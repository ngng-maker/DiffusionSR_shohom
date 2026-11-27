"""
Evaluate a 40 -> 20 -> 10 um cascade (stage A: 40 -> 20 um, stage B: 20 -> 10 um) on the 2026 regen data,
with the same metrics, files and extras as evaluate_si_schedules, so a chained eval compares one-to-one with
the direct 40 -> 10 um eval of the same schedule/start/solver/eps ("matched pairs").

Chaining, per test frame: stage A draws K samples at 20 um from the true 40 um input; each 20 um sample
(physical units, clipped like the dataset loader) is re-normalised with stage B's LR statistics, upscaled to
10 um resolution with the same interpolation as the dataset build (bilinear; nearest for liqlabel) and re-normalised with
stage B's upscaled-LR statistics; stage B then draws ONE 10 um sample from it. K stage-A samples -> K final
samples. Stage B was trained on TRUE 20 um simulations, so it sees its own predictions only here; that
train/test shift is part of what the chained numbers measure. Network evaluations add up (nfe A + nfe B).

The chain is wrapped as one model with the shared sample()/compute_x_e() interface and the test set as one
dataset (stage A inputs, stage B truth), so evaluate_si_schedules' resumable sampler and eval_extras are reused.
To fit that sampler's (N, C, H, W) arrays, the dataset returns stage A's upscaled-LR input nearest-upsampled to
10 um size; the cascade takes every second pixel back, which restores it exactly.

    python -m diffusionsr.analysis.evaluate_cascade --chain_name <name> \
        --a_config <cfgA.yml> --a_model_dir <runA> --a_enc_dir <encA> --a_data_root <regen_2d/x2_40to20> \
        --b_config <cfgB.yml> --b_model_dir <runB> --b_enc_dir <encB> --b_data_root <regen_2d/x2_20to10> \
        --solver heun --eps 0 --out_root eval_results/si_regen_cascade
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from diffusionsr.analysis.evaluate_coupled import compute_metrics, save_overlay_png
from diffusionsr.analysis.evaluate_si_schedules import (build_model, eval_name_for, ensemble_skill, ensemble_spread,
                                                        hr_px_um, largest_melt_frame, run_extras, sample_resumable)


def _key(path):
    """(condition folder, frame file) of a dataset path: the frame identity shared by all three task roots."""
    return os.path.basename(os.path.dirname(path)), os.path.basename(path)


class ChainDataset:
    """Test frames as the cascade sees them: stage A's inputs, stage B's ground truth."""

    def __init__(self, ds_a, ds_b):
        ka, kb = [_key(p) for p in ds_a.lr_paths], [_key(p) for p in ds_b.lr_paths]
        if ka != kb:
            idx = {k: i for i, k in enumerate(kb)}
            missing = [k for k in ka if k not in idx]
            if missing:
                raise SystemExit(f'{len(missing)} stage-A test frames have no stage-B twin, e.g. {missing[:3]}')
            self.b_index = [idx[k] for k in ka]
        else:
            self.b_index = list(range(len(ka)))
        self.a, self.b = ds_a, ds_b
        self.lr_paths = [ds_b.lr_paths[j] for j in self.b_index]   # P/V/timestep bookkeeping for eval_extras
        self.factor = ds_a.factor * ds_b.factor
        self.field_names = ds_b.field_names

    def __len__(self):
        return len(self.b_index)

    def __getitem__(self, i):
        _, _, lr_a, ulr_a = self.a[i][:4]
        res_b, hr_b = self.b[self.b_index[i]][:2]
        s = hr_b.shape[-1] // ulr_a.shape[-1]
        ulr_up = ulr_a.repeat_interleave(s, -2).repeat_interleave(s, -1)   # exact; undone by [::s, ::s]
        return res_b, hr_b, lr_a, ulr_up

    def unscale_data(self, array, input_type, **kw):
        if input_type == 'hr':
            return self.b.unscale_data(array, input_type='hr', **kw)
        if input_type == 'upscaled_lr':
            # stage A's upscaled LR was nearest-upsampled to stage-B size in __getitem__; its statistics are
            # per pixel at stage-A size, so un-scale at that size and upsample the physical field back.
            s = array.shape[-1] // int(self.a.img_shape)
            phys = self.a.unscale_data(array[..., ::s, ::s], input_type='upscaled_lr', **kw)
            return np.repeat(np.repeat(np.asarray(phys), s, axis=-2), s, axis=-1)
        return self.a.unscale_data(array, input_type=input_type, **kw)


def upscale_like_dataset(x, size, field_names, mode='dataset'):
    """Upscale (B, C, h, w) physical fields to `size` as the dataset build made LR/direct/<f>x.

    'dataset' (default): skimage.transform.resize order 1 (bilinear, no anti-aliasing) for continuous fields
    and order 0 (nearest) for the liquid label, as build_regen_dataset.py does and as reproduces data_fields'
    stored upscaled LR. For integer factors these equal torch bilinear (align_corners=False; skimage's
    reflect mode and torch's edge clamp agree at these sample points) and torch nearest.
    Any other value is passed to F.interpolate as the mode for every field.
    """
    if mode != 'dataset':
        return F.interpolate(x, size=size, mode=mode,
                             align_corners=False if mode in ('bilinear', 'bicubic') else None)
    up = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    for c, name in enumerate(field_names):
        if 'liqlabel' in name:
            up[:, c] = F.interpolate(x[:, c:c + 1], size=size, mode='nearest')[:, 0]
    return up


class CascadeModel:
    """Two trained models chained behind the shared sampling interface."""

    def __init__(self, model_a, model_b, ds_a, ds_b, upscale_mode='dataset'):
        self.A, self.B, self.da, self.db = model_a, model_b, ds_a, ds_b
        self.upscale_mode = upscale_mode
        self.model = torch.nn.ModuleList([model_a.model, model_b.model])       # for eval_extras.model_size
        encs = [m for m in (getattr(model_a, 'lr_enc', None), getattr(model_b, 'lr_enc', None)) if m is not None]
        self.lr_enc = torch.nn.ModuleList(encs) if encs else None
        self.device = model_b.device

    def _channel_names(self):
        """Field name of every channel: n_steps stacked frames x fields (channel = step * F + field)."""
        return list(self.db.field_names) * int(getattr(self.db, 'n_steps', 1))

    def _down(self, ulr_up):
        s = ulr_up.shape[-1] // int(self.da.img_shape)      # img_shape: HR side length of that task (int)
        return ulr_up[..., ::s, ::s]

    def compute_x_e(self, lr_a, ulr_up):
        return self.A.compute_x_e(lr_a, self._down(ulr_up))

    def _clip(self, phys):
        """Clip like SimulationXZDataset does on load (per-field thresholds, e.g. temperature 8000 K)."""
        thr = getattr(self.db, 'field_threshold', {})
        for c, name in enumerate(self._channel_names()):
            if name in thr and np.isfinite(thr[name]):
                phys[:, c] = phys[:, c].clamp(max=float(thr[name]))
        return phys

    def stage_b_inputs(self, x20_norm):
        """Stage-A output (stage A HR-standardised) -> stage B (lr, upscaled_lr), standardised."""
        phys = self.da.unscale_data(x20_norm, input_type='hr', maintain_torch=True)
        phys = self._clip(phys.float())
        lr_b = self.db.rescale_data(phys, input_type='lr', maintain_torch=True).float()
        H = W = int(self.db.img_shape)
        up = upscale_like_dataset(phys, (H, W), self._channel_names(), self.upscale_mode)
        ulr_b = self.db.rescale_data(up, input_type='upscaled_lr', maintain_torch=True).float()
        return lr_b, ulr_b

    @torch.no_grad()
    def sample(self, ulr_up, x_e_a, lr=None, solver='heun', n_steps=20, eps=0.0, generator=None, callback=None):
        cb_a = cb_b = None
        if callback is not None:
            n_a = [0]

            def cb_a(i, t, x):        # stage A state shown at 10 um size, in stage B's HR standardisation
                phys = self.da.unscale_data(x, input_type='hr', maintain_torch=True).float()
                up = F.interpolate(phys, size=(int(self.db.img_shape),) * 2, mode='nearest')
                callback(i, 0.5 * t, self.db.rescale_data(up, input_type='hr', maintain_torch=True).float())
                n_a[0] = i

            def cb_b(i, t, x):
                callback(n_a[0] + 1 + i, 0.5 + 0.5 * t, x)
        x20, nfe_a = self.A.sample(self._down(ulr_up), x_e_a, lr=lr, solver=solver, n_steps=n_steps, eps=eps,
                                   generator=generator, callback=cb_a)
        lr_b, ulr_b = self.stage_b_inputs(x20.to(self.device))
        x_e_b = self.B.compute_x_e(lr_b, ulr_b)
        x10, nfe_b = self.B.sample(ulr_b, x_e_b, lr=lr_b, solver=solver, n_steps=n_steps, eps=eps,
                                   generator=generator, callback=cb_b)
        return x10, nfe_a + nfe_b


def main():
    ap = argparse.ArgumentParser(description='Evaluate a two-stage SR cascade')
    ap.add_argument('--chain_name', required=True, help='e.g. si_regen_tempsdf_linear_enc_cascade')
    for s in ('a', 'b'):
        ap.add_argument(f'--{s}_config', required=True)
        ap.add_argument(f'--{s}_model_dir', required=True)
        ap.add_argument(f'--{s}_enc_dir', required=True)
        ap.add_argument(f'--{s}_data_root', required=True)
    ap.add_argument('--solver', required=True)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--n_steps', type=int, default=20)
    ap.add_argument('--k_samples', type=int, default=10)
    ap.add_argument('--frames_per_batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--weights', default='raw', choices=['raw', 'ema'])
    ap.add_argument('--upscale_mode', default='dataset',
                    help="'dataset' = bilinear + nearest for liqlabel, as the dataset build (LR/direct/<f>x)")
    ap.add_argument('--plot_frame_idx', type=int, default=-1)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--keep_partial', action='store_true')
    ap.add_argument('--save_predictions', default='full', choices=['full', 'q16', 'none'],
                    help="as evaluate_si_schedules: 'full' float32, 'q16' 16-bit + exact outliers, 'none'")
    ap.add_argument('--partial_current_frame', action='store_true',
                    help='keep only frame t of n_steps-stacked outputs in partial/ (5x less temporary disk)')
    ap.add_argument('--no_extras', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    built = {}
    for s in ('a', 'b'):
        cfg = yaml.safe_load(open(getattr(args, f'{s}_config')))
        ns = SimpleNamespace(model_dir=getattr(args, f'{s}_model_dir'), enc_dir=getattr(args, f'{s}_enc_dir'),
                             data_root=getattr(args, f'{s}_data_root'), weights=args.weights)
        built[s] = (cfg,) + tuple(build_model(ns, cfg, device))
    cfg_a, model_a, ds_a, _ = built['a']
    cfg_b, model_b, ds_b, field_names = built['b']
    if list(ds_a.field_names) != list(ds_b.field_names):
        raise SystemExit(f'stage fields differ: {ds_a.field_names} vs {ds_b.field_names}')

    chain_ds = ChainDataset(ds_a, ds_b)
    model = CascadeModel(model_a, model_b, ds_a, ds_b, args.upscale_mode)
    args.eval_name = eval_name_for(args.chain_name, args.solver, args.eps, args.n_steps)
    args.out_dir = os.path.join(args.out_root, args.eval_name)
    args.model_dir, args.enc_dir = args.b_model_dir, args.b_enc_dir      # eval_extras.model_size paths (stage B)
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f'{args.eval_name}_results.json')
    npz_path = os.path.join(args.out_dir, 'predictions.npz')
    if os.path.exists(json_path) and (os.path.exists(npz_path) or args.save_predictions == 'none'):
        print(f'{json_path} exists; nothing to do')
        return

    from diffusionsr.analysis.eval_extras import current_frame, save_predictions, to_current_frame
    cur = current_frame(cfg_b.get('n_steps', 1), len(field_names)) if args.partial_current_frame else None
    preds, gts, ulrs, nfe, wall = sample_resumable(model, chain_ds, args, device, cur=cur)
    # score frame t only
    preds, gts, ulrs = to_current_frame(preds, gts, ulrs, cfg_b.get('n_steps', 1), len(field_names))
    px_um = hr_px_um(cfg_b, ds_b)
    if args.plot_frame_idx < 0:
        args.plot_frame_idx = largest_melt_frame(gts)
    results = compute_metrics(preds, gts, px_um)
    results.update(ensemble_spread(preds, gts))
    results.update(ensemble_skill(preds, gts))
    results.update(train_name=args.chain_name, eval_name=args.eval_name, stage_a=args.a_model_dir,
                   stage_b=args.b_model_dir, si_schedule=cfg_b.get('si_schedule', cfg_b.get('model_kind')),
                   si_base=cfg_b.get('si_base', 'enc'), sigma=float(cfg_b.get('si_sigma', 0.0)),
                   solver=args.solver, eps=args.eps, n_steps=args.n_steps, nfe_per_sample=float(nfe),
                   wall_clock_s=float(wall), weights=args.weights, seed=args.seed, hr_px_um=px_um,
                   field_names=list(field_names), model_kind='cascade', upscale_mode=args.upscale_mode,
                   plot_frame_idx=int(args.plot_frame_idx))
    if not args.no_extras:
        results.update(run_extras(args, cfg_b, model, chain_ds, field_names, preds, gts, px_um, device))
        for s, m in (('a', model_a), ('b', model_b)):
            p = os.path.join(getattr(args, f'{s}_model_dir'), 'bestmodel_saved.pth')
            results[f'ckpt_MB_stage_{s}'] = float(os.path.getsize(p) / 2 ** 20) if os.path.exists(p) else float('nan')
    overlay_ulrs = np.asarray(ulrs)
    save_overlay_png(preds, gts, overlay_ulrs, args.out_dir, int(args.plot_frame_idx))
    results['predictions_saved'] = (save_predictions(npz_path, preds, gts, field_names, args.save_predictions)
                                    if args.save_predictions != 'none' else 'none')
    with open(json_path + '.tmp', 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(json_path + '.tmp', json_path)         # atomic: the JSON marks the eval as done
    if not args.keep_partial:
        del preds, gts, ulrs
        import gc
        gc.collect()
        shutil.rmtree(os.path.join(args.out_dir, 'partial'), ignore_errors=True)
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == '__main__':
    main()
