"""
Ablation v2: evaluate trained runs on many test frames, so differences between configurations get confidence
intervals instead of resting on the 5 QSS frames of the preview.

Frames: the LAST frame of each test simulation (one quasi-steady-state frame per power/velocity case), up to
--n_frames, spread evenly over the simulations in sorted order. The same frames are used for every variant.

Sampling: the preview settings (DiffusionSR/LDM ancestral DDPM over the trained T, FM Euler with fm_n_steps),
--n_samples samples per frame, seed 3000+frame. Also saves the frozen encoder's own output (enc_only).

Variants: any run directory under runs/ablation_v2 whose name starts with diffusion_/fm_/ldm_; the model is
built from that run's own training config (encoding, conditioning, encoder_normalize, seed, latent U-Net size).

Writes <out_dir>/<variant>.npz: samples (N, S, H, W) temperature in K, enc_only (N, H, W), gts, ulrs,
frame_idx, sims, seconds; and <out_dir>/stats_<field>.npz with the per-pixel and global HR statistics
(temperature channel) under both normalisations, for the std-map check. Metrics are computed offline.

Run on a GPU from the repo root on TRACE, e.g. inside a running job:
  srun --jobid=<job> --overlap --gres=gpu:1 python scripts/eval_ablation_v2_frames.py \
      --out_dir diffusionsr/runs/ablation_v2/_eval_frames --variants diffusion_enc_temp_std ...
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
_spec = importlib.util.spec_from_file_location('pv', SCRIPT_DIR / 'precompute_ablation_v2_preview.py')
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)

FIELD_OF = {'temperature': 'temp', 'temperature_sdfliqlabel': 'multi'}
NORM_OF = {'standardize': 'std', 'global_standardize': 'gstd'}
DEFAULT_VARIANTS = [f'{m}_enc_{f}_{n}' for m in ('diffusion', 'fm', 'ldm') for f in pv.FIELDS for n in pv.NORMS]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--variants', nargs='*', default=DEFAULT_VARIANTS)
    p.add_argument('--n_frames', type=int, default=48)
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def select_frames(test_ds, n_frames):
    """Index of the last frame of each test simulation, evenly subsampled to n_frames."""
    sims = np.array([Path(p).parent.name for p in test_ds.hr_paths])
    last = {}
    for i, s in enumerate(sims):  # hr_paths are sorted, so the last index per sim is its final frame
        last[s] = i
    names = sorted(last)
    if len(names) > n_frames:
        names = [names[int(round(j))] for j in np.linspace(0, len(names) - 1, n_frames)]
    return [last[s] for s in names], names


def build_model(variant, cfg, run_dir, train_ds, dev_ds, test_ds, device, settings):
    m = variant.split('_')[0]
    encoding = bool(cfg.get('encoding', False))
    common = dict(results_folder=str(run_dir),
                  lr_encoder_folder=str(run_dir / 'encoder') if encoding else None,
                  train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
                  encoding=encoding, device=device, conditioning=cfg.get('conditioning', 'implicit'),
                  timesteps=settings['timesteps'], schedule=settings['schedule'],
                  seed=int(cfg.get('seed', 0)))
    if m == 'ldm':
        from diffusionsr.runners.train_ldm import LDMModel
        model = LDMModel(vae_folder=cfg.get('vae_results_dir', str(run_dir)),
                         latent_unet_dim=cfg.get('latent_unet_dim'),
                         latent_unet_mults=tuple(cfg.get('latent_unet_mults') or (1,)), **common)
    elif m == 'fm':
        from diffusionsr.runners.train_flow_matching import FlowMatchingModel
        model = FlowMatchingModel(enc_output=False, fm_timescale=settings['fm_timescale'],
                                  encoder_normalize=cfg.get('encoder_normalize'), **common)
    else:
        from diffusionsr.runners.train_diffusion import DiffusionModel
        model = DiffusionModel(enc_output=False, encoder_normalize=cfg.get('encoder_normalize'), **common)
    assert model.timesteps == settings['timesteps'], (model.timesteps, settings)
    best = run_dir / 'bestmodel_saved.pth'
    path = best if best.exists() else run_dir / 'ckpt.pth'
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt[0] if isinstance(ckpt, (list, tuple)) else ckpt.get('model', ckpt))
    model.model.eval()
    return model, path


def enc_only(model, lr, device):
    """Frozen encoder's own HR estimate (normalised units of the dataset)."""
    if getattr(model, 'encoder_normalize', None) is not None:
        return model.encode_lr(lr.unsqueeze(0), output=True)
    from diffusionsr.runners.train_diffusion import forwardpass
    return forwardpass(model.lr_enc, lr.unsqueeze(0).to(device).float(), factor=model.train_dataset.factor,
                       output=True, transform_rescale=model.transform_rescale, dataset=model.train_dataset)


def temp_phys(ds, x, n_ch):
    """(B, n_steps*n_ch, H, W) normalised -> (B, H, W) temperature in K at the last timestep."""
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return np.stack([ds.unscale_data(xi, input_type='hr')[-n_ch] for xi in x])


def save_stats(out_dir, f):
    path = out_dir / f'stats_{f}.npz'
    if path.exists():
        return
    _, _, ds = pv.load_datasets(f, 'std')
    n_ch = len(pv.FIELD_NAMES[f])
    np.savez(path, mean_hr=ds.mean_hr[-n_ch], std_hr=ds.std_hr[-n_ch],
             global_mean_hr=float(np.atleast_1d(ds.global_mean_hr)[-n_ch]),
             global_std_hr=float(np.atleast_1d(ds.global_std_hr)[-n_ch]))


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device {device}, out {out_dir}', flush=True)
    for variant in args.variants:
        out = out_dir / f'{variant}.npz'
        if out.exists() and not args.force:
            print(f'{variant}: exists, skipped', flush=True)
            continue
        m = variant.split('_')[0]
        run_dir = pv.RUNS_ROOT / variant
        cfg, _ = pv.load_run_config(variant, run_dir)
        f = FIELD_OF[str(cfg['fields'])]
        n = NORM_OF[cfg['normalize_method']]
        save_stats(out_dir, f)
        settings = pv.sampling_settings(m, cfg, 20)
        train_ds, dev_ds, test_ds = pv.load_datasets(f, n)
        model, ckpt_path = build_model(variant, cfg, run_dir, train_ds, dev_ds, test_ds, device, settings)
        n_ch = len(pv.FIELD_NAMES[f])
        idx, sims = select_frames(test_ds, args.n_frames)
        print(f'\n=== {variant}: {len(idx)} frames x {args.n_samples} samples, {settings["sampler"]} '
              f'{settings["n_sampling_steps"]} steps, ckpt {ckpt_path.name} ===', flush=True)
        gts, ulrs, samples, encs = [], [], [], []
        t0 = time.time()
        with torch.no_grad():
            for k, i in enumerate(idx):
                res, hr, lr, ulr = test_ds[i][:4]
                gts.append(temp_phys(test_ds, hr.unsqueeze(0), n_ch)[0])
                ulrs.append(np.stack([test_ds.unscale_data(ulr.numpy(), input_type='upscaled_lr')[-n_ch]])[0])
                if model.encoding:
                    encs.append(temp_phys(test_ds, enc_only(model, lr, device), n_ch)[0])
                x_e = model.compute_x_e(lr.unsqueeze(0).to(device).float(), ulr.unsqueeze(0).to(device).float())
                x_e_b = None if x_e is None else x_e.repeat(args.n_samples, *[1] * (x_e.ndim - 1))
                hr_b = hr.unsqueeze(0).repeat(args.n_samples, 1, 1, 1).to(device).float()
                torch.manual_seed(3000 + k)
                samples.append(temp_phys(test_ds, pv.draw_samples(model, m, settings, hr_b, x_e_b, test_ds), n_ch))
                if k % 8 == 0:
                    print(f'  frame {k + 1}/{len(idx)}  {time.time() - t0:6.0f} s', flush=True)
        np.savez_compressed(out, samples=np.stack(samples).astype(np.float32), gts=np.stack(gts).astype(np.float32),
                            ulrs=np.stack(ulrs).astype(np.float32),
                            enc_only=np.stack(encs).astype(np.float32) if encs else np.zeros(0),
                            frame_idx=np.array(idx), sims=np.array(sims), seconds=np.array(time.time() - t0),
                            sampler=settings['sampler'], n_sampling_steps=settings['n_sampling_steps'])
        print(f'  saved {out.name} ({time.time() - t0:.0f} s)', flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
