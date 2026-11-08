"""
Evaluate one (schedule, base, solver, eps) configuration of the SI-schedules study.

Metrics, overlay and prediction format are the previous SI experiment's, by construction:
  * metrics: evaluate_coupled.compute_metrics, verbatim (MAE/RMSE per sample, ensemble-mean MAE,
    MP/KH depth MAE, area MAE, calibration_2sigma, n_melt_frames), on the full test split, K = 10;
  * overlay: evaluate_coupled.save_overlay_png, verbatim (frame 70);
  * predictions.npz: pred (N, K, C, H, W), gt (N, C, H, W), ens_std, field_names, physical units, so
    notebook 21's loading code reads these files unchanged;
  * weights: bestmodel_saved.pth entry [0] (the raw weights), as evaluate_coupled loads them.
Extra keys are only ADDED to results.json (solver, eps, nfe_per_sample, wall_clock_s, ...).

Robust to preemption/requeue on TRACE and ORCHARD:
  * Sampling is checkpointed after every batch into <out_dir>/partial/ (memory-mapped arrays plus a
    progress file); a restarted job continues from the next unfinished batch.
  * Each batch is seeded from (seed, batch index), so a resumed eval draws exactly the samples an
    uninterrupted one would have.
  * If results already exist, the job only (re)uploads; W&B deduplicates identical files.

W&B: one deterministic eval run per configuration (resume='allow') and ONE artifact,
`<eval_name>_eval_predictions`, holding a single version: each upload commits a new version and deletes
every other one (utils.log_single_version_artifact).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from diffusionsr.analysis.evaluate_coupled import _px_um, compute_metrics, save_overlay_png
from diffusionsr.analysis.si_solvers import SDE_SOLVERS, SOLVERS


def eps_tag(eps: float) -> str:
    """0.05 -> 'eps0p05'; filesystem- and W&B-safe."""
    return 'eps' + f'{eps:g}'.replace('.', 'p')


def eval_name_for(train_name: str, solver: str, eps: float) -> str:
    return f'{train_name}__{solver}_{eps_tag(eps)}'


def build_model(args, cfg, device):
    from diffusionsr.datasets.flipped_dataset import FlippedDataset
    from diffusionsr.runners.train_si_schedules import ScheduledInterpolantModel
    raw = cfg.get('fields', 'temperature')
    field_names = raw.split('_') if isinstance(raw, str) else list(raw)
    kw = dict(downscale_method=cfg.get('downscale_method', 'direct'), root_folder=args.data_root,
              normalize=cfg.get('normalize_method', 'standardize'), n_steps=int(cfg.get('n_steps', 1)),
              field_names=field_names, p_flip=0.0)
    train_ds = FlippedDataset(split='train', **kw)
    dev_ds = FlippedDataset(split='dev', **kw)
    test_ds = FlippedDataset(split='test', **kw)

    best_path = os.path.join(args.model_dir, 'bestmodel_saved.pth')
    ckpt = torch.load(best_path, map_location='cpu')
    unet_dim = int(ckpt[0]['time_mlp.1.weight'].shape[1])   # as evaluate_coupled: dim from the checkpoint

    model = ScheduledInterpolantModel(
        results_folder=args.model_dir, lr_encoder_folder=args.enc_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        encoding=bool(cfg.get('encoding', True)), conditioning=cfg.get('conditioning', 'implicit'),
        sigma=float(cfg['si_sigma']), si_cond=cfg.get('si_cond', 'enc_implicit'),
        si_base=cfg.get('si_base', 'enc'), si_schedule=cfg['si_schedule'],
        si_gamma_scale=float(cfg.get('si_gamma_scale', 0.0)),
        fm_timescale=float(cfg.get('fm_timescale', 1000.0)),
        image_size_override=unet_dim, enc_output=False, device=device)
    state = ckpt[5] if args.weights == 'ema' else ckpt[0]
    model.model.load_state_dict(state)
    model.model.eval()
    print(f'Loaded {best_path} ({args.weights} weights, best epoch {ckpt[2]})')
    return model, test_ds, field_names


@torch.no_grad()
def sample_resumable(model, test_ds, args, device):
    """Fill <out_dir>/partial/{pred,gt,ulr}.npy batch by batch; resume from progress.json."""
    part = os.path.join(args.out_dir, 'partial')
    os.makedirs(part, exist_ok=True)
    N, K = len(test_ds), args.k_samples
    if getattr(args, 'max_frames', None):        # smoke runs only; full evals use the whole test split
        N = min(N, args.max_frames)
    loader = DataLoader(Subset(test_ds, range(N)), batch_size=args.frames_per_batch, shuffle=False,
                        drop_last=False)
    prog_path = os.path.join(part, 'progress.json')
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}
    sig = dict(N=N, K=K, solver=args.solver, eps=args.eps, n_steps=args.n_steps, seed=args.seed,
               frames_per_batch=args.frames_per_batch, weights=args.weights)
    if prog and prog.get('sig') != sig:
        print('partial/ was written with different settings; starting over')
        shutil.rmtree(part); os.makedirs(part); prog = {}
    done = prog.get('done', 0)
    nfe_sum, wall = prog.get('nfe_sum', 0), prog.get('wall_s', 0.0)

    arrays = {}
    for i, (res, hr, lr, ulr) in enumerate(loader):
        if i == 0:
            C, H, W = hr.shape[1:]
            mode = 'r+' if done else 'w+'
            arrays['pred'] = np.lib.format.open_memmap(os.path.join(part, 'pred.npy'), mode, np.float32, (N, K, C, H, W))
            arrays['gt'] = np.lib.format.open_memmap(os.path.join(part, 'gt.npy'), mode, np.float32, (N, C, H, W))
            arrays['ulr'] = np.lib.format.open_memmap(os.path.join(part, 'ulr.npy'), mode, np.float32, (N, C, H, W))
        if i < done:
            continue
        t0 = time.time()
        s0 = i * args.frames_per_batch
        B = hr.shape[0]
        hr, lr, ulr = hr.to(device).float(), lr.to(device).float(), ulr.to(device).float()
        x_e = model.compute_x_e(lr, ulr)            # encoder features, once per frame
        rep = lambda x: None if x is None else x.repeat_interleave(K, dim=0)   # (B*K, ...) frame-major
        gen = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + i)
        x1, nfe = model.sample(rep(ulr), rep(x_e), lr=rep(lr), solver=args.solver,
                               n_steps=args.n_steps, eps=args.eps, generator=gen)
        x1 = x1.cpu().numpy().reshape(B, K, *x1.shape[1:])
        arrays['pred'][s0:s0 + B] = np.stack([np.stack([test_ds.unscale_data(x1[b, k], input_type='hr')
                                                        for k in range(K)]) for b in range(B)])
        arrays['gt'][s0:s0 + B] = np.stack([test_ds.unscale_data(hr[b].cpu().numpy(), input_type='hr') for b in range(B)])
        arrays['ulr'][s0:s0 + B] = np.stack([test_ds.unscale_data(ulr[b].cpu().numpy(), input_type='upscaled_lr') for b in range(B)])
        for a in arrays.values():
            a.flush()
        nfe_sum += nfe
        wall += time.time() - t0
        json.dump(dict(sig=sig, done=i + 1, nfe_sum=nfe_sum, wall_s=wall), open(prog_path + '.tmp', 'w'))
        os.replace(prog_path + '.tmp', prog_path)     # atomic: a kill never leaves a torn progress file
        print(f'  batch {i + 1}/{len(loader)}  nfe={nfe}', flush=True)

    n_batches = len(loader)
    return (np.load(os.path.join(part, 'pred.npy'), mmap_mode='r'),
            np.load(os.path.join(part, 'gt.npy'), mmap_mode='r'),
            np.load(os.path.join(part, 'ulr.npy'), mmap_mode='r'),
            nfe_sum / max(n_batches, 1), wall)


def ensemble_spread(preds, gts, ch: int = 0, threshold: float = 1900.0, chunk: int = 256) -> dict:
    """Notebook 21 section 6's spread numbers, computed here so the notebook needs no prediction arrays.

    Same formula: per-pixel std over the K samples (ddof 0), averaged over all pixels of all frames, and
    over the pixels inside the GROUND-TRUTH melt pool (T > 1900 K).
    """
    N = preds.shape[0]
    tot = tot_mp = 0.0
    n_mp = 0
    for s in range(0, N, chunk):
        sd = np.asarray(preds[s:s + chunk, :, ch], dtype=np.float32).std(axis=1)
        mask = np.asarray(gts[s:s + chunk, ch]) > threshold
        tot += float(sd.sum())
        tot_mp += float(sd[mask].sum())
        n_mp += int(mask.sum())
    return {'spread_all_px_K': tot / sd[0].size / N, 'spread_melt_px_K': tot_mp / max(n_mp, 1)}


def _sorted_pair_weights(K: int) -> np.ndarray:
    """w_j = 2j - K + 1 (j = 0..K-1): sum_{k<j} |x_(j) - x_(k)| = sum_j w_j x_(j) for sorted x."""
    return (2.0 * np.arange(K) - K + 1).astype(np.float64)


def ensemble_skill(preds, gts, ch: int = 0, threshold: float = 1900.0, chunk: int = 128,
                   seed: int = 0) -> dict:
    """Ensemble-verification scores, added for the SI-noise study (all in physical units, K).

    Computed over ALL pixels and over the GROUND-TRUTH melt pool (T > 1900 K), as ensemble_spread is.

    crps_*_K      fair CRPS (Ferro 2014): mean_k |x_k - y| - sum_{k != j} |x_k - x_j| / (2 K (K - 1)).
                  A proper score: lowest only when the predictive distribution matches the truth's, so
                  noise that is mostly error raises it (unlike calibration_2sigma). Unbiased for finite K.
                  K = 1 (a point prediction): CRPS = MAE.
    ssr_*         spread-skill ratio sqrt((K+1)/K) * sqrt(mean ensemble variance (ddof 1)) / RMSE of the
                  ensemble mean (Fortin et al. 2014). 1 = spread matches error; < 1 under-dispersed.
    rank_hist_*   counts of the truth's rank among the K members (K + 1 bins; ties broken at random).
                  Flat = consistent; U-shape = under-dispersed; hump = over-dispersed.
    """
    N, K = preds.shape[:2]
    w = _sorted_pair_weights(K)
    rng = np.random.default_rng(seed)
    acc = {r: dict(crps=0.0, var=0.0, se=0.0, n=0, hist=np.zeros(K + 1, np.int64)) for r in ('all', 'melt')}
    for s in range(0, N, chunk):
        x = np.asarray(preds[s:s + chunk, :, ch], dtype=np.float64)            # (n, K, H, W)
        y = np.asarray(gts[s:s + chunk, ch], dtype=np.float64)                 # (n, H, W)
        mae_term = np.abs(x - y[:, None]).mean(axis=1)
        if K > 1:
            xs = np.sort(x, axis=1)
            pair = np.tensordot(w, xs, axes=(0, 1))                            # sum_{k<j} |x_j - x_k|
            crps = mae_term - pair / (K * (K - 1))
            var = x.var(axis=1, ddof=1)
        else:
            crps, var = mae_term, np.zeros_like(y)
        se = (x.mean(axis=1) - y) ** 2
        below = (x < y[:, None]).sum(axis=1)
        ties = (x == y[:, None]).sum(axis=1)
        rank = below + np.floor(rng.random(y.shape) * (ties + 1)).astype(np.int64)
        mask = y > threshold
        for r, sel in (('all', np.ones_like(mask)), ('melt', mask)):
            a = acc[r]
            a['crps'] += float(crps[sel].sum()); a['var'] += float(var[sel].sum())
            a['se'] += float(se[sel].sum()); a['n'] += int(sel.sum())
            a['hist'] += np.bincount(rank[sel], minlength=K + 1)
    out = {}
    for r, a in acc.items():
        n = max(a['n'], 1)
        rmse = (a['se'] / n) ** 0.5
        out[f'crps_{r}_K'] = a['crps'] / n
        out[f'ssr_{r}'] = (((K + 1) / K) ** 0.5 * (a['var'] / n) ** 0.5 / rmse) if (K > 1 and rmse > 0) else float('nan')
        out[f'rank_hist_{r}'] = a['hist'].tolist()
    return out


def upload(args, json_path, npz_path, overlay_path, results):
    import wandb
    from diffusionsr.utils import log_single_version_artifact, wandb_safe_id
    run = wandb.init(id=wandb_safe_id(f'{args.eval_name}-{args.run_tag}'), resume='allow',
                     project=args.wandb_project, entity=args.wandb_entity or None,
                     name=args.eval_name, group=os.getenv('SI_WANDB_GROUP', 'si_schedules'), job_type='eval',
                     config=dict(train_name=args.train_name, solver=args.solver, eps=args.eps,
                                 n_steps=args.n_steps, k_samples=args.k_samples, weights=args.weights))
    run.summary.update({k: v for k, v in results.items() if isinstance(v, (int, float, str))})
    ok = log_single_version_artifact(
        f'{args.eval_name}_eval_predictions', 'eval_predictions',
        {'predictions.npz': npz_path, 'results.json': json_path, 'overlay.png': overlay_path},
        metadata=results, description=f'SI-schedules eval: {args.eval_name}', run=run)
    run.finish()
    if not ok:
        raise SystemExit('W&B upload failed; results are saved locally, rerun to retry the upload')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--train_name', required=True, help='config stem, e.g. si_sched_linear_enc_temp')
    ap.add_argument('--config', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--enc_dir', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--solver', required=True, choices=SOLVERS)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--n_steps', type=int, default=20)
    ap.add_argument('--k_samples', type=int, default=10)
    ap.add_argument('--frames_per_batch', type=int, default=4, help='frames per batch (x K samples)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--weights', default='raw', choices=['raw', 'ema'],
                    help="'raw' (default) matches how the previous SI evals loaded checkpoints")
    ap.add_argument('--plot_frame_idx', type=int, default=70)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--ckpt_artifact', default='', help='restore bestmodel_saved.pth from here if missing')
    ap.add_argument('--keep_partial', action='store_true')
    ap.add_argument('--max_frames', type=int, default=None, help='smoke runs only: first N test frames')
    ap.add_argument('--no_upload', action='store_true')
    ap.add_argument('--run_tag', default=os.getenv('SI_SCHED_TAG', 'v1'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--wandb_entity', default=os.getenv('WANDB_ENTITY', ''))
    ap.add_argument('--wandb_project', default='Flow3D_SuperResolution')
    args = ap.parse_args()

    if args.eps > 0 and args.solver not in SDE_SOLVERS:
        raise SystemExit(f'{args.solver} is ODE-only; eps must be 0')
    args.eval_name = eval_name_for(args.train_name, args.solver, args.eps)
    args.out_dir = os.path.join(args.out_root, args.eval_name)
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f'{args.eval_name}_results.json')
    npz_path = os.path.join(args.out_dir, 'predictions.npz')
    overlay_path = os.path.join(args.out_dir, f'overlay_frame{args.plot_frame_idx}.png')

    if os.path.exists(json_path) and os.path.exists(npz_path) and os.path.exists(overlay_path):
        print(f'{json_path} exists: skipping sampling, (re)uploading only')
        results = json.load(open(json_path))
    else:
        if not os.path.exists(os.path.join(args.model_dir, 'bestmodel_saved.pth')) and args.ckpt_artifact:
            from diffusionsr.utils import restore_single_version_artifact
            restore_single_version_artifact(args.wandb_entity, args.wandb_project, args.ckpt_artifact,
                                            args.model_dir, only=['bestmodel_saved.pth', 'configuration.yml'])
        cfg = yaml.safe_load(open(args.config))
        device = args.device if torch.cuda.is_available() else 'cpu'
        model, test_ds, field_names = build_model(args, cfg, device)
        preds, gts, ulrs, nfe, wall = sample_resumable(model, test_ds, args, device)

        results = compute_metrics(preds, gts, _px_um(test_ds.factor))
        results.update(ensemble_spread(preds, gts))
        results.update(ensemble_skill(preds, gts))
        results.update(train_name=args.train_name, eval_name=args.eval_name,
                       si_schedule=cfg['si_schedule'], si_gamma_scale=float(cfg.get('si_gamma_scale', 0.0)),
                       si_base=cfg.get('si_base', 'enc'),
                       sigma=float(cfg['si_sigma']), solver=args.solver, eps=args.eps,
                       n_steps=args.n_steps, nfe_per_sample=float(nfe), wall_clock_s=float(wall),
                       weights=args.weights, seed=args.seed)
        overlay_path = save_overlay_png(preds, gts, ulrs, args.out_dir, args.plot_frame_idx)
        np.savez_compressed(npz_path + '.tmp.npz', pred=np.asarray(preds, dtype=np.float32),
                            gt=np.asarray(gts, dtype=np.float32),
                            ens_std=np.asarray(preds, dtype=np.float32).std(axis=1),
                            field_names=np.array(field_names, dtype=object))
        os.replace(npz_path + '.tmp.npz', npz_path)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        if not args.keep_partial:
            # Release the memory maps first: on NFS an open file is silly-renamed (.nfsXXXX) rather than
            # unlinked, which left an empty partial/ behind in the TRACE smoke run.
            del preds, gts, ulrs
            import gc
            gc.collect()
            shutil.rmtree(os.path.join(args.out_dir, 'partial'), ignore_errors=True)
        print(json.dumps(results, indent=2))

    if not args.no_upload:
        upload(args, json_path, npz_path, overlay_path, results)


if __name__ == '__main__':
    main()
