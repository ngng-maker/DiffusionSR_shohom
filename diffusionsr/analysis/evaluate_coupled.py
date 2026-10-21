"""
Evaluation harness for the coupled stochastic-interpolant model.

For each test frame, draws K=10 samples from the model and computes:
  - MAE [K], RMSE [K] (per sample, then averaged)
  - MAE of ensemble mean
  - MP-MAE (melt-pool boundary depth error in µm)
  - Keyhole depth MAE in µm
  - Depth [µm] and area [µm²] from melt-pool profile
  - Calibration: fraction of GT keyhole depths inside mean ± 2*std of ensemble
  - Per-pixel ensemble std (for std-map figures)

Results are saved as a .npz artifact on the model's W&B training run.
Figures are saved locally under --out_dir.

Usage (on TRACE):
  python -m diffusionsr.analysis.evaluate_coupled \\
      --run_name   si_opt_a_s010_ulr_temp \\
      --wandb_run_name  <training run display name> \\
      --model_dir  /path/to/si_opt_a_s010_ulr_temp \\
      --data_root  /trace/group/forgelab/ngng/multifield/data_fields \\
      --field_names temperature \\
      --sampler    heun \\
      --n_steps    20 \\
      --k_samples  10 \\
      --out_dir    /path/to/eval_outputs/si \\
      --wandb_entity $WANDB_ENTITY
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader


def _find_root():
    s = Path(__file__).resolve()
    for p in [s, *s.parents]:
        if (p / 'setup.py').exists() and (p / 'diffusionsr').exists():
            return p
    raise RuntimeError('Cannot find project root')

PROJECT_ROOT = _find_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIXEL_SIZE_UM = {'2x': 10.0, '4x': 5.0}   # µm per HF pixel for each task


def _px_um(factor: int) -> float:
    return PIXEL_SIZE_UM.get(f'{factor}x', 5.0)


def _mp_geometry(temp: np.ndarray, px_um: float, threshold: float = 1900.0) -> dict:
    """
    Melt-pool geometry of one (H, W) physical temperature field.

    Orientation follows analysis_functions.get_profile: after temp.T the rows are z (the free surface
    is at HIGH z, cold gas above it) and the columns are x. Gas, the keyhole cavity and the cold solid
    are all at ambient, so per melted column the hot pixels run from the melt-pool bottom up to either
    the free surface or, under a keyhole, the keyhole bottom.

    get_profile pads columns WITHOUT melt with the column height, and the old metrics took max() and
    len() over that padded profile -- so every depth was the padding value and every width the full
    domain, for prediction and ground truth alike. MP/KH/depth/area MAE were always 0 and calibration
    always 1. Only melted columns are used here.

      surface   = highest hot pixel over all melted columns (the undisturbed melt surface)
      mp_depth  = surface - deepest hot pixel + 1          (melt-pool depth, um)
      kh_depth  = surface - lowest top-of-hot-region pixel (keyhole depression depth, um; 0 if flat)
      area      = hot-pixel count * px_um^2                (um^2)
    A frame with no melt has all four equal to 0.
    """
    hot = temp.T > threshold
    cols = np.where(hot.any(axis=0))[0]
    if len(cols) == 0:
        return {'mp_depth': 0.0, 'kh_depth': 0.0, 'width': 0.0, 'area': 0.0}
    bottoms = np.array([np.where(hot[:, c])[0].min() for c in cols])
    tops = np.array([np.where(hot[:, c])[0].max() for c in cols])
    surface = tops.max()
    return {
        'mp_depth': float(surface - bottoms.min() + 1) * px_um,
        'kh_depth': float(surface - tops.min()) * px_um,
        'width': float(len(cols)) * px_um,
        'area': float(hot.sum()) * px_um ** 2,
    }


def _phys_metrics(pred_t: np.ndarray, gt_t: np.ndarray, px_um: float):
    """
    Compute melt-pool metrics for a single (H, W) temperature field pair.
    Returns dict of scalars.
    """
    gp = _mp_geometry(pred_t, px_um)
    gg = _mp_geometry(gt_t, px_um)
    return {
        'mp_mae': abs(gp['mp_depth'] - gg['mp_depth']),   # um
        'kh_mae': abs(gp['kh_depth'] - gg['kh_depth']),   # um
        'depth_p': gp['mp_depth'], 'depth_g': gg['mp_depth'],
        'area_p': gp['area'], 'area_g': gg['area'],
    }


def compute_metrics(preds: np.ndarray, gts: np.ndarray, px_um: float,
                    temp_field_idx: int = 0) -> dict:
    """
    All scalar metrics from physical-unit predictions (N, K, C, H, W) and ground truth (N, C, H, W).
    Factored out of evaluate() so saved predictions (the W&B predictions.npz) can be re-scored with
    exactly the code an eval run uses.
    """
    N, K, C, H, W = preds.shape
    # ---- Per-sample metrics ----
    maes, rmses, mp_maes, kh_maes = [], [], [], []
    depths_p, depths_g, areas_p, areas_g = [], [], [], []

    for n in range(N):
        gt = gts[n]   # (C, H, W)
        for k in range(K):
            p = preds[n, k]
            maes.append(float(np.abs(p - gt).mean()))
            rmses.append(float(np.sqrt(np.mean((p - gt)**2))))
            if C > 0:
                m = _phys_metrics(p[temp_field_idx], gt[temp_field_idx], px_um)
                mp_maes.append(m['mp_mae'])
                kh_maes.append(m['kh_mae'])
                depths_p.append(m['depth_p']); depths_g.append(m['depth_g'])
                areas_p.append(m['area_p']);   areas_g.append(m['area_g'])

    # Ensemble mean MAE
    ens_mean = preds.mean(axis=1)   # (N, C, H, W)
    ens_mae  = float(np.abs(ens_mean - gts).mean())

    # Calibration: fraction of GT melt-pool depths inside the ensemble's mean ± 2*std, over frames whose
    # GT HAS a melt pool. A no-melt frame that the ensemble also leaves unmelted has mu = sig = 0 and
    # would count as "calibrated" trivially, inflating the score; such frames are excluded.
    # depths_p / depths_g were filled n-major, k-minor above, so they reshape to (N, K).
    n_melt_frames = 0
    if depths_g:
        dp = np.array(depths_p).reshape(N, K)
        dg = np.array(depths_g).reshape(N, K)[:, 0]
        melt = dg > 0
        n_melt_frames = int(melt.sum())
        if n_melt_frames:
            mu, sig = dp[melt].mean(axis=1), dp[melt].std(axis=1)
            calibration = float(np.mean(np.abs(dg[melt] - mu) <= 2 * sig))
        else:
            calibration = float('nan')
    else:
        calibration = float('nan')

    results = {
        'N': N, 'K': K,
        'MAE_mean': float(np.mean(maes)), 'MAE_std': float(np.std(maes)),
        'RMSE_mean': float(np.mean(rmses)),
        'MAE_ensemble_mean': ens_mae,
        'MP_MAE_mean': float(np.mean(mp_maes)) if mp_maes else float('nan'),
        'KH_MAE_mean': float(np.mean(kh_maes)) if kh_maes else float('nan'),
        'depth_mae_um': float(np.mean(np.abs(np.array(depths_p) - np.array(depths_g)))) if depths_p else float('nan'),
        'area_mae_um2': float(np.mean(np.abs(np.array(areas_p)  - np.array(areas_g))))  if areas_p  else float('nan'),
        'calibration_2sigma': calibration,
        # MP/KH/depth/area MAE average over ALL frames, including no-melt ones (where an unmelted
        # prediction scores 0); this count says how many frames actually carry a melt pool.
        'n_melt_frames': n_melt_frames,
    }
    return results


def evaluate(
    model,
    test_ds,
    test_loader,
    k_samples: int = 10,
    sampler: str = 'heun',
    n_steps: int = 20,
    eps0: float = 0.0,
    device: str = 'cuda',
    out_dir: str = '/tmp/si_eval',
    plot_frame_idx: int = 70,
    temp_field_idx: int = 0,
) -> dict:
    """
    Run full evaluation on the test set.

    Returns a results dict, also saves figures and .npz.
    """
    os.makedirs(out_dir, exist_ok=True)
    factor = test_ds.factor
    px_um = _px_um(factor)

    all_preds = []       # (N, K, C, H, W) physical units
    all_gts   = []       # (N, C, H, W) physical units
    all_ulrs  = []       # (N, C, H, W) physical units (upscaled LR)

    for i, (res, hr, lr, ulr) in enumerate(test_loader):
        with torch.no_grad():
            hr  = hr.to(device).float()
            lr  = lr.to(device).float()
            ulr = ulr.to(device).float()
            x_e = model.compute_x_e(lr, ulr)

            # Draw K samples
            sample_batch = []
            for _ in range(k_samples):
                imgs = model.batch_sample(
                    dataset=test_ds, batch=hr, x_e=x_e,
                    sampler=sampler, n_steps=n_steps,
                    upscaled_lr=ulr, eps0=eps0,
                    lr=lr,  # used only when si_base='enc'; the DDPM baselines take it via **kwargs
                )
                # (B, C, H, W) normalized. .cpu(): the SI samplers return host tensors, but the DDPM
                # baseline's DDIM sampler (evaluate_baseline_matched) returns them on the GPU.
                pred_norm = imgs[-1].detach().cpu()
                pred_phys = np.stack([
                    test_ds.unscale_data(pred_norm[s].numpy(), input_type='hr')
                    for s in range(pred_norm.shape[0])
                ])
                sample_batch.append(pred_phys)   # (B, C, H, W)

            # (B, K, C, H, W)
            samples_phys = np.stack(sample_batch, axis=1)

        gt_phys = np.stack([
            test_ds.unscale_data(hr.cpu().numpy()[s], input_type='hr')
            for s in range(hr.shape[0])
        ])
        ulr_phys = np.stack([
            test_ds.unscale_data(ulr.cpu().numpy()[s], input_type='upscaled_lr')
            for s in range(ulr.shape[0])
        ])

        all_preds.append(samples_phys)
        all_gts.append(gt_phys)
        all_ulrs.append(ulr_phys)
        print(f'  batch {i+1}/{len(test_loader)}', flush=True)

    # (N, K, C, H, W) and (N, C, H, W)
    preds = np.concatenate(all_preds, axis=0)
    gts   = np.concatenate(all_gts,   axis=0)
    ulrs  = np.concatenate(all_ulrs,  axis=0)
    N, K, C, H, W = preds.shape

    results = compute_metrics(preds, gts, px_um, temp_field_idx)
    ens_mean = preds.mean(axis=1)   # (N, C, H, W)
    ens_std  = preds.std(axis=1)    # (N, C, H, W) per-pixel ensemble std

    # ---- Overlay figures at plot_frame_idx ----
    idx = min(plot_frame_idx, N - 1)
    fig, axes = plt.subplots(2, K + 2, figsize=((K + 2) * 2.5, 5), dpi=120)
    vmin, vmax = 293.0, float(np.percentile(gts[:, temp_field_idx], 99))
    axes[0, 0].imshow(ulrs[idx, temp_field_idx].T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Upscaled LR', fontsize=7); axes[0, 0].axis('off')
    axes[0, 1].imshow(gts[idx, temp_field_idx].T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Ground truth', fontsize=7); axes[0, 1].axis('off')
    for k in range(K):
        axes[0, k+2].imshow(preds[idx, k, temp_field_idx].T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axes[0, k+2].set_title(f'Sample {k+1}', fontsize=7); axes[0, k+2].axis('off')
    # Std map row
    axes[1, 0].axis('off')
    axes[1, 1].imshow(ens_mean[idx, temp_field_idx].T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Ensemble mean', fontsize=7); axes[1, 1].axis('off')
    im_std = axes[1, 2].imshow(ens_std[idx, temp_field_idx].T, origin='lower', cmap='hot', vmin=0)
    axes[1, 2].set_title('Ensemble std', fontsize=7); axes[1, 2].axis('off')
    plt.colorbar(im_std, ax=axes[1, 2], fraction=0.046)
    for k in range(3, K+2):
        axes[1, k].axis('off')
    plt.suptitle(f'Frame {idx} — temperature (K)', fontsize=9)
    plt.tight_layout()
    overlay_path = os.path.join(out_dir, 'overlay_frame{}.png'.format(idx))
    plt.savefig(overlay_path, dpi=120, bbox_inches='tight'); plt.close('all')

    return results, preds, gts, ens_std, overlay_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name',       required=True)
    ap.add_argument('--wandb_run_name', default=None,
                    help='W&B display name of the training run; if omitted, eval results are '
                         'saved locally only (no artifact upload).')
    ap.add_argument('--model_dir',      required=True)
    ap.add_argument('--enc_dir',        default=None)
    ap.add_argument('--config',         default=None,
                    help='Training config YAML; if provided, data_root and field_names are '
                         'read from it when not given on the command line.')
    ap.add_argument('--data_root',      default=None,
                    help='Path to data directory (root_folder). Required if --config is not given.')
    ap.add_argument('--field_names',    nargs='+', default=None)
    ap.add_argument('--downscale',      default='direct')
    ap.add_argument('--sampler',        default='heun', choices=['heun', 'dopri5', 'em'])
    ap.add_argument('--n_steps',        type=int, default=20)
    ap.add_argument('--eps0',           type=float, default=0.0,
                    help='SDE diffusion coefficient for --sampler em (Option B only)')
    ap.add_argument('--k_samples',      type=int, default=10)
    ap.add_argument('--batch_size',     type=int, default=4)
    ap.add_argument('--sigma',          type=float, default=0.1)
    ap.add_argument('--gamma_c',        type=float, default=0.0)
    ap.add_argument('--si_cond',        default='ulr_concat',
                    choices=['ulr_concat', 'enc_implicit'])
    ap.add_argument('--si_base',        default='ulr', choices=['ulr', 'enc'],
                    help="start of the transport: bicubic upscale ('ulr') or the encoder's "
                         "prediction ('enc'). Must match training.")
    ap.add_argument('--fm_timescale',   type=float, default=1000.0)
    ap.add_argument('--plot_frame_idx', type=int, default=70)
    ap.add_argument('--out_dir',        required=True)
    ap.add_argument('--device',         default='cuda')
    ap.add_argument('--wandb_entity',   default=os.getenv('WANDB_ENTITY', ''))
    ap.add_argument('--wandb_project',  default='Flow3D_SuperResolution')
    ap.add_argument('--artifact_name',  default=None)
    args = ap.parse_args()

    # Fill data_root and field_names from config YAML when not given on CLI
    if args.config is not None:
        import yaml
        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)
        if args.data_root is None:
            args.data_root = _cfg.get('root_folder')
        if args.field_names is None:
            _raw = _cfg.get('fields', 'temperature')
            args.field_names = _raw.split('_') if isinstance(_raw, str) else list(_raw)
    if args.field_names is None:
        args.field_names = ['temperature']
    if args.data_root is None:
        ap.error('--data_root is required when --config is not provided')

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    from diffusionsr.datasets.flipped_dataset import FlippedDataset
    kw = dict(downscale_method=args.downscale, root_folder=args.data_root,
              normalize='standardize', n_steps=1, field_names=args.field_names, p_flip=0.0)
    train_ds = FlippedDataset(split='train', **kw)
    dev_ds   = FlippedDataset(split='dev',   **kw)
    test_ds  = FlippedDataset(split='test',  **kw)
    print(f'Fields: {test_ds.field_names}  factor={test_ds.factor}  n_test={len(test_ds)}')

    from diffusionsr.runners.train_coupled import CoupledInterpolantModel
    encoding = args.enc_dir is not None

    # Peek at the checkpoint to recover the U-Net dim that was used at training time.
    # This is necessary because DiffusionModel derives dim from dataset.img_shape, which
    # can differ between training and eval environments.
    best_path = os.path.join(args.model_dir, 'bestmodel_saved.pth')
    ckpt_path = os.path.join(args.model_dir, 'ckpt.pth')
    _load_path = best_path if os.path.exists(best_path) else ckpt_path
    _peek = torch.load(_load_path, map_location='cpu')
    # time_mlp.1 is Linear(dim, 4*dim) — shape[1] is the true Unet dim.
    # init_conv.weight.shape[0] is init_dim (defaults to 64, not dim).
    _unet_dim = int(_peek[0]['time_mlp.1.weight'].shape[1])
    del _peek
    print(f'Recovered U-Net dim={_unet_dim} from checkpoint')

    model = CoupledInterpolantModel(
        results_folder=args.model_dir,
        lr_encoder_folder=args.enc_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        encoding=encoding, sigma=args.sigma, gamma_c=args.gamma_c,
        si_cond=args.si_cond, si_base=args.si_base, fm_timescale=args.fm_timescale,
        image_size_override=_unet_dim,
        enc_output=False,  # SI training uses enc_output=False (init_dim=None → Unet default 64)
        device=device,
    )
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.model.load_state_dict(ckpt[0])
        print(f'Loaded bestmodel_saved.pth')
    else:
        model.load_saved_model()
    model.model.eval()

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    results, preds, gts, ens_std, overlay_path = evaluate(
        model, test_ds, test_loader,
        k_samples=args.k_samples, sampler=args.sampler,
        n_steps=args.n_steps, eps0=args.eps0, device=device,
        out_dir=args.out_dir, plot_frame_idx=args.plot_frame_idx,
    )

    import json
    print('\n=== RESULTS ===')
    print(json.dumps(results, indent=2))
    json_path = os.path.join(args.out_dir, f'{args.run_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    if args.wandb_run_name is None:
        print('--wandb_run_name not provided; skipping W&B artifact upload.')
        return

    # Upload artifact to existing W&B training run
    api = wandb.Api()
    existing = list(api.runs(f'{args.wandb_entity}/{args.wandb_project}',
                              filters={'display_name': args.wandb_run_name}))
    if existing:
        run = wandb.init(id=existing[0].id, resume='allow',
                         project=args.wandb_project, entity=args.wandb_entity)
    else:
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=f'{args.run_name}_eval', job_type='eval')

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, 'predictions.npz')
        np.savez(npz_path,
                 pred=preds.astype(np.float32),
                 gt=gts.astype(np.float32),
                 ens_std=ens_std.astype(np.float32),
                 field_names=np.array(args.field_names, dtype=object))
        art_name = args.artifact_name or f'{args.run_name}_eval_predictions'
        art = wandb.Artifact(name=art_name, type='eval_predictions',
                             description=f'SI eval — {args.run_name}')
        art.add_file(npz_path, name='predictions.npz')
        art.add_file(json_path, name='results.json')
        art.add_file(overlay_path, name='overlay.png')
        run.log_artifact(art, aliases=['latest'])
        run.finish()
    print('Artifact uploaded.')


if __name__ == '__main__':
    main()
