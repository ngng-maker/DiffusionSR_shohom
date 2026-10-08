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


def _mp_profile(temp: np.ndarray, threshold: float = 1900.0) -> np.ndarray:
    """
    Extract melt-pool depth profile as a function of x position.
    temp: (H, W) physical temperature array.
    Returns depth[x] = index of first pixel above threshold along z.
    """
    from diffusionsr.analysis.analysis_functions import get_profile
    # get_profile expects (1, H, W); returns (mp_profile, kh_profile)
    mp, kh = get_profile(temp[None])
    return np.array(mp), np.array(kh)


def _phys_metrics(pred_t: np.ndarray, gt_t: np.ndarray, px_um: float):
    """
    Compute melt-pool metrics for a single (H, W) temperature field pair.
    Returns dict of scalars.
    """
    mp_p, kh_p = _mp_profile(pred_t)
    mp_g, kh_g = _mp_profile(gt_t)

    # Depth (minimum depth of melt pool surface, i.e. deepest extent)
    depth_p = float(np.max(mp_p) * px_um) if len(mp_p) > 0 else 0.0
    depth_g = float(np.max(mp_g) * px_um) if len(mp_g) > 0 else 0.0

    # Width (number of x-positions above threshold)
    width_p = float(len(mp_p) * px_um)
    width_g = float(len(mp_g) * px_um)

    # Area (depth × width, µm²)
    area_p = depth_p * width_p
    area_g = depth_g * width_g

    mp_mae = abs(depth_p - depth_g)   # µm
    kh_depth_p = float(np.max(kh_p) * px_um) if len(kh_p) > 0 else 0.0
    kh_depth_g = float(np.max(kh_g) * px_um) if len(kh_g) > 0 else 0.0
    kh_mae = abs(kh_depth_p - kh_depth_g)

    return {
        'mp_mae': mp_mae, 'kh_mae': kh_mae,
        'depth_p': depth_p, 'depth_g': depth_g,
        'area_p': area_p, 'area_g': area_g,
    }


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
                )
                pred_norm = imgs[-1]   # (B, C, H, W) normalized
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
    ens_std  = preds.std(axis=1)    # (N, C, H, W) per-pixel ensemble std

    # Calibration: fraction of GT keyhole depths inside mean ± 2*std
    if depths_g:
        # Recompute per-simulation: for each n, compare GT depth against ensemble mean ± 2 std
        calib_list = []
        for n in range(N):
            gt_d = _phys_metrics(gts[n, temp_field_idx], gts[n, temp_field_idx], px_um)['depth_g']
            ens_depths = [
                _phys_metrics(preds[n, k, temp_field_idx], gts[n, temp_field_idx], px_um)['depth_p']
                for k in range(K)
            ]
            mu = np.mean(ens_depths); sig = np.std(ens_depths)
            calib_list.append(float(abs(gt_d - mu) <= 2 * sig))
        calibration = float(np.mean(calib_list))
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
    }

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
    ap.add_argument('--wandb_run_name', required=True)
    ap.add_argument('--model_dir',      required=True)
    ap.add_argument('--enc_dir',        default=None)
    ap.add_argument('--config',         default=None, help='Training config YAML; read for encoder_type/kwargs')
    ap.add_argument('--data_root',      required=True)
    ap.add_argument('--field_names',    nargs='+', default=['temperature'])
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
    ap.add_argument('--fm_timescale',   type=float, default=1000.0)
    ap.add_argument('--plot_frame_idx', type=int, default=70)
    ap.add_argument('--out_dir',        required=True)
    ap.add_argument('--device',         default='cuda')
    ap.add_argument('--wandb_entity',   default=os.getenv('WANDB_ENTITY', ''))
    ap.add_argument('--wandb_project',  default='Flow3D_SuperResolution')
    ap.add_argument('--artifact_name',  default=None)
    args = ap.parse_args()

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
    _unet_dim = int(_peek[0]['init_conv.weight'].shape[0])
    del _peek
    print(f'Recovered U-Net dim={_unet_dim} from checkpoint')

    model = CoupledInterpolantModel(
        results_folder=args.model_dir,
        lr_encoder_folder=args.enc_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        encoding=encoding, sigma=args.sigma, gamma_c=args.gamma_c,
        si_cond=args.si_cond, fm_timescale=args.fm_timescale,
        image_size_override=_unet_dim,
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
