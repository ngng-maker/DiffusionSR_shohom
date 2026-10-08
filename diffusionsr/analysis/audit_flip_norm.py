"""
Audit the flip / normalization ordering in DiffusionSR.

Code-inspection findings (documented in AUDIT_FLIP_NORM.md):
  - CNN encoder (train_rrdn_encoder.py:146): flip with p=0.2 applied AFTER the
    dataset has already returned normalized tensors.  Only `hr` and `lr` (true LR)
    are flipped; `upscaled_lr` and `residual` are not.
  - Diffusion / FM / LDM (train_diffusion.py): NO flip augmentation at all.
  - The flip axis is dim=2 (W = z / depth), not dim=1 (H = x / laser direction).

This script quantifies how much the ordering error matters:
  1. Loads cached pixel-wise statistics (mean_hr, std_hr, mean_upscaled_lr,
     std_upscaled_lr) and measures their asymmetry along the flipped axis.
  2. Samples N training frames and computes the per-pixel difference between
     "normalize then flip" (what the code does) vs "flip then normalize" (correct),
     in normalized units and in Kelvin, split by melt-pool vs background region.
  3. Saves figures and a JSON summary, then uploads as a W&B artifact.

Usage (on TRACE):
  cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
  conda activate <env>

  # Single-field (temperature only)
  python -m diffusionsr.analysis.audit_flip_norm \\
      --data_root    /trace/group/forgelab/ngng/multifield/data_fields \\
      --downscale    direct \\
      --field_names  temperature \\
      --tag          single_field \\
      --out_dir      /trace/group/forgelab/ngng/multifield/eval_outputs/audit_flip_norm \\
      --wandb_entity <entity> \\
      --n_samples    50

  # Multifield (temperature + sdfliqlabel)
  python -m diffusionsr.analysis.audit_flip_norm \\
      --data_root    /trace/group/forgelab/ngng/multifield/data_fields \\
      --downscale    direct \\
      --field_names  temperature sdfliqlabel \\
      --tag          multifield \\
      --out_dir      /trace/group/forgelab/ngng/multifield/eval_outputs/audit_flip_norm \\
      --wandb_entity <entity> \\
      --n_samples    50
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
def _find_root():
    s = Path(__file__).resolve()
    for p in [s, *s.parents]:
        if (p / 'setup.py').exists() and (p / 'diffusionsr').exists():
            return p
    raise RuntimeError('Cannot find project root')

PROJECT_ROOT = _find_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flip_w(arr: np.ndarray) -> np.ndarray:
    """Mirror along axis=2 (W = z/depth) — matching training code: torch.flip(hr, dims=[2])."""
    return np.flip(arr, axis=2).copy()


def _savefig(path: str, fig=None, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    (fig or plt).savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close('all')


def _melt_mask(temp_phys: np.ndarray, threshold: float = 1900.0) -> np.ndarray:
    """Boolean (H, W) mask where T > threshold K."""
    return temp_phys > threshold


# ---------------------------------------------------------------------------
# 1. Statistics asymmetry
# ---------------------------------------------------------------------------

def analyse_stats_asymmetry(ds, tag: str, out_dir: str) -> dict:
    """
    Compare each cached statistic map against its W-axis flip.
    ds must have attributes: mean_hr, std_hr, mean_upscaled_lr, std_upscaled_lr, field_names.
    All arrays are (C, H, W) in physical-unit scale (K, etc.).
    """
    field_names = ds.field_names
    pairs = [
        (ds.mean_hr,          ds.std_hr,          'mean_hr',          'HF'),
        (ds.mean_upscaled_lr, ds.std_upscaled_lr,  'mean_upscaled_lr', 'Upscaled-LR'),
    ]
    results = {}

    for mean_arr, std_arr, key, label in pairs:
        mean = np.array(mean_arr)   # (C, H, W)
        std  = np.array(std_arr)

        mean_flip = _flip_w(mean)
        abs_diff  = np.abs(mean - mean_flip)   # (C, H, W)
        std_safe  = np.where(std > 1e-6, std, 1e-6)
        rel_diff  = abs_diff / std_safe

        results[key] = {}
        for fi, fname in enumerate(field_names):
            ad = abs_diff[fi]   # (H, W)
            rd = rel_diff[fi]
            max_idx = np.unravel_index(np.argmax(ad), ad.shape)

            results[key][fname] = {
                'max_abs_diff_K':  float(ad.max()),
                'mean_abs_diff_K': float(ad.mean()),
                'max_rel_diff':    float(rd.max()),
                'mean_rel_diff':   float(rd.mean()),
                'max_asym_loc':    [int(max_idx[0]), int(max_idx[1])],
            }

            # Figure: mean | flipped mean | |diff|
            m_ch = mean[fi]
            vmin = float(np.percentile(m_ch, 1))
            vmax = float(np.percentile(m_ch, 99))
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
            axes[0].imshow(m_ch.T,         origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
            axes[0].set_title(f'{label} mean', fontsize=9); axes[0].axis('off')
            axes[1].imshow(mean_flip[fi].T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
            axes[1].set_title(f'{label} mean (W-flipped)', fontsize=9); axes[1].axis('off')
            im = axes[2].imshow(ad.T, origin='lower', cmap='hot', vmin=0)
            axes[2].set_title('|diff| (K)', fontsize=9); axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046)
            plt.suptitle(f'[{tag}] {key} — {fname} — W-axis asymmetry', fontsize=10)
            plt.tight_layout()
            _savefig(os.path.join(out_dir, f'{tag}_{key}_{fname}_asymmetry.png'), fig)

            # Figure: relative asymmetry (in std units)
            fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=150)
            im2 = ax2.imshow(rd.T, origin='lower', cmap='hot', vmin=0)
            plt.colorbar(im2, ax=ax2, label='|diff| / std')
            ax2.set_title(f'[{tag}] {key} {fname} — asymmetry / std', fontsize=9)
            ax2.axis('off')
            plt.tight_layout()
            _savefig(os.path.join(out_dir, f'{tag}_{key}_{fname}_asymmetry_rel.png'), fig2)

    return results


# ---------------------------------------------------------------------------
# 2. Effect-size: normalize-then-flip vs flip-then-normalize
# ---------------------------------------------------------------------------

def analyse_effect_size(ds, tag: str, out_dir: str, n_samples: int = 50,
                        seed: int = 42) -> dict:
    """
    For N random training samples, measure the per-pixel error from doing the
    flip AFTER normalization (code behaviour) vs BEFORE normalization (correct).

    Strategy:
      - The dataset already returns normalized tensors (norm_hr).
      - Unscale back to physical units via ds.unscale_data(norm_hr, 'hr').
      - flip-then-normalize (FtN, correct):  (flip(phys) - mean) / std
      - normalize-then-flip (NtF, code bug): flip(norm_hr) == flip((phys-mean)/std)
      - Error = |FtN - NtF|
    """
    np.random.seed(seed)
    field_names = ds.field_names
    temp_idx = field_names.index('temperature') if 'temperature' in field_names else 0

    mean_hr  = np.array(ds.mean_hr)   # (C, H, W)
    std_hr   = np.array(ds.std_hr)
    std_safe = np.where(std_hr > 1e-6, std_hr, 1e-6)

    indices = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    errs_norm, errs_K = [], []
    errs_mp_norm, errs_bg_norm = [], []
    errs_mp_K,   errs_bg_K    = [], []

    ex_ftn = ex_ntf = ex_diff_K = ex_phys = None   # saved for illustration

    for i, idx in enumerate(indices):
        sample  = ds[idx]
        norm_hr = np.array(sample[1])   # (C, H, W) normalized

        # Unscale to physical units
        phys_hr = np.array(ds.unscale_data(norm_hr, input_type='hr'))   # (C, H, W)

        # flip-then-normalize (correct)
        ftn = (np.flip(phys_hr, axis=2) - mean_hr) / std_safe

        # normalize-then-flip (code bug) — identical to flipping the already-normed tensor
        ntf = np.flip(norm_hr, axis=2)

        err      = np.abs(ftn - ntf)
        err_phys = err * std_safe   # convert back to Kelvin-scale error

        errs_norm.append(float(err.mean()))
        errs_K.append(float(err_phys.mean()))

        # Split by melt-pool region
        temp_ftn_K = ftn[temp_idx] * std_safe[temp_idx] + mean_hr[temp_idx]
        mp = _melt_mask(temp_ftn_K)
        bg = ~mp
        if mp.any():
            errs_mp_norm.append(float(err[temp_idx][mp].mean()))
            errs_mp_K.append(float(err_phys[temp_idx][mp].mean()))
        if bg.any():
            errs_bg_norm.append(float(err[temp_idx][bg].mean()))
            errs_bg_K.append(float(err_phys[temp_idx][bg].mean()))

        if i == 0:
            ex_ftn    = ftn.copy()
            ex_ntf    = ntf.copy()
            ex_diff_K = err_phys.copy()
            ex_phys   = phys_hr.copy()

    def _nm(lst):  return float(np.mean(lst))        if lst else float('nan')
    def _p95(lst): return float(np.percentile(lst, 95)) if lst else float('nan')

    results = {
        'n_samples':        len(indices),
        'mean_err_norm':    _nm(errs_norm),
        'mean_err_K':       _nm(errs_K),
        'mean_err_mp_norm': _nm(errs_mp_norm),
        'mean_err_mp_K':    _nm(errs_mp_K),
        'mean_err_bg_norm': _nm(errs_bg_norm),
        'mean_err_bg_K':    _nm(errs_bg_K),
        'p95_err_K':        _p95(errs_K),
    }

    # Figure: per-sample error distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].hist(errs_norm, bins=20, color='steelblue', edgecolor='w', alpha=0.85)
    axes[0].axvline(results['mean_err_norm'], color='red', lw=1.5,
                    label=f"mean={results['mean_err_norm']:.4f}")
    axes[0].set_xlabel('Mean error (norm units)'); axes[0].set_ylabel('Count')
    axes[0].set_title(f'[{tag}] NtF vs FtN — normalized'); axes[0].legend(fontsize=8)

    axes[1].hist(errs_K, bins=20, color='tomato', edgecolor='w', alpha=0.85)
    axes[1].axvline(results['mean_err_K'], color='darkred', lw=1.5,
                    label=f"mean={results['mean_err_K']:.1f} K")
    axes[1].set_xlabel('Mean error (K)')
    axes[1].set_title(f'[{tag}] NtF vs FtN — Kelvin'); axes[1].legend(fontsize=8)
    plt.tight_layout()
    _savefig(os.path.join(out_dir, f'{tag}_effect_size.png'), fig)

    # Figure: by region
    cats   = ['Overall', 'Melt pool', 'Background']
    values = [results['mean_err_K'], results['mean_err_mp_K'], results['mean_err_bg_K']]
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax2.bar(cats, values, color=['#4c72b0', '#dd8452', '#55a868'],
                   edgecolor='w', alpha=0.9)
    ax2.set_ylabel('Mean |error| (K)')
    ax2.set_title(f'[{tag}] NtF vs FtN error by region')
    for bar, val in zip(bars, values):
        if not (np.isnan(val) or val == 0):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', fontsize=9)
    plt.tight_layout()
    _savefig(os.path.join(out_dir, f'{tag}_effect_size_by_region.png'), fig2)

    # Figure: one example frame
    if ex_diff_K is not None:
        fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
        t_phys = ex_phys[temp_idx]
        im0 = axes3[0].imshow(t_phys.T, origin='lower', cmap='jet', vmin=293, vmax=5000)
        axes3[0].set_title('Physical HF temp (K)', fontsize=9); axes3[0].axis('off')
        plt.colorbar(im0, ax=axes3[0], fraction=0.046)

        t_ftn_K = ex_ftn[temp_idx] * std_safe[temp_idx] + mean_hr[temp_idx]
        im1 = axes3[1].imshow(t_ftn_K.T, origin='lower', cmap='jet', vmin=293, vmax=5000)
        axes3[1].set_title('FtN (correct) in K', fontsize=9); axes3[1].axis('off')
        plt.colorbar(im1, ax=axes3[1], fraction=0.046)

        im2 = axes3[2].imshow(ex_diff_K[temp_idx].T, origin='lower', cmap='hot', vmin=0)
        axes3[2].set_title('|FtN − NtF| (K)', fontsize=9); axes3[2].axis('off')
        plt.colorbar(im2, ax=axes3[2], fraction=0.046)

        plt.suptitle(f'[{tag}] Example ordering-error map', fontsize=10)
        plt.tight_layout()
        _savefig(os.path.join(out_dir, f'{tag}_example_error_map.png'), fig3)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Audit flip/normalization ordering in DiffusionSR')
    ap.add_argument('--data_root',     required=True,  help='Root folder of the dataset')
    ap.add_argument('--downscale',     default='direct')
    ap.add_argument('--field_names',   nargs='+', default=['temperature'])
    ap.add_argument('--tag',           default='audit',
                    help='Label for file names (e.g. single_field, multifield)')
    ap.add_argument('--out_dir',       required=True,  help='Output directory for figures/JSON')
    ap.add_argument('--wandb_entity',  default=os.getenv('WANDB_ENTITY', ''))
    ap.add_argument('--wandb_project', default='Flow3D_SuperResolution')
    ap.add_argument('--n_samples',     type=int, default=50)
    ap.add_argument('--seed',          type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from diffusionsr.datasets.dataset import SimulationXZDataset
    print(f'Loading dataset [{args.tag}]: {args.data_root}  fields={args.field_names}')
    ds = SimulationXZDataset(
        downscale_method=args.downscale,
        root_folder=args.data_root,
        split='train',
        normalize='standardize',
        field_names=args.field_names,
        n_steps=1,
    )
    print(f'  {len(ds)} training samples  factor={ds.factor}')

    print('--- Asymmetry analysis ---')
    asym = analyse_stats_asymmetry(ds, args.tag, args.out_dir)

    print('--- Effect-size analysis ---')
    effect = analyse_effect_size(ds, args.tag, args.out_dir, args.n_samples, args.seed)

    summary = {
        'tag':         args.tag,
        'data_root':   args.data_root,
        'field_names': args.field_names,
        'asymmetry':   asym,
        'effect_size': effect,
    }
    json_path = os.path.join(args.out_dir, f'{args.tag}_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary written → {json_path}')

    if args.wandb_entity:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f'audit_flip_norm_{args.tag}',
            job_type='audit',
        )
        art = wandb.Artifact(
            name=f'audit_flip_norm_{args.tag}',
            type='audit',
            description=f'Flip/normalization order audit — {args.tag}',
        )
        for fname in sorted(os.listdir(args.out_dir)):
            if fname.startswith(args.tag) and (fname.endswith('.png') or fname.endswith('.json')):
                art.add_file(os.path.join(args.out_dir, fname))
        run.log_artifact(art, aliases=['latest'])
        run.finish()
        print(f'Artifact audit_flip_norm_{args.tag} uploaded to W&B.')
    else:
        print('WANDB_ENTITY not set — skipping W&B upload.')

    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
