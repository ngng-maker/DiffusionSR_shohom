"""
Precompute QSS sample predictions and ensemble std maps for all ablation_v2 runs.

Saves {run_dir}/preview/qss_samples.npz with keys:
  preds   (N_QSS, C, H, W)  -- mean prediction, physical units
  gts     (N_QSS, C, H, W)  -- ground truth, physical units
  ulrs    (N_QSS, C, H, W)  -- upscaled LR input, physical units
  stds    (N_QSS, H, W)     -- per-pixel std across N_SAMPLES draws, temperature channel
  samples (N_SAMPLES, N_QSS, C, H, W) -- all individual draws, physical units

Run as a SLURM job:
  sbatch --partition=batch --gres=gpu:1 --time=12:00:00 \\
    --wrap="cd /trace/.../DiffusionSR_shohom; /path/to/python scripts/precompute_ablation_v2_preview.py"

Optional args:
  --variants   space-separated list of variant names to process (default: all 24)
  --n_qss      number of QSS test frames (default: 5)
  --n_samples  ensemble samples per frame (default: 5)
  --skip       DDIM skip interval (default: 20)
  --force      recompute even if qss_samples.npz already exists
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

RUNS_ROOT = Path('/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom/diffusionsr/runs/ablation_v2')
DATA_ROOT = Path('/trace/group/forgelab/ngng/multifield/data_fields')

MODELS = ['diffusion', 'fm', 'ldm']
ENCS   = ['enc', 'noenc']
FIELDS = ['temp', 'multi']
NORMS  = ['std', 'gstd']

FIELD_NAMES = {'temp': ['temperature'], 'multi': ['temperature', 'sdfliqlabel']}
NORM_KEY    = {'std': 'standardize',    'gstd': 'global_standardize'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='*', default=None,
                   help='Variant names to process. Default: all 24.')
    p.add_argument('--n_qss',    type=int, default=5)
    p.add_argument('--n_samples',type=int, default=5)
    p.add_argument('--skip',     type=int, default=20,
                   help='DDIM skip interval (ignored for flow-matching).')
    p.add_argument('--force',    action='store_true',
                   help='Recompute even if qss_samples.npz exists.')
    p.add_argument('--runs_root', type=str, default=None,
                   help='Path to ablation_v2 runs directory. Default: TRACE path.')
    p.add_argument('--data_root', type=str, default=None,
                   help='Path to data_fields directory. Default: TRACE path.')
    return p.parse_args()


def load_datasets(f, n):
    from diffusionsr.datasets.flipped_dataset import FlippedDataset
    common = dict(downscale_method='direct', root_folder=str(DATA_ROOT),
                  normalize=NORM_KEY[n], n_steps=5, field_names=FIELD_NAMES[f], p_flip=0.0)
    train_ds = FlippedDataset(split='train', **common)
    dev_ds   = FlippedDataset(split='dev',   **common)
    test_ds  = FlippedDataset(split='test',  **common)
    return train_ds, dev_ds, test_ds


def load_model(m, e, f, n, run_dir, train_ds, dev_ds, test_ds, device):
    enc_dir = run_dir / 'encoder' if e == 'enc' else None
    try:
        if m == 'ldm':
            from diffusionsr.runners.train_ldm import LDMModel
            cfg_path = run_dir / 'configuration.yml'
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
            vae_dir = cfg.get('vae_results_dir', str(run_dir))
            model = LDMModel(vae_folder=vae_dir, results_folder=str(run_dir),
                             lr_encoder_folder=str(enc_dir) if enc_dir else None,
                             train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
                             encoding=(e == 'enc'), device=device)
        elif m == 'fm':
            from diffusionsr.runners.train_flow_matching import FlowMatchingModel
            model = FlowMatchingModel(results_folder=str(run_dir),
                                      lr_encoder_folder=str(enc_dir) if enc_dir else None,
                                      train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
                                      encoding=(e == 'enc'), device=device)
        else:
            from diffusionsr.runners.train_diffusion import DiffusionModel
            model = DiffusionModel(results_folder=str(run_dir),
                                   lr_encoder_folder=str(enc_dir) if enc_dir else None,
                                   train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
                                   encoding=(e == 'enc'), device=device)

        best_path = run_dir / 'bestmodel_saved.pth'
        load_path = best_path if best_path.exists() else run_dir / 'ckpt.pth'
        if load_path.exists():
            ckpt = torch.load(load_path, map_location=device)
            state = ckpt[0] if isinstance(ckpt, (list, tuple)) else ckpt.get('model', ckpt)
            model.model.load_state_dict(state)
        model.model.eval()
        return model
    except Exception as exc:
        print(f'  [WARN] Model load failed: {exc}')
        return None


def compute_variant(m, e, f, n, run_dir, args, device):
    out_dir = run_dir / 'preview'
    out_file = out_dir / 'qss_samples.npz'
    if out_file.exists() and not args.force:
        print(f'  Skipping (already exists): {out_file}')
        return True

    print(f'  Loading datasets ({f}, {n})...')
    try:
        train_ds, dev_ds, test_ds = load_datasets(f, n)
    except Exception as exc:
        print(f'  [WARN] Dataset load failed: {exc}')
        return False

    print(f'  Loading model {m}/{e}/{f}/{n}...')
    model = load_model(m, e, f, n, run_dir, train_ds, dev_ds, test_ds, device)
    if model is None:
        return False

    n_ch      = len(FIELD_NAMES[f])
    qss_idx   = list(range(len(test_ds) - args.n_qss, len(test_ds)))
    sampler   = 'euler' if m == 'fm' else 'DDIM'
    skip      = args.skip

    all_preds, gts, ulrs, all_samples = [], [], [], []

    for idx in qss_idx:
        print(f'    frame {idx} ({qss_idx.index(idx)+1}/{args.n_qss})')
        res, hr, lr, ulr = test_ds[idx]
        hr_last  = hr[-n_ch:].unsqueeze(0).to(device).float()
        ulr_last = ulr[-n_ch:].unsqueeze(0).to(device).float()
        lr_t     = lr.unsqueeze(0).to(device).float()
        ulr_full = ulr.unsqueeze(0).to(device).float()

        frame_samples = []
        with torch.no_grad():
            for s_i in range(args.n_samples):
                raw = model.batch_sample(ulr_full, lr_t, sampler=sampler, skip=skip)
                # raw may be (T,B,C,H,W) for DDIM or (B,C,H,W) for final-only
                if raw.ndim == 5:
                    raw = raw[-1]  # take last reverse step
                pred_phys = test_ds.unscale_data(raw.cpu().numpy()[0], input_type='hr')
                frame_samples.append(pred_phys)  # (C, H, W)

        frame_samples_arr = np.stack(frame_samples)  # (N_SAMPLES, C, H, W)
        mean_pred = frame_samples_arr.mean(axis=0)   # (C, H, W)

        gt_phys  = test_ds.unscale_data(hr_last.cpu().numpy()[0],  input_type='hr')
        ulr_phys = test_ds.unscale_data(ulr_last.cpu().numpy()[0], input_type='upscaled_lr')

        all_preds.append(mean_pred)
        gts.append(gt_phys)
        ulrs.append(ulr_phys)
        all_samples.append(frame_samples_arr)

    preds_arr   = np.stack(all_preds)    # (N_QSS, C, H, W)
    gts_arr     = np.stack(gts)
    ulrs_arr    = np.stack(ulrs)
    samples_arr = np.stack(all_samples)  # (N_QSS, N_SAMPLES, C, H, W)
    stds_arr    = samples_arr[:, :, 0, :, :].std(axis=1)  # temperature ch, (N_QSS, H, W)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_file),
                        preds=preds_arr, gts=gts_arr, ulrs=ulrs_arr,
                        samples=samples_arr, stds=stds_arr)
    print(f'  Saved → {out_file}')
    return True


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    runs_root = Path(args.runs_root) if args.runs_root else RUNS_ROOT
    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    print(f'runs_root: {runs_root}')
    print(f'data_root: {data_root}')

    # Monkey-patch the module-level constants so helper functions pick them up
    global RUNS_ROOT, DATA_ROOT
    RUNS_ROOT = runs_root
    DATA_ROOT = data_root

    # Build variant list
    all_variants = [f'{m}_{e}_{f}_{n}' for m in MODELS for e in ENCS for f in FIELDS for n in NORMS]
    variants = args.variants if args.variants else all_variants

    for variant in variants:
        run_dir = RUNS_ROOT / variant
        if not run_dir.exists():
            print(f'[SKIP] {variant}: directory not found')
            continue
        if not (run_dir / 'bestmodel_saved.pth').exists() and not (run_dir / 'ckpt.pth').exists():
            print(f'[SKIP] {variant}: no checkpoint')
            continue

        # Parse variant name
        parts = variant.split('_', 3)  # e.g. diffusion_enc_temp_std
        if len(parts) != 4:
            print(f'[SKIP] {variant}: unexpected name format')
            continue
        m, e, f, n = parts

        print(f'\n=== {variant} ===')
        ok = compute_variant(m, e, f, n, run_dir, args, device)
        print(f'  {"OK" if ok else "FAILED"}')

    print('\nDone.')


if __name__ == '__main__':
    main()
