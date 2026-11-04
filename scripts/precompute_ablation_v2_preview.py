"""
Precompute QSS sample predictions and ensemble std maps for all ablation_v2 runs.

Saves {run_dir}/preview/qss_samples.npz with keys:
  preds   (N_QSS, C, H, W)  -- mean prediction, physical units
  gts     (N_QSS, C, H, W)  -- ground truth, physical units
  ulrs    (N_QSS, C, H, W)  -- upscaled LR input, physical units
  stds    (N_QSS, H, W)     -- per-pixel std across N_SAMPLES draws, temperature channel
  samples (N_QSS, N_SAMPLES, C, H, W) -- all individual draws, physical units
  sampling_* (scalars) -- timesteps/schedule/sampler/steps used; read from the run's training config.
                          Existing previews without matching stamps are recomputed automatically.

Run as a SLURM job:
  sbatch --partition=batch --gres=gpu:1 --time=12:00:00 \\
    --wrap="cd /trace/.../DiffusionSR_shohom; /path/to/python scripts/precompute_ablation_v2_preview.py"

Optional args:
  --variants   space-separated list of variant names to process (default: all 36; missing run dirs are skipped)
  --n_qss      number of QSS test frames (default: 5)
  --n_samples  ensemble samples per frame (default: 5)
  --sampler    DiffusionSR/LDM sampler: ddpm (default; ancestral, all `timesteps` steps) or ddim (eta=0)
  --skip       DDIM skip interval, only with --sampler ddim (default: 20)

DDPM is the default since 2026-11-03: on the v2 encoder checkpoints, deterministic DDIM (eta=0) left
stray wrong pixels that more DDIM steps did not remove (50/200/1000 steps identical), while ancestral
DDPM removed them and cut MP-MAE 0.60 -> 0.44 (scripts/diag_ablation_v2_sampling.py). Flow-matching
always uses its Euler sampler. Previews stamped with a different sampler are recomputed automatically.
  --force      recompute even if qss_samples.npz already exists
"""
import argparse
import sys
import os
from pathlib import Path

os.environ.setdefault('TQDM_DISABLE', '1')  # DiffusionModel.p_sample_loop wraps its 1000 steps in tqdm

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
ENCS   = ['enc', 'noenc', 'lrcond']  # lrcond: no encoder, conditioned on upscaled LR
FIELDS = ['temp', 'multi']
NORMS  = ['std', 'gstd']

FIELD_NAMES = {'temp': ['temperature'], 'multi': ['temperature', 'sdfliqlabel']}
NORM_KEY    = {'std': 'standardize',    'gstd': 'global_standardize'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='*', default=None,
                   help='Variant names to process. Default: all 36 (missing run dirs are skipped).')
    p.add_argument('--n_qss',    type=int, default=5)
    p.add_argument('--n_samples',type=int, default=5)
    p.add_argument('--sampler',  choices=['ddpm', 'ddim'], default='ddpm',
                   help='DiffusionSR/LDM sampler (flow-matching always uses Euler). Default: ddpm.')
    p.add_argument('--skip',     type=int, default=20,
                   help='DDIM skip interval (only with --sampler ddim).')
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


REPO_CFG_DIR = REPO_ROOT / 'diffusionsr' / 'configs' / 'ablation_v2'


def load_run_config(variant, run_dir):
    """Return (cfg, path) for the config this run was TRAINED with.

    Prefers the copy train_srdiff saved next to the checkpoint, then the repo copy. Raises rather
    than guessing: building a model without the trained `timesteps` silently falls back to the
    constructor default (200), which samples a 1000-step model with the wrong noise schedule.
    """
    for path in (run_dir / 'configuration.yml', REPO_CFG_DIR / f'{variant}.yml'):
        if path.exists():
            cfg = yaml.safe_load(path.read_text()) or {}
            if 'timesteps' in cfg:
                return cfg, path
    raise ValueError(f'{variant}: no training config with `timesteps` found '
                     f'(looked in {run_dir / "configuration.yml"} and {REPO_CFG_DIR / (variant + ".yml")})')


def sampling_settings(m, cfg, skip, sampler='ddpm'):
    """Schedule/sampler settings matching training (mirrors train_srdiff.py's defaults)."""
    timesteps = int(cfg['timesteps'])
    s = {'timesteps': timesteps, 'schedule': cfg.get('schedule', 'linear')}
    if m == 'fm':
        s['sampler'] = 'euler'
        s['fm_timescale'] = float(cfg.get('fm_timescale', timesteps))
        s['n_sampling_steps'] = int(cfg.get('fm_n_steps', 100))
    elif sampler == 'ddpm':
        s['sampler'] = 'DDPM'
        s['n_sampling_steps'] = timesteps
    else:
        s['sampler'] = 'DDIM'
        s['skip'] = skip
        s['n_sampling_steps'] = len(range(0, timesteps, skip))
    return s


def load_model(m, e, f, n, run_dir, train_ds, dev_ds, test_ds, device, cfg, settings):
    enc_dir = run_dir / 'encoder' if e == 'enc' else None
    conditioning = cfg.get('conditioning', 'implicit')
    # timesteps/schedule MUST match training: they define the DDPM noise schedule (diffusion, LDM)
    # and, via fm_timescale, the time-embedding scale (FM).
    common = dict(results_folder=str(run_dir),
                  lr_encoder_folder=str(enc_dir) if enc_dir else None,
                  train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
                  encoding=(e == 'enc'), device=device, conditioning=conditioning,
                  timesteps=settings['timesteps'], schedule=settings['schedule'])
    try:
        if m == 'ldm':
            from diffusionsr.runners.train_ldm import LDMModel
            vae_dir = cfg.get('vae_results_dir', str(run_dir))
            model = LDMModel(vae_folder=vae_dir, **common)
        elif m == 'fm':
            from diffusionsr.runners.train_flow_matching import FlowMatchingModel
            model = FlowMatchingModel(enc_output=False, fm_timescale=settings['fm_timescale'], **common)
        else:
            from diffusionsr.runners.train_diffusion import DiffusionModel
            model = DiffusionModel(enc_output=False, **common)
        assert model.timesteps == settings['timesteps'], (model.timesteps, settings)
        if m == 'fm':
            assert model.fm_timescale == settings['fm_timescale'], (model.fm_timescale, settings)

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


def draw_samples(model, m, settings, hr_batch, x_e, ds):
    """One reverse chain per batch row; returns the final (B, C, H, W) normalised samples."""
    if settings['sampler'] == 'euler':
        out = model.batch_sample(ds, hr_batch, x_e, sampler='euler', n_steps=settings['n_sampling_steps'])
    elif settings['sampler'] == 'DDIM':
        out = model.batch_sample(ds, hr_batch, x_e, sampler='DDIM', skip=settings['skip'])
    elif m == 'ldm':  # LDMModel.batch_sample('DDPM') samples at latent size and decodes
        out = model.batch_sample(ds, hr_batch, x_e, sampler='DDPM')
    else:
        # DiffusionModel.batch_sample('DDPM') hard-codes batch_size=2, so call the ancestral loop directly.
        out = model.p_sample_loop(model.model, x_e, shape=tuple(hr_batch.shape), timesteps=model.timesteps)
    return out[-1]  # every path returns the per-step trajectory; keep the last reverse step


def compute_variant(m, e, f, n, run_dir, args, device):
    out_dir = run_dir / 'preview'
    out_file = out_dir / 'qss_samples.npz'
    variant = f'{m}_{e}_{f}_{n}'
    try:
        cfg, cfg_path = load_run_config(variant, run_dir)
    except ValueError as exc:
        print(f'  [WARN] {exc}')
        return False
    settings = sampling_settings(m, cfg, args.skip, args.sampler)
    print(f'  Sampling settings (from {cfg_path}): {settings}')

    if out_file.exists() and not args.force:
        # Only reuse a preview stamped with these exact sampling settings; unstamped files predate the
        # timesteps fix (sampled with T=200) and are always recomputed.
        with np.load(str(out_file)) as old:
            stamped = {k[len('sampling_'):]: old[k].item() for k in old.files if k.startswith('sampling_')}
        if stamped == settings:
            print(f'  Skipping (already exists with matching sampling settings): {out_file}')
            return True
        print(f'  Recomputing stale preview (stamped {stamped or "none"}): {out_file}')

    print(f'  Loading datasets ({f}, {n})...')
    try:
        train_ds, dev_ds, test_ds = load_datasets(f, n)
    except Exception as exc:
        print(f'  [WARN] Dataset load failed: {exc}')
        return False

    print(f'  Loading model {m}/{e}/{f}/{n}...')
    model = load_model(m, e, f, n, run_dir, train_ds, dev_ds, test_ds, device, cfg, settings)
    if model is None:
        return False

    n_ch      = len(FIELD_NAMES[f])
    qss_idx   = list(range(len(test_ds) - args.n_qss, len(test_ds)))
    all_preds, gts, ulrs, all_samples = [], [], [], []

    for idx in qss_idx:
        print(f'    frame {idx} ({qss_idx.index(idx)+1}/{args.n_qss})')
        res, hr, lr, ulr = test_ds[idx]
        lr_t     = lr.unsqueeze(0).to(device).float()
        ulr_full = ulr.unsqueeze(0).to(device).float()
        # All ensemble members of a frame are drawn in one batch (independent noise per member).
        hr_batch = hr.unsqueeze(0).repeat(args.n_samples, 1, 1, 1).to(device).float()

        with torch.no_grad():
            x_e = model.compute_x_e(lr_t, ulr_full)
            if x_e is not None:
                x_e = x_e.repeat(args.n_samples, *[1] * (x_e.ndim - 1))
            raw = draw_samples(model, m, settings, hr_batch, x_e, test_ds)  # (N_SAMPLES, C, H, W)
        # unscale the full n_steps*n_ch tensor, then keep the last-timestep n_ch channels
        frame_samples_arr = np.stack([test_ds.unscale_data(r, input_type='hr')[-n_ch:]
                                      for r in raw.cpu().numpy()])  # (N_SAMPLES, n_ch, H, W)
        mean_pred = frame_samples_arr.mean(axis=0)   # (n_ch, H, W)

        gt_phys  = test_ds.unscale_data(hr.numpy(),  input_type='hr')[-n_ch:]
        ulr_phys = test_ds.unscale_data(ulr.numpy(), input_type='upscaled_lr')[-n_ch:]

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
                        samples=samples_arr, stds=stds_arr,
                        **{f'sampling_{k}': np.array(v) for k, v in settings.items()})
    print(f'  Saved → {out_file}')
    return True


def main():
    global RUNS_ROOT, DATA_ROOT
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    runs_root = Path(args.runs_root) if args.runs_root else RUNS_ROOT
    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    print(f'runs_root: {runs_root}')
    print(f'data_root: {data_root}')

    # Monkey-patch the module-level constants so helper functions pick them up
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
