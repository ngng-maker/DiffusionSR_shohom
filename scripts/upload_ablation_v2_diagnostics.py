"""
Upload the ablation_v2 diagnostic outputs to W&B (one single-version artifact each), for notebook 24.

  ablation_v2_sampler_diagnostics  <- runs/ablation_v2/_diagnostics           (scripts/diag_ablation_v2_sampling.py)
  ablation_v2_fm_vs_ddpm_probes    <- runs/ablation_v2/_diagnostics_fm_vs_ddpm (scripts/probe_fm_vs_ddpm.py)

Run on TRACE from the repo root:
  python scripts/upload_ablation_v2_diagnostics.py --wandb_entity ngng-
"""
import argparse
import os
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusionsr.utils import log_single_version_artifact

RUNS = Path(__file__).resolve().parent.parent / 'diffusionsr' / 'runs' / 'ablation_v2'
ARTIFACTS = {
    'ablation_v2_sampler_diagnostics': (RUNS / '_diagnostics',
        'Encoder runs re-sampled with DDIM 50/200/1000, DDPM 1000 (DiffusionSR, LDM) and Euler 50/200 (FM); '
        'RRDB encoder alone; LDM VAE reconstructions and latent stats. scripts/diag_ablation_v2_sampling.py'),
    'ablation_v2_fm_vs_ddpm_probes': (RUNS / '_diagnostics_fm_vs_ddpm',
        'Probe A: one-step clean-field estimates at matched noise levels (DiffusionSR vs FM); probe B: x0-hat along '
        'DDIM-50 / DDPM-1000 / FM Euler-50 trajectories. scripts/probe_fm_vs_ddpm.py'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wandb_project', default='Flow3D_SuperResolution')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--only', nargs='*', default=None, help='Subset of artifact names.')
    p.add_argument('--diag_dir', default=None, help='Override the _diagnostics source directory (e.g. a local copy).')
    p.add_argument('--probe_dir', default=None, help='Override the _diagnostics_fm_vs_ddpm source directory.')
    args = p.parse_args()
    for name, override in (('ablation_v2_sampler_diagnostics', args.diag_dir),
                           ('ablation_v2_fm_vs_ddpm_probes', args.probe_dir)):
        if override:
            ARTIFACTS[name] = (Path(override), ARTIFACTS[name][1])
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or os.getenv('WANDB_ENTITY'),
                     job_type='upload_diagnostics', name='upload_ablation_v2_diagnostics')
    ok = True
    for name, (src, desc) in ARTIFACTS.items():
        if args.only and name not in args.only:
            continue
        files = {f.name: str(f) for f in sorted(src.glob('*')) if f.suffix in ('.npz', '.json')}
        print(f'{name}: {len(files)} files from {src}')
        ok &= bool(files) and log_single_version_artifact(name, 'diagnostics', files, run=run, description=desc)
    run.finish()
    if not ok:
        raise SystemExit('An upload failed; see the [wandb] messages above.')


if __name__ == '__main__':
    main()
