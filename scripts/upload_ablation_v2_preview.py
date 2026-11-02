"""
Upload all precomputed preview/qss_samples.npz files to W&B as a single artifact.

Run on TRACE after precompute_ablation_v2_preview.py completes:
  /trace/.../envs/diffusion_SR/bin/python scripts/upload_ablation_v2_preview.py \
    --runs_root /trace/.../runs/ablation_v2 \
    --wandb_project Flow3D_SuperResolution

The artifact 'ablation_v2_preview' (type 'preview') is created in the project.
The notebook downloads it automatically when local npz files are absent.
"""
import argparse
import os
from pathlib import Path
import wandb

MODELS = ['diffusion', 'fm', 'ldm']
ENCS   = ['enc', 'noenc']
FIELDS = ['temp', 'multi']
NORMS  = ['std', 'gstd']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_root', required=True)
    p.add_argument('--wandb_project', default='Flow3D_SuperResolution')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--artifact_name', default='ablation_v2_preview')
    return p.parse_args()


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    entity = args.wandb_entity or os.getenv('WANDB_ENTITY')

    run = wandb.init(
        project=args.wandb_project,
        entity=entity,
        job_type='upload_preview',
        name='upload_ablation_v2_preview',
    )

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type='preview',
        description='Precomputed QSS samples (preds/gts/ulrs/stds/samples) for all 24 ablation_v2 variants.',
    )

    found = 0
    for m in MODELS:
        for e in ENCS:
            for f in FIELDS:
                for n in NORMS:
                    variant = f'{m}_{e}_{f}_{n}'
                    npz_path = runs_root / variant / 'preview' / 'qss_samples.npz'
                    if npz_path.exists():
                        artifact.add_file(str(npz_path), name=f'{variant}/qss_samples.npz')
                        found += 1
                        print(f'  Added {variant}')
                    else:
                        print(f'  [MISSING] {variant}')

    print(f'\nUploading {found}/24 files...')
    run.log_artifact(artifact)
    run.finish()
    print(f'Done. Artifact: {args.artifact_name}')


if __name__ == '__main__':
    main()
