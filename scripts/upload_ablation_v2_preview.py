"""
Upload precomputed preview data for all 24 ablation_v2 variants to W&B.

Bundles per variant:
  - preview/qss_samples.npz     (predictions, GT, stds)
  - meta.json                   (training_time_h, wandb_run_id)

Run on TRACE after precompute_ablation_v2_preview.py completes:
  /trace/.../envs/diffusion_SR/bin/python scripts/upload_ablation_v2_preview.py \
    --runs_root /trace/.../runs/ablation_v2 \
    --wandb_project Flow3D_SuperResolution \
    --wandb_entity ngng-

The artifact 'ablation_v2_preview' is downloaded automatically by notebook 18.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusionsr.utils import log_single_version_artifact

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

    files = {}  # {name_in_artifact: local_path}
    found = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for m in MODELS:
            for e in ENCS:
                for f in FIELDS:
                    for n in NORMS:
                        variant = f'{m}_{e}_{f}_{n}'
                        run_dir = runs_root / variant
                        npz_path = run_dir / 'preview' / 'qss_samples.npz'
                        if not npz_path.exists():
                            print(f'  [MISSING npz] {variant}')
                            continue
                        with np.load(str(npz_path)) as _z:
                            stamped = 'sampling_timesteps' in _z.files
                        if not stamped:
                            # Pre-fix previews were sampled with timesteps=200 instead of the trained 1000.
                            print(f'  [STALE npz, not uploaded] {variant}: no sampling_* stamp; '
                                  f're-run precompute_ablation_v2_preview.py')
                            continue

                        files[f'{variant}/qss_samples.npz'] = str(npz_path)

                        # Collect training time (seconds → hours)
                        t = 0.0
                        for fname in ['training_time.txt', 'vae_training_time.txt']:
                            p = run_dir / fname
                            if p.exists():
                                t += float(np.loadtxt(str(p)))

                        # Collect W&B run ID
                        run_id = None
                        id_file = run_dir / 'wandb_run_id.txt'
                        if id_file.exists():
                            run_id = id_file.read_text().strip()

                        # Write meta.json into a temp file and add to artifact
                        meta = {'training_time_h': t / 3600 if t > 0 else None,
                                'wandb_run_id': run_id}
                        meta_path = Path(tmpdir) / f'{variant}_meta.json'
                        meta_path.write_text(json.dumps(meta))
                        files[f'{variant}/meta.json'] = str(meta_path)

                        found += 1
                        print(f'  Added {variant}  (t={meta["training_time_h"]:.1f}h  run={run_id})'
                              if meta['training_time_h'] else f'  Added {variant}')

        print(f'\nUploading {found}/24 variants...')
        # Single version: commit the new one, then delete every older version of this artifact.
        ok = log_single_version_artifact(
            args.artifact_name, 'preview', files, run=run,
            description='Precomputed QSS samples (preds/gts/ulrs/stds/samples) for all 24 ablation_v2 variants.')

    run.finish()
    if not ok:
        raise SystemExit('Upload failed; see the [wandb] message above.')
    print(f'Done. Artifact: {args.artifact_name}')


if __name__ == '__main__':
    main()
