"""
One-time backfill: upload existing run checkpoints and loss curves to W&B.

Walks the runs directory, creates one W&B run per model run, uploads:
  - checkpoint artifact (ckpt.pth / bestmodel_saved.pth / vae_best.pth)
  - loss curves image (generated on the fly from loss .txt files)
  - configuration YAML (logged as config)

Usage (on TRACE):
  cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
  conda activate diffusion_SR
  python -m diffusionsr.runners.upload_checkpoints_to_wandb \\
      --runs_dir diffusionsr/runs \\
      --project Flow3D_SuperResolution \\
      --entity $WANDB_ENTITY

  # Dry-run to preview what would be uploaded (no W&B calls):
  python -m diffusionsr.runners.upload_checkpoints_to_wandb \\
      --runs_dir diffusionsr/runs --dry_run
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

# ── Model type detection from directory name ───────────────────────────────────

_DIR_TO_LABEL = {
    'diffusionimplicitencoded':  'diffusion_sr',
    'diffusionexplicitencoded':  'diffusion_sr',
    'diffusionimplicitupscaled': 'diffusion_sr',
    'diffusion':                 'diffusion_sr',
    'flowmatchingimplicitencoded': 'flow_matching',
    'flowmatching':              'flow_matching',
    'ldmimplicitencoded':        'latent_diffusion',
    'ldm':                       'latent_diffusion',
    'uncond_diffusion':          'uncond_diffusion',
    'encoder':                   'encoder',
    'mobilenet':                 'mobilenet',
    'vae':                       'vae',
}


def _model_label(folder_name: str) -> str:
    return _DIR_TO_LABEL.get(folder_name.lower(), folder_name)


def _run_name_from_path(run_dir: Path, model_label: str) -> str:
    """Derive a W&B run name from the run directory path.

    If the run directory leaf looks like a datetime string (YYYY_MM_DD_HH_MM_SS),
    convert it.  Otherwise use it as-is (e.g. 'cs_both_n3').
    """
    leaf = run_dir.name
    m = re.match(r'^(\d{4})_(\d{2})_(\d{2})_', leaf)
    if m:
        year, month, day = m.group(1), m.group(2), m.group(3)
        # Format as D_Mon_YYYY
        from datetime import date
        try:
            d = date(int(year), int(month), int(day))
            date_str = f"{d.day}_{d.strftime('%b')}_{d.year}"
        except ValueError:
            date_str = leaf
        return f"{date_str}_{model_label}"
    else:
        return f"{model_label}_{leaf}"


def _find_checkpoints(run_dir: Path):
    """Return list of (artifact_name, local_path) pairs for all checkpoints in run_dir."""
    found = []
    for fname in ('bestmodel_saved.pth', 'ckpt.pth', 'vae_best.pth', 'vae_ckpt.pth', 'encoder_ckpt.pth'):
        p = run_dir / fname
        if p.exists():
            found.append((fname, p))
    return found


def _load_config(run_dir: Path):
    cfg_path = run_dir / 'configuration.yml'
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _upload_run(run_dir: Path, model_label: str, project: str, entity: str,
                dry_run: bool = False):
    import wandb
    from diffusionsr.runners.plot_training_curves import plot_curves, collect_curves

    checkpoints = _find_checkpoints(run_dir)
    if not checkpoints:
        print(f"  skip (no checkpoints): {run_dir}")
        return

    run_name = _run_name_from_path(run_dir, model_label)
    config = _load_config(run_dir)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Uploading: {run_dir}")
    print(f"  run_name : {run_name}")
    print(f"  model    : {model_label}")
    print(f"  ckpts    : {[c[0] for c in checkpoints]}")

    if dry_run:
        return

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=model_label,
        tags=[model_label, 'backfill'],
        config=config,
        job_type='backfill',
    )

    # Upload checkpoints as artifacts
    for ckpt_name, ckpt_path in checkpoints:
        alias = 'best' if 'best' in ckpt_name else 'latest'
        art = wandb.Artifact(
            name=f"{run_name}_ckpt",
            type='checkpoint',
            metadata={'source_dir': str(run_dir), 'file': ckpt_name},
        )
        art.add_file(str(ckpt_path), name='ckpt.pth')
        run.log_artifact(art, aliases=[alias, ckpt_name.replace('.pth', '')])
        print(f"  uploaded artifact: {ckpt_name} → {run_name}_ckpt:{alias}")

    # Generate and upload loss curves
    curves = collect_curves(run_dir)
    if curves:
        curves_png = run_dir / 'loss_curves.png'
        try:
            plot_curves(run_dir, out_path=str(curves_png))
            if curves_png.exists():
                run.log({'loss_curves': wandb.Image(str(curves_png))})
                print(f"  uploaded loss curves image")
        except Exception as e:
            print(f"  loss curves failed: {e}")

        # Also log final loss values as summary metrics
        for label, values in curves:
            metric_key = label.lower().replace(' ', '_')
            run.summary[f'final/{metric_key}'] = float(values[-1])
            run.summary[f'epochs/{metric_key}'] = len(values)

    wandb.finish()


def _scan_and_upload(runs_dir: Path, project: str, entity: str, dry_run: bool):
    """Walk runs_dir two levels deep: <downscale>/<model>/<run_name>/"""
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        sys.exit(1)

    uploaded = 0
    skipped = 0

    for downscale_dir in sorted(runs_dir.iterdir()):
        if not downscale_dir.is_dir():
            continue
        for model_dir in sorted(downscale_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            label = _model_label(model_dir.name)
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                checkpoints = _find_checkpoints(run_dir)
                if checkpoints:
                    _upload_run(run_dir, label, project, entity, dry_run=dry_run)
                    uploaded += 1
                else:
                    skipped += 1

    print(f"\nDone. Uploaded: {uploaded}  Skipped (no ckpt): {skipped}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs_dir', default='diffusionsr/runs',
                        help='Root runs directory (default: diffusionsr/runs)')
    parser.add_argument('--project', default='Flow3D_SuperResolution',
                        help='W&B project name')
    parser.add_argument('--entity', default=os.getenv('WANDB_ENTITY', ''),
                        help='W&B entity (default: $WANDB_ENTITY)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview what would be uploaded without calling W&B')
    args = parser.parse_args()

    if not args.dry_run and not args.entity:
        print("Error: --entity or $WANDB_ENTITY must be set")
        sys.exit(1)

    _scan_and_upload(
        runs_dir=Path(args.runs_dir),
        project=args.project,
        entity=args.entity,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
