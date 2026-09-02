"""Standalone VAE pretraining for the ablation_20260901 experiment.

Usage:
    python -m diffusionsr.scripts.pretrain_vae_standalone \
        --root_folder /trace/.../data_fields \
        --vae_dir     /trace/.../runs/ablation_20260901/vae_sdf \
        --fields      temperature_sdfliqlabel \
        --wandb_run_name 1_Sep_2026_vae_sdf

pretrain_vae() already skips if vae_best.pth exists, so this is safe to
re-run; the SLURM setup script calls it for both field configs.
"""
import argparse
import os
import wandb
from pathlib import Path

from diffusionsr.datasets.dataset import SimulationXZDataset
from diffusionsr.runners.train_ldm import pretrain_vae
from diffusionsr.utils import config_to_field_names

FIELD_ALIASES = {
    "temperature_sdfliqlabel": "temperature_sdfliqlabel",
    "temperature":             "temperature",
    "temperature_liqlabel":    "temperature_liqlabel",
}

def main():
    parser = argparse.ArgumentParser(description="Pretrain KL-VAE standalone (ablation 2026-09-01)")
    parser.add_argument("--root_folder",     required=True)
    parser.add_argument("--vae_dir",         required=True)
    parser.add_argument("--fields",          default="temperature_sdfliqlabel",
                        choices=list(FIELD_ALIASES))
    parser.add_argument("--n_steps",         type=int, default=3)
    parser.add_argument("--downscale_method", default="direct")
    parser.add_argument("--normalize",       default="standardize")
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--wandb_run_name",  default=None)
    parser.add_argument("--wandb_project",   default="Flow3D_SuperResolution")
    args = parser.parse_args()

    field_names = config_to_field_names(args.fields)
    print(f"Field names: {field_names}")
    print(f"VAE dir:     {args.vae_dir}")

    if Path(args.vae_dir, "vae_best.pth").exists():
        print("VAE already trained — skipping.")
        return

    kw = dict(downscale_method=args.downscale_method, root_folder=args.root_folder,
              normalize=args.normalize, n_steps=args.n_steps, field_names=field_names)
    train_ds = SimulationXZDataset(split="train", **kw)
    dev_ds   = SimulationXZDataset(split="dev",   **kw)
    test_ds  = SimulationXZDataset(split="test",  **kw)
    print(f"Dataset loaded: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    entity = os.getenv("WANDB_ENTITY")
    run_name = args.wandb_run_name or f"vae_{args.fields}"
    wandb.init(project=args.wandb_project, entity=entity, name=run_name,
               group="ablation_20260901", tags=["vae", args.fields],
               config=vars(args), resume="allow")

    pretrain_vae(args.vae_dir, train_dataset=train_ds, dev_dataset=dev_ds,
                 test_dataset=test_ds, num_epochs=args.epochs)
    wandb.finish()
    print("VAE pretraining complete.")

if __name__ == "__main__":
    main()
