import argparse
from datetime import date
from pathlib import Path
from typing import List, Optional

import torch

import diffusionsr


PACKAGE_DIR = Path(diffusionsr.__file__).resolve().parent        # <repo>/diffusionsr
REPO_ROOT = PACKAGE_DIR.parent                                    # <repo>


# ── Beta schedules ────────────────────────────────────────────────────────────
# Canonical definitions — import from here instead of redefining locally.
# CLAUDE.md warning: all copies must stay in sync with the training schedule.

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(0.0001, 0.02, int(timesteps))


def quadratic_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(0.0001 ** 0.5, 0.02 ** 0.5, int(timesteps)) ** 2


def sigmoid_beta_schedule(timesteps: int) -> torch.Tensor:
    betas = torch.linspace(-6, 6, int(timesteps))
    return torch.sigmoid(betas) * (0.02 - 0.0001) + 0.0001


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672."""
    steps = int(timesteps) + 1
    x = torch.linspace(0, int(timesteps), steps)
    alphas_cumprod = torch.cos(((x / int(timesteps)) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# ── Config helpers ────────────────────────────────────────────────────────────

def dict2namespace(config: dict) -> argparse.Namespace:
    """Recursively convert a config dict to a nested argparse.Namespace."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def config_to_field_names(config_dict) -> Optional[List[str]]:
    """Map the 'fields' config value to a list of field name strings.

    Accepts either a config dict (reads ``config_dict['fields']``) or a raw
    string/list value directly.  Returns None for "all available fields".
    """
    fields = config_dict.get("fields") if isinstance(config_dict, dict) else config_dict
    field_aliases = {"melt_region": "meltregion"}
    if fields == "temperature":
        return ["temperature"]
    if fields == "temperature_liqlabel":
        return ["temperature", "liqlabel"]
    if fields == "temperature_sdfliqlabel":
        return ["temperature", "sdfliqlabel"]
    if fields in ["temperature_liqlabel_meltregion", "temperature_liqlabel_melt_region"]:
        return ["temperature", "liqlabel", "meltregion"]
    if fields == "all_but_pressure":
        return ["vx", "temperature", "vy", "vz", "liqlabel"]
    if fields in ["all", None]:
        return None
    if isinstance(fields, (list, tuple)):
        return [field_aliases.get(field, field) for field in fields]
    raise ValueError(f"Unsupported fields config: {fields}")


def relocate_config_paths(config):
    """Rewrite path-like fields pulled from a W&B config to point at the current repo.

    W&B runs store absolute paths from the training machine. This rewrites them
    to their equivalents under the current checkout:
      - root_folder: rerooted to <REPO_ROOT>/data/<basename> (matches download_data.sh)
      - encoder_results_dir, restart_dir: anchored on 'runs/' and re-rooted under
        <PACKAGE_DIR>, since training is launched from inside diffusionsr/

    Fields without a recognized structure are left untouched.
    """
    if "root_folder" in config:
        basename = Path(str(config["root_folder"]).rstrip("/")).name
        config["root_folder"] = str(REPO_ROOT / "data" / basename)

    for key in ("encoder_results_dir", "restart_dir"):
        if key not in config:
            continue
        s = str(config[key])
        i = s.find("runs/")
        if i != -1:
            config[key] = str(PACKAGE_DIR / s[i:])

    return config


# ── W&B run naming and checkpoint helpers ─────────────────────────────────────

def make_run_name(model_type: str, suffix: str = "") -> str:
    """Return a W&B run name in the format D_Mon_YYYY_<model_type>[_suffix].

    Example: make_run_name('latent_diffusion', 'direct') → '1_Aug_2026_latent_diffusion_direct'
    """
    today = date.today()
    name = f"{today.day}_{today.strftime('%b')}_{today.year}_{model_type}"
    if suffix:
        name += f"_{suffix}"
    return name


def upload_checkpoint_artifact(ckpt_path: str, run_name: str, epoch: int,
                                is_best: bool = False) -> None:
    """Upload a checkpoint file to W&B as a versioned artifact.

    The artifact is named ``{run_name}_ckpt`` (type ``checkpoint``) with
    aliases ``['latest', 'epoch_{epoch}']`` (plus ``'best'`` when is_best).
    No-op if ``wandb.run`` is None (W&B disabled or not initialised yet).
    """
    import wandb  # local import so utils stays importable without W&B installed
    if wandb.run is None:
        return
    art = wandb.Artifact(
        name=f"{run_name}_ckpt",
        type="checkpoint",
        metadata={"epoch": epoch},
    )
    art.add_file(str(ckpt_path), name="ckpt.pth")
    aliases = ["latest", f"epoch_{epoch}"]
    if is_best:
        aliases.append("best")
    wandb.run.log_artifact(art, aliases=aliases)


def restore_checkpoint_from_wandb(entity: str, project: str, run_name: str,
                                    local_dir: str, alias: str = "latest") -> bool:
    """Download a W&B checkpoint artifact to ``local_dir/ckpt.pth``.

    Returns True if the download succeeded and the file exists, False otherwise.
    Intended as a fallback when the local checkpoint is absent after SLURM
    preemption cleared the scratch filesystem.
    """
    import wandb
    api = wandb.Api()
    artifact_id = f"{entity}/{project}/{run_name}_ckpt:{alias}"
    try:
        artifact = api.artifact(artifact_id)
        artifact.download(root=local_dir)
        ckpt = Path(local_dir) / "ckpt.pth"
        if ckpt.exists():
            print(f"Restored checkpoint from W&B: {artifact_id} → {ckpt}")
            return True
        return False
    except Exception as e:
        print(f"W&B checkpoint restore failed ({artifact_id}): {e}")
        return False
