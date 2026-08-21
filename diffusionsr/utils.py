import argparse
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
