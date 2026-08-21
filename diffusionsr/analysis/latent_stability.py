"""Latent stability analysis helpers for VAE embeddings.

The companion ``latent_stability.ipynb`` notebook is intentionally thin: it
sets the few user-facing controls, then calls the functions here.  Keeping the
implementation in a module makes the notebook easier to rerun and lets this
analysis be reused from scripts.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusionsr.datasets.dataset import SimulationXZDataset
from diffusionsr.models.vae_model import VAE2D, VAE3D


TARGET_INDEX = {
    "residual": 0,
    "hr": 1,
    "high_resolution": 1,
    "lr": 2,
    "true_lr": 2,
    "upscaled_lr": 3,
}

PROPERTY_ALIASES = {
    "p": "power",
    "power": "power",
    "laser_power": "power",
    "laser power": "power",
    "v": "velocity",
    "velocity": "velocity",
    "scan_velocity": "velocity",
    "scan velocity": "velocity",
    "led": "led",
    "linear_energy_density": "led",
    "linear energy density": "led",
}


@dataclass
class LatentStabilityState:
    project_root: Path
    dataset_root: Path
    checkpoint_path: Path
    checkpoint_options: List[Path]
    config_path: Optional[Path]
    output_dir: Path
    config: Mapping
    checkpoint: Mapping
    model: torch.nn.Module
    model_config: Mapping
    trainer_config: Mapping
    spatial_dims: int
    depth_size: int
    input_type: str
    target_type: str
    embed_data_type: str
    datasets: Dict[str, SimulationXZDataset]
    device: torch.device

    def summary(self) -> Dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "dataset_root": str(self.dataset_root),
            "checkpoint_path": str(self.checkpoint_path),
            "config_path": str(self.config_path) if self.config_path else None,
            "output_dir": str(self.output_dir),
            "spatial_dims": self.spatial_dims,
            "depth_size": self.depth_size,
            "input_type": self.input_type,
            "target_type": self.target_type,
            "embed_data_type": self.embed_data_type,
            "splits": {split: len(dataset) for split, dataset in self.datasets.items()},
            "model_config": dict(self.model_config),
            "trainer_config": dict(self.trainer_config),
        }


def find_project_root(start: Optional[Union[str, Path]] = None) -> Path:
    start_path = Path(start or Path.cwd()).expanduser().resolve()
    for path in [start_path, *start_path.parents]:
        if (path / "setup.py").exists() and (path / "diffusionsr").exists():
            return path
    raise RuntimeError("Could not find the LPBFDiffusionSR project root.")


def resolve_path(
    path: Optional[Union[str, Path]],
    project_root: Optional[Union[str, Path]] = None,
    must_exist: bool = False,
) -> Optional[Path]:
    if path in (None, ""):
        return None
    project_root = Path(project_root or find_project_root()).expanduser().resolve()
    candidate = Path(path).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        Path.cwd() / candidate,
        project_root / candidate,
        project_root / "diffusionsr" / candidate,
    ]
    for item in candidates:
        if item.exists():
            return item.resolve()
    resolved = candidates[0].resolve()
    if must_exist:
        raise FileNotFoundError(f"Could not resolve path: {path}")
    return resolved


def slugify(value: Union[str, Path]) -> str:
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "latent_stability"


def canonical_data_type(data_type: str) -> str:
    aliases = {"high_resolution": "hr", "true_lr": "lr"}
    return aliases.get(str(data_type), str(data_type))


def canonical_property(name: str) -> str:
    key = str(name).strip().lower()
    if key not in PROPERTY_ALIASES:
        raise ValueError(f"Unknown property '{name}'. Use one of: power, velocity, led.")
    return PROPERTY_ALIASES[key]


def linear_energy_density(power: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    power = np.asarray(power, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    return np.divide(power, velocity, out=np.full_like(power, np.nan), where=velocity != 0)


def config_to_field_names(fields) -> Optional[List[str]]:
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


def discover_checkpoints(
    search_roots: Optional[Sequence[Union[str, Path]]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> List[Path]:
    project_root = Path(project_root or find_project_root()).expanduser().resolve()
    if search_roots is None:
        search_roots = [project_root / "diffusionsr" / "runs"]
    checkpoints: List[Path] = []
    for root in search_roots:
        root_path = resolve_path(root, project_root=project_root)
        if root_path is None or not root_path.exists():
            continue
        if root_path.is_file() and root_path.suffix == ".pth":
            checkpoints.append(root_path)
            continue
        checkpoints.extend(path for path in root_path.rglob("*.pth") if ".venv" not in path.parts)
    return sorted(set(checkpoints), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)


def select_checkpoint(
    vae_path: Optional[Union[str, Path]] = None,
    checkpoint_name: Optional[str] = "best_model.pth",
    checkpoint_index: int = 0,
    search_roots: Optional[Sequence[Union[str, Path]]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[Path, List[Path]]:
    project_root = Path(project_root or find_project_root()).expanduser().resolve()
    base = resolve_path(vae_path, project_root=project_root) if vae_path not in (None, "") else None

    if base is not None and base.is_file():
        return base, [base]

    if base is not None and base.exists():
        options = sorted(base.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not options:
            options = sorted(base.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    else:
        options = discover_checkpoints(search_roots=search_roots, project_root=project_root)

    if not options:
        raise FileNotFoundError("No .pth checkpoints found. Set VAE_PATH or CHECKPOINT_SEARCH_ROOTS.")

    if checkpoint_name:
        named = [path for path in options if path.name == checkpoint_name]
        if named:
            return named[0], options

    checkpoint_index = max(0, min(int(checkpoint_index), len(options) - 1))
    return options[checkpoint_index], options


def load_yaml(path: Union[str, Path]) -> Mapping:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def load_config_for_checkpoint(
    checkpoint_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[Mapping, Optional[Path]]:
    checkpoint_path = Path(checkpoint_path)
    resolved_config = resolve_path(config_path, project_root=project_root) if config_path else None
    if resolved_config is None:
        candidate = checkpoint_path.parent / "configuration.yml"
        resolved_config = candidate if candidate.exists() else None
    if resolved_config is None:
        return {}, None
    return load_yaml(resolved_config), resolved_config.resolve()


def model_from_checkpoint(checkpoint: Mapping, device: Union[str, torch.device]) -> Tuple[torch.nn.Module, Mapping, Mapping, int]:
    model_config = checkpoint.get("model_config", {})
    trainer_config = checkpoint.get("trainer_config", {})
    if not model_config:
        raise KeyError("The selected checkpoint does not contain model_config.")

    spatial_dims = int(model_config.get("spatial_dims", trainer_config.get("spatial_dims", 2)))
    model_cls = VAE2D if spatial_dims == 2 else VAE3D
    model = model_cls(
        input_channels=int(model_config["input_channels"]),
        output_channels=int(model_config.get("output_channels", model_config["input_channels"])),
        latent_channels=int(model_config["latent_channels"]),
        hidden_channels=int(model_config["hidden_channels"]),
        channel_multipliers=tuple(model_config.get("channel_multipliers", (1, 2, 4))),
        output_activation=model_config.get("output_activation"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, model_config, trainer_config, spatial_dims


def make_dataset(
    config: Mapping,
    dataset_root: Union[str, Path],
    split: str,
    return_info: bool = True,
) -> SimulationXZDataset:
    return SimulationXZDataset(
        downscale_method=config.get("downscale_method", "direct"),
        normalize=config.get("normalize_method", config.get("normalize", "standardize")),
        split=split,
        root_folder=str(dataset_root),
        n_steps=int(config.get("n_steps", 1)),
        field_names=config_to_field_names(config.get("fields", "temperature")),
        out_steps=config.get("out_steps", None),
        inflate_dim=config.get("inflate_dim", None),
        inflate_method=config.get("inflate_method", "repeat"),
        return_info=return_info,
    )


def setup_analysis(
    dataset_root: Optional[Union[str, Path]] = None,
    vae_path: Optional[Union[str, Path]] = None,
    checkpoint_name: Optional[str] = "best_model.pth",
    checkpoint_index: int = 0,
    config_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    splits: Sequence[str] = ("train", "test"),
    embed_data_type: Optional[str] = None,
    search_roots: Optional[Sequence[Union[str, Path]]] = None,
    save_selection: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> LatentStabilityState:
    project_root = find_project_root()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path, checkpoint_options = select_checkpoint(
        vae_path=vae_path,
        checkpoint_name=checkpoint_name,
        checkpoint_index=checkpoint_index,
        search_roots=search_roots,
        project_root=project_root,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config, resolved_config_path = load_config_for_checkpoint(
        checkpoint_path,
        config_path=config_path,
        project_root=project_root,
    )
    model, model_config, trainer_config, spatial_dims = model_from_checkpoint(checkpoint, device=device)

    configured_root = dataset_root if dataset_root not in (None, "") else config.get("root_folder")
    if configured_root in (None, ""):
        raise ValueError("Set DATASET_ROOT or use a VAE run with configuration.yml containing root_folder.")
    dataset_root_path = resolve_path(configured_root, project_root=project_root, must_exist=True)

    output_root = resolve_path(output_dir, project_root=project_root) if output_dir else None
    if output_root is None:
        run_slug = slugify("_".join(checkpoint_path.parent.parts[-4:]))
        output_root = project_root / "diffusionsr" / "analysis" / "latent_stability_outputs" / run_slug
    output_root.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, SimulationXZDataset] = {}
    ordered_splits = list(dict.fromkeys(["train", *splits]))
    for split in ordered_splits:
        if split in splits or split == "train":
            datasets[split] = make_dataset(config, dataset_root_path, split=split, return_info=True)
    datasets = {split: datasets[split] for split in splits}

    target_type = canonical_data_type(trainer_config.get("target_type", config.get("vae_target", "hr")))
    input_type = canonical_data_type(
        trainer_config.get("input_type", config.get("vae_input_type", config.get("vae_input", target_type)))
    )
    embed_data_type = canonical_data_type(embed_data_type or input_type)
    depth_size = int(trainer_config.get("depth_size") or config.get("inflate_dim") or 1)

    state = LatentStabilityState(
        project_root=project_root,
        dataset_root=dataset_root_path,
        checkpoint_path=checkpoint_path,
        checkpoint_options=checkpoint_options,
        config_path=resolved_config_path,
        output_dir=output_root,
        config=config,
        checkpoint=checkpoint,
        model=model,
        model_config=model_config,
        trainer_config=trainer_config,
        spatial_dims=spatial_dims,
        depth_size=depth_size,
        input_type=input_type,
        target_type=target_type,
        embed_data_type=embed_data_type,
        datasets=datasets,
        device=device,
    )

    if save_selection:
        selection_path = output_root / "latent_stability_selection.json"
        with open(selection_path, "w") as handle:
            json.dump(state.summary(), handle, indent=2, default=str)
    return state


def print_checkpoint_options(options: Sequence[Path], limit: int = 25) -> None:
    for index, path in enumerate(options[:limit]):
        print(f"{index:>3}: {path}")
    if len(options) > limit:
        print(f"... {len(options) - limit} more checkpoint(s). Increase the limit or set CHECKPOINT_INDEX.")


def format_tensor(tensor: torch.Tensor, spatial_dims: int, depth_size: int = 1) -> torch.Tensor:
    if spatial_dims == 2:
        if tensor.ndim == 3:
            return tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(f"Expected a 4D 2D batch, got shape {tuple(tensor.shape)}.")
        return tensor

    if tensor.ndim == 5:
        return tensor
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D flattened-volume batch or 5D volume, got {tuple(tensor.shape)}.")
    batch_size, channels, height, width = tensor.shape
    if depth_size <= 1:
        return tensor.reshape(batch_size, channels, 1, height, width)
    if channels % depth_size != 0:
        raise ValueError(f"Channel count {channels} is not divisible by depth size {depth_size}.")
    return tensor.reshape(batch_size, channels // depth_size, depth_size, height, width)


def select_tensor(batch, data_type: str, state: LatentStabilityState) -> torch.Tensor:
    data_type = canonical_data_type(data_type)
    tensor = batch[TARGET_INDEX[data_type]]
    return format_tensor(tensor, state.spatial_dims, state.depth_size).to(state.device).float()


def batch_info(batch, batch_size: int) -> np.ndarray:
    if isinstance(batch, (tuple, list)) and len(batch) > 4:
        info = batch[4]
        if torch.is_tensor(info):
            return info.detach().cpu().numpy().astype(np.float64)
        return np.asarray(info, dtype=np.float64)
    return np.full((batch_size, 3), np.nan, dtype=np.float64)


def as_numpy(array) -> np.ndarray:
    return array.detach().cpu().numpy() if torch.is_tensor(array) else np.asarray(array)


def _select_stats(dataset: SimulationXZDataset, data_type: str) -> Tuple[np.ndarray, np.ndarray]:
    data_type = canonical_data_type(data_type)
    if data_type == "hr":
        return np.asarray(dataset.mean_hr), np.asarray(dataset.std_hr)
    if data_type == "lr":
        return np.asarray(dataset.mean_lr), np.asarray(dataset.std_lr)
    if data_type == "upscaled_lr":
        return np.asarray(dataset.mean_upscaled_lr), np.asarray(dataset.std_upscaled_lr)
    if data_type == "residual":
        return np.asarray(dataset.mean_resid), np.asarray(dataset.std_resid)
    raise ValueError(f"Unsupported data_type: {data_type}")


def unscale_volume(dataset: SimulationXZDataset, volume: np.ndarray, data_type: str) -> np.ndarray:
    data_type = canonical_data_type(data_type)
    volume = np.asarray(volume)
    if volume.ndim != 4:
        raise ValueError(f"Expected a sample volume shaped (C,D,H,W), got {volume.shape}.")
    channels, depth, height, width = volume.shape
    flat = volume.reshape(channels * depth, height, width)

    if getattr(dataset, "normalize", None) == "rescaling":
        unscaled = np.asarray(dataset.unscale_data(flat, input_type=data_type))
        return unscaled.reshape(channels, depth, height, width)

    mean, std = _select_stats(dataset, data_type)
    if mean.ndim == 2:
        mean = np.broadcast_to(mean, flat.shape)
        std = np.broadcast_to(std, flat.shape)
        return (flat * std + mean).reshape(channels, depth, height, width)
    if mean.ndim == 3:
        unscaled = np.asarray(dataset.unscale_data(flat, input_type=data_type))
        return unscaled.reshape(channels, depth, height, width)
    if mean.ndim == 4:
        return volume * std + mean
    raise ValueError(f"Unsupported statistics shape: {mean.shape}")


def unscale_batch(dataset: SimulationXZDataset, tensor, data_type: str) -> np.ndarray:
    data_type = canonical_data_type(data_type)
    array = as_numpy(tensor)
    if array.ndim == 5:
        return np.stack([unscale_volume(dataset, sample, data_type) for sample in array], axis=0)
    return np.asarray(dataset.unscale_data(array, input_type=data_type))


def embed_splits(
    state: LatentStabilityState,
    splits: Sequence[str] = ("train", "test"),
    batch_size: int = 4,
    max_samples_per_split: Optional[int] = None,
    num_workers: int = 0,
    sample_posterior: bool = False,
    save_name: Optional[str] = None,
) -> Dict[str, object]:
    state.model.eval()
    results: Dict[str, object] = {
        "checkpoint_path": str(state.checkpoint_path),
        "dataset_root": str(state.dataset_root),
        "embed_data_type": state.embed_data_type,
        "target_type": state.target_type,
        "spatial_dims": state.spatial_dims,
        "depth_size": state.depth_size,
    }
    decode_output_shape: Optional[Tuple[int, ...]] = None
    latent_shape: Optional[Tuple[int, ...]] = None

    for split in splits:
        dataset = state.datasets[split]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        split_vectors = []
        split_latents = []
        split_info = []
        split_indices = []
        seen = 0
        progress = tqdm(loader, desc=f"embed {split}", leave=False)
        for batch in progress:
            model_input = select_tensor(batch, state.embed_data_type, state)
            with torch.no_grad():
                mu, logvar = state.model.encode(model_input)
                latent = state.model.reparameterize(mu, logvar) if sample_posterior else mu

            latent_np = latent.detach().cpu().numpy()
            vectors_np = latent_np.reshape(latent_np.shape[0], -1)
            info_np = batch_info(batch, latent_np.shape[0])
            take = latent_np.shape[0]
            if max_samples_per_split is not None:
                remaining = int(max_samples_per_split) - seen
                if remaining <= 0:
                    break
                take = min(take, remaining)

            split_latents.append(latent_np[:take])
            split_vectors.append(vectors_np[:take])
            split_info.append(info_np[:take])
            split_indices.extend(range(seen, seen + take))
            seen += take
            if decode_output_shape is None:
                decode_output_shape = tuple(model_input.shape[-state.spatial_dims:])
            if latent_shape is None:
                latent_shape = tuple(latent_np.shape[1:])

        if not split_vectors:
            raise ValueError(f"No samples embedded for split '{split}'.")

        vectors = np.concatenate(split_vectors, axis=0)
        latents = np.concatenate(split_latents, axis=0)
        info = np.concatenate(split_info, axis=0)
        power = info[:, 0]
        velocity = info[:, 1]
        timestep = info[:, 2] if info.shape[1] > 2 else np.full(len(info), np.nan)
        results[split] = {
            "vectors": vectors,
            "latents": latents,
            "power": power,
            "velocity": velocity,
            "led": linear_energy_density(power, velocity),
            "timestep": timestep,
            "sample_index": np.asarray(split_indices, dtype=np.int64),
        }

    results["decode_output_shape"] = tuple(decode_output_shape or ())
    results["latent_shape"] = tuple(latent_shape or ())

    save_name = save_name or f"embeddings_{state.embed_data_type}_{Path(state.checkpoint_path).stem}.npz"
    save_path = state.output_dir / save_name
    payload = {
        "checkpoint_path": np.asarray(str(state.checkpoint_path)),
        "dataset_root": np.asarray(str(state.dataset_root)),
        "embed_data_type": np.asarray(state.embed_data_type),
        "target_type": np.asarray(state.target_type),
        "decode_output_shape": np.asarray(results["decode_output_shape"]),
        "latent_shape": np.asarray(results["latent_shape"]),
    }
    for split in splits:
        for key, value in results[split].items():
            payload[f"{split}_{key}"] = np.asarray(value)
    np.savez_compressed(save_path, **payload)
    results["path"] = str(save_path)
    return results


def _combined_vectors(embeddings: Mapping, splits: Sequence[str] = ("train", "test")) -> Tuple[np.ndarray, np.ndarray]:
    vectors = []
    split_labels = []
    for split in splits:
        split_vectors = np.asarray(embeddings[split]["vectors"])
        vectors.append(split_vectors)
        split_labels.extend([split] * len(split_vectors))
    return np.concatenate(vectors, axis=0), np.asarray(split_labels)


def _split_projection(array: np.ndarray, split_labels: np.ndarray, split: str) -> np.ndarray:
    return array[split_labels == split]


def property_values(embeddings: Mapping, split: str, color_by: str) -> np.ndarray:
    return np.asarray(embeddings[split][canonical_property(color_by)], dtype=np.float64)


def projection_color_label(color_by: str) -> str:
    key = canonical_property(color_by)
    return {"power": "Laser power", "velocity": "Velocity", "led": "Linear energy density"}[key]


def plot_projection_pair(
    projections: Mapping,
    embeddings: Mapping,
    color_by: str = "led",
    projection_names: Sequence[str] = ("pca", "tsne"),
    save_path: Optional[Union[str, Path]] = None,
):
    color_by = canonical_property(color_by)
    projection_names = [name for name in projection_names if name in projections]
    fig, axes = plt.subplots(1, len(projection_names), figsize=(6 * len(projection_names), 4.8), dpi=160)
    axes = np.atleast_1d(axes)

    all_colors = np.concatenate([property_values(embeddings, split, color_by) for split in ("train", "test")])
    vmin, vmax = np.nanpercentile(all_colors, [1, 99])
    if np.isclose(vmin, vmax):
        vmin, vmax = None, None

    for ax, name in zip(axes, projection_names):
        last = None
        for split, marker, label in [("train", "o", "train"), ("test", "^", "test")]:
            coords = np.asarray(projections[name][split])
            colors = property_values(embeddings, split, color_by)
            last = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                s=22,
                marker=marker,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                alpha=0.78,
                edgecolors="none",
                label=label,
            )
        ax.set_title(name.upper())
        ax.set_xlabel(f"{name.upper()} 1")
        ax.set_ylabel(f"{name.upper()} 2")
        ax.legend(frameon=False)
        if last is not None:
            fig.colorbar(last, ax=ax, label=projection_color_label(color_by))
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def run_latent_projections(
    embeddings: Mapping,
    perplexity: float = 30.0,
    color_by: str = "led",
    pca_components: int = 8,
    random_state: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
) -> Dict[str, object]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    vectors, split_labels = _combined_vectors(embeddings)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    pca_components = max(2, min(int(pca_components), scaled.shape[0], scaled.shape[1]))
    pca_model = PCA(n_components=pca_components, random_state=random_state)
    pca_all = pca_model.fit_transform(scaled)

    tsne_perplexity = min(float(perplexity), max(1.0, (scaled.shape[0] - 1) / 3.0))
    tsne_perplexity = max(1.0, tsne_perplexity)
    tsne_model = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    tsne_all = tsne_model.fit_transform(scaled)

    projections: Dict[str, object] = {
        "scaler": scaler,
        "split_labels": split_labels,
        "pca_model": pca_model,
        "pca": {
            "all": pca_all,
            "train": _split_projection(pca_all, split_labels, "train"),
            "test": _split_projection(pca_all, split_labels, "test"),
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
        },
        "tsne": {
            "all": tsne_all,
            "train": _split_projection(tsne_all, split_labels, "train"),
            "test": _split_projection(tsne_all, split_labels, "test"),
            "perplexity": tsne_perplexity,
        },
    }

    if output_dir is not None and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "latent_projection_results.npz",
            pca_all=pca_all,
            tsne_all=tsne_all,
            split_labels=split_labels,
            pca_explained_variance_ratio=pca_model.explained_variance_ratio_,
            tsne_perplexity=np.asarray(tsne_perplexity),
        )
        plot_projection_pair(
            projections,
            embeddings,
            color_by=color_by,
            save_path=output_dir / f"projection_{canonical_property(color_by)}.png",
        )
    return projections


def make_projection_collage(
    projections: Mapping,
    embeddings: Mapping,
    color_modes: Sequence[str] = ("power", "velocity", "led"),
    projection_names: Sequence[str] = ("pca", "tsne"),
    save_path: Optional[Union[str, Path]] = None,
):
    color_modes = [canonical_property(color) for color in color_modes]
    projection_names = [name for name in projection_names if name in projections]
    fig, axes = plt.subplots(
        len(color_modes),
        len(projection_names),
        figsize=(5.4 * len(projection_names), 4.4 * len(color_modes)),
        dpi=160,
        squeeze=False,
    )

    for row, color_by in enumerate(color_modes):
        all_colors = np.concatenate([property_values(embeddings, split, color_by) for split in ("train", "test")])
        vmin, vmax = np.nanpercentile(all_colors, [1, 99])
        if np.isclose(vmin, vmax):
            vmin, vmax = None, None
        for col, name in enumerate(projection_names):
            ax = axes[row, col]
            last = None
            for split, marker, label in [("train", "o", "train"), ("test", "^", "test")]:
                coords = np.asarray(projections[name][split])
                colors = property_values(embeddings, split, color_by)
                last = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=colors,
                    s=18,
                    marker=marker,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.78,
                    edgecolors="none",
                    label=label,
                )
            ax.set_title(f"{name.upper()} colored by {projection_color_label(color_by)}")
            ax.set_xlabel(f"{name.upper()} 1")
            ax.set_ylabel(f"{name.upper()} 2")
            ax.legend(frameon=False)
            if last is not None:
                fig.colorbar(last, ax=ax, label=projection_color_label(color_by))
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "r2": float(r2_score(y_true[finite], y_pred[finite])),
        "rmse": float(math.sqrt(mean_squared_error(y_true[finite], y_pred[finite]))),
        "mae": float(mean_absolute_error(y_true[finite], y_pred[finite])),
    }


def _raw_linear_coefficients(pipeline) -> Tuple[np.ndarray, float]:
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["linearregression"]
    coef = np.asarray(model.coef_, dtype=np.float64) / np.asarray(scaler.scale_, dtype=np.float64)
    intercept = float(model.intercept_ - np.dot(coef, scaler.mean_))
    return coef, intercept


def fit_linear_probes(
    embeddings: Mapping,
    targets: Sequence[str] = ("power", "led", "velocity"),
    solve_splits_separately: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
) -> Dict[str, object]:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = np.asarray(embeddings["train"]["vectors"], dtype=np.float64)
    x_test = np.asarray(embeddings["test"]["vectors"], dtype=np.float64)
    targets = [canonical_property(target) for target in targets]
    results: Dict[str, object] = {"targets": targets, "models": {}, "raw": {}, "metrics": {}, "predictions": {}}

    for target in targets:
        y_train = property_values(embeddings, "train", target)
        y_test = property_values(embeddings, "test", target)
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(x_train, y_train)
        coef, intercept = _raw_linear_coefficients(model)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        results["models"][target] = model
        results["raw"][target] = {"coef": coef, "intercept": intercept}
        results["predictions"][target] = {"train": train_pred, "test": test_pred}
        results["metrics"][target] = {
            "train_fit": {
                "train": _regression_metrics(y_train, train_pred),
                "test": _regression_metrics(y_test, test_pred),
            }
        }

        if solve_splits_separately:
            split_models = {}
            for split, x_split, y_split in [("train", x_train, y_train), ("test", x_test, y_test)]:
                split_model = make_pipeline(StandardScaler(), LinearRegression())
                split_model.fit(x_split, y_split)
                split_pred = split_model.predict(x_split)
                split_coef, split_intercept = _raw_linear_coefficients(split_model)
                split_models[split] = {
                    "model": split_model,
                    "coef": split_coef,
                    "intercept": split_intercept,
                    "metrics": _regression_metrics(y_split, split_pred),
                }
            results["metrics"][target]["separate_fit"] = {
                split: split_models[split]["metrics"] for split in split_models
            }
            results.setdefault("separate_models", {})[target] = split_models

    fig = plot_probe_parity(results, embeddings)
    if output_dir is not None and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "linear_probe_parity.png", bbox_inches="tight")
        serializable_metrics = json.loads(json.dumps(results["metrics"], default=float))
        with open(output_dir / "linear_probe_metrics.json", "w") as handle:
            json.dump(serializable_metrics, handle, indent=2)
        np.savez_compressed(
            output_dir / "linear_probe_coefficients.npz",
            **{f"{target}_coef": results["raw"][target]["coef"] for target in targets},
            **{f"{target}_intercept": np.asarray(results["raw"][target]["intercept"]) for target in targets},
        )
    return results


def plot_probe_parity(probe_results: Mapping, embeddings: Mapping):
    targets = probe_results["targets"]
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.6), dpi=160)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, targets):
        values = {
            "train": property_values(embeddings, "train", target),
            "test": property_values(embeddings, "test", target),
        }
        predictions = probe_results["predictions"][target]
        combined = np.concatenate([values["train"], values["test"], predictions["train"], predictions["test"]])
        finite = combined[np.isfinite(combined)]
        if finite.size:
            low, high = float(np.min(finite)), float(np.max(finite))
            margin = 0.04 * (high - low if high > low else 1.0)
            low -= margin
            high += margin
        else:
            low, high = 0.0, 1.0

        train_metrics = probe_results["metrics"][target]["train_fit"]["train"]
        test_metrics = probe_results["metrics"][target]["train_fit"]["test"]
        ax.scatter(
            values["train"],
            predictions["train"],
            s=24,
            alpha=0.78,
            edgecolors="none",
            label=f"train R2={train_metrics['r2']:.3f}",
        )
        ax.scatter(
            values["test"],
            predictions["test"],
            s=28,
            alpha=0.78,
            marker="^",
            edgecolors="none",
            label=f"test R2={test_metrics['r2']:.3f}",
        )
        ax.plot([low, high], [low, high], color="black", linewidth=1.2, linestyle="--", label="parity y=x")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel(f"true {projection_color_label(target)}")
        ax.set_ylabel(f"predicted {projection_color_label(target)}")
        ax.set_title(f"{projection_color_label(target)} probe")
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def clamp_index(index: int, size: int) -> int:
    return max(0, min(int(size) - 1, int(index)))


def center_indices(shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(size // 2) for size in shape)


def slice_2d(array: np.ndarray, axis: str = "z", index: Optional[int] = None) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array.T
    if array.ndim != 3:
        raise ValueError(f"Expected 2D image or 3D volume, got {array.shape}.")
    axis = axis.lower()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    index = center_indices(array.shape)[axis_index] if index is None else clamp_index(index, array.shape[axis_index])
    if axis == "x":
        return array[index, :, :].T
    if axis == "y":
        return array[:, index, :].T
    return array[:, :, index].T


def field_name_for_channel(dataset: SimulationXZDataset, channel_index: int) -> str:
    field_names = list(getattr(dataset, "field_names", []) or [])
    if not field_names:
        return f"channel_{channel_index}"
    return field_names[int(channel_index) % len(field_names)]


def selected_channels(
    dataset: SimulationXZDataset,
    decoded: np.ndarray,
    channels: Optional[Sequence[Union[int, str]]] = None,
    max_channels: int = 3,
) -> List[int]:
    channel_count = int(decoded.shape[1])
    if channels is None:
        return list(range(min(channel_count, int(max_channels))))
    requested = [channels] if isinstance(channels, (str, int)) else list(channels)
    selected: List[int] = []
    for item in requested:
        if isinstance(item, int):
            selected.append(clamp_index(item, channel_count))
            continue
        matches = [
            idx for idx in range(channel_count)
            if field_name_for_channel(dataset, idx).lower() == str(item).lower()
        ]
        selected.extend(matches)
    return list(dict.fromkeys(selected))[:max_channels]


def decode_latent_vectors(
    state: LatentStabilityState,
    embeddings: Mapping,
    vectors: np.ndarray,
    split: str = "test",
    batch_size: int = 8,
    output_shape: Optional[Sequence[int]] = None,
    unscale: bool = True,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    latent_shape = tuple(int(value) for value in embeddings["latent_shape"])
    output_shape = tuple(output_shape or embeddings.get("decode_output_shape", ()))
    if not output_shape:
        raise ValueError("No decode_output_shape found. Rerun embed_splits.")

    decoded_batches = []
    state.model.eval()
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            latent = torch.from_numpy(vectors[start:start + batch_size].reshape((-1, *latent_shape))).to(state.device)
            decoded = state.model.decode(latent, output_shape=output_shape)
            decoded_batches.append(decoded.detach().cpu())
    decoded_tensor = torch.cat(decoded_batches, dim=0)
    if not unscale:
        return decoded_tensor.numpy()
    return unscale_batch(state.datasets[split], decoded_tensor, state.target_type)


def plot_decoded_sequence(
    state: LatentStabilityState,
    decoded: np.ndarray,
    split: str = "test",
    channels: Optional[Sequence[Union[int, str]]] = None,
    axis: str = "z",
    depth_index: Optional[int] = None,
    titles: Optional[Sequence[str]] = None,
    max_channels: int = 3,
    save_path: Optional[Union[str, Path]] = None,
):
    decoded = np.asarray(decoded)
    dataset = state.datasets[split]
    channel_indices = selected_channels(dataset, decoded, channels=channels, max_channels=max_channels)
    if not channel_indices:
        raise ValueError("No channels selected for decoded plot.")
    cols = decoded.shape[0]
    rows = len(channel_indices)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.6 * cols, 2.6 * rows),
        dpi=160,
        squeeze=False,
        constrained_layout=True,
    )
    titles = list(titles or [str(index) for index in range(cols)])

    for row, channel_index in enumerate(channel_indices):
        values = decoded[:, channel_index]
        finite = values[np.isfinite(values)]
        vmin, vmax = (None, None)
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [1, 99])
            if np.isclose(vmin, vmax):
                vmin, vmax = None, None
        field_name = field_name_for_channel(dataset, channel_index)
        cmap = "jet" if field_name == "temperature" else ("gray" if field_name in {"liqlabel", "meltregion"} else "viridis")
        last = None
        for col in range(cols):
            image = decoded[col, channel_index]
            index = depth_index
            if image.ndim == 3 and index is not None:
                index = clamp_index(index, image.shape[{"x": 0, "y": 1, "z": 2}[axis.lower()]])
            last = axes[row, col].imshow(slice_2d(image, axis=axis, index=index), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col])
            if col == 0:
                axes[row, col].set_ylabel(f"{channel_index}: {field_name}")
        if last is not None:
            fig.colorbar(last, ax=axes[row, :].tolist(), shrink=0.8)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def interpolate_between_samples(
    state: LatentStabilityState,
    embeddings: Mapping,
    split: str = "test",
    index_a: int = 0,
    index_b: int = 1,
    steps: int = 7,
    channels: Optional[Sequence[Union[int, str]]] = None,
    axis: str = "z",
    depth_index: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    vectors = np.asarray(embeddings[split]["vectors"])
    index_a = clamp_index(index_a, len(vectors))
    index_b = clamp_index(index_b, len(vectors))
    alphas = np.linspace(0.0, 1.0, int(steps))
    interp_vectors = (1.0 - alphas[:, None]) * vectors[index_a] + alphas[:, None] * vectors[index_b]
    decoded = decode_latent_vectors(state, embeddings, interp_vectors, split=split)
    titles = [f"a={alpha:.2f}" for alpha in alphas]
    fig = plot_decoded_sequence(
        state,
        decoded,
        split=split,
        channels=channels,
        axis=axis,
        depth_index=depth_index,
        titles=titles,
        save_path=save_path,
    )
    return {"mode": "samples", "vectors": interp_vectors, "decoded": decoded, "alphas": alphas, "figure": fig}


def _probe_target_matrix(
    embeddings: Mapping,
    fields: Sequence[str],
    target_values,
    steps: int,
) -> np.ndarray:
    fields = [canonical_property(field) for field in fields]
    if target_values is None:
        columns = []
        for field in fields:
            observed = np.concatenate([property_values(embeddings, "train", field), property_values(embeddings, "test", field)])
            finite = observed[np.isfinite(observed)]
            if finite.size == 0:
                raise ValueError(f"No finite observed values for {field}.")
            low, high = np.nanpercentile(finite, [10, 90])
            columns.append(np.linspace(low, high, int(steps)))
        return np.stack(columns, axis=1)

    if isinstance(target_values, Mapping):
        columns = []
        for field in fields:
            values = np.asarray(target_values[field], dtype=np.float64)
            if values.ndim == 0:
                values = np.repeat(values, int(steps))
            if values.size == 2 and int(steps) != 2:
                values = np.linspace(values[0], values[1], int(steps))
            if values.size != int(steps):
                raise ValueError(f"Target values for {field} must be scalar, length 2, or length steps.")
            columns.append(values)
        return np.stack(columns, axis=1)

    values = np.asarray(target_values, dtype=np.float64)
    if values.ndim == 1:
        if len(fields) != 1:
            raise ValueError("A 1D target_values array is only valid for one probe field.")
        return values.reshape(-1, 1)
    if values.ndim == 2 and values.shape[1] == len(fields):
        return values
    raise ValueError("target_values must be a mapping or an array shaped (steps, num_fields).")


def interpolate_along_probe_lines(
    state: LatentStabilityState,
    embeddings: Mapping,
    probe_results: Mapping,
    split: str = "test",
    anchor_index: int = 0,
    probe_fields: Sequence[str] = ("led",),
    target_values=None,
    steps: int = 7,
    channels: Optional[Sequence[Union[int, str]]] = None,
    axis: str = "z",
    depth_index: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    fields = [canonical_property(field) for field in probe_fields]
    vectors = np.asarray(embeddings[split]["vectors"], dtype=np.float64)
    anchor_index = clamp_index(anchor_index, len(vectors))
    anchor = vectors[anchor_index]
    targets = _probe_target_matrix(embeddings, fields, target_values, steps)

    coefs = np.stack([probe_results["raw"][field]["coef"] for field in fields], axis=0)
    intercepts = np.asarray([probe_results["raw"][field]["intercept"] for field in fields], dtype=np.float64)
    current = coefs @ anchor + intercepts
    gram_inverse = np.linalg.pinv(coefs @ coefs.T)

    interp_vectors = []
    predicted_values = []
    for desired in targets:
        rhs = desired - current #rhs is the difference in target space that we want to achieve
        delta = coefs.T @ gram_inverse @ rhs
        candidate = anchor + delta
        interp_vectors.append(candidate)
        predicted_values.append(coefs @ candidate + intercepts)
    interp_vectors = np.asarray(interp_vectors, dtype=np.float64)
    predicted_values = np.asarray(predicted_values, dtype=np.float64)
    decoded = decode_latent_vectors(state, embeddings, interp_vectors, split=split)

    titles = [
        ", ".join(f"{field}={value:.3g}" for field, value in zip(fields, row))
        for row in predicted_values
    ]
    fig = plot_decoded_sequence(
        state,
        decoded,
        split=split,
        channels=channels,
        axis=axis,
        depth_index=depth_index,
        titles=titles,
        save_path=save_path,
    )
    return {
        "mode": "probe",
        "fields": fields,
        "vectors": interp_vectors,
        "decoded": decoded,
        "target_values": targets,
        "predicted_values": predicted_values,
        "figure": fig,
    }


def interpolate_along_pca_components(
    state: LatentStabilityState,
    embeddings: Mapping,
    projections: Mapping,
    split: str = "test",
    anchor_index: int = 0,
    components: Sequence[int] = (0, 1, 2),
    sigma_range: Tuple[float, float] = (-2.0, 2.0),
    steps: int = 7,
    channels: Optional[Sequence[Union[int, str]]] = None,
    axis: str = "z",
    depth_index: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> Dict[int, Dict[str, object]]:
    vectors = np.asarray(embeddings[split]["vectors"], dtype=np.float64)
    anchor_index = clamp_index(anchor_index, len(vectors))
    anchor = vectors[anchor_index:anchor_index + 1]
    scaler = projections["scaler"]
    pca_model = projections["pca_model"]
    anchor_scaled = scaler.transform(anchor)
    pca_scores = np.asarray(projections["pca"]["all"])
    results: Dict[int, Dict[str, object]] = {}

    for component in components:
        component = int(component)
        if component < 0 or component >= pca_model.components_.shape[0]:
            raise ValueError(f"PCA component {component} is out of range.")
        score_std = float(np.nanstd(pca_scores[:, component])) or 1.0
        offsets = np.linspace(float(sigma_range[0]), float(sigma_range[1]), int(steps)) * score_std
        scaled_vectors = anchor_scaled + offsets[:, None] * pca_model.components_[component][None, :]
        interp_vectors = scaler.inverse_transform(scaled_vectors)
        decoded = decode_latent_vectors(state, embeddings, interp_vectors, split=split)
        titles = [f"PC{component + 1} {offset / score_std:+.1f} sigma" for offset in offsets]
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"pca_component_{component + 1}.png"
        fig = plot_decoded_sequence(
            state,
            decoded,
            split=split,
            channels=channels,
            axis=axis,
            depth_index=depth_index,
            titles=titles,
            save_path=save_path,
        )
        results[component] = {
            "mode": "pca",
            "component": component,
            "offsets": offsets,
            "vectors": interp_vectors,
            "decoded": decoded,
            "figure": fig,
        }
    return results


def run_latent_interpolation(
    state: LatentStabilityState,
    embeddings: Mapping,
    mode: str = "samples",
    projections: Optional[Mapping] = None,
    probe_results: Optional[Mapping] = None,
    split: str = "test",
    index_a: int = 0,
    index_b: int = 1,
    anchor_index: int = 0,
    steps: int = 7,
    channels: Optional[Sequence[Union[int, str]]] = None,
    axis: str = "z",
    depth_index: Optional[int] = None,
    probe_fields: Sequence[str] = ("led",),
    probe_target_values=None,
    pca_components: Sequence[int] = (0, 1, 2),
    pca_sigma_range: Tuple[float, float] = (-2.0, 2.0),
    output_dir: Optional[Union[str, Path]] = None,
) -> object:
    mode = str(mode).strip().lower()
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if mode in {"samples", "sample", "between"}:
        save_path = output_dir / "interpolation_between_samples.png" if output_dir else None
        return interpolate_between_samples(
            state,
            embeddings,
            split=split,
            index_a=index_a,
            index_b=index_b,
            steps=steps,
            channels=channels,
            axis=axis,
            depth_index=depth_index,
            save_path=save_path,
        )
    if mode in {"probe", "fit", "line"}:
        if probe_results is None:
            raise ValueError("probe_results is required for probe interpolation.")
        fields_slug = "_".join(canonical_property(field) for field in probe_fields)
        save_path = output_dir / f"interpolation_probe_{fields_slug}.png" if output_dir else None
        return interpolate_along_probe_lines(
            state,
            embeddings,
            probe_results,
            split=split,
            anchor_index=anchor_index,
            probe_fields=probe_fields,
            target_values=probe_target_values,
            steps=steps,
            channels=channels,
            axis=axis,
            depth_index=depth_index,
            save_path=save_path,
        )
    if mode in {"pca", "pc"}:
        if projections is None:
            raise ValueError("projections is required for PCA interpolation.")
        save_dir = output_dir / "pca_interpolations" if output_dir else None
        return interpolate_along_pca_components(
            state,
            embeddings,
            projections,
            split=split,
            anchor_index=anchor_index,
            components=pca_components,
            sigma_range=pca_sigma_range,
            steps=steps,
            channels=channels,
            axis=axis,
            depth_index=depth_index,
            save_dir=save_dir,
        )
    raise ValueError("mode must be 'samples', 'probe', or 'pca'.")
