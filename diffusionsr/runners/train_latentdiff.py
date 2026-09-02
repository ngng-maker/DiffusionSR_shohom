import argparse
import csv
import datetime
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusionsr.datasets.dataset import SimulationXZDataset
from diffusionsr.models.vae_model import VAE2D, VAE3D
from diffusionsr.runners.train_vae import parse_channel_multipliers
from diffusionsr.utils import (
    cosine_beta_schedule,
    linear_beta_schedule,
    dict2namespace,
    config_to_field_names,
)


_TARGET_INDEX = {
    "residual": 0,
    "hr": 1,
    "high_resolution": 1,
    "lr": 2,
    "true_lr": 2,
    "upscaled_lr": 3,
}

_LATENT_PREDICTION_TARGET_ALIASES = {
    "target": "target",
    "latent": "target",
    "target_latent": "target",
    "latent_residual": "latent_residual",
    "residual": "latent_residual",
    "delta": "latent_residual",
}


def normalize_latent_prediction_target(value: str) -> str:
    try:
        return _LATENT_PREDICTION_TARGET_ALIASES[str(value).lower()]
    except KeyError as exc:
        choices = sorted(_LATENT_PREDICTION_TARGET_ALIASES)
        raise ValueError(f"latent_prediction_target must be one of {choices}, got {value}") from exc


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _wandb_log(payload: dict, step: Optional[int] = None) -> None:
    if wandb.run is not None:
        wandb.log(payload, step=step)


def _groups_for_channels(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _conv(spatial_dims: int):
    if spatial_dims == 2:
        return nn.Conv2d
    if spatial_dims == 3:
        return nn.Conv3d
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")


def _interpolate_mode(spatial_dims: int) -> str:
    return "bilinear" if spatial_dims == 2 else "trilinear"


def resize_to_spatial(tensor: torch.Tensor, spatial_shape: Sequence[int], spatial_dims: int) -> torch.Tensor:
    spatial_shape = tuple(int(dim) for dim in spatial_shape)
    if tuple(tensor.shape[-spatial_dims:]) == spatial_shape:
        return tensor
    return F.interpolate(
        tensor,
        size=spatial_shape,
        mode=_interpolate_mode(spatial_dims),
        align_corners=False,
    )


def loss_fn(prediction: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(prediction, target)
    if loss_type == "l2":
        return F.mse_loss(prediction, target)
    if loss_type == "huber":
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return time.float()[:, None]
        scale = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class TimeResidualBlock(nn.Module):
    def __init__(self, spatial_dims: int, channels: int, time_dim: int):
        super().__init__()
        conv = _conv(spatial_dims)
        self.norm1 = nn.GroupNorm(_groups_for_channels(channels), channels)
        self.conv1 = conv(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(_groups_for_channels(channels), channels)
        self.conv2 = conv(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb).view(time_emb.shape[0], -1, *([1] * (x.ndim - 2)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConditionalLatentDenoiser(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        latent_channels: int,
        condition_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        time_dim: Optional[int] = None,
    ):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        self.latent_channels = int(latent_channels)
        self.condition_channels = int(condition_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        time_dim = int(time_dim or hidden_channels * 4)
        conv = _conv(self.spatial_dims)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_channels),
            nn.Linear(hidden_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = conv(latent_channels + condition_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [TimeResidualBlock(self.spatial_dims, hidden_channels, time_dim) for _ in range(self.num_blocks)]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups_for_channels(hidden_channels), hidden_channels),
            nn.SiLU(),
            conv(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )

    def config(self) -> dict:
        return {
            "model_type": "latent_diffusion_denoiser",
            "spatial_dims": self.spatial_dims,
            "latent_channels": self.latent_channels,
            "condition_channels": self.condition_channels,
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
        }

    def forward(self, noisy_latent: torch.Tensor, timestep: torch.Tensor, condition_latent: torch.Tensor) -> torch.Tensor:
        condition_latent = resize_to_spatial(
            condition_latent,
            noisy_latent.shape[-self.spatial_dims:],
            self.spatial_dims,
        )
        h = self.input(torch.cat([noisy_latent, condition_latent], dim=1))
        time_emb = self.time_mlp(timestep)
        for block in self.blocks:
            h = block(h, time_emb)
        return self.output(h)


class DirectLatentPredictor(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        conv = _conv(self.spatial_dims)

        layers = [conv(input_channels, hidden_channels, kernel_size=3, padding=1), nn.SiLU()]
        for _ in range(self.num_blocks):
            layers.extend(
                [
                    conv(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_groups_for_channels(hidden_channels), hidden_channels),
                    nn.SiLU(),
                ]
            )
        layers.append(conv(hidden_channels, output_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def config(self) -> dict:
        return {
            "model_type": "direct_latent_predictor",
            "spatial_dims": self.spatial_dims,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
        }

    def forward(self, condition_latent: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        condition_latent = resize_to_spatial(condition_latent, output_shape, self.spatial_dims)
        return self.net(condition_latent)


def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        return linear_beta_schedule(timesteps)
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    raise ValueError(f"Unsupported latent diffusion schedule: {schedule}")


def normalize_sampler_name(sampler: str) -> str:
    sampler_name = str(sampler or "DDPM").upper()
    if sampler_name not in {"DDPM", "DDIM"}:
        raise ValueError(f"Unsupported latent sampler: {sampler}. Use DDPM or DDIM.")
    return sampler_name


def extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    out = values.gather(0, timesteps.detach().to(values.device)).to(timesteps.device)
    return out.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class FrozenVAEEncoder:
    def __init__(self, checkpoint_path, device: torch.device):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        model_config = checkpoint["model_config"]
        spatial_dims = int(model_config["spatial_dims"])
        model_cls = VAE2D if spatial_dims == 2 else VAE3D
        model = model_cls(
            input_channels=int(model_config["input_channels"]),
            output_channels=int(model_config.get("output_channels", model_config["input_channels"])),
            latent_channels=int(model_config["latent_channels"]),
            hidden_channels=int(model_config["hidden_channels"]),
            channel_multipliers=tuple(model_config["channel_multipliers"]),
            output_activation=model_config.get("output_activation"),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.trainer_config = checkpoint.get("trainer_config", {})
        self.spatial_dims = spatial_dims
        self.input_channels = int(model_config["input_channels"])
        self.latent_channels = int(model_config["latent_channels"])

    @torch.no_grad()
    def encode(self, tensor: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        if tensor.shape[1] != self.input_channels:
            raise ValueError(
                f"Encoder {self.checkpoint_path} expects {self.input_channels} input channels, "
                f"got tensor shape {tuple(tensor.shape)}. Check latent_input_type/latent_target_type."
            )
        mu, logvar = self.model.encode(tensor)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu


def resolve_checkpoint(path_or_dir, checkpoint_name: str = "best_model.pth") -> Path:
    raw_path = Path(path_or_dir).expanduser()
    base_candidates = [
        raw_path,
        Path.cwd() / raw_path,
        diffusionsr_root() / raw_path,
        project_root() / raw_path,
    ]
    checked = []
    for base in base_candidates:
        checked.append(base)
        if base.is_file():
            return base.resolve()
        for name in [checkpoint_name, "ckpt.pth", "best_model.pth"]:
            candidate = base / name
            checked.append(candidate)
            if candidate.exists():
                return candidate.resolve()
    checked_text = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(f"No checkpoint found for {path_or_dir}. Checked: {checked_text}")


class LatentDiffusionTrainer:
    def __init__(
        self,
        results_folder,
        train_dataset,
        dev_dataset,
        input_encoder: FrozenVAEEncoder,
        target_encoder: FrozenVAEEncoder,
        input_type: str = "lr",
        target_type: str = "hr",
        mode: str = "diffusion",
        latent_prediction_target: str = "target",
        timesteps: int = 200,
        schedule: str = "linear",
        sampler: str = "DDPM",
        sample_timesteps: Optional[int] = None,
        ddim_eta: float = 0.0,
        loss_type: str = "huber",
        hidden_channels: int = 64,
        num_blocks: int = 4,
        sample_posterior: bool = False,
        depth_size: Optional[int] = None,
        num_workers: int = 0,
        log_interval: int = 50,
        sample_interval: int = 5,
        save_every: int = 0,
        grad_clip_norm: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        if mode not in ["diffusion", "nn"]:
            raise ValueError("mode must be 'diffusion' or 'nn'")
        if input_type not in _TARGET_INDEX:
            raise ValueError(f"Unsupported input_type: {input_type}")
        if target_type not in _TARGET_INDEX:
            raise ValueError(f"Unsupported target_type: {target_type}")
        latent_prediction_target = normalize_latent_prediction_target(latent_prediction_target)
        if latent_prediction_target == "latent_residual" and input_encoder.latent_channels != target_encoder.latent_channels:
            raise ValueError(
                "latent_residual prediction requires input and target encoders to use the same latent channel count; "
                f"got {input_encoder.latent_channels} and {target_encoder.latent_channels}."
            )
        if input_encoder.spatial_dims != target_encoder.spatial_dims:
            raise ValueError(
                "Input and target encoders must use the same spatial_dims; "
                f"got {input_encoder.spatial_dims} and {target_encoder.spatial_dims}."
            )

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.input_type = input_type
        self.target_type = target_type
        self.mode = mode
        self.latent_prediction_target = latent_prediction_target
        self.timesteps = int(timesteps)
        if self.timesteps < 1:
            raise ValueError("timesteps must be at least 1")
        self.schedule = str(schedule).lower()
        self.sampler = normalize_sampler_name(sampler)
        self.sample_timesteps = int(sample_timesteps or self.timesteps)
        if self.timesteps > 1 and self.sample_timesteps < 2:
            raise ValueError("sample_timesteps must be at least 2 when timesteps > 1")
        self.sample_timesteps = min(self.sample_timesteps, self.timesteps)
        self.ddim_eta = float(ddim_eta)
        if self.ddim_eta < 0:
            raise ValueError("ddim_eta must be non-negative")
        self.loss_type = loss_type
        self.sample_posterior = bool(sample_posterior)
        self.spatial_dims = target_encoder.spatial_dims
        self.depth_size = int(depth_size or getattr(train_dataset, "inflate_dim", None) or 1)
        self.num_workers = int(num_workers)
        self.log_interval = int(log_interval)
        self.sample_interval = int(sample_interval)
        self.save_every = int(save_every)
        self.grad_clip_norm = grad_clip_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_validation_loss = math.inf
        self.start_epoch = 0
        self.global_step = 0

        if self.mode == "diffusion":
            self.model = ConditionalLatentDenoiser(
                spatial_dims=self.spatial_dims,
                latent_channels=target_encoder.latent_channels,
                condition_channels=input_encoder.latent_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
            ).to(self.device)
            self._init_diffusion_schedule()
        else:
            self.model = DirectLatentPredictor(
                spatial_dims=self.spatial_dims,
                input_channels=input_encoder.latent_channels,
                output_channels=target_encoder.latent_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
            ).to(self.device)

        self.trainer_config = {
            "mode": self.mode,
            "input_type": self.input_type,
            "target_type": self.target_type,
            "latent_prediction_target": self.latent_prediction_target,
            "spatial_dims": self.spatial_dims,
            "depth_size": self.depth_size,
            "timesteps": self.timesteps,
            "schedule": self.schedule,
            "sampler": self.sampler,
            "sample_timesteps": self.sample_timesteps,
            "ddim_eta": self.ddim_eta,
            "loss_type": self.loss_type,
            "sample_posterior": self.sample_posterior,
            "hidden_channels": hidden_channels,
            "num_blocks": num_blocks,
            "num_workers": self.num_workers,
            "log_interval": self.log_interval,
            "sample_interval": self.sample_interval,
            "save_every": self.save_every,
            "grad_clip_norm": self.grad_clip_norm,
            "input_encoder_checkpoint": str(input_encoder.checkpoint_path),
            "target_encoder_checkpoint": str(target_encoder.checkpoint_path),
        }
        self._write_metadata()
        self._log_timestamp("initialized")

    def _init_diffusion_schedule(self) -> None:
        betas = make_beta_schedule(self.schedule, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(self.device)
        self.posterior_variance = posterior_variance.clamp_min(1e-20).to(self.device)

    def _write_metadata(self) -> None:
        metadata = {
            "created_at": now_iso(),
            "model_config": self.model.config(),
            "trainer_config": self.trainer_config,
            "input_encoder_model_config": self.input_encoder.model_config,
            "target_encoder_model_config": self.target_encoder.model_config,
            "train_samples": len(self.train_dataset),
            "dev_samples": len(self.dev_dataset),
        }
        with open(self.results_folder / "latentdiff_metadata.json", "w") as f:
            json.dump(_jsonable(metadata), f, indent=2)

    def _log_timestamp(self, event: str, **metadata) -> None:
        record = {"event": event, "timestamp": now_iso()}
        record.update(metadata)
        with open(self.results_folder / "timestamps.log", "a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")

    def _format_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2:
            if tensor.ndim == 3:
                return tensor.unsqueeze(1)
            if tensor.ndim != 4:
                raise ValueError(f"Expected 4D 2D batch, got {tuple(tensor.shape)}")
            return tensor

        if tensor.ndim == 5:
            return tensor
        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D flattened-volume or 5D volume batch, got {tuple(tensor.shape)}")
        batch_size, channels, height, width = tensor.shape
        if channels % self.depth_size != 0:
            raise ValueError(f"Channel count {channels} is not divisible by depth size {self.depth_size}")
        return tensor.reshape(batch_size, channels // self.depth_size, self.depth_size, height, width)

    def _select_tensor(self, batch, data_type: str) -> torch.Tensor:
        tensor = batch[_TARGET_INDEX[data_type]]
        return self._format_tensor(tensor).to(self.device).float()

    @torch.no_grad()
    def _encode_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self._select_tensor(batch, self.input_type)
        target_tensor = self._select_tensor(batch, self.target_type)
        input_latent = self.input_encoder.encode(input_tensor, sample_posterior=self.sample_posterior)
        target_latent = self.target_encoder.encode(target_tensor, sample_posterior=self.sample_posterior)
        return input_latent.detach(), target_latent.detach()

    def _latent_residual_base(self, condition_latent: torch.Tensor, reference_latent: torch.Tensor) -> torch.Tensor:
        base = resize_to_spatial(condition_latent, reference_latent.shape[-self.spatial_dims :], self.spatial_dims)
        if base.shape[1] != reference_latent.shape[1]:
            raise ValueError(
                "latent_residual prediction requires condition and target latents to have the same channel count; "
                f"got {base.shape[1]} and {reference_latent.shape[1]}."
            )
        return base

    def _model_target_latent(self, condition_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        if self.latent_prediction_target == "latent_residual":
            return target_latent - self._latent_residual_base(condition_latent, target_latent)
        return target_latent

    def _compose_prediction_latent(self, condition_latent: torch.Tensor, predicted_model_latent: torch.Tensor) -> torch.Tensor:
        if self.latent_prediction_target == "latent_residual":
            return self._latent_residual_base(condition_latent, predicted_model_latent) + predicted_model_latent
        return predicted_model_latent

    def q_sample(self, x_start: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timestep, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def _sample_timestep_indices(self, sample_timesteps: Optional[int] = None) -> torch.Tensor:
        step_count = int(sample_timesteps or self.sample_timesteps)
        if self.timesteps > 1 and step_count < 2:
            raise ValueError("sample_timesteps must be at least 2 when timesteps > 1")
        step_count = max(1, min(step_count, self.timesteps))
        if step_count == 1:
            return torch.tensor([self.timesteps - 1], device=self.device, dtype=torch.long)
        return torch.linspace(0, self.timesteps - 1, steps=step_count, device=self.device).round().long().unique()

    def _predict_x0_from_noise(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timestep, noisy_latent.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timestep, noisy_latent.shape)
        return (noisy_latent - sqrt_one_minus * predicted_noise) / sqrt_alpha.clamp_min(1e-12)

    @torch.no_grad()
    def _sample_ddpm(
        self,
        condition_latent: torch.Tensor,
        latent_shape: Sequence[int],
        sample_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        shape = (condition_latent.shape[0], self.target_encoder.latent_channels, *tuple(latent_shape))
        latent = torch.randn(shape, device=self.device)
        for time_index in reversed(self._sample_timestep_indices(sample_timesteps).tolist()):
            timestep = torch.full((shape[0],), int(time_index), device=self.device, dtype=torch.long)
            predicted_noise = self.model(latent, timestep, condition_latent)
            beta = extract(self.betas, timestep, latent.shape)
            sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timestep, latent.shape)
            sqrt_recip_alpha = extract(self.sqrt_recip_alphas, timestep, latent.shape)
            model_mean = sqrt_recip_alpha * (latent - beta * predicted_noise / sqrt_one_minus.clamp_min(1e-12))
            if time_index == 0:
                latent = model_mean
            else:
                variance = extract(self.posterior_variance, timestep, latent.shape)
                latent = model_mean + torch.sqrt(variance) * torch.randn_like(latent)
        return latent

    @torch.no_grad()
    def _sample_ddim(
        self,
        condition_latent: torch.Tensor,
        latent_shape: Sequence[int],
        sample_timesteps: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        shape = (condition_latent.shape[0], self.target_encoder.latent_channels, *tuple(latent_shape))
        latent = torch.randn(shape, device=self.device)
        eta = self.ddim_eta if eta is None else float(eta)
        indices = self._sample_timestep_indices(sample_timesteps).tolist()
        for position in reversed(range(len(indices))):
            time_index = int(indices[position])
            next_index = int(indices[position - 1]) if position > 0 else -1
            timestep = torch.full((shape[0],), time_index, device=self.device, dtype=torch.long)
            predicted_noise = self.model(latent, timestep, condition_latent)
            alpha = extract(self.alphas_cumprod, timestep, latent.shape)
            predicted_x0 = self._predict_x0_from_noise(latent, timestep, predicted_noise)
            if next_index < 0:
                latent = predicted_x0
                continue

            next_timestep = torch.full((shape[0],), next_index, device=self.device, dtype=torch.long)
            alpha_next = extract(self.alphas_cumprod, next_timestep, latent.shape)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_next) / (1.0 - alpha)).clamp_min(0.0)
                * (1.0 - alpha / alpha_next).clamp_min(0.0)
            )
            direction = torch.sqrt((1.0 - alpha_next - sigma.square()).clamp_min(0.0)) * predicted_noise
            latent = torch.sqrt(alpha_next) * predicted_x0 + direction
            if eta > 0:
                latent = latent + sigma * torch.randn_like(latent)
        return latent

    @torch.no_grad()
    def sample_latent(
        self,
        condition_latent: torch.Tensor,
        latent_shape: Sequence[int],
        sampler: Optional[str] = None,
        sample_timesteps: Optional[int] = None,
        ddim_eta: Optional[float] = None,
    ) -> torch.Tensor:
        latent_shape = tuple(int(dim) for dim in latent_shape)
        condition_latent = condition_latent.to(self.device)
        if self.mode == "nn":
            predicted_model_latent = self.model(condition_latent, output_shape=latent_shape)
            return self._compose_prediction_latent(condition_latent, predicted_model_latent)

        sampler_name = normalize_sampler_name(sampler or self.sampler)
        if sampler_name == "DDPM":
            predicted_model_latent = self._sample_ddpm(condition_latent, latent_shape, sample_timesteps=sample_timesteps)
        else:
            predicted_model_latent = self._sample_ddim(
                condition_latent,
                latent_shape,
                sample_timesteps=sample_timesteps,
                eta=ddim_eta,
            )
        return self._compose_prediction_latent(condition_latent, predicted_model_latent)

    def _diffusion_step(self, condition_latent: torch.Tensor, target_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        model_target_latent = self._model_target_latent(condition_latent, target_latent)
        timestep = torch.randint(0, self.timesteps, (model_target_latent.shape[0],), device=self.device).long()
        noise = torch.randn_like(model_target_latent)
        noisy_latent = self.q_sample(model_target_latent, timestep, noise)
        predicted_noise = self.model(noisy_latent, timestep, condition_latent)
        loss = loss_fn(predicted_noise, noise, self.loss_type)
        return {"loss": loss, "noise_loss": loss}

    def _nn_step(self, condition_latent: torch.Tensor, target_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        model_target_latent = self._model_target_latent(condition_latent, target_latent)
        prediction = self.model(condition_latent, output_shape=model_target_latent.shape[-self.spatial_dims:])
        latent_loss = loss_fn(prediction, model_target_latent, self.loss_type)
        return {"loss": latent_loss, "latent_loss": latent_loss}

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> Dict[str, float]:
        self.model.train(train)
        metric_names = ["loss", "noise_loss"] if self.mode == "diffusion" else ["loss", "latent_loss"]
        totals = {name: 0.0 for name in metric_names}
        sample_count = 0
        split = "train" if train else "validation"
        iterator = tqdm(loader, desc=f"{split} latent epoch {epoch + 1}", leave=False)

        for batch in iterator:
            condition_latent, target_latent = self._encode_batch(batch)
            batch_size = target_latent.shape[0]
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                losses = (
                    self._diffusion_step(condition_latent, target_latent)
                    if self.mode == "diffusion"
                    else self._nn_step(condition_latent, target_latent)
                )

            if train:
                losses["loss"].backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
                self.optimizer.step()
                self.global_step += 1

            for name in metric_names:
                totals[name] += losses[name].detach().item() * batch_size
            sample_count += batch_size
            iterator.set_postfix(loss=losses["loss"].detach().item())

            if train and self.log_interval > 0 and self.global_step % self.log_interval == 0:
                _wandb_log(
                    {f"{split}/{name}": losses[name].detach().item() for name in metric_names},
                    step=self.global_step,
                )

        divisor = max(1, sample_count)
        return {name: total / divisor for name, total in totals.items()}

    def _make_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _checkpoint_payload(self, epoch: int, metrics: dict) -> dict:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "metrics": metrics,
            "model_config": self.model.config(),
            "trainer_config": self.trainer_config,
            "saved_at": now_iso(),
        }

    def _log_artifact(self, path: Path, aliases=None, metadata=None) -> None:
        if wandb.run is None:
            return
        aliases = aliases or ["latest"]
        artifact_name = f"latentdiff_{self.mode}_{Path(self.results_folder).name}".replace("/", "_")
        try:
            artifact = wandb.Artifact(artifact_name, type="model", metadata=_jsonable(metadata or {}))
            artifact.add_file(str(path))
            wandb.log_artifact(artifact, aliases=aliases)
        except Exception as exc:
            print(f"Could not log wandb artifact {path}: {exc}")

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        payload = self._checkpoint_payload(epoch, metrics)
        latest_path = self.results_folder / "ckpt.pth"
        torch.save(payload, latest_path)
        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            torch.save(payload, self.results_folder / f"ckpt_epoch_{epoch + 1:04d}.pth")
        if is_best:
            best_path = self.results_folder / "best_model.pth"
            torch.save(payload, best_path)
            self._log_timestamp("best_model", epoch=epoch + 1, validation_loss=metrics["validation/loss"])
            self._log_artifact(best_path, aliases=["best", f"epoch-{epoch + 1}"], metadata=metrics)

    def _load_checkpoint(self, restart_dir: str) -> None:
        checkpoint_path = Path(restart_dir) / "ckpt.pth"
        if not checkpoint_path.exists():
            checkpoint_path = Path(restart_dir) / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No latent diffusion checkpoint found in {restart_dir}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self._log_timestamp("restart", checkpoint=str(checkpoint_path), start_epoch=self.start_epoch + 1)

    def _append_history(self, row: dict) -> None:
        path = self.results_folder / "history.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @torch.no_grad()
    def _save_latent_preview(self, loader: DataLoader, epoch: int) -> Optional[Path]:
        try:
            batch = next(iter(loader))
        except StopIteration:
            return None
        self.model.eval()
        condition_latent, target_latent = self._encode_batch(batch)

        preview = self.sample_latent(condition_latent, target_latent.shape[-self.spatial_dims:])

        target_slice = target_latent[0, 0].detach().cpu().numpy()
        preview_slice = preview[0, 0].detach().cpu().numpy()
        if self.spatial_dims == 3:
            center = target_slice.shape[0] // 2
            target_slice = target_slice[center]
            preview_slice = preview_slice[center]
        error_slice = np.abs(preview_slice - target_slice)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=180)
        preview_title = (
            "pred latent"
            if self.mode == "nn"
            else f"pred latent ({self.sampler}, {self.sample_timesteps} steps)"
        )
        for ax, array, title in zip(axes, [target_slice, preview_slice, error_slice], ["target latent", preview_title, "absolute error"]):
            cmap = "magma" if title == "absolute error" else "coolwarm"
            im = ax.imshow(array.T, origin="lower", cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        image_dir = self.results_folder / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"latent_preview_epoch_{epoch + 1:04d}.png"
        fig.savefig(path)
        plt.close(fig)
        _wandb_log({"validation/latent_preview": wandb.Image(str(path)), "epoch": epoch + 1}, step=self.global_step)
        return path

    def train(
        self,
        epochs: int,
        restart: bool = False,
        restart_dir: str = "",
        additional_epochs: Optional[int] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> dict:
        self.optimizer = Adam(self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        if restart:
            self._load_checkpoint(restart_dir)
        end_epoch = self.start_epoch + int(additional_epochs) if restart and additional_epochs is not None else int(epochs)

        train_loader = self._make_loader(self.train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = self._make_loader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        self._log_timestamp(
            "train_start",
            start_epoch=self.start_epoch + 1,
            end_epoch=end_epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mode=self.mode,
        )

        train_losses = []
        validation_losses = []
        final_metrics = {}
        for epoch in tqdm(range(self.start_epoch, end_epoch), desc=f"latent {self.mode} epochs"):
            started = time.time()
            train_metrics = self._run_epoch(train_loader, epoch, train=True)
            with torch.no_grad():
                validation_metrics = self._run_epoch(dev_loader, epoch, train=False)
            duration = time.time() - started

            metrics = {
                **{f"train/{name}": value for name, value in train_metrics.items()},
                **{f"validation/{name}": value for name, value in validation_metrics.items()},
                "epoch": epoch + 1,
                "epoch_duration_seconds": duration,
            }
            final_metrics = metrics
            train_losses.append(metrics["train/loss"])
            validation_losses.append(metrics["validation/loss"])
            np.savetxt(self.results_folder / "loss_epoch.txt", np.asarray(train_losses))
            np.savetxt(self.results_folder / "validation_loss_epoch.txt", np.asarray(validation_losses))

            is_best = metrics["validation/loss"] < self.best_validation_loss
            if is_best:
                self.best_validation_loss = metrics["validation/loss"]
            self._save_checkpoint(epoch, metrics, is_best=is_best)
            self._append_history({"timestamp": now_iso(), **metrics, "best_validation_loss": self.best_validation_loss})
            self._log_timestamp(
                "epoch_end",
                epoch=epoch + 1,
                train_loss=metrics["train/loss"],
                validation_loss=metrics["validation/loss"],
                duration_seconds=duration,
            )
            _wandb_log(metrics, step=self.global_step)

            if self.sample_interval > 0 and (epoch + 1) % self.sample_interval == 0:
                self._save_latent_preview(dev_loader, epoch)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            print(
                "Epoch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}, Time: {:.2f}s".format(
                    epoch + 1, metrics["train/loss"], metrics["validation/loss"], duration
                )
            )

        self._log_artifact(self.results_folder / "ckpt.pth", aliases=["latest"], metadata=final_metrics)
        self._log_timestamp("train_end", best_validation_loss=self.best_validation_loss)
        return final_metrics


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def diffusionsr_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    candidates = [
        path,
        Path.cwd() / path,
        Path.cwd() / "configs" / path,
        diffusionsr_root() / "configs" / path,
        project_root() / "diffusionsr" / "configs" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config path: {config_path}")


def resolve_root_folder(config_dict: dict) -> str:
    root_folder = Path(config_dict["root_folder"]).expanduser()
    if root_folder.is_absolute():
        return str(root_folder)
    candidates = [Path.cwd() / root_folder, diffusionsr_root() / root_folder, project_root() / root_folder]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((diffusionsr_root() / root_folder).resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Train latent diffusion or a direct latent NN between two frozen VAE encoders.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path or file name under diffusionsr/configs")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--restart_dir", type=str, default="", help="Existing latentdiff run directory to resume")
    parser.add_argument("--additional_epochs", type=int, default=None, help="Extra epochs to run when restarting")
    parser.add_argument("--mode", type=str, default=None, choices=["diffusion", "nn"], help="Override latent_training_mode")
    parser.add_argument(
        "--latent_prediction_target",
        type=str,
        default=None,
        choices=sorted(_LATENT_PREDICTION_TARGET_ALIASES),
        help="Train on the target latent directly or on target_latent - input_latent.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override latent_timesteps for diffusion training")
    parser.add_argument("--sampler", type=str, default=None, choices=["DDPM", "DDIM", "ddpm", "ddim"], help="Sampler for latent previews/generation")
    parser.add_argument("--sample_timesteps", type=int, default=None, help="Reverse sampling steps for DDPM/DDIM previews")
    parser.add_argument("--ddim_eta", type=float, default=None, help="DDIM stochasticity; 0.0 is deterministic")
    return parser.parse_args()


def build_dataset(config: dict, split: str, spatial_dims: int):
    inflate_dim = config.get("inflate_dim") if spatial_dims == 3 else None
    return SimulationXZDataset(
        downscale_method=config["downscale_method"],
        normalize=config["normalize_method"],
        split=split,
        root_folder=resolve_root_folder(config),
        n_steps=int(config.get("n_steps", 1)),
        field_names=config_to_field_names(config),
        out_steps=config.get("out_steps"),
        inflate_dim=inflate_dim,
        inflate_method=config.get("inflate_method", "repeat"),
        crop_mode=config.get("crop_mode"),
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_encoder_checkpoint = resolve_checkpoint(
        config["latent_input_encoder_dir"],
        config.get("latent_input_encoder_checkpoint", "best_model.pth"),
    )
    target_encoder_checkpoint = resolve_checkpoint(
        config["latent_target_encoder_dir"],
        config.get("latent_target_encoder_checkpoint", "best_model.pth"),
    )
    input_encoder = FrozenVAEEncoder(input_encoder_checkpoint, device=device)
    target_encoder = FrozenVAEEncoder(target_encoder_checkpoint, device=device)
    if input_encoder.spatial_dims != target_encoder.spatial_dims:
        raise ValueError("Input and target VAE encoders must both be 2D or both be 3D.")

    spatial_dims = target_encoder.spatial_dims
    train_dataset = build_dataset(config, "train", spatial_dims)
    dev_dataset = build_dataset(config, "dev", spatial_dims)

    restart = bool(args.restart_dir)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    mode = args.mode or config.get("latent_training_mode", "diffusion")
    latent_prediction_target = normalize_latent_prediction_target(
        args.latent_prediction_target or config.get("latent_prediction_target", "target")
    )
    timesteps = int(args.timesteps or config.get("latent_timesteps", config.get("timesteps", 200)))
    sampler = args.sampler or config.get("latent_sampler", "DDPM")
    sample_timesteps = args.sample_timesteps
    if sample_timesteps is None:
        sample_timesteps = config.get("latent_sample_timesteps", timesteps)
    sample_timesteps = int(sample_timesteps)
    ddim_eta = args.ddim_eta
    if ddim_eta is None:
        ddim_eta = config.get("latent_ddim_eta", 0.0)
    ddim_eta = float(ddim_eta)
    combined_config = {
        **config,
        **vars(args),
        "effective_latent_training_mode": mode,
        "effective_latent_prediction_target": latent_prediction_target,
        "effective_latent_timesteps": timesteps,
        "effective_latent_sampler": normalize_sampler_name(sampler),
        "effective_latent_sample_timesteps": sample_timesteps,
        "effective_latent_ddim_eta": ddim_eta,
    }
    if restart:
        results_dir = Path(args.restart_dir)
    else:
        mode_tag = mode if latent_prediction_target == "target" else f"{mode}_{latent_prediction_target}"
        results_dir = Path("runs") / config["downscale_method"] / f"latentdiff_{mode_tag}" / timestamp / config["normalize_method"] / f"n_steps_{int(config.get('n_steps', 1))}"
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, results_dir / "configuration.yml")
    with open(results_dir / "information.txt", "w") as f:
        f.write(
            f"mode: {mode}\n"
            f"latent_prediction_target: {latent_prediction_target}\n"
            f"diffusion_timesteps: {timesteps}\n"
            f"sampler: {normalize_sampler_name(sampler)}\n"
            f"sample_timesteps: {sample_timesteps}\n"
            f"ddim_eta: {ddim_eta}\n"
            f"input_type: {config.get('latent_input_type', 'lr')}\n"
            f"target_type: {config.get('latent_target_type', 'hr')}\n"
            f"input_encoder: {input_encoder_checkpoint}\n"
            f"target_encoder: {target_encoder_checkpoint}\n"
            f"created_at: {timestamp}\n"
        )

    wandb.init(
        project=config.get("wandb_project", "Flow3D_SuperResolution"),
        entity=os.getenv("WANDB_ENTITY"),
        config=combined_config,
        mode=config.get("wandb_mode", "online"),
    )

    trainer = LatentDiffusionTrainer(
        results_folder=results_dir,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        input_encoder=input_encoder,
        target_encoder=target_encoder,
        input_type=config.get("latent_input_type", "lr"),
        target_type=config.get("latent_target_type", "hr"),
        mode=mode,
        latent_prediction_target=latent_prediction_target,
        timesteps=timesteps,
        schedule=config.get("latent_schedule", "linear"),
        sampler=sampler,
        sample_timesteps=sample_timesteps,
        ddim_eta=ddim_eta,
        loss_type=config.get("latent_loss_type", "huber"),
        hidden_channels=int(config.get("latent_hidden_channels", 64)),
        num_blocks=int(config.get("latent_num_blocks", 4)),
        sample_posterior=bool(config.get("latent_sample_posterior", False)),
        depth_size=config.get("inflate_dim"),
        num_workers=int(config.get("latent_num_workers", 0)),
        log_interval=int(config.get("latent_log_interval", 50)),
        sample_interval=int(config.get("latent_sample_interval", 5)),
        save_every=int(config.get("latent_save_every", 0)),
        grad_clip_norm=config.get("latent_grad_clip_norm"),
        device=device,
    )
    trainer.train(
        epochs=int(config.get("epochs", 100)),
        restart=restart,
        restart_dir=args.restart_dir,
        additional_epochs=args.additional_epochs,
        batch_size=int(config.get("batch_size", 16)),
        learning_rate=float(config.get("latent_learning_rate") or config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("latent_weight_decay", 0.0)),
    )


if __name__ == "__main__":
    main()
