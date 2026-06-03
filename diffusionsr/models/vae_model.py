from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


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


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_groups_for_channels(channels), channels)


def _interpolate_mode(spatial_dims: int) -> str:
    return "bilinear" if spatial_dims == 2 else "trilinear"


def _as_tuple(values: Sequence[int], expected_dims: int) -> Tuple[int, ...]:
    values = tuple(int(v) for v in values)
    if len(values) == expected_dims:
        return values
    if len(values) == expected_dims + 2:
        return values[-expected_dims:]
    raise ValueError(f"Expected {expected_dims} spatial dimensions, got {values}")


class _ConvBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        conv = _conv(spatial_dims)
        self.block = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
            _norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ConvVAE(nn.Module):
    """Convolutional VAE that returns spatial latent maps for latent diffusion."""

    def __init__(
        self,
        spatial_dims: int,
        input_channels: int,
        output_channels: Optional[int] = None,
        latent_channels: int = 4,
        hidden_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        output_activation: Optional[str] = None,
        min_logvar: float = -30.0,
        max_logvar: float = 20.0,
    ):
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not channel_multipliers:
            raise ValueError("channel_multipliers must contain at least one entry")

        self.spatial_dims = int(spatial_dims)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels or input_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.channel_multipliers = tuple(int(multiplier) for multiplier in channel_multipliers)
        self.output_activation = output_activation
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        conv = _conv(self.spatial_dims)
        hidden_dims = [self.hidden_channels * multiplier for multiplier in self.channel_multipliers]

        encoder_layers = []
        in_channels = self.input_channels
        for out_channels in hidden_dims:
            encoder_layers.append(_ConvBlock(self.spatial_dims, in_channels, out_channels))
            encoder_layers.extend(
                [
                    conv(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    _norm(out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.to_mu = conv(in_channels, self.latent_channels, kernel_size=1)
        self.to_logvar = conv(in_channels, self.latent_channels, kernel_size=1)

        self.decoder_input = conv(self.latent_channels, hidden_dims[-1], kernel_size=1)
        decoder_layers = []
        in_channels = hidden_dims[-1]
        decoder_dims = list(reversed(hidden_dims[:-1])) + [hidden_dims[0]]
        for out_channels in decoder_dims:
            decoder_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    _ConvBlock(self.spatial_dims, in_channels, out_channels),
                ]
            )
            in_channels = out_channels

        self.decoder = nn.Sequential(*decoder_layers)
        self.output = conv(in_channels, self.output_channels, kernel_size=3, padding=1)

    @property
    def latent_downsample_factor(self) -> int:
        return 2 ** len(self.channel_multipliers)

    def latent_shape(self, spatial_shape: Sequence[int]) -> Tuple[int, ...]:
        spatial_shape = _as_tuple(spatial_shape, self.spatial_dims)
        factor = self.latent_downsample_factor
        return tuple(max(1, (dim + factor - 1) // factor) for dim in spatial_shape)

    def config(self) -> dict:
        return {
            "spatial_dims": self.spatial_dims,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "latent_channels": self.latent_channels,
            "hidden_channels": self.hidden_channels,
            "channel_multipliers": self.channel_multipliers,
            "output_activation": self.output_activation,
            "min_logvar": self.min_logvar,
            "max_logvar": self.max_logvar,
        }

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(self.min_logvar, self.max_logvar)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(
        self,
        z: torch.Tensor,
        output_shape: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        reconstruction = self.decoder(self.decoder_input(z))
        reconstruction = self.output(reconstruction)
        if output_shape is not None:
            spatial_shape = _as_tuple(output_shape, self.spatial_dims)
            if tuple(reconstruction.shape[-self.spatial_dims:]) != spatial_shape:
                reconstruction = F.interpolate(
                    reconstruction,
                    size=spatial_shape,
                    mode=_interpolate_mode(self.spatial_dims),
                    align_corners=False,
                )

        if self.output_activation == "tanh":
            reconstruction = torch.tanh(reconstruction)
        elif self.output_activation == "sigmoid":
            reconstruction = torch.sigmoid(reconstruction)
        elif self.output_activation is not None:
            raise ValueError(f"Unsupported output_activation: {self.output_activation}")
        return reconstruction

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        output_shape: Optional[Sequence[int]] = None,
    ) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        reconstruction = self.decode(z, output_shape=output_shape or x.shape[-self.spatial_dims:])
        return VAEOutput(reconstruction=reconstruction, mu=mu, logvar=logvar, z=z)


class VAE2D(_ConvVAE):
    def __init__(
        self,
        input_channels: int,
        output_channels: Optional[int] = None,
        latent_channels: int = 4,
        hidden_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        output_activation: Optional[str] = None,
    ):
        super().__init__(
            spatial_dims=2,
            input_channels=input_channels,
            output_channels=output_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            channel_multipliers=channel_multipliers,
            output_activation=output_activation,
        )


class VAE3D(_ConvVAE):
    def __init__(
        self,
        input_channels: int,
        output_channels: Optional[int] = None,
        latent_channels: int = 4,
        hidden_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        output_activation: Optional[str] = None,
    ):
        super().__init__(
            spatial_dims=3,
            input_channels=input_channels,
            output_channels=output_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            channel_multipliers=channel_multipliers,
            output_activation=output_activation,
        )


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(reconstruction, target)
    if loss_type == "l2":
        return F.mse_loss(reconstruction, target)
    if loss_type == "huber":
        return F.smooth_l1_loss(reconstruction, target)
    raise ValueError(f"Unsupported reconstruction loss: {loss_type}")


def vae_loss(
    output: VAEOutput,
    target: torch.Tensor,
    beta: float = 1e-4,
    loss_type: str = "l1",
) -> dict:
    recon = reconstruction_loss(output.reconstruction, target, loss_type=loss_type)
    kl = kl_divergence(output.mu, output.logvar)
    total = recon + float(beta) * kl
    return {"loss": total, "reconstruction_loss": recon, "kl_loss": kl}


ConvVAE2D = VAE2D
ConvVAE3D = VAE3D

__all__ = [
    "VAEOutput",
    "VAE2D",
    "VAE3D",
    "ConvVAE2D",
    "ConvVAE3D",
    "kl_divergence",
    "reconstruction_loss",
    "vae_loss",
]
