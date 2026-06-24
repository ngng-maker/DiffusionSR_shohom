from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

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

def _latent_vectors(z: torch.Tensor) -> torch.Tensor:
    if z.ndim < 3:
        raise ValueError(f"Expected spatial latent maps with at least 3 dimensions, got {tuple(z.shape)}")
    channel_axis = 1
    spatial_axes = tuple(axis for axis in range(z.ndim) if axis not in (0, channel_axis))
    return z.permute(0, *spatial_axes, channel_axis).reshape(-1, z.shape[channel_axis])


def codebook_clustering_loss(
    z: torch.Tensor,
    codebook: Union[nn.Embedding, torch.Tensor],
    beta: float = 0.25,
) -> Dict[str, torch.Tensor]:
    entries = codebook.weight if isinstance(codebook, nn.Embedding) else codebook
    if entries.ndim != 2:
        raise ValueError(f"Expected a 2D codebook, got shape {tuple(entries.shape)}")
    if entries.shape[1] != z.shape[1]:
        raise ValueError(
            f"Codebook vectors have {entries.shape[1]} channels but latent maps have {z.shape[1]}"
        )

    vectors = _latent_vectors(z)
    distances = (
        vectors.pow(2).sum(dim=1, keepdim=True)
        - 2.0 * vectors @ entries.t()
        + entries.pow(2).sum(dim=1).unsqueeze(0)
    )
    encoding_indices = distances.argmin(dim=1)
    nearest = entries.index_select(0, encoding_indices)

    embedding = F.mse_loss(nearest, vectors.detach())
    commitment = F.mse_loss(vectors, nearest.detach())
    total = embedding + float(beta) * commitment

    encodings = F.one_hot(encoding_indices, entries.shape[0]).type_as(vectors)
    avg_probs = encodings.mean(dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return {
        "codebook_loss": total,
        "codebook_embedding_loss": embedding,
        "codebook_commitment_loss": commitment,
        "codebook_perplexity": perplexity,
    }


def code_book_loss(
    z: torch.Tensor,
    codebook: Union[nn.Embedding, torch.Tensor],
    beta: float = 0.25,
) -> torch.Tensor:
    return codebook_clustering_loss(z, codebook, beta=beta)["codebook_loss"]


def feature_matching_loss(
    reconstruction_features: torch.Tensor,
    target_features: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(reconstruction_features, target_features)
    if loss_type == "l2":
        return F.mse_loss(reconstruction_features, target_features)
    if loss_type == "huber":
        return F.smooth_l1_loss(reconstruction_features, target_features)
    raise ValueError(f"Unsupported feature loss: {loss_type}")


class VGGFeatureLoss(nn.Module):
    """VGG-style perceptual loss for 2D images or slices from 3D volumes."""

    def __init__(
        self,
        model_name: str = "vgg19",
        feature_node: str = "features.35",
        pretrained: bool = True,
        resize: Optional[int] = 224,
        loss_type: str = "l1",
        input_normalization: str = "minmax",
        slice_mode: str = "center",
    ):
        super().__init__()
        from torchvision import models, transforms
        from torchvision.models.feature_extraction import create_feature_extractor

        model_name = model_name.lower()
        if model_name not in {"vgg16", "vgg19"}:
            raise ValueError("feature loss supports vgg16 and vgg19")

        model_factory = getattr(models, model_name)
        weights = None
        if pretrained:
            weight_name = "VGG16_Weights" if model_name == "vgg16" else "VGG19_Weights"
            weights_enum = getattr(models, weight_name, None)
            if weights_enum is not None:
                weights = weights_enum.IMAGENET1K_V1
                model = model_factory(weights=weights)
            else:
                model = model_factory(pretrained=True)
        else:
            try:
                model = model_factory(weights=None)
            except TypeError:
                model = model_factory(pretrained=False)

        self.feature_node = feature_node
        self.feature_extractor = create_feature_extractor(model, [feature_node])
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.resize = int(resize) if resize else None
        self.loss_type = loss_type
        self.input_normalization = input_normalization.lower()
        self.slice_mode = slice_mode.lower()

        if weights is not None and hasattr(weights, "transforms"):
            transform = weights.transforms()
            mean = getattr(transform, "mean", [0.485, 0.456, 0.406])
            std = getattr(transform, "std", [0.229, 0.224, 0.225])
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _to_2d_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 5:
            if self.slice_mode == "center":
                tensor = tensor[:, :, tensor.shape[2] // 2]
            elif self.slice_mode == "all":
                tensor = tensor.permute(0, 2, 1, 3, 4).reshape(-1, tensor.shape[1], tensor.shape[3], tensor.shape[4])
            else:
                raise ValueError("feature slice_mode must be 'center' or 'all'")
        elif tensor.ndim != 4:
            raise ValueError(f"Expected a 4D image batch or 5D volume batch, got {tuple(tensor.shape)}")

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] == 2:
            tensor = torch.cat([tensor, tensor[:, :1]], dim=1)
        elif tensor.shape[1] > 3:
            tensor = tensor[:, :3]

        if self.input_normalization == "minmax":
            dims = tuple(range(2, tensor.ndim))
            min_value = tensor.amin(dim=dims, keepdim=True)
            max_value = tensor.amax(dim=dims, keepdim=True)
            tensor = (tensor - min_value) / (max_value - min_value).clamp_min(1e-6)
        elif self.input_normalization == "none":
            pass
        else:
            raise ValueError("feature input_normalization must be 'minmax' or 'none'")

        if self.resize is not None and tuple(tensor.shape[-2:]) != (self.resize, self.resize):
            tensor = F.interpolate(tensor, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        return self.normalize(tensor)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reconstruction_tensor = self._to_2d_batch(reconstruction)
        target_tensor = self._to_2d_batch(target)
        reconstruction_features = self.feature_extractor(reconstruction_tensor)[self.feature_node]
        with torch.no_grad():
            target_features = self.feature_extractor(target_tensor)[self.feature_node]
        return feature_matching_loss(reconstruction_features, target_features, loss_type=self.loss_type)


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        input_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        max_channels: int = 256,
    ):
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        conv = _conv(spatial_dims)
        layers = [
            conv(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = hidden_channels
        for layer_index in range(1, num_layers):
            out_channels = min(hidden_channels * (2 ** layer_index), max_channels)
            layers.extend(
                [
                    conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    _norm(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels
        layers.append(conv(in_channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generator_gan_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def discriminator_gan_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)

def vae_loss(
    output: VAEOutput,
    target: torch.Tensor,
    beta: float = 1e-4,
    loss_type: str = "l1",
    feature_loss: Optional[torch.Tensor] = None,
    feature_weight: float = 0.0,
    adversarial_loss: Optional[torch.Tensor] = None,
    adversarial_weight: float = 0.0,
    codebook: Optional[Union[nn.Embedding, torch.Tensor]] = None,
    codebook_weight: float = 0.0,
    codebook_beta: float = 0.25,
    additional_terms: Optional[dict] = None,
) -> dict:
    if additional_terms is not None:
        feature_loss = additional_terms.get("feature_loss", feature_loss)
        feature_weight = additional_terms.get("feature_weight", feature_weight)
        adversarial_loss = additional_terms.get("adversarial_loss", adversarial_loss)
        adversarial_weight = additional_terms.get("adversarial_weight", adversarial_weight)
        codebook = additional_terms.get("codebook", codebook)
        codebook_weight = additional_terms.get("codebook_weight", codebook_weight)
        codebook_beta = additional_terms.get("codebook_beta", codebook_beta)

    recon = reconstruction_loss(output.reconstruction, target, loss_type=loss_type)
    kl = kl_divergence(output.mu, output.logvar)
    total = recon + float(beta) * kl
    losses = {"loss": total, "reconstruction_loss": recon, "kl_loss": kl}

    if feature_loss is not None:
        losses["feature_loss"] = feature_loss
        total = total + float(feature_weight) * feature_loss
    if adversarial_loss is not None:
        losses["adversarial_loss"] = adversarial_loss
        total = total + float(adversarial_weight) * adversarial_loss
    if codebook is not None and float(codebook_weight) != 0.0:
        codebook_losses = codebook_clustering_loss(output.z, codebook, beta=codebook_beta)
        losses.update(codebook_losses)
        total = total + float(codebook_weight) * codebook_losses["codebook_loss"]

    losses["loss"] = total
    return losses


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
    "codebook_clustering_loss",
    "code_book_loss",
    "feature_matching_loss",
    "VGGFeatureLoss",
    "PatchDiscriminator",
    "generator_gan_loss",
    "discriminator_gan_loss",
    "vae_loss",
]
