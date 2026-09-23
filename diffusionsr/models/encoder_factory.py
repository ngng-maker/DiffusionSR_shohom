"""Single construction point for conditioning encoders.

Both `runners/train_rrdn_encoder.pretrain_encoder` (stage 1) and
`runners/train_diffusion.DiffusionModel.initialize_encoder` (stage 2) must build *identical*
architectures, or the stage-2 `load_state_dict` will fail or, worse, silently load into a
mismatched module. Routing both through this factory keeps them in lockstep.

Selected by the config key `encoder_type` (default 'rrdb', which preserves existing behaviour
for every config already in the repo). FNO-specific options come from the optional
`encoder_kwargs` mapping.
"""

from typing import Any, Mapping, Optional  # Mapping is the read-only dict type used for the kwargs blob

import torch.nn as nn  # Only needed for the return type annotation

from diffusionsr.models.lr_encoder_model import rrdbnet_encoder  # The RRDB CNN baseline encoder factory
from diffusionsr.models.fno_encoder_model import fno_encoder  # The FNO encoder factory added for this study

__all__ = ["build_encoder", "ENCODER_TYPES"]

ENCODER_TYPES = ("rrdb", "fno")  # Valid `encoder_type` values; used for validation and error messages


def build_encoder(
    encoder_type: str,  # Which architecture to construct, one of ENCODER_TYPES
    upscale_factor: int,  # HR/LR grid ratio, taken from `dataset.factor`
    in_channels: int,  # Input field channels, normally n_steps * num_fields
    out_channels: int,  # Output field channels, normally out_steps * num_fields
    encoder_kwargs: Optional[Mapping[str, Any]] = None,  # Architecture-specific extras from config
) -> nn.Module:
    """Construct a conditioning encoder exposing `forward` and `conditioning_features`."""
    # Normalise the selector so 'RRDB', ' rrdb ' and 'rrdb' all resolve identically; config files are
    # hand-edited and this removes a whole class of silent misconfiguration.
    encoder_type = (encoder_type or "rrdb").strip().lower()

    # dict(...) both copies the mapping (so we never mutate the caller's config) and converts None to
    # an empty dict, letting the code below use it unconditionally.
    kwargs = dict(encoder_kwargs or {})

    if encoder_type == "rrdb":
        # num_blocks=8 is the value hardcoded at both original call sites, preserved as the default so
        # existing configs reproduce byte-identical architectures. `setdefault` only fills it in when
        # the config did not already specify a value.
        kwargs.setdefault("num_blocks", 8)
        return rrdbnet_encoder(
            upscale_factor=upscale_factor,  # Number of 2x stages the RRDB upsampling stack will use
            in_channels=in_channels,  # Physical fields entering the encoder
            out_channels=out_channels,  # Physical fields the L1 pretraining loss compares against
            **kwargs,  # num_blocks plus any other RRDBNet options supplied by config
        )

    if encoder_type == "fno":
        return fno_encoder(
            upscale_factor=upscale_factor,  # Used to size the interpolation or spectral upsampling target
            in_channels=in_channels,  # Physical fields entering the encoder
            out_channels=out_channels,  # Physical fields the L1 pretraining loss compares against
            **kwargs,  # feature_channels, latent_channels, num_fno_modes, upsample_mode, backend, ...
        )

    # Anything else is a typo or an unimplemented variant; fail immediately rather than defaulting,
    # so a misspelled config never silently trains the wrong architecture.
    raise ValueError(f"Unknown encoder_type {encoder_type!r}; expected one of {ENCODER_TYPES}")
