"""Fourier Neural Operator conditioning encoder.

Drop-in replacement for the RRDB CNN encoder in `lr_encoder_model.py`. It obeys the same
two-level contract the diffusion pipeline depends on:

    forward(x)                -> [B, out_channels,     H_hr, W_hr]   (the L1 pretraining target)
    conditioning_features(x)  -> [B, feature_channels, H_hr, W_hr]   (what the U-Net consumes)

`feature_channels` must equal the U-Net's `init_dim`, which `DiffusionModel.__init__` sets to
`self.channels` when `enc_output=True` and leaves as `None` (-> 64) when `enc_output=False`.
The production path (`train_srdiff.py` defaults `enc_output=False`) uses the 64-channel
intermediate, mirroring the paper's use of the RRDB tensor before its last convolution.
"""

from typing import Any, Optional, Sequence, Tuple  # Type hints; Optional marks args that may be None, Sequence covers list/tuple mode counts

import torch  # Core PyTorch package, provides the Tensor type and FFT routines used below
import torch.nn as nn  # Neural-network building blocks (Module, Conv2d, Parameter, ...)
import torch.nn.functional as F  # Stateless functional ops (interpolate, gelu, pad) that carry no learnable weights

__all__ = [
    "SpectralConv2d",
    "FNOEncoder",
    "fno_encoder",
    "PHYSICSNEMO_AVAILABLE",
]

# Try to import NVIDIA PhysicsNeMo's FNO. It is Linux/CUDA-first and will usually be absent on a
# Windows dev box, so the import is guarded and the result recorded as a module-level flag rather
# than allowed to raise. `FNOEncoder` consults this flag when `backend='auto'`.
try:
    from physicsnemo.models.fno import FNO as _PhysicsNeMoFNO  # The packaged FNO; aliased with a leading underscore to keep it out of `__all__`

    PHYSICSNEMO_AVAILABLE = True  # Set when the import succeeded, so callers can branch without repeating the try/except
except ImportError:  # Raised when the physicsnemo package is not installed in the active environment
    _PhysicsNeMoFNO = None  # Bind the name to None so later `is None` checks are well-defined rather than NameError
    PHYSICSNEMO_AVAILABLE = False  # Records that only the builtin backend can be used in this environment


# --- ALGORITHM: Spectral convolution, the core operator of a Fourier Neural Operator (Li et al., 2021) ---
# CONCEPT: An ordinary convolution mixes each pixel with a small fixed neighbourhood, so information
# travels only a few pixels per layer. An FNO instead performs convolution in the *frequency* domain.
# The convolution theorem says convolution in space equals pointwise multiplication in Fourier space,
# so instead of learning a spatial kernel, the layer learns one complex weight matrix per Fourier mode
# and multiplies the transformed input by it. Because every Fourier mode is global (a sine wave spanning
# the whole domain), a single layer has a *global* receptive field.
# The second key idea is truncation: only the lowest `modes` frequencies are kept and learned; everything
# above is discarded. This bounds the parameter count independently of grid size, and — crucially for
# this study — it makes the learned operator *resolution-independent*, because the same set of modes can
# be evaluated on any grid. That is the property H3 in the study plan is testing.
# IMPLEMENTATION: `weights1`/`weights2` hold the learned per-mode complex matrices for the positive and
# negative vertical-frequency bands respectively (rfft2 keeps only non-negative horizontal frequencies,
# so the horizontal axis needs one band but the vertical axis needs two). `out_size` optionally writes
# the result into a *larger* spectrum than the input, which zero-pads in frequency space and therefore
# performs a band-limited interpolation onto a finer grid — spectral upsampling.
class SpectralConv2d(nn.Module):
    """2D spectral convolution with optional spectral (band-limited) upsampling."""

    def __init__(
        self,
        in_channels: int,  # Number of feature channels entering the layer
        out_channels: int,  # Number of feature channels the layer produces
        modes1: int,  # How many Fourier modes to keep along the first spatial axis (height)
        modes2: int,  # How many Fourier modes to keep along the second spatial axis (width)
    ) -> None:
        super().__init__()  # Run nn.Module's constructor so parameter/buffer registration machinery is set up
        self.in_channels = in_channels  # Stored for the einsum contraction and for zero-tensor allocation in forward
        self.out_channels = out_channels  # Stored so forward knows the channel width of the output spectrum
        self.modes1 = modes1  # Retained truncation limit for the height axis
        self.modes2 = modes2  # Retained truncation limit for the width axis

        # Scale the random initialisation by 1/(in*out). Without this the summed contribution of many
        # input channels would make activations blow up; this is the standard FNO initialisation.
        scale = 1.0 / (in_channels * out_channels)

        # weights1 covers the *positive* vertical-frequency band (rows 0..modes1). torch.cfloat is the
        # complex64 dtype, so each entry is a complex number with its own learnable real and imaginary part.
        # nn.Parameter registers the tensor with the module so the optimiser will update it.
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        # weights2 covers the *negative* vertical-frequency band (the last `modes1` rows of the spectrum).
        # A real-valued 2D signal has a conjugate-symmetric spectrum; rfft2 exploits this to drop redundant
        # horizontal frequencies, but the vertical axis still carries both signs, hence a second weight block.
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _complex_matmul(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Contract the channel dimension of a batched spectrum against per-mode weight matrices."""
        # torch.einsum performs a named-index tensor contraction. "bixy,ioxy->boxy" reads: for every
        # batch b, output channel o and mode (x, y), sum over input channel i the product
        # inputs[b,i,x,y] * weights[i,o,x,y]. It is a per-mode matrix-vector product across channels,
        # applied independently at each Fourier mode; complex arithmetic is handled natively for cfloat.
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Apply the spectral convolution, optionally resampling onto a grid of size `out_size`."""
        batch, _, height, width = x.shape  # Unpack the input shape; the channel count is recovered from self.in_channels

        # torch.fft.rfft2 computes the 2D discrete Fourier transform of a *real* input. Because the
        # spectrum of a real signal is conjugate-symmetric, it returns only the non-redundant half along
        # the last axis, giving shape (B, C, height, width//2 + 1) with complex entries. Each entry is
        # the amplitude and phase of one spatial frequency present in the image.
        x_ft = torch.fft.rfft2(x)

        # Default to preserving the input grid; when `out_size` is supplied we will instead write into a
        # larger spectrum, which after the inverse transform yields a finer-resolution field.
        out_height, out_width = out_size if out_size is not None else (height, width)

        # Allocate the destination spectrum filled with zeros. Every mode we do not explicitly write stays
        # zero — that is exactly the low-pass truncation the FNO relies on, and when out_size is larger
        # than the input it is also the zero-padding that performs band-limited interpolation.
        out_ft = torch.zeros(
            batch,  # Same batch size as the input
            self.out_channels,  # Channel width of this layer's output
            out_height,  # Full (non-halved) height, because rfft2 only halves the last axis
            out_width // 2 + 1,  # Halved width plus the DC/Nyquist column, matching rfft2's output layout
            dtype=torch.cfloat,  # Complex dtype so it can hold Fourier coefficients
            device=x.device,  # Keep the allocation on the same device as the input to avoid an implicit transfer
        )

        # Clamp the truncation limits so we never index past the end of either the source or destination
        # spectrum. This is what makes the layer safe to evaluate on a grid coarser than it was trained on:
        # if the new grid supports fewer modes than `self.modes1`, we simply use the modes that exist.
        mode_h = min(self.modes1, height // 2, out_height // 2)  # Usable vertical modes, limited by both grids' Nyquist
        mode_w = min(self.modes2, width // 2 + 1, out_width // 2 + 1)  # Usable horizontal modes, limited by both grids

        # Positive vertical-frequency band: take the top-left corner of the source spectrum (lowest
        # frequencies in both axes) and multiply it by the corresponding learned weights.
        out_ft[:, :, :mode_h, :mode_w] = self._complex_matmul(
            x_ft[:, :, :mode_h, :mode_w],  # Lowest `mode_h` vertical and `mode_w` horizontal frequencies of the input
            self.weights1[:, :, :mode_h, :mode_w],  # Matching slice of the learned weights, in case clamping shrank the band
        )

        # Negative vertical-frequency band: `-mode_h:` selects the *last* rows of the spectrum, which in
        # FFT layout hold the negative vertical frequencies. Handling them separately (rather than letting
        # them stay zero) is what keeps the operator able to represent asymmetric vertical structure.
        out_ft[:, :, -mode_h:, :mode_w] = self._complex_matmul(
            x_ft[:, :, -mode_h:, :mode_w],  # Highest-magnitude negative vertical frequencies of the input
            self.weights2[:, :, :mode_h, :mode_w],  # The second learned weight block, sliced to the clamped band
        )

        # torch.fft.irfft2 inverts the transform back to a real-valued spatial field. `s=` states the
        # desired output spatial size explicitly, which is required because the halved last axis alone
        # is ambiguous about whether the original width was even or odd — and it is how the enlarged
        # spectrum is realised as a higher-resolution image.
        return torch.fft.irfft2(out_ft, s=(out_height, out_width))


class _FNOBlock(nn.Module):
    """One FNO layer: a global spectral path summed with a local pointwise path, then a nonlinearity."""

    def __init__(self, channels: int, modes1: int, modes2: int) -> None:
        super().__init__()  # Initialise nn.Module bookkeeping before assigning submodules
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)  # Global path: mixes information across the whole domain via Fourier modes
        # Local path: a 1x1 convolution acts independently at each pixel, mixing only across channels.
        # It carries the high-frequency content the spectral path discards by truncation, so the two
        # paths are complementary — this residual pairing is standard in the FNO architecture.
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        spectral_out = self.spectral(x, out_size=out_size)  # Global Fourier path, optionally resampling to `out_size`

        if out_size is not None:
            # When the spectral path changed resolution, the pointwise path's output no longer matches
            # its spatial shape, so resample the input first. bilinear interpolation is used because the
            # pointwise branch is a local correction and does not need band-limited (spectral) accuracy.
            # align_corners=False matches PyTorch's recommended convention for resizing feature maps.
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

        local_out = self.pointwise(x)  # Local per-pixel channel mixing on the (possibly resampled) input

        # Sum the two paths and apply GELU. GELU is a smooth activation that gates each value by the
        # probability it would be kept under a standard normal, giving softer gradients than ReLU; smooth
        # activations are conventional in FNOs because the target fields are themselves smooth.
        return F.gelu(spectral_out + local_out)


class _BuiltinFNOBody(nn.Module):
    """Self-contained FNO used when PhysicsNeMo is unavailable, or when spectral upsampling is required.

    Mirrors the standard lift -> spectral blocks -> project structure, and matches the role of
    PhysicsNeMo's `FNO` so the two backends are interchangeable behind `FNOEncoder`.
    """

    def __init__(
        self,
        in_channels: int,  # Channels of the incoming physical field(s)
        out_channels: int,  # Channels the body should emit (the encoder's `feature_channels`)
        latent_channels: int,  # Width of the internal representation the spectral blocks operate on
        num_layers: int,  # How many spectral blocks to stack
        modes: Tuple[int, int],  # Per-axis Fourier mode truncation limits
        padding: int,  # How many cells of zero-padding to add before the FFT (see forward)
    ) -> None:
        super().__init__()  # Standard nn.Module initialisation
        self.padding = padding  # Retained so forward knows how much padding to add and later strip

        # Lifting layer: a 1x1 convolution that widens the input from its small physical channel count
        # (e.g. 1 temperature field) to the much wider latent width the spectral blocks operate in.
        self.lift = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

        # nn.ModuleList holds submodules in a list while still registering their parameters with the
        # parent module; a plain Python list would leave them invisible to the optimiser.
        self.blocks = nn.ModuleList([_FNOBlock(latent_channels, modes[0], modes[1]) for _ in range(num_layers)])

        # Projection head: two 1x1 convolutions with a nonlinearity between them, collapsing the latent
        # width down to the requested output width. The hidden layer gives the projection some capacity
        # rather than making it a bare linear map.
        self.project = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),  # First pointwise layer, keeps the latent width
            nn.GELU(),  # Smooth nonlinearity between the two projection convolutions
            nn.Conv2d(latent_channels, out_channels, kernel_size=1),  # Second pointwise layer, emits the requested channel count
        )

    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self.lift(x)  # Widen the physical fields into the latent representation

        # The FFT treats the domain as periodic, wrapping the right edge around to the left. Melt-pool
        # cross-sections and wall-bounded convection are *not* periodic, so that wrap-around injects a
        # spurious discontinuity at the boundary (Gibbs ringing). Padding with zeros pushes the artificial
        # seam into a margin that is discarded afterwards, keeping it out of the physical domain.
        if self.padding > 0:
            # F.pad's pad tuple is ordered last-axis-first: (left, right, top, bottom). Padding only the
            # right and bottom keeps the original content anchored at index 0, which makes the later crop
            # a simple slice from the origin.
            x = F.pad(x, (0, self.padding, 0, self.padding), mode="constant", value=0.0)

        # Work out the padded target size for the final block when spectral upsampling is requested, so
        # the resize happens *inside* the padded domain and the crop below still removes the right amount.
        padded_out_size = None if out_size is None else (out_size[0] + self.padding, out_size[1] + self.padding)

        for index, block in enumerate(self.blocks):  # enumerate yields (position, block) so we can detect the last one
            is_last = index == len(self.blocks) - 1  # Only the final block performs the resolution change, if any
            # Pass the enlarged target size to the last block alone; earlier blocks keep the input grid,
            # which keeps their cost low and concentrates the upsampling in one well-defined place.
            x = block(x, out_size=padded_out_size if is_last else None)

        if self.padding > 0:
            # Slice away the padding margin. Negative-free slicing from the origin works because the
            # padding was appended to the right and bottom only.
            x = x[..., : x.shape[-2] - self.padding, : x.shape[-1] - self.padding]

        return self.project(x)  # Collapse the latent width down to the requested output channel count


class FNOEncoder(nn.Module):
    """FNO conditioning encoder exposing the same interface as `RRDBNet`.

    `upsample_mode` selects one of the three variants in the study plan:

      'pre'      (FNO-A) bicubic-upsample the input to HR, then run the FNO entirely at HR.
      'spectral' (FNO-B) run the FNO at LR and upsample by zero-padding the spectrum. Requires the
                         builtin backend, since PhysicsNeMo does not expose per-layer output sizing.
      'conv'     (FNO-C) run the FNO at LR, then use the *same* nearest-interp + conv upsampling
                         blocks RRDB uses, isolating the trunk as the only difference.
    """

    def __init__(
        self,
        upscale_factor: int,  # Ratio between HR and LR grid size, e.g. 4 for 20x20 -> 80x80
        in_channels: int = 1,  # Physical field channels entering the encoder
        out_channels: int = 1,  # Physical field channels the L1 pretraining objective compares against
        feature_channels: int = 64,  # Width of the conditioning tensor; must equal the U-Net's init_dim
        latent_channels: int = 64,  # Internal width of the spectral blocks
        num_fno_layers: int = 4,  # Depth of the spectral stack
        num_fno_modes: Sequence[int] = (16, 16),  # Per-axis Fourier truncation; see the Nyquist note in the factory
        padding: int = 8,  # Non-periodic padding margin (0 disables it, appropriate for periodic domains)
        upsample_mode: str = "pre",  # Which of the three variants above to build
        backend: str = "auto",  # 'auto' | 'physicsnemo' | 'builtin'
        **body_kwargs: Any,  # Extra keyword arguments forwarded to the PhysicsNeMo FNO constructor
    ) -> None:
        super().__init__()  # Initialise nn.Module before registering any submodules

        if upsample_mode not in ("pre", "spectral", "conv"):  # Guard against silent typos in a config file
            raise ValueError(f"upsample_mode must be 'pre', 'spectral' or 'conv', got {upsample_mode!r}")

        self.upscale_factor = upscale_factor  # Needed by forward to size the interpolation / spectral target
        self.in_channels = in_channels  # Recorded for checkpoint metadata and debugging
        self.out_channels = out_channels  # Recorded so the head's width can be verified on load
        self.feature_channels = feature_channels  # The channel count `conditioning_features` promises to return
        self.upsample_mode = upsample_mode  # Branch selector consulted in conditioning_features

        # Normalise the mode specification to a 2-tuple so downstream code never has to handle both an
        # int and a sequence. An int means "same truncation on both axes".
        modes = (int(num_fno_modes), int(num_fno_modes)) if isinstance(num_fno_modes, int) else tuple(int(m) for m in num_fno_modes)

        # Resolve which FNO implementation to use. 'auto' prefers PhysicsNeMo (the study's stated target)
        # and silently falls back to the builtin one so the code remains testable on machines without it.
        resolved_backend = backend
        if resolved_backend == "auto":
            resolved_backend = "physicsnemo" if PHYSICSNEMO_AVAILABLE else "builtin"

        # Spectral upsampling needs to resize inside the last spectral layer, which PhysicsNeMo's packaged
        # FNO does not expose. Rather than silently producing a different architecture than requested,
        # fail loudly so the experiment record stays honest about which variant was trained.
        if self.upsample_mode == "spectral" and resolved_backend != "builtin":
            raise ValueError("upsample_mode='spectral' requires backend='builtin'; PhysicsNeMo's FNO does not expose per-layer output sizing")

        if resolved_backend == "physicsnemo" and not PHYSICSNEMO_AVAILABLE:  # Explicit request for a backend that is not installed
            raise ImportError("backend='physicsnemo' requested but physicsnemo is not installed; install nvidia-physicsnemo or use backend='builtin'")

        self.backend = resolved_backend  # Persisted so it can be written into the checkpoint and logged to W&B

        if resolved_backend == "physicsnemo":
            # PhysicsNeMo's FNO already implements lift -> spectral blocks -> decoder. Asking it for
            # `feature_channels` outputs makes its decoder output *our* conditioning tensor, so the
            # structural parallel with RRDB (trunk -> 64-ch features -> 1x1 head) is preserved exactly.
            self.body = _PhysicsNeMoFNO(
                in_channels=in_channels,  # Physical fields in
                out_channels=feature_channels,  # Emit the conditioning width, not the physical width
                dimension=2,  # 2D spatial problem (the x-z cross-section)
                latent_channels=latent_channels,  # Internal spectral width
                num_fno_layers=num_fno_layers,  # Depth of the spectral stack
                num_fno_modes=list(modes),  # PhysicsNeMo accepts a per-axis list of mode counts
                padding=padding,  # Its own non-periodic padding handling, equivalent in intent to the builtin one
                **body_kwargs,  # Any further PhysicsNeMo-specific options passed through from config
            )
        else:
            self.body = _BuiltinFNOBody(
                in_channels=in_channels,  # Physical fields in
                out_channels=feature_channels,  # Emit the conditioning width
                latent_channels=latent_channels,  # Internal spectral width
                num_layers=num_fno_layers,  # Depth of the spectral stack
                modes=modes,  # Per-axis truncation limits
                padding=padding,  # Zero-pad margin to suppress FFT wrap-around artefacts
            )

        # In 'conv' mode the body runs at LR, so we need explicit upsampling blocks afterwards. These
        # deliberately replicate RRDBNet's upsampling stack (nearest interpolation, 3x3 conv, LeakyReLU
        # with negative slope 0.2) so that FNO-C differs from RRDB *only* in its trunk.
        self.upsampling = nn.ModuleList()
        if self.upsample_mode == "conv":
            num_doublings = self._log2_exact(upscale_factor)  # How many 2x steps are needed; raises if the factor is not a power of two
            for _ in range(num_doublings):  # One nearest-interp + conv + activation stage per doubling
                self.upsampling.append(
                    nn.Sequential(
                        nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),  # 3x3 conv smooths the blocky nearest-neighbour output
                        nn.LeakyReLU(0.2, inplace=True),  # Matches RRDBNet's activation exactly, including the 0.2 negative slope
                    )
                )
            # RRDBNet applies one further conv+activation after all upsampling (its `conv3`), which is the
            # layer whose output the paper extracts as conditioning. Replicated here for the same reason.
            self.upsampling.append(
                nn.Sequential(
                    nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),  # Final refinement conv at HR
                    nn.LeakyReLU(0.2, inplace=True),  # Final activation before the conditioning tensor is read off
                )
            )

        # The output head, mirroring RRDBNet's `conv4`: a single convolution mapping the 64-channel
        # conditioning features down to the physical field channels used by the L1 pretraining loss.
        # Keeping it to one layer means `conditioning_features` really is "everything but the last conv".
        self.head = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _log2_exact(factor: int) -> int:
        """Return log2(factor), rejecting anything that is not an exact power of two."""
        exponent = 0  # Counts how many times `factor` can be halved
        value = int(factor)  # Work on a local integer copy so the caller's argument is untouched
        while value > 1:  # Keep halving until we reach 1
            if value % 2 != 0:  # An odd value above 1 means the original was not a power of two
                raise ValueError(f"upsample_mode='conv' requires a power-of-two upscale_factor, got {factor}")
            value //= 2  # Integer division halves the value for the next iteration
            exponent += 1  # Record that one doubling stage is needed
        return exponent  # Number of 2x upsampling stages required

    def conditioning_features(self, x: torch.Tensor, factor: Optional[int] = None) -> torch.Tensor:
        """Return the HR feature map the diffusion U-Net consumes as `x_e`.

        This is the FNO analogue of reading RRDBNet's `conv3` output, i.e. everything except the
        final channel-collapsing convolution.

        `factor` overrides the upscale ratio this encoder was constructed with. It exists to match
        `RRDBNet.conditioning_features`' signature, but for the 'pre' and 'spectral' variants it is
        also the zero-shot upscale-factor probe (study plan H3): an FNO trained at 4x can be asked
        for 8x here without retraining, because its learned Fourier modes are grid-independent.
        The 'conv' variant cannot honour an override, since its upsampling stack has a fixed depth.
        """
        # Default to the ratio baked in at construction; an explicit argument wins so callers can
        # evaluate the same weights at a different resolution.
        if factor is None:
            factor = self.upscale_factor

        if self.upsample_mode == "conv" and factor != self.upscale_factor:
            # FNO-C's upsampling is a fixed stack of 2x convolution stages, so it physically cannot
            # produce a different ratio. Fail loudly rather than silently returning the wrong size.
            raise ValueError(f"upsample_mode='conv' cannot override factor ({factor} != {self.upscale_factor}); its upsampling depth is fixed at construction")

        target_height = x.shape[-2] * factor  # HR height implied by the LR input and the (possibly overridden) factor
        target_width = x.shape[-1] * factor  # HR width, computed the same way

        if self.upsample_mode == "pre":
            # FNO-A: resample onto the HR grid first, then let the operator work entirely at HR.
            # bicubic interpolation fits a cubic polynomial through a 4x4 neighbourhood, giving a smoother
            # starting field than bilinear; align_corners=False follows PyTorch's standard resize convention.
            x = F.interpolate(x, size=(target_height, target_width), mode="bicubic", align_corners=False)
            return self.body(x)  # Body already emits `feature_channels`, so no further projection is needed

        if self.upsample_mode == "spectral":
            # FNO-B: stay on the LR grid and let the final spectral layer write into an enlarged spectrum,
            # which is a band-limited interpolation onto the HR grid. This is the most operator-native
            # route and the one that makes the discretisation-invariance claim cleanest.
            return self.body(x, out_size=(target_height, target_width))

        # FNO-C: run the operator at LR, then upsample with RRDB's own convolutional stack.
        features = self.body(x)  # Spectral trunk output, still at LR resolution
        for index, stage in enumerate(self.upsampling):  # Walk the doubling stages, then the final refinement conv
            if index < len(self.upsampling) - 1:  # Every stage except the last is preceded by a 2x nearest-neighbour resize
                # 'nearest' simply repeats each pixel into a 2x2 block. RRDBNet uses exactly this before
                # each upsampling conv, so replicating it keeps FNO-C's upsampling path bit-for-bit
                # comparable to the baseline's.
                features = F.interpolate(features, scale_factor=2, mode="nearest")
            features = stage(features)  # Apply the conv + LeakyReLU for this stage
        return features  # HR feature map of width `feature_channels`

    def forward(self, x: torch.Tensor, factor: Optional[int] = None) -> torch.Tensor:
        """Map the LR field to an HR field — the target of the L1 encoder-pretraining objective."""
        features = self.conditioning_features(x, factor=factor)  # Shared trunk, identical to what the U-Net will later consume
        return self.head(features)  # Collapse conditioning width down to the physical output channels


def fno_encoder(upscale_factor: int, **kwargs: Any) -> FNOEncoder:
    """Construct an `FNOEncoder`, mirroring the `rrdbnet_encoder(...)` factory signature.

    Note on mode counts: the Nyquist limit caps the useful truncation at half the grid size along each
    axis. At 4x upscaling the LR grid is 20x20, so `num_fno_modes` above 10 is a no-op for the 'spectral'
    and 'conv' variants; 'pre' works on the 80x80 HR grid where the ceiling is 40.
    """
    return FNOEncoder(upscale_factor=upscale_factor, **kwargs)  # Forward every argument through to the class constructor
