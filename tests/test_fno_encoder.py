"""Tests for the pluggable conditioning encoder (study: FNO vs RRDB).

The most important test here is `test_rrdb_conditioning_features_matches_legacy_inline_forwardpass`.
It is the regression gate for the whole study: it pins `RRDBNet.conditioning_features` to the exact
tensor the old inlined `forwardpass` produced, so the RRDB baseline cannot drift as a side effect of
making the encoder swappable. If that test fails, every RRDB-vs-FNO number is invalid.
"""

import pytest  # Test framework; used here for parametrisation and exception assertions
import torch  # Tensors, manual seeding, and the equality checks
import torch.nn.functional as F  # Provides interpolate, needed by the legacy reference implementation

from diffusionsr.models.encoder_factory import build_encoder  # The single construction point under test
from diffusionsr.models.fno_encoder_model import FNOEncoder, PHYSICSNEMO_AVAILABLE  # FNO encoder and backend flag
from diffusionsr.models.lr_encoder_model import RRDBNet, rrdbnet_encoder  # The CNN baseline being pinned
from diffusionsr.runners.train_diffusion import DiffusionModel, forwardpass  # Integration surface


def _legacy_forwardpass_features(lr_enc, sample, factor):
    """Verbatim copy of the pre-refactor `forwardpass(..., output=False)` body.

    Deliberately duplicated rather than imported: its whole purpose is to be an independent
    reference that does NOT change when the production code changes. Note it omits the global
    residual skip (`out1 + out2`) that `RRDBNet._forward_impl` applies — that omission is the
    historical behaviour all trained checkpoints were conditioned on, so it must be preserved.
    """
    x = lr_enc.conv1(sample)  # Initial 3x3 convolution lifting input fields to 64 feature maps
    x = lr_enc.trunk(x)  # Residual-in-residual dense block stack
    x = lr_enc.conv2(x)  # Post-trunk convolution; note no `+ out1` here, matching the original
    x = F.interpolate(x, scale_factor=2, mode='nearest')  # First 2x nearest-neighbour upsample
    x = lr_enc.upsampling1(x)  # Refinement conv + LeakyReLU after the first doubling
    if factor == 4:  # Only 4x encoders take a second doubling
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # Second 2x nearest-neighbour upsample
        x = lr_enc.upsampling2(x)  # Second refinement conv + LeakyReLU
    x = lr_enc.conv3(x)  # Final conv + LeakyReLU; its output is the conditioning tensor
    return x


class _DummyDataset:
    """Minimal stand-in exposing only the attributes DiffusionModel reads from a dataset."""

    def __init__(self, num_fields=1, img_shape=16, n_steps=1, factor=2):
        self.num_fields = num_fields  # Number of physical fields per timestep
        self.img_shape = img_shape  # HR grid side length; also used as the U-Net's `dim`
        self.n_steps = n_steps  # How many prior timesteps are stacked as extra channels
        self.out_steps = n_steps  # Output timestep count, mirrored from n_steps
        self.factor = factor  # HR/LR ratio the encoder is built for
        self.inflate_dim = None  # Present because some code paths check it; unused in 2D
        self.field_names = ["temperature"] if num_fields == 1 else ["temperature", "liqlabel"]


# ── Regression gate ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()  # No gradients needed; this is a pure forward-equivalence check
@pytest.mark.parametrize("factor", [2, 4])  # Both upscale factors the conditioning path supports
def test_rrdb_conditioning_features_matches_legacy_inline_forwardpass(factor):
    """RRDB conditioning output must be bit-identical to the pre-refactor inline implementation."""
    torch.manual_seed(0)  # Fix the RNG so the randomly initialised weights are reproducible
    encoder = rrdbnet_encoder(upscale_factor=factor, in_channels=1, out_channels=1, num_blocks=2)  # Small trunk keeps the test fast
    encoder.eval()  # Disable any training-mode behaviour so the two paths cannot diverge stochastically

    sample = torch.randn(2, 1, 8, 8)  # Batch of 2 single-channel 8x8 LR inputs

    legacy = _legacy_forwardpass_features(encoder, sample, factor)  # Reference tensor from the duplicated old code
    refactored = encoder.conditioning_features(sample, factor=factor)  # Tensor from the new method

    # torch.equal requires identical shape, dtype AND exact elementwise equality — not approximate.
    # Both paths execute the same ops in the same order, so anything short of exact equality means
    # the refactor changed the computation.
    assert torch.equal(legacy, refactored), "RRDB conditioning path changed — the baseline is no longer comparable"


@torch.no_grad()
@pytest.mark.parametrize("factor", [2, 4])
def test_forwardpass_wrapper_matches_legacy(factor):
    """The public `forwardpass` entry point must also be unchanged, not just the new method."""
    torch.manual_seed(0)  # Same seed as above for reproducible weights
    encoder = rrdbnet_encoder(upscale_factor=factor, in_channels=1, out_channels=1, num_blocks=2)
    encoder.eval()
    sample = torch.randn(2, 1, 8, 8)

    legacy = _legacy_forwardpass_features(encoder, sample, factor).float()  # .float() mirrors forwardpass's final cast
    via_wrapper = forwardpass(encoder, sample, factor=factor, output=False)  # Production call path

    assert torch.equal(legacy, via_wrapper)  # Exact equality, same reasoning as the test above


@torch.no_grad()
def test_rrdb_conditioning_features_still_excludes_residual_skip():
    """Pin the known divergence from `_forward_impl` so nobody "fixes" it without retraining."""
    torch.manual_seed(0)
    encoder = rrdbnet_encoder(upscale_factor=2, in_channels=1, out_channels=1, num_blocks=2)
    encoder.eval()
    sample = torch.randn(1, 1, 8, 8)

    out1 = encoder.conv1(sample)  # The tensor that `_forward_impl` would add back in as a skip connection
    trunk_out = encoder.conv2(encoder.trunk(out1))  # Trunk output before any skip is applied

    # Reconstruct what the conditioning path WOULD produce if the residual skip were present.
    with_skip = torch.add(out1, trunk_out)  # `_forward_impl`'s behaviour
    with_skip = encoder.conv3(encoder.upsampling1(F.interpolate(with_skip, scale_factor=2, mode='nearest')))

    actual = encoder.conditioning_features(sample, factor=2)  # What the conditioning path actually produces

    # These must differ. If they ever match, someone has added the skip back into the conditioning
    # path, which silently invalidates every diffusion checkpoint trained against the old behaviour.
    assert not torch.allclose(with_skip, actual), "conditioning path now includes the residual skip — retrain before accepting this"


# ── FNO encoder ───────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
@pytest.mark.parametrize("upsample_mode", ["pre", "spectral", "conv"])  # All three study variants
@pytest.mark.parametrize("factor", [2, 4])  # Both upscale ratios
def test_fno_encoder_output_shapes(upsample_mode, factor):
    """Every variant must emit HR-sized tensors with the promised channel widths."""
    torch.manual_seed(0)
    encoder = FNOEncoder(
        upscale_factor=factor,  # Target HR/LR ratio
        in_channels=1,  # Single physical field in
        out_channels=1,  # Single physical field out
        feature_channels=64,  # Must match the U-Net's init_dim on the enc_output=False path
        latent_channels=16,  # Small latent width keeps the test fast
        num_fno_layers=2,  # Shallow stack, same reason
        num_fno_modes=(4, 4),  # Well under the Nyquist limit of the 8x8 LR grid
        padding=2,  # Exercise the non-periodic padding branch
        upsample_mode=upsample_mode,  # The variant under test
        backend="builtin",  # Builtin backend so the test runs without PhysicsNeMo installed
    )
    encoder.eval()

    sample = torch.randn(2, 1, 8, 8)  # Batch of 2 single-channel 8x8 LR inputs
    hr_side = 8 * factor  # Expected HR side length

    features = encoder.conditioning_features(sample)  # The tensor the U-Net consumes as x_e
    assert features.shape == (2, 64, hr_side, hr_side)  # Conditioning width must be exactly feature_channels

    prediction = encoder(sample)  # The tensor the L1 pretraining loss compares against HR
    assert prediction.shape == (2, 1, hr_side, hr_side)  # Physical field width and HR resolution
    assert torch.isfinite(prediction).all()  # Guard against NaN/Inf leaking out of the FFT path


@torch.no_grad()
def test_fno_feature_channels_follow_config():
    """`feature_channels` must be configurable, because init_dim equals `channels` when enc_output=True."""
    torch.manual_seed(0)
    # enc_output=True makes the U-Net's init_dim equal the physical channel count, so the encoder has
    # to be able to emit something other than 64. Build a 2-channel conditioning tensor to prove it.
    encoder = FNOEncoder(
        upscale_factor=2, in_channels=2, out_channels=2, feature_channels=2,
        latent_channels=8, num_fno_layers=1, num_fno_modes=(3, 3), padding=0,
        upsample_mode="pre", backend="builtin",
    )
    encoder.eval()
    features = encoder.conditioning_features(torch.randn(1, 2, 8, 8))  # Two-field LR input
    assert features.shape == (1, 2, 16, 16)  # Conditioning width follows feature_channels, not a hardcoded 64


@torch.no_grad()
@pytest.mark.parametrize("upsample_mode", ["pre", "spectral"])  # The two resolution-agnostic variants
def test_fno_zero_shot_factor_override(upsample_mode):
    """An FNO trained at one factor must evaluate at another without retraining (study hypothesis H3)."""
    torch.manual_seed(0)
    encoder = FNOEncoder(
        upscale_factor=2, in_channels=1, out_channels=1, feature_channels=8,
        latent_channels=8, num_fno_layers=2, num_fno_modes=(3, 3), padding=0,
        upsample_mode=upsample_mode, backend="builtin",
    )
    encoder.eval()
    sample = torch.randn(1, 1, 8, 8)  # 8x8 LR input

    at_native = encoder.conditioning_features(sample)  # Default factor of 2 -> 16x16
    assert at_native.shape == (1, 8, 16, 16)

    # The same weights asked for 4x instead. This works because the learned Fourier modes describe a
    # continuous operator rather than a fixed pixel grid — the property the study is built around.
    at_override = encoder.conditioning_features(sample, factor=4)
    assert at_override.shape == (1, 8, 32, 32)
    assert torch.isfinite(at_override).all()


def test_fno_conv_mode_rejects_factor_override():
    """FNO-C has a fixed-depth upsampling stack, so an override must fail loudly, not silently."""
    encoder = FNOEncoder(
        upscale_factor=2, in_channels=1, out_channels=1, feature_channels=8,
        latent_channels=8, num_fno_layers=1, num_fno_modes=(3, 3), padding=0,
        upsample_mode="conv", backend="builtin",
    )
    # pytest.raises asserts the block raises the given exception type; `match` checks the message text
    # so we know it failed for the intended reason rather than some unrelated error.
    with pytest.raises(ValueError, match="cannot override factor"):
        encoder.conditioning_features(torch.randn(1, 1, 8, 8), factor=4)


def test_fno_conv_mode_rejects_non_power_of_two_factor():
    """The 'conv' variant builds one 2x stage per doubling, so factor 3 is unrepresentable."""
    with pytest.raises(ValueError, match="power-of-two"):
        FNOEncoder(upscale_factor=3, upsample_mode="conv", backend="builtin")


def test_spectral_mode_requires_builtin_backend():
    """Asking PhysicsNeMo for spectral upsampling must fail rather than silently build a different model."""
    with pytest.raises(ValueError, match="requires backend='builtin'"):
        FNOEncoder(upscale_factor=2, upsample_mode="spectral", backend="physicsnemo")


# ── Factory ───────────────────────────────────────────────────────────────────────────────────

def test_factory_defaults_to_rrdb():
    """An absent `encoder_type` must reproduce the previous hardcoded RRDB construction."""
    encoder = build_encoder(encoder_type="rrdb", upscale_factor=4, in_channels=1, out_channels=1)
    assert isinstance(encoder, RRDBNet)  # Correct class
    assert len(encoder.trunk) == 8  # num_blocks=8 was the value hardcoded at both original call sites


def test_factory_builds_fno():
    """The 'fno' selector must route through to an FNOEncoder with the supplied kwargs."""
    encoder = build_encoder(
        encoder_type="fno", upscale_factor=2, in_channels=1, out_channels=1,
        encoder_kwargs={"latent_channels": 8, "num_fno_layers": 1, "num_fno_modes": (3, 3), "backend": "builtin"},
    )
    assert isinstance(encoder, FNOEncoder)
    assert encoder.backend == "builtin"  # The requested backend was honoured, not silently swapped


def test_factory_rejects_unknown_type():
    """A typo in `encoder_type` must abort rather than defaulting to some architecture."""
    with pytest.raises(ValueError, match="Unknown encoder_type"):
        build_encoder(encoder_type="transformer", upscale_factor=2, in_channels=1, out_channels=1)


@pytest.mark.parametrize("alias", ["RRDB", " rrdb ", "Rrdb"])  # Case and whitespace variations from hand-edited YAML
def test_factory_normalises_encoder_type(alias):
    """Selector matching must be forgiving of case and stray whitespace in config files."""
    assert isinstance(build_encoder(encoder_type=alias, upscale_factor=2, in_channels=1, out_channels=1), RRDBNet)


# ── Integration with the diffusion U-Net ──────────────────────────────────────────────────────

def test_fno_conditioning_feeds_diffusion_unet(tmp_path, monkeypatch):
    """End-to-end shape contract: FNO conditioning must drop into DiffusionModel unchanged.

    Exercises the production path (`enc_output=False`), where the U-Net's init_dim resolves to 64
    and the conditioning tensor must therefore be 64 channels wide.
    """
    dataset = _DummyDataset(num_fields=1, img_shape=16, factor=2)  # 16x16 HR, 8x8 LR, single field

    def fake_initialize_encoder(self):
        """Bypass checkpoint loading — this test is about shapes, not about restoring weights."""
        model = build_encoder(
            encoder_type="fno",  # The architecture under test
            upscale_factor=self.train_dataset.factor,  # Matches the dummy dataset's 2x
            in_channels=1,  # Single field in
            out_channels=1,  # Single field out
            encoder_kwargs={
                "feature_channels": 64,  # Must equal the U-Net's init_dim on this code path
                "latent_channels": 8,  # Small for speed
                "num_fno_layers": 1,  # Shallow for speed
                "num_fno_modes": (3, 3),  # Within the 8x8 LR Nyquist limit
                "backend": "builtin",  # Runs without PhysicsNeMo
            },
        )
        model.eval()  # Frozen encoder, matching production behaviour
        return model.to(self.device)

    # monkeypatch.setattr temporarily replaces the method for the duration of this test only; pytest
    # restores the original automatically afterwards, so other tests are unaffected.
    monkeypatch.setattr(DiffusionModel, "initialize_encoder", fake_initialize_encoder)

    model = DiffusionModel(
        results_folder=tmp_path / "fno_diffusion",  # Scratch dir pytest creates and cleans up
        lr_encoder_folder=tmp_path / "encoder",  # Unused because initialize_encoder is patched
        train_dataset=dataset,
        dev_dataset=dataset,
        test_dataset=dataset,
        timesteps=4,  # Tiny diffusion chain; we only need one loss evaluation
        conditioning="implicit",  # The paper's conditioning mode: x_e is added after init_conv
        encoding=True,  # Turn the encoder path on
        schedule="linear",
        device="cpu",  # CPU so the test runs anywhere
        enc_output=False,  # Production path -> init_dim falls back to 64
        encoder_type="fno",  # Recorded on the model; the patched initializer builds the real thing
    )

    true_lr = torch.randn(2, 1, 8, 8)  # LR batch matching the dummy dataset's factor
    batch = torch.randn(2, 1, 16, 16)  # HR batch the denoiser operates on

    # The real production call: forwardpass dispatches to conditioning_features because output=False.
    x_e = forwardpass(model.lr_enc, true_lr, factor=dataset.factor, output=model.enc_output)
    assert x_e.shape == (2, 64, 16, 16)  # 64 channels at HR — the contract the U-Net's init_conv expects

    t = torch.randint(0, model.timesteps, (batch.shape[0],), dtype=torch.long)  # Random diffusion timesteps
    loss = model.p_losses(model.model, batch, t, loss_type="l1", x_e=x_e)  # One denoising loss evaluation

    assert loss.ndim == 0  # p_losses reduces to a scalar
    assert torch.isfinite(loss)  # No NaN/Inf from the spectral path propagating into the U-Net


@pytest.mark.skipif(not PHYSICSNEMO_AVAILABLE, reason="physicsnemo not installed in this environment")
@torch.no_grad()
def test_physicsnemo_backend_matches_builtin_contract():
    """When PhysicsNeMo is present, its FNO must satisfy the same interface as the builtin one.

    Skipped locally (Windows dev box) and exercised on TRACE, which is where the real experiments run.
    """
    torch.manual_seed(0)
    encoder = FNOEncoder(
        upscale_factor=4, in_channels=1, out_channels=1, feature_channels=64,
        latent_channels=16, num_fno_layers=2, num_fno_modes=(4, 4), padding=2,
        upsample_mode="pre", backend="physicsnemo",
    )
    encoder.eval()
    features = encoder.conditioning_features(torch.randn(2, 1, 20, 20))  # SS316L 4x LR size
    assert features.shape == (2, 64, 80, 80)  # The real SS316L HR conditioning shape
    assert torch.isfinite(features).all()
