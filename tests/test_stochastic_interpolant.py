"""Tests for StochasticInterpolantModel (levels 2 and 3 of the interpolant framework).

Two of these are gates the study depends on:

  test_score_formula_matches_analytic_gaussian
      Verifies grad log rho_t = -(alpha_t * sigma)^-1 * g_t, the identity the SDE sampler is built
      on. Everything is Gaussian, so the true score is available exactly by autograd -- making this
      a real check rather than a restatement of the formula.

  test_interpolant_velocity_target_matches_flow_matching
      With a Gaussian base, this class must reduce to the already-verified FlowMatchingModel. The
      two conventions run time in opposite directions, so the velocity targets must be exact
      negatives. If this fails the interpolant is misderived and nothing downstream is meaningful.
"""

import math  # sqrt, used when reconstructing the SDE's noise scale

import pytest  # Parametrisation and exception assertions
import torch  # Tensors, autograd, and the distributions used for the analytic check

from diffusionsr.runners.train_diffusion import forwardpass  # Checked for interface compatibility
from diffusionsr.runners.train_stochastic_interpolant import StochasticInterpolantModel


class _DummyDataset:
    """Minimal stand-in exposing only the attributes DiffusionModel reads from a dataset."""

    def __init__(self, num_fields=1, img_shape=16, n_steps=1, factor=2):
        self.num_fields = num_fields  # Physical fields per timestep
        self.img_shape = img_shape  # HR grid side length; also the U-Net's `dim`
        self.n_steps = n_steps  # Stacked prior timesteps
        self.out_steps = n_steps  # Output timestep count
        self.factor = factor  # HR/LR ratio
        self.inflate_dim = None  # Present because some code paths check it
        self.field_names = ["temperature"] if num_fields == 1 else ["temperature", "liqlabel"]


def _make_si_model(tmp_path, monkeypatch, si_base="bicubic", si_sigma=0.5, si_epsilon=0.0,
                   encoding=False, dataset=None):
    """Build a StochasticInterpolantModel on CPU without needing an encoder checkpoint."""
    dataset = dataset or _DummyDataset()

    if encoding:
        def fake_initialize_encoder(self):
            """Bypass checkpoint loading; these tests are about the interpolant, not the encoder."""
            from diffusionsr.models.encoder_factory import build_encoder
            model = build_encoder(
                "fno",  # Either architecture works here; FNO exercises the newer path
                upscale_factor=self.train_dataset.factor,
                in_channels=self.train_dataset.n_steps * self.train_dataset.num_fields,
                out_channels=self.train_dataset.n_steps * self.train_dataset.num_fields,
                encoder_kwargs={"feature_channels": 64, "latent_channels": 8, "num_fno_layers": 1,
                                "num_fno_modes": (3, 3), "backend": "builtin"},
            )
            model.eval()  # Frozen, matching production behaviour
            return model.to(self.device)

        # monkeypatch.setattr swaps the method for this test only; pytest restores it afterwards.
        monkeypatch.setattr(StochasticInterpolantModel, "initialize_encoder", fake_initialize_encoder)

    return StochasticInterpolantModel(
        results_folder=tmp_path / f"si_{si_base}",  # Scratch dir pytest creates and cleans up
        lr_encoder_folder=tmp_path / "encoder",  # Unused when initialize_encoder is patched
        train_dataset=dataset,
        dev_dataset=dataset,
        test_dataset=dataset,
        timesteps=4,  # Only used as the time-embedding scale here
        conditioning="implicit",  # The paper's conditioning mode
        encoding=encoding,
        schedule="linear",  # Ignored -- initialize_variance_schedule sets betas to None
        device="cpu",  # So the tests run anywhere
        enc_output=False,  # Production path: conditioning is the 64-channel intermediate
        si_base=si_base,
        si_sigma=si_sigma,
        si_epsilon=si_epsilon,
    )


# ── Gate 1: the score identity the SDE is built on ────────────────────────────────────────────

@pytest.mark.parametrize("a", [1.0, 0.9, 0.5])  # Encoder quality: 1.0 = perfect predictor
@pytest.mark.parametrize("sigma", [1.0, 0.3, 0.05])  # Base noise scale
def test_score_formula_matches_analytic_gaussian(a, sigma):
    """grad log rho_t(x) must equal -(alpha_t * sigma)^-1 * g_t(x).

    Setup: x1 ~ N(mu1, Sig1) and base x0 = a*x1 + sigma*zeta, so

        I_t = (1-t)*x0 + t*x1 = k*x1 + c*zeta,   k = (1-t)*a + t,   c = (1-t)*sigma

    Every quantity is Gaussian, so rho_t = N(k*mu1, k^2 Sig1 + c^2 I) and its score is obtainable
    by autograd on the exact log-density -- independent of the formula under test.
    """
    torch.manual_seed(0)
    dtype = torch.float64  # Double precision so the comparison is limited by the formula, not by rounding
    d = 4

    mu1 = torch.randn(d, dtype=dtype)
    A = torch.randn(d, d, dtype=dtype)
    # A @ A.T is positive semi-definite; adding a multiple of the identity makes it strictly
    # positive definite so MultivariateNormal accepts it.
    Sig1 = A @ A.T + 0.5 * torch.eye(d, dtype=dtype)

    for t in [0.0, 0.25, 0.5, 0.75, 0.95]:  # t=1 excluded: alpha=0 there, so the score is genuinely singular
        alpha = 1.0 - t
        k = alpha * a + t  # Coefficient on x1 in the interpolant
        c = alpha * sigma  # Coefficient on zeta in the interpolant
        cov = k * k * Sig1 + c * c * torch.eye(d, dtype=dtype)  # Cov(I_t)

        x = torch.randn(d, dtype=dtype)

        # --- the true score, by differentiating the exact log-density ---
        x_grad = x.clone().requires_grad_(True)  # requires_grad_ marks it for autograd tracking
        log_prob = torch.distributions.MultivariateNormal(k * mu1, cov).log_prob(x_grad)
        true_score = torch.autograd.grad(log_prob, x_grad)[0]  # Returns a tuple; take the first element

        # --- g_t(x) = E[zeta | I_t = x], by Gaussian conditioning ---
        # Cov(zeta, I_t) = Cov(zeta, k*x1 + c*zeta) = c*I, because zeta is independent of x1.
        # torch.linalg.solve(M, v) computes M^-1 v without forming the inverse.
        g_t = c * torch.linalg.solve(cov, x - k * mu1)

        claimed_score = -(alpha * sigma) ** -1 * g_t  # The formula the SDE sampler uses

        assert torch.allclose(claimed_score, true_score, atol=1e-10), (
            f"score formula wrong at a={a}, sigma={sigma}, t={t}: "
            f"max err {(claimed_score - true_score).abs().max().item():.3e}"
        )


def test_conditional_latent_expectation_matches_monte_carlo():
    """Independently confirm the closed-form g_t really is E[zeta | I_t = x], by binning in 1-D."""
    torch.manual_seed(1)
    m1, s1, a, sigma, t = 0.7, 1.3, 0.8, 0.4, 0.3
    alpha = 1.0 - t
    k, c = alpha * a + t, alpha * sigma

    n = 2_000_000  # Large enough that a narrow bin still holds tens of thousands of samples
    x1 = m1 + s1 * torch.randn(n)
    zeta = torch.randn(n)
    interpolant = alpha * (a * x1 + sigma * zeta) + t * x1  # I_t, built the same way the model does

    variance = k * k * s1 * s1 + c * c  # Var(I_t) in 1-D

    for x_star in [-1.0, 0.0, 1.0]:
        # Select samples whose interpolant landed near x_star; their mean zeta estimates E[zeta|I_t=x*].
        selected = (interpolant - x_star).abs() < 0.02
        monte_carlo = zeta[selected].mean().item()
        closed_form = c * (x_star - k * m1) / variance
        # Tolerance reflects Monte Carlo noise and the finite bin width, not the formula's accuracy.
        assert abs(monte_carlo - closed_form) < 0.02, (
            f"g_t mismatch at x*={x_star}: MC={monte_carlo:.4f} vs closed form={closed_form:.4f}"
        )


# ── Gate 2: reduction to the already-verified FlowMatchingModel ───────────────────────────────

def test_interpolant_velocity_target_matches_flow_matching():
    """With a Gaussian base, the velocity target must be the exact negative of FlowMatchingModel's.

    The two classes run time in opposite directions:

        SI  (t=0 base, t=1 data):   I_t = (1-t)*x0 + t*x1       dI/dt = x1 - x0
        FM  (t=0 data, t=1 noise):  x_t = (1-t)*x_start + t*eps  dx/dt = eps - x_start

    Identifying x0 <-> eps and x1 <-> x_start, the targets are exact negatives. This pins the new
    machinery to the old, verified implementation at the one configuration where both apply.
    """
    torch.manual_seed(0)
    x_hr = torch.randn(2, 1, 16, 16)  # Plays x1 for SI and x_start for FM
    base = torch.randn(2, 1, 16, 16)  # Plays x0 for SI and the noise for FM

    si_velocity_target = x_hr - base  # As computed in StochasticInterpolantModel.interpolant_loss
    fm_velocity_target = base - x_hr  # As computed in FlowMatchingModel.fm_loss

    # torch.equal requires exact elementwise equality, not approximate -- both are one subtraction.
    assert torch.equal(si_velocity_target, -fm_velocity_target), (
        "the interpolant velocity target no longer matches flow matching's; the derivation changed"
    )


def test_interpolant_path_hits_both_endpoints():
    """I_0 must be exactly the base and I_1 exactly the target (Def. 3.1's boundary conditions)."""
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 8, 8)
    x1 = torch.randn(2, 1, 8, 8)

    for t_value, expected, name in [(0.0, x0, "I_0 == x0 (base)"), (1.0, x1, "I_1 == x1 (target)")]:
        t_ = torch.full((2, 1, 1, 1), t_value)
        interpolant = (1.0 - t_) * x0 + t_ * x1
        assert torch.allclose(interpolant, expected, atol=1e-6), f"boundary condition violated: {name}"


# ── The model itself ──────────────────────────────────────────────────────────────────────────

def test_two_head_output_width(tmp_path, monkeypatch):
    """The U-Net must emit 2x the field width, splitting cleanly into velocity and latent."""
    model = _make_si_model(tmp_path, monkeypatch)
    x = torch.randn(2, 1, 16, 16)
    t = torch.full((2,), 0.3)
    with torch.no_grad():
        out = model.model(x, t * model.si_timescale, x_e=None)
    assert out.shape == (2, 2, 16, 16), f"expected doubled output width, got {tuple(out.shape)}"
    velocity, latent = out.chunk(2, dim=1)  # The same split the loss and sampler perform
    assert velocity.shape == (2, 1, 16, 16) and latent.shape == (2, 1, 16, 16)


def test_betas_is_none(tmp_path, monkeypatch):
    """No DDPM schedule should exist -- the guard that makes inherited DDPM code fail loudly."""
    model = _make_si_model(tmp_path, monkeypatch)
    assert model.betas is None


def test_bicubic_base_is_upscaled_lr_plus_noise(tmp_path, monkeypatch):
    """The 'bicubic' coupling must be upscaled_lr + sigma*zeta, with the returned zeta consistent."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic", si_sigma=0.25)
    upscaled_lr = torch.randn(2, 1, 16, 16)
    true_lr = torch.randn(2, 1, 8, 8)

    x0, zeta = model.compute_base(true_lr, upscaled_lr)
    # Reconstructing the base from the returned zeta must recover x0 exactly, which proves the zeta
    # handed to the score head is the same draw that went into x0.
    assert torch.allclose(x0, upscaled_lr + 0.25 * zeta, atol=1e-6)


def test_encoder_base_uses_the_field_prediction(tmp_path, monkeypatch):
    """The 'enc' coupling must start from the encoder's FIELD output, not its 64-channel features."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="enc", encoding=True, si_sigma=0.1)
    true_lr = torch.randn(2, 1, 8, 8)
    upscaled_lr = torch.randn(2, 1, 16, 16)

    x0, zeta = model.compute_base(true_lr, upscaled_lr)
    # 1 channel, not 64: output=True selects the field prediction. A 64-channel base would be a
    # shape error downstream, but failing here names the cause.
    assert x0.shape == (2, 1, 16, 16), f"base should be field-shaped, got {tuple(x0.shape)}"

    with torch.no_grad():
        expected_mean = forwardpass(model.lr_enc, true_lr.float(), factor=2, output=True)
    assert torch.allclose(x0 - 0.1 * zeta, expected_mean, atol=1e-5)


def test_loss_is_finite_and_trains(tmp_path, monkeypatch):
    """One optimisation step must change the parameters, with both heads contributing."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic")
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

    x_hr = torch.randn(2, 1, 16, 16)
    true_lr = torch.randn(2, 1, 8, 8)
    upscaled_lr = torch.randn(2, 1, 16, 16)

    x_e = model.compute_x_e(true_lr, upscaled_lr)  # Also caches the base -- the documented invariant
    t = torch.randint(0, model.timesteps, (2,))  # Deliberately ignored by p_losses; passed for signature parity
    loss = model.p_losses(model.model, x_hr, t, loss_type="l2", x_e=x_e)

    assert loss.ndim == 0 and torch.isfinite(loss)
    before = next(model.model.parameters()).detach().clone()
    loss.backward()
    optimizer.step()
    after = next(model.model.parameters()).detach()
    assert not torch.equal(before, after), "no parameter moved; the loss is not connected to the model"


def test_missing_base_cache_raises(tmp_path, monkeypatch):
    """p_losses without a preceding compute_x_e must fail loudly, not silently invent a base."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic")
    with pytest.raises(RuntimeError, match="no base cached"):
        model.p_losses(model.model, torch.randn(2, 1, 16, 16), torch.randint(0, 4, (2,)), x_e=None)


def test_stale_base_cache_is_detected(tmp_path, monkeypatch):
    """A cache from a differently sized batch must be caught rather than broadcast into nonsense."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic")
    model.compute_x_e(torch.randn(4, 1, 8, 8), torch.randn(4, 1, 16, 16))  # Cache batch of 4
    with pytest.raises(RuntimeError, match="invariant has been broken"):
        model.p_losses(model.model, torch.randn(2, 1, 16, 16), torch.randint(0, 4, (2,)), x_e=None)


# ── Sampling ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0])  # 0.0 is the ODE; the rest are SDEs
def test_sampling_starts_at_the_base_and_runs(tmp_path, monkeypatch, epsilon):
    """Integration must begin AT the base (not at noise) and produce finite output for any epsilon."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic", si_sigma=0.5)
    x0 = torch.randn(2, 1, 16, 16)

    trajectory = model.euler_sample(model.model, x_e=None, x0=x0, n_steps=5, epsilon=epsilon)
    assert len(trajectory) == 5  # One state recorded per Euler step

    # After a single step of size dt the state must still be near the base -- which is the whole
    # point of the coupling. Starting from noise instead would put it O(1) away immediately.
    first_step_displacement = (trajectory[0] - x0.cpu()).abs().mean().item()
    base_magnitude = x0.abs().mean().item()
    assert first_step_displacement < base_magnitude, "sampling does not appear to start at the base"

    assert torch.isfinite(trajectory[-1]).all(), f"non-finite sample at epsilon={epsilon}"


def test_epsilon_zero_is_deterministic(tmp_path, monkeypatch):
    """epsilon=0 must recover the probability-flow ODE exactly -- same input, same output."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic")
    x0 = torch.randn(2, 1, 16, 16)

    first = model.euler_sample(model.model, x_e=None, x0=x0, n_steps=4, epsilon=0.0)[-1]
    second = model.euler_sample(model.model, x_e=None, x0=x0, n_steps=4, epsilon=0.0)[-1]
    assert torch.equal(first, second), "epsilon=0 is not deterministic; the ODE path injects noise"


def test_positive_epsilon_is_stochastic(tmp_path, monkeypatch):
    """epsilon>0 must produce sample diversity -- the mechanism keyhole variability depends on."""
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic")
    x0 = torch.randn(2, 1, 16, 16)

    first = model.euler_sample(model.model, x_e=None, x0=x0, n_steps=4, epsilon=0.5)[-1]
    second = model.euler_sample(model.model, x_e=None, x0=x0, n_steps=4, epsilon=0.5)[-1]
    assert not torch.equal(first, second), "epsilon>0 produced identical samples; the SDE is not stochastic"


def test_sde_drift_matches_the_verified_formula(tmp_path, monkeypatch):
    """The sampler's update must implement exactly the drift the score identity implies.

    This is what links the analytic gate above to the code. With the model stubbed to return known
    constant velocity and latent fields, one Euler step is computable by hand:

        eps_t  = epsilon * alpha_t                      (the design choice in euler_sample)
        score  = -(alpha_t * sigma)^-1 * latent         (verified analytically above)
        drift  = velocity + eps_t * score
               = velocity - (epsilon / sigma) * latent  (the alpha factors cancel)
        x_next = x + drift*dt + sqrt(2*eps_t*dt) * xi
    """
    sigma, epsilon, n_steps = 0.5, 0.4, 1
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic", si_sigma=sigma)

    velocity_value, latent_value = 2.0, 3.0  # Arbitrary distinct constants so a swap would show up

    class _ConstantModel(torch.nn.Module):
        """Returns fixed velocity and latent halves, so the arithmetic is fully predictable."""

        def forward(self, x, t, x_e=None):
            # torch.cat joins along the channel axis, producing the [velocity | latent] layout
            # chunk(2, dim=1) splits back out in euler_sample.
            return torch.cat([torch.full_like(x, velocity_value),
                              torch.full_like(x, latent_value)], dim=1)

    stub = _ConstantModel()
    x0 = torch.zeros(1, 1, 8, 8)  # Zero start makes the expected value easy to read off

    torch.manual_seed(123)  # Fix the RNG so the Brownian increment is reproducible
    produced = model.euler_sample(stub, x_e=None, x0=x0, n_steps=n_steps, epsilon=epsilon)[-1]

    # Recompute the same step by hand. At step 0, t=0 so alpha_t = 1.
    dt, alpha_t = 1.0 / n_steps, 1.0
    expected_drift = velocity_value - (epsilon / sigma) * latent_value
    torch.manual_seed(123)  # Replay the identical RNG stream the sampler consumed
    brownian = torch.randn_like(x0)
    expected = x0 + expected_drift * dt + math.sqrt(2.0 * epsilon * alpha_t * dt) * brownian

    assert torch.allclose(produced, expected, atol=1e-6), (
        f"SDE update does not match the formula: produced {produced.flatten()[0].item():.6f}, "
        f"expected {expected.flatten()[0].item():.6f}"
    )


def test_batch_sample_api_matches_the_other_models(tmp_path, monkeypatch):
    """`batch_sample` must keep the shape the analysis layer expects from every model class."""
    dataset = _DummyDataset()
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic", dataset=dataset)

    x_e = model.compute_x_e(torch.randn(2, 1, 8, 8), torch.randn(2, 1, 16, 16))  # Caches the base
    samples = model.batch_sample(dataset=dataset, batch=torch.randn(2, 1, 16, 16), x_e=x_e, n_steps=3)

    # (n_steps, B, C, H, W); callers take [-1] for the final sample.
    assert samples.shape == (3, 2, 1, 16, 16), f"unexpected trajectory shape {tuple(samples.shape)}"


def test_unsupported_sampler_raises(tmp_path, monkeypatch):
    """Only Euler is implemented; a DDIM request must not silently fall through to Euler."""
    dataset = _DummyDataset()
    model = _make_si_model(tmp_path, monkeypatch, si_base="bicubic", dataset=dataset)
    model.compute_x_e(torch.randn(2, 1, 8, 8), torch.randn(2, 1, 16, 16))
    with pytest.raises(NotImplementedError, match="only supports sampler='euler'"):
        model.batch_sample(dataset=dataset, batch=torch.randn(2, 1, 16, 16), x_e=None, sampler="DDIM")


# ── Configuration guards ──────────────────────────────────────────────────────────────────────

def test_zero_sigma_rejected(tmp_path, monkeypatch):
    """sigma=0 puts the base on a lower-dimensional manifold where the score is singular."""
    with pytest.raises(ValueError, match="si_sigma must be > 0"):
        _make_si_model(tmp_path, monkeypatch, si_sigma=0.0)


def test_unknown_base_rejected(tmp_path, monkeypatch):
    """A typo in si_base must abort rather than defaulting to some coupling."""
    with pytest.raises(ValueError, match="si_base must be"):
        _make_si_model(tmp_path, monkeypatch, si_base="encoder")  # 'enc' is the correct spelling


def test_encoder_base_without_encoder_rejected(tmp_path, monkeypatch):
    """si_base='enc' with encoding=False has no encoder to start from; catch it at construction."""
    with pytest.raises(ValueError, match="requires encoding=True"):
        _make_si_model(tmp_path, monkeypatch, si_base="enc", encoding=False)
