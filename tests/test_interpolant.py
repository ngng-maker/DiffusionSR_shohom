"""Tests for the stochastic-interpolant math in `runners/interpolant.py`.

Three of these are correctness gates for claims the sampler and training loop rest on, rather than
smoke tests:

  test_option_b_score_identity_matches_analytic_gaussian
      `CoupledInterpolantModel`'s SDE uses score = -g_hat/gamma(t). That identity is asserted by the
      framework but never checked numerically. Here everything is Gaussian, so the true score is
      available exactly by autograd on the log-density -- making this an independent check rather
      than a restatement.

  test_ulr_to_hr_space_round_trip  /  test_ulr_to_hr_space_puts_ulr_in_hr_statistics
      `ulr_to_hr_space` exists because the dataset normalises `upscaled_lr` and `hr` with *separate*
      statistics, so an interpolant between them would otherwise have endpoints in different
      normalised spaces. That is a silent failure -- training converges to something subtly wrong
      rather than crashing -- so it is worth pinning.

  test_option_a_velocity_target_matches_flow_matching
      Option A must reduce to the already-verified FlowMatchingModel. The two run time in opposite
      directions, so the velocity targets must be exact negatives.
"""

import pytest  # Parametrisation and exception assertions
import torch  # Tensors, autograd, and the distributions used for the analytic checks

from diffusionsr.runners.interpolant import (  # The module under test
    gamma,
    gamma_dot,
    interpolate_A,
    interpolate_B,
    make_x0,
    sde_eps,
    si_loss_A,
    si_loss_B,
    ulr_to_hr_space,
    velocity_target_A,
    velocity_target_B,
)


# ── Gate 1: the score identity the Option-B SDE rests on ──────────────────────────────────────

@pytest.mark.parametrize("a", [1.0, 0.9, 0.5])  # Base quality: 1.0 = base equals the target up to noise
@pytest.mark.parametrize("sigma", [0.2, 0.05])  # Noise added when building x0
@pytest.mark.parametrize("c", [0.25, 0.10])  # Option-B path-noise amplitude
def test_option_b_score_identity_matches_analytic_gaussian(a, sigma, c):
    """score(I_t) = -g_hat/gamma(t) must be the true score of rho_t, where g_hat = E[z | I_t].

    Construction, all Gaussian so rho_t is Gaussian too:

        x1 ~ N(mu1, Sig1)
        x0  = a*x1 + sigma*zeta                      the data-dependent coupling
        I_t = (1-t)*x0 + t*x1 + gamma(t)*z           Option B

    Collecting terms, I_t = k*x1 + c_zeta*zeta + c_z*z with

        k = (1-t)*a + t,    c_zeta = (1-t)*sigma,    c_z = gamma(t)

    so rho_t = N(k*mu1, k^2 Sig1 + (c_zeta^2 + c_z^2) I) and its score follows by autograd on the
    exact log-density -- independently of the identity under test.
    """
    torch.manual_seed(0)
    dtype = torch.float64  # Double precision so the comparison is limited by the identity, not rounding
    d = 4

    mu1 = torch.randn(d, dtype=dtype)
    A = torch.randn(d, d, dtype=dtype)
    # A @ A.T is positive semi-definite; the identity term makes it strictly positive definite so
    # MultivariateNormal accepts it as a covariance.
    Sig1 = A @ A.T + 0.5 * torch.eye(d, dtype=dtype)

    # t = 0 and t = 1 are excluded: gamma vanishes there, so the score is genuinely singular. That is
    # exactly why `sde_eps` is designed to make eps(t)/gamma(t) constant -- see the test below.
    for t_value in [0.1, 0.25, 0.5, 0.75, 0.9]:
        t = torch.tensor(t_value, dtype=dtype)
        k = (1.0 - t_value) * a + t_value  # Coefficient on x1
        c_zeta = (1.0 - t_value) * sigma  # Coefficient on zeta (from x0)
        c_z = gamma(t, c).item()  # Coefficient on z (the Option-B path noise)

        cov = k * k * Sig1 + (c_zeta ** 2 + c_z ** 2) * torch.eye(d, dtype=dtype)  # Cov(I_t)
        x = torch.randn(d, dtype=dtype)

        # --- the true score, by differentiating the exact log-density ---
        x_grad = x.clone().requires_grad_(True)  # Marks the tensor for autograd tracking
        log_prob = torch.distributions.MultivariateNormal(k * mu1, cov).log_prob(x_grad)
        true_score = torch.autograd.grad(log_prob, x_grad)[0]  # grad returns a tuple; take element 0

        # --- g_hat = E[z | I_t = x], by Gaussian conditioning ---
        # Cov(z, I_t) = Cov(z, k*x1 + c_zeta*zeta + c_z*z) = c_z * I, since z is independent of both
        # x1 and zeta. linalg.solve(M, v) computes M^-1 v without forming the inverse.
        g_hat = c_z * torch.linalg.solve(cov, x - k * mu1)

        claimed_score = -g_hat / c_z  # The identity CoupledInterpolantModel's SDE sampler uses

        assert torch.allclose(claimed_score, true_score, atol=1e-10), (
            f"score = -g_hat/gamma(t) is wrong at a={a}, sigma={sigma}, c={c}, t={t_value}: "
            f"max err {(claimed_score - true_score).abs().max().item():.3e}"
        )


def test_sde_eps_over_gamma_is_constant():
    """eps(t)/gamma(t) must be exactly eps0/c, including arbitrarily close to t=0 and t=1.

    This is the property that keeps the SDE step finite: the drift correction is
    eps(t) * score = -(eps(t)/gamma(t)) * g_hat, and gamma(t) -> 0 at both endpoints. If the ratio
    were not constant the sampler would divide by (near) zero there.
    """
    eps0, c = 0.3, 0.25
    expected = eps0 / c

    # Includes values close enough to the endpoints that a non-constant ratio would blow up.
    for t_value in [1e-8, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-4, 1 - 1e-8]:
        t = torch.tensor(t_value, dtype=torch.float64)
        ratio = (sde_eps(t, eps0, c) / gamma(t, c)).item()
        assert abs(ratio - expected) < 1e-9, f"eps/gamma = {ratio} != {expected} at t={t_value}"


def test_sde_eps_zero_recovers_the_ode():
    """eps0 = 0 must give eps(t) = 0 everywhere, i.e. a deterministic sampler with no retraining."""
    for t_value in [0.0, 0.3, 0.7, 1.0]:
        t = torch.tensor(t_value)
        assert sde_eps(t, 0.0, 0.25).item() == 0.0


# ── Gate 2: the normalisation fix ─────────────────────────────────────────────────────────────

def test_ulr_to_hr_space_round_trip():
    """Re-normalising must be exactly invertible: ULR-stat -> physical -> HR-stat -> back."""
    torch.manual_seed(0)
    shape = (2, 1, 8, 8)

    # Deliberately different statistics for the two spaces -- that difference is the whole reason
    # this function exists. std tensors are offset from zero so no division is degenerate.
    mean_ulr = torch.randn(1, 8, 8)
    std_ulr = torch.rand(1, 8, 8) + 0.5
    mean_hr = torch.randn(1, 8, 8)
    std_hr = torch.rand(1, 8, 8) + 0.5

    ulr = torch.randn(shape)
    in_hr_space = ulr_to_hr_space(ulr, mean_ulr, std_ulr, mean_hr, std_hr)
    # Applying the inverse mapping (swap the two stat pairs) must recover the original.
    back = ulr_to_hr_space(in_hr_space, mean_hr, std_hr, mean_ulr, std_ulr)

    assert torch.allclose(back, ulr, atol=1e-5), "ulr_to_hr_space is not invertible"


def test_ulr_to_hr_space_puts_ulr_in_hr_statistics():
    """The output must actually be standardised by the HR statistics, not merely rescaled.

    Concretely: a field that is *exactly* the HR mean in physical units must map to zero in HR-stat
    space. If the two endpoints of the interpolant were in different normalised spaces, this is the
    offset that would silently bias training.
    """
    torch.manual_seed(0)
    mean_ulr = torch.full((1, 4, 4), 3.0)
    std_ulr = torch.full((1, 4, 4), 2.0)
    mean_hr = torch.full((1, 4, 4), -1.0)
    std_hr = torch.full((1, 4, 4), 0.5)

    # A tensor whose *physical* value equals mean_hr. In ULR-stat space that is
    # (mean_hr - mean_ulr)/std_ulr = (-1 - 3)/2 = -2.
    ulr_encoding_of_mean_hr = torch.full((1, 1, 4, 4), -2.0)
    result = ulr_to_hr_space(ulr_encoding_of_mean_hr, mean_ulr, std_ulr, mean_hr, std_hr)

    # Being exactly the HR mean, it must standardise to zero.
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6), (
        f"a field equal to the HR mean should map to 0 in HR-stat space, got {result.flatten()[0].item()}"
    )


def test_make_x0_is_base_plus_sigma_zeta():
    """x0 = ulr_hr + sigma*zeta, with the supplied zeta used verbatim when given."""
    torch.manual_seed(0)
    ulr_hr = torch.randn(2, 1, 8, 8)
    zeta = torch.randn(2, 1, 8, 8)

    # Passing zeta explicitly is what lets the caller reuse the SAME draw as the score target.
    assert torch.allclose(make_x0(ulr_hr, 0.25, zeta), ulr_hr + 0.25 * zeta, atol=1e-6)
    # sigma = 0 gives a deterministic base (documented behaviour), though the score is then singular.
    assert torch.allclose(make_x0(ulr_hr, 0.0, zeta), ulr_hr, atol=1e-6)


# ── Gate 3: reduction to the already-verified FlowMatchingModel ───────────────────────────────

def test_option_a_velocity_target_matches_flow_matching():
    """Option A's velocity target must be the exact negative of FlowMatchingModel's.

    The two run time in opposite directions:

        Option A (t=0 base, t=1 data):  I_t = (1-t)*x0 + t*x1       dI/dt = x1 - x0
        FlowMatching (t=0 data, t=1 noise): x_t = (1-t)*x_start + t*eps   dx/dt = eps - x_start

    Identifying x0 <-> eps and x1 <-> x_start, the targets are exact negatives. This pins the
    interpolant math to the verified flow-matching implementation at the one configuration where
    both apply (independent Gaussian base, gamma = 0).
    """
    torch.manual_seed(0)
    x_hr = torch.randn(2, 1, 16, 16)  # Plays x1 for Option A and x_start for FlowMatching
    base = torch.randn(2, 1, 16, 16)  # Plays x0 for Option A and the noise for FlowMatching

    option_a_target = velocity_target_A(base, x_hr)  # x1 - x0
    flow_matching_target = base - x_hr  # exactly FlowMatchingModel.fm_loss's `noise - x_start`

    # torch.equal demands exact elementwise equality; both sides are a single subtraction.
    assert torch.equal(option_a_target, -flow_matching_target), (
        "Option A's velocity target no longer matches flow matching's; the derivation changed"
    )


# ── Interpolant boundary conditions and consistency ───────────────────────────────────────────

def test_interpolants_hit_both_endpoints():
    """Def. 3.1's boundary conditions: I_0 == x0 (base) and I_1 == x1 (target), for both options."""
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 8, 8)
    x1 = torch.randn(2, 1, 8, 8)
    z = torch.randn(2, 1, 8, 8)

    for t_value, expected, label in [(0.0, x0, "I_0 == x0"), (1.0, x1, "I_1 == x1")]:
        t = torch.full((2,), t_value)
        assert torch.allclose(interpolate_A(x0, x1, t), expected, atol=1e-6), f"Option A: {label}"
        # Option B also holds because gamma(0) = gamma(1) = 0 kills the path-noise term.
        assert torch.allclose(interpolate_B(x0, x1, t, z, c=0.25), expected, atol=1e-6), f"Option B: {label}"


def test_option_b_reduces_to_option_a_when_c_is_zero():
    """c = 0 must make Option B identical to Option A, in both the path and the velocity target."""
    torch.manual_seed(0)
    x0, x1, z = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    t = torch.rand(2)

    assert torch.allclose(interpolate_B(x0, x1, t, z, c=0.0), interpolate_A(x0, x1, t), atol=1e-6)
    assert torch.allclose(velocity_target_B(x0, x1, z, t, c=0.0), velocity_target_A(x0, x1), atol=1e-6)


def test_gamma_boundary_and_peak():
    """gamma(t) = c*4t(1-t): zero at both endpoints, equal to c at the midpoint."""
    c = 0.3
    assert gamma(torch.tensor(0.0), c).item() == pytest.approx(0.0)
    assert gamma(torch.tensor(1.0), c).item() == pytest.approx(0.0)
    # 4 * 0.5 * 0.5 = 1, so gamma(0.5) = c exactly -- which is what makes c readable as "peak noise".
    assert gamma(torch.tensor(0.5), c).item() == pytest.approx(c)


def test_gamma_dot_is_the_derivative_of_gamma():
    """gamma_dot must match a central finite difference of gamma everywhere."""
    c = 0.3
    h = 1e-6  # Small enough for accuracy, large enough to avoid float64 cancellation
    for t_value in [0.1, 0.25, 0.5, 0.75, 0.9]:
        t = torch.tensor(t_value, dtype=torch.float64)
        numerical = ((gamma(t + h, c) - gamma(t - h, c)) / (2 * h)).item()
        analytic = gamma_dot(t, c).item()
        assert abs(numerical - analytic) < 1e-6, (
            f"gamma_dot wrong at t={t_value}: analytic {analytic:.6f} vs numerical {numerical:.6f}"
        )


def test_velocity_target_a_is_constant_along_the_path():
    """Option A's target must not depend on t -- the defining property of a straight-line path."""
    torch.manual_seed(0)
    x0, x1 = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    reference = velocity_target_A(x0, x1)
    for t_value in [0.0, 0.3, 0.7, 1.0]:
        # interpolate_A is evaluated purely to confirm the path is traversed; the target is unchanged.
        interpolate_A(x0, x1, torch.full((2,), t_value))
        assert torch.equal(velocity_target_A(x0, x1), reference)


# ── Losses ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("loss_type", ["l1", "l2", "huber"])
def test_losses_are_zero_at_a_perfect_prediction(loss_type):
    """All three loss types must bottom out at exactly zero when the prediction is exact."""
    torch.manual_seed(0)
    target = torch.randn(2, 1, 8, 8)
    z = torch.randn(2, 1, 8, 8)

    assert si_loss_A(target.clone(), target, loss_type).item() == pytest.approx(0.0, abs=1e-9)
    # Option B sums the velocity and denoiser terms, so a perfect prediction of both gives zero.
    assert si_loss_B(target.clone(), z.clone(), target, z, loss_type).item() == pytest.approx(0.0, abs=1e-9)


def test_option_b_loss_is_the_sum_of_its_two_terms():
    """si_loss_B must be exactly si_loss_A(velocity) + si_loss_A(denoiser), with equal weights."""
    torch.manual_seed(0)
    b_hat, g_hat = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    velocity_target, z = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)

    combined = si_loss_B(b_hat, g_hat, velocity_target, z, "l2")
    separate = si_loss_A(b_hat, velocity_target, "l2") + si_loss_A(g_hat, z, "l2")
    assert torch.allclose(combined, separate, atol=1e-9)


def test_unknown_loss_type_rejected():
    """A typo in loss_type must abort rather than silently defaulting to one of them."""
    torch.manual_seed(0)
    x = torch.randn(2, 1, 8, 8)
    with pytest.raises(ValueError, match="Unknown loss_type"):
        si_loss_A(x, x, "mse")  # 'l2' is the correct spelling in this module
