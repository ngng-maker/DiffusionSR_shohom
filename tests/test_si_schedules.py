"""Tests for the scheduled interpolant (runners/si_schedules.py) and its solvers (analysis/si_solvers.py).

The end-to-end gate is test_every_schedule_solver_and_eps_transports_base_to_target: with a Gaussian
target the exact v = E[alpha_dot m + beta_dot x1 | I_t] and eta = E[n | I_t] are available in closed
form, so plugging them into the solvers must reproduce the target distribution for EVERY schedule,
solver and eps. A wrong Gamma_dot, a sign error in the eps*Gamma*eta score term, a wrong
sqrt(2 eps) Gamma diffusion, or a broken final denoising jump each shows up as a wrong mean or std.
"""

import math

import numpy as np

import pytest
import torch

from diffusionsr.analysis.si_solvers import SDE_SOLVERS, SOLVERS, integrate
from diffusionsr.runners.si_schedules import (SCHEDULES, T_END, denoise_to_target, get_schedule,
                                              make_interpolant)

SIGMA = 0.1
NAMES = sorted(SCHEDULES)


@pytest.mark.parametrize('name', NAMES)
def test_endpoints(name):
    s = get_schedule(name)
    t0, t1 = torch.tensor(0.0), torch.tensor(1.0)
    assert float(s.alpha(t0)) == pytest.approx(1.0, abs=1e-6)
    assert float(s.beta(t0)) == pytest.approx(0.0, abs=1e-6)
    assert float(s.gamma_sq(t0)) == pytest.approx(0.0, abs=1e-12)
    assert float(s.alpha(t1)) == pytest.approx(0.0, abs=1e-6)
    assert float(s.beta(t1)) == pytest.approx(1.0, abs=1e-6)
    assert float(s.gamma_sq(t1)) == pytest.approx(0.0, abs=1e-12)
    # Gamma_0 = sigma for every schedule: the base's own noise.
    assert float(s.Gamma(t0, SIGMA)) == pytest.approx(SIGMA, rel=1e-6)


@pytest.mark.parametrize('name', NAMES)
def test_derivatives_match_finite_differences(name):
    s = get_schedule(name)
    t = torch.linspace(0.02, 0.98, 49, dtype=torch.float64)
    h = 1e-6
    fd = lambda f: (f(t + h) - f(t - h)) / (2 * h)
    torch.testing.assert_close(s.alpha_dot(t), fd(s.alpha), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s.beta_dot(t), fd(s.beta), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s.gamma_gamma_dot(t), 0.5 * fd(s.gamma_sq), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s.Gamma_dot(t, SIGMA), fd(lambda u: s.Gamma(u, SIGMA)), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('name', NAMES)
def test_vp_style_identities(name):
    s = get_schedule(name)
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    if name in ('trig', 'vp'):
        torch.testing.assert_close(s.alpha(t) ** 2 + s.beta(t) ** 2, torch.ones_like(t))
    assert torch.all(s.Gamma(t[:-1], SIGMA) > 0)    # score defined on [0, 1)


def test_vp_matches_diffusionsr_ddpm_schedule():
    """The VP arm's signal coefficient follows the same alphabar as DiffusionSR's linear-beta DDPM."""
    from diffusionsr.utils import linear_beta_schedule
    betas = linear_beta_schedule(1000).double()
    abar = torch.cumprod(1 - betas, 0)                      # abar[k] at s = (k+1)/1000
    s = torch.arange(1, 1001, dtype=torch.float64) / 1000
    vp = get_schedule('vp')
    r = vp._r(1 - s)                                        # sqrt(alphabar(s)), continuous
    torch.testing.assert_close(r, abar.sqrt(), rtol=2e-2, atol=2e-3)


def test_two_noise_and_one_noise_forms_agree():
    """alpha sigma zeta + gamma z has the same law as Gamma n, and projects onto n with Gamma_dot."""
    torch.manual_seed(0)
    s = get_schedule('bridge')
    t = torch.tensor(0.3)
    a, ad = s.alpha(t), s.alpha_dot(t)
    g = s.gamma_sq(t).sqrt()
    gd = s.gamma_gamma_dot(t) / g
    zeta, z = torch.randn(2, 400_000, dtype=torch.float64)
    noise = a * SIGMA * zeta + g * z
    dnoise = ad * SIGMA * zeta + gd * z
    G, Gd = s.Gamma(t, SIGMA), s.Gamma_dot(t, SIGMA)
    assert float(noise.std()) == pytest.approx(float(G), rel=5e-3)
    n = noise / G
    slope = float((dnoise * n).mean() / (n * n).mean())     # regression of d(noise)/dt on n
    assert slope == pytest.approx(float(Gd), rel=1e-2)


def test_make_interpolant_endpoints_and_targets():
    torch.manual_seed(0)
    m, x1 = torch.randn(2, 3, 1, 4, 4)
    for name in NAMES:
        s = get_schedule(name)
        I0, v0, n = make_interpolant(s, m, x1, torch.zeros(3), SIGMA)
        torch.testing.assert_close(I0, m + SIGMA * n)
        I, v, n = make_interpolant(s, m, x1, torch.full((3,), 0.4), SIGMA)
        c = s.coeffs(torch.tensor(0.4), SIGMA)
        torch.testing.assert_close(v, c['alpha_dot'] * m + c['beta_dot'] * x1)
        # The Tweedie jump inverts the interpolant when given the true noise.
        torch.testing.assert_close(denoise_to_target(s, I, m, n, 0.4, SIGMA), x1, rtol=1e-4, atol=1e-4)


# ── The end-to-end gate ────────────────────────────────────────────────────────────────────────

MU1, S1, M = 1.0, 0.7, 0.4      # target N(MU1, S1^2); base mean M (deliberately off-target)


def _exact_field(schedule):
    """Closed-form (v, eta) for x1 ~ N(MU1, S1^2) and a fixed base mean M."""
    def field(x, t):
        c = schedule.coeffs(torch.tensor(float(t), dtype=x.dtype), SIGMA)
        mean = c['alpha'] * M + c['beta'] * MU1
        var = c['beta'] ** 2 * S1 ** 2 + c['Gamma'] ** 2
        e_x1 = MU1 + c['beta'] * S1 ** 2 / var * (x - mean)
        e_n = c['Gamma'] / var * (x - mean)
        return c['alpha_dot'] * M + c['beta_dot'] * e_x1, e_n
    return field


CASES = [(sched, solver, eps) for sched in NAMES for solver in SOLVERS
         for eps in ((0.0, 0.1, 1.0) if solver in SDE_SOLVERS else (0.0,))]


@pytest.mark.parametrize('sched,solver,eps', CASES)
def test_every_schedule_solver_and_eps_transports_base_to_target(sched, solver, eps):
    g = torch.Generator().manual_seed(1)
    s = get_schedule(sched)
    x0 = M + SIGMA * torch.randn(40_000, dtype=torch.float64, generator=g)
    m = torch.full_like(x0, M)
    # Euler is first order and the bridge's Gamma_dot has a 1/sqrt(1-t) singularity at t -> 1, so its
    # bias decays slowly (std 1.50 / 0.80 / 0.71 at 20 / 200 / 2000 steps against 0.70). That is a real
    # property of the pair, which the experiment measures; here it only needs to converge.
    n_steps = 2000 if (sched, solver) == ('bridge', 'euler') else 200
    x1, nfe = integrate(_exact_field(s), s, x0, m, SIGMA, solver=solver, n_steps=n_steps, eps=eps,
                        generator=g)
    assert float(x1.mean()) == pytest.approx(MU1, abs=0.02)
    assert float(x1.std()) == pytest.approx(S1, rel=0.03)
    expected_nfe = {'euler': n_steps + 1, 'heun': 2 * n_steps + 1, 'rk4': 4 * n_steps + 1}.get(solver)
    if expected_nfe:
        assert nfe == expected_nfe
    if solver == 'dopri5':
        assert nfe < 400        # an O(dt) error estimate (the old tableau typo) needed ~15,000


def test_eps_zero_is_deterministic_and_ode_solvers_reject_eps():
    s = get_schedule('linear')
    x0 = M + SIGMA * torch.randn(1000, dtype=torch.float64)
    m = torch.full_like(x0, M)
    a, _ = integrate(_exact_field(s), s, x0, m, SIGMA, solver='heun', n_steps=10, eps=0.0)
    b, _ = integrate(_exact_field(s), s, x0, m, SIGMA, solver='heun', n_steps=10, eps=0.0)
    torch.testing.assert_close(a, b)
    for solver in ('rk4', 'dopri5'):
        with pytest.raises(ValueError):
            integrate(_exact_field(s), s, x0, m, SIGMA, solver=solver, eps=0.1)


def test_higher_order_solvers_converge_faster():
    """At 10 steps on the trig ODE, the error ordering must be euler > heun > rk4."""
    s = get_schedule('trig')
    x0 = M + SIGMA * torch.linspace(-3, 3, 501, dtype=torch.float64)
    m = torch.full_like(x0, M)
    ref, _ = integrate(_exact_field(s), s, x0, m, SIGMA, solver='rk4', n_steps=2000)
    errs = {sv: float((integrate(_exact_field(s), s, x0, m, SIGMA, solver=sv, n_steps=10)[0] - ref).abs().max())
            for sv in ('euler', 'heun', 'rk4')}
    assert errs['euler'] > errs['heun'] > errs['rk4']


# ── W&B: one artifact, one version ────────────────────────────────────────────────────────────

class _FakeVersion:
    def __init__(self, store, version):
        self.store, self.version, self.aliases = store, version, []

    def delete(self, delete_aliases=False):
        assert delete_aliases, 'W&B refuses to delete an aliased version without delete_aliases=True'
        self.store.remove(self)


class _FakeWandb:
    """Just enough of wandb for log_single_version_artifact: versions accumulate until pruned."""

    def __init__(self):
        self.versions = []
        self.counter = 100          # W&B version numbers only ever increase
        outer = self

        class Artifact:
            def __init__(self, name, type, metadata=None, description=None):
                self.name, self.files = name, []

            def add_file(self, path, name):
                self.files.append(name)

        class Run:
            entity, project = 'ent', 'proj'

            def log_artifact(self, art, aliases):
                outer.counter += 1
                v = _FakeVersion(outer.versions, f'v{outer.counter}')
                outer.versions.append(v)
                v.wait = lambda: v
                return v

        class Api:
            def __init__(self, timeout=None):
                pass

            def artifacts(self, type_, path):
                return list(outer.versions)

        self.Artifact, self.Api, self.run = Artifact, Api, Run()


def test_single_version_artifact_keeps_exactly_one_version(tmp_path, monkeypatch):
    import sys
    from diffusionsr.utils import log_single_version_artifact
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    f = tmp_path / 'ckpt.pth'
    f.write_bytes(b'x')
    for epoch in range(4):                       # e.g. 4 syncs, across a preemption and a restart
        assert log_single_version_artifact('si_sched_vp_enc_temp_ckpt', 'checkpoint',
                                           {'ckpt.pth': str(f), 'missing.txt': str(tmp_path / 'nope')},
                                           metadata={'epoch': epoch})
        assert len(fake.versions) == 1           # the newest version only, never an accumulation
    # A leftover version from an interrupted earlier upload is removed by the next upload.
    fake.versions.insert(0, _FakeVersion(fake.versions, 'v1'))
    log_single_version_artifact('si_sched_vp_enc_temp_ckpt', 'checkpoint', {'ckpt.pth': str(f)})
    assert [v.version for v in fake.versions] == ['v105']


def test_wandb_safe_id_is_deterministic_and_valid():
    from diffusionsr.utils import wandb_safe_id
    a = wandb_safe_id('si_sched_bridge_ulr_temp__heun_eps0p05-v1')
    assert a == wandb_safe_id('si_sched_bridge_ulr_temp__heun_eps0p05-v1')
    assert a == 'si-sched-bridge-ulr-temp-heun-eps0p05-v1' and len(a) <= 64


# ── SI-noise study: linear_gamma schedule, large eps, larger sigma, ensemble-skill scores ─────────

GAMMA_SCALES = (0.5, 1.0, 2.0)


@pytest.mark.parametrize('c', GAMMA_SCALES)
def test_linear_gamma_endpoints_derivatives_and_bounded_gamma_dot(c):
    s = get_schedule('linear_gamma', c)
    t0, t1 = torch.tensor(0.0), torch.tensor(1.0)
    assert float(s.gamma_sq(t0)) == 0.0 and float(s.gamma_sq(t1)) == 0.0
    assert float(s.Gamma(t0, SIGMA)) == pytest.approx(SIGMA)
    assert float(s.gamma_sq(torch.tensor(0.5)).sqrt()) == pytest.approx(c / 4)      # peak gamma = c/4
    t = torch.linspace(0.02, 0.98, 49, dtype=torch.float64)
    h = 1e-6
    fd = lambda f: (f(t + h) - f(t - h)) / (2 * h)
    torch.testing.assert_close(s.gamma_gamma_dot(t), 0.5 * fd(s.gamma_sq), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s.Gamma_dot(t, SIGMA), fd(lambda u: s.Gamma(u, SIGMA)), rtol=1e-5, atol=1e-6)
    # Unlike the bridge, Gamma_dot stays bounded up to t = 1 (gamma vanishes linearly, like alpha).
    tt = torch.tensor([0.99, 0.999, 0.9999], dtype=torch.float64)
    assert float(s.Gamma_dot(tt, SIGMA).abs().max()) < 2 * (SIGMA ** 2 + c ** 2) ** 0.5


def test_get_schedule_rejects_misplaced_gamma_scale():
    with pytest.raises(ValueError):
        get_schedule('linear', 1.0)
    with pytest.raises(ValueError):
        get_schedule('linear_gamma', 0.0)


NOISE_CASES = ([('linear_gamma', c, SIGMA, sv, e) for c in GAMMA_SCALES for sv in SOLVERS
                for e in ((0.0, 1.0, 5.0) if sv in SDE_SOLVERS else (0.0,))]
               + [('linear', 0.0, sg, sv, e) for sg in (0.25, 0.5) for sv in ('euler', 'heun', 'rk4')
                  for e in ((0.0, 5.0) if sv in SDE_SOLVERS else (0.0,))])


@pytest.mark.parametrize('sched,c,sigma,solver,eps', NOISE_CASES)
def test_si_noise_arms_transport_base_to_target(sched, c, sigma, solver, eps):
    g = torch.Generator().manual_seed(1)
    s = get_schedule(sched, c)
    x0 = M + sigma * torch.randn(40_000, dtype=torch.float64, generator=g)
    x1, _ = integrate(_exact_field_sigma(s, sigma), s, x0, torch.full_like(x0, M), sigma, solver=solver,
                      n_steps=200, eps=eps, generator=g)
    assert float(x1.mean()) == pytest.approx(MU1, abs=0.02)
    assert float(x1.std()) == pytest.approx(S1, rel=0.03)


@pytest.mark.parametrize('solver', SDE_SOLVERS)
@pytest.mark.parametrize('eps', (2.0, 5.0, 10.0, 20.0))
def test_phase0_large_eps_is_stable_at_the_experiments_20_steps(solver, eps):
    """Phase 0 samples the existing linear model with eps up to 20 at 20 steps: no blow-up, and the
    SDE's std is no further from the target than the eps = 0 ODE's (more eps corrects Euler's
    20-step under-dispersion: 0.59 at eps 0 -> 0.68 at eps 20, target 0.70)."""
    s = get_schedule('linear')
    stds = {}
    for e in (0.0, eps):
        g = torch.Generator().manual_seed(1)
        x0 = M + SIGMA * torch.randn(40_000, dtype=torch.float64, generator=g)
        x1, _ = integrate(_exact_field(s), s, x0, torch.full_like(x0, M), SIGMA, solver=solver,
                          n_steps=20, eps=e, generator=g)
        assert torch.isfinite(x1).all()
        assert float(x1.mean()) == pytest.approx(MU1, abs=0.02)
        stds[e] = float(x1.std())
    assert abs(stds[eps] - S1) <= abs(stds[0.0] - S1) + 0.01


def _exact_field_sigma(schedule, sigma):
    def field(x, t):
        c = schedule.coeffs(torch.tensor(float(t), dtype=x.dtype), sigma)
        mean = c['alpha'] * M + c['beta'] * MU1
        var = c['beta'] ** 2 * S1 ** 2 + c['Gamma'] ** 2
        return (c['alpha_dot'] * M + c['beta_dot'] * (MU1 + c['beta'] * S1 ** 2 / var * (x - mean)),
                c['Gamma'] / var * (x - mean))
    return field


def _skill(pred, gt):
    from diffusionsr.analysis.evaluate_si_schedules import ensemble_skill
    return ensemble_skill(pred[:, :, None], gt[:, None], threshold=-1e9)   # every pixel is "melt"


def test_ensemble_skill_on_a_consistent_ensemble():
    """Truth and K members iid N(0, s^2): SSR = 1, flat rank histogram, and the fair CRPS equals the
    exact expected CRPS of N(0, s^2) against y ~ N(0, s^2), which is s / sqrt(pi)."""
    rng = np.random.default_rng(0)
    s, N, K = 50.0, 2000, 10
    gt = rng.normal(0, s, (N, 8, 8))
    pred = rng.normal(0, s, (N, K, 8, 8))
    out = _skill(pred, gt)
    assert out['ssr_all'] == pytest.approx(1.0, abs=0.02)
    assert out['crps_all_K'] == pytest.approx(s / math.sqrt(math.pi), rel=0.02)
    h = np.array(out['rank_hist_all']) / sum(out['rank_hist_all'])
    assert np.abs(h - 1 / (K + 1)).max() < 0.01
    assert out['crps_melt_K'] == out['crps_all_K']


def test_ensemble_skill_flags_under_dispersion_and_matches_mae_for_k1():
    rng = np.random.default_rng(1)
    gt = rng.normal(0, 50.0, (2000, 8, 8))
    tight = rng.normal(0, 10.0, (2000, 10, 8, 8))           # members far too close together
    out = _skill(tight, gt)
    assert out['ssr_all'] < 0.3
    h = np.array(out['rank_hist_all'])
    assert h[0] + h[-1] > 0.6 * h.sum()                      # U-shape: truth outside the ensemble
    point = rng.normal(0, 50.0, (2000, 1, 8, 8))
    out1 = _skill(point, gt)
    assert out1['crps_all_K'] == pytest.approx(float(np.abs(point[:, 0] - gt).mean()), rel=1e-9)
    assert math.isnan(out1['ssr_all'])
