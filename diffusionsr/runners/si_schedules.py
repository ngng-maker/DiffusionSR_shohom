"""
Interpolant schedules for the paired-coupling stochastic interpolant (the "SI schedules" study).

Reference: Albergo, Boffi & Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for
Flows and Diffusions" (JMLR 2025, arXiv:2303.08797), and Albergo et al., "Stochastic Interpolants
with Data-Dependent Couplings" (ICML 2024, arXiv:2310.03725).

Convention (the papers', NOT FlowMatchingModel's): t = 0 is the base, t = 1 is the target.

The interpolant
---------------
    x0  = m + sigma * zeta                  paired base: m = enc(LR) or bicubic(LR), in HR-stat space
    I_t = alpha_t * x0 + beta_t * x1 + gamma_t * z
        = alpha_t * m + beta_t * x1 + (alpha_t * sigma * zeta + gamma_t * z)

Given the conditioning (the LR field), m is deterministic, so the two Gaussian noises collapse into ONE
with standard deviation

    Gamma_t = sqrt(alpha_t^2 sigma^2 + gamma_t^2)            ("effective noise scale")
    I_t     = alpha_t * m + beta_t * x1 + Gamma_t * n,        n ~ N(0, I)

which is equal in distribution to the two-noise form, and is how training draws I_t.

What the network learns, and how velocity and score follow from it
------------------------------------------------------------------
Two output heads (one U-Net, out_dim = 2C):

    v_hat   ~ E[ alpha_dot * m + beta_dot * x1 | I_t ]       "signal velocity"
    eta_hat ~ E[ n | I_t ]                                    denoiser

The noise part of dI/dt is alpha_dot*sigma*zeta + gamma_dot*z. Its projection onto n is Gamma_dot*n and
the orthogonal remainder is independent of I_t, so E[noise part | I_t] = Gamma_dot * E[n | I_t]. Hence

    b(x, t)     = v_hat + Gamma_dot * eta_hat                 velocity  (drives the ODE)
    score(x, t) = -eta_hat / Gamma_t                          score     (needed when eps > 0)

Both are exact at the L2 optimum. This decomposition is used instead of regressing b directly because
two schedules have integrable singularities at t = 1 (Gamma_dot -> -inf for the Brownian bridge,
alpha_dot -> -inf for VP). The v target stays bounded on [0, T_END]; the singular coefficient is
known in closed form and only multiplies eta_hat at sampling time.

Gamma_0 = sigma for every schedule (alpha_0 = 1, gamma_0 = 0), and Gamma_1 = 0.

The SDE and the meaning of eps
------------------------------
    dX = [ b + eps_t * score ] dt + sqrt(2 eps_t) dW,     with   eps_t = eps * Gamma_t^2
       = [ v_hat + (Gamma_dot - eps * Gamma_t) * eta_hat ] dt + sqrt(2 eps) * Gamma_t dW

eps_t = eps * Gamma_t^2 makes eps dimensionless and comparable across schedules: per step the SDE
refreshes a fraction ~sqrt(2 eps dt) of the path's own noise, whatever Gamma_t's magnitude, and the
score term eps*Gamma_t*eta_hat never divides by Gamma_t -> 0. eps = 0 is exactly the probability-flow
ODE. eps is a SAMPLING parameter: one trained model covers the whole eps sweep.

Time range
----------
Training samples t ~ U(0, T_END) and sampling integrates 0 -> T_END, then takes one denoising jump
    x1_hat = (X - alpha * m - Gamma * eta_hat) / beta        (Tweedie, at t = T_END)
which removes the residual Gamma(T_END) noise (about 0.045 normalised units for the bridge, ~1e-4 for
the gamma = 0 schedules). The same T_END and final jump are used for every schedule and solver.
"""

from __future__ import annotations

import math

import torch

T_END = 1.0 - 1e-3   # end of integration; see "Time range" above


def _as_tensor(t, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t
    kw = {} if like is None else {'device': like.device, 'dtype': like.dtype}
    return torch.tensor(float(t), **kw)


class Schedule:
    """Base class. Subclasses define alpha, beta, gamma^2 and their time derivatives.

    All methods take t as a tensor of any shape and return tensors of the same shape.
    gamma is described through gamma^2 and d(gamma^2)/dt / 2 = gamma * gamma_dot, which is finite even
    where gamma_dot itself is not (the bridge's sqrt at t = 0).
    """

    name: str = 'base'
    description: str = ''

    def alpha(self, t):          raise NotImplementedError
    def alpha_dot(self, t):      raise NotImplementedError
    def beta(self, t):           raise NotImplementedError
    def beta_dot(self, t):       raise NotImplementedError

    def gamma_sq(self, t):
        return torch.zeros_like(t)

    def gamma_gamma_dot(self, t):
        """gamma_t * d(gamma_t)/dt = d(gamma_t^2)/dt / 2."""
        return torch.zeros_like(t)

    # -- derived -------------------------------------------------------------------------------
    def Gamma(self, t, sigma: float):
        a = self.alpha(t)
        return torch.sqrt(a * a * sigma * sigma + self.gamma_sq(t))

    def Gamma_dot(self, t, sigma: float):
        a = self.alpha(t)
        G = self.Gamma(t, sigma)
        num = a * self.alpha_dot(t) * sigma * sigma + self.gamma_gamma_dot(t)
        # Gamma > 0 on [0, T_END] whenever sigma > 0; the clamp only guards t = 1 exactly.
        return num / G.clamp_min(1e-12)

    def coeffs(self, t, sigma: float) -> dict:
        """Every coefficient the loss and the samplers need, at once."""
        return {
            'alpha': self.alpha(t), 'alpha_dot': self.alpha_dot(t),
            'beta': self.beta(t), 'beta_dot': self.beta_dot(t),
            'Gamma': self.Gamma(t, sigma), 'Gamma_dot': self.Gamma_dot(t, sigma),
        }


class LinearSchedule(Schedule):
    """alpha = 1 - t, beta = t, gamma = 0.  Rectified flow / flow matching (the previous run's Option A)."""
    name = 'linear'
    description = 'Linear / flow matching: alpha=1-t, beta=t, gamma=0'

    def alpha(self, t):     return 1.0 - t
    def alpha_dot(self, t): return -torch.ones_like(t)
    def beta(self, t):      return t
    def beta_dot(self, t):  return torch.ones_like(t)


class TrigSchedule(Schedule):
    """alpha = cos(pi t / 2), beta = sin(pi t / 2), gamma = 0.  Variance-preserving path ('GVP')."""
    name = 'trig'
    description = 'Trigonometric (GVP): alpha=cos(pi t/2), beta=sin(pi t/2), gamma=0'

    def alpha(self, t):     return torch.cos(0.5 * math.pi * t)
    def alpha_dot(self, t): return -0.5 * math.pi * torch.sin(0.5 * math.pi * t)
    def beta(self, t):      return torch.sin(0.5 * math.pi * t)
    def beta_dot(self, t):  return 0.5 * math.pi * torch.cos(0.5 * math.pi * t)


class VPSchedule(Schedule):
    """Variance-preserving diffusion schedule, matched to DiffusionSR's own DDPM.

    DiffusionSR's linear beta schedule (1e-4 .. 0.02 over 1000 steps) is, in continuous time s in [0,1]
    (s = 0 data, s = 1 noise), the VP-SDE with beta(s) = b0 + (b1 - b0) s, b0 = 0.1, b1 = 20, and

        alphabar(s) = exp(-(b0 s + (b1 - b0) s^2 / 2)).

    Here s = 1 - t (our t runs base -> target). The signal coefficient sqrt(alphabar) is rescaled so the
    interpolant hits its endpoints exactly (sqrt(alphabar(1)) = 0.0066, not 0), and alpha is set by
    alpha^2 + beta^2 = 1:

        r(t)  = sqrt(alphabar(1 - t)),    beta = (r - r0) / (1 - r0),    alpha = sqrt(1 - beta^2).
    """
    name = 'vp'
    description = 'VP diffusion (DDPM-matched): beta=sqrt(alphabar), alpha=sqrt(1-beta^2), gamma=0'

    b0, b1 = 0.1, 20.0

    def _r(self, t):
        s = 1.0 - t
        return torch.exp(-0.5 * (self.b0 * s + 0.5 * (self.b1 - self.b0) * s * s))

    @property
    def _r0(self):
        return math.exp(-0.5 * (self.b0 + 0.5 * (self.b1 - self.b0)))

    def beta(self, t):
        return ((self._r(t) - self._r0) / (1.0 - self._r0)).clamp(0.0, 1.0)

    def beta_dot(self, t):
        s = 1.0 - t
        return self._r(t) * 0.5 * (self.b0 + (self.b1 - self.b0) * s) / (1.0 - self._r0)

    def alpha(self, t):
        b = self.beta(t)
        return torch.sqrt((1.0 - b * b).clamp_min(0.0))

    def alpha_dot(self, t):
        # d/dt sqrt(1 - beta^2) = -beta beta_dot / alpha. Integrable singularity at t = 1 only.
        return -self.beta(t) * self.beta_dot(t) / self.alpha(t).clamp_min(1e-12)


class BridgeSchedule(Schedule):
    """alpha = 1 - t, beta = t, gamma = sqrt(2 t (1 - t)).  Brownian bridge (diffusion bridge, I2SB-like)."""
    name = 'bridge'
    description = 'Brownian bridge: alpha=1-t, beta=t, gamma=sqrt(2t(1-t))'

    def alpha(self, t):           return 1.0 - t
    def alpha_dot(self, t):       return -torch.ones_like(t)
    def beta(self, t):            return t
    def beta_dot(self, t):        return torch.ones_like(t)
    def gamma_sq(self, t):        return 2.0 * t * (1.0 - t)
    def gamma_gamma_dot(self, t): return 1.0 - 2.0 * t


SCHEDULES = {cls.name: cls for cls in (LinearSchedule, TrigSchedule, VPSchedule, BridgeSchedule)}


def get_schedule(name: str) -> Schedule:
    try:
        return SCHEDULES[name]()
    except KeyError:
        raise ValueError(f'Unknown si_schedule={name!r}; choose from {sorted(SCHEDULES)}') from None


# ---------------------------------------------------------------------------------------------
# Training target and the sampling-time drift, written once so training and sampling cannot drift
# ---------------------------------------------------------------------------------------------

def bcast(c: torch.Tensor, ndim: int) -> torch.Tensor:
    """(B,) -> (B, 1, 1, 1) for broadcasting over (C, H, W)."""
    return c.view(-1, *([1] * (ndim - 1))) if c.dim() == 1 else c


def make_interpolant(schedule: Schedule, m, x1, t, sigma: float, n=None):
    """Return (I_t, v_target, n) for training. m, x1: (B,C,H,W); t: (B,)."""
    if n is None:
        n = torch.randn_like(x1)
    c = {k: bcast(v, x1.dim()) for k, v in schedule.coeffs(t, sigma).items()}
    I_t = c['alpha'] * m + c['beta'] * x1 + c['Gamma'] * n
    v_target = c['alpha_dot'] * m + c['beta_dot'] * x1
    return I_t, v_target, n


def drift_and_diffusion(schedule: Schedule, v_hat, eta_hat, t: float, sigma: float, eps: float):
    """SDE drift f and scalar diffusion g at time t (eps = 0 gives the ODE drift b and g = 0)."""
    tt = _as_tensor(t, v_hat)
    G = schedule.Gamma(tt, sigma)
    Gd = schedule.Gamma_dot(tt, sigma)
    f = v_hat + (Gd - eps * G) * eta_hat
    g = math.sqrt(2.0 * eps) * float(G) if eps > 0 else 0.0
    return f, g


def denoise_to_target(schedule: Schedule, x, m, eta_hat, t: float, sigma: float):
    """Tweedie jump at time t: x1_hat = (x - alpha m - Gamma eta_hat) / beta."""
    tt = _as_tensor(t, x)
    c = schedule.coeffs(tt, sigma)
    return (x - c['alpha'] * m - c['Gamma'] * eta_hat) / c['beta']
