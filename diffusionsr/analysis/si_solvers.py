"""
Integrators for the scheduled stochastic interpolant (runners/si_schedules.py).

Every solver integrates t: 0 -> T_END with the drift/diffusion defined ONCE in
si_schedules.drift_and_diffusion, then takes the shared final denoising jump at T_END. So the solver is
the only thing that changes between solver arms.

    solver     order (ODE)   network evals            eps > 0 ?
    ---------  -----------   ----------------------   ---------------------------------------------
    euler      1             n_steps + 1              yes: Euler-Maruyama
    heun       2             2 n_steps + 1            yes: stochastic Heun (additive-noise predictor-
                                                           corrector; reduces to Heun at eps = 0)
    rk4        4             4 n_steps + 1            no  (ODE only)
    dopri5     5(4) adapt.   adaptive (+1)            no  (ODE only), rtol = atol = 1e-4 as before

The "+1" is the final denoising evaluation. The number of network evaluations actually spent is
returned, so the results can be compared at matched compute as well as at matched step count.

The field passed in is a callable  field(x, t) -> (v_hat, eta_hat)  that wraps the U-Net and the fixed
conditioning; it knows nothing about solvers, and solvers know nothing about U-Nets.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from diffusionsr.runners.si_schedules import T_END, Schedule, denoise_to_target, drift_and_diffusion

Field = Callable[[torch.Tensor, float], tuple]

SOLVERS = ('euler', 'heun', 'rk4', 'dopri5')
SDE_SOLVERS = ('euler', 'heun')
SOLVER_LABEL = {'euler': 'Euler / Euler-Maruyama', 'heun': 'Heun / stochastic Heun',
                'rk4': 'RK4', 'dopri5': 'Dopri5 (adaptive)'}


class _Counted:
    """Wrap a field and count calls (one call = one U-Net forward pass)."""

    def __init__(self, field: Field):
        self.field, self.nfe = field, 0

    def __call__(self, x, t):
        self.nfe += 1
        return self.field(x, t)


def _noise(x, generator):
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


@torch.no_grad()
def integrate(field: Field, schedule: Schedule, x0: torch.Tensor, m: torch.Tensor, sigma: float,
              solver: str = 'heun', n_steps: int = 20, eps: float = 0.0,
              t_end: float = T_END, generator: torch.Generator | None = None,
              rtol: float = 1e-4, atol: float = 1e-4) -> tuple[torch.Tensor, int]:
    """Integrate from X_0 = x0 to t_end, then denoise to t = 1. Returns (x1_hat, nfe)."""
    if solver not in SOLVERS:
        raise ValueError(f'Unknown solver={solver!r}; choose from {SOLVERS}')
    if eps < 0:
        raise ValueError(f'eps must be >= 0, got {eps}')
    if eps > 0 and solver not in SDE_SOLVERS:
        raise ValueError(f'{solver} is an ODE solver and cannot integrate the SDE (eps={eps}); '
                         f'use one of {SDE_SOLVERS}')

    f_ = _Counted(field)

    def fg(x, t):
        v, eta = f_(x, t)
        return drift_and_diffusion(schedule, v, eta, t, sigma, eps)

    x = x0.clone()
    if solver == 'dopri5':
        x = _dopri5(fg, x, t_end, rtol, atol)
    else:
        dt = t_end / n_steps
        sq = math.sqrt(dt)
        for i in range(n_steps):
            t = i * dt
            if solver == 'euler':
                f1, g1 = fg(x, t)
                x = x + f1 * dt + (g1 * sq * _noise(x, generator) if g1 else 0.0)
            elif solver == 'heun':
                f1, g1 = fg(x, t)
                dW = sq * _noise(x, generator) if eps > 0 else None
                x_pred = x + f1 * dt + (g1 * dW if dW is not None else 0.0)
                f2, g2 = fg(x_pred, t + dt)
                x = x + 0.5 * (f1 + f2) * dt + (0.5 * (g1 + g2) * dW if dW is not None else 0.0)
            elif solver == 'rk4':
                k1, _ = fg(x, t)
                k2, _ = fg(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3, _ = fg(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4, _ = fg(x + dt * k3, t + dt)
                x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Final denoising jump at t_end (one more network evaluation).
    _, eta = f_(x, t_end)
    x1 = denoise_to_target(schedule, x, m, eta, t_end, sigma)
    return x1, f_.nfe


# Dormand-Prince 5(4) tableau. The 4th-order weight for stage 3 is 7571/16695; sample_coupled.py had
# 7571/57600, which left the error weights summing to -0.32 instead of 0, so the error estimate was O(dt)
# and the controller shrank the step forever (~15k evaluations on a smooth 1-D test). Fixed there too.
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0)
_A = ((),
      (1 / 5,),
      (3 / 40, 9 / 40),
      (44 / 45, -56 / 15, 32 / 9),
      (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
      (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656))
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)
_E = (5179 / 57600 - 35 / 384, 0.0, 7571 / 16695 - 500 / 1113, 393 / 640 - 125 / 192,
      -92097 / 339200 + 2187 / 6784, 187 / 2100 - 11 / 84, 1 / 40)


def _dopri5(fg, x, t_end, rtol, atol, dt0=0.05, min_dt=1e-5, max_dt=0.5):
    """Adaptive Dormand-Prince with FSAL; same error norm and step controller as DopRK45Sampler."""
    t, dt = 0.0, dt0
    k1, _ = fg(x, t)
    while t < t_end - 1e-9:
        dt = max(min(dt, t_end - t, max_dt), min_dt)
        ks = [k1]
        for i in range(1, 6):
            xi = x + dt * sum(a * k for a, k in zip(_A[i], ks))
            ks.append(fg(xi, t + _C[i] * dt)[0])
        x5 = x + dt * sum(b * k for b, k in zip(_B5, ks))
        k7, _ = fg(x5, t + dt)
        err = dt * (sum(e * k for e, k in zip(_E[:6], ks)) + _E[6] * k7).abs()
        scale = atol + rtol * torch.maximum(x.abs(), x5.abs())
        err_norm = float((err / scale).pow(2).mean().sqrt())
        if err_norm <= 1.0 or dt <= min_dt:
            x, t, k1 = x5, t + dt, k7          # accept; FSAL reuses k7 as the next k1
        dt = dt * max(0.1, min(10.0, 0.9 * max(err_norm, 1e-10) ** -0.2))
    return x
