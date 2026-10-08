"""
Samplers for the coupled stochastic-interpolant model.

Three integrators are implemented here from scratch (no torchdiffeq dependency):

  HeunSampler    — fixed-step 2nd-order Runge-Kutta (Heun's method) for the ODE.
                   Used as the primary speed/quality trade-off sampler (N=10/20/50).
                   Also used for Option B with eps0=0 (deterministic ODE path).

  DopRK45Sampler — Dormand-Prince adaptive step-size RK4(5) for the ODE.
                   Used as the high-accuracy reference.  No external library needed:
                   the 6-stage Butcher tableau is hard-coded below.

  EMSampler      — Euler-Maruyama for the Option-B forward SDE.
                   Uses the time-dependent diffusion coefficient
                   eps(t) = eps0 * 4t(1-t) so noise vanishes at the endpoints.

All samplers share the same call signature so they are interchangeable:
    imgs = sampler.sample(model, x0, x_e, ...)

Reference: Albergo, Goldstein, Boffi, Ranganath, Vanden-Eijnden,
  "Stochastic Interpolants with Data-Dependent Couplings", ICML 2024.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Type alias for clarity
# ---------------------------------------------------------------------------
# A velocity function: (x_t, t_scalar_float, x_e) -> dx/dt with shape (B,C,H,W)
VelocityFn = Callable[[torch.Tensor, float, torch.Tensor | None], torch.Tensor]


# ---------------------------------------------------------------------------
# Heun sampler (fixed-step, 2nd-order Runge-Kutta)
# ---------------------------------------------------------------------------

class HeunSampler:
    """
    2nd-order Runge-Kutta (Heun's method) ODE integrator.

    Heun is the standard step in flow-matching literature because it is
    2nd-order accurate with only 2 network evaluations per step:
      k1 = f(x_t,     t)
      k2 = f(x_t+k1, t+dt)   (predictor step)
      x_{t+dt} = x_t + 0.5*(k1 + k2)*dt   (corrector)

    Integrates from t=0 (x0 = data-dependent base) to t=1 (HF prediction).
    """

    def __init__(self, fm_timescale: float = 1000.0):
        self.fm_timescale = fm_timescale

    def _velocity(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t_scalar: float,
        x_e: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Query the velocity network b_hat(x_t, t, x_e).

        The U-Net expects the time embedded as t * fm_timescale (float),
        matching how FlowMatchingModel passes time during training.
        """
        B = x.shape[0]
        device = x.device
        t_batch = torch.full((B,), t_scalar * self.fm_timescale, device=device)
        return model(x, t_batch, x_e)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x_e: torch.Tensor | None,
        n_steps: int = 20,
        return_trajectory: bool = False,
    ) -> list[torch.Tensor]:
        """
        Integrate dx/dt = b_hat(x, t, x_e) from t=0 to t=1.

        Returns a list of tensors.  If return_trajectory=False, only the
        final x (at t=1) is returned as a length-1 list, which is enough
        for evaluation.  If True, every intermediate step is returned
        (useful for visualising the transport).
        """
        model.eval()
        x = x0.clone()
        dt = 1.0 / n_steps
        imgs = []

        for i in range(n_steps):
            t = i * dt  # current time: steps 0, dt, 2dt, ..., (n-1)*dt

            # Predictor (Euler step)
            k1 = self._velocity(model, x, t, x_e)
            x_pred = x + k1 * dt

            # Corrector (average the two velocities)
            k2 = self._velocity(model, x_pred, t + dt, x_e)
            x = x + 0.5 * (k1 + k2) * dt

            if return_trajectory or i == n_steps - 1:
                imgs.append(x.cpu())

        return imgs


# ---------------------------------------------------------------------------
# Dormand-Prince RK4(5) adaptive sampler
# ---------------------------------------------------------------------------

class DopRK45Sampler:
    """
    Dormand-Prince RK4(5) adaptive step-size ODE integrator.

    This is the same algorithm behind scipy's solve_ivp(method='RK45') and
    torchdiffeq's dopri5.  It uses two embedded Runge-Kutta formulas of
    orders 4 and 5 to estimate the local truncation error and adapt dt.

    The Butcher tableau (Dormand-Prince 1980):
      c  = [0,    1/5,   3/10,  4/5,   8/9,   1,   1]
      a21 = 1/5
      a31 = 3/40,        a32 = 9/40
      a41 = 44/45,       a42 = -56/15,    a43 = 32/9
      a51 = 19372/6561,  a52 = -25360/2187, a53 = 64448/6561, a54 = -212/729
      a61 = 9017/3168,   a62 = -355/33,   a63 = 46732/5247,  a64 = 49/176, a65 = -5103/18656
      b  = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]       (5th order)
      b* = [5179/57600, 0, 7571/57600, 393/640, -92097/339200, 187/2100, 1/40]  (4th order, for error)

    Note: stages k1-k6 plus k7=k1(next) in the FSAL property (first-same-as-last).
    For simplicity we do not exploit FSAL here — one extra call per step.
    """

    # Dormand-Prince Butcher tableau constants
    _c2, _c3, _c4, _c5 = 1/5, 3/10, 4/5, 8/9
    _a21 = 1/5
    _a31, _a32 = 3/40, 9/40
    _a41, _a42, _a43 = 44/45, -56/15, 32/9
    _a51, _a52, _a53, _a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    _a61, _a62, _a63, _a64, _a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    # 5th-order solution weights
    _b1, _b3, _b4, _b5, _b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
    # 4th-order embedded solution weights (for error estimate)
    _e1, _e3, _e4, _e5, _e6, _e7 = (
        5179/57600 - 35/384,
        7571/57600 - 500/1113,
        393/640   - 125/192,
        -92097/339200 - (-2187/6784),
        187/2100  - 11/84,
        1/40,
    )

    def __init__(
        self,
        fm_timescale: float = 1000.0,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        min_dt: float = 1e-5,
        max_dt: float = 0.5,
    ):
        self.fm_timescale = fm_timescale
        self.rtol = rtol
        self.atol = atol
        self.min_dt = min_dt
        self.max_dt = max_dt

    def _velocity(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t_scalar: float,
        x_e: torch.Tensor | None,
    ) -> torch.Tensor:
        B = x.shape[0]
        t_batch = torch.full((B,), t_scalar * self.fm_timescale, device=x.device)
        return model(x, t_batch, x_e)

    def _step(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: float,
        dt: float,
        x_e: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        One Dormand-Prince RK4(5) step from t to t+dt.

        Returns:
          x5    : 5th-order solution (used to advance state)
          err   : (B,C,H,W) element-wise error estimate |x5 - x4|
          dt    : the dt that was actually used (same as input here; adaptive
                  step rejection happens in .sample())
        """
        k1 = self._velocity(model, x, t, x_e)
        k2 = self._velocity(model, x + dt * self._a21 * k1,                         t + dt * self._c2, x_e)
        k3 = self._velocity(model, x + dt * (self._a31*k1 + self._a32*k2),          t + dt * self._c3, x_e)
        k4 = self._velocity(model, x + dt * (self._a41*k1 + self._a42*k2 + self._a43*k3), t + dt * self._c4, x_e)
        k5 = self._velocity(model, x + dt * (self._a51*k1 + self._a52*k2 + self._a53*k3 + self._a54*k4), t + dt * self._c5, x_e)
        k6 = self._velocity(model, x + dt * (self._a61*k1 + self._a62*k2 + self._a63*k3 + self._a64*k4 + self._a65*k5), t + dt, x_e)

        # 5th-order solution (used to advance)
        x5 = x + dt * (self._b1*k1 + self._b3*k3 + self._b4*k4 + self._b5*k5 + self._b6*k6)

        # Error estimate (difference between 5th and 4th order solutions)
        k7 = self._velocity(model, x5, t + dt, x_e)   # FSAL
        err = dt * torch.abs(self._e1*k1 + self._e3*k3 + self._e4*k4 + self._e5*k5 + self._e6*k6 + self._e7*k7)

        return x5, err, dt

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x_e: torch.Tensor | None,
        return_trajectory: bool = False,
    ) -> list[torch.Tensor]:
        """
        Integrate from t=0 to t=1 with adaptive step size.
        """
        model.eval()
        x = x0.clone()
        t = 0.0
        dt = 0.1   # initial step guess
        imgs = []

        while t < 1.0 - 1e-8:
            # Clamp so we do not overshoot t=1
            dt = min(dt, 1.0 - t, self.max_dt)
            dt = max(dt, self.min_dt)

            x5, err, dt_used = self._step(model, x, t, dt, x_e)

            # Error norm: RMS over all elements (scalar)
            scale = self.atol + self.rtol * torch.maximum(x.abs(), x5.abs())
            err_norm = float((err / scale).pow(2).mean().sqrt())

            if err_norm <= 1.0 or dt <= self.min_dt:
                # Accept step
                x = x5
                t += dt_used
                if return_trajectory:
                    imgs.append(x.cpu())

            # Adjust step size: standard PI controller step-size factor
            # Factor = 0.9 * (1 / err_norm)^(1/5), clamped to [0.1, 10]
            factor = 0.9 * (max(err_norm, 1e-10) ** (-0.2))
            factor = max(0.1, min(10.0, factor))
            dt = dt_used * factor

        if not return_trajectory:
            imgs.append(x.cpu())
        return imgs


# ---------------------------------------------------------------------------
# Euler-Maruyama SDE sampler (Option B)
# ---------------------------------------------------------------------------

class EMSampler:
    """
    Euler-Maruyama integrator for the Option-B forward SDE:

      dX = [b_hat(X,t) - eps(t) * g_hat(X,t) / gamma(t)] dt + sqrt(2*eps(t)) dW

    where:
      eps(t)         = eps0 * 4t(1-t)        (vanishes at endpoints)
      eps(t)/gamma(t) = eps0 / c              (constant, avoids 0/0)
      gamma(t)       = c * 4t(1-t)
      b_hat          = first C channels of model output
      g_hat          = last  C channels of model output

    When eps0=0, the SDE reduces to the deterministic ODE (same network weights),
    so this sampler handles both the ODE and SDE cases.

    N ∈ {50, 100, 200} steps recommended from the spec.
    """

    def __init__(self, fm_timescale: float = 1000.0, gamma_c: float = 0.25):
        self.fm_timescale = fm_timescale
        self.gamma_c = gamma_c   # the c in gamma(t) = c*4t(1-t)

    def _query(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t_scalar: float,
        x_e: torch.Tensor | None,
        C: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the dual-head model and return (b_hat, g_hat).

        The model outputs 2*C channels: first C are b_hat (velocity),
        last C are g_hat (denoiser).
        """
        B = x.shape[0]
        t_batch = torch.full((B,), t_scalar * self.fm_timescale, device=x.device)
        out = model(x, t_batch, x_e)   # (B, 2C, H, W)
        return out[:, :C], out[:, C:]

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x_e: torch.Tensor | None,
        n_steps: int = 100,
        eps0: float = 0.25,
        return_trajectory: bool = False,
    ) -> list[torch.Tensor]:
        """
        Integrate the forward SDE from t=0 to t=1.

        eps0 can be swept at inference time without retraining (only the
        network weights and the number of steps are fixed at training time).
        eps0 = 0 gives the deterministic ODE.
        """
        model.eval()
        x = x0.clone()
        C = x.shape[1]
        dt = 1.0 / n_steps
        imgs = []
        c = self.gamma_c

        for i in range(n_steps):
            t = i * dt

            # Avoid t=0 and t=1 where gamma=0 (score would blow up).
            # In practice eps(t)=0 there anyway, so the score term vanishes.
            t_safe = t + 1e-6 if t < 1e-6 else t

            b_hat, g_hat = self._query(model, x, t_safe, x_e, C)

            # Compute eps(t) = eps0 * 4t(1-t).
            # Note: eps(t)/gamma(t) = eps0/c (constant), so we never divide
            # by gamma directly.  The score correction becomes:
            #   eps(t) * g_hat / gamma(t) = (eps0/c) * g_hat
            eps_over_gamma = eps0 / c if c > 0 else 0.0

            # Deterministic drift component
            drift = b_hat - eps_over_gamma * g_hat

            # Stochastic diffusion term: sqrt(2*eps(t)) * dW
            # eps(t) = eps0 * 4*t*(1-t)
            eps_t = eps0 * 4.0 * t_safe * (1.0 - t_safe)
            diffusion_coef = math.sqrt(max(2.0 * eps_t, 0.0))
            dW = torch.randn_like(x) * math.sqrt(dt)

            x = x + drift * dt + diffusion_coef * dW

            if return_trajectory or i == n_steps - 1:
                imgs.append(x.cpu())

        return imgs
