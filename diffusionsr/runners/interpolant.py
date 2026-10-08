"""
Core stochastic-interpolant math for Options A and B.

Reference: Albergo, Goldstein, Boffi, Ranganath, Vanden-Eijnden,
  "Stochastic Interpolants with Data-Dependent Couplings", ICML 2024.
  https://arxiv.org/abs/2310.03725

All functions are stateless and operate on (B, C, H, W) float tensors
unless stated otherwise.  No neural-network code lives here.

Notation
--------
x0   : base sample (start of transport), in HF-stat space.
       x0 = ulr_in_hr + sigma * zeta, where ulr_in_hr is the bicubic-
       upscaled LF field re-normalized to HF statistics, and zeta ~ N(0,I).
x1   : target (ground truth HF field), in HF-stat space.
t    : interpolation time, scalar or (B,) tensor in [0, 1].
z    : Option-B interpolant noise, z ~ N(0, I), independent of zeta.
gamma: Option-B noise schedule, scalar function of t.

Option A (gamma = 0):
  I_t = (1-t)*x0 + t*x1                    (straight-line path)
  dI_t/dt = x1 - x0                        (constant velocity along path)
  Loss: E[|b_hat(I_t, t, xi) - (x1-x0)|^2]

Option B (gamma > 0):
  I_t = (1-t)*x0 + t*x1 + gamma(t)*z       (noisy path)
  dI_t/dt = x1 - x0 + gamma_dot(t)*z
  Two output heads: b_hat (velocity), g_hat (denoiser = E[z|I_t])
  Loss: E[|b_hat - (x1-x0+gamma_dot*z)|^2] + E[|g_hat - z|^2]
  Score: s(I_t, t) = -g_hat / gamma(t)      (used in SDE sampler)
"""

import torch


# ---------------------------------------------------------------------------
# Re-normalization: upscaled_lr stats → HF stats
# ---------------------------------------------------------------------------

def ulr_to_hr_space(
    upscaled_lr: torch.Tensor,
    mean_ulr: torch.Tensor,
    std_ulr: torch.Tensor,
    mean_hr: torch.Tensor,
    std_hr: torch.Tensor,
) -> torch.Tensor:
    """
    Re-normalize upscaled_lr from its own statistics to the HF statistics.

    The SI requires that the source (x0) and target (x1) live in the same
    normalized space.  The dataset normalizes them separately, so we undo
    upscaled_lr's normalization and apply HF normalization instead.

    All stat tensors should be (C, H, W) or broadcastable to (B, C, H, W).
    """
    # Step 1: unscale from upscaled_lr-stat space → physical units (Kelvin)
    phys = upscaled_lr * std_ulr + mean_ulr
    # Step 2: rescale into HF-stat space
    return (phys - mean_hr) / std_hr


# ---------------------------------------------------------------------------
# Base sample construction
# ---------------------------------------------------------------------------

def make_x0(
    ulr_hr: torch.Tensor,
    sigma: float,
    zeta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build the base sample x0 = U(x_lf) + sigma * zeta.

    ulr_hr  : (B, C, H, W) — upscaled LF field in HF-stat space.
    sigma   : float — noise level.  sigma=0 gives a deterministic base.
    zeta    : optional pre-sampled N(0,I) noise; sampled fresh if None.

    A fresh zeta is drawn every call so each ensemble member sees different
    noise at both training time and sampling time.
    """
    if zeta is None:
        zeta = torch.randn_like(ulr_hr)
    return ulr_hr + sigma * zeta


# ---------------------------------------------------------------------------
# Option A: linear interpolant, no path noise
# ---------------------------------------------------------------------------

def interpolate_A(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute I_t = (1-t)*x0 + t*x1 for Option A.

    t may be shape (B,) or scalar; it is broadcast over spatial dims.
    """
    # Reshape t for broadcasting: (B,) → (B, 1, 1, 1)
    if t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    return (1.0 - t) * x0 + t * x1


def velocity_target_A(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Velocity target for Option A: dI_t/dt = x1 - x0.

    This is constant along the straight-line path (independent of t).
    """
    return x1 - x0


# ---------------------------------------------------------------------------
# Option B: noisy interpolant, gamma(t) schedule
# ---------------------------------------------------------------------------

def gamma(t: torch.Tensor, c: float) -> torch.Tensor:
    """
    Noise schedule for Option B: gamma(t) = c * 4t(1-t).

    Properties:
      - gamma(0) = gamma(1) = 0  →  interpolant is exact at endpoints.
      - gamma(0.5) = c            →  peak noise at mid-path.
      - Smooth, so gamma_dot is bounded everywhere.

    t may be any broadcastable shape.
    """
    return c * 4.0 * t * (1.0 - t)


def gamma_dot(t: torch.Tensor, c: float) -> torch.Tensor:
    """
    Time derivative: d/dt [c * 4t(1-t)] = 4c(1 - 2t).
    """
    return 4.0 * c * (1.0 - 2.0 * t)


def interpolate_B(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    z: torch.Tensor,
    c: float,
) -> torch.Tensor:
    """
    Noisy interpolant: I_t = (1-t)*x0 + t*x1 + gamma(t)*z.

    z ~ N(0, I) is drawn independently of zeta (the noise in x0).
    """
    if t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    gam = gamma(t, c)
    return (1.0 - t) * x0 + t * x1 + gam * z


def velocity_target_B(
    x0: torch.Tensor,
    x1: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    c: float,
) -> torch.Tensor:
    """
    Full velocity target for Option B: dI_t/dt = (x1 - x0) + gamma_dot(t) * z.
    """
    if t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    return (x1 - x0) + gamma_dot(t, c) * z


# ---------------------------------------------------------------------------
# Option B: SDE drift correction and noise
# ---------------------------------------------------------------------------

def sde_eps(t: torch.Tensor, eps0: float, c: float) -> torch.Tensor:
    """
    Time-dependent diffusion coefficient for the forward SDE sampler.

    eps(t) = eps0 * gamma(t) / c = eps0 * 4t(1-t).

    Crucially, eps(t)/gamma(t) = eps0/c = constant, which avoids
    dividing by gamma = 0 at t = 0 and t = 1 during the SDE step.

    eps0 = 0 recovers the deterministic ODE (identical network, no retraining).
    """
    return eps0 * 4.0 * t * (1.0 - t)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def si_loss_A(
    b_hat: torch.Tensor,
    velocity_tgt: torch.Tensor,
    loss_type: str = 'l2',
) -> torch.Tensor:
    """MSE / Huber / L1 between predicted and target velocity (Option A)."""
    import torch.nn.functional as F
    if loss_type == 'l2':
        return F.mse_loss(b_hat, velocity_tgt)
    elif loss_type == 'l1':
        return F.l1_loss(b_hat, velocity_tgt)
    elif loss_type == 'huber':
        return F.smooth_l1_loss(b_hat, velocity_tgt)
    else:
        raise ValueError(f'Unknown loss_type={loss_type!r}')


def si_loss_B(
    b_hat: torch.Tensor,
    g_hat: torch.Tensor,
    velocity_tgt: torch.Tensor,
    z: torch.Tensor,
    loss_type: str = 'l2',
) -> torch.Tensor:
    """
    Combined velocity + denoiser loss (Option B).

    L = E[|b_hat - dI_t|^2] + E[|g_hat - z|^2]

    b_hat predicts the full velocity (including the gamma_dot*z term).
    g_hat predicts z, the raw interpolant noise (score = -g_hat/gamma).
    Both terms are equally weighted; the paper does not prescribe a ratio.
    """
    import torch.nn.functional as F
    loss_fn = {'l2': F.mse_loss, 'l1': F.l1_loss, 'huber': F.smooth_l1_loss}[loss_type]
    return loss_fn(b_hat, velocity_tgt) + loss_fn(g_hat, z)
