"""
Sub-pixel melt-pool depth metrics and depth calibration, back-filled from saved predictions.

WHY
---
`evaluate_coupled._mp_geometry` measures depths in whole pixels (10 um at 2x). Every sample's depth is
then a multiple of 10 um, and when all K samples land on the same pixel their std is exactly 0, so the
+/-2 sigma calibration test passes only if the ensemble is exactly right. That made the pixel-based
calibration_2sigma partly a test of the pixel grid rather than of the model: on the SI-schedules linear
model 42% of melted frames had sigma = 0 (DDPM baseline: 22%).

Here the 1900 K isotherm is located to sub-pixel precision by linear interpolation between the last
hot pixel and its cold neighbour, at the melt-pool bottom (per column) and at the top of the hot region
(per column). Depths then vary continuously and the calibration test measures the model.

  mp_depth = surface - deepest bottom crossing     (um)
  kh_depth = surface - lowest top crossing         (um; ~0 with no keyhole)
  surface  = highest top crossing over melted columns
Same definitions as _mp_geometry, with crossings in place of pixel indices; the pixel version's "+1"
corresponds to the half pixel each crossing sits beyond the last hot pixel centre on average.

Written to <dir>/depth_subpx.json; results.json is never touched, so published numbers stay as they are.

    python -m diffusionsr.analysis.add_depth_metrics eval_results/si_schedules/*/ \
        eval_results/_predictions_cache/si_baseline_ddpm_matched_temp
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from diffusionsr.analysis.add_skill_metrics import load_pair

THRESHOLD_K = 1900.0
PX_UM_2X = 10.0


def mp_depth_subpixel(temp: np.ndarray, px_um: float = PX_UM_2X, threshold: float = THRESHOLD_K):
    """temp (..., H, W) physical K, oriented as in _mp_geometry (temp.T rows = z, surface at high z).
    Returns (mp_depth, kh_depth, melted), each of shape (...); depths in um, 0 where nothing melted."""
    A = np.swapaxes(np.asarray(temp, dtype=np.float64), -1, -2)       # (..., Z, X)
    Z = A.shape[-2]
    hot = A > threshold
    col_melt = hot.any(-2)                                             # (..., X)
    b = hot.argmax(-2)                                                 # lowest hot row per column
    t = Z - 1 - hot[..., ::-1, :].argmax(-2)                           # highest hot row per column

    def take(idx):
        return np.take_along_axis(A, idx[..., None, :], -2)[..., 0, :]

    # Bottom crossing between row b-1 (<= threshold) and row b (> threshold); domain edge -> half pixel.
    Ab, Abm = take(b), take(np.maximum(b - 1, 0))
    has_below = b > 0
    zb = np.where(has_below, b - (Ab - threshold) / np.where(has_below, Ab - Abm, 1.0), b - 0.5)
    # Top crossing between row t (> threshold) and row t+1 (<= threshold).
    At, Atp = take(t), take(np.minimum(t + 1, Z - 1))
    has_above = t < Z - 1
    zt = np.where(has_above, t + (At - threshold) / np.where(has_above, At - Atp, 1.0), t + 0.5)

    melted = col_melt.any(-1)
    surface = np.where(col_melt, zt, -np.inf).max(-1)
    bottom = np.where(col_melt, zb, np.inf).min(-1)
    kh_top = np.where(col_melt, zt, np.inf).min(-1)
    mp = np.where(melted, surface - bottom, 0.0) * px_um
    kh = np.where(melted, surface - kh_top, 0.0) * px_um
    return mp, kh, melted


def depth_metrics(preds, gts, px_um: float = PX_UM_2X, ch: int = 0, chunk: int = 64) -> dict:
    """preds (N, K, C, H, W), gts (N, C, H, W) physical units -> sub-pixel depth metrics."""
    N, K = preds.shape[:2]
    dp, kp = np.zeros((N, K)), np.zeros((N, K))
    dg, kg, mg = np.zeros(N), np.zeros(N), np.zeros(N, bool)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        dp[s:e], kp[s:e], _ = mp_depth_subpixel(np.asarray(preds[s:e, :, ch]), px_um)
        dg[s:e], kg[s:e], mg[s:e] = mp_depth_subpixel(np.asarray(gts[s:e, ch]), px_um)
    mu, sd = dp[mg].mean(1), dp[mg].std(1)
    err = np.abs(dg[mg] - mu)
    return {
        'MP_MAE_subpx_mean': float(np.abs(dp - dg[:, None]).mean()),
        'KH_MAE_subpx_mean': float(np.abs(kp - kg[:, None]).mean()),
        # Same rule as calibration_2sigma (truth within ensemble mean +/- 2 std, melted GT frames only),
        # with continuous depths. Ideal 0.954 for a Gaussian ensemble; 1 sigma ideal 0.683.
        'calibration_2sigma_subpx': float(np.mean(err <= 2 * sd)),
        'calibration_1sigma_subpx': float(np.mean(err <= sd)),
        'depth_sd_median_subpx_um': float(np.median(sd)),
        'depth_err_median_subpx_um': float(np.median(err)),
        'frac_depth_sd0_subpx': float(np.mean(sd == 0)),
        'n_melt_frames': int(mg.sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description='Back-fill sub-pixel depth metrics into depth_subpx.json')
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--force', action='store_true', help='recompute even if depth_subpx.json exists')
    ap.add_argument('--px_um', type=float, default=PX_UM_2X)
    args = ap.parse_args(argv)
    for d in args.dirs:
        d = d.rstrip('/')
        out = os.path.join(d, 'depth_subpx.json')
        if os.path.exists(out) and not args.force:
            print(f'skip (exists): {out}')
            continue
        pair = load_pair(d)
        if pair is None:
            print(f'skip (no predictions): {d}')
            continue
        m = depth_metrics(*pair, px_um=args.px_um)
        with open(out, 'w') as f:
            json.dump(m, f, indent=2)
        print(f'{d}: cal2s_subpx={m["calibration_2sigma_subpx"]:.3f} MP_subpx={m["MP_MAE_subpx_mean"]:.2f} um')


if __name__ == '__main__':
    main()
