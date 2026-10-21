"""Tests for the melt-pool metrics in `analysis/evaluate_coupled.py`.

These pin a bug that made every physics metric a constant. `get_profile` pads columns without melt
with the column height, and the old metrics took max()/len() over that padded profile, so depth was
the padding value and width the whole domain for prediction and ground truth alike: MP/KH/depth/area
MAE were always 0 and calibration always 1, in every arm.

Fields are built in the orientation the metrics expect: temp.T has rows = z (free surface at HIGH z,
ambient gas above) and columns = x. So temp[x, z].
"""

import numpy as np  # Synthetic temperature fields
import pytest  # Approximate comparisons

from diffusionsr.analysis.evaluate_coupled import _mp_geometry, _phys_metrics, compute_metrics

AMBIENT, MELT = 293.0, 3000.0
H = W = 60
PX = 10.0  # um per pixel


def _field(melt_cols, top, bottom):
    """temp[x, z] with melt in columns melt_cols spanning z = bottom..top[x] inclusive."""
    t = np.full((W, H), AMBIENT)
    for x in melt_cols:
        t[x, bottom[x]:top[x] + 1] = MELT
    return t


def test_no_melt_gives_zero_geometry():
    g = _mp_geometry(np.full((W, H), AMBIENT), PX)
    assert g == {'mp_depth': 0.0, 'kh_depth': 0.0, 'width': 0.0, 'area': 0.0}


def test_flat_pool_depth_width_area():
    cols = range(20, 30)
    top = {x: 40 for x in cols}       # flat free surface at z = 40
    bottom = {x: 35 for x in cols}    # pool bottom at z = 35 -> 6 pixels deep
    g = _mp_geometry(_field(cols, top, bottom), PX)
    assert g['mp_depth'] == pytest.approx(6 * PX)
    assert g['kh_depth'] == pytest.approx(0.0)          # flat surface, no depression
    assert g['width'] == pytest.approx(10 * PX)
    assert g['area'] == pytest.approx(10 * 6 * PX ** 2)


def test_keyhole_depression_is_measured_from_the_surface():
    cols = range(20, 30)
    top = {x: 40 for x in cols}
    top[25] = 30                      # keyhole: the cavity above z = 30 is cold, like real frames
    bottom = {x: 35 for x in cols}
    bottom[25] = 22                   # the pool reaches deepest under the keyhole
    g = _mp_geometry(_field(cols, top, bottom), PX)
    assert g['kh_depth'] == pytest.approx(10 * PX)      # surface 40 - keyhole bottom 30
    assert g['mp_depth'] == pytest.approx(19 * PX)      # surface 40 - deepest melt 22 + 1


def test_metrics_now_distinguish_prediction_from_ground_truth():
    cols = range(20, 30)
    gt = _field(cols, {x: 40 for x in cols}, {x: 35 for x in cols})       # 6 px deep
    pred = _field(cols, {x: 40 for x in cols}, {x: 32 for x in cols})     # 9 px deep
    m = _phys_metrics(pred, gt, PX)
    assert m['mp_mae'] == pytest.approx(3 * PX)        # was always 0 before the fix
    assert m['area_p'] != m['area_g']


def test_calibration_ignores_frames_without_melt():
    cols = range(20, 30)
    melt_gt = _field(cols, {x: 40 for x in cols}, {x: 35 for x in cols})
    cold = np.full((W, H), AMBIENT)
    # Frame 0 has melt, but every sample is far too deep with no spread -> outside mean +- 2 std.
    # Frame 1 has no melt anywhere; it must not count as a calibrated hit.
    deep = _field(cols, {x: 40 for x in cols}, {x: 20 for x in cols})
    preds = np.stack([np.stack([deep] * 4), np.stack([cold] * 4)])[:, :, None]   # (N=2, K=4, C=1)
    gts = np.stack([melt_gt, cold])[:, None]                                     # (N=2, C=1)
    r = compute_metrics(preds, gts, PX)
    assert r['n_melt_frames'] == 1
    assert r['calibration_2sigma'] == pytest.approx(0.0)   # old code: 0.5 from the no-melt frame
