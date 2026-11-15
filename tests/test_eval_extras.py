"""CPU tests for analysis/eval_extras.py and the SI trajectory callback (no data, no GPU)."""
import os

import numpy as np
import pytest
import torch

from diffusionsr.analysis import eval_extras as X
from diffusionsr.analysis.si_solvers import integrate

Z = W = 24


def melt_field(bottom, top=20.6, amb=293.0):
    """(X, Z) temperature with a hot band from z=bottom to z=top in columns 6..17."""
    z = np.arange(Z, dtype=float)
    col = np.clip(1900 + 100 * np.minimum(z - bottom, top - z), amb, None)
    T = np.full((W, Z), amb)
    T[6:18] = col
    return T


class FakeDS:
    """Minimal stand-in for FlippedDataset: 2 conditions x 3 timesteps, standardisation = identity."""
    def __init__(self, n_fields=1):
        self.lr_paths = [f'/d/test/LR/direct/1x/power{P}velocity{V}/_{t}.npy'
                         for P, V in ((100, 500), (300, 200)) for t in (3, 1, 2)]
        self.n_fields = n_fields

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, i):
        f = torch.tensor(np.stack([melt_field(8 + i % 3)] * self.n_fields), dtype=torch.float32)
        return f, f, f, f

    def unscale_data(self, a, input_type='hr'):
        return np.asarray(a)


class FakeModel:
    """sample() adds per-sample noise to the upscaled LR, in n_steps steps, calling the callback."""
    def __init__(self):
        self.model = torch.nn.Linear(3, 4)
        self.lr_enc = torch.nn.Linear(2, 2)

    def compute_x_e(self, lr, ulr):
        return ulr

    def sample(self, ulr, x_e, lr=None, solver='heun', n_steps=4, eps=0.0, generator=None, callback=None):
        x = ulr.clone()
        noise = torch.randn(x.shape, generator=generator) * 30.0
        for i in range(n_steps + 1):
            if callback is not None:
                callback(i, i / n_steps, x + noise * i / n_steps)
        return x + noise, 2 * n_steps + 1


def make_preds(N=6, K=5, seed=0):
    rng = np.random.default_rng(seed)
    gts = np.stack([melt_field(8 + n % 3)[None] for n in range(N)])
    preds = np.stack([np.stack([melt_field(8 + n % 3 + rng.uniform(-0.6, 0.6))[None] for _ in range(K)])
                      for n in range(N)])
    return preds, gts


def test_frame_info_parses_condition_and_timestep():
    info = X.frame_info(FakeDS())
    assert info[0] == dict(P=100, V=500, t=3, cond='P100_V500')
    assert {d['cond'] for d in info} == {'P100_V500', 'P300_V200'}


def test_pv_metrics_per_condition():
    preds, gts = make_preds()
    out = X.pv_metrics(preds, gts, X.frame_info(FakeDS()), px_um=10.0)
    assert out['n_conditions'] == 2
    for r in out['per_condition']:
        assert r['n_frames'] == 3 and r['n_melt'] == 3
        assert 0.0 <= r['calibration_2sigma_subpx'] <= 1.0
        assert r['true_mp_depth_um'] > 0 and np.isfinite(r['crps_melt_K'])


def test_field_metrics_iou():
    preds, gts = make_preds()
    lab_g = (gts > 1900).astype(float); lab_p = (preds > 1900).astype(float)
    P = np.concatenate([preds, lab_p], axis=2); G = np.concatenate([gts, lab_g], axis=1)
    m = X.field_metrics(P, G, ['temperature', 'liqlabel'], liquid_field='liqlabel')
    assert 0.5 < m['IoU_liquid_liqlabel'] <= 1.0 and m['MAE_liqlabel'] >= 0
    perfect = X.field_metrics(np.repeat(G[:, None], 3, 1), G, ['temperature', 'liqlabel'], 'liqlabel')
    assert perfect['IoU_liquid_liqlabel'] == 1.0 and perfect['MAE_temperature'] == 0.0


def test_model_size_and_benchmark(tmp_path):
    ck = tmp_path / 'b.pth'
    torch.save({'a': torch.zeros(1000)}, ck)
    s = X.model_size(FakeModel(), str(ck))
    assert s['params_generator'] == 16 and s['params_encoder'] == 6 and s['ckpt_MB'] > 0
    b = X.benchmark_sampling(FakeModel(), FakeDS(), 'heun', 4, 0.0, k=3, frames=2, device='cpu', reps=2)
    assert b['bench_frames'] == 2 and b['bench_nfe'] == 9 and b['sec_per_sample'] > 0


def test_gifs(tmp_path):
    p = X.trajectory_gif(FakeModel(), FakeDS(), 1, str(tmp_path / 't.gif'), n_steps=4, k=3, device='cpu')
    from PIL import Image
    im = Image.open(p)
    assert im.n_frames >= 5
    preds, gts = make_preds()
    q = X.simtime_gif(preds, gts, X.frame_info(FakeDS()), str(tmp_path / 's.gif'))
    assert Image.open(q).n_frames >= 3


def test_si_callback_does_not_change_result():
    from diffusionsr.runners.si_schedules import get_schedule
    sched = get_schedule("linear")
    torch.manual_seed(0)
    m = torch.randn(2, 1, 4, 4); x0 = m + 0.1 * torch.randn_like(m)

    def field(x, t):
        return x * 0.0 + 1.0, x * 0.0
    for solver in ('euler', 'heun', 'rk4', 'dopri5'):
        a, n1 = integrate(field, sched, x0, m, 0.1, solver=solver, n_steps=5)
        seen = []
        b, n2 = integrate(field, sched, x0, m, 0.1, solver=solver, n_steps=5,
                          callback=lambda i, t, x: seen.append((i, t)))
        assert torch.allclose(a, b) and n1 == n2
        assert seen[0] == (0, 0.0) and len(seen) >= 2
        if solver != 'dopri5':
            assert len(seen) == 6


def test_trajectory_gif_single_sample(tmp_path):
    """K = 1 (the encoder alone) has no second sample; the GIF shows sample 1 twice instead of failing."""
    from PIL import Image
    p = X.trajectory_gif(FakeModel(), FakeDS(), 0, str(tmp_path / 'k1.gif'), n_steps=3, k=1, device='cpu')
    assert Image.open(p).n_frames >= 4
