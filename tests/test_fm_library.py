"""CPU tests for runners/train_fm_library.FMLibraryModel (flow_matching library, single velocity head)."""
from pathlib import Path

import pytest
import torch

pytest.importorskip('flow_matching')

from diffusionsr.models.lr_encoder_model import rrdbnet_encoder  # noqa: E402
from diffusionsr.runners.train_diffusion import DiffusionModel  # noqa: E402
from diffusionsr.runners.train_fm_library import FMLibraryModel, build_fm_model  # noqa: E402

from test_training_models import TrainableDummyDataset  # noqa: E402  (tests/ is on sys.path under pytest)

IMG, FACTOR = 16, 2


def _fake_encoder(self):
    enc = rrdbnet_encoder(upscale_factor=self.train_dataset.factor,
                          in_channels=self.train_dataset.n_steps * self.train_dataset.num_fields,
                          out_channels=self.train_dataset.out_steps * self.train_dataset.num_fields)
    return enc.eval().to(self.device)


def _make(tmp_path: Path, monkeypatch, fm_source='paired', num_fields=1, sigma=0.1):
    monkeypatch.setattr(DiffusionModel, 'initialize_encoder', _fake_encoder)
    ds = TrainableDummyDataset(num_fields=num_fields, img_shape=IMG, factor=FACTOR)
    torch.manual_seed(0)
    return FMLibraryModel(
        results_folder=tmp_path / f'fm_{fm_source}_{num_fields}', lr_encoder_folder=tmp_path / 'enc',
        train_dataset=ds, dev_dataset=ds, test_dataset=ds, timesteps=1000,
        conditioning='implicit', encoding=True, schedule='linear', device='cpu', enc_output=False,
        sigma=sigma, si_cond='enc_implicit', si_base='enc', fm_timescale=1000.0, fm_source=fm_source)


def _batch(num_fields=1, B=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    hr = torch.randn(B, num_fields, IMG, IMG, generator=g)
    lr = torch.randn(B, num_fields, IMG // FACTOR, IMG // FACTOR, generator=g)
    ulr = torch.randn(B, num_fields, IMG, IMG, generator=g)
    return hr, lr, ulr


@pytest.mark.parametrize('num_fields', [1, 2])
def test_single_velocity_head(tmp_path, monkeypatch, num_fields):
    model = _make(tmp_path, monkeypatch, num_fields=num_fields)
    hr, lr, ulr = _batch(num_fields)
    x_e = model.compute_x_e(lr, ulr)
    out = model.model(hr, torch.full((hr.shape[0],), 500.0), x_e)
    assert out.shape == hr.shape            # C outputs, not the SI dual head's 2C


@pytest.mark.parametrize('fm_source', ['noise', 'paired'])
def test_loss_finite_and_decreasing(tmp_path, monkeypatch, fm_source):
    model = _make(tmp_path, monkeypatch, fm_source=fm_source)
    hr, lr, ulr = _batch(B=4)
    with torch.no_grad():                       # the encoder is frozen in training; keep it off the graph
        m = model.base_hr(lr, ulr)
        x_e = model.compute_x_e(lr, ulr)
    opt = torch.optim.Adam(model.model.parameters(), lr=1e-3)
    losses = []
    for step in range(60):
        torch.manual_seed(0)                    # identical t/x0 draws every step: a fixed, learnable target
        loss = model._si_forward(hr, m, x_e, 'l2')
        assert torch.isfinite(loss)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < 0.7 * sum(losses[:10]) / 10


@pytest.mark.parametrize('solver,per_step', [('euler', 1), ('midpoint', 2), ('rk4', 4)])
@pytest.mark.parametrize('n_steps', [3, 7])
def test_sample_shape_nfe_and_callback(tmp_path, monkeypatch, solver, per_step, n_steps):
    model = _make(tmp_path, monkeypatch)
    hr, lr, ulr = _batch()
    x_e = model.compute_x_e(lr, ulr)
    seen = []
    x1, nfe = model.sample(ulr, x_e, lr=lr, solver=solver, n_steps=n_steps,
                           callback=lambda i, t, x: seen.append((i, t, tuple(x.shape))))
    assert x1.shape == hr.shape and torch.isfinite(x1).all()
    assert nfe == per_step * n_steps
    assert [s[0] for s in seen] == list(range(n_steps + 1))
    assert seen[0][1] == 0.0 and abs(seen[-1][1] - 1.0) < 1e-6
    assert all(s[2] == tuple(hr.shape) for s in seen)


def test_dopri5_adaptive_and_eps_rejected(tmp_path, monkeypatch):
    model = _make(tmp_path, monkeypatch)
    hr, lr, ulr = _batch()
    x_e = model.compute_x_e(lr, ulr)
    x1, nfe = model.sample(ulr, x_e, lr=lr, solver='dopri5')
    assert x1.shape == hr.shape and nfe > 0
    with pytest.raises(ValueError):
        model.sample(ulr, x_e, lr=lr, solver='euler', eps=0.5)
    with pytest.raises(ValueError):
        model.sample(ulr, x_e, lr=lr, solver='heun')


@pytest.mark.parametrize('fm_source', ['noise', 'paired'])
def test_source_start_point(tmp_path, monkeypatch, fm_source):
    sigma = 0.3
    model = _make(tmp_path, monkeypatch, fm_source=fm_source, sigma=sigma)
    hr, lr, ulr = _batch()
    x_e = model.compute_x_e(lr, ulr)
    seen = {}
    gen = torch.Generator().manual_seed(42)
    model.sample(ulr, x_e, lr=lr, solver='euler', n_steps=2, generator=gen,
                 callback=lambda i, t, x: seen.setdefault(i, x.clone()))
    zeta = torch.randn(hr.shape, generator=torch.Generator().manual_seed(42))
    expect = zeta if fm_source == 'noise' else model.base_hr(lr, ulr) + sigma * zeta
    assert torch.allclose(seen[0], expect, atol=1e-6)


def test_paired_requires_enc_base_and_lr(tmp_path, monkeypatch):
    model = _make(tmp_path, monkeypatch, fm_source='paired')
    _, lr, ulr = _batch()
    with pytest.raises(ValueError):
        model.sample(ulr, model.compute_x_e(lr, ulr), lr=None)


def test_batch_sample_api(tmp_path, monkeypatch):
    model = _make(tmp_path, monkeypatch)
    hr, lr, ulr = _batch()
    out = model.batch_sample(None, hr, model.compute_x_e(lr, ulr), sampler='euler', n_steps=4,
                             upscaled_lr=ulr, lr=lr)
    assert out.shape == (1, *hr.shape) and model.last_nfe == 4


@pytest.mark.parametrize('weights,idx', [('raw', 0), ('ema', 5)])
def test_build_fm_model_loads_raw_or_ema(tmp_path, monkeypatch, weights, idx):
    model = _make(tmp_path, monkeypatch)
    raw = {k: v.clone() for k, v in model.model.state_dict().items()}
    ema = {k: v + 1.0 if v.is_floating_point() else v.clone() for k, v in raw.items()}
    run = tmp_path / 'run'; run.mkdir()
    torch.save([raw, {}, 7, 0, 0.5, ema], run / 'bestmodel_saved.pth')

    ds = TrainableDummyDataset(num_fields=1, img_shape=IMG, factor=FACTOR)
    import diffusionsr.datasets.flipped_dataset as fd
    monkeypatch.setattr(fd, 'FlippedDataset', lambda split, **kw: ds)

    class A:
        data_root = 'unused'; model_dir = str(run); enc_dir = str(tmp_path / 'enc')
    A.weights = weights
    cfg = dict(fields='temperature', si_sigma=0.1, si_base='enc', fm_source='paired')
    built, test_ds, fields = build_fm_model(A, cfg, 'cpu')
    want = (raw, {}, 7, 0, 0.5, ema)[idx]
    got = built.model.state_dict()
    assert all(torch.equal(got[k], want[k]) for k in want)
    assert fields == ['temperature'] and test_ds is ds
