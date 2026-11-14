"""CPU tests for analysis/evaluate_cascade.py (chaining, alignment, shapes) and analysis/ddpm_adapter.py."""
import numpy as np
import torch

from diffusionsr.analysis.ddpm_adapter import DDPMAdapter
from diffusionsr.analysis.evaluate_cascade import CascadeModel, ChainDataset
from diffusionsr.analysis.evaluate_si_schedules import sample_resumable


class StageDS:
    """Fake task root: HR side `hr`, LR side `lr`; per-pixel standardisation stats like the real dataset."""
    def __init__(self, hr, lr, order=None, mean=1000.0, std=500.0):
        self.img_shape, self.lr_side = hr, lr
        self.factor = hr // lr
        conds = [(100, 500), (300, 200)]
        paths = [f'/r/test/LR/direct/1x/power{P}velocity{V}/_{t}.npy' for P, V in conds for t in (1, 2)]
        self.lr_paths = [paths[i] for i in (order or range(len(paths)))]
        self.field_names = ['temperature']
        self.field_threshold = {'temperature': 8000}
        self.num_fields = 1
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.lr_paths)

    def _f(self, side, i):
        return torch.full((1, side, side), 2000.0 + 10 * i)

    def __getitem__(self, i):
        hr = (self._f(self.img_shape, i) - self.mean) / self.std
        lr = (self._f(self.lr_side, i) - self.mean) / self.std
        ulr = (self._f(self.img_shape, i) - self.mean) / self.std
        return hr, hr, lr, ulr

    def unscale_data(self, a, input_type='hr', maintain_torch=False):
        return a * self.std + self.mean

    def rescale_data(self, a, input_type='hr', maintain_torch=False):
        return (a - self.mean) / self.std


class Upsampler:
    """'Model' that returns its upscaled-LR input plus small noise; counts nfe = 3."""
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.lr_enc = torch.nn.Linear(2, 3)
        self.device = 'cpu'
        self.calls = []

    def compute_x_e(self, lr, ulr):
        return ulr

    def sample(self, ulr, x_e, lr=None, solver='heun', n_steps=20, eps=0.0, generator=None, callback=None):
        self.calls.append(tuple(ulr.shape))
        if callback is not None:
            callback(0, 0.0, ulr)
            callback(1, 1.0, ulr)
        return ulr + 0.01 * torch.randn(ulr.shape, generator=generator), 3


def test_chain_dataset_aligns_and_restores_ulr():
    a, b = StageDS(4, 2), StageDS(8, 4, order=[3, 2, 1, 0])
    ch = ChainDataset(a, b)
    assert ch.b_index == [3, 2, 1, 0]
    _, hr_b, lr_a, ulr_up = ch[1]
    assert hr_b.shape[-1] == 8 and ulr_up.shape[-1] == 8
    assert torch.equal(ulr_up[..., ::2, ::2], a[1][3])


def test_cascade_sample_shapes_nfe_and_resumable(tmp_path):
    a, b = StageDS(4, 2), StageDS(8, 4, mean=900.0, std=400.0)
    A, B = Upsampler(), Upsampler()
    cm = CascadeModel(A, B, a, b)
    ch = ChainDataset(a, b)
    _, hr, lr, ulr = (x[None] for x in ch[0])
    seen = []
    x10, nfe = cm.sample(ulr.repeat(3, 1, 1, 1), cm.compute_x_e(lr, ulr).repeat(3, 1, 1, 1), lr=lr.repeat(3, 1, 1, 1),
                         generator=torch.Generator().manual_seed(0), callback=lambda i, t, x: seen.append((i, t, x.shape)))
    assert x10.shape == (3, 1, 8, 8) and nfe == 6
    assert A.calls[-1] == (3, 1, 4, 4) and B.calls[-1] == (3, 1, 8, 8)
    assert [s[2][-1] for s in seen] == [8, 8, 8, 8] and seen[-1][1] == 1.0
    # physical value survives the chain: stage A out ~2000 K -> stage B in/out ~2000 K in B's units
    assert abs(float(b.unscale_data(x10).mean()) - 2000.0) < 20
    args = type('A', (), dict(out_dir=str(tmp_path), k_samples=2, max_frames=None, frames_per_batch=2, solver='heun',
                              eps=0.0, n_steps=20, seed=0, weights='raw'))()
    preds, gts, ulrs, nfe_mean, _ = sample_resumable(cm, ch, args, 'cpu')
    assert preds.shape == (4, 2, 1, 8, 8) and gts.shape == (4, 1, 8, 8) and nfe_mean == 6


class FakeDDPM:
    def __init__(self, T=10):
        self.timesteps, self.device = T, 'cpu'
        self.betas = torch.linspace(1e-4, 0.02, T)
        al = 1 - self.betas
        ac = torch.cumprod(al, 0)
        self.sqrt_recip_alphas = (1 / al).sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - ac).sqrt()
        ac_prev = torch.cat([torch.ones(1), ac[:-1]])
        self.posterior_variance = self.betas * (1 - ac_prev) / (1 - ac)
        self.model = lambda x, t, x_e: torch.zeros_like(x)
        self.lr_enc = None

    def compute_x_e(self, lr, ulr):
        return ulr

    def ddim_compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1), beta], 0)
        return (1 - beta).cumprod(0).index_select(0, t + 1).view(-1, 1, 1, 1)


def test_ddpm_adapter_ancestral_and_ddim():
    ad = DDPMAdapter(FakeDDPM(T=10))
    u = torch.zeros(2, 1, 4, 4)
    seen = []
    x, nfe = ad.sample(u, u, solver='ddpm', generator=torch.Generator().manual_seed(1),
                       callback=lambda i, t, x: seen.append(i))
    assert x.shape == u.shape and nfe == 10 and seen == list(range(11))
    x2, nfe2 = ad.sample(u, u, solver='ddpm', generator=torch.Generator().manual_seed(1))
    assert torch.equal(x, x2)                                   # seeded -> reproducible
    _, nfe3 = ad.sample(u, u, solver='ddim', n_steps=5, generator=torch.Generator().manual_seed(1))
    assert nfe3 == 5
