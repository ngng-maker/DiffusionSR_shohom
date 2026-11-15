"""
Ablation v3 models (DiffusionSR, Flow-Matching and LDM as trained by train_srdiff) behind the SAME sampling
interface as the SI and FM models, so they go through evaluate_si_schedules unchanged: resumable sampling
over the full test split, metrics, eval_extras (P/V calibration, timing, model size, GIFs).

    sample(upscaled_lr, x_e, lr=None, solver=..., n_steps=..., eps=0.0, generator=None, callback=None)
        -> (x0_hat in HR-normalised space, last timestep only, nfe)

    model_kind  solver  sampler
    abl_dsr     ddpm    ancestral DDPM over all `timesteps` steps (user rule; nfe = 1000)
    abl_fm      euler   Euler from t = 1 (noise) to t = 0 with n_steps steps (fm_n_steps, 50)
    abl_ldm     ddpm    ancestral DDPM in the VAE latent (nfe = 1000), decoded by the frozen VAE
    abl_enc     encoder the shared frozen encoder alone (one pass; K = 1)

The ablation models take n_steps > 1 consecutive simulation timesteps as channels and predict all of them;
the eval scores only the LAST timestep (the current frame), as eval_ablation_v2_frames did. LastStepDataset
hands the harness that frame's channels as `hr` while keeping the full inputs as `lr`/`upscaled_lr`.
"""

from __future__ import annotations

import os

import numpy as np
import torch

KINDS = ('abl_dsr', 'abl_fm', 'abl_ldm', 'abl_enc')


class LastStepDataset(torch.utils.data.Dataset):
    """A SimulationXZDataset whose `res`/`hr` are cut to the last timestep's n_ch channels.

    `lr` and `upscaled_lr` stay full (the models need every timestep). unscale_data/rescale_data accept
    either the full channel count or the last-timestep slice; every other attribute is the wrapped dataset's.
    """

    def __init__(self, ds, n_ch: int):
        self.ds, self.n_ch = ds, int(n_ch)
        self.c_full = ds.n_steps * ds.num_fields
        self.sl = slice(self.c_full - self.n_ch, self.c_full)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        res, hr = item[0][self.sl], item[1][self.sl]
        return (res, hr) + tuple(item[2:])

    def __getattr__(self, name):            # only called for attributes not found on the wrapper
        return getattr(self.__dict__['ds'], name)

    def _full(self, x):
        """Embed a last-timestep array/tensor (n_ch, ...) or (B, n_ch, ...) into the full channel count."""
        ch_ax = 0 if x.ndim == 3 else 1
        shape = list(x.shape)
        shape[ch_ax] = self.c_full
        out = torch.zeros(shape, dtype=x.dtype, device=x.device) if torch.is_tensor(x) else np.zeros(shape, x.dtype)
        idx = [slice(None)] * x.ndim
        idx[ch_ax] = self.sl
        out[tuple(idx)] = x
        return out, ch_ax

    def _apply(self, fn, x, *a, **kw):
        ch_ax = 0 if x.ndim == 3 else 1
        if x.shape[ch_ax] == self.c_full:
            return fn(x, *a, **kw)
        full, ch_ax = self._full(x)
        y = fn(full, *a, **kw)
        idx = [slice(None)] * y.ndim
        idx[ch_ax] = self.sl
        return y[tuple(idx)]

    def unscale_data(self, x, *a, **kw):
        return self._apply(self.ds.unscale_data, x, *a, **kw)

    def rescale_data(self, x, *a, **kw):
        return self._apply(self.ds.rescale_data, x, *a, **kw)


class AblationAdapter:
    def __init__(self, m, kind: str, n_ch: int, fm_n_steps: int = 50):
        self.m, self.kind, self.n_ch, self.fm_n_steps = m, kind, int(n_ch), int(fm_n_steps)
        # the (latent) U-Net, for eval_extras.model_size; the encoder alone has no generative network
        self.model = torch.nn.Identity() if kind == 'abl_enc' else m.model
        self.lr_enc = getattr(m, 'lr_enc', None)
        self.device = m.device
        self.timesteps = int(m.timesteps)
        self.vae = getattr(m, 'vae', None)

    def compute_x_e(self, lr, ulr):
        return self.m.compute_x_e(lr, ulr)

    def _last(self, x):
        return x[:, -self.n_ch:]

    def _to_pixels(self, z):
        return self.vae.decode(z) if self.kind == 'abl_ldm' else z

    @torch.no_grad()
    def sample(self, upscaled_lr, x_e, lr=None, solver='ddpm', n_steps=20, eps=0.0, generator=None,
               callback=None):
        if eps:
            raise ValueError('eps applies to the SI samplers only')
        if self.kind == 'abl_enc':                 # the frozen encoder alone: deterministic, use K = 1
            out = self._last(self.m.encode_lr(lr, output=True))
            if callback is not None:
                callback(0, 0.0, out)
            return out, 1
        dev = upscaled_lr.device
        B, C, H, W = upscaled_lr.shape
        cb = None
        if callback is not None:
            cb = lambda i, t, z: callback(i, t, self._last(self._to_pixels(z)))
        if self.kind == 'abl_fm':
            if solver != 'euler':
                raise ValueError(f"Flow-Matching samples with 'euler', got {solver!r}")
            steps = int(n_steps or self.fm_n_steps)
            x = torch.randn((B, C, H, W), device=dev, generator=generator)
            if cb is not None:
                cb(0, 0.0, x)
            dt = 1.0 / steps
            for k, i in enumerate(reversed(range(steps)), start=1):     # t: 1 -> 0, as euler_sample
                t = torch.full((B,), (i + 1) / steps, device=dev)
                x = x - self.m.model(x, t * self.m.fm_timescale, x_e) * dt
                if cb is not None:
                    cb(k, 1.0 - i / steps, x)
            return self._last(x), steps
        if solver != 'ddpm':
            raise ValueError(f"DiffusionSR and LDM sample with ancestral 'ddpm', got {solver!r}")
        shape = (B, C, H, W)
        if self.kind == 'abl_ldm':
            from diffusionsr.runners.train_ldm import LATENT_CH
            shape = (B, LATENT_CH, H // 4, W // 4)
        T = self.timesteps
        f = lambda a, i: a[i].to(dev).float() if torch.is_tensor(a) else torch.tensor(float(a[i]), device=dev)
        x = torch.randn(shape, device=dev, generator=generator)
        if cb is not None:
            cb(0, 0.0, x)
        for step, i in enumerate(reversed(range(T)), start=1):
            tt = torch.full((B,), i, device=dev, dtype=torch.long)
            e = self.m.model(x, tt, x_e)
            mean = f(self.m.sqrt_recip_alphas, i) * (x - f(self.m.betas, i) * e / f(self.m.sqrt_one_minus_alphas_cumprod, i))
            if i > 0:
                x = mean + f(self.m.posterior_variance, i).sqrt() * torch.randn(x.shape, device=dev, generator=generator)
            else:
                x = mean
            if cb is not None:
                cb(step, 1.0 - i / T, x)
        return self._last(self._to_pixels(x)), T


def build_ablation_model(args, cfg, device):
    """(AblationAdapter, LastStepDataset(test), field_names) for model_kind abl_dsr | abl_fm | abl_ldm | abl_enc.

    The frozen encoder comes from args.enc_dir (shared per field and normalisation); the LDM's frozen VAE from
    cfg['vae_results_dir']; the model from args.model_dir/bestmodel_saved.pth (raw weights: these loops keep
    no EMA)."""
    from diffusionsr.datasets.dataset import SimulationXZDataset
    from diffusionsr.runners.train_diffusion import remove_module_prefix
    kind = cfg['model_kind']
    if kind not in KINDS:
        raise ValueError(f'not an ablation model: model_kind={kind!r}')
    if getattr(args, 'weights', 'raw') != 'raw':
        raise SystemExit('ablation models keep no EMA weights; use --weights raw')
    raw = cfg.get('fields', 'temperature')
    field_names = raw.split('_') if isinstance(raw, str) else list(raw)
    kw = dict(downscale_method=cfg.get('downscale_method', 'direct'), root_folder=args.data_root,
              normalize=cfg.get('normalize_method', 'standardize'), n_steps=int(cfg.get('n_steps', 1)),
              field_names=field_names, history=cfg.get('history', 'legacy'))   # as trained (train_srdiff)
    train_ds, dev_ds, test_ds = (SimulationXZDataset(split=s, **kw) for s in ('train', 'dev', 'test'))
    encoding = bool(cfg.get('encoding', True))
    common = dict(results_folder=args.model_dir, lr_encoder_folder=args.enc_dir if encoding else None,
                  train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds, encoding=encoding,
                  device=device, conditioning=cfg.get('conditioning', 'implicit'),
                  timesteps=int(cfg.get('timesteps', 1000)), schedule=cfg.get('schedule', 'linear'),
                  seed=int(cfg.get('seed', 0)))
    if kind == 'abl_ldm':
        from diffusionsr.runners.train_ldm import LDMModel
        m = LDMModel(vae_folder=cfg['vae_results_dir'], latent_unet_dim=cfg.get('latent_unet_dim'),
                     latent_unet_mults=tuple(cfg.get('latent_unet_mults') or (1,)), **common)
    elif kind == 'abl_fm':
        from diffusionsr.runners.train_flow_matching import FlowMatchingModel
        m = FlowMatchingModel(enc_output=False, fm_timescale=float(cfg.get('fm_timescale', 1000)), **common)
    else:
        from diffusionsr.runners.train_diffusion import DiffusionModel
        m = DiffusionModel(enc_output=False, **common)
    n_ch = len(field_names)
    if kind == 'abl_enc':
        if not encoding:
            raise SystemExit('abl_enc needs encoding: true')
        print(f'Encoder alone from {args.enc_dir} (n_steps={kw["n_steps"]})')
        return AblationAdapter(m, kind, n_ch), LastStepDataset(test_ds, n_ch), field_names
    best = os.path.join(args.model_dir, 'bestmodel_saved.pth')
    path = best if os.path.exists(best) else os.path.join(args.model_dir, 'ckpt.pth')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt[0] if isinstance(ckpt, (list, tuple)) else ckpt.get('model', ckpt)
    m.model.load_state_dict(remove_module_prefix(state))
    m.model.eval()
    if kind == 'abl_ldm':
        m.vae.eval()
    print(f'Loaded {path} as {kind} (encoding={encoding}, n_steps={kw["n_steps"]})')
    return AblationAdapter(m, kind, n_ch, int(cfg.get('fm_n_steps', 50))), LastStepDataset(test_ds, n_ch), field_names
