"""
The DDPM baselines (task-0 'ddpm_matched' and as-published 'diffusionsr') behind the SAME sampling interface
as the SI and FM models, so they go through evaluate_si_schedules unchanged: resumable sampling, metrics,
eval_extras (P/V calibration, timing, model size, GIFs) and the cascade.

    sample(upscaled_lr, x_e, lr=None, solver='ddpm', n_steps=20, eps=0.0, generator=None, callback=None)
        -> (x0_hat in HR-standardised space, nfe)

solver 'ddpm' = ancestral DDPM over all `timesteps` steps (the user's rule for DDPM-type models, nfe = 1000);
solver 'ddim' = deterministic DDIM (eta 0) with skip = timesteps / n_steps, as the older studies used
(n_steps 20 -> skip 50). Noise comes from `generator`, so a seeded eval is reproducible batch by batch.
"""

from __future__ import annotations

import os

import torch


class DDPMAdapter:
    def __init__(self, ddpm_model):
        self.m = ddpm_model
        self.model = ddpm_model.model          # the U-Net, for eval_extras.model_size
        self.lr_enc = getattr(ddpm_model, 'lr_enc', None)
        self.device = ddpm_model.device
        self.timesteps = int(ddpm_model.timesteps)

    def compute_x_e(self, lr, ulr):
        return self.m.compute_x_e(lr, ulr)

    def _eps(self, x, i, x_e):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        return self.m.model(x, t, x_e)

    @torch.no_grad()
    def sample(self, upscaled_lr, x_e, lr=None, solver='ddpm', n_steps=20, eps=0.0, generator=None,
               callback=None):
        if eps:
            raise ValueError('eps applies to the SI samplers only')
        dev = upscaled_lr.device
        T = self.timesteps
        x = torch.randn(upscaled_lr.shape, device=dev, generator=generator)
        if callback is not None:
            callback(0, 0.0, x)
        f = lambda a, i: a[i].to(dev).float() if torch.is_tensor(a) else torch.tensor(float(a[i]), device=dev)
        if solver == 'ddpm':
            for step, i in enumerate(reversed(range(T)), start=1):
                e = self._eps(x, i, x_e)
                mean = f(self.m.sqrt_recip_alphas, i) * (x - f(self.m.betas, i) * e / f(self.m.sqrt_one_minus_alphas_cumprod, i))
                if i > 0:
                    z = torch.randn(x.shape, device=dev, generator=generator)
                    x = mean + f(self.m.posterior_variance, i).sqrt() * z
                else:
                    x = mean
                if callback is not None:
                    callback(step, 1.0 - i / T, x)
            return x, T
        if solver == 'ddim':
            skip = max(1, T // int(n_steps))
            seq = list(range(0, T, skip))
            seq_next = [-1] + seq[:-1]
            for step, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next)), start=1):
                at = self.m.ddim_compute_alpha(self.m.betas, torch.full((x.shape[0],), i, device=dev).long())
                at_next = self.m.ddim_compute_alpha(self.m.betas, torch.full((x.shape[0],), j, device=dev).long())
                e = self._eps(x, i, x_e)
                x0_t = (x - e * (1 - at).sqrt()) / at.sqrt()
                x = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * e
                if callback is not None:
                    callback(step, 1.0 - j / T if j >= 0 else 1.0, x)
            return x, len(seq)
        raise ValueError(f"DDPM baselines sample with 'ddpm' or 'ddim', got {solver!r}")


def build_ddpm_model(args, cfg, device):
    """(DDPMAdapter, test_ds, field_names) for a config with model_kind 'ddpm_matched' or 'diffusionsr'.

    Mirrors evaluate_baseline_matched: U-Net dim from the checkpoint, FlippedDataset with p_flip 0, and
    args.weights 'ema' (checkpoint[5], harness-trained models only) or 'raw' (checkpoint[0])."""
    from diffusionsr.datasets.flipped_dataset import FlippedDataset
    from diffusionsr.runners.train_diffusion import remove_module_prefix
    raw = cfg.get('fields', 'temperature')
    field_names = raw.split('_') if isinstance(raw, str) else list(raw)
    kw = dict(downscale_method=cfg.get('downscale_method', 'direct'), root_folder=args.data_root,
              normalize=cfg.get('normalize_method', 'standardize'), n_steps=int(cfg.get('n_steps', 1)),
              field_names=field_names, p_flip=0.0)
    train_ds, dev_ds, test_ds = (FlippedDataset(split=s, **kw) for s in ('train', 'dev', 'test'))
    best = os.path.join(args.model_dir, 'bestmodel_saved.pth')
    path = best if os.path.exists(best) else os.path.join(args.model_dir, 'ckpt.pth')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    common = dict(results_folder=args.model_dir, lr_encoder_folder=args.enc_dir,
                  train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
                  timesteps=int(cfg.get('timesteps', 1000)), conditioning=cfg.get('conditioning', 'implicit'),
                  encoding=True, schedule=cfg.get('schedule', 'linear'), device=device, enc_output=False,
                  image_size_override=int(ckpt[0]['time_mlp.1.weight'].shape[1]))
    kind = cfg['model_kind']
    if kind == 'ddpm_matched':
        from diffusionsr.runners.train_ddpm_matched import DDPMMatchedHarnessModel
        model = DDPMMatchedHarnessModel(sigma=float(cfg.get('si_sigma', 0.1)), gamma_c=0.0, si_cond='enc_implicit',
                                        fm_timescale=float(cfg.get('fm_timescale', 1000.0)), **common)
    elif kind == 'diffusionsr':
        from diffusionsr.runners.train_diffusion import DiffusionModel
        model = DiffusionModel(**common)
    else:
        raise ValueError(f'not a DDPM baseline: model_kind={kind!r}')
    if args.weights == 'ema':
        if len(ckpt) <= 5 or ckpt[5] is None:
            raise SystemExit(f'{path} holds no EMA weights (plain DiffusionModel loop); use --weights raw')
        state = ckpt[5]
    else:
        state = ckpt[0]
    model.model.load_state_dict(remove_module_prefix(state))
    model.model.eval()
    print(f'Loaded {path} ({args.weights} weights) as {kind}')
    return DDPMAdapter(model), test_ds, field_names
