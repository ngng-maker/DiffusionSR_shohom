"""
FMLibraryModel: standard flow matching built with facebookresearch/flow_matching (pip flow_matching 1.0.10),
inside the SAME training harness, U-Net and frozen-encoder conditioning as the SI arms.

Experiment 2 of docs/plans/2026-11_si_multifield_fm_plan.md compares three models that share every
architectural and training choice except the flow-matching formulation:

  FM-noise  (fm_source 'noise')   x0 ~ N(0, I): textbook flow matching, independent coupling.
  FM-paired (fm_source 'paired')  x0 = enc(LR) + sigma*zeta: the SI arms' paired coupling (si_base 'enc').
  SI-FM                           ScheduledInterpolantModel, schedule 'linear', eps = 0 (configs/si_schedules).

FM-paired vs SI-FM isolates the IMPLEMENTATION: library path/loss/solver and a SINGLE velocity head
(U-Net out_dim = C), against SI's dual head (v_hat + eta_hat, out_dim = 2C), its t < 0.999 training
range and its final denoising jump. FM-noise vs FM-paired isolates the COUPLING.

Library pieces used, unchanged:
  * path   AffineProbPath(scheduler=CondOTScheduler()): x_t = t*x1 + (1-t)*x0, dx_t = x1 - x0,
           t ~ U(0, 1), x0 at t = 0 and data at t = 1 (the library's convention).
  * loss   regression of the velocity v_theta(x_t, t) onto path_sample.dx_t (MSE for loss_type 'l2').
  * solver ODESolver over a ModelWrapper: torchdiffeq 'euler', 'midpoint', 'rk4' (fixed grid of n_steps
           steps) and 'dopri5' (adaptive, atol = rtol = 1e-4).
The time fed to the U-Net embedding is t * fm_timescale, exactly as the SI arms do.

Inherited unchanged from ScheduledInterpolantModel / CoupledInterpolantModel: the training loop (EMA 0.999,
grad clip 1.0, per-step StepLR, FlippedDataset p_flip 0.2), checkpoint layout (raw weights at [0], EMA at
[5]), best-epoch selection on validation loss, restart logic, the deterministic W&B run id and the ONE
single-version checkpoint artifact per config (utils.log_single_version_artifact), and base_hr() /
compute_x_e() (frozen RRDB, implicit 64-channel conditioning added after the first conv at EVERY network
evaluation).

Train: `si_process: "fm_library"` in the config, `--modeltype coupled` (slurm/fm_library/01_train_array.sh).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from diffusionsr.models.diffusion_model import Unet
from diffusionsr.runners.train_si_schedules import ScheduledInterpolantModel
from diffusionsr.utils import log_single_version_artifact

FM_SOURCES = ('noise', 'paired')
FM_SOLVERS = ('euler', 'midpoint', 'rk4', 'dopri5')
FM_STEPS_PER_STEP = {'euler': 1, 'midpoint': 2, 'rk4': 4}   # network evaluations per fixed step
FM_ATOL = FM_RTOL = 1e-4
_LOSS = {'l2': F.mse_loss, 'l1': F.l1_loss, 'huber': F.smooth_l1_loss}

CKPT_FILES = ('ckpt.pth', 'bestmodel_saved.pth', 'loss_epoch.txt', 'validation_loss_epoch.txt',
              'configuration.yml', 'wandb_run_id.txt')


class _Velocity(ModelWrapper):
    """The U-Net as the library's velocity model: v(x, t), conditioning x_e fixed, calls counted."""

    def __init__(self, unet, x_e, fm_timescale: float):
        super().__init__(unet)
        self.x_e = x_e
        self.fm_timescale = fm_timescale
        self.nfe = 0

    def forward(self, x, t, **extras):
        self.nfe += 1
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        tb = t.reshape(-1).expand(x.shape[0]) if t.numel() == 1 else t.reshape(x.shape[0])
        return self.model(x, tb * self.fm_timescale, self.x_e)


class FMLibraryModel(ScheduledInterpolantModel):
    """Single-head conditional flow matching from the flow_matching library, in the SI harness."""

    def __init__(self, *args, fm_source: str = 'paired', **kwargs):
        if fm_source not in FM_SOURCES:
            raise ValueError(f'fm_source must be one of {FM_SOURCES}, got {fm_source!r}')
        # The schedule is unused (the library path replaces it); 'linear' only satisfies the parent.
        kwargs['si_schedule'] = 'linear'
        kwargs['si_gamma_scale'] = 0.0
        super().__init__(*args, **kwargs)
        if fm_source == 'paired' and self.si_base != 'enc':
            raise ValueError("fm_source='paired' is the SI start x0 = enc(LR) + sigma*zeta, so it needs "
                             f"si_base='enc' (got {self.si_base!r})")
        self.fm_source = fm_source
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Rebuild the U-Net with a SINGLE velocity head (the parent built the SI dual head, 2C outputs).
        base_C = self.train_dataset.n_steps * self.train_dataset.num_fields
        self.model = Unet(
            dim=self.image_size,
            channels=base_C,
            init_dim=base_C if self.enc_output else None,
            encoder_flag=self.encoding,
            dim_mults=(1, 2, 4,),
            conditioning=self.conditioning,
            out_dim=base_C,
        )
        self.model.to(self.device)
        print(f'FMLibraryModel: flow_matching CondOT path, single velocity head, fm_source={fm_source}'
              f'{" (x0 = enc(LR) + sigma*zeta, sigma=%g)" % self.sigma if fm_source == "paired" else " (x0 ~ N(0, I))"}')

    # ------------------------------------------------------------------
    # Source sample x0
    # ------------------------------------------------------------------

    def _source(self, shape_like, m=None, generator=None):
        """x0 for the batch. 'noise': N(0, I). 'paired': m + sigma*zeta with m = enc(LR) (base_hr)."""
        z = torch.randn(shape_like.shape, device=shape_like.device, dtype=shape_like.dtype,
                        generator=generator)
        return z if self.fm_source == 'noise' else m + self.sigma * z

    # ------------------------------------------------------------------
    # Training objective (the inherited loop's one loss hook)
    # ------------------------------------------------------------------

    def _si_forward(self, hr, base_mean, x_e, loss_type):
        """hr = x1 (HR-stat space); base_mean = m = enc(LR) from base_hr() (ignored for 'noise')."""
        x0 = self._source(hr, base_mean)
        t = torch.rand(hr.shape[0], device=hr.device)
        ps = self.path.sample(x_0=x0, x_1=hr, t=t)
        v_hat = self.model(ps.x_t, t * self.fm_timescale, x_e)
        return _LOSS[loss_type](v_hat, ps.dx_t)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, upscaled_lr, x_e, lr=None, solver='euler', n_steps=20, eps=0.0, generator=None,
               callback=None):
        """One sample per input. Returns (x1_hat in HR-stat space, network evaluations used).

        euler / midpoint / rk4: torchdiffeq fixed-grid solver on the uniform grid t_i = i/n_steps
        (step 1/n_steps); nfe = n_steps * (1, 2, 4). dopri5: adaptive, atol = rtol = 1e-4; nfe as used.
        callback(i, t_i, x_i) is called for i = 0..n_steps on that grid (dopri5: interpolated output).
        """
        if eps != 0:
            raise ValueError(f'flow matching samples the deterministic ODE only; got eps={eps}')
        if solver not in FM_SOLVERS:
            raise ValueError(f'Unknown solver={solver!r}; choose from {FM_SOLVERS}')
        self.model.eval()
        ref = upscaled_lr.to(self.device).float()
        if self.fm_source == 'paired':
            if lr is None:
                raise ValueError("fm_source='paired' starts from enc(LR), so the true LR must be passed")
            m = self.base_hr(lr, ref)
        else:
            m = None
        x0 = self._source(ref, m, generator)

        vel = _Velocity(self.model, x_e, self.fm_timescale)
        ode = ODESolver(velocity_model=vel)
        want_path = callback is not None
        if want_path or solver != 'dopri5':
            grid = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device)
        else:
            grid = torch.tensor([0.0, 1.0], device=x0.device)
        # step_size=None makes the fixed-grid solvers step exactly on `grid` (n_steps steps of 1/n_steps).
        sol = ode.sample(x_init=x0, step_size=None, method=solver, atol=FM_ATOL, rtol=FM_RTOL,
                         time_grid=grid, return_intermediates=want_path)
        if want_path:
            for i in range(sol.shape[0]):
                callback(i, float(grid[i]), sol[i])
            sol = sol[-1]
        return sol, vel.nfe

    def batch_sample(self, dataset, batch, x_e, sampler='euler', n_steps=20, upscaled_lr=None,
                     eps0=0.0, lr=None, eps=None, generator=None, **kwargs):
        """API-compatible with ScheduledInterpolantModel.batch_sample: returns a (1, B, C, H, W) stack."""
        if upscaled_lr is None:
            raise ValueError('upscaled_lr is required (it fixes the HR shape and, for paired, the base)')
        x1, nfe = self.sample(upscaled_lr.to(self.device).float(), x_e, lr=lr, solver=sampler,
                              n_steps=n_steps, eps=eps0 if eps is None else eps, generator=generator)
        self.last_nfe = nfe
        return x1.unsqueeze(0).cpu()

    # ------------------------------------------------------------------
    # W&B checkpoint artifact metadata
    # ------------------------------------------------------------------

    def _upload_state(self, epoch, final=False):
        files = {name: str(self.results_folder / name) for name in CKPT_FILES}
        log_single_version_artifact(
            self.ckpt_artifact, 'checkpoint', files,
            metadata={'epoch': int(epoch), 'final': bool(final), 'model_kind': 'fm_library',
                      'fm_source': self.fm_source, 'si_base': self.si_base, 'sigma': self.sigma},
            description=f'Resumable training state for {self.ckpt_artifact} (single version)')


def build_fm_model(args, cfg, device):
    """Mirror of analysis/evaluate_si_schedules.build_model for `model_kind: fm_library` configs.

    args needs data_root, model_dir, enc_dir, weights ('raw' -> checkpoint[0], 'ema' -> checkpoint[5]).
    Returns (model, test_ds, field_names).
    """
    from diffusionsr.datasets.flipped_dataset import FlippedDataset
    raw = cfg.get('fields', 'temperature')
    field_names = raw.split('_') if isinstance(raw, str) else list(raw)
    kw = dict(downscale_method=cfg.get('downscale_method', 'direct'), root_folder=args.data_root,
              normalize=cfg.get('normalize_method', 'standardize'), n_steps=int(cfg.get('n_steps', 1)),
              field_names=field_names, p_flip=0.0)
    train_ds = FlippedDataset(split='train', **kw)
    dev_ds = FlippedDataset(split='dev', **kw)
    test_ds = FlippedDataset(split='test', **kw)

    best_path = os.path.join(args.model_dir, 'bestmodel_saved.pth')
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    unet_dim = int(ckpt[0]['time_mlp.1.weight'].shape[1])

    model = FMLibraryModel(
        results_folder=args.model_dir, lr_encoder_folder=args.enc_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        encoding=bool(cfg.get('encoding', True)), conditioning=cfg.get('conditioning', 'implicit'),
        sigma=float(cfg.get('si_sigma', 0.1)), si_cond=cfg.get('si_cond', 'enc_implicit'),
        si_base=cfg.get('si_base', 'enc'), fm_source=cfg.get('fm_source', 'paired'),
        fm_timescale=float(cfg.get('fm_timescale', 1000.0)),
        image_size_override=unet_dim, enc_output=False, device=device)
    weights = getattr(args, 'weights', 'raw')
    state = ckpt[5] if weights == 'ema' else ckpt[0]
    model.model.load_state_dict(state)
    model.model.eval()
    print(f'Loaded {best_path} ({weights} weights, best epoch {ckpt[2]})')
    return model, test_ds, field_names
