"""
ScheduledInterpolantModel: paired-coupling stochastic interpolant with a selectable schedule
(linear/FM, trigonometric/GVP, VP diffusion, Brownian bridge), sampled with a selectable solver and eps.

What is identical to the previous SI experiment (slurm/interpolant, notebook 21)
---------------------------------------------------------------------------------
Inherited unchanged from CoupledInterpolantModel: the training loop (EMA 0.999, grad-clip 1.0, per-step
StepLR, FlippedDataset p_flip=0.2), U-Net architecture, frozen RRDB encoder, best-epoch checkpointing,
restart logic, and the paired base x0 = m + sigma*zeta with m = enc(LR) (si_base 'enc') or the bicubic
upscale (si_base 'ulr').

CONDITIONING, as in DiffusionSR: si_cond 'enc_implicit'. The frozen encoder's 64-channel feature map
x_e = enc_features(LR) is computed ONCE per LR input and ADDED to the U-Net after its first conv at
EVERY network evaluation of every solver step (and of every DDPM/DDIM step in DiffusionSR). Nothing
about the conditioning changes between the DDPM baseline, the previous SI arms and these arms.

What is new
-----------
  * The schedule (alpha_t, beta_t, gamma_t), from runners/si_schedules.py.
  * A dual-head U-Net for EVERY schedule (out_dim = 2C): v_hat and eta_hat, so every arm has a score and
    supports the eps > 0 SDE. The previous Option-A arms were single-head and ODE-only.
  * Loss = |v_hat - (alpha_dot m + beta_dot x1)|^2 + |eta_hat - n|^2 (see si_schedules.py for why this
    decomposition rather than direct velocity regression).
  * Solvers from analysis/si_solvers.py; eps is a sampling parameter, so one training run per
    (schedule, base) covers every solver and eps.
  * W&B: a DETERMINISTIC run id per (schedule, base), resumed with resume='allow', and ONE checkpoint
    artifact per (schedule, base), overwritten on every sync (utils.log_single_version_artifact). A
    restart on any cluster, even with the local run directory gone, restores from that artifact and
    resumes the same run.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import wandb

from diffusionsr.analysis.si_solvers import integrate
from diffusionsr.models.diffusion_model import Unet
from diffusionsr.runners.si_schedules import T_END, get_schedule, make_interpolant
from diffusionsr.runners.train_coupled import CoupledInterpolantModel
from diffusionsr.utils import log_single_version_artifact

_LOSS = {'l2': F.mse_loss, 'l1': F.l1_loss, 'huber': F.smooth_l1_loss}

# Files that make up a resumable training state. Restoring all of them on a fresh node reproduces the
# local run directory exactly (loss history included), so a restarted run continues seamlessly.
CKPT_FILES = ('ckpt.pth', 'bestmodel_saved.pth', 'loss_epoch.txt', 'validation_loss_epoch.txt',
              'configuration.yml', 'wandb_run_id.txt')


class ScheduledInterpolantModel(CoupledInterpolantModel):

    def __init__(self, *args, si_schedule: str = 'linear', ckpt_artifact: str | None = None,
                 wandb_run_id: str | None = None, **kwargs):
        # gamma_c is the previous experiment's Option-B knob; the schedule now owns gamma_t. Forcing 0
        # stops the parent building its own dual head, so the one below is the only U-Net rebuild.
        kwargs['gamma_c'] = 0.0
        super().__init__(*args, **kwargs)
        self.si_schedule_name = si_schedule
        self.si_sched = get_schedule(si_schedule)   # NOT self.schedule: that is DDPM's beta-schedule name
        self.ckpt_artifact = ckpt_artifact
        self.wandb_run_id = wandb_run_id
        self.last_nfe = None

        base_C = self.train_dataset.n_steps * self.train_dataset.num_fields
        self.model = Unet(
            dim=self.image_size,
            channels=base_C,
            init_dim=base_C if self.enc_output else None,
            encoder_flag=self.encoding,
            dim_mults=(1, 2, 4,),
            conditioning=self.conditioning,
            out_dim=2 * base_C,              # v_hat (first C) + eta_hat (last C)
        )
        self.model.to(self.device)
        print(f'ScheduledInterpolantModel: schedule={si_schedule} ({self.si_sched.description}), '
              f'si_base={self.si_base}, si_cond={self._si_cond}, sigma={self.sigma}')

    # ------------------------------------------------------------------
    # Training objective (the inherited loop's one loss hook)
    # ------------------------------------------------------------------

    def _split(self, out, C):
        return out[:, :C], out[:, C:]

    def _si_forward(self, hr, base_mean, x_e, loss_type):
        """hr = x1 (HR-stat space); base_mean = m, from base_hr() (enc(LR) or bicubic per si_base)."""
        B, C = hr.shape[:2]
        t = torch.rand(B, device=hr.device) * T_END
        I_t, v_tgt, n = make_interpolant(self.si_sched, base_mean, hr, t, self.sigma)
        v_hat, eta_hat = self._split(self.model(I_t, t * self.fm_timescale, x_e), C)
        loss_fn = _LOSS[loss_type]
        return loss_fn(v_hat, v_tgt) + loss_fn(eta_hat, n)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _field(self, x_e):
        """(x, t) -> (v_hat, eta_hat), with the conditioning x_e fixed for the whole trajectory."""
        def field(x, t):
            tb = torch.full((x.shape[0],), float(t) * self.fm_timescale, device=x.device)
            return self._split(self.model(x, tb, x_e), x.shape[1])
        return field

    @torch.no_grad()
    def sample(self, upscaled_lr, x_e, lr=None, solver='heun', n_steps=20, eps=0.0, generator=None):
        """One sample per input. Returns (x1_hat in HR-stat space, network evaluations used)."""
        self.model.eval()
        if self.si_base == 'enc' and lr is None:
            raise ValueError("si_base='enc' starts from enc(LR), so the true LR must be passed")
        m = self.base_hr(lr, upscaled_lr)
        zeta = torch.randn(m.shape, device=m.device, dtype=m.dtype, generator=generator)
        x0 = m + self.sigma * zeta
        return integrate(self._field(x_e), self.si_sched, x0, m, self.sigma, solver=solver,
                         n_steps=n_steps, eps=eps, generator=generator)

    def batch_sample(self, dataset, batch, x_e, sampler='heun', n_steps=20, upscaled_lr=None,
                     eps0=0.0, lr=None, eps=None, generator=None, **kwargs):
        """API-compatible with CoupledInterpolantModel.batch_sample: returns a (1, B, C, H, W) stack."""
        if upscaled_lr is None:
            raise ValueError('upscaled_lr is required (it defines the paired base)')
        x1, nfe = self.sample(upscaled_lr.to(self.device).float(), x_e, lr=lr, solver=sampler,
                              n_steps=n_steps, eps=eps0 if eps is None else eps, generator=generator)
        self.last_nfe = nfe
        return x1.unsqueeze(0).cpu()

    # ------------------------------------------------------------------
    # W&B: one deterministic run, one checkpoint artifact
    # ------------------------------------------------------------------

    def _wandb_init(self, wandb_project, wandb_entity, wandb_run_name, cfg):
        if not self.wandb_run_id:
            return super()._wandb_init(wandb_project, wandb_entity, wandb_run_name, cfg)
        cfg = dict(cfg, si_schedule=self.si_schedule_name, ckpt_artifact=self.ckpt_artifact)
        # resume='allow': creates the run on the first launch and resumes the SAME run after any
        # preemption/requeue on any cluster, whether or not the local run directory survived.
        wandb.init(id=self.wandb_run_id, resume='allow', project=wandb_project,
                   entity=wandb_entity or os.getenv('WANDB_ENTITY', ''),
                   name=wandb_run_name or self.wandb_run_id, group='si_schedules', job_type='train',
                   config=cfg)
        with open(self.results_folder / 'wandb_run_id.txt', 'w') as f:
            f.write(wandb.run.id)

    def _upload_state(self, epoch, final=False):
        files = {name: str(self.results_folder / name) for name in CKPT_FILES}
        log_single_version_artifact(
            self.ckpt_artifact, 'checkpoint', files,
            metadata={'epoch': int(epoch), 'final': bool(final), 'si_schedule': self.si_schedule_name,
                      'si_base': self.si_base, 'sigma': self.sigma},
            description=f'Resumable training state for {self.ckpt_artifact} (single version)')

    def _sync_checkpoints(self, epoch, ckpt_path, best_path):
        if not self.ckpt_artifact:
            return super()._sync_checkpoints(epoch, ckpt_path, best_path)
        self._upload_state(epoch)

    def _final_sync(self, ckpt_path, best_path):
        if not self.ckpt_artifact:
            return super()._final_sync(ckpt_path, best_path)
        last = wandb.run.summary.get('last_epoch', -1) if wandb.run is not None else -1
        self._upload_state(last, final=True)
