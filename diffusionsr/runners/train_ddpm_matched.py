"""DDPM trained in the *identical* harness as the stochastic-interpolant arms.

WHY THIS EXISTS
---------------
Comparing `--modeltype diffusion` against `--modeltype coupled` is not a controlled experiment.
`DiffusionModel.train()` and `CoupledInterpolantModel.train()` differ in four ways that have nothing
to do with the generative process:

    harness feature        DiffusionModel.train()   CoupledInterpolantModel.train()
    ---------------------  -----------------------  -------------------------------
    EMA of the weights     none                     ema_decay (0.999 by default)
    gradient clipping      none                     clip_grad_norm_(grad_clip)
    LR schedule            fixed                    per-step StepLR(1000, 0.99)
    dataset                SimulationXZDataset      FlippedDataset(p_flip=0.2)

EMA alone usually improves sample quality noticeably, and the flip augmentation changes the
effective dataset size. So an interpolant arm could beat the DDPM baseline on the *harness* while
the generative process made no difference at all -- or made things worse.

This class removes that confound from the other direction: rather than crippling the interpolant
arms, it gives the DDPM everything they have. It subclasses `CoupledInterpolantModel` and overrides
only the two things that define the generative process:

    _si_forward   -> DDPM noising + epsilon-prediction, instead of the interpolant velocity loss
    batch_sample  -> the DDPM/DDIM reverse chain, instead of Euler/Heun integration

Everything else -- EMA, gradient clipping, the per-step LR schedule, FlippedDataset, the
single-W&B-run machinery, checkpointing and restart -- is inherited unchanged, because the training
loop is not touched. The result is an experiment in which the generative process is the only
variable.

`initialize_variance_schedule` is also restored: `CoupledInterpolantModel` inherits
`FlowMatchingModel`'s version, which sets `self.betas = None` as a guard, and DDPM needs a real
beta/alpha schedule.

Usage: add `si_process: "ddpm"` to an otherwise-interpolant config and run `--modeltype coupled`.
See `configs/stochastic_interpolant/baseline_ddpm_matched_temp.yml`.
"""

import torch  # Tensors, RNG, and the no_grad context
import torch.nn.functional as F  # The l1/l2/huber loss functions

from diffusionsr.runners.train_coupled import CoupledInterpolantModel  # The harness being borrowed
from diffusionsr.runners.train_diffusion import DiffusionModel  # Source of the DDPM schedule and sampler


class DDPMMatchedHarnessModel(CoupledInterpolantModel):
    """DDPM generative process, stochastic-interpolant training harness."""

    def __init__(self, *args, **kwargs):
        # gamma_c must be 0. CoupledInterpolantModel rebuilds the U-Net with out_dim = 2*C when
        # gamma_c > 0 (for Option B's dual velocity/denoiser head), but DDPM predicts a single
        # epsilon field and needs out_dim = C. Forcing it here keeps the inherited single-head
        # U-Net, and also means this model is architecturally identical to an Option A arm.
        if float(kwargs.get('gamma_c', 0.0)) != 0.0:
            raise ValueError(
                f"DDPMMatchedHarnessModel requires gamma_c=0 (it predicts a single epsilon field, "
                f"so the U-Net must stay single-head); got gamma_c={kwargs.get('gamma_c')}"
            )
        kwargs['gamma_c'] = 0.0
        super().__init__(*args, **kwargs)
        # sigma is part of the interpolant's base construction and has no DDPM counterpart. It is
        # accepted (so the configs stay structurally identical) but never read by this class.
        print(f"DDPMMatchedHarnessModel: DDPM process in the SI harness "
              f"(ema_decay={self.ema_decay}, grad_clip={self.grad_clip}); si_sigma is ignored")

    def initialize_variance_schedule(self):
        """Restore the DDPM beta/alpha schedule.

        `CoupledInterpolantModel` inherits `FlowMatchingModel.initialize_variance_schedule`, which
        sets `self.betas = None` deliberately so that stray calls into DDPM code fail loudly. This
        model *is* DDPM code, so it needs the real schedule -- delegated to the base implementation
        rather than reimplemented, so the betas, alphas_cumprod and posterior variance are
        bit-identical to what `--modeltype diffusion` would build.
        """
        DiffusionModel.initialize_variance_schedule(self)

    @staticmethod
    def _regression_loss(prediction, target, loss_type: str):
        """The same three loss functions the interpolant's si_loss_A offers, so `loss_type` matches."""
        if loss_type == 'l2':
            return F.mse_loss(prediction, target)  # The repo's SI configs all use l2
        if loss_type == 'l1':
            return F.l1_loss(prediction, target)
        if loss_type == 'huber':
            return F.smooth_l1_loss(prediction, target)
        raise ValueError(f'Unknown loss_type={loss_type!r}')

    def _si_forward(self, hr, ulr_hr, x_e, loss_type):
        """Standard DDPM loss, slotted into the interpolant harness's one loss hook.

        The signature is dictated by the inherited training loop, which calls
        `self._si_forward(hr, ulr_hr, x_e, loss_type)` once per batch. `ulr_hr` is accepted and
        ignored: the DDPM forward process starts from the HR target and walks toward noise, so it
        has no base-density argument. (The loop still computes `ulr_hr` either way, which costs one
        cheap renormalisation per batch and keeps the loop untouched.)

        Deliberately mirrors `DiffusionModel.p_losses` rather than inventing anything: sample an
        integer timestep, noise the target with the closed-form forward process, and regress the
        network's output onto the noise that was added.
        """
        batch_size = hr.shape[0]

        # Integer timestep per sample, uniform over the chain. randint's upper bound is exclusive,
        # so this covers 0 .. timesteps-1, matching DiffusionModel.train()'s sampling.
        t = torch.randint(0, self.timesteps, (batch_size,), device=hr.device).long()

        noise = torch.randn_like(hr)  # The epsilon the network is trained to recover

        # q_sample is inherited from DiffusionModel and applies the closed-form
        # x_t = sqrt(alphas_cumprod[t]) * x_0 + sqrt(1 - alphas_cumprod[t]) * noise,
        # which is why initialize_variance_schedule had to be restored above.
        x_t = self.q_sample(x_start=hr, t=t, noise=noise)

        # The U-Net takes the integer timestep directly here, NOT scaled by fm_timescale. The
        # interpolant arms rescale their continuous t in [0,1] up into the embedding's integer
        # range; DDPM's t is already in that range, so rescaling would double-count.
        predicted_noise = self.model(x_t, t, x_e=x_e)

        return self._regression_loss(predicted_noise, noise, loss_type)

    def batch_sample(self, dataset, batch, x_e, sampler='DDIM', skip=50, **kwargs):
        """Sample with the DDPM/DDIM reverse chain instead of Euler/Heun integration.

        `CoupledInterpolantModel` overrides `batch_sample` for the interpolant samplers, so the
        DDPM implementation is reached explicitly through `DiffusionModel` rather than by `super()`.

        The default `skip=50` gives 1000/50 = 20 effective steps, matching the 20-step budget the
        interpolant arms are evaluated at -- so neither side wins on sampling compute. `**kwargs`
        absorbs interpolant-only arguments (`n_steps`, `eps0`, `upscaled_lr`) that the shared
        evaluation harness passes to every model.
        """
        if sampler not in ('DDPM', 'DDIM'):
            # Guard against an interpolant sampler name silently falling through to DDIM, which
            # would make a logged "heun" run actually be a DDIM run.
            raise NotImplementedError(
                f"DDPMMatchedHarnessModel samples with 'DDPM' or 'DDIM', got {sampler!r}. "
                f"The interpolant samplers (euler/heun/dopri5/em) do not apply to a DDPM."
            )
        return DiffusionModel.batch_sample(self, dataset, batch, x_e, sampler=sampler, skip=skip)
