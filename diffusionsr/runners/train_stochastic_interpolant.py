"""Stochastic interpolants with data-dependent couplings.

Replaces the DDPM generative process entirely. See `FNO/INTERPOLANT_PLAN.md` for the study design.

    Albergo, Boffi & Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for Flows and
      Diffusions", JMLR 26 (2025), arXiv:2303.08797.          -- the framework
    Albergo, Goldstein, Boffi, Ranganath & Vanden-Eijnden, "Stochastic Interpolants with
      Data-Dependent Couplings", ICML 2024, arXiv:2310.03725.  -- the super-resolution instantiation

What is replaced relative to `DiffusionModel`:

    component        DDPM                                   here
    ---------------  -------------------------------------  ------------------------------------
    forward process  x_t = sqrt(abar_t) x0 + sqrt(1-abar) e  I_t = (1-t) x0 + t x1
    schedule         betas / alphas_cumprod, 1000 steps      alpha_t, beta_t; self.betas is None
    training target  the noise epsilon                       velocity x1 - x0, and the latent zeta
    sampler          reverse Markov chain / DDIM             Euler integration of an ODE or SDE
    starting point   pure Gaussian noise                     enc(LR) + sigma*zeta  (the coupling)

Retained from `DiffusionModel`: the U-Net (a function approximator, not part of the generative
process) and the frozen conditioning encoder. `compute_x_e` is inherited unchanged, so conditioning
still arrives as `x_e` exactly as in the paper's `xi`.

The level-1 case of this framework already exists in the repo as `FlowMatchingModel` (linear path,
*independent* coupling, gamma=0, ODE only). This class adds level 2 (the data-dependent coupling)
and level 3 (the SDE with a tunable diffusion coefficient).
"""

import math  # sqrt for the SDE's Brownian increment

import torch  # Tensors, RNG, and the no_grad context used at sampling time
import torch.nn.functional as F  # The l1/l2/huber loss functions

from diffusionsr.runners.train_diffusion import DiffusionModel, forwardpass  # Base class and the encoder-dispatch helper

# Below this value of gamma_eff = alpha_t * sigma the score term is numerically unsafe. With the
# eps_t = eps * alpha_t schedule chosen in `euler_sample` the alpha factors cancel analytically, so
# this is only a guard against sigma itself being set to (near) zero.
_GAMMA_FLOOR = 1e-8


class StochasticInterpolantModel(DiffusionModel):
    """Conditional generative model built on a stochastic interpolant with a coupled base.

    The interpolant is the linear, gamma=0 instantiation used for super-resolution in the couplings
    paper (its Eq. 20):

        I_t = alpha_t * x0 + beta_t * x1,     alpha_t = 1 - t,  beta_t = t,   t in [0, 1]

    so that I_0 = x0 (the base) and I_1 = x1 (the HR target). The time derivative is

        dI/dt = x1 - x0

    which is *constant along the path* -- a property of the linear interpolant, verified numerically
    in `tests/test_stochastic_interpolant.py`.

    Two quantities are learned, both as plain L2-style regressions (couplings paper Thm. 3.1):

        b_t(x) = E[dI/dt | I_t = x]     the velocity -- drives both the ODE and the SDE
        g_t(x) = E[zeta  | I_t = x]     the latent   -- only needed when epsilon > 0

    They share one U-Net whose output width is doubled and split in half, so `epsilon` becomes a
    pure *sampling-time* parameter: one training run covers the whole epsilon sweep, with
    `epsilon = 0` recovering the probability-flow ODE exactly.
    """

    def __init__(
        self,
        *args,  # Forwarded positionally to DiffusionModel (results_folder, lr_encoder_folder, datasets, ...)
        si_base: str = "enc",  # 'enc' = frozen encoder's HR prediction, 'bicubic' = upscaled LR
        si_sigma: float = 1.0,  # Base noise scale; see the class docstring and PLAN section 2.5
        si_epsilon: float = 0.0  ,  # Default SDE diffusion coefficient; 0.0 = deterministic ODE
        si_n_steps: int = 50,  # Default number of Euler steps at sampling time
        si_timescale: float = None,  # Scale applied to t before the U-Net's time embedding
        si_score_weight: float = 1.0,  # Relative weight of the score regression in the total loss
        **kwargs,  # Forwarded as keywords to DiffusionModel (timesteps, conditioning, encoding, ...)
    ) -> None:
        super().__init__(*args, **kwargs)  # Builds the encoder, the (single-head) U-Net, and all training scaffolding

        if si_base not in ("enc", "bicubic"):  # Guard against a typo in a config file
            raise ValueError(f"si_base must be 'enc' or 'bicubic', got {si_base!r}")
        if si_base == "enc" and not self.encoding:
            # The encoder base requires a frozen encoder to have been loaded. Failing here is far
            # better than discovering a None encoder mid-training.
            raise ValueError("si_base='enc' requires encoding=True so the frozen encoder is loaded")

        self.si_base = si_base  # Which coupling to use; the central variable of this study
        self.si_sigma = float(si_sigma)  # Stored as float so a string from YAML cannot sneak through
        self.si_epsilon = float(si_epsilon)  # Default epsilon, overridable per sampling call
        self.si_n_steps = int(si_n_steps)  # Default Euler step count, overridable per sampling call
        self.si_score_weight = float(si_score_weight)  # Balances the two regression terms

        # The U-Net's time embedding was designed for integer DDPM timesteps in [0, timesteps). Our t
        # is continuous in [0, 1], so it is rescaled before embedding -- exactly the trick
        # FlowMatchingModel uses with `fm_timescale`.
        self.si_timescale = float(si_timescale) if si_timescale is not None else float(self.timesteps)

        if self.si_sigma <= 0.0:
            # sigma = 0 puts the base on a lower-dimensional manifold, where the score is singular
            # (couplings paper section 4.2). Refuse rather than produce NaNs later.
            raise ValueError(f"si_sigma must be > 0 (the base would be degenerate otherwise), got {self.si_sigma}")

        self._rebuild_two_head_unet()  # Replace the single-head U-Net super().__init__ built

        # Holds (x0, zeta) for the current batch. See `compute_x_e` for the invariant that keeps it
        # valid, and `_take_cached_base` for the guard that detects a stale one.
        self._si_base_cache = None

    # ── Architecture ──────────────────────────────────────────────────────────────────────────

    def _rebuild_two_head_unet(self) -> None:
        """Swap the inherited U-Net for one with a doubled output width.

        The velocity and the latent are both fields of the same shape as the data, so rather than
        training two networks they are emitted as two halves of one output tensor. Follows the same
        rebuild-after-super pattern `LDMModel._rebuild_model_for_latent` uses.
        """
        from diffusionsr.models.diffusion_model import Unet  # Imported locally to mirror LDMModel and avoid a cycle

        field_channels = self.train_dataset.n_steps * self.train_dataset.num_fields  # Width of one field tensor
        if field_channels != self.channels:
            # A channels_override (used by LDMModel for latent-space operation) would break the
            # chunk(2) split below, so reject it explicitly rather than mis-slicing.
            raise ValueError(
                f"StochasticInterpolantModel expects channels == n_steps*num_fields "
                f"({self.channels} != {field_channels}); channels_override is not supported"
            )
        self._si_field_channels = field_channels  # Stored so the sampler knows where to split the output

        # Reproduce DiffusionModel's init_dim logic exactly: enc_output=True conditions on the
        # encoder's final field (so init_dim is the field width), otherwise on its 64-channel
        # intermediate (so init_dim falls back to Unet's default of 64).
        init_dim = self.channels if self.enc_output else None

        self.model = Unet(
            dim=self.image_size,  # Same spatial base dimension as the inherited model
            channels=self.channels,  # Input width: one field tensor (the interpolant I_t)
            init_dim=init_dim,  # Must match the conditioning tensor's channel count
            encoder_flag=self.encoding,  # Whether x_e conditioning is wired in
            dim_mults=(1, 2, 4,),  # Identical to DiffusionModel, so capacity is comparable
            conditioning=self.conditioning,  # 'implicit' (added) or 'explicit' (concatenated)
            out_dim=2 * field_channels,  # [velocity | latent] stacked on the channel axis
        ).to(self.device)

    def initialize_variance_schedule(self) -> None:
        """No beta/alpha schedule exists for an interpolant.

        Setting `self.betas = None` rather than leaving it unset means any accidental call into
        inherited DDPM sampling code (`p_sample`, `p_sample_loop`, `ddim_compute_alpha`) fails loudly
        instead of silently operating on a stale schedule. Same guard `FlowMatchingModel` uses.
        """
        self.betas = None

    # ── The coupling ──────────────────────────────────────────────────────────────────────────

    def compute_base(self, true_lr, upscaled_lr):
        """Draw the base sample x0 conditionally on the target -- i.e. the data-dependent coupling.

        Returns `(x0, zeta)`. `zeta` is returned as well as used because it is the regression target
        for the score head, so it must be the *same* draw that went into x0.

        The two modes are the two treatment arms of the study:
          'enc'     x0 = enc(LR) + sigma*zeta   -- the frozen encoder's own HR field prediction.
                    This is the study's contribution; the couplings paper has no encoder and uses
                    bicubic. Strongest coupling, so the shortest transport.
          'bicubic' x0 = U(D(x1)) + sigma*zeta  -- exactly the paper's section 4.2 setup, used here
                    as the control that isolates whether the *encoder* matters or merely proximity.
        """
        if self.si_base == "enc":
            # output=True selects the encoder's final field prediction rather than its 64-channel
            # intermediate. The intermediate is what conditioning uses; the field is what the
            # transport starts from. Both come from the same frozen encoder at no extra cost.
            m = forwardpass(
                self.lr_enc,  # The frozen encoder loaded by DiffusionModel.initialize_encoder
                true_lr.to(self.device).float(),  # Native-resolution LR input
                factor=self.train_dataset.factor,  # HR/LR ratio, forwarded for RRDB's upsampling depth
                output=True,  # Final field prediction, not the conditioning features
                transform_rescale=self.transform_rescale,  # Honour the inherited normalisation option
                dataset=self.train_dataset,  # Required by forwardpass when transform_rescale is set
            )
        else:  # 'bicubic' -- validated in __init__, so no further branch is needed
            m = upscaled_lr.to(self.device).float()  # The dataset's bicubic-upscaled LR, already HR-sized

        # zeta smooths the base off the lower-dimensional manifold it would otherwise sit on, and is
        # what keeps the score well-defined despite gamma_t = 0 (couplings paper section 3.2).
        zeta = torch.randn_like(m)
        return m + self.si_sigma * zeta, zeta

    def compute_x_e(self, true_lr, upscaled_lr):
        """Return the conditioning tensor, and cache the base for this batch as a side effect.

        INVARIANT, and the reason this is safe: `DiffusionModel.train()` calls `compute_x_e` and then
        `p_losses` on the same batch with nothing in between (train_diffusion.py around lines 772-780
        and 808-823), and `analysis/eval_predictions.py` likewise calls `compute_x_e` immediately
        before `batch_sample`. The base therefore cannot be stale by the time it is read.

        `p_losses` cannot compute the base itself because it never receives the LR tensors, and
        duplicating the whole inherited training loop just to thread two extra arguments through
        would be a far larger and more fragile change than this cache. `_take_cached_base` verifies
        the batch size as a guard against the invariant ever being broken.
        """
        x_e = super().compute_x_e(true_lr, upscaled_lr)  # Unchanged conditioning path
        self._si_base_cache = self.compute_base(true_lr, upscaled_lr)  # (x0, zeta) for this batch
        return x_e

    def _take_cached_base(self, expected_batch_size: int):
        """Read the cached base, checking it belongs to the batch being processed."""
        if self._si_base_cache is None:
            raise RuntimeError(
                "StochasticInterpolantModel: no base cached. compute_x_e(true_lr, upscaled_lr) must "
                "be called immediately before p_losses/batch_sample so the coupled base is available."
            )
        x0, zeta = self._si_base_cache
        if x0.shape[0] != expected_batch_size:
            # Catches the realistic failure mode: a cache left over from a differently sized batch.
            raise RuntimeError(
                f"StochasticInterpolantModel: cached base has batch size {x0.shape[0]} but the "
                f"current batch has {expected_batch_size}. The compute_x_e -> loss/sample invariant "
                f"has been broken."
            )
        return x0, zeta

    # ── Training ──────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _regression_loss(prediction, target, loss_type: str):
        """One of the three loss functions the repo's configs select between."""
        if loss_type == "l1":
            return F.l1_loss(prediction, target)  # Mean absolute error
        if loss_type == "l2":
            return F.mse_loss(prediction, target)  # Mean squared error -- the objective the theory assumes
        if loss_type == "huber":
            return F.smooth_l1_loss(prediction, target)  # L2 near zero, L1 in the tails
        raise NotImplementedError(loss_type)

    def interpolant_loss(self, denoise_model, x1, x0, zeta, x_e=None, loss_type="l1"):
        """The two quadratic regressions of couplings paper Thm. 3.1, on one shared network."""
        x1 = x1.to(self.device).float()  # The HR target
        x0 = x0.to(self.device).float()  # The coupled base

        # t ~ Uniform(0,1), one draw per sample in the batch. torch.rand (continuous) rather than
        # torch.randint, because an interpolant has no discretisation.
        t = torch.rand(x1.shape[0], device=self.device)
        t_ = t.view(-1, 1, 1, 1)  # Reshape to (B,1,1,1) so it broadcasts over channels and space

        interpolant = (1.0 - t_) * x0 + t_ * x1  # I_t, with alpha_t = 1-t and beta_t = t
        velocity_target = x1 - x0  # dI/dt for a linear interpolant -- constant in t

        # Rescale t into the range the U-Net's sinusoidal time embedding was built for.
        prediction = denoise_model(interpolant, t * self.si_timescale, x_e=x_e)

        # chunk(2, dim=1) splits the doubled output width back into the two heads. Order matches
        # `euler_sample`; swapping them would train the score as the velocity and vice versa.
        velocity_prediction, latent_prediction = prediction.chunk(2, dim=1)

        loss_velocity = self._regression_loss(velocity_prediction, velocity_target, loss_type)
        loss_latent = self._regression_loss(latent_prediction, zeta, loss_type)

        # The score head is only consumed when epsilon > 0, but it is trained unconditionally so
        # that epsilon stays a sampling-time knob -- one training run covers the whole sweep.
        return loss_velocity + self.si_score_weight * loss_latent

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", x_e=None):
        """Inherited-signature entry point so `DiffusionModel.train()` needs no changes.

        The integer `t` the training loop samples via `torch.randint(0, self.timesteps, ...)` is
        intentionally IGNORED -- an interpolant uses continuous t, resampled inside
        `interpolant_loss`. Flagged explicitly rather than left as a silent surprise: the randint
        call still runs each step and its result is discarded. `noise` is likewise unused, since the
        latent comes from `compute_base` where it is tied to x0.
        """
        x0, zeta = self._take_cached_base(x_start.shape[0])
        return self.interpolant_loss(denoise_model, x_start, x0, zeta, x_e=x_e, loss_type=loss_type)

    # ── Sampling ──────────────────────────────────────────────────────────────────────────────

    @torch.no_grad()  # Sampling never needs gradients; this also halves memory
    def euler_sample(self, model, x_e, x0, n_steps=None, epsilon=None):
        """Integrate base -> target with forward Euler, as an ODE (epsilon=0) or an SDE (epsilon>0).

        Deterministic case, the probability flow of couplings paper Eq. 8:

            dX/dt = b_t(X)

        Stochastic case, their forward SDE Eq. 11, using score = -(alpha_t*sigma)^-1 * g_t:

            dX = [ b_t(X) - (eps_t / (alpha_t*sigma)) * g_t(X) ] dt + sqrt(2*eps_t) dW

        DESIGN CHOICE -- the schedule of the diffusion coefficient. The score term carries a factor
        1/alpha_t, which diverges as t -> 1 (where alpha_t -> 0). Taking

            eps_t = epsilon * alpha_t

        makes the alpha factors cancel exactly, leaving a bounded drift correction -(epsilon/sigma)*g,
        and drives the injected noise sqrt(2*epsilon*alpha_t) to zero at t=1 as the framework's
        boundary conditions require. One dimensionless knob, no singularity at either end. This
        particular choice is ours, not stated verbatim in the paper, so it is recorded here.
        """
        n_steps = int(self.si_n_steps if n_steps is None else n_steps)  # Allow a per-call override
        eps = float(self.si_epsilon if epsilon is None else epsilon)  # Likewise for epsilon
        if eps < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {eps}")

        x = x0.clone().to(self.device).float()  # Start AT THE BASE -- not at noise. This is the coupling.
        dt = 1.0 / n_steps  # Uniform step size over t in [0, 1]
        trajectory = []  # Collected per-step states, matching the inherited samplers' return shape

        for step_index in range(n_steps):  # Integrate forward in time, t: 0 -> 1
            t_value = step_index / n_steps  # Left endpoint of this step (explicit Euler)
            alpha_t = 1.0 - t_value  # The interpolant's coefficient on x0 at this time
            # torch.full builds a (B,) tensor of the same t for every sample in the batch, which is
            # what the time embedding expects.
            t_batch = torch.full((x.shape[0],), t_value, device=self.device)

            prediction = model(x, t_batch * self.si_timescale, x_e)  # Same rescaling as training
            velocity, latent = prediction.chunk(2, dim=1)  # Split order must match interpolant_loss

            drift = velocity  # The ODE drift
            if eps > 0.0:
                gamma_eff = alpha_t * self.si_sigma  # The effective latent coefficient
                if gamma_eff > _GAMMA_FLOOR:
                    # eps_t = eps * alpha_t, so eps_t / gamma_eff = eps / sigma -- alpha cancels.
                    drift = drift - (eps / self.si_sigma) * latent
                # Brownian increment with variance 2*eps_t*dt; vanishes as alpha_t -> 0 at t=1.
                noise_scale = math.sqrt(2.0 * eps * alpha_t * dt)
                x = x + drift * dt + noise_scale * torch.randn_like(x)
            else:
                x = x + drift * dt  # Pure probability-flow ODE

            trajectory.append(x.cpu())  # Moved to CPU to keep long trajectories off the GPU

        return trajectory

    @torch.no_grad()
    def sample(self, model, x_e, image_size, n_steps=None, batch_size=16, channels=3, x0=None, epsilon=None):
        """Shape-based entry point mirroring the inherited `sample` signature."""
        if x0 is None:
            # Falls back to the cached base when the caller has not supplied one explicitly.
            x0, _ = self._take_cached_base(batch_size)
        return self.euler_sample(model, x_e, x0=x0, n_steps=n_steps, epsilon=epsilon)

    def batch_sample(self, dataset, batch, x_e, sampler="euler", n_steps=None, epsilon=None, x0=None, **kwargs):
        """API-compatible with `DiffusionModel.batch_sample` / `FlowMatchingModel.batch_sample`.

        Keeping the `(dataset, batch, x_e, sampler=..., **kwargs)` shape means the analysis layer
        (`analysis/eval_predictions.py` and the result notebooks) can treat this class
        interchangeably with the DDPM and flow-matching models.

        `x0` may be passed explicitly; otherwise the base cached by `compute_x_e` is used, which is
        how `eval_predictions.py` already calls it (compute_x_e, then batch_sample).
        """
        if sampler != "euler":
            # No other integrator is implemented. Fail loudly rather than quietly using Euler.
            raise NotImplementedError(f"StochasticInterpolantModel only supports sampler='euler', got {sampler!r}")

        # Derive the batch size from whatever was actually provided, so x and x_e shapes agree.
        if batch is not None:
            actual_batch_size = batch.shape[0]
        elif x_e is not None:
            actual_batch_size = x_e.shape[0]
        elif x0 is not None:
            actual_batch_size = x0.shape[0]
        else:
            raise ValueError("batch_sample needs at least one of batch, x_e or x0 to infer the batch size")

        if x0 is None:
            x0, _ = self._take_cached_base(actual_batch_size)

        trajectory = self.euler_sample(self.model, x_e=x_e, x0=x0, n_steps=n_steps, epsilon=epsilon)
        # torch.stack along a new leading axis gives (n_steps, B, C, H, W), the shape the inherited
        # samplers return and which callers index with [-1] to get the final sample.
        return torch.stack(trajectory, dim=0)
