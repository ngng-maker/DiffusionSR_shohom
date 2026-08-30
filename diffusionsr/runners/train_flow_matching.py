import torch
import torch.nn.functional as F

from diffusionsr.runners.train_diffusion import DiffusionModel, num_to_groups


class FlowMatchingModel(DiffusionModel):
    """Conditional rectified-flow / conditional-OT flow-matching model.

    Differs from DiffusionModel ONLY in the training objective and sampler:
      - DiffusionModel: discrete-timestep DDPM, predicts noise, reverse Markov chain sampling.
      - FlowMatchingModel: continuous t in [0, 1], predicts the velocity field v_theta(x_t, t, x_e)
        along the linear interpolation path x_t = (1 - t) * x_0 + t * noise, sampled via Euler
        integration of the ODE dx/dt = v_theta(x, t, x_e) from t=1 (noise) to t=0 (data).

    Conditioning design note (deliberate, not an oversight):
    This class keeps EXACTLY the same encoder-based conditioning pipeline as DiffusionSR
    (frozen RRDB encoder output x_e, conditioning='implicit' injection in Unet.forward). An
    alternative design was considered and intentionally NOT built here: conditioning flow
    matching directly on bicubic-upscaled LR (x_e = upscaled_lr), mirroring
    implicit_diffusion_direct_noencoder.yml's simpler conditioning. That alternative would make
    this class the flow-matching analogue of vanilla diffusion instead of DiffusionSR. The chosen
    design isolates "DDPM vs. flow matching" as the only difference vs. DiffusionSR (the
    encoder-conditioned model). If the rejected alternative is wanted later, no new training-loop
    code is needed -- DiffusionModel.train()'s `encoding` branch (forwardpass(...) vs.
    upscaled_lr.repeat(...)) is inherited unchanged, so constructing this class with
    encoding=False would produce it directly.
    """

    def __init__(self, *args, fm_timescale=None, **kwargs):
        # `timesteps` (inherited constructor arg) is repurposed for FlowMatchingModel as:
        #   (a) the time-embedding scale constant, via fm_timescale defaulting to timesteps, and
        #   (b) nothing else -- it is NOT a discretization count, since the forward process here
        #       is continuous. The number of Euler steps at sampling time is a separate `n_steps`
        #       argument passed to batch_sample/sample/euler_sample, not `timesteps`.
        super().__init__(*args, **kwargs)
        self.fm_timescale = float(fm_timescale) if fm_timescale is not None else float(self.timesteps)

    # ---- override DDPM-specific scaffolding; everything else (encoder, Unet, train() loop,
    # ---- checkpointing, W&B logging) is inherited unchanged from DiffusionModel ----

    def initialize_variance_schedule(self):
        # No beta/alpha schedule exists for flow matching. Setting self.betas = None (rather than
        # leaving it unset) means any accidental call into inherited DDPM sampling code
        # (p_sample, p_sample_loop, ddim_compute_alpha) fails loudly with an AttributeError-style
        # TypeError instead of silently operating on a stale/wrong schedule.
        self.betas = None

    def sample_t(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def fm_loss(self, denoise_model, x_start, x_e=None, loss_type="l1", t=None, noise=None):
        """
        x_start: x_0, the HR target batch, shape (B, C, H, W).
        Forward process: x_t = (1 - t) * x_0 + t * noise, t ~ Uniform(0, 1).
        Target velocity: noise - x_0 (constant along the linear path).
        """
        if noise is None:
            noise = torch.randn_like(x_start, dtype=torch.float32)
        if t is None:
            t = self.sample_t(x_start.shape[0], x_start.device)
        t_ = t.view(-1, 1, 1, 1)
        x_t = (1 - t_) * x_start + t_ * noise
        target_velocity = noise - x_start
        embedding_t = t * self.fm_timescale
        predicted_velocity = denoise_model(x_t, embedding_t, x_e=x_e)
        if loss_type == "l1":
            loss = F.l1_loss(target_velocity, predicted_velocity)
        elif loss_type == "l2":
            loss = F.mse_loss(target_velocity, predicted_velocity)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(target_velocity, predicted_velocity)
        else:
            raise NotImplementedError(loss_type)
        return loss

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", x_e=None):
        # DiffusionModel.train() (inherited unchanged) calls
        # self.p_losses(self.model, batch, t, loss_type=..., x_e=x_e) where `t` was sampled as
        # torch.randint(0, self.timesteps, ...) for the DDPM case. That integer `t` is intentionally
        # IGNORED here -- we resample our own continuous t inside fm_loss instead, so train()'s call
        # site needs no changes. Flagged explicitly (not a silent bug): the randint call still runs
        # every step and its result is simply discarded.
        return self.fm_loss(denoise_model, x_start, x_e=x_e, loss_type=loss_type, t=None, noise=noise)

    @torch.no_grad()
    def euler_sample(self, model, x_e, shape, n_steps=100):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)  # x_1 = noise, t = 1
        dt = 1.0 / n_steps
        imgs = []
        for i in reversed(range(n_steps)):  # integrate t: 1 -> 0
            t_val = (i + 1) / n_steps
            t_batch = torch.full((shape[0],), t_val, device=device)
            v = model(x, t_batch * self.fm_timescale, x_e)
            x = x - v * dt
            imgs.append(x.cpu())
        return imgs

    @torch.no_grad()
    def sample(self, model, x_e, image_size, n_steps=100, batch_size=16, channels=3):
        return self.euler_sample(model, x_e, shape=(batch_size, channels, image_size, image_size), n_steps=n_steps)

    def batch_sample(self, dataset, batch, x_e, sampler='euler', n_steps=100, **kwargs):
        # Mirrors DiffusionModel.batch_sample's signature/dispatch shape (dataset, batch, x_e,
        # sampler=..., **kwargs) so analysis-layer callers can treat FlowMatchingModel and
        # DiffusionModel instances interchangeably wherever only .batch_sample(...) is called.
        if sampler != 'euler':
            raise NotImplementedError(f"FlowMatchingModel only supports the euler sampler, got {sampler}")
        # Derive batch size from the actual batch so x and x_e shapes agree in euler_sample.
        if batch is not None:
            actual_bs = batch.shape[0]
        elif x_e is not None:
            actual_bs = x_e.shape[0]
        else:
            actual_bs = 1
        imgs = self.euler_sample(self.model, x_e=x_e,
                                 shape=(actual_bs, self.channels, dataset.img_shape, dataset.img_shape),
                                 n_steps=n_steps)
        return torch.stack(imgs, dim=0)
