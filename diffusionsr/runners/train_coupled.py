"""
CoupledInterpolantModel: stochastic interpolant with data-dependent couplings.

Inherits the encoder initialization, U-Net architecture, checkpointing, and
W&B logging from DiffusionModel / FlowMatchingModel.  Overrides ONLY:
  - The training loop (uses SI loss, EMA, grad clipping, per-step LR schedule)
  - The sampling interface (starts from data-dependent x0, not pure noise)

W&B single-run guarantee:
  On first launch the run ID is written to <results_folder>/wandb_run_id.txt.
  On any subsequent restart/requeue, that file is read and the same W&B run
  is resumed (resume='must'), so all training curves accumulate in one place
  with no duplicate runs.

Usage (via CLI):
    python -m diffusionsr.runners.train_srdiff \
        --config configs/stochastic_interpolant/si_opt_a_sigma010_ulr_temp.yml \
        --gpu 0 \
        --modeltype coupled
"""

import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusionsr.models.diffusion_model import Unet
from diffusionsr.runners.train_diffusion import (
    DiffusionModel,
    forwardpass,
    num_to_groups,
    upload_checkpoint_artifact,
    cleanup_old_checkpoint_versions,
    remove_module_prefix,
)
from diffusionsr.runners.train_flow_matching import FlowMatchingModel
from diffusionsr.runners.interpolant import (
    ulr_to_hr_space,
    make_x0,
    interpolate_A,
    velocity_target_A,
    interpolate_B,
    velocity_target_B,
    si_loss_A,
    si_loss_B,
)
from diffusionsr.analysis.sample_coupled import HeunSampler, DopRK45Sampler, EMSampler


class CoupledInterpolantModel(FlowMatchingModel):
    """
    Stochastic interpolant model with data-dependent coupling x0 = U(x_lf) + sigma*zeta.

    Key hyperparameters
    -------------------
    sigma     : float — noise level added to the base x0 (sweep: 0.05, 0.1, 0.2).
    gamma_c   : float — Option-A: 0.0 (default).  Option-B: 0.1 or 0.25.
    si_cond   : str   — 'ulr_concat'  (concat upscaled_lr as conditioning channel, primary)
                       'enc_implicit' (frozen RRDB encoder features, ablation).
    ema_decay : float — EMA weight decay (0.999).
    grad_clip : float — gradient norm clip threshold (1.0).
    """

    def __init__(
        self,
        *args,
        sigma: float = 0.1,
        gamma_c: float = 0.0,
        si_cond: str = 'ulr_concat',
        ema_decay: float = 0.999,
        grad_clip: float = 1.0,
        fm_timescale: float = 1000.0,
        **kwargs,
    ):
        # For 'ulr_concat', the U-Net input is [x_t; ulr_in_hr] = 2*C channels.
        # We double channels before the parent builds the U-Net.
        self._si_cond = si_cond
        self._double_input = (si_cond == 'ulr_concat')

        if self._double_input:
            # Use conditioning='explicit' so Unet.forward concatenates x_e before
            # the first conv.  The Unet is built with channels=C and its first conv
            # is sized channels*2=2*C internally — no channels_override needed.
            kwargs.setdefault('conditioning', 'explicit')

        super().__init__(*args, fm_timescale=fm_timescale, **kwargs)

        self.sigma = sigma
        self.gamma_c = gamma_c
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip

        # For Option B, the U-Net must output 2*C channels (b_hat + g_hat).
        # Rebuild the model with doubled out_dim.
        if gamma_c > 0.0:
            base_C = self.train_dataset.n_steps * self.train_dataset.num_fields
            init_dim = base_C if self.enc_output else None
            # channels=base_C; explicit conditioning doubles input internally to 2*C.
            self.model = Unet(
                dim=self.image_size,
                channels=base_C,
                init_dim=init_dim,
                encoder_flag=self.encoding,
                dim_mults=(1, 2, 4,),
                conditioning=self.conditioning,
                out_dim=base_C * 2,   # dual head: b_hat + g_hat
            )
            self.model.to(self.device)

        # Build stat tensors on CPU; moved to device lazily in ulr_to_hr().
        std_min = 1.0
        def _t(arr):
            return torch.tensor(np.array(arr, dtype=np.float32))

        self._mean_ulr = _t(self.train_dataset.mean_upscaled_lr)
        self._std_ulr  = _t(self.train_dataset.std_upscaled_lr).clamp(min=std_min)
        self._mean_hr  = _t(self.train_dataset.mean_hr)
        self._std_hr   = _t(self.train_dataset.std_hr).clamp(min=std_min)

        # Sampler objects (created lazily on first call)
        self._heun_sampler  = HeunSampler(fm_timescale=fm_timescale)
        self._dop_sampler   = DopRK45Sampler(fm_timescale=fm_timescale)
        self._em_sampler    = EMSampler(fm_timescale=fm_timescale, gamma_c=gamma_c)

    # ------------------------------------------------------------------
    # Re-normalization helper
    # ------------------------------------------------------------------

    def ulr_to_hr(self, upscaled_lr: torch.Tensor) -> torch.Tensor:
        """
        Re-normalize upscaled_lr from its own pixel-wise stats to HF pixel-wise stats.
        Operates in-device (no CPU round-trip).
        """
        dev = upscaled_lr.device
        mu_u = self._mean_ulr.to(dev)
        sd_u = self._std_ulr.to(dev)
        mu_h = self._mean_hr.to(dev)
        sd_h = self._std_hr.to(dev)
        return ulr_to_hr_space(upscaled_lr, mu_u, sd_u, mu_h, sd_h)

    # ------------------------------------------------------------------
    # Conditioning tensor
    # ------------------------------------------------------------------

    def compute_x_e(self, true_lr, upscaled_lr):
        """
        For 'ulr_concat' (primary): return ulr_in_hr_space.
        For 'enc_implicit' (ablation): return frozen encoder features (inherited).
        The U-Net conditioning is handled by the 'explicit'/'implicit' flag set at init.
        """
        if self._si_cond == 'enc_implicit':
            return super().compute_x_e(true_lr, upscaled_lr)
        # ulr_concat: the conditioning tensor IS the upscaled_lr in HF space.
        ulr = upscaled_lr.to(self.device).float()
        return self.ulr_to_hr(ulr)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _si_forward(self, hr, ulr_hr, x_e, loss_type):
        """
        One forward pass of the stochastic interpolant loss.

        hr     : (B, C, H, W) — x1 (HF target), already in HR-stat space.
        ulr_hr : (B, C, H, W) — U(x_lf) in HR-stat space.
        x_e    : (B, ?, H, W) — conditioning tensor.
        """
        B, C = hr.shape[:2]
        device = hr.device

        # Sample continuous time t ~ Uniform(0,1)
        t = torch.rand(B, device=device)
        t_ = t.view(B, 1, 1, 1)   # for broadcasting over (C, H, W)

        # Build x0 = U(x_lf) + sigma * zeta
        x0 = make_x0(ulr_hr, self.sigma)

        if self.gamma_c == 0.0:
            # Option A: straight-line interpolant, no path noise
            x_t = interpolate_A(x0, hr, t)
            vtgt = velocity_target_A(x0, hr)
            t_embed = t * self.fm_timescale
            b_hat = self.model(x_t, t_embed, x_e)
            loss = si_loss_A(b_hat, vtgt, loss_type)
        else:
            # Option B: noisy interpolant, dual-head network
            z = torch.randn_like(hr)
            x_t = interpolate_B(x0, hr, t, z, self.gamma_c)
            vtgt = velocity_target_B(x0, hr, z, t, self.gamma_c)
            t_embed = t * self.fm_timescale
            out = self.model(x_t, t_embed, x_e)   # (B, 2C, H, W)
            b_hat = out[:, :C]
            g_hat = out[:, C:]
            loss = si_loss_B(b_hat, g_hat, vtgt, z, loss_type)

        return loss

    # ------------------------------------------------------------------
    # Training loop (full override)
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int,
        restart: bool = False,
        restart_dir: str = '',
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        loss_type: str = 'l2',
        lr_step_size: int = 1000,
        lr_gamma: float = 0.99,
        wandb_run_name: str = '',
        wandb_entity: str = '',
        wandb_project: str = 'Flow3D_SuperResolution',
        wandb_config: dict | None = None,
    ):
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        # Per-step LR schedule: ×0.99 every 1000 gradient steps (coupling paper)
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_gamma)

        start_epoch = 0
        global_step = 0
        best_val_loss = float('inf')
        all_train_losses, all_val_losses = [], []

        # EMA model (shadow copy of weights, used for evaluation / checkpointing)
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()

        # ------------------------------------------------------------------
        # Restore from checkpoint if restarting
        # ------------------------------------------------------------------
        if restart:
            ckpt_path = os.path.join(restart_dir or str(self.results_folder), 'ckpt.pth')
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(remove_module_prefix(ckpt[0]))
                self.optimizer.load_state_dict(ckpt[1])
                start_epoch = ckpt[2] + 1
                global_step = ckpt[3]
                if len(ckpt) > 4:
                    best_val_loss = ckpt[4]
                if len(ckpt) > 5:
                    ema_model.load_state_dict(remove_module_prefix(ckpt[5]))
                # Restore loss history so .txt files accumulate all epochs
                for (attr, fname) in [(all_train_losses, 'loss_epoch.txt'),
                                       (all_val_losses,  'validation_loss_epoch.txt')]:
                    p = self.results_folder / fname
                    if p.exists():
                        attr += list(np.loadtxt(str(p)).reshape(-1))
                print(f'Resumed from epoch {start_epoch}, global_step={global_step}')

        # ------------------------------------------------------------------
        # W&B: resume or start exactly ONE run per experiment
        # ------------------------------------------------------------------
        wandb_id_file = self.results_folder / 'wandb_run_id.txt'
        if wandb_id_file.exists():
            with open(wandb_id_file) as f:
                run_id = f.read().strip()
            wandb.init(
                id=run_id,
                resume='must',
                project=wandb_project,
                entity=wandb_entity or os.getenv('WANDB_ENTITY', ''),
                name=wandb_run_name or None,
            )
        else:
            cfg = dict(
                sigma=self.sigma, gamma_c=self.gamma_c, si_cond=self._si_cond,
                ema_decay=self.ema_decay, grad_clip=self.grad_clip,
                fm_timescale=self.fm_timescale, epochs=epochs,
                batch_size=batch_size, learning_rate=learning_rate,
                loss_type=loss_type,
                **(wandb_config or {}),
            )
            wandb.init(
                project=wandb_project,
                entity=wandb_entity or os.getenv('WANDB_ENTITY', ''),
                name=wandb_run_name or None,
                config=cfg,
            )
            with open(wandb_id_file, 'w') as f:
                f.write(wandb.run.id)

        # ------------------------------------------------------------------
        # Data loaders
        # ------------------------------------------------------------------
        self.dev_loader = DataLoader(
            self.dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # ------------------------------------------------------------------
        # Epoch loop
        # ------------------------------------------------------------------
        for epoch in tqdm(range(start_epoch, epochs)):
            # Resample the training loader each epoch
            train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            self.model.train()
            train_losses = []

            for step, (res, hr, lr, ulr) in enumerate(train_loader):
                hr  = hr.to(self.device).float()
                lr  = lr.to(self.device).float()
                ulr = ulr.to(self.device).float()

                # Conditioning and x0 source
                ulr_hr = self.ulr_to_hr(ulr)
                x_e = self.compute_x_e(lr, ulr)

                self.optimizer.zero_grad()
                loss = self._si_forward(hr, ulr_hr, x_e, loss_type)
                loss.backward()

                # Gradient norm clipping (prevents explosive updates early in training)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                self.scheduler.step()   # per-step schedule
                global_step += 1

                # EMA update: ema_params = decay * ema_params + (1-decay) * params
                with torch.no_grad():
                    for p, p_ema in zip(self.model.parameters(), ema_model.parameters()):
                        p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

                train_losses.append(loss.item())

            mean_train = float(np.mean(train_losses))
            all_train_losses.append(mean_train)
            np.savetxt(str(self.results_folder / 'loss_epoch.txt'), all_train_losses)

            # ---- Validation ----
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for res, hr, lr, ulr in self.dev_loader:
                    hr  = hr.to(self.device).float()
                    lr  = lr.to(self.device).float()
                    ulr = ulr.to(self.device).float()
                    ulr_hr = self.ulr_to_hr(ulr)
                    x_e    = self.compute_x_e(lr, ulr)
                    val_losses.append(self._si_forward(hr, ulr_hr, x_e, loss_type).item())

            mean_val = float(np.mean(val_losses))
            all_val_losses.append(mean_val)
            np.savetxt(str(self.results_folder / 'validation_loss_epoch.txt'), all_val_losses)

            print(f'Epoch {epoch} | train={mean_train:.4f} | val={mean_val:.4f}')
            wandb.log({'train_loss': mean_train, 'val_loss': mean_val,
                       'lr': self.scheduler.get_last_lr()[0]}, step=epoch)

            # ---- Checkpoint ----
            is_best = mean_val < best_val_loss
            if is_best:
                best_val_loss = mean_val

            states = [
                self.model.state_dict(),
                self.optimizer.state_dict(),
                epoch,
                global_step,
                best_val_loss,
                ema_model.state_dict(),   # save EMA weights alongside main weights
            ]
            ckpt_path = str(self.results_folder / 'ckpt.pth')
            torch.save(states, ckpt_path)
            if is_best:
                torch.save(states, str(self.results_folder / 'bestmodel_saved.pth'))
            _run_name = wandb.run.name if wandb.run is not None else 'run'
            if is_best:
                upload_checkpoint_artifact(ckpt_path, _run_name, epoch, is_best=True)

        if wandb.run is not None:
            cleanup_old_checkpoint_versions(wandb.run.name)
            wandb.finish()

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _make_x0_batch(self, upscaled_lr: torch.Tensor) -> torch.Tensor:
        """Build x0 = ulr_in_hr_space + sigma * zeta for a batch."""
        ulr = upscaled_lr.to(self.device).float()
        ulr_hr = self.ulr_to_hr(ulr)
        return make_x0(ulr_hr, self.sigma)

    @torch.no_grad()
    def heun_sample(self, upscaled_lr: torch.Tensor, x_e: torch.Tensor | None,
                    n_steps: int = 20) -> list[torch.Tensor]:
        x0 = self._make_x0_batch(upscaled_lr)
        return self._heun_sampler.sample(self.model, x0, x_e, n_steps=n_steps)

    @torch.no_grad()
    def dopri5_sample(self, upscaled_lr: torch.Tensor,
                      x_e: torch.Tensor | None) -> list[torch.Tensor]:
        x0 = self._make_x0_batch(upscaled_lr)
        return self._dop_sampler.sample(self.model, x0, x_e)

    @torch.no_grad()
    def em_sample(self, upscaled_lr: torch.Tensor, x_e: torch.Tensor | None,
                  n_steps: int = 100, eps0: float = 0.25) -> list[torch.Tensor]:
        """Euler-Maruyama SDE sampler (Option B). eps0=0 gives the deterministic ODE."""
        if self.gamma_c == 0.0:
            raise ValueError('em_sample is for Option B (gamma_c > 0); use heun_sample for Option A.')
        x0 = self._make_x0_batch(upscaled_lr)
        return self._em_sampler.sample(self.model, x0, x_e, n_steps=n_steps, eps0=eps0)

    def batch_sample(
        self,
        dataset,
        batch,
        x_e,
        sampler: str = 'heun',
        n_steps: int = 20,
        upscaled_lr: torch.Tensor | None = None,
        eps0: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        API-compatible drop-in for DiffusionModel.batch_sample / FlowMatchingModel.batch_sample.

        upscaled_lr must be provided (it is the source distribution for SI sampling).
        If None, a zero tensor is used as a fallback (pure noise mode), which degrades
        the method to vanilla flow matching.
        """
        if upscaled_lr is None:
            # Fallback: no upscaled_lr available — treat as zero-mean source
            shape = (batch.shape[0], dataset.n_steps * dataset.num_fields,
                     dataset.img_shape, dataset.img_shape)
            ulr = torch.zeros(shape, device=self.device)
        else:
            ulr = upscaled_lr

        if sampler == 'heun':
            imgs = self.heun_sample(ulr, x_e, n_steps=n_steps)
        elif sampler == 'dopri5':
            imgs = self.dopri5_sample(ulr, x_e)
        elif sampler == 'em':
            imgs = self.em_sample(ulr, x_e, n_steps=n_steps, eps0=eps0)
        else:
            raise NotImplementedError(f'Unknown sampler={sampler!r}')

        return torch.stack(imgs, dim=0)
