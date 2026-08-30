"""
Latent Diffusion Model (LDM) runner.

Two-stage training:
  Stage 1 (pretrain_vae): train a KL-VAE on HR fields.
  Stage 2 (LDMModel):     freeze RRDB encoder + VAE encoder; train a DDPM U-Net
                           in the 4x-compressed latent space.

Conditioning (matches DiffusionSR):
  x_e = VAE_enc(RRDB(LR))  -- latent embedding, 4 channels, H/4 x W/4
  Injected implicitly: z_after_init_conv + z_e  (same as DiffusionSR in pixel space)

The only variable vs. DiffusionSR is pixel-space vs. latent-space diffusion.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import wandb

from diffusionsr.models.vae_model import VAE2D, vae_loss
from diffusionsr.runners.train_diffusion import DiffusionModel, forwardpass
from diffusionsr.utils import upload_checkpoint_artifact

LATENT_CH = 4  # latent channels; must match VAE latent_channels
_VAE_CHANNEL_MULTS = (1, 2)  # two stride-2 stages → 4x spatial reduction, same as original KLVAE


# ── VAE pretraining ────────────────────────────────────────────────────────────

def pretrain_vae(results_dir, train_dataset, dev_dataset, test_dataset,
                 num_epochs=100, base_ch=64, lr=1e-4, beta_kl=1e-6,
                 epoch_subsample_frac=None):
    """
    Train a KL-VAE on HR fields. Saves:
      vae_ckpt.pth    -- epoch-level restart checkpoint
      vae_best.pth    -- best validation-loss weights
    Skipped if vae_best.pth already exists.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if (results_dir / 'vae_best.pth').exists():
        print(f"VAE already trained at {results_dir} — skipping.")
        return 0

    in_ch = train_dataset.n_steps * train_dataset.num_fields
    vae = VAE2D(input_channels=in_ch, latent_channels=LATENT_CH,
                hidden_channels=base_ch, channel_multipliers=_VAE_CHANNEL_MULTS).cuda()
    opt = Adam(vae.parameters(), lr=lr)

    # Restart from epoch checkpoint if interrupted
    start_epoch = 0
    ckpt_path = results_dir / 'vae_ckpt.pth'
    best_val = float('inf')
    train_losses, val_losses = [], []
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        vae.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', float('inf'))
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        print(f"Resuming VAE from epoch {start_epoch}")

    dev_dl = DataLoader(dev_dataset, batch_size=16, shuffle=False, drop_last=False)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        if epoch_subsample_frac is not None and epoch_subsample_frac < 1.0:
            n_sub = max(16, int(epoch_subsample_frac * len(train_dataset)))
            sub_idx = torch.randperm(len(train_dataset))[:n_sub].tolist()
            train_dl = DataLoader(Subset(train_dataset, sub_idx), batch_size=16, shuffle=True, drop_last=True)
        else:
            train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

        vae.train()
        ep_loss = []
        for _, hr, _, _ in train_dl:
            hr = hr.cuda().float()
            out = vae(hr)
            loss = vae_loss(out, hr, beta=beta_kl)['loss']
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss.append(loss.item())
        train_mean = np.mean(ep_loss)
        train_losses.append(train_mean)

        vae.eval()
        val_ep = []
        with torch.no_grad():
            for _, hr, _, _ in dev_dl:
                hr = hr.cuda().float()
                out = vae(hr)
                loss = vae_loss(out, hr, beta=beta_kl)['loss']
                val_ep.append(loss.item())
        val_mean = np.mean(val_ep)
        val_losses.append(val_mean)

        np.savetxt(results_dir / 'vae_train_loss.txt', train_losses)
        np.savetxt(results_dir / 'vae_val_loss.txt', val_losses)
        wandb.log({'vae_train_loss': train_mean, 'vae_val_loss': val_mean}, step=epoch)
        print(f"VAE epoch {epoch}: train={train_mean:.4f}  val={val_mean:.4f}")

        if val_mean < best_val:
            best_val = val_mean
            best_path = results_dir / 'vae_best.pth'
            torch.save(vae.state_dict(), best_path)
            if wandb.run is not None:
                upload_checkpoint_artifact(str(best_path), wandb.run.name + '_vae',
                                           epoch, is_best=True)

        torch.save({
            'epoch': epoch,
            'model': vae.state_dict(),
            'optimizer': opt.state_dict(),
            'best_val': best_val,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, ckpt_path)
        if wandb.run is not None:
            upload_checkpoint_artifact(str(ckpt_path), wandb.run.name + '_vae',
                                       epoch, is_best=False)

    return num_epochs


# ── LDMModel ──────────────────────────────────────────────────────────────────

class LDMModel(DiffusionModel):
    """
    Latent Diffusion Model: DiffusionModel operating in the 4x-compressed VAE latent space.

    Differences from DiffusionModel:
      - Loads a pre-trained KL-VAE and freezes its weights.
      - prepare_batch(): encodes HR pixel → latent z (mu only, for training stability).
      - compute_x_e(): RRDB(LR) → pixel x_e → VAE_enc(x_e) → latent z_e (4ch, H/4 x W/4).
      - batch_sample(): samples z_0 in latent space, then decodes to pixel-space HR.
      - U-Net channel count = LATENT_CH (4), image_size = HR_size // 4.
    """

    def __init__(self, vae_folder, *args, **kwargs):
        # Override channel count and U-Net dim to match latent space dimensions
        # (set before super().__init__() builds the U-Net)
        kwargs['channels_override'] = LATENT_CH
        # image_size is used as the dim multiplier base in U-Net; scale to latent spatial size
        # We don't have the dataset yet, so we pass image_size_override via a sentinel and
        # handle it after super().__init__() — see _rebuild_model_for_latent().
        super().__init__(*args, **kwargs)
        self.vae = self._load_vae(vae_folder)
        # Rebuild U-Net with correct latent image_size (HR // 4)
        self._rebuild_model_for_latent()

    def _load_vae(self, vae_folder):
        in_ch = self.train_dataset.n_steps * self.train_dataset.num_fields
        vae = VAE2D(input_channels=in_ch, latent_channels=LATENT_CH,
                    channel_multipliers=_VAE_CHANNEL_MULTS).to(self.device)
        vae_path = os.path.join(vae_folder, 'vae_best.pth')
        state = torch.load(vae_path, map_location=self.device)
        vae.load_state_dict(state)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        print(f"Loaded frozen VAE from {vae_path}")
        return vae

    def _rebuild_model_for_latent(self):
        """Replace the pixel-space U-Net built by super().__init__() with a latent-space one."""
        from diffusionsr.models.diffusion_model import Unet
        latent_image_size = self.train_dataset.img_shape // 4
        self.image_size = latent_image_size
        # dim must be even: SinusoidalPositionEmbeddings outputs 2*(dim//2) dims
        unet_dim = latent_image_size if latent_image_size % 2 == 0 else latent_image_size + 1
        self.model = Unet(
            dim=unet_dim,
            channels=LATENT_CH,
            init_dim=LATENT_CH,
            encoder_flag=self.encoding,
            dim_mults=(1,),
            conditioning=self.conditioning,
            out_dim=LATENT_CH,
        ).to(self.device)

    # ── Overrides for latent-space operation ──────────────────────────────────

    def compute_x_e(self, true_lr, upscaled_lr):
        """Return latent conditioning: VAE_enc(RRDB(LR)) -> z_e (B, 4, H/4, W/4)."""
        x_e_pixel = forwardpass(self.lr_enc, true_lr.to(self.device).float(),
                                factor=self.train_dataset.factor, output=True,
                                transform_rescale=self.transform_rescale,
                                dataset=self.train_dataset)
        with torch.no_grad():
            mu_e, _ = self.vae.encode(x_e_pixel)
        return mu_e  # use mean for stable conditioning

    def prepare_batch(self, batch):
        """Encode HR pixel batch -> latent z (mu, no noise for training targets)."""
        with torch.no_grad():
            mu, _ = self.vae.encode(batch.to(self.device).float())
        return mu

    def batch_sample(self, dataset, batch, x_e, sampler='DDIM', skip=None, **kwargs):
        """Sample z_0 in latent space via DDPM/DDIM, then decode to pixel space."""
        latent_h = batch.shape[-2] // 4
        latent_w = batch.shape[-1] // 4
        batch_size = 2

        if sampler == 'DDPM':
            # Parent DDPM uses dataset.img_shape (pixel size) for noise — wrong for latent space.
            # Call p_sample_loop directly with the correct latent spatial size.
            latent_samples = self.p_sample_loop(
                self.model, x_e,
                shape=(batch_size, LATENT_CH, latent_h, latent_w),
                timesteps=self.timesteps)
        else:
            # DDIM uses batch.shape for noise — pass latent_batch so it gets the right size.
            latent_batch = torch.zeros(batch_size, LATENT_CH, latent_h, latent_w, device=self.device)
            latent_samples = super().batch_sample(dataset=dataset, batch=latent_batch,
                                                  x_e=x_e, sampler=sampler, skip=skip, **kwargs)

        # Decode each reverse-step sample back to pixel space
        decoded = []
        for z in latent_samples:
            with torch.no_grad():
                pix = self.vae.decode(z.to(self.device))
            decoded.append(pix.cpu())
        return torch.stack(decoded, dim=0)  # (T, B, C, H, W)
