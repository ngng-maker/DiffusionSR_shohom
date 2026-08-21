"""
KL-VAE for Latent Diffusion Model.

Architecture (Rombach et al. 2022 SR variant):
  Encoder: (B, C, H, W) -> mu, logvar each (B, latent_ch, H/4, W/4)
           Two stride-2 downsampling stages with residual blocks + self-attention bottleneck.
  Decoder: (B, latent_ch, H/4, W/4) -> (B, C, H, W), mirror of encoder.
  Loss: L1 reconstruction + beta * KL(q(z|x) || N(0,I)), beta=1e-6.

Spatial downsampling factor is 4x. HR images must be divisible by 4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch):
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            _gn(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            _gn(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SelfAttn(nn.Module):
    """Single-head self-attention (efficient for small spatial dims)."""
    def __init__(self, ch):
        super().__init__()
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, -1).transpose(1, 2)   # (B, HW, C)
        k = k.reshape(B, C, -1)                    # (B, C, HW)
        v = v.reshape(B, C, -1).transpose(1, 2)   # (B, HW, C)
        attn = (q @ k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class VAEEncoder(nn.Module):
    def __init__(self, in_channels, base_ch=64, latent_ch=4):
        super().__init__()
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, padding=1),
            ResBlock(ch),
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),   # H -> H/2
            ResBlock(ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1),  # H/2 -> H/4
            ResBlock(ch * 4),
            SelfAttn(ch * 4),
            ResBlock(ch * 4),
            _gn(ch * 4), nn.SiLU(),
        )
        self.mu_proj = nn.Conv2d(ch * 4, latent_ch, 1)
        self.logvar_proj = nn.Conv2d(ch * 4, latent_ch, 1)

    def forward(self, x):
        h = self.net(x.float())
        return self.mu_proj(h), self.logvar_proj(h)


class VAEDecoder(nn.Module):
    def __init__(self, out_channels, base_ch=64, latent_ch=4):
        super().__init__()
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, ch * 4, 3, padding=1),
            ResBlock(ch * 4),
            SelfAttn(ch * 4),
            ResBlock(ch * 4),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),  # H/4 -> H/2
            ResBlock(ch * 2),
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),       # H/2 -> H
            ResBlock(ch),
            _gn(ch), nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, z):
        return self.net(z.float())


class KLVAE(nn.Module):
    """
    KL-regularized VAE with 4x spatial downsampling.
    Used as stage 1 of the Latent Diffusion Model.
    """
    LATENT_CH = 4

    def __init__(self, in_channels, base_ch=64, latent_ch=4):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder = VAEEncoder(in_channels, base_ch, latent_ch)
        self.decoder = VAEDecoder(in_channels, base_ch, latent_ch)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss(recon, target, mu, logvar, beta=1e-6):
        recon_loss = F.l1_loss(recon, target.float())
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss.item(), kl.item()
