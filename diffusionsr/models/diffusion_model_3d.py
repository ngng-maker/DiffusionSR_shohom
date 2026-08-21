import math
from functools import partial
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample3D(dim):
    return nn.ConvTranspose3d(dim, dim, 4, 2, 1)


def Downsample3D(dim):
    return nn.Conv3d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvNextBlock3D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv3d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class Attention3D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        batch, _, depth, height, width = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda tensor: rearrange(
                tensor,
                "b (h c) d x y -> b h c (d x y)",
                h=self.heads,
            ),
            qkv,
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(
            out,
            "b h (d x y) c -> b (h c) d x y",
            d=depth,
            x=height,
            y=width,
        )
        return self.to_out(out)


class LinearAttention3D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        _, _, depth, height, width = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda tensor: rearrange(
                tensor,
                "b (h c) d x y -> b h c (d x y)",
                h=self.heads,
            ),
            qkv,
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(
            out,
            "b h c (d x y) -> b (h c) d x y",
            h=self.heads,
            d=depth,
            x=height,
            y=width,
        )
        return self.to_out(out)


class PreNorm3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        encoder_flag,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=3,
        with_time_emb=True,
        use_convnext=True,
        convnext_mult=2,
        conditioning="implicit",
    ):
        super().__init__()
        self.channels = channels
        self.conditioning = conditioning
        self.encoder_flag = encoder_flag

        init_dim = default(init_dim, 64)
        if self.conditioning == "implicit":
            self.init_conv = nn.Conv3d(channels, init_dim, 7, padding=3)
        elif self.conditioning == "explicit":
            if not self.encoder_flag:
                self.init_conv = nn.Conv3d(2 * channels, init_dim, 7, padding=3)
            else:
                self.init_conv = nn.Conv3d(channels + init_dim, init_dim, 7, padding=3)
        else:
            raise ValueError(f"Unknown conditioning mode: {conditioning}")

        self.mish = nn.Mish()
        dims = [init_dim, *map(lambda mult: dim * mult, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ConvNextBlock3D, mult=convnext_mult) if use_convnext else ConvNextBlock3D

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm3D(dim_out, LinearAttention3D(dim_out))),
                        Downsample3D(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm3D(mid_dim, Attention3D(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = index >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm3D(dim_in, LinearAttention3D(dim_in))),
                        Upsample3D(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block_klass(dim, dim), nn.Conv3d(dim, out_dim, 1))

    def forward(self, x, time, x_e=None):
        x = x.float()
        if self.conditioning == "implicit":
            x = self.mish(self.init_conv(x))
            x = x + x_e
        else:
            x = self.init_conv(torch.cat((x, x_e), dim=1))

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        hidden = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            hidden.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, hidden.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)