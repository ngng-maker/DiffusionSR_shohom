import os
from inspect import isfunction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from datasets.dataset import TemperatureXZDataset
from models.diffusion_model import Unet
from einops import rearrange
from models.lr_encoder_model import rrdbnet_encoder
from PIL import Image
from pylab import gca
from torch import einsum, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import (CenterCrop, Compose, Lambda, Resize,
                                    ToPILImage, ToTensor)
from torchvision.utils import save_image
from tqdm.auto import tqdm

# from models.train_rrdn_encoder import pretrain_encoder

#os.environ['CUDA_VISIBLE_DEVICES']  = "6"


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


'''
Make frame thicker, make tick pointing inside, make tick thicker
default frame width is 2, default tick width is 1.5
'''


def frame_tick(frame_width=2, tick_width=1.5):
    ax = gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction='in',
                    width=tick_width)


'''
legend:
default location : upper left
default fontsize: 8
Frame is always off
'''
def forwardpass(lr_enc, sample):
    x = lr_enc.conv1(sample)
    x = lr_enc.trunk(x)
    x = lr_enc.conv2(x)
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    x = lr_enc.upsampling1(x)
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    x = lr_enc.upsampling2(x)
    x = lr_enc.conv3(x)
    return x


def legend(location='upper left', fontsize=8):
    plt.legend(loc=location, fontsize=fontsize, frameon=False)


'''
savefig:
bbox_inches is always tight
'''


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


torch.manual_seed(0)

@torch.no_grad()
def p_sample(model, x, x_e,  t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, x_e) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def savefig(filename):
    plt.savefig(filename, bbox_inches='tight')

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

@torch.no_grad()
def p_sample_loop(model, x_e, shape, timesteps):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, x_e,  torch.full(
            (b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu())
    return imgs

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

@torch.no_grad()
def sample(model, x_e, image_size, timesteps, batch_size=16, channels=3):
    return p_sample_loop(model, x_e, timesteps=timesteps, shape=(batch_size, channels, image_size, image_size))

if schedule == 'linear':
    betas = linear_beta_schedule(timesteps=timesteps)
elif schedule == 'quadratic':
    betas = quadratic_beta_schedule(timesteps=timesteps)
elif schedule == 'sigmoid':
    betas = sigmoid_beta_schedule(timesteps=timesteps)
print("TIMESTEPS == {}, Schedule = {}".format(timesteps, schedule))

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * \
    (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
