import os
from inspect import isfunction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from datasets.dataset import TemperatureXZDataset
from models.modified_diffusion import Unet
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

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

'''
savefig:
bbox_inches is always tight
'''


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def train_diffusion(results_folder,
                    lr_encoder_folder,
                    train_dataset,
                    dev_dataset,
                    test_dataset,
                    timesteps=200,
                    restart=False,
                    restart_dir='',
                    conditioning = 'implicit',
                    schedule='linear'):
    print("Now training diffusion model")
    # breakpoint()
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

    # We'll illustrate with a cats image how noise is added at each time step of the diffusion process.

    image_size = 128
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),

    ])

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    # forward diffusion
    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        # use seed for reproducability

    def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(
            figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [image] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title='Original image')
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()

    def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1", x_e=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, x_e=x_e)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok=True)
    image_size = 80
    channels = train_dataset.n_steps

    batch_size = 16

    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # std_lr =  train_dataset.std_lr
    # mean_lr = train_dataset.mean_lr
    # std_resid =  train_dataset.std_resid
    # mean_resid =  train_dataset.mean_resid

    # Algorithm 2 but save all images:

    device = 'cuda'
    lr_enc = rrdbnet_encoder(in_channels=train_dataset.n_steps,
                             out_channels=train_dataset.n_steps, num_blocks=8)

    lr_enc.to(device)
    learning_rate = 5e-5
    lr_encoptimizer = torch.optim.Adam(
        lr_enc.parameters(), lr=learning_rate, weight_decay=0)
    lrenc_fname = os.path.join(lr_encoder_folder, 'model_saved.pth')
    lrenc_checkpoint = torch.load(lrenc_fname)
    lr_enc.load_state_dict(lrenc_checkpoint['model_state_dict'])
    lr_encoptimizer.load_state_dict(lrenc_checkpoint['optimizer_state_dict'])
    lr_enc.eval()

    for param in lr_enc.parameters():
        param.requires_grad = False

    save_and_sample_every = 50

    from torch.optim import Adam

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        conditioning = conditioning,
        out_dim = 1
    )

    model.to(device)
    optimizer = Adam(model.parameters(), lr=5e-5)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    if restart:
        checkpoint = torch.load(os.path.join(restart_dir, 'ckpt.pth'))
        model.load_state_dict(checkpoint[0])
        optimizer.load_state_dict(checkpoint[1])
        epoch = checkpoint[2]
        step = checkpoint[3]
        start_epoch = epoch
        # lr = 5e-6
    epochs = 300
    if not restart:
        start_epoch = 0
    all_train_losses = []
    unaveraged_train_losses = []
    all_test_losses = []
    unaveraged_test_losses = []
    for epoch in tqdm(range(start_epoch, epochs)):
        image_num = 0
        losses = []
        model.train()
        for step, (res, hr, true_lr, upscaled_lr) in enumerate(dataloader):

            batch = hr
            if len(true_lr.shape) < 4:
                true_lr = true_lr.view(
                    true_lr.shape[0], 1, true_lr.shape[1], true_lr.shape[2]).to(device).float()
            if conditioning == 'implicit':
                x_e = forwardpass(lr_enc, true_lr.to(device).float())
            else:
                x_e = upscaled_lr.to(device).float().repeat(1,1,1, 1)
            optimizer.zero_grad()

            num_batch = batch.shape[0]
            len_batch = batch.shape[1]
            width_batch = batch.shape[2]
            if len(batch.shape) < 4:
                batch = torch.reshape(
                    batch, (num_batch, 1, len_batch, width_batch))
            batch = batch.to(device)
            t = torch.randint(0, timesteps, (batch_size,),
                              device=device).long()
            loss = p_losses(model, batch, t, loss_type="huber", x_e=x_e)

            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        # save generated images
        mean_loss = np.mean(losses)
        all_train_losses.append(mean_loss)
        unaveraged_train_losses.extend(losses)

        if restart:
            np.savetxt(os.path.join(results_folder,
                       "reloadedloss_epoch.txt"), all_train_losses)
            np.savetxt(os.path.join(results_folder,
                       "reloadedloss_iterations.txt"), unaveraged_train_losses)
        else:
            np.savetxt(os.path.join(results_folder,
                       "loss_epoch.txt"), all_train_losses)
            np.savetxt(os.path.join(results_folder,
                       "loss_iterations.txt"), unaveraged_train_losses)
        milestone = epoch
        if epoch % 2 == 0:

            batches = num_to_groups(1, batch_size)
            all_images_list = list(map(lambda n: sample(model, timesteps=timesteps, x_e=x_e,
                                   image_size=80,  batch_size=batch.shape[0], channels=channels), batches))[0]
            all_images = torch.stack(all_images_list, dim=0)

            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]

            torch.save(states, os.path.join(results_folder, "ckpt.pth"))
            plt.imshow(train_dataset.unscale_data(all_images.numpy(
            )[-1, 0][0], input_type='hr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            # plt.imshow((all_images.numpy()[-1, 0][0]*std_resid + mean_resid).T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)

            plt.title(f'Conditional Sampled High Resolution, Epoch = {milestone}')
            plt.colorbar()
            frame_tick()
            plt.savefig(os.path.join(results_folder,
                        f'individual-sample-{milestone}.png'))
            plt.clf()
            # plt.imshow((all_images.numpy()[-1, 0][0]*std_resid + mean_resid).T + (upscaled_lr[0].numpy()*std_upscaled + mean_upscaled).T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)

            # plt.imshow(train_dataset.unscale_data(all_images.numpy()[-1, 0][0], input_type='residual').T + train_dataset.unscale_data(
            #     upscaled_lr[0, 0].numpy(), input_type='upscaled_lr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            # plt.title(f'Conditional Sampled HR, Epoch = {milestone}')
            # plt.colorbar()
            # frame_tick()
            # plt.savefig(os.path.join(results_folder,
            #             f'individual-combinedsample-{milestone}.png'))
            # plt.clf()

            plt.imshow((train_dataset.unscale_data(hr[0, 0].cpu(
            ), input_type='hr')).T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            # plt.imshow((hr[0].detach().cpu().numpy()*std + mean).T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
            plt.title(f'High Resolution GT, Epoch = {milestone}')
            frame_tick()

            plt.colorbar()
            plt.savefig(os.path.join(results_folder,
                        f'hr-sample-{milestone}.png'))
            plt.clf()
            plt.imshow(train_dataset.unscale_data(upscaled_lr[0, 0].numpy(
            ), input_type='upscaled_lr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            plt.title(f'Low Resolution Upscaled GT, Epoch = {milestone}')
            plt.colorbar()
            frame_tick()

            plt.savefig(os.path.join(results_folder,
                        f'lr-sample-{milestone}.png'))
            plt.clf()

            plt.imshow(train_dataset.unscale_data(true_lr[0, 0].numpy(
            ), input_type='lr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            plt.title(f'Low Resolution Downscaled GT, Epoch = {milestone}')
            plt.colorbar()

            plt.savefig(os.path.join(results_folder,
                        f'lr-downsampled-{milestone}.png'))
            plt.clf()
            frame_tick()
            if train_dataset.normalize == 'standardize':
                np.savez_compressed(os.path.join(results_folder, f'all_data-{epoch}.npz'), res=res.cpu().numpy(), hr=hr.detach().cpu().numpy(), true_lr=true_lr.cpu().numpy(), upscaled_lr=upscaled_lr.cpu().numpy(
                ), sampled=all_images[-1].detach().cpu().numpy(), mean_lrs=train_dataset.mean_lr, std_lrs=train_dataset.std_lr, mean_resid=train_dataset.mean_resid, std_resid=train_dataset.std_resid)
            if train_dataset.normalize == 'rescaling':
                np.savez_compressed(os.path.join(results_folder, f'all_data-{epoch}.npz'), res=res.cpu().numpy(), hr=hr.detach().cpu().numpy(), true_lr=true_lr.cpu().numpy(), upscaled_lr=upscaled_lr.cpu().numpy(
                ), sampled=all_images[-1].detach().cpu().numpy())
            save_image(all_images[:, 0, ], str(
                results_folder / f'sample-{epoch}.png'), nrow=6)

        test_losses = []

        model.eval()
        for step, (res, hr, true_lr, upscaled_lr) in enumerate(dev_dataloader):
            batch = hr
            if len(true_lr.shape) < 4:
                true_lr = true_lr.view(
                    true_lr.shape[0], 1, true_lr.shape[1], true_lr.shape[2]).to(device).float()
            if conditioning ==  'implicit':
                x_e = forwardpass(lr_enc, true_lr.to(device).float())
            elif conditioning == 'explicit':
                x_e = upscaled_lr.to(device).float().repeat(1,1,1, 1)
            
            num_batch = batch.shape[0]
            len_batch = batch.shape[1]
            width_batch = batch.shape[2]
            if len(batch.shape) < 4:
                batch = torch.reshape(
                    batch, (num_batch, 1, len_batch, width_batch))

            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,),
                              device=device).long()
            loss = p_losses(model, batch, t, loss_type="huber", x_e=x_e)
            test_losses.append(loss.item())

        test_mean_loss = np.mean(test_losses)
        all_test_losses.append(test_mean_loss)
        unaveraged_test_losses.extend(test_losses)
        if restart:
            np.savetxt(os.path.join(results_folder,
                       "reloadedtest_loss_epoch.txt"), all_test_losses)
            np.savetxt(os.path.join(
                results_folder, "reloadedtest_loss_iterations.txt"), unaveraged_test_losses)
        else:
            np.savetxt(os.path.join(results_folder,
                       "test_loss_epoch.txt"), all_test_losses)
            np.savetxt(os.path.join(results_folder,
                       "test_loss_iterations.txt"), unaveraged_test_losses)
        print("Epoch: {}, Average Train Loss: {:.04}, Average Test Loss: {:.04}".format(
            epoch, mean_loss, test_mean_loss))

        if epoch % 2 == 0:

            batches = num_to_groups(1, batch_size)
            all_images_list = list(map(lambda n: sample(model, x_e=x_e, timesteps=timesteps,
                                   image_size=80,  batch_size=batch.shape[0], channels=channels), batches))[0]
            all_images = torch.stack(all_images_list, dim=0)

            # all_images = (all_images + 1) * 0.5
            plt.imshow(dev_dataset.unscale_data(all_images.numpy(
            )[-1, 0][0], input_type='hr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            plt.title(
                f'Test, Conditional High Resolution, Epoch = {milestone}')

            plt.colorbar()
            frame_tick()
            plt.savefig(os.path.join(results_folder,
                        f'test_individual-sample-{milestone}.png'))
            plt.clf()

            # plt.imshow(dev_dataset.unscale_data(all_images.numpy()[-1, 0][0], input_type='residual').T + dev_dataset.unscale_data(
            #     upscaled_lr[0, 0].numpy(), input_type='upscaled_lr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            # plt.title(f'Test, Conditional Sampled HR, Epoch = {milestone}')

            # plt.colorbar()
            # frame_tick()
            # plt.savefig(os.path.join(results_folder,
            #             f'test_individual-combinedsample-{milestone}.png'))
            # plt.clf()

            plt.imshow((dev_dataset.unscale_data(hr[0, 0].cpu(
            ), input_type='hr')).T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            plt.title(f'Test, High Resolution GT, Epoch = {milestone}')
            frame_tick()

            plt.colorbar()
            plt.savefig(os.path.join(results_folder,
                        f'test_hr-sample-{milestone}.png'))
            plt.clf()
            plt.imshow(train_dataset.unscale_data(upscaled_lr[0, 0].numpy(
            ), input_type='upscaled_lr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            plt.title(f'Test, Low Resolution Upscaled GT, Epoch = {milestone}')
            plt.colorbar()
            frame_tick()

            plt.savefig(os.path.join(results_folder,
                        f'test_lr-sample-{milestone}.png'))
            plt.clf()

            plt.imshow(train_dataset.unscale_data(true_lr[0, 0].numpy(
            ), input_type='lr').T, origin='lower', cmap='jet', vmin=293, vmax=5000)
            # plt.imshow((all_images.numpy()[-1][0]*(7000-293) + 293).T, origin = 'lower', cmap = 'jet')
            plt.title(
                f'Test, Low Resolution Downscaledscaled GT, Epoch = {milestone}')
            plt.colorbar()

            plt.savefig(os.path.join(results_folder,
                        f'test_lr-downsampled-{milestone}.png'))
            plt.clf()
            frame_tick()

            # np.savez_compressed(os.path.join(results_folder, f'testall_data-{epoch}.npz'), res=res.cpu().numpy(), hr=hr.detach().cpu().numpy(), true_lr=true_lr.cpu().numpy(), upscaled_lr=upscaled_lr.cpu().numpy(
            # ), sampled=all_images[-1].detach().cpu().numpy(), mean_lrs=mean_lr, std_lrs=std_lr, mean_resid=train_dataset.mean_resid, std_resid=train_dataset.std_resid)
