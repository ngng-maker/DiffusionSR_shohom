import os
from inspect import isfunction
from pathlib import Path
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusionsr.models.diffusion_model import Unet
from diffusionsr.models.lr_encoder_model import rrdbnet_encoder
from diffusionsr.utils import (
    cosine_beta_schedule,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    upload_checkpoint_artifact,
)
from pylab import gca
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
import wandb
from torch import nn

def remove_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from all keys in a state dictionary.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k.split("module.")[-1]
            new_state_dict[new_key] = v  # Remove the 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def exists(x):
    '''
    Check if array x is not None
    '''
    return x is not None


def default(val, d):
    '''
    If val is not none, return val
    If val is none, return d
    '''
    if exists(val):
        return val
    return d() if isfunction(d) else d





def frame_tick(frame_width=2, tick_width=1.5):
    '''
    Make frame thicker, make tick pointing inside, make tick thicker
    default frame width is 2, default tick width is 1.5
    '''
    ax = gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction='in',
                    width=tick_width)

def forwardpass(lr_enc, sample, factor = 4, output = False, transform_rescale = False, dataset = None):
    '''
    Pass the array "sample" through the RRDB encoder
    '''
   
    if transform_rescale and dataset is None:
        raise AssertionError("Dataset must be specified in order to use transform_rescale option")

    if output:
        if transform_rescale:
            
            unscaled_sample= dataset.unscale_data(sample, input_type = 'lr', maintain_torch = True)
            rescaling_sample = dataset.rescale_data(unscaled_sample, input_type = 'lr', normalize = 'rescaling', maintain_torch = True)
            sample = rescaling_sample
        x = lr_enc(sample)
        if transform_rescale:
            x = dataset.unscale_data(x, input_type = 'hr', normalize = 'rescaling', maintain_torch = True)
            # breakpoint()
            x = dataset.rescale_data(x, input_type = 'hr', normalize = 'standardize', maintain_torch = True)
    else:
        if transform_rescale:
            raise NotImplementedError()
        x = lr_enc.conv1(sample)
        x = lr_enc.trunk(x)
        x = lr_enc.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = lr_enc.upsampling1(x)
        if factor == 4:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = lr_enc.upsampling2(x)
        x = lr_enc.conv3(x)
    return x.float()


def legend(location='upper left', fontsize=8):
    '''
    legend:
    default location : upper left
    default fontsize: 8
    Frame is always off
    '''
    plt.legend(loc=location, fontsize=fontsize, frameon=False)
def savefig(filename):
    '''
    savefig:
    bbox_inches is always tight
    '''
    plt.savefig(filename, bbox_inches='tight')

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionModel():
    def __init__(self,
                 results_folder,
                 lr_encoder_folder,
                 train_dataset,
                 dev_dataset,
                 test_dataset,
                 timesteps=200,
                 conditioning='implicit',
                 encoding = True,
                 schedule='linear',
                 device = 'cuda',
                 enc_output = True,
                 out_steps = None,
                 transform_rescale = False,
                 channels_override = None,
                 image_size_override = None,
                 epoch_subsample_frac = None,
                 ):

        self.results_folder = results_folder
        self.lr_encoder_folder  = lr_encoder_folder
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.timesteps = timesteps
        self.transform_rescale = transform_rescale
        self.encoding = encoding
        self.conditioning = conditioning
        self.schedule = schedule
        self.enc_output = enc_output
        self.image_size = image_size_override if image_size_override is not None else self.train_dataset.img_shape
        self.device = device
        torch.manual_seed(0)
        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        if channels_override is not None:
            self.channels = channels_override
        elif out_steps is None:
            self.channels = self.train_dataset.n_steps*self.train_dataset.num_fields
        else:
            self.channels = out_steps
        if self.encoding:
            print(f"Loading encoder ... encoding = {encoding}")
            self.lr_enc = self.initialize_encoder()
        else:
            print(f'not loading encoder, ..., encoding = {encoding} ')
        if self.enc_output:
            init_dim= self.channels
        else:
            init_dim = None
        self.model = Unet(
            dim=self.image_size,
            channels=self.channels,
            init_dim = init_dim, 
            encoder_flag=self.encoding,
            dim_mults=(1, 2, 4,),
            conditioning=conditioning,
            out_dim=self.train_dataset.n_steps*self.train_dataset.num_fields
        )
        self.initialize_variance_schedule()
        # self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

       
        self.epoch_subsample_frac = epoch_subsample_frac
        self.save_prefix = ''
    def load_saved_model(self):
        print(f"Loading model from {self.results_folder}")
            # breakpoint()
        if os.path.exists(os.path.join(self.results_folder, 'ckpt.pth')):
            print("model found, loading")
            checkpoint = torch.load(os.path.join(self.results_folder, 'ckpt.pth'), map_location=self.device)
            self.model.load_state_dict(checkpoint[0])
            self.optimizer = Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint[1])
            epoch = checkpoint[2]
            self.step = checkpoint[3]
            self.start_epoch = epoch
            self.save_prefix = 'evaluation'
        else:
            print('model not found')    
        
    def initialize_variance_schedule(self):
        if self.schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps)
        elif self.schedule == 'quadratic':
            self.betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif self.schedule == 'sigmoid':
            self.betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        elif self.schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        else:
            raise NotImplementedError(f"Undefined schedule, {self.schedule}")
        print("TIMESTEPS == {}, Schedule = {}".format(self.timesteps, self.schedule))

        # define alphas
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)



    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", x_e=None):
        if noise is None:
            noise = torch.randn_like(x_start, dtype = torch.float32)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, x_e=x_e)
        # breakpoint()
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def compute_x_e(self, true_lr, upscaled_lr):
        """Return the conditioning tensor from LR input. Override in subclasses."""
        if self.encoding:
            return forwardpass(self.lr_enc, true_lr.to(self.device).float(),
                               factor=self.train_dataset.factor, output=self.enc_output,
                               transform_rescale=self.transform_rescale,
                               dataset=self.train_dataset)
        elif self.conditioning == 'none':
            return None
        else:
            return upscaled_lr.to(self.device).float()

    def prepare_batch(self, batch):
        """Return the batch tensor to use for loss computation. Override in subclasses."""
        return batch

    def initialize_encoder(self):
        '''
        Inititalize the low resolution encoder for converting the LR data to the preliminary HR space
        '''
        lr_enc = rrdbnet_encoder(upscale_factor = self.train_dataset.factor, in_channels=self.train_dataset.n_steps*self.train_dataset.num_fields,
                        out_channels=self.train_dataset.n_steps*self.train_dataset.num_fields, num_blocks=8)

        lr_enc.to(self.device)
        # Optimizer is reinitialized here to make sure outputs are reproducible, encoder is not trained
        lr_encoptimizer = torch.optim.Adam(
        lr_enc.parameters(), weight_decay=0)
        lrenc_fname = os.path.join(self.lr_encoder_folder, 'bestmodel_saved.pth')
        if not os.path.exists(lrenc_fname):
            lrenc_fname = os.path.join(self.lr_encoder_folder, 'model_saved.pth')
        lrenc_checkpoint = torch.load(lrenc_fname, map_location=self.device)
        lr_enc.load_state_dict(lrenc_checkpoint['model_state_dict'])
        lr_encoptimizer.load_state_dict(lrenc_checkpoint['optimizer_state_dict'])
        lr_enc.eval()

        for param in lr_enc.parameters():
            param.requires_grad = False
        return lr_enc
    
    @torch.no_grad()
    def p_sample(self,model, x, x_e,  t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, x_e) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self,model, x_e, shape, timesteps):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            img = self.p_sample(model, img, x_e,  torch.full(
                (b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())
        return imgs
    @torch.no_grad()
    def sample(self,model, x_e, image_size, timesteps, batch_size=16, channels=3):
        return self.p_sample_loop(model, x_e, timesteps=timesteps, shape=(batch_size, channels, image_size, image_size))

    def ddim_compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0).to(self.device)
        # print(beta.device, t.device)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    def save_and_log_images(self, split, all_images, epoch, step, hr, true_lr, upscaled_lr, plotting_field='temperature', vmin=293, vmax=5000):
        """
        Saves the model state, generates plots of the images, and logs them to wandb.
        """
        # Select the appropriate dataset
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'validation':
            dataset = self.dev_dataset
        else:
            raise ValueError(f"Invalid split value: {split}")

        # Save model and optimizer states
        states = [
            self.model.state_dict(),
            self.optimizer.state_dict(),
            epoch,
            step,
        ]
        ckpt_filename = os.path.join(self.results_folder, f"ckpt_epoch_{epoch}_step_{step}.pth")
        torch.save(states, ckpt_filename)

        # Get the index for the plotting field
        try:
            temp_idx = dataset.field_names.index(plotting_field)
        except ValueError:
            temp_idx = 0
            plotting_field = dataset.field_names[0]

        # Define a helper function for plotting
        def plot_and_log(image_data, title, filename_suffix):
            plt.figure(dpi=300)
            frame_tick()
            plt.imshow(image_data.T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
            plt.title(f'{title}, Epoch = {epoch}')
            plt.colorbar()
            filename = os.path.join(self.results_folder, f"{self.save_prefix}{split}_{filename_suffix}_{epoch}.png")
            plt.savefig(filename)
            wandb.log({f'{split}_{filename_suffix}': wandb.Image(filename)}, step=epoch)
            plt.clf()

        # Generate and log images
        sample_image = dataset.unscale_data(all_images.numpy()[-1, 0], input_type='hr')[temp_idx]
        plot_and_log(sample_image, 'Conditional Sampled High Resolution', 'individual_sample')

        hr_image = dataset.unscale_data(hr[0].cpu().numpy(), input_type='hr')[temp_idx]
        plot_and_log(hr_image, 'High Resolution Ground Truth', 'hr_sample')

        upscaled_lr_image = dataset.unscale_data(upscaled_lr[0].numpy(), input_type='upscaled_lr')[temp_idx]
        plot_and_log(upscaled_lr_image, 'Low Resolution Upscaled Ground Truth', 'lr_sample')

        true_lr_image = dataset.unscale_data(true_lr[0].numpy(), input_type='lr')[temp_idx]
        plot_and_log(true_lr_image, 'Low Resolution Downscaled Ground Truth', 'lr_downsampled')

        # Additional plotting for multiple channels
        if hr.shape[1] > 1:
            # Implement additional plotting as needed, using the helper function
            pass

    def save(self, split, all_images, epoch, step, res, hr, true_lr, upscaled_lr, plotting_field = 'temperature' ):
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'validation':
            dataset = self.dev_dataset
        states = [
            self.model.state_dict(),
            self.optimizer.state_dict(),
            epoch,
            step,
        ]
        try:
            temp_idx = dataset.field_names.index(plotting_field)
        except ValueError:
            temp_idx = 0
            plotting_field = dataset.field_names[0]

        torch.save(states, os.path.join(self.results_folder, "ckpt.pth"))
        plt.imshow(dataset.unscale_data(all_images.numpy(
        )[-1, 0], input_type='hr')[temp_idx].T, origin='lower', cmap='jet', vmin=293, vmax=5000)


        plt.title(f'Conditional Sampled High Resolution, Epoch = {epoch}')
        plt.colorbar()
        frame_tick()
        plt.savefig(os.path.join(self.results_folder,self.save_prefix +
                    f'{split}-individual-sample-{epoch}.png'))
        wandb.log({f'{split}-individual-sample': wandb.Image(os.path.join(self.results_folder,self.save_prefix +
                    f'{split}-individual-sample-{epoch}.png'))}, step = epoch)
        plt.clf()
                


        plt.imshow((dataset.unscale_data(hr[0].cpu(
        ).numpy(), input_type='hr')[temp_idx]).T, origin='lower', cmap='jet', vmin=293, vmax=5000)
        
        plt.title(f'High Resolution GT, Epoch = {epoch}')
        frame_tick()

        plt.colorbar()
        plt.savefig(os.path.join(self.results_folder,
                   self.save_prefix + f'{split}-hr-sample-{epoch}.png'))
        wandb.log({f'{split}-hr-sample': wandb.Image(os.path.join(self.results_folder,
                   self.save_prefix + f'{split}-hr-sample-{epoch}.png'))}, step = epoch)
        
        plt.clf()
        plt.imshow(dataset.unscale_data(upscaled_lr[0].numpy(
        ), input_type='upscaled_lr')[temp_idx].T, origin='lower', cmap='jet', vmin=293, vmax=5000)
        plt.title(f'Low Resolution Upscaled GT, Epoch = {epoch}')
        plt.colorbar()
        frame_tick()

        plt.savefig(os.path.join(self.results_folder,
                   self.save_prefix + f'{split}-lr-sample-{epoch}.png'))
        wandb.log({f'{split}-lr-sample': wandb.Image(os.path.join(self.results_folder,
                     self.save_prefix + f'{split}-lr-sample-{epoch}.png'))}, step = epoch)
        plt.clf()

        plt.imshow(dataset.unscale_data(true_lr[0].numpy(
        ), input_type='lr')[temp_idx].T, origin='lower', cmap='jet', vmin=293, vmax=5000)
        plt.title(f'Low Resolution Downscaled GT, Epoch = {epoch}')
        plt.colorbar()

        plt.savefig(os.path.join(self.results_folder,
                   self.save_prefix + f'{split}-lr-downsampled-{epoch}.png'))
        wandb.log({f'{split}-lr-downsampled': wandb.Image(os.path.join(self.results_folder,
                     self.save_prefix + f'{split}-lr-downsampled-{epoch}.png'))}, step = epoch)
        plt.clf()

        frame_tick()
        sample = dataset.unscale_data(all_images.numpy()[-1, 0], input_type='hr')
        if hr.shape[1] > 1:
            plt.clf()
            plt.figure(dpi = 300)
            frame_tick()
            plt.imshow(sample[1].T, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
            plt.title(f'Generated Fluid Fraction, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(self.results_folder, f'generated-fluid-fraction-{epoch}.png'))
            wandb.log({"generated_fluid_fraction": wandb.Image(os.path.join(self.results_folder, f'generated-fluid-fraction-{epoch}.png'))}, step = epoch)

            plt.clf()
            plt.figure(dpi = 300)
            frame_tick()
            plt.imshow(sample[1].T > 0.5, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
            plt.title(f'Generated Binarized Fluid Fraction, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(self.results_folder, f'generated-binarized-fluid-fraction-{epoch}.png'))
            plt.clf()
            wandb.log({"generated_binarized_fluid_fraction": wandb.Image(os.path.join(self.results_folder, f'generated-binarized-fluid-fraction-{epoch}.png'))}, step = epoch)
            plt.clf()
            plt.figure(dpi = 300)
            frame_tick()
            plt.imshow((dataset.unscale_data(hr[0].cpu(), input_type = 'hr')[1]).T, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
            # breakpoint()
            # plt.imshow((hr[0].detach().cpu().numpy()*std + mean).T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
            plt.title(f'High Resolution GT Fluid Fraction, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(self.results_folder, f'hr-sample-fluid-fraction-{epoch}.png'))
            wandb.log({"high_res_gt_fluid_fraction": wandb.Image(os.path.join(self.results_folder, f'hr-sample-fluid-fraction-{epoch}.png'))}, step = epoch)
            plt.clf()
            plt.figure(dpi = 300)
            frame_tick()
            plt.imshow(dataset.unscale_data(true_lr[0].cpu().numpy(), input_type = 'lr')[1].T, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
            plt.title(f'Low Resolution Downscaled GT Fluid Fraction, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(self.results_folder, f'lr-downsampled-fluid-fraction-{epoch}.png'))
            wandb.log({"low_res_gt_fluid_fraction": wandb.Image(os.path.join(self.results_folder, f'lr-downsampled-fluid-fraction-{epoch}.png'))}, step = epoch)
            plt.clf()
            plt.close('all')
            plt.figure(dpi= 300)
            frame_tick()
            plt.imshow(sample[1].T, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1, alpha = 0.5)
            plt.imshow(sample[temp_idx].T,  origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000, alpha = 0.5)
            plt.title(f'Overlaid Generated HR Sample and Fluid Fraction, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(self.results_folder, f'overlaid-generated-fluid-fraction-{epoch}.png'))
            plt.clf()
            wandb.log({"overlaid_generated_fluid_fraction": wandb.Image(os.path.join(self.results_folder, f'overlaid-generated-fluid-fraction-{epoch}.png'))}, step = epoch)
            plt.figure(dpi= 300)
            frame_tick()
            plt.imshow((dataset.unscale_data(hr[0].cpu(), input_type = 'hr')[1]).T, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1, alpha = 0.5)
            plt.imshow(dataset.unscale_data(hr[0].cpu().detach().numpy()[0], input_type = 'hr')[temp_idx].T,  origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000, alpha = 0.5)
            plt.title(f'Overlaid GT Sample and Fluid Fraction, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(self.results_folder, f'overlaid-gt-fluid-fraction-{epoch}.png'))
            wandb.log({"overlaid_gt_fluid_fraction": wandb.Image(os.path.join(self.results_folder, f'overlaid-gt-fluid-fraction-{epoch}.png'))}, step = epoch)
            plt.clf()
            plt.figure(dpi= 300)
            frame_tick()
            temperature = sample[temp_idx]
            fluid_fraction = sample[1]
            masked_temperature= np.copy(temperature)
            masked_temperature[fluid_fraction < 0.5] = 293 
            plt.imshow(masked_temperature.T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)

            plt.title(f'Masked Generated HR Sample and Fluid Fraction, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(self.results_folder, f'masked-generated-fluid-fraction-{epoch}.png'))
            plt.clf()
            wandb.log({"masked_generated_fluid_fraction": wandb.Image(os.path.join(self.results_folder, f'masked-generated-fluid-fraction-{epoch}.png'))}, step = epoch)

        scaling_factor = 1

        labels = ['Input', 'Bicubic Upscaling', 'Diffusion', 'Target']
       
        field_idx = 0
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(7.8*scaling_factor, 3*scaling_factor), dpi=300)
        fig.patch.set_alpha(0)

        input = dataset.unscale_data(true_lr[0].numpy(), input_type='lr')[temp_idx].T
        result_diffusion = dataset.unscale_data(all_images.numpy()[-1, 0], input_type='hr')[temp_idx].T
        target = dataset.unscale_data(hr[0].cpu().numpy(), input_type='hr')[temp_idx].T
        upscaled_lr_data = dataset.unscale_data(upscaled_lr[0].cpu().numpy(), input_type = 'upscaled_lr')[temp_idx].T
        for i, (ax, array, label) in enumerate(zip(axs,[input, upscaled_lr_data, result_diffusion, target], labels )):

            if i == 0:
                division_factor = 2
            else:
                division_factor  = 1
            xx, yy = np.meshgrid(np.arange(array.shape[-1])*10*division_factor, np.arange(array.shape[-1])*10*division_factor)
            im = ax.pcolormesh(xx, yy, array, vmin=293, vmax=5000, cmap='jet')
            ax.axis('equal')
            ax.set_ylim([yy.min(), yy.max()])
            ax.set_title(label, fontsize = 15)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft =False)
            ax.set_xticks([])
            ax.set_yticks([])
            if  i == 0:
                ax.set_ylabel(r'z $[\mu m]$')
                ax.set_xlabel(r'x $[\mu m]$')
        fig.subplots_adjust(wspace = 0.01)#, hspace = 0.1)
        cax = fig.add_axes([0.91, 0.12, 0.02, 0.77])
        clb = fig.colorbar(im, cax=cax)
        clb.set_ticks([293, 1000, 2000, 3000, 4000, 5000])
        clb.ax.set_title(r'T$[K]$', fontsize=15)
        plt.savefig(os.path.join(self.results_folder,
                self.save_prefix + f'{split}-panel-{epoch}.png'))
        wandb.log({f'{split}-panel': wandb.Image(os.path.join(self.results_folder,
                    self.save_prefix + f'{split}-panel-{epoch}.png'))}, step = epoch)
        plt.clf()
        
    def batch_sample(self, dataset, batch, x_e, sampler = 'DDPM', skip = None, **kwargs):
        
        timesteps = self.timesteps
        if sampler == 'DDPM':
            batch_size = 2
            batches = num_to_groups(1, batch_size)
            all_images_list = list(map(lambda n: self.sample(self.model, timesteps=timesteps, x_e=x_e,
                                image_size=dataset.img_shape,  batch_size=batch_size, channels=self.channels), batches))[0]
            
        elif sampler == 'DDIM':
            shape = batch.shape
            if skip is None:
                skip = 1
            seq = range(0,timesteps, skip)
            with torch.no_grad():
                x = torch.randn(shape, device=self.device)
                n = x.size(0)
                seq_next = [-1] + list(seq[:-1])
                x0_preds = []
                xs = [x]
                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = self.ddim_compute_alpha(self.betas, t.long())
                    at_next = self.ddim_compute_alpha(self.betas, next_t.long())
                    xt = xs[-1].to('cuda')
                    et = self.model(xt, t, x_e)
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    x0_preds.append(x0_t.to('cpu'))
                    c1 = (
                        kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    )
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                    xs.append(xt_next.to(self.device))

            all_images_list = xs
        else:
            raise NotImplementedError(f"No such sampler, {sampler}")
        all_images = torch.stack(all_images_list, dim=0)
        return all_images


    def sample_and_save(self, batch, res, hr, true_lr, upscaled_lr, x_e, step, epoch, split = 'train'):
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'validation':
            dataset = self.dev_dataset

        all_images = self.batch_sample(dataset, batch[:2], x_e[:2] if x_e is not None else None)
        self.save(split, all_images, epoch = epoch, step = step , res =res, hr = hr, true_lr = true_lr, upscaled_lr = upscaled_lr)
        x_e_str = f'{x_e.min():.3f} {x_e.max():.3f}' if x_e is not None else 'None'
        print(hr.min(), hr.max(), true_lr.min(), true_lr.max(), x_e_str, all_images[-1].min(), all_images[-1].max())
  
    def train(self,
              epochs,
              restart = False,
              restart_dir = '',
              batch_size = 16,
              learning_rate = 1e-5,
              loss_type = 'huber', 
              ):
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.restart = restart
        self.batch_size = batch_size
        self.restart_dir = restart_dir
        # self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.epochs = epochs
        if restart:
            print("Resuming training...")
            checkpoint = torch.load(os.path.join(self.restart_dir, 'ckpt.pth'), map_location=self.device)
            state_dict = remove_module_prefix(checkpoint[0])
            try:
                self.model.load_state_dict(checkpoint[0])
            except:
                self.model.load_state_dict(state_dict)
                
            self.optimizer.load_state_dict(checkpoint[1])
            epoch = checkpoint[2]
            self.step = checkpoint[3]
            self.start_epoch = epoch
            self.save_prefix = 'restart'
        else:
            self.start_epoch = 0
        
        self.dev_loader = DataLoader(
            self.dev_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        all_train_losses = []
        unaveraged_train_losses = []
        all_test_losses = []
        unaveraged_test_losses = []
        best_val_loss = float('inf')

        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            # Resample a random subset of the training set each epoch
            if self.epoch_subsample_frac is not None and self.epoch_subsample_frac < 1.0:
                n_sub = max(self.batch_size, int(self.epoch_subsample_frac * len(self.train_dataset)))
                sub_idx = torch.randperm(len(self.train_dataset))[:n_sub].tolist()
                self.train_loader = DataLoader(
                    Subset(self.train_dataset, sub_idx), batch_size=self.batch_size, shuffle=True, drop_last=True)
            else:
                self.train_loader = DataLoader(
                    self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            losses = []
            
            self.model.train()


            for step, (res, hr, true_lr, upscaled_lr) in tqdm(enumerate(self.train_loader), total = len(self.train_loader)):
                # Create batch for diffusion training
                
                batch = hr
                num_batch = batch.shape[0]
                len_batch = batch.shape[1]
                width_batch = batch.shape[2]
                if len(batch.shape) < 4:
                    batch = torch.reshape(
                        batch, (num_batch, 1, len_batch, width_batch))
                batch = batch.to(self.device).float()
                # Create lr input for conditioning
                if len(true_lr.shape) < 4:
                    true_lr = true_lr.view(
                        true_lr.shape[0], 1, true_lr.shape[1], true_lr.shape[2]).to(self.device).float()
                
                self.optimizer.zero_grad()
                x_e = self.compute_x_e(true_lr, upscaled_lr)
                batch_for_loss = self.prepare_batch(batch)

                # Sample a timestep
                t = torch.randint(0, self.timesteps, (batch_for_loss.shape[0],),
                                device=self.device).long()

                # Calculate loss
                loss = self.p_losses(self.model, batch_for_loss, t, loss_type=self.loss_type, x_e=x_e)

                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()


            # save generated images
            mean_loss = np.mean(losses)
            all_train_losses.append(mean_loss)
            unaveraged_train_losses.extend(losses)

            np.savetxt(os.path.join(self.results_folder, self.save_prefix +
                                    "loss_epoch.txt"), all_train_losses)
            np.savetxt(os.path.join(self.results_folder, self.save_prefix +
                                    "loss_iterations.txt"), unaveraged_train_losses)


            if epoch % 2 == 0:
                self.sample_and_save(batch, res, hr, true_lr, upscaled_lr, x_e, step, epoch, split = 'train')

            test_losses = []
            self.model.eval()
            for step, (res, hr, true_lr, upscaled_lr) in enumerate(self.dev_loader):
                batch = hr
                if len(true_lr.shape) < 4:
                    true_lr = true_lr.view(
                        true_lr.shape[0], 1, true_lr.shape[1], true_lr.shape[2]).to(self.device).float()
                x_e = self.compute_x_e(true_lr, upscaled_lr)

                num_batch = batch.shape[0]
                len_batch = batch.shape[1]
                width_batch = batch.shape[2]
                if len(batch.shape) < 4:
                    batch = torch.reshape(
                        batch, (num_batch, 1, len_batch, width_batch))

                batch = batch.to(self.device)
                batch_for_loss = self.prepare_batch(batch)

                # Algorithm 1 line 3: sample t uniformly for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_for_loss.shape[0],),
                                device=self.device).long()
                loss = self.p_losses(self.model, batch_for_loss, t, loss_type=self.loss_type, x_e=x_e)
                test_losses.append(loss.item())

            test_mean_loss = np.mean(test_losses)
            all_test_losses.append(test_mean_loss)
            unaveraged_test_losses.extend(test_losses)

            np.savetxt(os.path.join(self.results_folder, self.save_prefix +
                                    "validation_loss_epoch.txt"), all_test_losses)
            np.savetxt(os.path.join(self.results_folder, self.save_prefix +
                                    "validation_loss_iterations.txt"), unaveraged_test_losses)
            print("Epoch: {}, Average Train Loss: {:.04}, Average Test Loss: {:.04}".format(
                epoch, mean_loss, test_mean_loss))
            _wb_step = epoch + getattr(self, '_wandb_step_offset', 0)
            wandb.log({'train_loss': mean_loss, 'val_loss': test_mean_loss}, step=_wb_step)

            # Save checkpoint every epoch so SLURM preemption/requeue can restore it.
            states = [self.model.state_dict(), self.optimizer.state_dict(), epoch, step]
            ckpt_path = os.path.join(self.results_folder, "ckpt.pth")
            torch.save(states, ckpt_path)
            is_best = test_mean_loss < best_val_loss
            if is_best:
                best_val_loss = test_mean_loss
                torch.save(states, os.path.join(self.results_folder, "bestmodel_saved.pth"))
            _run_name = wandb.run.name if wandb.run is not None else "run"
            upload_checkpoint_artifact(ckpt_path, _run_name, epoch, is_best=is_best)

            if epoch % 2 == 0:
                  self.sample_and_save(batch, res, hr, true_lr, upscaled_lr, x_e, step, epoch, split = 'validation')

        # After training: generate and upload loss curves to W&B.
        try:
            from diffusionsr.runners.plot_training_curves import plot_curves
            curves_png = os.path.join(self.results_folder, "loss_curves.png")
            plot_curves(self.results_folder, out_path=curves_png)
            if wandb.run is not None and os.path.exists(curves_png):
                wandb.log({"loss_curves": wandb.Image(curves_png)})
        except Exception as _e:
            print(f"Loss-curve plot skipped: {_e}")

