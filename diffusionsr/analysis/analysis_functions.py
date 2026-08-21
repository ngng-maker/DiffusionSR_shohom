import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusionsr.models.diffusion_model import Unet
from diffusionsr.models.lr_encoder_model import rrdbnet_encoder
from diffusionsr.models.mobilenet_model import MobileNetv2_SISR
from tqdm import tqdm
from diffusionsr.datasets.dataset import SimulationXZDataset
from diffusionsr.runners.train_diffusion import forwardpass, num_to_groups
from skimage.metrics import structural_similarity as ssim_id
from diffusionsr.runners.train_diffusion import DiffusionModel
from diffusionsr.analysis.plotting_functions import frame_tick, legend
from diffusionsr.analysis.metrics import PSNR, SSIM
from diffusionsr.utils import (
    cosine_beta_schedule,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)

device = 'cuda'



def predict_mobilenet(model, res, hr, lr, upscaled_lr, dataset):

    '''
    Return the predictions for the MobileNet model given an input batch
    Parameters:
        model: Torch network module object, representing the trained MobileNet model
        res: Torch tensor, Residual between HR and LR data
        hr: Torch tensor, High Resolution data
        lr: Torch tensor, Low Resolution data
        upscaled_lr: Torch tensor, Bicubic upscaled low resolution data
        dataset: Dataset object, used for rescaling data
    Returns:
        Low resolution data, scaled to original space, 4-D numpy array (batch, channels, height, width)
        Output (Super-resolution), scaled to original space, 4-D numpy array
        High resolution data, scaled to original space, 4-D numpy array 
    '''


    img = upscaled_lr.view(upscaled_lr.shape[0], 1, upscaled_lr.shape[1], upscaled_lr.shape[2])
    target = hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2])
    img = img.to(device)
    target = target.to(device)
    output = model(img)
    return dataset.unscale_data(lr, input_type='lr'), dataset.unscale_data(output.detach().cpu(), input_type = 'hr'), dataset.unscale_data(target.cpu(), input_type = 'hr') 


def predict_lrenc(lr_enc, res, hr, lr, upscaled_lr, dataset):

    '''
    Return the predictions for the MobileNet model given an input batch
    Parameters:
        lr_enc: Torch network module object, representing the trained RRDN encoder model
        res: Torch tensor, Residual between HR and LR data
        hr: Torch tensor, High Resolution data
        lr: Torch tensor, Low Resolution data
        upscaled_lr: Torch tensor, Bicubic upscaled low resolution data
        dataset: Dataset object, used for rescaling data
    Returns:
        Low resolution data, scaled to original space, 4-D numpy array (batch, channels, height, width)
        Output (Super-resolution), scaled to original space, 4-D numpy array
        High resolution data, scaled to original space, 4-D numpy array 
    '''

    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
    else:
        img = lr.to(device)
    if len(hr.shape) < 4:
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        target = hr.to(device)
    x_e = lr_enc(img.float())

    return dataset.unscale_data(lr, input_type='lr'), dataset.unscale_data(x_e.detach().cpu(), input_type = 'hr'), dataset.unscale_data(target.cpu(), input_type = 'hr')
def predict_diffusion(model, lr_enc, res, hr, lr, upscaled_lr, dataset, timesteps = 200, schedule = 'linear'):
    '''
    Return the predictions for the MobileNet model given an input batch
    Parameters:
        lr_enc: Torch network module object, representing the trained RRDN encoder model
        res: Torch tensor, Residual between HR and LR data
        hr: Torch tensor, High Resolution data
        lr: Torch tensor, Low Resolution data
        upscaled_lr: Torch tensor, Bicubic upscaled low resolution data
        dataset: Dataset object, used for rescaling data
    Returns:
        Low resolution data, scaled to original space, 4-D numpy array (batch, channels, height, width)
        Output (Super-resolution), scaled to original space, 4-D numpy array
        High resolution data, scaled to original space, 4-D numpy array 
    '''

    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        img = lr.to(device)
        target = hr.to(device)
    if len(lr.shape)< 4:

        x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float(), factor = dataset.factor)
    else:
        x_e = forwardpass(lr_enc, lr.to(device).float(), factor = dataset.factor)
    batches = num_to_groups(1, lr.shape[0])

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

    @torch.no_grad()
    def sample(model, x_e, image_size, timesteps, batch_size=16, channels=3):
        return p_sample_loop(model, x_e, timesteps=timesteps, shape=(batch_size, channels, image_size, image_size))

    if schedule == 'linear':
        betas = linear_beta_schedule(timesteps=timesteps)
    elif schedule == 'quadratic':
        betas = quadratic_beta_schedule(timesteps=timesteps)
    elif schedule == 'cosine':
        betas = quadratic_beta_schedule(timesteps=timesteps)
    elif schedule == 'sigmoid':
        betas = sigmoid_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
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


    all_images_list = list(map(lambda n: sample(model, x_e = x_e, timesteps = timesteps, image_size = 80,  batch_size=img.shape[0], channels=1), batches))[0]
    all_images = torch.stack(all_images_list, dim=0)
    result = dataset.unscale_data(all_images.numpy()[-1, 0], input_type = 'residual') + dataset.unscale_data(upscaled_lr.numpy(), input_type = 'upscaled_lr')
    
    return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr')


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def predict_ddim_diffusion(model, lr_enc, res, hr, lr, upscaled_lr, dataset, seq, timesteps = 200, skip = 1, schedule = 'linear',transform_rescale = False,  **kwargs):
    
    # skip =timesteps // self.args.timesteps
    seq = range(0, timesteps, skip)
    
    if schedule == 'linear':
        betas = linear_beta_schedule(timesteps=timesteps)
    elif schedule == 'quadratic':
        betas = quadratic_beta_schedule(timesteps=timesteps)
    elif schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps=timesteps)
    elif schedule == 'sigmoid':
        betas = sigmoid_beta_schedule(timesteps=timesteps)
    b = betas

    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        img = lr.to(device)
        target = hr.to(device)
    if len(lr.shape)< 4:

        x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float(), factor = dataset.factor, transform_rescale=transform_rescale, dataset = dataset)
    else:
        x_e = forwardpass(lr_enc, lr.to(device).float(), factor = dataset.factor, transform_rescale=transform_rescale, dataset = dataset)
    batches = num_to_groups(1, lr.shape[0])
    shape=hr.shape
    # print(timesteps, batches, img.shape[0])
    # print(x_e.shape)
    with torch.no_grad():
        n = img.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        
        x = torch.randn(shape, device=device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(x, t, x_e)#model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    # return xs, x0_preds
    result = dataset.unscale_data(x0_preds, input_type = 'hr') 
    
    return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr')

def predict_modified_diffusion(model, lr_enc, res, hr, lr, upscaled_lr, dataset, timesteps = 200, schedule = 'linear', transform_rescale = False):
    # take in all 4
    # return rescaled input, result, target
    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        img = lr.to(device)
        target = hr.to(device)
    if len(lr.shape)< 4:

        x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float(), factor = dataset.factor, transform_rescale =transform_rescale, dataset = dataset)
    else:
        x_e = forwardpass(lr_enc, lr.to(device).float(), factor = dataset.factor, transform_rescale =transform_rescale, dataset =dataset)
    batches = num_to_groups(1, lr.shape[0])

    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    # torch.manual_seed(0)

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

    @torch.no_grad()
    def sample(model, x_e, image_size, timesteps, batch_size=16, channels=3):
        return p_sample_loop(model, x_e, timesteps=timesteps, shape=(batch_size, channels, image_size, image_size))

    if schedule == 'linear':
        betas = linear_beta_schedule(timesteps=timesteps)
    elif schedule == 'quadratic':
        betas = quadratic_beta_schedule(timesteps=timesteps)
    elif schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps=timesteps)
    elif schedule == 'sigmoid':
        betas = sigmoid_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
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


    all_images_list = list(map(lambda n: sample(model, x_e = x_e, timesteps = timesteps, image_size = x_e.shape[-1],  batch_size=img.shape[0], channels=1), batches))[0]
    all_images = torch.stack(all_images_list, dim=0)
    result = dataset.unscale_data(all_images.numpy()[-1], input_type = 'hr') 
    
    return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr')


def get_profile(image, mesh = 5, max_idx = None):
    image = image[0].T
    binarized = image > 1900
    profile = []
    keyholeprofile = []
    for i in range(binarized.shape[1]):
        coords = np.where(binarized[:,i])[0]
        if len(coords) > 0:
            profile.append(np.min(coords))
            keyholeprofile.append(np.max(coords))
        else:
            if max_idx == None:

                profile.append(binarized.shape[1])
                keyholeprofile.append(binarized.shape[1])
            else:
                profile.append(max_idx)
                # keyholeprofile.append(binarized.shape[1])
                keyholeprofile.append(max_idx)
    profile = np.array(profile)
    keyholeprofile = np.array(keyholeprofile)
    return(profile, keyholeprofile)
def plot_images(input, result, target, modeltype, split = 'train', timesteps = 200, title = '', timestep = None, save =False):
    
    result  = result
    scaling_factor = 1.5
    dpi = 150
    min_temp = 293
    max_temp = 5000
    method = 'Direct'
    # os.makedirs('lrenc_saved_figures', exist_ok = True)
    plt.figure(dpi = dpi, figsize = np.array([4,3])*scaling_factor)
    plt.imshow(input.T, cmap = 'jet', vmin = min_temp, vmax=max_temp, origin = 'lower',  extent=[0*5,80*5,0*5,80*5])

    plt.xlabel(r'x $[\mu m]$')
    plt.ylabel(r'z $[\mu m]$')
    clb = plt.colorbar()
    clb.ax.set_title(r'T$[K]$',fontsize=10)
    frame_tick()
    if timestep is None:
        plt.title('[{} Data] Low Resolution, {}'.format(split, method), fontsize = 10)
    else:
        plt.title(r'[{} Data] Low Resolution, {}, $i$ = {}'.format(split, method, timestep), fontsize = 10)
    if save:
        plt.savefig( title + 'input.png')
    else:
        plt.show()
    # plt.show()
    plt.clf()
    plt.figure(dpi = dpi, figsize = np.array([4,3])*scaling_factor)

    plt.imshow((target).T, cmap = 'jet', vmin = min_temp, vmax=max_temp, origin = 'lower',  extent=[0*5,80*5,0*5,80*5])

    plt.xlabel(r'x $[\mu m]$')
    plt.ylabel(r'z $[\mu m]$')
    clb = plt.colorbar()
    clb.ax.set_title(r'T$[K]$',fontsize=10)
    frame_tick()
    # plt.title('[{} Data] High Resolution'.format(split), fontsize = 10)
    plt.title(r'[{} Data] High Resolution, {}, $i$ = {}'.format(split, method, timestep), fontsize = 10)
    # plt.savefig()
    # plt.savefig('lrenc_saved_figures/target' + title + '.png')
    if save:
        plt.savefig( title + 'target.png')
    else:
        plt.show()

    # plt.show()
    plt.clf()

    plt.figure(dpi = dpi, figsize = np.array([4,3])*scaling_factor)

    plt.imshow((result).T, cmap = 'jet', vmin = min_temp, vmax=max_temp, origin = 'lower',  extent=[0*5,80*5,0*5,80*5])

    plt.xlabel(r'x $[\mu m]$')
    plt.ylabel(r'z $[\mu m]$')
    clb = plt.colorbar()
    clb.ax.set_title(r'T$[K]$',fontsize=10)
    frame_tick()
    # plt.title('[{} Data], {} Timesteps, {} Output'.format(split,str(timesteps), modeltype), fontsize = 10)
    plt.title(r'[{} Data], {} Timesteps, {} Output, $i$ = {}'.format(split,str(timesteps), modeltype, str(timestep)), fontsize = 8)
    # plt.savefig()
    # plt.savefig('lrenc_saved_figures/result'+ title + '.png')
    if save:
        plt.savefig( title + 'result.png')
    else:
        plt.show()

    # plt.show()
    plt.clf()
    plt.close('all')
def multifield_plot_images(input, result, target, modeltype, field_idx, field_key, split = 'train', timesteps = 200, title = '', timestep = None):
    
    result  = result
    scaling_factor = 1.5
    dpi = 90
    min_temp = 293
    max_temp = 5000
    method = 'Direct'

    axis_lim_min= {'vx': -30, 'vy':-30, 'vz':-30, 'temperature':293, 'pressure': 900000, 'liqlabel':0}
    axis_lim_max= {'vx': 30, 'vy':30, 'vz':30, 'temperature':5000, 'pressure': 4*1000000, 'liqlabel':1}
    titles= {'vx': r"$v_x$ [cm/s]", 'vy':r"$v_y$ [cm/s]", 'vz':r"$v_z$ [cm/s]", 'temperature':"T [K]", 'pressure': "P [Pa]", 'liqlabel':"Liquid Volume Fraction"}

    # os.makedirs('lrenc_saved_figures', exist_ok = True)
    plt.figure(dpi = dpi, figsize = np.array([4,3])*scaling_factor)
    plt.imshow(input[field_idx].T, cmap = 'jet',vmin = axis_lim_min[field_key], vmax = axis_lim_max[field_key], origin = 'lower',  extent=[0*5,80*5,0*5,80*5])

    plt.xlabel(r'x $[\mu m]$')
    plt.ylabel(r'z $[\mu m]$')
    clb = plt.colorbar()
    clb.ax.set_title(r'T$[K]$',fontsize=10)
    frame_tick()

    plt.xlabel(r'x [$\mu m$]')
    plt.ylabel(titles[field_key])
    # plt.title('t = {} s, P = {} W, V = {} mm/s'.format(0.5*i/100, power, velocity))

    if timestep is None:
        plt.title('[{} Data] Low Resolution, {}'.format(split, method), fontsize = 10)
    else:
        plt.title(r'[{} Data] Low Resolution, {}, $i$ = {}'.format(split, method, timestep), fontsize = 10)
        
    plt.savefig( title + 'input.png')

    # plt.show()
    plt.clf()
    plt.figure(dpi = dpi, figsize = np.array([4,3])*scaling_factor)

    plt.imshow((target[field_idx]).T, cmap = 'jet', vmin = axis_lim_min[field_key], vmax = axis_lim_max[field_key], origin = 'lower',  extent=[0*5,80*5,0*5,80*5])

    plt.xlabel(r'x $[\mu m]$')
    plt.ylabel(r'z $[\mu m]$')
    # clb = plt.colorbar()
    clb = plt.colorbar()
    clb.ax.set_title(titles[field_key])
    # clb.ax.set_title(r'T$[K]$',fontsize=10)
    frame_tick()
    # plt.title('[{} Data] High Resolution'.format(split), fontsize = 10)
    plt.title(r'[{} Data] High Resolution, {}, $i$ = {}'.format(split, method, timestep), fontsize = 10)
    # plt.savefig()
    # plt.savefig('lrenc_saved_figures/target' + title + '.png')
    plt.savefig( title + 'target.png')


    # plt.show()
    plt.clf()

    plt.figure(dpi = dpi, figsize = np.array([4,3])*scaling_factor)

    plt.imshow((result[field_idx]).T, cmap = 'jet', vmin = axis_lim_min[field_key], vmax = axis_lim_max[field_key], origin = 'lower',  extent=[0*5,80*5,0*5,80*5])

    plt.xlabel(r'x $[\mu m]$')
    plt.ylabel(r'z $[\mu m]$')
    clb = plt.colorbar()
    clb.ax.set_title(titles[field_key])
    frame_tick()
    # plt.title('[{} Data], {} Timesteps, {} Output'.format(split,str(timesteps), modeltype), fontsize = 10)
    plt.title(r'[{} Data], {} Timesteps, {} Output, $i$ = {}'.format(split,str(timesteps), modeltype, str(timestep)), fontsize = 10)
    # plt.savefig()
    # plt.savefig('lrenc_saved_figures/result'+ title + '.png')
    plt.savefig( title + 'result.png')


    # plt.show()
    plt.clf()
    plt.close('all')
def load_mobilenet(mobilenet_results_dir):
    device = 'cuda'
    mobilenet = MobileNetv2_SISR()

    mobilenet.to(device)
    mobilenetoptimizer = torch.optim.Adam(mobilenet.parameters())
    mobilenet_fname = os.path.join(mobilenet_results_dir, 'SISR_mv2f.pth')
    mobilenet_checkpoint = torch.load(mobilenet_fname)
    mobilenet.load_state_dict(mobilenet_checkpoint['model_state_dict'])
    mobilenetoptimizer.load_state_dict(mobilenet_checkpoint['optimizer_state_dict'])
    mobilenet.eval()
    return mobilenet



def predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, dataset, skip = 50, sampler = 'DDPM'):

    '''
    Return the predictions for the Diffusion model given an input batch
    Parameters:
    diff_model: DiffusionModel object
    lr_enc: Torch network module object, representing the trained RRDN encoder model
    res: Torch tensor, Residual between HR and LR data
    hr: Torch tensor, High Resolution data
    lr: Torch tensor, Low Resolution data
    upscaled_lr: Torch tensor, Bicubic upscaled low resolution data
    sampler: String, Sampling method for the diffusion model (either DDIM or DDPM). Default is DDPM
    skip: Integer, Number of timesteps to skip in the diffusion model. Default is 50. Ignored if DDPM.
    dataset: Dataset object, used for rescaling data
    Returns:
    Low resolution data, scaled to original space, 4-D numpy array (batch, channels, height, width)
    Output (Super-resolution), scaled to original space, 4-D numpy array
    High resolution data, scaled to original space, 4-D numpy array 
    '''

    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        img = lr.to(device)
        target = hr.to(device)
    if len(lr.shape) < 4:

        x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float(), factor = dataset.factor)
    else:
        x_e = forwardpass(lr_enc, lr.to(device).float(), factor = dataset.factor)
    
    all_images = diff_model.batch_sample(dataset = dataset, batch = hr.to(device), x_e = x_e.to(device), sampler = sampler, skip=skip)                
    result = dataset.unscale_data(all_images.cpu().numpy()[-1, 0], input_type = 'residual') + dataset.unscale_data(upscaled_lr.numpy(), input_type = 'upscaled_lr')
    
    return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr')




def load_diffusion(diffusion_results_dir, dataset, conditioning = 'implicit', encoder_flag= True):
    model = Unet(
            dim=dataset.img_shape,
            channels=dataset.n_steps*dataset.num_fields,
            dim_mults=(1, 2, 4,),
            conditioning=conditioning,
            out_dim=dataset.n_steps*dataset.num_fields,
            encoder_flag=encoder_flag
        )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    from torch.optim.lr_scheduler import StepLR
    scheduler= StepLR(optimizer, step_size = 10, gamma = 0.5)
    model_fname = os.path.join(diffusion_results_dir, 'ckpt.pth')
    model_checkpoint = torch.load(model_fname)
    model.load_state_dict(model_checkpoint[0])
    optimizer.load_state_dict(model_checkpoint[1])

    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    return model
def load_encoder(encoder_results_dir, dataset):
    device = 'cuda'
    lr_enc  = rrdbnet_encoder(upscale_factor = dataset.factor, in_channels = dataset.n_steps*dataset.num_fields, out_channels = dataset.out_steps*dataset.num_fields)
    # lr_enc = rrdbnet_x4(upscale_factor = 4, num_blocks = 8)

    lr_enc.to(device)
    lr_encoptimizer = torch.optim.Adam(lr_enc.parameters())
    lrenc_fname = os.path.join(encoder_results_dir, 'model_saved.pth')
    lrenc_checkpoint = torch.load(lrenc_fname)
    lr_enc.load_state_dict(lrenc_checkpoint['model_state_dict'])
    lr_encoptimizer.load_state_dict(lrenc_checkpoint['optimizer_state_dict'])
    lr_enc.eval()
    return lr_enc

def plot_srdiff_contours(input, result, target, modeltype, split = 'train', max_temp = 5000, min_temp = 293, dpi = 90, scaling_factor = 1.5, method = 'Direct'):
    profile_result, kp_result = get_profile(result)

    profile_gt, kp_gt = get_profile(target)
    profile_input, kp_input = get_profile(input, mesh = 20)

    plt.figure(dpi = 150, figsize = np.array([4,3])*1.25)
    plt.plot(20*np.arange(20), kp_input*20 - 400, '--', label = 'Low Resolution (Input)')
    plt.plot(5*np.arange(80), kp_result*5 - 400, label = 'Super Resolution (Output)' )
    plt.plot(5*np.arange(80), kp_gt*5 - 400, 'k', linewidth = 2.0, label = 'High Resolution')
    frame_tick()

    legend()
    plt.title('[{} Data] Keyhole Depth, {}'.format(split, modeltype), fontsize = 10)
    plt.xlabel(r'x [$\mu m$] ')
    plt.ylabel(r'z [$\mu m$] ')
    plt.show()

    plt.figure(dpi = 150, figsize = np.array([4,3])*1.25)
    plt.plot(20*np.arange(20),profile_input*20 - 400, '--', label = 'Low Resolution (Input)')
    plt.plot(5*np.arange(80), profile_result*5 - 400, label = 'Super Resolution (Output)' )
    plt.plot(5*np.arange(80),profile_gt*5 - 400, 'k', linewidth = 2.0, label = 'High Resolution')
    frame_tick()

    legend()
    plt.title('[{} Data] Melt Pool Depth, {}'.format(split, modeltype), fontsize = 10)
    plt.xlabel(r'x [$\mu m$] ')
    plt.ylabel(r'z [$\mu m$] ')
    plt.show()
def initialize_diffusion(diff_dir, enc_dir, datasets, timesteps, conditioning, encoding, schedule, device):
    ''' 
    Parameters:
        diff_dir: (str) Diffusion results directory
        enc_dir: (str) Encoder results directory
        datasets: (tuple or list) The three dataset objects corresponding to the train/validation/test splits, in the order (train, validation, test).
        timesteps: (int) The number of timesteps used for the diffusion model during training
        conditioning: (boolean) If true, the diffusion model assumes the LR passes through an encoder before being used for conditioning
        schedule: (str) Variance schedule used for training the diffusion model
        device: (str) 'cuda' for GPU, 'cpu' for CPU.
    Returns:
        diffusion_model: Custom DiffusionModel object that enables sampling with either DDIM or DDPM samplers.
    '''

    diffusion_model = DiffusionModel(results_folder=diff_dir,
                                    lr_encoder_folder=enc_dir,
                                    train_dataset=datasets[0],
                                    dev_dataset=datasets[1],
                                    test_dataset=datasets[2],
                                    timesteps=timesteps,
                                    conditioning=conditioning,
                                    encoding=encoding,
                                    schedule=schedule,
                                    device=device,enc_output = False
                                    )
    diffusion_model.load_saved_model()
    return diffusion_model

