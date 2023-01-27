import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models.diffusion_model import Unet
from models.lr_encoder_model import rrdbnet_encoder as rrdbnet_x4
from models.mobilenet_model import MobileNetv2_SISR
from pylab import gca
from tqdm import tqdm

from runners.train_diffusion import forwardpass, num_to_groups
from skimage.metrics import structural_similarity as ssim_id

device = 'cuda'

def frame_tick(frame_width = 2, tick_width = 1.5):
    ax = gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction = 'in', 
                    width = tick_width)
def legend(location = 'best', fontsize = 8):
        plt.legend(loc = location, fontsize = fontsize, frameon = False)
def predict_mobilenet(model, res, hr, lr, upscaled_lr, dataset):
    img = upscaled_lr.view(upscaled_lr.shape[0], 1, upscaled_lr.shape[1], upscaled_lr.shape[2])
    target = hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2])
    img = img.to(device)
    target = target.to(device)
    
    output = model(img)
    return dataset.unscale_data(lr, input_type='lr'), dataset.unscale_data(output.detach().cpu(), input_type = 'hr'), dataset.unscale_data(target.cpu(), input_type = 'hr') 
def predict_lrenc(lr_enc, res, hr, lr, upscaled_lr, dataset):
    # take in all 4
    # return rescaled input, result, target
    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
    else:
        img = lr.to(device)
    if len(hr.shape) < 4:
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        target = hr.to(device)
    x_e = lr_enc(img.float())
    result = x_e.cpu().detach().numpy()[:,0]
    return dataset.unscale_data(lr, input_type='lr'), dataset.unscale_data(x_e.detach().cpu(), input_type = 'hr'), dataset.unscale_data(target.cpu(), input_type = 'hr')
def predict_diffusion(model, lr_enc, res, hr, lr, upscaled_lr, dataset, timesteps = 200, schedule = 'linear'):
    # take in all 4
    # return rescaled input, result, target
    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        img = lr.to(device)
        target = hr.to(device)
    if len(lr.shape)< 4:

        x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float())
    else:
        x_e = forwardpass(lr_enc, lr.to(device).float())
    batches = num_to_groups(1, lr.shape[0])
    print(timesteps, batches, img.shape[0])
    print(x_e.shape)


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

    # def num_to_groups(num, divisor):
    #     groups = num // divisor
    #     remainder = num % divisor
    #     arr = [divisor] * groups
    #     if remainder > 0:
    #         arr.append(remainder)
    #     return arr

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
    print("TIMESTEPS == {}, Schedule = {}".format(timesteps, schedule))

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

def predict_ddim_diffusion(model, lr_enc, res, hr, lr, upscaled_lr, dataset, seq, timesteps = 200, skip = 1, schedule = 'linear', **kwargs):
    
    # skip =timesteps // self.args.timesteps
    seq = range(0, timesteps, skip)
    
    def cosine_beta_schedule(timesteps, s=0.008):

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

    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
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

        x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float())
    else:
        x_e = forwardpass(lr_enc, lr.to(device).float())
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

def predict_modified_diffusion(model, lr_enc, res, hr, lr, upscaled_lr, dataset, timesteps = 200, schedule = 'linear'):
    # take in all 4
    # return rescaled input, result, target
    if len(lr.shape) < 4:
        img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
        target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
    else:
        img = lr.to(device)
        target = hr.to(device)
    if len(lr.shape)< 4:

        x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float())
    else:
        x_e = forwardpass(lr_enc, lr.to(device).float())
    batches = num_to_groups(1, lr.shape[0])
    print(timesteps, batches, img.shape[0])
    print(x_e.shape)


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

    # def num_to_groups(num, divisor):
    #     groups = num // divisor
    #     remainder = num % divisor
    #     arr = [divisor] * groups
    #     if remainder > 0:
    #         arr.append(remainder)
    #     return arr

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
    print("TIMESTEPS == {}, Schedule = {}".format(timesteps, schedule))

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
    result = dataset.unscale_data(all_images.numpy()[-1], input_type = 'hr') 
    
    return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr')


def get_profile(image, mesh = 5):
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
            profile.append(binarized.shape[1])
            keyholeprofile.append(binarized.shape[1])
    profile = np.array(profile)
    keyholeprofile = np.array(keyholeprofile)
    return(profile, keyholeprofile)
def plot_images(input, result, target, modeltype, split = 'train', timesteps = 200, title = '', timestep = None):
    
    result  = result
    scaling_factor = 1.5
    dpi = 90
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
        
    plt.savefig( title + 'input.png')

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
    plt.savefig( title + 'target.png')


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

def load_diffusion(diffusion_results_dir):
    image_size = 80
    channels = 1
    batch_size = 4
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
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
def load_encoder(encoder_results_dir):
    device = 'cuda'
    lr_enc = rrdbnet_x4(upscale_factor = 4, num_blocks = 8)

    lr_enc.to(device)
    lr_encoptimizer = torch.optim.Adam(lr_enc.parameters())
    lrenc_fname = os.path.join(encoder_results_dir, 'model_saved.pth')
    lrenc_checkpoint = torch.load(lrenc_fname)
    lr_enc.load_state_dict(lrenc_checkpoint['model_state_dict'])
    lr_encoptimizer.load_state_dict(lrenc_checkpoint['optimizer_state_dict'])
    lr_enc.eval()
    return lr_enc

def PSNR(op, t, batch_size): 
    mse = torch.sum((t - op) ** 2) 
    mse /= (batch_size*80*80)
    max_pixel = torch.max(t)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr 
def SSIM(op, t, batch_size):
    ssim = 0 
    for i in range(op.shape[0]):

        # print(out[0,0].size())
        # print(op.shape, t.shape)
        score = ssim_id(op[0], t[0])#, full=True)
        ssim+=score/batch_size
    
        #print("SSIM: {}".format(score))
    return ssim
