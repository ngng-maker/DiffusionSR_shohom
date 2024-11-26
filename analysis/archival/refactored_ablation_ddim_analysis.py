
import glob
import os
import time
import argparse
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets.dataset import SimulationXZDataset
from pylab import gca
from runners.train_diffusion import forwardpass
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn
from sklearn.metrics import mean_absolute_error
from analysis_functions import (PSNR, SSIM, get_profile, load_encoder, plot_images,
                            predict_lrenc)
import multiprocessing

import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
import sklearn
from sklearn.metrics import mean_absolute_error
from runners.train_diffusion import DiffusionModel


device = 'cuda'


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0).to(device)
    # print(beta.device, t.device)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a



def predict_modified_ddim_diffusion(model, lr_enc, res, hr, lr, upscaled_lr, encoding, dataset, seq, timesteps = 200, skip = 1, schedule = 'linear',transform_rescale= False, **kwargs):
    
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
    if encoding:
            # x_e = forwardpass(lr_enc, lr.to(device).float(), factor = train_dataset.factor)
        if len(lr.shape)< 4:
            x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float(), transform_rescale=False)
        else:
            x_e = forwardpass(lr_enc, lr.to(device).float(), transform_rescale=False)
    else:
        x_e = upscaled_lr.to(device).float().repeat(1,1,1, 1)
    
    # batches = num_to_groups(1, lr.shape[0])
    shape=hr.shape
    # print(timesteps, batches, img.shape[0])
    # print(x_e.shape)
    # with torch.no_grad():
    #     n = img.size(0)
    #     seq_next = [-1] + list(seq[:-1])
    #     x0_preds = []
        
    #     x = torch.randn(shape, device=device)
    #     xs = [x]
    #     for i, j in zip(reversed(seq), reversed(seq_next)):
    #         t = (torch.ones(n) * i).to(x.device)
    #         next_t = (torch.ones(n) * j).to(x.device)
    #         at = compute_alpha(b, t.long())
    #         at_next = compute_alpha(b, next_t.long())
    #         xt = xs[-1].to('cuda')
    #         et = model(x, t, x_e)#model(xt, t)
    #         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    #         x0_preds.append(x0_t.to('cpu'))
    #         c1 = (
    #             kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    #         )
    #         c2 = ((1 - at_next) - c1 ** 2).sqrt()
    #         xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
    #         xs.append(xt_next.to('cpu'))
    with torch.no_grad():
        x = torch.randn(shape, device=device)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t, x_e)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    # print(len(x0_preds),x0_preds[0].shape, len(xs))
    # return xs, x0_preds
    result = dataset.unscale_data(xs[-1], input_type = 'hr') 
    return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr'), xs, b




def get_inst_profile(image, mesh = 5):
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

def predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, dataset, skip = 50, device = 'cuda', sampler = 'DDIM'):

    '''
    Return the predictions for the Diffusion model given an input batch
    Parameters:
    diff_model: DiffusionModel object
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

    if diff_model.encoding:
        if len(lr.shape) < 4:

            x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float(), factor = dataset.factor)
        else:
            x_e = forwardpass(lr_enc, lr.to(device).float(), factor = dataset.factor)
            x_e = forwardpass(lr_enc, lr.to(device).float())
    else:
        x_e = upscaled_lr.to(diff_model.device).float().repeat(1,1,1, 1)
    all_images = diff_model.batch_sample(dataset = dataset, batch = hr, x_e = x_e, sampler = sampler, skip= skip)                
    result = dataset.unscale_data(all_images.cpu().numpy()[-1, 0], input_type = 'residual') + dataset.unscale_data(upscaled_lr.numpy(), input_type = 'upscaled_lr')
        
    return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr')


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
    encoding_flag = encoding  == 'True'
    diffusion_model = DiffusionModel(results_folder=diff_dir,
                                    lr_encoder_folder=enc_dir,
                                    train_dataset=datasets[0],
                                    dev_dataset=datasets[1],
                                    test_dataset=datasets[2],
                                    timesteps=timesteps,
                                    conditioning=conditioning,
                                    encoding=encoding_flag,
                                    schedule=schedule,
                                    device=device,
                                    )
    diffusion_model.load_saved_model()
    return diffusion_model

class Simulation():
    def __init__(self, folder, diff_steps, schedule):
        self.folder = folder
        self.diff_steps = diff_steps
        self.schedule = schedule
        self.kp = None
        self.profile = None
        self.max_depth = None
        self.kp_depth = None
    def get_loss_curve(self):
        test_loss = np.loadtxt(os.path.join(self.folder, 'validation_loss_epoch.txt'))
        train_loss = np.loadtxt(os.path.join(self.folder, 'loss_epoch.txt'))
        return train_loss, test_loss

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
def frame_tick(frame_width = 2, tick_width = 1.5):
    ax = gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction = 'in', 
                    width = tick_width)
def legend(location = 'best', fontsize = 8):
        plt.legend(loc = location, fontsize = fontsize, frameon = False)


def plot_srdiff_contours(input, result, target, modeltype, split = 'train', max_temp = 5000, min_temp = 293, dpi = 90, scaling_factor = 1.5, method = 'Direct', save = False, mpfname = '', kpfname = ''):
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
    if save:
  
        plt.savefig(kpfname)
    else:
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
    if save:
        plt.savefig(mpfname)
    else:
        plt.show()
    plt.show()
def plot_temp_lines(input, result, target, modeltype, split = 'train', save = False, fname = ''):
    scaling_factor = 1.5
    dpi = 90
    min_temp = 293
    max_temp = 8000
    method = 'direct'
    plt.figure(dpi = dpi, figsize = np.array([4,3])*scaling_factor)
    plt.title(r'{} model, Cross section: x = 200 $\mu m$'.format(modeltype))
    plt.plot(80*5*(np.arange(input.shape[-1])/input.shape[-1]),input[10,:],label = 'Low-Resolution Input')
    plt.plot(80*5*(np.arange(target.shape[-1])/target.shape[-1]),target[40,:], label = 'High-Resolution Input') 
    plt.plot(80*5*(np.arange(target.shape[-1])/target.shape[-1]),result[40,:], label = 'Output')
    legend()
    plt.ylabel(r'$T [K]$')
    plt.xlabel(r'$z [\mu m$]')
    frame_tick()
    plt.ylim(min_temp, max_temp)
    plt.xlim(0, 400)
    if save:
        plt.savefig(fname)
    else:
        plt.show()
    # plt.show()




# def process_case(diff_dir, limit =None, skip = 50, gpu  = 1, run_skip_ablation = False):
def find_simulation_case(case):

    downscale_method = case['downscale_method']
    encoding = case['encoding']
    encode_flag = case['encode_flag']
    conditioning = case['conditioning']
    # loss_type = case['loss_type']
    data_sorted = False
    print('#######################################################################')
    print(f"Now processing case: {downscale_method}, {conditioning}, {encode_flag}")
    print('#######################################################################')
    device= 'cuda'
    run_paths = f'/home/oogoke/DiffusionSR/runs/{downscale_method}/diffusion{conditioning}{encode_flag}'

    possible_paths= glob.glob(os.path.join(run_paths, '*/**/*'))
    len_paths  = [len(os.listdir(p)) for p in possible_paths]
    actual_path_idx = np.argmax(len_paths)
    max_len = np.max(len_paths)
    diffusion_results_dir = possible_paths[actual_path_idx]
    print(f"Choosing path with {max_len} files, {diffusion_results_dir}")
    if downscale_method == 'direct' and encoding == 'True' and encode_flag == 'encoded' and conditioning == 'implicit':
        diffusion_results_dir    = '/home/oogoke/DiffusionSR/runs/direct/diffusionimplicitencoded/2023_02_19_15_08_28/standardize/n_steps_1'
    return diffusion_results_dir
def process(diffusion_results_dir,  limit =None, skip = 50, gpu  = 1, run_skip_ablation = False):

    device = 'cuda'



    os.path.exists(os.path.join(diffusion_results_dir, 'configuration.yml'))
    with open(os.path.join(os.path.join(diffusion_results_dir, 'configuration.yml')), "r") as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)
    conditioning = new_config.conditioning
    downscale_method = new_config.downscale_method
    # encode_flag= new_config.encode_flag
    encoding = str(new_config.encoding)
 
    if encoding == 'True':
        encode_flag = 'encoded'
    else:
        encode_flag = 'upscaled'
    if 'loss_type' in new_config:
        loss_type = new_config.loss_type
    else:
        loss_type = 'huber'
    timesteps = int(new_config.timesteps)
    schedule = new_config.schedule

    if encoding == 'True':
        with open(os.path.join(diff_dir, 'information.txt')) as f:
            lines = f.readlines()
        encoder_results_dir = os.path.join('/home/oogoke/DiffusionSR', lines[1].split(' pretrained_encoder: ')[1].split('\n')[0])
        if encoder_results_dir.startswith('/home/cmu'):
            print(f"Skipping {diffusion_results_dir}")
            return
            # encoder_results_dir = encoder_results_dir.replace('/home/cmu/github/LPBFDiffusionSR', '/home/oogoke/DiffusionSR')
    else:
        encoder_results_dir = None


    os.environ['CUDA_VISIBLE_DEVICES']  = str(gpu)
    batch_size = 16
    analysis_folder = f'/home/oogoke/DiffusionSR/runs/analysis/analyzed_figures/{timesteps}_{conditioning}_{schedule}_{downscale_method}_{encode_flag}_{loss_type}'
    os.makedirs( analysis_folder, exist_ok = True)
    os.makedirs(os.path.join(analysis_folder, 'csv_files'), exist_ok = True)

    data_folder = '/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data'
    train_dataset = SimulationXZDataset(downscale_method = downscale_method, split = 'train', root_folder = data_folder , return_info = True)
    test_dataset = SimulationXZDataset(downscale_method =downscale_method, split = 'test', root_folder = data_folder, return_info = True)
    dev_dataset = SimulationXZDataset(downscale_method = downscale_method, split = 'dev', root_folder = data_folder, return_info = True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)


    
    # ### Plot Train, Test Loss for diffusion



    test_loss = np.loadtxt(os.path.join(diffusion_results_dir, 'validation_loss_epoch.txt'))
    train_loss = np.loadtxt(os.path.join(diffusion_results_dir, 'loss_epoch.txt'))

    plt.figure(figsize = np.array([4,3])*1.25, dpi = 150)

    plt.plot(train_loss[:], label = 'Training Loss')
    plt.plot(test_loss[:], label = 'Validation Loss')
    plt.xlabel(r'Epoch ($3000$ iterations)')
    plt.ylabel(r'$L_1$ Loss')
    legend()
    plt.title('Diffusion Model Loss, Linear Schedule: 1000 timesteps')
    plt.grid(which = 'major')
    frame_tick()
    plt.savefig(os.path.join(analysis_folder, 'diff_loss_linear_scale.png'))
    plt.show()
    plt.clf()
    plt.figure(figsize = np.array([4,3])*1.25, dpi = 150)
    plt.plot(train_loss[:], label = 'Training Loss')
    plt.plot(test_loss[:], label = 'Validation Loss')
    plt.xlabel(r'Epoch ($3000$ iterations)')
    plt.ylabel(r'$L_1$ Loss')
    legend()
    plt.title('Diffusion Model Loss, Linear Schedule: 1000 timesteps')

    plt.grid(which = 'major')
    frame_tick()
    plt.yscale('log')
    plt.savefig(os.path.join(analysis_folder, 'diff_loss_log_scale.png'))
    plt.show()
    plt.clf()
    plt.close('all')

    
    # ### Defining simulation object for plotting loss curves


  

    
    # ### Demonstration: Plotting loss curves 


    if encoding == 'True':
        try:
            test_loss = np.loadtxt(os.path.join(encoder_results_dir, 'test_loss.txt'))
        except OSError:
            test_loss = np.loadtxt(os.path.join(encoder_results_dir, 'validation_loss_epoch.txt'))

        train_loss = np.loadtxt(os.path.join(encoder_results_dir, 'train_loss.txt'))

        plt.figure(figsize = np.array([4,3])*1.25, dpi = 150)

        plt.plot(train_loss[:], label = 'Training Loss')
        plt.plot(test_loss, label = "Validation Loss")

        plt.xlabel(r'Epoch ($3000$ iterations)')
        plt.ylabel(r'$L_1$ Loss')
        legend()
        plt.title('Encoder Model Loss')
        plt.grid(which = 'major')
        frame_tick()
        plt.savefig(os.path.join(analysis_folder, 'encoder_loss_linear_scale.png'))
        plt.clf()




    
    # ### Initialize encoder and diffusion models



    diff_model = initialize_diffusion(diff_dir=diffusion_results_dir,
                                    enc_dir=encoder_results_dir,
                                    datasets=[train_dataset,
                                                dev_dataset, test_dataset],
                                    timesteps=timesteps,
                                    conditioning=conditioning,
                                    encoding=encoding,
                                    schedule=schedule,
                                    device=device)

    if encoding == 'True':

        lr_enc = load_encoder(encoder_results_dir, dataset = train_dataset)
    else:
        lr_enc = None


    
    # ### Define diffusion prediction function
    # Note: This may eventually be moved to analysis_functions.py, but is provided here to allow for easier editing/debugging


    for batch_idx, (res, hr, lr, upscaled_lr, info) in enumerate(train_dataloader):            
        input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, train_dataloader.dataset)
        break



    if encoding == 'True':

        input, lr_enc_result, target = predict_lrenc(lr_enc,res, hr, lr, upscaled_lr, train_dataset)
    input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, train_dataloader.dataset)
    split = 'Train'
    kpfname = os.path.join(analysis_folder, f'diffusion_contour_line_{split}_plot_kp.png')
    mpfname = os.path.join(analysis_folder, f'diffusion_contour_line_{split}_plot_mp.png')
    plot_srdiff_contours(input, result_diffusion, target[0], split = 'Train', modeltype = "Diffusion", save = True, kpfname = kpfname, mpfname = mpfname )
    # plt.savefig(os.path.join(analysis_folder, 'diffusion_train_contour_plot'))

    kp_fname = os.path.join(analysis_folder, f'encoder_contour_line_{split}_plot_kp.png')
    mpfname = os.path.join(analysis_folder, f'encoder_contour_line_{split}_plot_mp.png')
    if encoding == 'True':

        plot_srdiff_contours(input, lr_enc_result, target[0], split = 'Train', modeltype = "Encoder", save = True, kpfname = kpfname, mpfname = mpfname)
    # plt.savefig(os.path.join(analysis_folder, 'encoder_train_contour_plot.png'))

    if encoding == 'True':

        plot_images(input[0], lr_enc_result[0,0], target[0][0], timesteps =  diff_model.timesteps//skip, split = 'Train', modeltype = "Encoder", save = True, title = os.path.join(analysis_folder, 'encoder_train_image_plot'))
    # plt.savefig()

    plot_images(input[0], result_diffusion[0], target[0][0], split = 'Train', timesteps =  diff_model.timesteps//skip,modeltype = "Diffusion", save = True, title = os.path.join(analysis_folder, 'diffusion_train_image_plot') )


    #### Plot temperature contours
    for split, dataloader in zip(['Train', 'Validation', 'Test'], [train_dataloader, dev_dataloader, test_dataloader]):

        
        fname = os.path.join(analysis_folder, f'diffusion_temperature_line_{split}_plot_kp.png')
        plot_temp_lines(input[0,0], result_diffusion[0,0], target[0][0], split = split, modeltype = "Diffusion", save = True,fname =fname)
        # plt.savefig()
        if encoding == 'True':
            fname = os.path.join(analysis_folder, f'encoder_temperature_line_{split}_plot_mp.png')

            input, lr_enc_result, target = predict_lrenc(lr_enc,res, hr, lr, upscaled_lr, dataloader.dataset)
            plot_temp_lines(input[0,0], lr_enc_result[0][0], target[0][0], split = split, modeltype = "Encoder", save = True,fname =fname)
        # plt.savefig(os.path.join(analysis_folder, f'encoder_temperature_line_{split}_plot.png'))


    
    # ##### Plot melt pool images in the validation dataset. Save figures in order of power, velocity, and time.
    plot_all_images = False
    if plot_all_images:
        last_power = ''
        modeltype = 'Diffusion' # Defines part of the title of the image
        verbose= False # controls how much information to print 
        if limit is None:
            limit = len(dev_dataloader)  
        for batch_idx, (res, hr, lr, upscaled_lr, info) in enumerate(dev_dataloader):  
            for skips in [1]:
                oldtime = time.time()
                if verbose:
                    print(res.shape)
                if encoding == 'True':
                    input, result, target = predict_lrenc(lr_enc,res, hr, lr, upscaled_lr, train_dataset)
                input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, train_dataloader.dataset)
                info = info[0]
                time_elapsed = time.time()-oldtime
                if verbose:
                    print(info)
                if last_power != 'power' + str(info[0].item()) + 'vel' + str(info[1].item()):
                        timestep = 0
                else:
                    timestep += 1
                folder = os.path.join(analysis_folder, 'time_stepped/'  + 'power' + str(info[0].item()) + 'vel' + str(info[1].item()))

                os.makedirs(folder,exist_ok = True)
                title  = os.path.join(folder, 'batch_{:06}_'.format(batch_idx)+ 'time' + '{:07}'.format(int(np.round(info[2].item()*1e6))))
                if verbose:
                    print(title, timestep)
                plot_images(input[0], result_diffusion[0], target[0][0], timestep = timestep, split = 'Validation', modeltype =modeltype, timesteps = timesteps//skips, save = True, title =  title)
                if verbose:
                    print(time_elapsed, "elapsed time")
                last_power = 'power' + str(info[0].item()) + 'vel' + str(info[1].item())
            
            if batch_idx > limit:
                break


    
    # #### Calculate melt pool dimensions in the validation dataset, order the results in terms of processing conditions
    data_sorted = False
    if data_sorted:
        last_power = ''
        modeltype = 'Diffusion' # Defines part of the title of the image
        enc_depths_sim = []
        diff_depths_sim = []
        gt_depths_sim = []

        enc_kp_sim = []
        diff_kp_sim = []
        gt_kp_sim = []
        gt_results = {}
        enc_results = {}
        diff_results = {}
        
        for batch_idx, (res, hr, lr, upscaled_lr, info) in tqdm(enumerate(dev_dataloader), total = len(dev_dataloader)):  
        

            oldtime = time.time()
            if encoding == 'True':

                input, result, target = predict_lrenc(lr_enc,res, hr, lr, upscaled_lr, train_dataset)
            input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, dev_dataloader.dataset)
            info = info[0] # info is technically returned with an extra dimension, this command removes it.
            time_elapsed = time.time()-oldtime

            if last_power != 'power' + str(info[0].item()) + 'vel' + str(info[1].item()):
                timestep = 0

                enc_depths_sim = []
                diff_depths_sim = []
                gt_depths_sim = []

                enc_kp_sim = []
                diff_kp_sim = []
                gt_kp_sim = []

            else:
                timestep += 1
            profile_diff, kp_diff = get_profile(result_diffusion[0])
            profile_gt, kp_gt = get_profile(target[0])
            if encoding == 'True':
            
                profile_enc, kp_enc = get_profile(result[0])

            if encoding == 'True':

                enc_kp_sim.append(5*(80-np.min(kp_enc)))
            gt_kp_sim.append(5*(80-np.min(kp_gt)))
            diff_kp_sim.append(5*(80-np.min(kp_diff)))
            if encoding == 'True':

                enc_depths_sim.append(5*(80-np.min(profile_enc)))
            gt_depths_sim.append(5*(80-np.min(profile_gt)))
            diff_depths_sim.append(5*(80-np.min(profile_diff)))

            last_power = 'power' + str(info[0].item()) + 'vel' + str(info[1].item())  
            gt_results['key_power' + str(info[0].item()) + 'vel' + str(info[1].item()) ] = gt_kp_sim
            if encoding == 'True':
            
                enc_results['key_power' + str(info[0].item()) + 'vel' + str(info[1].item()) ] = enc_kp_sim
            diff_results['key_power' + str(info[0].item()) + 'vel' + str(info[1].item()) ] = diff_kp_sim
            gt_results['depth_power' + str(info[0].item()) + 'vel' + str(info[1].item()) ] = gt_depths_sim
            if encoding == 'True':
                enc_results['depth_power' + str(info[0].item()) + 'vel' + str(info[1].item()) ] = enc_depths_sim
            diff_results['depth_power' + str(info[0].item()) + 'vel' + str(info[1].item()) ] = diff_depths_sim
            if limit is None:
                limit = len(dev_dataloader)
            if batch_idx > limit:
                break
                # break



        
        pd.DataFrame.from_dict(gt_results).to_csv(os.path.join(analysis_folder, 'csv_files/gt_results.csv'))
        if encoding == 'True':
            pd.DataFrame.from_dict(enc_results).to_csv(os.path.join(analysis_folder, 'csv_files/enc_results.csv'))
        pd.DataFrame.from_dict(diff_results).to_csv(os.path.join(analysis_folder, 'csv_files/diff_results.csv'))


    
        # ##### Plot keyhole and melt pool depths over time for each power/velocity combination


        depth_folder = os.path.join(analysis_folder, 'depth_over_time')
        os.makedirs(depth_folder, exist_ok = True)
        for key in enc_results.keys():
            if encoding == 'True':  
                plt.plot(enc_results[key], label = 'Encoder Model Output')
            plt.plot(diff_results[key], label = 'Diffusion Model Output')
            plt.plot(gt_results[key], label = 'Ground Truth')
            power_text = key.split('power')[1].split('vel')[0]
            vel_text = key.split('vel')[1]
            plt.title(f'Power: {power_text} W , Velocity: {vel_text} mm/s')
            frame_tick()
            plt.legend()
            
            plt.savefig(os.path.join(depth_folder, f'{key}.png'))
            plt.clf()
            # plt.show()

        
    # #### Calculate melt pool/keyhole dimensions for DDIM sampler for varying number of timesteps


    
    if run_skip_ablation:
        max_depth_gt = []
        max_depth_input = []
        kp_depth_gt = []
        kp_depth_input = []
        kp_depth_diffs = []
        max_depth_diffs = []

        psnrs_lr = {}
        psnrs_diff= {}

        ssims_diff = {}
        ssims_lr = {}

        l1_lr = {}
        l1_diff = {}

        sample_times = []
        steps_list= []
        psnr_mean_lr = []
        ssim_mean_lr = []
        l1_mean_lr = []
        ssim_mean_diff = []
        psnr_mean_diff = []
        l1_mean_diff = []
        l1loss = nn.L1Loss()
        factor = 2
        mesh_hr = 5
        mesh_lr = 20
        # for skips in [1, 2, 3, 5, 10, 15, 20, 50]:
        for skips in [0, 1, 5,  50]:


        # for skips in [15, 20, 50]:
            diff_steps = diff_model.timesteps
            kp_depth_diff = []
            max_depth_diff = []
            max_depth_input= []
            kp_depth_input = []
            max_depth_gt  = []
            kp_depth_gt = []
            psnr_batches_lr = []
            psnr_batches_diff = []
            l1_batches_diff = []
            l1_batches_lr = []
            ssim_batches_lr = []
            ssim_batches_diff = []
            if limit is None:
                limit = len(dev_dataloader)
            for batch_idx, (res, hr, lr, upscaled_lr, info) in enumerate(dev_dataloader):  
                oldtime = time.time()
                if skips == 0:
                    sampler = 'DDPM'
                else:
                    sampler = 'DDIM'
                input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, train_dataloader.dataset, skip = skips, sampler = sampler)
                upscale = dev_dataset.unscale_data(upscaled_lr, input_type = 'upscaled_lr')
                lrpsnr = PSNR(upscale, target, target.shape[0]).item()

                diffpsnr = PSNR(result_diffusion, target, target.shape[0]).item()
                psnr_batches_diff.append(diffpsnr)
                psnr_batches_lr.append((lrpsnr))
                lrssim = SSIM(upscale, target.view(upscale.shape), target.shape[0])
                diffssim = SSIM(result_diffusion, target.view(upscale.shape), target.shape[0])
                lrl1 = l1loss(upscale, target).item()
                diffl1 = l1loss(torch.tensor(result_diffusion), target).item()
                ssim_batches_lr.append(lrssim)
                ssim_batches_diff.append(diffssim)
                l1_batches_lr.append(lrl1)
                l1_batches_diff.append(diffl1)
                for i, rd in enumerate(result_diffusion):
                    elapsed = time.time() - oldtime
                    if batch_idx == 0:
                        sample_times.append(elapsed)
                    if skips == 0:
                        steps_list.append(timesteps + 1)
                    else:
                        steps_list.append(timesteps//skips)

                    profile_diff, kp_diff = get_profile(result_diffusion[i])
                    profile_gt, kp_gt = get_profile(target[i])
                    profile_input, kp_input = get_profile(input[i], mesh = 20)

                    max_depth_gt.append(mesh_hr*(20*factor-5*factor - np.min(profile_gt)))
                    max_depth_diff.append(mesh_hr*(20*factor-5*factor  -np.min(profile_diff)))
                    max_depth_input.append(mesh_lr*(20-5- np.min(profile_input)))
                    kp_depth_gt.append(mesh_hr*(20*factor-5*factor-np.min(kp_gt)))
                    kp_depth_input.append(mesh_lr*(20-5-np.min(kp_input)))
                    kp_depth_diff.append(mesh_hr*(20*factor-5*factor-np.min(kp_diff)))
                if batch_idx > limit:
                    break
                    
            kp_depth_diffs.append(kp_depth_diff)
            max_depth_diffs.append(max_depth_diff)
            # ssims_diff.append(ssim_batches_diff)
            # ssims_lr.append(ssim_batches_lr)
            # l1_lr.append(l1_batches_lr)
            l1_lr[str(skips)] = l1_batches_lr
            l1_diff[str(skips)] = l1_batches_diff
            psnrs_lr[str(skips)] = psnr_batches_lr
            psnrs_diff[str(skips)] = psnr_batches_diff
            ssims_lr[str(skips)] = ssim_batches_lr
            ssims_diff[str(skips)] = ssim_batches_diff
            # psnrs_lr.append(psnr_batches_lr)
            # psnrs_diff.append(psnr_batches_diff)
            # l1_diff.append(l1_batches_diff)
            psnr_mean_lr.append(np.mean(psnr_batches_lr))
            psnr_mean_diff.append(np.mean(psnr_batches_diff))
            ssim_mean_diff.append(np.mean(ssim_batches_diff))
            ssim_mean_lr.append(np.mean(ssim_batches_lr))
            l1_mean_lr.append(np.mean(l1_batches_lr))
            l1_mean_diff.append(np.mean(l1_batches_diff))
        import pandas as pd
        pd.DataFrame.from_dict(l1_lr).to_csv(os.path.join(analysis_folder, 'csv_files/l1_lr.csv'))
        pd.DataFrame.from_dict(l1_diff).to_csv(os.path.join(analysis_folder, 'csv_files/l1_diff.csv'))
        pd.DataFrame.from_dict(psnrs_lr).to_csv(os.path.join(analysis_folder, 'csv_files/psnrs_lr.csv'))
        pd.DataFrame.from_dict(psnrs_diff).to_csv(os.path.join(analysis_folder, 'csv_files/psnrs_diff.csv'))
        pd.DataFrame.from_dict(ssims_diff).to_csv(os.path.join(analysis_folder, 'csv_files/ssims_diff.csv'))
        pd.DataFrame.from_dict(ssims_lr).to_csv(os.path.join(analysis_folder, 'csv_files/ssims_lr.csv'))
        pd.DataFrame.from_dict(ssims_diff).to_csv(os.path.join(analysis_folder, 'csv_files/ssims_diff.csv'))



    
    # #### Plot melt pool statistics


    from sklearn.metrics import mean_absolute_error

    if run_skip_ablation:
        plt.figure(figsize = [4,3], dpi = 150)
        plt.plot(np.arange(300),np.arange(300), 'k-', linewidth = 2.0)
        plt.scatter(max_depth_gt, max_depth_input, s = 3.0, label = 'Low Resolution')
        maes_depth = []
        for max_depth, skips in zip(max_depth_diffs, [1, 2, 3, 5, 10, 15, 20, 50]):
            plt.scatter(max_depth_gt, max_depth, s = 3.0, label = 'Super Resolution, {}'.format(timesteps//skips))
        
            mae_input = mean_absolute_error(max_depth_gt, max_depth_input)
            mae_depth_diff = mean_absolute_error(max_depth_gt, max_depth)
            maes_depth.append(mae_depth_diff)
            
            plt.xlim(0,300)
            plt.ylim(0,300)
            frame_tick()
        plt.xlabel(r'Melt Pool Depth [$\mu m]$ (High Resolution) ')
        plt.ylabel(r'Melt Pool Depth [$\mu m]$ (Estimated) ')
        plt.title('Validation Set')
        plt.text(25, 250,r'MAE LR: {:.03}$\mu m$'.format(mae_input), fontsize = 8 )
        plt.text(25, 225,r'MAE SRDiff: {:.03}$\mu m$'.format(mae_depth_diff), fontsize = 8 )
        plt.legend(fontsize = 5, loc = 'lower right')
        plt.show()

    
    # #### Plot keyhole statistics




    if run_skip_ablation:

        plt.figure(figsize = [4,3], dpi = 150)
        plt.plot(np.arange(300),np.arange(300), 'k-', linewidth = 2.0)
        plt.scatter(max_depth_gt, kp_depth_input, s = 3.0, label = 'Low Resolution')
        maes_depth = []
        for kp_depth, skips in zip(kp_depth_diffs, [1, 2, 3, 5, 10, 15, 20, 50]):
            plt.scatter(kp_depth_gt, kp_depth, s = 3.0, label = 'Super Resolution, {}'.format(timesteps//skips))
        
            mae_input = mean_absolute_error(kp_depth_gt, kp_depth_input)
            mae_depth_diff = mean_absolute_error(kp_depth_gt, kp_depth)
            maes_depth.append(mae_depth_diff)
        
            plt.xlim(0,300)
            plt.ylim(0,300)
            frame_tick()
            
        plt.xlabel(r'Keyhole Depth [$\mu m]$ (High Resolution) ')
        plt.ylabel(r'Keyhole Depth [$\mu m]$ (Estimated) ')
        plt.title('Validation Set')
        plt.text(25, 250,r'MAE LR: {:.03}$\mu m$'.format(mae_input), fontsize = 8 )
        plt.text(25, 225,r'MAE SRDiff: {:.03}$\mu m$'.format(mae_depth_diff), fontsize = 8 )
        plt.legend(fontsize = 5, loc = 'lower right')
        plt.show()


    # if run_skip_ablation:

    #     plt.plot(steps_list[::12], maes_depth, '.')
    #     frame_tick()
    #     plt.xlabel('Sampling Timesteps')
    #     plt.ylabel(r'$L_1$ Error: Melt Pool Depth')
    #     plt.show()
    #     frame_tick()

    #     plt.plot(steps_list[::12], sample_times[::12], '.')
    #     plt.xlabel('Sampling Timesteps')
    #     plt.ylabel(r'Elapsed Time (s)')
    #     plt.show()


    #     plt.plot(steps_list[::12], maes_depth, '.')
    #     frame_tick()
    #     plt.xlabel('Sampling Timesteps')
    #     plt.ylabel(r'$L_1$ Error: Melt Pool Depth')
    #     plt.show()
    #     frame_tick()


    
    # #### Calculate the melt pool/keyhole dimensions for a single value of DDIM sampling timesteps


    max_depth_gt = []
    max_depth_input = []
    kp_depth_gt = []
    kp_depth_input = []
    kp_depth_diffs = []
    max_depth_diffs = []
    sample_times = []
    steps_list= []
    skip = 50
    mesh_hr = 5
    factor =4 
    mesh_lr = 20
    kp_depth_diff = []
    max_depth_diff = []
    max_depth_input= []
    kp_depth_input = []
    max_depth_gt  = []
    kp_depth_gt = []
    if limit is None:
        limit = len(dev_dataloader)
    for batch_idx, (res, hr, lr, upscaled_lr, info) in tqdm(enumerate(dev_dataloader), total = len(dev_dataloader)):  
        oldtime = time.time()
        encode_bool = encoding == 'True'
        input, result_diffusion, target, _, _ = predict_modified_ddim_diffusion(diff_model.model, lr_enc, res, hr, lr, upscaled_lr,encoding = encode_bool,dataset =  train_dataset,seq= None, timesteps = timesteps,skip = skip, schedule = 'linear')

        # input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, train_dataloader.dataset, skip = skip)
        for i, rd in enumerate(result_diffusion):
            elapsed = time.time() - oldtime
            if batch_idx == 0:
                sample_times.append(elapsed)
                steps_list.append(timesteps//skip)
            profile_diff, kp_diff = get_profile(result_diffusion[i])
            profile_gt, kp_gt = get_profile(target[i])
            profile_input, kp_input = get_profile(input[i], mesh = 20)
        
            max_depth_gt.append(mesh_hr*(20*factor - np.min(profile_gt)))
            max_depth_diff.append(mesh_hr*(20*factor  -np.min(profile_diff)))
            max_depth_input.append(mesh_lr*(20 - np.min(profile_input)))
            kp_depth_gt.append(mesh_hr*(20*factor-np.min(kp_gt)))
            kp_depth_input.append(mesh_lr*(20-np.min(kp_input)))
            kp_depth_diff.append(mesh_hr*(20*factor-np.min(kp_diff)))
        if batch_idx > limit:
            break
    kp_depth_diffs.append(kp_depth_diff)
    max_depth_diffs.append(max_depth_diff)




    plt.figure(figsize = [4,3], dpi = 150)
    plt.plot(np.arange(300),np.arange(300), 'k-', linewidth = 2.0)
    plt.scatter(max_depth_gt, max_depth_input, s = 3.0, label = 'Low Resolution')
    plt.scatter(max_depth_gt, max_depth_diff, s = 3.0, label = 'Super Resolution')

    mae_input = mean_absolute_error(max_depth_gt, max_depth_input)
    mae_depth_diff = mean_absolute_error(max_depth_gt, max_depth_diff)

    plt.xlim(0,300)
    plt.ylim(0,300)
    frame_tick()
    plt.xlabel(r'Meltpool Depth [$\mu m]$ (High Resolution) ')
    plt.ylabel(r'Meltpool Depth [$\mu m]$ (Estimated) ')
    plt.title('Validation Set')
    plt.text(25, 250,r'MAE LR: {:.03}$\mu m$'.format(mae_input), fontsize = 8 )
    plt.text(25, 225,r'MAE SRDiff: {:.03}$\mu m$'.format(mae_depth_diff), fontsize = 8 )
    plt.savefig(os.path.join(analysis_folder, 'melt_pool_depth_validation_gridsearch.png'))
    plt.show()




    plt.figure(figsize = [4,3], dpi = 150)
    plt.plot(np.arange(300),np.arange(300), 'k-', linewidth = 2.0)
    plt.scatter(kp_depth_gt, kp_depth_input, s = 3.0, label = 'Low Resolution')
    plt.scatter(kp_depth_gt, kp_depth_diff, s = 3.0, label = 'Super Resolution')

    mae_input = mean_absolute_error(kp_depth_gt, kp_depth_input)
    mae_depth = mean_absolute_error(kp_depth_gt, kp_depth_diff)

    plt.xlim(0,300)
    plt.ylim(0,300)
    frame_tick()
    plt.xlabel(r'Keyhole Depth [$\mu m]$ (High Resolution) ')
    plt.ylabel(r'Keyhole Depth [$\mu m]$ (Estimated) ')
    plt.title('Test Set')
    plt.text(25, 250,r'MAE LR: {:.03}$\mu m$'.format(mae_input) )
    plt.text(25, 225,r'MAE SR: {:.03}$\mu m$'.format(mae_depth) )
    plt.savefig(os.path.join(analysis_folder, 'keyhole_depth_test_gridsearch.png'))

    plt.show()




# cases = []
# downscale = 'lanczos'
# for encoding in ['False', 'True']:
#     for condition in ['implicit', 'explicit']:
#         case = {}
#         case['downscale_method'] = downscale
#         case['encoding'] =encoding
#         case['conditioning'] = condition
        # if encoding == 'True':
        #     case['encode_flag'] = 'encoded'
        # else:
        #     case['encode_flag'] = 'upscaled'
#         cases.append(case)

# downscale = 'direct'
# for encoding, condition in zip(['False', 'True', 'False', 'True'], ['implicit', 'explicit', 'explicit', 'implicit']):
#     case = {}
#     case['downscale_method'] = downscale
#     case['encoding'] =encoding
#     case['conditioning'] = condition
#     if encoding == 'True':
#         case['encode_flag'] = 'encoded'
#     else:
#         case['encode_flag'] = 'upscaled'
#     cases.append(case)


# print(f"Found {len(cases)} to process")

diff_dirs =  ['/home/oogoke/DiffusionSR/runs/direct/diffusionimplicitencoded/2023_02_19_15_58_08/standardize/n_steps_1', '/home/oogoke/DiffusionSR/runs/direct/diffusionimplicitencoded/2023_02_19_15_57_33/standardize/n_steps_1', '/home/oogoke/DiffusionSR/runs/direct/diffusionimplicitencoded/2023_02_19_15_08_28/standardize/n_steps_1' ]
# for i, case in enumerate(cases):
#     diff_dir = find_simulation_case(case)
for i, diff_dir in enumerate(diff_dirs):
    # process(diff_dir, limit = None, skip = 50, gpu = i , run_skip_ablation = False)
# i = 0

    p = multiprocessing.Process(target = process, args = (diff_dir, None, 50, i, False))
    p.start()
p.join()



