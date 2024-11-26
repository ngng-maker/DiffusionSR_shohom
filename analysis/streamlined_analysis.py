import sys
# sys.path.append('/home/cmu/github/LPBFDiffusionSR/datasets')
import numpy as np
from pylab import gca
import numpy as np
import math
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
import time
import os
from skimage.metrics import structural_similarity as ssim_id
from analysis.plotting_functions import frame_tick, legend
import cv2
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
import pdb
from PIL import Image
import matplotlib.pyplot as plt 
# print(os.listdir('.'))
from datasets.dataset import SimulationXZDataset
import wandb
from runners.train_diffusion import DiffusionModel
from analysis_functions import initialize_diffusion

# from datasets.dataset import TemperatureXZDataset
from runners.train_diffusion import forwardpass
from analysis_functions import predict_lrenc, predict_refactored_diffusion, predict_mobilenet, predict_ddim_diffusion,predict_modified_diffusion, predict_diffusion, plot_images, get_profile, load_mobilenet, load_encoder, load_diffusion, PSNR, SSIM, multifield_plot_images
from models.diffusion_model import Unet
from models.lr_encoder_model import rrdbnet_encoder as rrdbnet_x4
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--wandb_id', type=str, help='Wandb ID', default = '3921imsz')
args = parser.parse_args()

wandb_id = args.wandb_id
api  = wandb.Api()
run = api.run(f'fogoke/Flow3D_SuperResolution/{wandb_id}')
config = run.config


diffusion_results_dir = config['restart_dir']
encoder_results_dir = config['encoder_results_dir']
timesteps = config['timesteps']
conditioning = config['conditioning']
encoding = config['encoding']
schedule = config['schedule']
device= 'cuda'
encode_bool = encoding == 'True'



os.environ['CUDA_VISIBLE_DEVICES']  = "0"
batch_size = 1
downscale_method = 'direct'
analysis_folder = f'analyzed_figures_paper_3_20/{timesteps}_{conditioning}_{schedule}_{downscale_method}'
os.makedirs( analysis_folder, exist_ok = True)

data_folder = config['root_folder']
train_dataset = SimulationXZDataset(downscale_method = 'direct', split = 'train', root_folder = data_folder , return_info = True)
test_dataset = SimulationXZDataset(downscale_method ='direct', split = 'test', root_folder = data_folder, return_info = True)
dev_dataset = SimulationXZDataset(downscale_method = 'direct', split = 'dev', root_folder = data_folder, return_info = True)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

diff_model = initialize_diffusion(diff_dir=diffusion_results_dir,
                                  enc_dir=encoder_results_dir,
                                  datasets=[train_dataset,
                                            dev_dataset, test_dataset],
                                  timesteps=timesteps,
                                  conditioning=conditioning,
                                  encoding=encoding,
                                  schedule=schedule,
                                  device=device)


lr_enc = load_encoder(encoder_results_dir, dataset = train_dataset)

# def predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, dataset, skip = 50):

#     '''
#     Return the predictions for the Diffusion model given an input batch
#     Parameters:
#     diff_model: DiffusionModel object
#     lr_enc: Torch network module object, representing the trained RRDN encoder model
#     res: Torch tensor, Residual between HR and LR data
#     hr: Torch tensor, High Resolution data
#     lr: Torch tensor, Low Resolution data
#     upscaled_lr: Torch tensor, Bicubic upscaled low resolution data
#     dataset: Dataset object, used for rescaling data
#     Returns:
#     Low resolution data, scaled to original space, 4-D numpy array (batch, channels, height, width)
#     Output (Super-resolution), scaled to original space, 4-D numpy array
#     High resolution data, scaled to original space, 4-D numpy array 
#     '''

#     if len(lr.shape) < 4:
#         img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
#         target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
#     else:
#         img = lr.to(device)
#         target = hr.to(device)
#     if len(lr.shape) < 4:

#         x_e = forwardpass(lr_enc, lr.view(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(device).float(), factor = dataset.factor)
#     else:
#         x_e = forwardpass(lr_enc, lr.to(device).float(), factor = dataset.factor)
#         # x_e = forwardpass(lr_enc, lr.to(device).float())
#         all_images = diff_model.batch_sample(dataset = dataset, batch = hr.to(device), x_e = x_e.to(device), sampler = 'DDPM', skip= skip)                
#         result = dataset.unscale_data(all_images.cpu().numpy()[-1, 0], input_type = 'residual') + dataset.unscale_data(upscaled_lr.numpy(), input_type = 'upscaled_lr')
        
#     return dataset.unscale_data(lr, input_type='lr'), result, dataset.unscale_data(target.cpu(), input_type = 'hr')


# for array in [input, upscaled_lr_data, result_diffusion, target]:

scaling_factor = 1

labels = ['Input', 'Bicubic Upscaling', 'CNN', 'Diffusion', 'Target']
batch_idxs = [399,2617,1708] # 260v900 364v900 400v650
# batch_idxs = [0,1,2]
timesteps = 1000
skip = 50
for k in range(10):
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(2.8*7.4*scaling_factor, 10*scaling_factor), dpi=300)
    fig.patch.set_alpha(0)
    for j,(row, batch_index) in enumerate(zip(axs, batch_idxs)):
        for batch_idx, (res, hr, lr, upscaled_lr, info_full) in tqdm(enumerate(test_dataloader), total = len(test_dataloader) ):
            if batch_idx == batch_index:
                input, result, target = predict_lrenc(lr_enc,res, hr, lr, upscaled_lr, train_dataset)
                input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, test_dataloader.dataset, skip = skip)
                # input, result_diffusion, target, _, _ = predict_modified_ddim_diffusion(diff_model.model, lr_enc, res, hr, lr, upscaled_lr,encoding = encode_bool,dataset =  train_dataset,seq= None, timesteps = timesteps,skip = skip, schedule = 'linear')
                upscaled_lr_data = test_dataloader.dataset.unscale_data(upscaled_lr, input_type = 'upscaled_lr')
                for i, (ax, array, label) in enumerate(zip(row,[input, upscaled_lr_data, result, result_diffusion, target], labels )):
                    if i == 0:
                        division_factor = 2
                        bound = 10
                    else:
                        division_factor  = 1
                        bound = 20
                    print("DIVISION FACTOR", division_factor)
                    xx, yy = np.meshgrid(np.arange(28//division_factor)*10*division_factor, np.arange(20//division_factor)*10*division_factor)
                    print(array.shape[-1])
                    im = ax.pcolormesh(xx, yy, array[0,0][12//division_factor:40//division_factor, bound:-bound].T,vmin = 293, vmax = 5000,cmap='jet')
                    ax.axis('equal')
                    ax.set_ylim([yy.min(), yy.max()])
                    # if i ==  0 and j == 0:
                        
                        
                        # frame_tick()
                    # else:
                    # ax.axis('off')
                    ax.set_title(label, fontsize = 15)
                    ax.xaxis.set_tick_params(labelbottom=False)
                    ax.yaxis.set_tick_params(labelleft =False)
                    # ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j == len(axs) - 1  and i == 0:
                        ax.set_ylabel(r'z $[\mu m]$')
                        ax.set_xlabel(r'x $[\mu m]$')
            elif batch_idx > batch_index:
                break
            
    fig.subplots_adjust(wspace = 0.01)#, hspace = 0.1)
    # fig.subplots_adjust()


    # Add colorbar
    cax = fig.add_axes([0.91, 0.12, 0.02, 0.77])
    clb = fig.colorbar(im, cax=cax)

    clb.set_ticks([293, 1000, 2000, 3000, 4000, 5000])
    clb.ax.set_title(r'T$[K]$', fontsize=15)
    plt.savefig('test.png')
    plt.clf()