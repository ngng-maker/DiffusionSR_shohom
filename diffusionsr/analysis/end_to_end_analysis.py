import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

import wandb
from diffusionsr.datasets.dataset import SimulationXZDataset
from diffusionsr.analysis.analysis_functions import (
    initialize_diffusion,
    predict_lrenc,
    predict_refactored_diffusion,
    load_encoder
)


# Parse WandB ID
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_id', type=str, default='3921imsz', help='Wandb ID')
wandb_id = parser.parse_args().wandb_id

# Load WandB run config
run = wandb.Api().run(f'fogoke/Flow3D_SuperResolution/{wandb_id}')
config = run.config

# Extract configuration details
device = 'cuda:0'
encode_bool = config['encoding'] == 'True'
analysis_folder = f"analyzed_figures_paper_3_20/{config['timesteps']}_{config['conditioning']}_{config['schedule']}_direct"
os.makedirs(analysis_folder, exist_ok=True)

# Dataset and DataLoader setup
data_folder = config['root_folder']
datasets = {split: SimulationXZDataset(downscale_method='direct', split=split, root_folder=data_folder, return_info=True)
            for split in ['train', 'dev', 'test']}
dataloaders = {split: DataLoader(datasets[split], batch_size=1, shuffle=False, drop_last=True)
               for split in ['train', 'dev', 'test']}
test_dataloader = dataloaders['test']  
train_dataset = datasets['train']



# Initialize diffusion model and encoder
diff_model = initialize_diffusion(
    diff_dir=config['restart_dir'],
    enc_dir=config['encoder_results_dir'],
    datasets=list(datasets.values()),
    timesteps=config['timesteps'],
    conditioning=config['conditioning'],
    encoding=config['encoding'],
    schedule=config['schedule'],
    device=device
)

lr_enc = load_encoder(config['encoder_results_dir'], dataset=datasets['train'])

SCALING_FACTOR = 1
FIGURE_HSIZE = 2.8*7.4
timesteps = 1000
skip = 50

labels = ['Input', 'Bicubic Upscaling', 'CNN', 'Diffusion', 'Target']
batch_idxs = [399,2617,1708] # p_v_: 260v900 364v900 400v650 
# def plot_rows(batch_idxs, test_dataloader, train_dataset, lr_enc, diff_model, skip ):

for k in range(10):
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(FIGURE_HSIZE*SCALING_FACTOR, 10*SCALING_FACTOR), dpi=300)
    fig.patch.set_alpha(0)
    for j,(row, batch_index) in enumerate(zip(axs, batch_idxs)):
        for batch_idx, (res, hr, lr, upscaled_lr, info_full) in tqdm(enumerate(test_dataloader), total = len(test_dataloader) ):
            if batch_idx == batch_index:
                input, result, target = predict_lrenc(lr_enc,res, hr, lr, upscaled_lr, train_dataset)
                input, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, test_dataloader.dataset, skip = skip)
                upscaled_lr_data = test_dataloader.dataset.unscale_data(upscaled_lr, input_type = 'upscaled_lr')
                for i, (ax, array, label) in enumerate(zip(row,[input, upscaled_lr_data, result, result_diffusion, target], labels )):
                    if i == 0:
                        division_factor = 2
                        bound = 10
                    else:
                        division_factor  = 1
                        bound = 20
                    xx, yy = np.meshgrid(np.arange(28//division_factor)*10*division_factor, np.arange(20//division_factor)*10*division_factor)
                    im = ax.pcolormesh(xx, yy, array[0,0][12//division_factor:40//division_factor, bound:-bound].T,vmin = 293, vmax = 5000,cmap='jet')
                    ax.axis('equal')
                    ax.set_ylim([yy.min(), yy.max()])
                    ax.set_title(label, fontsize = 15)
                    ax.xaxis.set_tick_params(labelbottom=False)
                    ax.yaxis.set_tick_params(labelleft=False)
                    # ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j == len(axs) - 1  and i == 0:
                        ax.set_ylabel(r'z $[\mu m]$')
                        ax.set_xlabel(r'x $[\mu m]$')
            elif batch_idx > batch_index:
                break
            
    fig.subplots_adjust(wspace = 0.01)
    
    # Add colorbar
    cax = fig.add_axes([0.91, 0.12, 0.02, 0.77])
    clb = fig.colorbar(im, cax=cax)

    clb.set_ticks([293, 1000, 2000, 3000, 4000, 5000])
    clb.ax.set_title(r'T$[K]$', fontsize=15)
    plt.savefig('test.png')
    plt.clf()