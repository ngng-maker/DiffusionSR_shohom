import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from diffusionsr.datasets.dataset import SimulationXZDataset
import wandb
from diffusionsr.analysis.analysis_functions import initialize_diffusion
from diffusionsr.analysis.analysis_functions import predict_lrenc, predict_refactored_diffusion, load_encoder
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--wandb_id', type=str, help='Wandb ID', default = '3921imsz')
parser.add_argument('--wandb_entity', type=str, default=os.getenv("WANDB_ENTITY"),
                    help='Wandb entity (defaults to WANDB_ENTITY env var)')
parser.add_argument('--wandb_project', type=str, default='Flow3D_SuperResolution',
                    help='Wandb project name')
args = parser.parse_args()

if args.wandb_entity is None:
    raise ValueError("Set --wandb_entity or the WANDB_ENTITY env var.")

wandb_id = args.wandb_id
api  = wandb.Api()
run = api.run(f'{args.wandb_entity}/{args.wandb_project}/{wandb_id}')
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
                input_sample, result, target = predict_lrenc(lr_enc,res, hr, lr, upscaled_lr, train_dataset)
                input_sample, result_diffusion, target = predict_refactored_diffusion(diff_model, lr_enc, res, hr, lr, upscaled_lr, test_dataloader.dataset, skip = skip)
                upscaled_lr_data = test_dataloader.dataset.unscale_data(upscaled_lr, input_type = 'upscaled_lr')
                for i, (ax, array, label) in enumerate(zip(row,[input_sample, upscaled_lr_data, result, result_diffusion, target], labels )):
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