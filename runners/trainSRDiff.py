import datetime
import os
import matplotlib.pyplot as plt
from datasets.dataset import TemperatureXZDataset
from pylab import gca
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from train_rrdn_encoder import pretrain_encoder
from train_mobilenet import train_mobilenet
# Define parameters
os.environ['CUDA_VISIBLE_DEVICES']  = "3"
residual_flag = False
modeltype = 'diffusion' # possible options: diffusion, mobilenet, encoder
normalize_method = 'standardize'
conditioning = 'explicit' # possible options: explicit, implicit
downscale_method = 'direct'
mobilenet_flag = False
use_pretrained= False
restart = False
restart_dir = '/home/oogoke/DiffusionSR/runs/clean/direct/diffusion/2022_12_12_02_31_25/standardize/n_steps_1'
encoder_results_dir = '/home/oogoke/DiffusionSR/runs/clean/direct/encoder/2023_01_04_21_59_47/standardize/n_steps_3'##'/home/oogoke/DiffusionSR/runs/clean/direct/encoder/2022_12_04_11_32_01/standardize'
root_folder = 'datasets/update_v2_laser_velocity_xz_cross_section_data'
schedule = 'linear'
n_steps = 3
timesteps = 2000
batch_size= 16

# Define dataset
train_dataset = TemperatureXZDataset(downscale_method=downscale_method,
                                     normalize=normalize_method,
                                     split='train',
                                     root_folder=root_folder,
                                     n_steps=n_steps)
test_dataset = TemperatureXZDataset(downscale_method=downscale_method,
                                    normalize=normalize_method,
                                    split='test',
                                    root_folder=root_folder,
                                    n_steps=n_steps)
dev_dataset = TemperatureXZDataset(downscale_method=downscale_method,
                                   normalize=normalize_method,
                                   split='dev',
                                   root_folder=root_folder,
                                   n_steps=n_steps)


dataloader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)

dev_dataloader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)



# Define which version of the model to use
# if conditioning == 'explicit':
#     if residual_flag:
#         from misc.residual_train_no_encoder_diffusion import train_diffusion
#     else:
#         from  misc.train_diffusion_no_encoder import train_diffusion
# elif conditioning == 'implicit':
#     if residual_flag:
#         from misc.residualtrain_diffusion import train_diffusion
#     else:
from runners.train_diffusion import train_diffusion

# Create folder to save results
now = datetime.datetime.now()
print ("Current date and time : ")
datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
print(datetime_string)


if residual_flag:
    residual_tag = 'residual'
else:
    residual_tag = ''



if modeltype == 'mobilenet':
    results_dir = os.path.join('runs',  downscale_method, modeltype, datetime_string, normalize_method, 'n_steps_{}'.format(n_steps))
    os.makedirs(results_dir, exist_ok=True)
    train_mobilenet(results_folder=results_dir,
                    train_dataset=train_dataset,
                    dev_dataset=dev_dataset,
                    test_dataset=test_dataset)

if modeltype == 'diffusion':
    if use_pretrained:
        print("Using pretrained, " , encoder_results_dir)
    else:
        print('Training encoder from scratch')
        encoder_results_dir = os.path.join('runs',   downscale_method, 'encoder', datetime_string, normalize_method, 'n_steps_{}'.format(n_steps))
        os.makedirs(encoder_results_dir, exist_ok=True)
        pretrain_encoder(encoder_results_dir,
                         train_dataset=train_dataset,
                         dev_dataset=dev_dataset,
                         test_dataset=test_dataset)

    if restart: # Resume training
        diffusion_results_dir = restart_dir
    else:
        diffusion_results_dir = os.path.join('runs',  downscale_method, modeltype + residual_tag+conditioning, datetime_string, normalize_method, 'n_steps_{}'.format(n_steps))
    
    os.makedirs(diffusion_results_dir, exist_ok=True)
    message = ''
    with open(os.path.join(diffusion_results_dir, 'information.txt'), 'w') as f:
        f.write('schedule: {}, timesteps: {}'.format(schedule, str(timesteps)) + '\n '+ 'pretrained_encoder: {}'.format(encoder_results_dir) + '\n  diffusion timesteps {}'.format(timesteps))
    if restart:
        with open(os.path.join(diffusion_results_dir, 'information_restart.txt'), 'w') as f:
            f.write('schedule: {}, timesteps: {}'.format(schedule, str(timesteps)) + '\n '+ 'pretrained_encoder: {}'.format(encoder_results_dir) + '\n  diffusion timesteps {}'.format(timesteps))

    train_diffusion(results_folder= diffusion_results_dir,
                    lr_encoder_folder=encoder_results_dir,
                    train_dataset=train_dataset,
                    dev_dataset=dev_dataset,
                    test_dataset=test_dataset,
                    timesteps=timesteps,
                    restart=restart,
                    restart_dir=diffusion_results_dir,
                    conditioning = conditioning,
                    schedule=schedule)

