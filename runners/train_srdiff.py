import argparse
import datetime
import os
import shutil

import torch
import yaml
from datasets.dataset import SimulationXZDataset
from torch.utils.data import DataLoader

from runners.train_diffusion import DiffusionModel
from runners.train_mobilenet import train_mobilenet
from runners.train_rrdn_encoder import pretrain_encoder
import wandb

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--config', type = str, default='implicit_diffusion.yml', help = "Path to the config file")
    parser.add_argument('--gpu', type = str, default = "7", help = 'Index of GPU to use (if only a single GPU is available, enter "0".' )
    parser.add_argument('--modeltype', type = str, default = 'mobilenet', help = "SR model to run. Options are diffusion, encoder, MobileNet")
    parser.add_argument('--restart_dir', type = str, default = '')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config
# Define parameters

args,  new_config = parse_args_and_config()
combined_dict = vars(args)
combined_dict.update(vars(new_config))

os.environ['CUDA_VISIBLE_DEVICES']  = args.gpu#"5"
residual_flag = new_config.residual_flag
modeltype = args.modeltype # possible options: diffusion, mobilenet, encoder
normalize_method = new_config.normalize_method
conditioning = new_config.conditioning # possible options: explicit, implicit
downscale_method = new_config.downscale_method
use_pretrained= new_config.use_pretrained
if 'enc_output' in new_config:
    enc_output = new_config.enc_output
else:
    enc_output = False

if args.restart_dir == '':
    restart = False
    restart_dir = ''
else:
    restart = True
    restart_dir = args.restart_dir
use_pretrained = new_config.use_pretrained
if use_pretrained:
    encoder_results_dir = new_config.encoder_results_dir
root_folder = new_config.root_folder
schedule = new_config.schedule
n_steps = int(new_config.n_steps)
timesteps = int(new_config.timesteps)
batch_size= int(new_config.batch_size)
encoding_flag = bool(new_config.encoding)
if encoding_flag:
    encoder = 'encoded'
else:
    encoder = 'upscaled'
# breakpoint()
field_names = None
if new_config.fields == 'temperature':
    field_names = ['temperature']
elif new_config.fields == 'all':
    field_names = None
elif new_config.fields == 'all_but_pressure':
    field_names = ['vx', 'temperature',  'vy', 'vz', 'liqlabel']
elif new_config.fields == 'temperature_liqlabel':
    field_names = ['temperature', 'liqlabel']
print(field_names)
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = int(new_config.epochs)
learning_rate = float(new_config.learning_rate)
if not 'loss_type' in new_config:
    loss_type = 'huber'
else:
    loss_type  = new_config.loss_type
if 'out_steps' in new_config:
    out_steps = new_config.out_steps
else:
    out_steps = None
if 'transform_rescale' in new_config:
    transform_rescale = new_config.transform_rescale
else:
    transform_rescale  = False
# Define dataset
train_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                     normalize=normalize_method,
                                     split='train',
                                     root_folder=root_folder,
                                     n_steps=n_steps, field_names = field_names, out_steps = out_steps)
test_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                    normalize=normalize_method,
                                    split='test',
                                    root_folder=root_folder,
                                    n_steps=n_steps, field_names = field_names, out_steps = out_steps)
dev_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                   normalize=normalize_method,
                                   split='dev',
                                   root_folder=root_folder,
                                   n_steps=n_steps, field_names = field_names, out_steps = out_steps)

# Define dataloader
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
# breakpoint()
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
                    test_dataset=test_dataset,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    learning_rate =learning_rate)

if modeltype == 'diffusion':
    if encoding_flag:
        if use_pretrained:
            print("Using pretrained model ... , " , encoder_results_dir)
        else:
            print('Pre-training encoder from scratch...')
            # Train encoder model
            encoder_results_dir = os.path.join('runs',   downscale_method, 'encoder', datetime_string, normalize_method, 'n_steps_{}'.format(n_steps))

            os.makedirs(encoder_results_dir, exist_ok=True)
            shutil.copy(os.path.join("configs", args.config), os.path.join(encoder_results_dir, 'configuration.yml'))
            pretrain_encoder(encoder_results_dir,
                            train_dataset=train_dataset,
                            dev_dataset=dev_dataset,
                            test_dataset=test_dataset, 
                            config = combined_dict)
    else:
        encoder_results_dir = 'no_encoder_used'
    if restart: # Resume training
        diffusion_results_dir = restart_dir
    else:
        diffusion_results_dir = os.path.join('runs',  downscale_method, modeltype + residual_tag+conditioning+encoder, datetime_string, normalize_method, 'n_steps_{}'.format(n_steps))
    # Train diffusion model
    os.makedirs(diffusion_results_dir, exist_ok=True)
    message = ''
    shutil.copy(os.path.join("configs", args.config), os.path.join(diffusion_results_dir, 'configuration.yml'))
    with open(os.path.join(diffusion_results_dir, 'information.txt'), 'w') as f:
        f.write(f'schedule: {schedule}, timesteps: {timesteps} fields: temp' + '\n '+ f'pretrained_encoder: {encoder_results_dir}' + f'\n  diffusion timesteps {timesteps}')
    if restart:
        with open(os.path.join(diffusion_results_dir, 'information_restart.txt'), 'w') as f:
            f.write(f'schedule: {schedule}, timesteps: {timesteps}, fields: temp' + '\n '+ f'pretrained_encoder: {encoder_results_dir}' + f'\n  diffusion timesteps {timesteps}')
    print("Training Diffusion...")

    wandb.init(
        project="Flow3D_SuperResolution",
        entity = "fogoke", 
        config=combined_dict,
        # mode = 'disabled' if config['data']['debug'] else 'online'
    )

    diffusion_model = DiffusionModel(results_folder=diffusion_results_dir,
                                     lr_encoder_folder=encoder_results_dir,
                                     train_dataset=train_dataset,
                                     dev_dataset=dev_dataset,
                                     test_dataset=test_dataset,
                                     timesteps=timesteps,
                                     conditioning=conditioning,
                                     encoding=encoding_flag,
                                     schedule=schedule,
                                     device=f'cuda:0',#{args.gpu.split(",")[0]}',
                                     enc_output = enc_output, 
                                     out_steps = out_steps, transform_rescale=transform_rescale
                                     )
    diffusion_model.train(epochs=epochs,
                          restart=restart,
                          restart_dir=restart_dir,

                          batch_size=batch_size,
                          learning_rate=learning_rate,
                          loss_type=loss_type
                          )
