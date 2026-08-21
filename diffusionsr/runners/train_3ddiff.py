import argparse
import datetime
import os
import shutil

import torch
import yaml
from diffusionsr.datasets.dataset import SimulationXZDataset
from torch.utils.data import DataLoader

from diffusionsr.runners.train_diffusion_3d import DiffusionModel3D
from diffusionsr.runners.train_mobilenet import train_mobilenet
from diffusionsr.runners.train_rrdn_encoder import pretrain_encoder
from diffusionsr.runners.train_vae import VAETrainer, parse_channel_multipliers
from diffusionsr.utils import dict2namespace, config_to_field_names
import wandb

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--config', type = str, default='implicit_diffusion.yml', help = "Path to the config file")
    parser.add_argument('--gpu', type = str, default = "7", help = 'Index of GPU to use (if only a single GPU is available, enter "0".' )
    parser.add_argument('--modeltype', type = str, default = 'mobilenet', help = "SR model to run. Options are diffusion, encoder, MobileNet")
    parser.add_argument('--restart_dir', type = str, default = '')
    parser.add_argument('--additional_epochs', type=int, default=None, help='When restarting, train for this many extra epochs beyond the checkpoint epoch.')
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
residual_flag = new_config.residual_flag  # Used in diffusionsr/runners/train_srdiff.py to set the diffusion run-folder tag via residual_tag.
modeltype = args.modeltype # possible options: diffusion, mobilenet, encoder; used in diffusionsr/runners/train_srdiff.py to choose the training branch and name mobilenet/diffusion run folders.
normalize_method = new_config.normalize_method  # Used in diffusionsr/runners/train_srdiff.py when building datasets, then consumed in diffusionsr/datasets/dataset.py for rescale/unscale behavior, and included in run-folder names here.
conditioning = new_config.conditioning # possible options: explicit, implicit; passed from diffusionsr/runners/train_srdiff.py into diffusionsr/runners/train_diffusion.py::DiffusionModel, where it selects the diffusion conditioning mode and contributes to the run-folder name here.
downscale_method = new_config.downscale_method  # Used in diffusionsr/runners/train_srdiff.py when building dataset paths and run-folder names, then consumed in diffusionsr/datasets/dataset.py to locate LR data.
use_pretrained= new_config.use_pretrained  # Used in diffusionsr/runners/train_srdiff.py to decide whether to reuse encoder_results_dir or train a fresh encoder before diffusion training.
inflate_dim = int(new_config.inflate_dim) if hasattr(new_config, 'inflate_dim') and new_config.inflate_dim is not None else None
inflate_method = new_config.inflate_method if hasattr(new_config, 'inflate_method') else 'repeat'


if hasattr(new_config, 'enc_output'):
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
    if use_pretrained:
        print("Ignoring use_pretrained because encoding is disabled.")
# breakpoint()
field_names = config_to_field_names(new_config.fields)
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = int(new_config.epochs)
if args.additional_epochs is not None:
    additional_epochs = args.additional_epochs
elif hasattr(new_config, 'additional_epochs') and new_config.additional_epochs is not None:
    additional_epochs = int(new_config.additional_epochs)
else:
    additional_epochs = None
learning_rate = float(new_config.learning_rate)
if not hasattr(new_config, 'loss_type'):
    loss_type = 'huber'
else:
    loss_type  = new_config.loss_type
if hasattr(new_config, 'out_steps'):
    out_steps = new_config.out_steps
else:
    out_steps = None
if hasattr(new_config, 'transform_rescale'):
    transform_rescale = new_config.transform_rescale
else:
    transform_rescale  = False
voxel_save_interval = int(new_config.voxel_save_interval) if hasattr(new_config, 'voxel_save_interval') and new_config.voxel_save_interval is not None else 0
voxel_sample_batch_idx = int(new_config.voxel_sample_batch_idx) if hasattr(new_config, 'voxel_sample_batch_idx') and new_config.voxel_sample_batch_idx is not None else 0
voxel_threshold = float(new_config.voxel_threshold) if hasattr(new_config, 'voxel_threshold') and new_config.voxel_threshold is not None else 1800.0
voxel_channel = int(new_config.voxel_channel) if hasattr(new_config, 'voxel_channel') and new_config.voxel_channel is not None else 0
voxel_sampler = new_config.voxel_sampler if hasattr(new_config, 'voxel_sampler') and new_config.voxel_sampler is not None else 'DDIM'
voxel_skip = int(new_config.voxel_skip) if hasattr(new_config, 'voxel_skip') and new_config.voxel_skip is not None else 10
print(field_names)
print(
    {
        'config': args.config,
        'modeltype': modeltype,
        'encoding': encoding_flag,
        'use_pretrained': use_pretrained,
        'additional_epochs': additional_epochs,
        'inflate_dim': inflate_dim,
        'inflate_method': inflate_method,
        'voxel_save_interval': voxel_save_interval,
        'voxel_sample_batch_idx': voxel_sample_batch_idx,
    }
)
# Define dataset
train_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                     normalize=normalize_method,
                                     split='train',
                                     root_folder=root_folder,
                                     n_steps=n_steps, field_names = field_names, out_steps = out_steps,
                                     inflate_dim=inflate_dim, inflate_method=inflate_method)
test_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                    normalize=normalize_method,
                                    split='test',
                                    root_folder=root_folder,
                                    n_steps=n_steps, field_names = field_names, out_steps = out_steps,
                                    inflate_dim=inflate_dim, inflate_method=inflate_method)
dev_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                   normalize=normalize_method,
                                   split='dev',
                                   root_folder=root_folder,
                                   n_steps=n_steps, field_names = field_names, out_steps = out_steps,
                                   inflate_dim=inflate_dim, inflate_method=inflate_method)

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

if modeltype in ['vae', 'vae3d']:
    if restart:
        vae_results_dir = restart_dir
    else:
        vae_results_dir = os.path.join('runs', downscale_method, 'vae3d', datetime_string, normalize_method, 'n_steps_{}'.format(n_steps))
    os.makedirs(vae_results_dir, exist_ok=True)
    shutil.copy(os.path.join("configs", args.config), os.path.join(vae_results_dir, 'configuration.yml'))
    vae_target = getattr(new_config, 'vae_target', 'hr')
    vae_input_type = getattr(new_config, 'vae_input_type', getattr(new_config, 'vae_input', vae_target))
    with open(os.path.join(vae_results_dir, 'information.txt'), 'w') as f:
        f.write(
            f'vae_spatial_dims: 3\nfields: {field_names}\ncreated_at: {datetime_string}\n'
            f'inflate_dim: {inflate_dim}\ninflate_method: {inflate_method}\n'
            f'input: {vae_input_type}\ntarget: {vae_target}'
        )

    print('Training 3D VAE...')
    wandb.init(
        project="Flow3D_SuperResolution",
        entity=os.getenv("WANDB_ENTITY"),
        config=combined_dict,
    )

    output_activation = getattr(new_config, 'vae_output_activation', None)
    if output_activation in ['', 'none', 'None']:
        output_activation = None
    vae_trainer = VAETrainer(
        results_folder=vae_results_dir,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        spatial_dims=3,
        target_type=vae_target,
        input_type=vae_input_type,
        input_channels=getattr(new_config, 'vae_input_channels', None),
        output_channels=getattr(new_config, 'vae_output_channels', None),
        latent_channels=int(getattr(new_config, 'vae_latent_channels', 4)),
        hidden_channels=int(getattr(new_config, 'vae_hidden_channels', 16)),
        channel_multipliers=parse_channel_multipliers(getattr(new_config, 'vae_channel_multipliers', (1, 2, 4))),
        output_activation=output_activation,
        beta=float(getattr(new_config, 'vae_beta', 1e-4)),
        reconstruction_loss=getattr(new_config, 'vae_reconstruction_loss', 'l1'),
        kl_anneal_epochs=int(getattr(new_config, 'vae_kl_anneal_epochs', 0)),
        num_workers=int(getattr(new_config, 'vae_num_workers', 0)),
        log_interval=int(getattr(new_config, 'vae_log_interval', 50)),
        sample_interval=int(getattr(new_config, 'vae_sample_interval', 1)),
        save_every=int(getattr(new_config, 'vae_save_every', 0)),
        grad_clip_norm=getattr(new_config, 'vae_grad_clip_norm', None),
        depth_size=getattr(train_dataset, 'inflate_dim', None),
    )
    vae_trainer.train(
        epochs=epochs,
        restart=restart,
        restart_dir=restart_dir,
        additional_epochs=additional_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=float(getattr(new_config, 'vae_weight_decay', 0.0)),
    )

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
        print('Encoder path disabled because encoding is False.')
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
        entity=os.getenv("WANDB_ENTITY"),
        config=combined_dict,
        # mode = 'disabled' if config['data']['debug'] else 'online'
    )

    diffusion_model = DiffusionModel3D(results_folder=diffusion_results_dir,
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
                          additional_epochs=additional_epochs,
                          batch_size=batch_size,
                          learning_rate=learning_rate,
                          loss_type=loss_type,
                          voxel_save_interval=voxel_save_interval,
                          voxel_sample_batch_idx=voxel_sample_batch_idx,
                          voxel_threshold=voxel_threshold,
                          voxel_channel=voxel_channel,
                          voxel_sampler=voxel_sampler,
                          voxel_skip=voxel_skip,
                          )
