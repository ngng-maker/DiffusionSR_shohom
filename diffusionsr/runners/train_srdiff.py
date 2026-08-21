import argparse
import datetime
import os
import shutil

import torch
import yaml
from diffusionsr.datasets.dataset import SimulationXZDataset
from torch.utils.data import DataLoader

from diffusionsr.runners.train_diffusion import DiffusionModel
from diffusionsr.runners.train_mobilenet import train_mobilenet
from diffusionsr.runners.train_rrdn_encoder import pretrain_encoder
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
    parser.add_argument('--config', type=str, default='implicit_diffusion.yml',
                        help="Path to the config file (relative to configs/)")
    parser.add_argument('--gpu', type=str, default="7",
                        help='Index of GPU to use')
    parser.add_argument('--modeltype', type=str, default='mobilenet',
                        help="SR model to run. Options: diffusion | flow_matching | uncond_diffusion | ldm | encoder | mobilenet")
    parser.add_argument('--restart_dir', type=str, default='',
                        help="Legacy: diffusion restart dir (use --force_run_dir instead)")
    # Stable-path args for preemption-safe SLURM restart
    parser.add_argument('--force_enc_dir', type=str, default='',
                        help="Fixed encoder results dir (overrides datetime-based naming). "
                             "If bestmodel_saved.pth exists here, encoder training is skipped.")
    parser.add_argument('--force_run_dir', type=str, default='',
                        help="Fixed diffusion results dir (overrides datetime-based naming). "
                             "If ckpt.pth exists here, training is automatically resumed.")
    args = parser.parse_args()

    # Support absolute paths, cwd-relative paths (e.g. configs/multifield/...),
    # and legacy bare filenames (looked up under configs/)
    _cfg = args.config
    if os.path.isabs(_cfg) or os.path.exists(_cfg):
        config_path = _cfg
    else:
        config_path = os.path.join("configs", _cfg)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config, config_path

args, new_config, config_path = parse_args_and_config()
combined_dict = vars(args)
combined_dict.update(vars(new_config))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
residual_flag = new_config.residual_flag
modeltype = args.modeltype
normalize_method = new_config.normalize_method
conditioning = new_config.conditioning
downscale_method = new_config.downscale_method
use_pretrained = new_config.use_pretrained
if 'enc_output' in new_config:
    enc_output = new_config.enc_output
else:
    enc_output = False

# Determine restart state for diffusion model
if args.force_run_dir:
    # Stable path: always use force_run_dir; auto-detect whether to restart
    diffusion_run_dir_fixed = args.force_run_dir
    restart = os.path.exists(os.path.join(args.force_run_dir, 'ckpt.pth'))
    restart_dir = args.force_run_dir if restart else ''
elif args.restart_dir:
    diffusion_run_dir_fixed = None
    restart = True
    restart_dir = args.restart_dir
else:
    diffusion_run_dir_fixed = None
    restart = False
    restart_dir = ''

use_pretrained = new_config.use_pretrained
if use_pretrained and not args.force_enc_dir:
    encoder_results_dir = new_config.encoder_results_dir
root_folder = new_config.root_folder
schedule = new_config.schedule
n_steps = int(new_config.n_steps)
timesteps = int(new_config.timesteps)
batch_size = int(new_config.batch_size)
encoding_flag = bool(new_config.encoding)
if encoding_flag:
    encoder = 'encoded'
else:
    encoder = 'upscaled'

# Field selection
field_names = None
fields_val = getattr(new_config, 'fields', 'all')
if fields_val == 'temperature':
    field_names = ['temperature']
elif fields_val == 'liqlabel':
    field_names = ['liqlabel']
elif fields_val == 'temperature_liqlabel':
    field_names = ['temperature', 'liqlabel']
elif fields_val == 'temperature_liqlabel_pressure':
    field_names = ['temperature', 'liqlabel', 'pressure']
elif fields_val == 'all_but_pressure':
    field_names = ['vx', 'temperature', 'vy', 'vz', 'liqlabel']
elif fields_val == 'all':
    field_names = None
print(field_names)

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = int(new_config.epochs)
learning_rate = float(new_config.learning_rate)
loss_type = getattr(new_config, 'loss_type', 'huber')
out_steps = getattr(new_config, 'out_steps', None)
transform_rescale = getattr(new_config, 'transform_rescale', False)

# Datasets
train_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                     normalize=normalize_method,
                                     split='train',
                                     root_folder=root_folder,
                                     n_steps=n_steps, field_names=field_names, out_steps=out_steps)
test_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                    normalize=normalize_method,
                                    split='test',
                                    root_folder=root_folder,
                                    n_steps=n_steps, field_names=field_names, out_steps=out_steps)
dev_dataset = SimulationXZDataset(downscale_method=downscale_method,
                                   normalize=normalize_method,
                                   split='dev',
                                   root_folder=root_folder,
                                   n_steps=n_steps, field_names=field_names, out_steps=out_steps)

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

now = datetime.datetime.now()
print("Current date and time:")
datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
print(datetime_string)

residual_tag = 'residual' if residual_flag else ''


# ── MOBILENET ─────────────────────────────────────────────────────────────────
if modeltype == 'mobilenet':
    results_dir = os.path.join('runs', downscale_method, modeltype, datetime_string,
                               normalize_method, f'n_steps_{n_steps}')
    os.makedirs(results_dir, exist_ok=True)
    train_mobilenet(results_folder=results_dir,
                    train_dataset=train_dataset,
                    dev_dataset=dev_dataset,
                    test_dataset=test_dataset,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate)


# ── SHARED: encoder training helper ───────────────────────────────────────────
def _run_or_skip_encoder(enc_dir):
    """Train encoder unless bestmodel_saved.pth already exists (prior run completed)."""
    os.makedirs(enc_dir, exist_ok=True)
    if os.path.exists(os.path.join(enc_dir, 'bestmodel_saved.pth')):
        print(f"Encoder already trained at {enc_dir} — skipping encoder stage.")
        return
    shutil.copy(config_path, os.path.join(enc_dir, 'configuration.yml'))
    pretrain_encoder(enc_dir,
                     train_dataset=train_dataset,
                     dev_dataset=dev_dataset,
                     test_dataset=test_dataset,
                     config=combined_dict)


# ── DIFFUSION (DiffusionSR) ───────────────────────────────────────────────────
if modeltype == 'diffusion':
    if encoding_flag:
        if args.force_enc_dir:
            encoder_results_dir = args.force_enc_dir
        elif use_pretrained:
            encoder_results_dir = new_config.encoder_results_dir
        else:
            encoder_results_dir = os.path.join(
                'runs', downscale_method, 'encoder', datetime_string,
                normalize_method, f'n_steps_{n_steps}')
        _run_or_skip_encoder(encoder_results_dir)
    else:
        encoder_results_dir = 'no_encoder_used'

    if diffusion_run_dir_fixed:
        diffusion_results_dir = diffusion_run_dir_fixed
    else:
        diffusion_results_dir = os.path.join(
            'runs', downscale_method,
            f'diffusion{residual_tag}{conditioning}{encoder}',
            datetime_string, normalize_method, f'n_steps_{n_steps}')

    os.makedirs(diffusion_results_dir, exist_ok=True)
    shutil.copy(config_path,
                os.path.join(diffusion_results_dir, 'configuration.yml'))
    with open(os.path.join(diffusion_results_dir, 'information.txt'), 'w') as f:
        f.write(f'schedule: {schedule}, timesteps: {timesteps} fields: {fields_val}\n'
                f'pretrained_encoder: {encoder_results_dir}\n'
                f'diffusion timesteps {timesteps}')
    if restart:
        with open(os.path.join(diffusion_results_dir, 'information_restart.txt'), 'w') as f:
            f.write(f'Restarted from ckpt.pth')

    print("Training DiffusionSR...")
    wandb.init(project="Flow3D_SuperResolution", entity=os.getenv("WANDB_ENTITY"),
               config=combined_dict)

    diffusion_model = DiffusionModel(
        results_folder=diffusion_results_dir,
        lr_encoder_folder=encoder_results_dir,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        timesteps=timesteps,
        conditioning=conditioning,
        encoding=encoding_flag,
        schedule=schedule,
        device='cuda:0',
        enc_output=enc_output,
        out_steps=out_steps,
        transform_rescale=transform_rescale,
    )
    diffusion_model.train(epochs=epochs, restart=restart, restart_dir=restart_dir,
                          batch_size=batch_size, learning_rate=learning_rate,
                          loss_type=loss_type)


# ── FLOW MATCHING ─────────────────────────────────────────────────────────────
if modeltype == 'flow_matching':
    from diffusionsr.runners.train_flow_matching import FlowMatchingModel
    fm_timescale = float(getattr(new_config, 'fm_timescale', timesteps))
    fm_n_steps = int(getattr(new_config, 'fm_n_steps', 100))

    if encoding_flag:
        if args.force_enc_dir:
            encoder_results_dir = args.force_enc_dir
        elif use_pretrained:
            encoder_results_dir = new_config.encoder_results_dir
        else:
            encoder_results_dir = os.path.join(
                'runs', downscale_method, 'encoder', datetime_string,
                normalize_method, f'n_steps_{n_steps}')
        _run_or_skip_encoder(encoder_results_dir)
    else:
        encoder_results_dir = 'no_encoder_used'

    if diffusion_run_dir_fixed:
        diffusion_results_dir = diffusion_run_dir_fixed
    else:
        diffusion_results_dir = os.path.join(
            'runs', downscale_method,
            f'flowmatching{conditioning}{encoder}',
            datetime_string, normalize_method, f'n_steps_{n_steps}')

    os.makedirs(diffusion_results_dir, exist_ok=True)
    shutil.copy(config_path,
                os.path.join(diffusion_results_dir, 'configuration.yml'))

    print("Training Flow Matching...")
    wandb.init(project="Flow3D_SuperResolution", entity=os.getenv("WANDB_ENTITY"),
               config=combined_dict)

    fm_model = FlowMatchingModel(
        results_folder=diffusion_results_dir,
        lr_encoder_folder=encoder_results_dir,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        timesteps=timesteps,
        conditioning=conditioning,
        encoding=encoding_flag,
        schedule=schedule,
        device='cuda:0',
        enc_output=enc_output,
        out_steps=out_steps,
        transform_rescale=transform_rescale,
        fm_timescale=fm_timescale,
    )
    fm_model.train(epochs=epochs, restart=restart, restart_dir=restart_dir,
                   batch_size=batch_size, learning_rate=learning_rate,
                   loss_type=loss_type)


# ── UNCONDITIONAL DIFFUSION ────────────────────────────────────────────────────
if modeltype == 'uncond_diffusion':
    if diffusion_run_dir_fixed:
        diffusion_results_dir = diffusion_run_dir_fixed
    else:
        diffusion_results_dir = os.path.join(
            'runs', downscale_method, 'uncond_diffusion',
            datetime_string, normalize_method, f'n_steps_{n_steps}')

    os.makedirs(diffusion_results_dir, exist_ok=True)
    shutil.copy(config_path,
                os.path.join(diffusion_results_dir, 'configuration.yml'))

    print("Training Unconditional Diffusion...")
    wandb.init(project="Flow3D_SuperResolution", entity=os.getenv("WANDB_ENTITY"),
               config=combined_dict)

    uncond_model = DiffusionModel(
        results_folder=diffusion_results_dir,
        lr_encoder_folder='no_encoder_used',
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        timesteps=timesteps,
        conditioning='none',
        encoding=False,
        schedule=schedule,
        device='cuda:0',
        enc_output=False,
        out_steps=out_steps,
        transform_rescale=False,
    )
    uncond_model.train(epochs=epochs, restart=restart, restart_dir=restart_dir,
                       batch_size=batch_size, learning_rate=learning_rate,
                       loss_type=loss_type)


# ── LATENT DIFFUSION MODEL ────────────────────────────────────────────────────
if modeltype == 'ldm':
    from diffusionsr.runners.train_ldm import pretrain_vae, LDMModel

    vae_epochs = int(getattr(new_config, 'vae_epochs', 100))

    # Stage 1: encoder
    if encoding_flag:
        if args.force_enc_dir:
            encoder_results_dir = args.force_enc_dir
        elif use_pretrained:
            encoder_results_dir = new_config.encoder_results_dir
        else:
            encoder_results_dir = os.path.join(
                'runs', downscale_method, 'encoder', datetime_string,
                normalize_method, f'n_steps_{n_steps}')
        _run_or_skip_encoder(encoder_results_dir)
    else:
        encoder_results_dir = 'no_encoder_used'

    # Stage 2: VAE
    vae_dir_key = getattr(new_config, 'vae_results_dir', '')
    if vae_dir_key:
        vae_results_dir = vae_dir_key
    elif args.force_enc_dir:
        # Store VAE alongside the encoder
        vae_results_dir = args.force_enc_dir.replace('/encoder/', '/vae/') \
            if '/encoder/' in args.force_enc_dir \
            else args.force_enc_dir + '_vae'
    else:
        vae_results_dir = os.path.join(
            'runs', downscale_method, 'vae', datetime_string,
            normalize_method, f'n_steps_{n_steps}')

    # Stage 3: latent DDPM
    if diffusion_run_dir_fixed:
        diffusion_results_dir = diffusion_run_dir_fixed
    else:
        diffusion_results_dir = os.path.join(
            'runs', downscale_method, f'ldm{conditioning}{encoder}',
            datetime_string, normalize_method, f'n_steps_{n_steps}')

    os.makedirs(diffusion_results_dir, exist_ok=True)
    shutil.copy(config_path,
                os.path.join(diffusion_results_dir, 'configuration.yml'))

    print("Training Latent Diffusion Model...")
    wandb.init(project="Flow3D_SuperResolution", entity=os.getenv("WANDB_ENTITY"),
               config=combined_dict)

    os.makedirs(vae_results_dir, exist_ok=True)
    pretrain_vae(vae_results_dir, train_dataset=train_dataset,
                 dev_dataset=dev_dataset, test_dataset=test_dataset,
                 num_epochs=vae_epochs)

    ldm_model = LDMModel(
        vae_folder=vae_results_dir,
        results_folder=diffusion_results_dir,
        lr_encoder_folder=encoder_results_dir,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        timesteps=timesteps,
        conditioning=conditioning,
        encoding=encoding_flag,
        schedule=schedule,
        device='cuda:0',
        enc_output=enc_output,
        out_steps=out_steps,
    )
    ldm_model.train(epochs=epochs, restart=restart, restart_dir=restart_dir,
                    batch_size=batch_size, learning_rate=learning_rate,
                    loss_type=loss_type)


# ── ENCODER ONLY ──────────────────────────────────────────────────────────────
if modeltype == 'encoder':
    if args.force_enc_dir:
        encoder_results_dir = args.force_enc_dir
    else:
        encoder_results_dir = os.path.join(
            'runs', downscale_method, 'encoder', datetime_string,
            normalize_method, f'n_steps_{n_steps}')
    _run_or_skip_encoder(encoder_results_dir)
