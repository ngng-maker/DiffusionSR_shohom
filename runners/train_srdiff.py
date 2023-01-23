import datetime
import torch
import os
from torch.utils.data import DataLoader
from datasets.dataset import SimulationXZDataset as TemperatureXZDataset
from runners.train_mobilenet import train_mobilenet
from runners.train_rrdn_encoder import pretrain_encoder
from runners.train_diffusion import DiffusionModel
import argparse
import yaml
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
    parser.add_argument('--gpu', type = str, default = "0", help = 'Index of GPU to use (if only a single GPU is available, enter "0".' )
    parser.add_argument('--modeltype', type = str, default = 'diffusion', help = "SR model to run. Options are diffusion, encoder, MobileNet")
    parser.add_argument('--restart_dir', type = str, default = '')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config
# Define parameters

args,  new_config = parse_args_and_config()
os.environ['CUDA_VISIBLE_DEVICES']  = args.gpu#"5"
residual_flag = new_config.residual_flag
modeltype = args.modeltype # possible options: diffusion, mobilenet, encoder
normalize_method = new_config.normalize_method
conditioning = new_config.conditioning # possible options: explicit, implicit
downscale_method = new_config.downscale_method
use_pretrained= new_config.use_pretrained
if args.restart_dir == '':
    restart = False
else:
    restart_dir = args.restart_dir
use_pretrained = new_config.use_pretrained
if use_pretrained:
    encoder_results_dir = new_config.encoder_results_dir
root_folder = new_config.root_folder
schedule = new_config.schedule
n_steps = int(new_config.n_steps)
timesteps = int(new_config.timesteps)
batch_size= int(new_config.batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = int(new_config.epochs)
learning_rate = float(new_config.learning_rate)

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

if modeltype == 'diffusion':
    if use_pretrained:
        print("Using pretrained model ... , " , encoder_results_dir)
    else:
        print('Pre-training encoder from scratch...')
        # Train encoder model
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
    # Train diffusion model
    os.makedirs(diffusion_results_dir, exist_ok=True)
    message = ''
    with open(os.path.join(diffusion_results_dir, 'information.txt'), 'w') as f:
        f.write('schedule: {}, timesteps: {}'.format(schedule, str(timesteps)) + '\n '+ 'pretrained_encoder: {}'.format(encoder_results_dir) + '\n  diffusion timesteps {}'.format(timesteps))
    if restart:
        with open(os.path.join(diffusion_results_dir, 'information_restart.txt'), 'w') as f:
            f.write('schedule: {}, timesteps: {}'.format(schedule, str(timesteps)) + '\n '+ 'pretrained_encoder: {}'.format(encoder_results_dir) + '\n  diffusion timesteps {}'.format(timesteps))
    print("Training Diffusion...")
    # train_diffusion(results_folder= diffusion_results_dir,
    #                 lr_encoder_folder=encoder_results_dir,
    #                 train_dataset=train_dataset,
    #                 dev_dataset=dev_dataset,
    #                 test_dataset=test_dataset,
    #                 timesteps=timesteps,
    #                 restart=restart,
    #                 restart_dir=diffusion_results_dir,
    #                 conditioning = conditioning,
    #                 schedule=schedule,
    #                 device = device)
    diffusion_model = DiffusionModel(results_folder =diffusion_results_dir,
                 lr_encoder_folder =encoder_results_dir ,
                 train_dataset = train_dataset,
                 dev_dataset= dev_dataset,
                 test_dataset = test_dataset,
                 timesteps=timesteps,
                 conditioning=conditioning,
                 schedule=schedule,
                 epochs=epochs,
                 image_size=40,
                 batch_size=batch_size,
                 learning_rate=learning_rate,
                 device = 'cuda', 
                 loss_type = 'huber')
    diffusion_model.train()
