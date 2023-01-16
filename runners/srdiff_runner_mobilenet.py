
import os
import matplotlib.pyplot as plt
from datasets.dataset import TemperatureXZDataset
from pylab import gca
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from misc.residualtrain_diffusion import train_diffusion
from train_mobilenet import train_mobilenet
from train_rrdn_encoder import pretrain_encoder
os.environ['CUDA_VISIBLE_DEVICES']  = "4"

from PIL import Image
import requests
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

'''
Make frame thicker, make tick pointing inside, make tick thicker
default frame width is 2, default tick width is 1.5
'''
def frame_tick(frame_width = 2, tick_width = 1.5):
    ax = gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction = 'in', 
                    width = tick_width)

'''
legend:
default location : upper left
default fontsize: 8
Frame is always off
'''
def legend(location = 'upper left', fontsize = 8):
    plt.legend(loc = location, fontsize = fontsize, frameon = False)
    
'''
savefig:
bbox_inches is always tight
'''
def savefig(filename):
    plt.savefig(filename, bbox_inches = 'tight')

method = 'lanczos'
encoder_results_dir = os.path.join('runs', 'r2_results_normalized_test',  method, 'encoder')
import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
print(datetime_string)
mobilenet_flag = True
use_pretrained= False
if mobilenet_flag:
    mobilenet_results_dir = os.path.join('runs', 'clean',  method, 'mobilenet', datetime_string)
    message = 'tanh removed'
    os.makedirs(mobilenet_results_dir, exist_ok=True)
    with open(os.path.join(mobilenet_results_dir, 'information.txt'), 'w') as f:
        f.write(message)
if use_pretrained:
    encoder_results_dir = '/home/oogoke/runs/clean/lanczos/encoder/2022_11_28_01_23_44'#os.path.join('runs', 'clean',  method, 'encoder', datetime_string)
    print("Using pretrained, " , encoder_results_dir)
else:
    encoder_results_dir = os.path.join('runs', 'clean',  method, 'encoder', datetime_string)
diffusion_results_dir = os.path.join('runs', 'clean',  method, 'diffusion', datetime_string)
os.makedirs(diffusion_results_dir, exist_ok=True)
train_dataset = TemperatureXZDataset(method = method, split = 'train', root_folder = '/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data')
test_dataset = TemperatureXZDataset(method = method, split = 'test', root_folder = '/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data')
dev_dataset = TemperatureXZDataset(method = method, split = 'dev', root_folder = '/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data')

batch_size= 16

std =  train_dataset.std_hr
mean = train_dataset.mean_hr
std_lrs = train_dataset.std_lr
mean_lrs = train_dataset.mean_lr
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
if mobilenet_flag:
    train_mobilenet(results_folder= mobilenet_results_dir, train_dataset = train_dataset, dev_dataset = dev_dataset, test_dataset = test_dataset)
if not use_pretrained:
    pretrain_encoder(encoder_results_dir, train_dataset = train_dataset, dev_dataset = dev_dataset, test_dataset = test_dataset)
train_diffusion(results_folder = diffusion_results_dir, lr_encoder_folder  = encoder_results_dir, train_dataset = train_dataset, dev_dataset = dev_dataset, test_dataset = test_dataset)

