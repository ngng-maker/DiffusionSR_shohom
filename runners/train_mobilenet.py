from random import random
import numpy as np
import math
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
import time
import os
import cv2
import pdb
from PIL import Image
import matplotlib.pyplot as plt 
from skimage.metrics import structural_similarity as ssim
from models.mobilenet_model import MobileNetv2_SISR, test_predictions, train_epoch, dev_epoch
from torch.utils.data import DataLoader
import numpy as np
import torchsummary
import matplotlib.pyplot as plt



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)

def train_mobilenet(results_folder, train_dataset, dev_dataset, test_dataset, num_epochs, batch_size, learning_rate):
    num_epochs = 20
    batch_size = 3
    learning_rate = 1e-4


    model = MobileNetv2_SISR(train_dataset.n_steps*train_dataset.num_fields)
    model.apply(init_weights)
    device = torch.device("cuda")
    model.eval()
    model.to(device)     

    device = 'cuda'



    std =  train_dataset.std_hr
    mean = train_dataset.mean_hr
    std_lrs = train_dataset.std_lr
    mean_lrs = train_dataset.mean_lr
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # train_dataset = MyDataset(image_paths, target_paths, transform=img_transform)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=8)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    print(train_dataset.__len__())

    # dev_dataset = MyDataset(image_paths_dev, target_paths_dev, transform=test_transform)
    dev_loader_args = dict(batch_size=batch_size, shuffle=False, num_workers=8)
    dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)
    print(dev_dataset.__len__())

    # test_dataset = MyDataset(image_paths_test, target_paths_test, transform=test_transform)/
    test_loader_args = dict(batch_size=1, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_dataset, **test_loader_args)
    print(test_dataset.__len__())
    cuda = torch.cuda.get_device_name(0)
    print(cuda)

    print(os.getcwd())
    np.random.random()
    num_epochs = 150
    Train_Loss = []
    Dev_Loss = []
    Dev_Acc = []
    train_psnrs = []
    dev_psnrs  = []
    train_ssims = []
    dns_ssim = []
    les_ssim = []
    psnrs = []
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay = 1e-2)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.95, patience=3, threshold=5e-4, eps=1e-6)
    model_dir = results_folder
    img_dir = os.path.join(results_folder, 'images')
    os.makedirs(model_dir,exist_ok = True)
    os.makedirs(img_dir,exist_ok = True)



    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            test_predictions(model, test_loader, epoch  = epoch, img_dir = img_dir)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch = epoch, image_dir=img_dir)
        dev_loss, psnr_les, psnr, ssim_les, ssim_dns = dev_epoch(model, dev_loader, criterion, epoch = epoch, image_dir=img_dir)
        Train_Loss.append(train_loss)
        print(Train_Loss)
        Dev_Loss.append(dev_loss)
        dev_psnrs.append(psnr)
        dns_ssim.append(ssim_dns)
        les_ssim.append(ssim_les)
        psnrs.append(psnr_les)
        scheduler.step(dev_loss)
        np.save(model_dir + '/mse_train_progress_MSE{}'.format(epoch), Train_Loss)
        np.save(model_dir + '/mse_train_progress_ssimdns{}'.format(epoch), dns_ssim)
        np.save(model_dir + '/mse_dev_progress_MSE{}'.format(epoch), Dev_Loss)
        np.save(model_dir + '/mse_train_progress_ssimles{}'.format(epoch),les_ssim)
        np.save(model_dir + '/mse_train_progress_psnr{}'.format(epoch), dev_psnrs)
        
        
        print(' ')
        print('epoch [{}/{}], Train_Loss:{:.6f}, Dev_Loss:{:.6f}'.format(epoch+1, num_epochs, train_loss, dev_loss))
        print('PSNR_DNS:{:.4f}, PSNR_LES:{:.4f}, SSIM_LES:{:.4f}, SSIM_DNS:{:.4f}'.format(psnr, psnr_les, ssim_les, ssim_dns))

        torch.save(model.state_dict(), model_dir+'/SISR_mv2f.pth')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, model_dir+'/SISR_mv2f.pth')
        scheduler.step(train_loss)
        print(optimizer)
        print('='*100)
