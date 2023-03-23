from tqdm import tqdm
import time


import math
# from datasets import load_dataset
import os
import pdb
import time
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import torchvision
import torchvision.transforms.functional as TF
# from datasets.dataset import TemperatureXZDataset
from models.lr_encoder_model import rrdbnet_encoder as rrdbnet_upscaled
from PIL import Image
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm

# from torchvision.models.feature_extraction import create_feature_extractor
def npl1loss(arr1, arr2):
    return (np.mean(np.abs(arr1 - arr2)))

# os.environ['CUDA_VISIBLE_DEVICES']  = "2"
def pretrain_encoder(results_dir, train_dataset, dev_dataset, test_dataset):
    print("Now training encoder...")

    lr_enc = rrdbnet_upscaled(upscale_factor = train_dataset.factor, in_channels = train_dataset.n_steps*train_dataset.num_fields, out_channels = train_dataset.out_steps*train_dataset.num_fields)
    # breakpoint()
    # !pip install torchsummary
    

    torchsummary.summary(lr_enc.to('cuda'), input_size = (train_dataset.n_steps*train_dataset.num_fields, 20, 20))

    
    tensor = torch.tensor(np.ones((1,train_dataset.n_steps,20,20))).float()

    
    # plt.imshow(lr_enc(tensor.to('cuda'))[0,0].detach().cpu().numpy())
    # plt.colorbar()
    # plt.show()

    os.makedirs(results_dir, exist_ok = True)
    # image_size = 80
    # channels = 1
    batch_size = 128
    # std =  train_dataset.std_hr
    # mean = train_dataset.mean_hr
    # std_lrs = train_dataset.std_lr
    # mean_lrs = train_dataset.mean_lr

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = 'cuda'
    criterion = torch.nn.L1Loss()
    data_loader = dataloader
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(lr_enc.parameters(), lr=learning_rate,weight_decay = 0)
    temp_idx = train_dataset.field_names.index('temperature')
    min_test_loss = 1e10
    losses = []
    test_losses = []
    scaled_losses = []
    test_scaled_losses = []
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [75, 150, 225], gamma = 0.5)
    for epoch in range(250):
        lr_enc.train()

        running_loss = 0
        avg_psnr = 0
        avg_psnr_les = 0
        psnr = None
        running_scaled_loss = 0
        start_time = time.time()
        print('Train Loop')
        for batch_num, (res, hr, lr, upscaled_lr) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):
            # breakpoint()
            # print('hr', hr.shape)
            flip = False
            if np.random.uniform() < 0.2:
                hr = torch.flip(hr, dims = [2])
                lr = torch.flip(lr, dims = [2])
                flip = True
            if len(lr.shape)  == 3:
                img = (lr.view(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(device))
                target = (hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2]).to(device))
            else:
                img = lr.to(device)
                target = hr.to(device)
            output = lr_enc((img.float()))
            new_out = output
            # breakpoint()
            if data_loader.dataset.out_steps == 1 and data_loader.dataset.num_fields == 1:
                loss = criterion(output.float(), (target[:,-1:].float()))
                # if data_loader.dataset.normalize == 'rescaling':
                #     scaled_loss = criterion(dataloader.dataset.unscale_data(output, input_type = 'hr', maintain_torch = True),dataloader.dataset.unscale_data(target[:,-1:], input_type = 'hr'), maintain_torch = True ).item()
                # elif data_loader.dataset.normalize == 'standardize':
                scaled_loss = npl1loss(dataloader.dataset.unscale_data(output, input_type = 'hr'),dataloader.dataset.unscale_data(target[:,-1:], input_type = 'hr') )

                    # scaled_loss = criterion(dataloader.dataset.unscale_data(output, input_type = 'hr'),dataloader.dataset.unscale_data(target[:,-1:], input_type = 'hr') ).item()

                # breakpoint()
            else:
                loss = criterion(output.float(), (target.float()))
                # breakpoint()
                # if data_loader.dataset.normalize == 'rescaling':
                #     scaled_loss = criterion(dataloader.dataset.unscale_data(output, input_type = 'hr', maintain_torch = True),dataloader.dataset.unscale_data(target, input_type = 'hr'), maintain_torch = True ).item()

                # elif data_loader.dataset.normalize == 'standardize':

                scaled_loss = npl1loss(dataloader.dataset.unscale_data(output, input_type = 'hr'),dataloader.dataset.unscale_data(target, input_type = 'hr') )
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()/len(data_loader)
            running_scaled_loss += scaled_loss/len(data_loader)

            if batch_num % 400 == 0:
                if flip:
                    hr = torch.flip(hr, dims = [2])
                    lr = torch.flip(lr, dims = [2])
                    new_out = torch.flip(new_out, dims = [2])

        #         plt.clf()
        #         plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),train_dataset.unscale_data(target.cpu(), input_type = 'hr')[0][0][40,:], label = 'High-Resolution Target')
        #         plt.plot(80*5*(np.arange(img.shape[2])/img.shape[2]),(train_dataset.unscale_data(img.cpu(), input_type = 'lr'))[0][0][10,:], label = 'Low-Resolution Input')
        #         plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),(train_dataset.unscale_data(new_out.cpu().detach(), input_type = 'hr'))[0][0][40,:], label = 'Output')
        #         plt.title(r'Cross section: x = 200 $\mu m$, L1:{}'.format(loss.item()), fontsize = 10)
        #         #             legend()
        #         plt.legend()
        #         plt.ylabel(r'T [K]')
        #         plt.xlabel(r'z [$\mu m$]')
        #         plt.savefig(results_dir + '/image{}_{}.png'.format(epoch, batch_num))
        #         plt.show()
        #         plt.clf()
        #         plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),(target.cpu().numpy())[0][0][40,:], label = 'High-Resolution Target')
        #         plt.plot(80*5*(np.arange(img.shape[2])/img.shape[2]),(img.cpu().numpy())[0][0][10,:], label = 'Low-Resolution Input')
        #         print(new_out.min(), "MINIMUM")
        #         plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),(new_out.detach().cpu().numpy())[0][0][40,:], label = 'Output')
        #         plt.title(r'Cross section: x = 200 $\mu m$, L1:{}'.format(loss.item()), fontsize = 10)
        # #             legend()
        #         plt.legend()
        #         plt.ylabel(r'T [K]')
        #         plt.xlabel(r'z [$\mu m$]')
        #         plt.savefig(results_dir + '/nonormalimage{}_{}.png'.format(epoch, batch_num))
        #         plt.show()
        #         plt.close('all')

               


                plt.clf()
                # breakpoint()
                plt.imshow((train_dataset.unscale_data(hr[0].cpu(), input_type = 'hr')[temp_idx]).T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                # breakpoint()
                # plt.imshow((hr[0].detach().cpu().numpy()*std + mean).T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'High Resolution GT, Epoch = {epoch}')
                plt.colorbar()
                plt.savefig(os.path.join(results_dir, f'hr-sample-{epoch}.png'))
                plt.clf()


                plt.imshow(train_dataset.unscale_data(lr[0].cpu().numpy(), input_type = 'lr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Low Resolution Downscaled GT, Epoch = {epoch}')
                plt.colorbar()

                plt.savefig(os.path.join(results_dir, f'lr-downsampled-{epoch}.png'))
                plt.clf()

                plt.imshow(train_dataset.unscale_data(new_out.cpu().detach().numpy()[0], input_type = 'hr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Generated HR Sample, Epoch = {epoch}')
                plt.colorbar()
                plt.savefig(os.path.join(results_dir, f'generated-sample-{epoch}.png'))
                plt.clf()
                plt.close('all')
                torch.save({
                'epoch': epoch,
                'model_state_dict': lr_enc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, results_dir + '/model_saved.pth')
        print("epoch: {}".format(epoch))
        losses.append(running_loss)
        scaled_losses.append(running_scaled_loss)
        np.savetxt(results_dir + '/train_loss.txt', losses)
        np.savetxt(results_dir + '/scaled_train_loss.txt', scaled_losses)

        print('Train_Loss:{:.6f}'.format(running_loss))
        torch.cuda.empty_cache()
        end_time = time.time()
        del img
        del target
        del loss

        
        print("Train Time: {:.2f} s".format(end_time-start_time))

        print(running_loss)
        lr_enc.eval()

        testrunning_loss = 0
        testrunning_scaled_loss = 0
        avg_psnr = 0
        avg_psnr_les = 0
        psnr = None
        start_time = time.time()
        print('Test Loop')
        
        all_losses = []
        all_scaled_losses = []
        for batch_num, (res, hr, lr, upscaled_lr) in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), ascii=True):
            # print(hr.shape)
            if len(lr.shape) == 3:
                img = (lr.view(lr.shape[0], dev_dataloader.dataset.n_steps, lr.shape[1], lr.shape[2]).to(device))
                target = (hr.view(hr.shape[0], dev_dataloader.dataset.n_steps, hr.shape[1], hr.shape[2]).to(device))
            img = (lr.to(device))
            target = (hr.to(device))
            output = lr_enc((img.float()))

            new_out = output
            
            # loss = criterion(output.float(), (target.float()))

            if data_loader.dataset.out_steps == 1 and data_loader.dataset.num_fields == 1:

                loss = criterion(output.float(), (target[:,-1:].float()))
                # if data_loader.dataset.normalize == 'rescaling':
                #     testscaled_loss = criterion(dataloader.dataset.unscale_data(output, input_type = 'hr'),dataloader.dataset.unscale_data(target[:,-1:], input_type = 'hr') ).item()
                # elif data_loader.dataset.normalize == 'standardize':
                testscaled_loss = npl1loss(dataloader.dataset.unscale_data(output, input_type = 'hr'),dataloader.dataset.unscale_data(target[:,-1:], input_type = 'hr') )

                # breakpoint()
            else:
                loss = criterion(output.float(), (target.float()))
                # if data_loader.dataset.normalize == 'rescaling':
                #     testscaled_loss = criterion(dataloader.dataset.unscale_data(output, input_type = 'hr'),dataloader.dataset.unscale_data(target, input_type = 'hr') ).item()

                # elif data_loader.dataset.normalize == 'standardize':

                testscaled_loss = npl1loss(dataloader.dataset.unscale_data(output, input_type = 'hr'),dataloader.dataset.unscale_data(target, input_type = 'hr') )
            # breakpoint()
            # print(testrunning_loss)
            # if loss.item() > 0.5:
            #     breakpoint()
            testrunning_loss += loss.item()/len(dev_dataloader)
            testrunning_scaled_loss += testscaled_loss/len(dev_dataloader)
            all_losses.append(testrunning_loss)
            all_scaled_losses.append(testrunning_scaled_loss)
            if batch_num % 400 == 0:
                # print(target.shape)
                # plt.clf()
                # plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),(dev_dataset.unscale_data(target.cpu(), input_type = 'hr'))[0][0][40,:], label = 'High-Resolution Target')
                # plt.plot(80*5*(np.arange(img.shape[2])/img.shape[2]),(dev_dataset.unscale_data(img.cpu(), input_type = 'lr'))[0][0][10,:], label = 'Low-Resolution Input')

                # plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),(dev_dataset.unscale_data(target.cpu(), input_type = 'hr'))[0][0][40,:], label = 'Output')
                # plt.title(r'Cross section: x = 200 $\mu m$, L1:{}'.format(loss.item()), fontsize = 10)
                # plt.legend()
                # plt.ylabel(r'T [K]')
                # plt.xlabel(r'z [$\mu m$]')
                # plt.savefig(results_dir + '/testimage{}_{}.png'.format(epoch, batch_num))
                # plt.show()
                # plt.clf()
                # plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),(target.cpu().numpy())[0][0][40,:], label = 'Test, High-Resolution Target')
                # plt.plot(80*5*(np.arange(img.shape[2])/img.shape[2]),(img.cpu().numpy())[0][0][10,:], label = 'Test, Low-Resolution Input')

                # plt.plot(80*5*(np.arange(target.shape[2])/target.shape[2]),(new_out.detach().cpu().numpy())[0][0][40,:], label = 'Output')
                # plt.title(r'Cross section: x = 200 $\mu m$, L1:{}'.format(loss.item()), fontsize = 10)
                # plt.legend()
                # plt.ylabel(r'T [K]')
                # plt.xlabel(r'z [$\mu m$]')
                # plt.savefig(results_dir + '/testnonormalimage{}_{}.png'.format(epoch, batch_num))
                # plt.show()
                # plt.close('all')
                plt.imshow((train_dataset.unscale_data(hr[0].cpu(), input_type = 'hr'))[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                # plt.imshow((hr[0].detach().cpu().numpy()*std + mean).T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Test, High Resolution GT, Epoch = {epoch}')
                plt.colorbar()
                plt.savefig(os.path.join(results_dir, f'testhr-sample-{epoch}.png'))
                plt.clf()


                plt.imshow(train_dataset.unscale_data(lr[0].cpu().numpy(), input_type = 'lr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Test, Low Resolution Downscaled GT, Epoch = {epoch}')
                plt.colorbar()

                plt.savefig(os.path.join(results_dir, f'testlr-downsampled-{epoch}.png'))
                plt.clf()

                plt.imshow(train_dataset.unscale_data(new_out.cpu().detach().numpy()[0], input_type = 'hr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Test, Generated HR Sample, Epoch = {epoch}')
                plt.colorbar()
                plt.savefig(os.path.join(results_dir, f'testgenerated-sample-{epoch}.png'))
                plt.clf()
                plt.close('all')
        print('Test_Loss:{:.6f}'.format(testrunning_loss))
        scheduler.step()
        # breakpoint()
        test_losses.append(testrunning_loss)
        test_scaled_losses.append(testrunning_scaled_loss)
        np.savetxt(results_dir + '/test_loss.txt', test_losses)
        np.savetxt(results_dir + '/scaled_test_loss.txt', test_scaled_losses)
        if testrunning_loss < min_test_loss:
            min_test_loss = testrunning_loss
            torch.save({
                            'epoch': epoch,
                            'model_state_dict': lr_enc.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, results_dir + '/bestmodel_saved.pth')
        # breakpoint()
