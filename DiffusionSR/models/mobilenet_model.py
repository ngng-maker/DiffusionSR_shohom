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

def to_img(x):
#     breakpoint()
#     x = 0.5 * (x + 1)
#     x = x.clamp(0, 1)
    x = x.view(x.size(0), -1, x.size(2),x.size(2))
#     breakpoint()
    return x
from torch.utils.data import DataLoader
import numpy as np
import torchsummary
import matplotlib.pyplot as plt
# from lr_encoder_model import rrdbnet_x4
device = torch.device("cuda")


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetv2_SISR(nn.Module):
    def __init__(self, n_channels = 1):
        super(MobileNetv2_SISR, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.conv1 = nn.Sequential(
        nn.Conv2d(n_channels, 32, 3, stride=1, padding=1),  # b, 32, 64, 64
        nn.BatchNorm2d(32),
        nn.LeakyReLU()
        )
        self.bottlenecks1 = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 16, 1, 1),
            InvertedResidual(16, 24, 1, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
        )

        self.bottlenecks2 = nn.Sequential(
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 2, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 256, 1, 6),
            InvertedResidual(256, 256, 1, 6),
        )
        self.deconv1 = nn.Sequential(
        nn.ConvTranspose2d(64, n_channels, 3, stride=1, padding=1),  # b, 1, 64, 64
        #nn.Tanh()
        )
        self.pix_shuffle = nn.PixelShuffle(2)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = x.float()
#         breakpoint()
        out = self.conv1(x)
        
        out = self.bottlenecks1(out)
        int1 = out
        out = self.dropout(out)
        out = self.bottlenecks2(out)
        out = self.pix_shuffle(out)
        out = torch.add(out, int1)

        out = self.deconv1(out)
        
        out1 = out#self.tanh(out)+0.0001*out
        return out1

        #return out
        
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

def PSNR(op, t, batch_size): 
    mse = torch.sum((t - op) ** 2) 
    #print(mse.size())
    mse /= (batch_size*64*64)

    max_pixel = 1.
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    #print(psnr.size())
    return psnr 

def KE(img, op, t): 
    ke_recon = torch.sum(op ** 2)/op.size()[0] 
    ke_dns = torch.sum(t ** 2)/t.size()[0] 
    ke_les = torch.sum(img ** 2)/img.size()[0] 
 
    return ke_les, ke_recon, ke_dns

def Avg_KE(img, op, t): 
    # pdb.set_trace()

    op = np.squeeze(op)
    img = np.squeeze(img)
    t = np.squeeze(t)
    # pdb.set_trace()

    ke_recon = torch.mean(torch.abs(op - torch.mean(op)))
    ke_dns = torch.mean(torch.abs(t - torch.mean(t)))
    ke_les = torch.mean(torch.abs(img - torch.mean(img)))
 
    return ke_les, ke_recon, ke_dns



def SSIM(op, t, batch_size):
    ssim = 0 
    #print(op.size(), t.size())
    for i in range(op.size()[0]):
        tar = to_img(t[i])
        out = to_img(op[i])
        # print(out[0,0].size())
        (score, diff) = ssim(out[0,0].cpu().detach().numpy(), tar[0,0].cpu().numpy(), full=True)
        ssim+=score/batch_size
    
        #print("SSIM: {}".format(score))
    return ssim

def train_epoch(model, data_loader, criterion, optimizer, epoch = 0, image_dir = ''):
    dataset = data_loader.dataset
    model.train()
    factor = data_loader.dataset.factor
    running_loss = 0
    avg_psnr = 0
    avg_psnr_les = 0
    psnr = None
    start_time = time.time()
    print('Train Loop')
    temp_idx = data_loader.dataset.field_names.index('temperature')
    for batch_num, (res, hr, true_lr, upscaled_lr) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):
        if len(upscaled_lr.shape)< 4:
            img = upscaled_lr.view(upscaled_lr.shape[0], 1, upscaled_lr.shape[1], upscaled_lr.shape[2])
            target = hr.view(hr.shape[0], 1, hr.shape[1], hr.shape[2])
        else:
            img = upscaled_lr
            target = hr
        img = img.to(device)
        target = target.to(device)
        
        output = model(img)
        new_out = output#*(8000-293) + 293
        loss = criterion(output.float(), target.float())
#         breakpoint()
        psnr = PSNR(output, target, img.shape[0])
        psnr_les = PSNR(img, target, img.shape[0])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()/len(data_loader)
        avg_psnr += psnr.item()/len(data_loader)
        avg_psnr_les += psnr_les.item()/len(data_loader)
        
        if batch_num % 500 == 0:
            plot_directory = image_dir + '/epoch_{}/good_train_examples/'.format(epoch)
            os.makedirs(plot_directory,exist_ok=True)


            plt.imshow(dataset.unscale_data(true_lr[0].detach().cpu().numpy(), input_type = 'lr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
            plt.title(f'Train, Low Resolution Downscaled GT, Epoch = {epoch}')
            plt.colorbar()

            plt.savefig(os.path.join(plot_directory, f'trainlr-downsampled-{epoch}.png'))
            plt.clf()

            plt.imshow(dataset.unscale_data(new_out.detach().cpu().detach().numpy()[0], input_type = 'hr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
            plt.title(f'Train, Generated HR Sample, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(plot_directory, f'traingenerated-sample-{epoch}.png'))
            plt.clf()

            plt.imshow(dataset.unscale_data(hr.cpu().detach().numpy()[0], input_type = 'hr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
            plt.title(f'Train, Generated HR Sample, Epoch = {epoch}')
            plt.colorbar()
            plt.savefig(os.path.join(plot_directory, f'trainhr-sample-{epoch}.png'))
            plt.clf()
            plt.close('all')


            plt.plot(data_loader.dataset.img_shape*(20/factor)*(np.arange(target.shape[2])/target.shape[2]),img[0][temp_idx][int(10*factor),:].cpu().numpy(),label = 'Low-Resolution Input')
            plt.plot(data_loader.dataset.img_shape*(20/factor)*(np.arange(target.shape[2])/target.shape[2]),target[0][temp_idx][int(10*factor),:].cpu().numpy(), label = 'High-Resolution Input')
            plt.plot(data_loader.dataset.img_shape*(20/factor)*(np.arange(target.shape[2])/target.shape[2]),new_out[0][temp_idx][int(10*factor),:].cpu().detach().numpy(), label = 'Output')
            plt.title(r'Cross section: x = 200 $\mu m$, L1:{}'.format(loss.item()), fontsize = 10)
#             legend()
            plt.legend()
            plt.ylabel(r'T [K]')
            plt.xlabel(r'z [$\mu m$]')
            plt.savefig(os.path.join(plot_directory, 'line_plot{}.png'.format(epoch)), bbox_inches='tight')
            # print('saved')
            # print(image_dir + '/epoch_{}/good_train_examples/line_plot{}.png'.format(epoch,batch_num))
            plt.clf()
            
            
    if not psnr:
        return -1
    #print(' ')
    print('Train_Loss:{:.6f}, PSNR_DNS:{:.4f}, PSNR_LES:{:.4f}'.format(running_loss, psnr, psnr_les))
    torch.cuda.empty_cache()
    end_time = time.time()
    del img
    del target
    del loss
    print("Train Time: {:.2f} s".format(end_time-start_time))

    return running_loss


def dev_epoch(model, data_loader, criterion, epoch = 0, image_dir = ''):
    with torch.no_grad():
        model.eval()

        running_loss = 0
        avg_psnr = 0
        avg_psnr_les = 0
        avg_ssim_dns = 0
        avg_ssim_les = 0

        start_time = time.time()
        print('Dev Loop')
        print(' ')
        dataset = data_loader.dataset
        temp_idx = data_loader.dataset.field_names.index('temperature')

        factor = dataset.factor
        loss = None
        for batch_num, (res, hr, true_lr, upscaled_lr) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):
            if len(upscaled_lr.shape) == 3:
                img = upscaled_lr.view(upscaled_lr.shape[0],1,upscaled_lr.shape[1], upscaled_lr.shape[2])#(img)# - 293)/(8000 - 293)
                target = hr.view(hr.shape[0],1, hr.shape[1], hr.shape[2])#(target)# - 293)/(8000 - 293)
            else:
                img = upscaled_lr
                target = hr
            img = img.to(device)
            target = target.to(device)

            output = model(img)
            loss = criterion(output, target)
            psnr = PSNR(output, target,img.shape[0])
            psnr_les = PSNR(img, target,img.shape[0])
            ssim_dns =0# SSIM(output, target, img.shape[0])
            ssim_les =0# SSIM(img, target, img.shape[0])

            running_loss += loss.item()/len(data_loader)
            avg_psnr += psnr.item()/len(data_loader)
            avg_psnr_les += psnr_les.item()/len(data_loader)
            # avg_ssim_dns += ssim_dns.item()/len(data_loader)
            # avg_ssim_les += ssim_les.item()/len(data_loader)
            if batch_num % 500 == 0:
                plot_directory = image_dir + '/epoch_{}/good_train_examples/'.format(epoch)
                os.makedirs(plot_directory,exist_ok=True)


                plt.imshow(dataset.unscale_data(true_lr[0].detach().cpu().numpy(), input_type = 'lr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Test, Low Resolution Downscaled GT, Epoch = {epoch}')
                plt.colorbar()

                plt.savefig(os.path.join(plot_directory, f'testlr-downsampled-{epoch}.png'))
                plt.clf()

                plt.imshow(dataset.unscale_data(output.detach().cpu().detach().numpy()[0], input_type = 'hr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Test, Generated HR Sample, Epoch = {epoch}')
                plt.colorbar()
                plt.savefig(os.path.join(plot_directory, f'testgenerated-sample-{epoch}.png'))
                plt.clf()

                plt.imshow(dataset.unscale_data(hr.cpu().detach().numpy()[0], input_type = 'hr')[temp_idx].T, origin = 'lower', cmap = 'jet', vmin = 293, vmax = 5000)
                plt.title(f'Test, Generated HR Sample, Epoch = {epoch}')
                plt.colorbar()
                plt.savefig(os.path.join(plot_directory, f'testhr-sample-{epoch}.png'))
                plt.clf()
                plt.close('all')


                plt.plot(data_loader.dataset.img_shape*(20/factor)*(np.arange(target.shape[2])/target.shape[2]),img[0][temp_idx][int(10*factor),:].cpu().numpy(),label = 'Low-Resolution Input')
                plt.plot(data_loader.dataset.img_shape*(20/factor)*(np.arange(target.shape[2])/target.shape[2]),target[0][temp_idx][int(10*factor),:].cpu().numpy(), label = 'High-Resolution Input')
                plt.plot(data_loader.dataset.img_shape*(20/factor)*(np.arange(target.shape[2])/target.shape[2]),output[0][temp_idx][int(10*factor),:].cpu().detach().numpy(), label = 'Output')
                plt.title(r'Cross section: x = 200 $\mu m$, L1:{}'.format(loss.item()), fontsize = 10)
    #             legend()
                plt.legend()
                plt.ylabel(r'T [K]')
                plt.xlabel(r'z [$\mu m$]')
                plt.savefig(os.path.join(plot_directory, 'line_plot{}.png'.format(epoch)), bbox_inches='tight')
                # print('saved')
                # print(image_dir + '/epoch_{}/good_train_examples/line_plot{}.png'.format(epoch,batch_num))
                plt.clf()
        torch.cuda.empty_cache()
        end_time = time.time()
        if not loss:
            return running_loss, avg_psnr_les, avg_psnr, avg_ssim_les, avg_ssim_dns
        del img
        del target
        del loss
        del psnr
        del psnr_les

        print("Dev Time: {:.2f} s".format(end_time-start_time))

        return running_loss, avg_psnr_les, avg_psnr, avg_ssim_les, avg_ssim_dns

def transform(self, image, mask):
    # Resize
    resize = transforms.Resize(size=(520, 520))
    image = resize(image)
    mask = resize(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(512, 512))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    return image, mask


def train_predictions(model, train_loader, img_dir, epoch):
    with torch.no_grad():
        model.eval()
        avg_psnr = 0
        avg_psnr_les = 0
        avg_dns_ke = 0
        avg_les_ke = 0
        avg_recon_ke = 0
        avg_ssim_les = 0
        avg_ssim_dns = 0

        P = []

        for batch_idx, (img, target) in enumerate(train_loader):  
    
            # if batch_idx > 3:
            #     break 

            img = img.to(device)
            target = target.to(device)

            out = model((img- 293)/(8000 - 293))
            psnr = PSNR(out, target, img.shape[0])
            psnr_les = PSNR(img, target, img.shape[0])
            les_ke, recon_ke, dns_ke = KE(img, out, target)
            ssim_dns = SSIM(out, target, img.shape[0])
            ssim_les = SSIM(img, target, img.shape[0])

            pic = to_img(out.cpu()) #Only the first image of the batch.
#             breakpoint()
#             plt.imsave('test_filtered_3c/image_tr_{}.png'.format(batch_idx),pic[0][0].T, origin = 'lower')
# #             save_image(pic, 'test_filtered_3c/image_tr_{}.png'.format(batch_idx))
#             t = to_img(target) #Only the first image of the batch.
#             plt.imsave('test_filtered_3c/image_tr_{}.png'.format(batch_idx),pic[0][0].T, origin = 'lower')
# #             save_image(t, 'test_filtered_3c/image_tr_{}_t.png'.format(batch_idx))
#             i = to_img(img) #Only the first image of the batch.
#             plt.imsave('test_filtered_3c/image_tr_{}.png'.format(batch_idx),pic[0][0].T, origin = 'lower')
#             save_image(i, 'test_filtered_3c/image_tr_{}_og.png'.format(batch_idx))


            if batch_idx % 100 == 0:
                os.makedirs(img_dir + '/epoch_{}'.format(epoch)+ '/', exist_ok = True)
                pic = to_img(out.cpu()*(8000 - 293) + 293) #Only the first image of the batch
    #             plt.imsave(out.cpu().numpy().T, cmap = 'jet')
                np.save(img_dir + '/epoch_{}'.format(epoch)+ '/image_tr_{}.npy'.format(batch_idx),np.array(pic.cpu().numpy()))
                plt.imsave(img_dir + '/epoch_{}'.format(epoch)+ '/image_tr_{}.png'.format(batch_idx), out.cpu().numpy().T, cmap = 'jet', vmin = 300, vmax = 8000, origin = 'lower')
    #             save_image(pic, 'test_1c/image_{}.png'.format(batch_idx))
                t = to_img(target) #Only the first image of the batch.
                np.save(img_dir + '/epoch_{}'.format(epoch)+ '/image_tr_{}_t.npy'.format(batch_idx),np.array(target.cpu().numpy()))
        
                plt.imsave(img_dir + '/epoch_{}'.format(epoch)+ '/image_tr_{}_t.png'.format(batch_idx), t.cpu().numpy().T, cmap = 'jet', vmin = 300, vmax = 8000, origin = 'lower')
    #             save_image(t, 'test_1c/image_{}_t.png'.format(batch_idx))
                i = to_img(img) #Only the first image of the batch.
                np.save(img_dir + '/epoch_{}'.format(epoch)+ '/image_tr_{}_og.npy'.format(batch_idx),np.array(i.cpu().numpy()))
        
                plt.imsave(img_dir + '/epoch_{}'.format(epoch)+ '/image_tr_{}_og.png'.format(batch_idx), i.cpu().numpy().T, cmap = 'jet', vmin = 300, vmax = 8000, origin = 'lower')

            del img
            del target

            avg_psnr += psnr.item()/len(train_loader)
            avg_psnr_les += psnr_les.item()/len(train_loader)

            avg_dns_ke += dns_ke.item()/len(train_loader)
            avg_recon_ke += recon_ke.item()/len(train_loader)
            avg_les_ke += les_ke.item()/len(train_loader)

            avg_ssim_les += ssim_les.item()/len(train_loader)
            avg_ssim_dns += ssim_dns.item()/len(train_loader)

    return avg_psnr, avg_psnr_les, avg_les_ke, avg_recon_ke, avg_dns_ke, avg_ssim_les, avg_ssim_dns

def test_predictions(model, test_loader, epoch = 0, img_dir = ''):
    with torch.no_grad():
        model.eval()
        avg_psnr = 0
        avg_psnr_les = 0
        avg_dns_ke = 0
        avg_les_ke = 0
        avg_recon_ke = 0
        avg_ssim_les = 0
        avg_ssim_dns = 0

        P = []
        L = []
        R = []
        D = []

        L_ = []
        R_ = []
        D_ = []

        for batch_idx, (res, hr, true_lr, upscaled_lr) in enumerate(test_loader):   
#             breakpoint()
#             if img.shape[1] == 1:
#                 continue
#             if np.sum(img.shape[2:] != 160):
#                 continue
            if len(res.shape) == 3:
                img= upscaled_lr.view(upscaled_lr.shape[0],1, upscaled_lr.shape[1], upscaled_lr.shape[2])#(img - 293)/(8000 - 293)
                target = hr.view(hr.shape[0],1, hr.shape[1], hr.shape[2])#(target - 293)/(8000 -293)
            else:
                img =upscaled_lr
                target = hr
            img = img.to(device)
            target = target.to(device)
            
            out = model(img)# - 293)/(8000 - 293)
            psnr = PSNR(out, target, img.shape[0])
            psnr_les = PSNR(img, target, img.shape[0])

            les_ke, recon_ke, dns_ke = KE(img, out, target)
            les_ake, recon_ake, dns_ake = Avg_KE(img, out, target)

            ssim_dns =0# SSIM(out, target, img.shape[0])
            ssim_les = 0#SSIM(img, target, img.shape[0])
            L.append(les_ke.cpu().numpy())
            R.append(recon_ke.cpu().numpy())
            D.append(dns_ke.cpu().numpy())

            L_.append(les_ake.cpu().numpy())
            R_.append(recon_ake.cpu().numpy())
            D_.append(dns_ake.cpu().numpy())
#             breakpoint()

#             if batch_idx % 100 == 0:
#                 os.makedirs(img_dir + '/epoch_{}'.format(epoch)+ '/', exist_ok = True)
#                 pic = to_img(out.cpu()*(8000 - 293) + 293) #Only the first image of the batch
#     #             plt.imsave(out.cpu().numpy().T, cmap = 'jet')
               
#                 np.save(img_dir + '/epoch_{}'.format(epoch)+ '/image_{}.npy'.format(batch_idx),np.array(pic.cpu().numpy()))
#                 plt.imsave(img_dir + '/epoch_{}'.format(epoch)+ '/image_{}.png'.format(batch_idx), out.cpu().numpy().T, cmap = 'jet', vmin = 300, vmax = 1800, origin = 'lower')
#     #             save_image(pic, 'test_1c/image_{}.png'.format(batch_idx))
#                 t = to_img(target*(8000 - 293) + 293) #Only the first image of the batch.
#                 np.save(img_dir + '/epoch_{}'.format(epoch)+ '/image_{}_t.npy'.format(batch_idx),np.array(t.cpu().numpy()))
        
#                 plt.imsave(img_dir + '/epoch_{}'.format(epoch)+ '/image_{}_t.png'.format(batch_idx), t.cpu().numpy().T, cmap = 'jet', vmin = 300, vmax = 1800, origin = 'lower')
#     #             save_image(t, 'test_1c/image_{}_t.png'.format(batch_idx))
#                 i = to_img(img*(8000 - 293) + 293) #Only the first image of the batch.
#                 np.save(img_dir + '/epoch_{}'.format(epoch)+ '/image_{}_og.npy'.format(batch_idx),np.array(i.cpu().numpy()))
        
#                 plt.imsave(img_dir + '/epoch_{}'.format(epoch)+ '/image_{}_og.png'.format(batch_idx), i.cpu().numpy().T, cmap = 'jet', vmin = 300, vmax = 1800, origin = 'lower')
#     #             save_image(i, 'test_1c/image_{}_og.png'.format(batch_idx))
#     #             breakpoint()
#             del img
#             del target

            avg_psnr += psnr.item()/len(test_loader)
            avg_psnr_les += psnr_les.item()/len(test_loader)

            avg_dns_ke += dns_ke.item()/len(test_loader)
            avg_recon_ke += recon_ke.item()/len(test_loader)
            avg_les_ke += les_ke.item()/len(test_loader)

            avg_ssim_les += 0# ssim_les.item()/len(test_loader)
            avg_ssim_dns += 0#ssim_dns.item()/len(test_loader)

    # plot_MAE(L, R, D)
    # plot_Avg_MAE(L_, R_, D_)
    return avg_psnr, avg_psnr_les, avg_les_ke, avg_recon_ke, avg_dns_ke, avg_ssim_les, avg_ssim_dns

def plot_MAE(L, R, D):

    MAE = np.abs((np.array(R) - np.array(D))/np.array(D))
    print(MAE.shape)
    # print(MAE)
    avg = np.mean(MAE)
    plt.xticks([])
    plt.title('Reconstruction')
    plt.ylabel('MAE')
    plt.ylim(0,1.2)
    plt.plot(MAE, 'ko', fillstyle = 'none')
    plt.axhline(avg, color = 'r')
    plt.savefig('MAE.eps')
    plt.close()

    MAE_L = np.abs((np.array(L) - np.array(D))/np.array(D))
    print(MAE.shape)
    # print(MAE)
    avg_l = np.mean(MAE_L)
    plt.xticks([])
    plt.title('LES')
    plt.ylabel('MAE')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(MAE_L, 'ko', fillstyle = 'none')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_l, color = 'r')
    plt.savefig('MAE_L.eps')  # plt.show()
    plt.close()

#combined plot for turbulent velocity
def plot_Avg_MAE(L, R, D):

    # MAE = np.abs((np.array(R) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
    avg = np.mean(np.abs(R))
    # plt.xticks([])
    # plt.title('Reconstruction')
    plt.ylabel('Average Turbulent Velocity')
    plt.xlabel('Samples')
    # plt.ylim(0,1.2)
    plt.plot(np.abs(R),  marker = '^', c = 'dodgerblue', label='Recon',ls=' ', ms='3.5')


    # MAE_L = np.abs((np.array(L) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
    avg_l = np.mean(np.abs(D))
    # plt.xticks([])
    # plt.title('DNS')
    # plt.ylabel('Average Turbulent Velocity')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(np.abs(D), marker = '+', c = 'darkorange', label='DNS',ls=' ', ms='4')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_l, color = 'red', ls='-.',label='Avg. DNS', lw='1.5')
    plt.axhline(avg, color = 'k', ls='-.', label='Avg. Recon', lw='1.5')
    plt.legend(loc='best')
    plt.savefig('combined.eps')  # plt.show()
    plt.close()


def plot_Avg_MAE(L, R, D):

    # MAE = np.abs((np.array(R) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
    avg = np.mean(np.abs(R))
    # plt.xticks([])
    plt.title('Reconstruction')
    plt.ylabel('Average Turbulent Velocity')
    # plt.ylim(0,1.2)
    plt.plot(np.abs(R), 'ko', fillstyle = 'none')
    plt.axhline(avg, color = 'r')
    plt.savefig('AM.eps')
    plt.close()

    # MAE_L = np.abs((np.array(L) - np.array(D))/np.array(D))
    # print(MAE.shape)
    # print(MAE)
    avg_l = np.mean(np.abs(D))
    # plt.xticks([])
    plt.title('DNS')
    plt.ylabel('Average Turbulent Velocity')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(np.abs(D), 'ko', fillstyle = 'none')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_l, color = 'r')
    plt.savefig('AM_L.eps')  # plt.show()
    plt.close()

    error = np.abs(np.array(D) - np.array(R))
    avg_e = np.mean(np.abs(error))
    # plt.xticks([])
    plt.title('Error')
    plt.ylim(0, 0.35)
    plt.ylabel('Turbulent Velocity Error')
    # plt.plot(MAE, 'yo', fillstyle = 'none')
    plt.plot(error, 'ko', fillstyle = 'none')
    # plt.axhline(avg, color = 'r')
    plt.axhline(avg_e, color = 'r')
    plt.savefig('error.eps')  # plt.show()
    plt.close()

