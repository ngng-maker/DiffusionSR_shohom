
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
import glob

class TemperatureXZDataset(Dataset):
    def __init__(self,
                 downscale_method,
                 root_folder='/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data',
                 split='train',
                 normalize='standardize',
                 return_info=False,
                 n_steps=1):
        # Use default HR paths, always the same
        # Use LR path as specified, may change
        # downscale_method: scaling downscale_method
        self.root_folder = root_folder
        self.split = split
        self.normalize = normalize
        self.hr_path = os.path.join(self.root_folder, split, 'HR') + '/'
        self.threshold_T = 8000
        self.powers = []
        self.velocities = []
        self.n_steps = n_steps
        self.times = []
        self.return_info = return_info
        self.factor = 4
        if downscale_method == 'all':
            type = 'area'
            self.lr_path = os.path.join(
                self.root_folder, split, 'LR', type, '1x')+'/'
            self.upscaled_lr_path = os.path.join(
                self.root_folder, split, 'LR', type, '4x')+'/'
            img_names = os.listdir(self.lr_path)
            targ_names = os.listdir(self.hr_path)
            intersection_list = list(set(img_names) & set(targ_names))
            self.lr_paths = [
                self.lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')]
            self.hr_paths = [
                self.hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')]
            self.upscaled_lr_paths = [
                self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')]
            for type in ['cubic', 'lanczos', 'linear', 'area']:
                self.lr_path = os.path.join(
                    self.root_folder, split, 'LR', type, '1x')+'/'
                self.upscaled_lr_path = os.path.join(
                    self.root_folder, split, 'LR', type, '4x')+'/'
                self.lr_paths.extend(
                    [self.lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')])
                self.hr_paths.extend(
                    [self.hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')])
                self.upscaled_lr_paths.extend(
                    [self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')])
        else:
            breakpoint()
            self.lr_path = os.path.join(
                self.root_folder, split, 'LR', downscale_method, '1x')+'/'

            img_names = os.listdir(self.lr_path)
            targ_names = os.listdir(self.hr_path)
            intersection_list = list(set(img_names) & set(targ_names))
            self.lr_path = os.path.join(
                self.root_folder, split, 'LR', downscale_method, '1x')+'/'
            self.upscaled_lr_path = os.path.join(
                self.root_folder, split, 'LR', downscale_method, '4x')+'/'
            self.lr_paths = np.array(
                [self.lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object')
            self.hr_paths = np.array(
                [self.hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object')
            self.upscaled_lr_paths = np.array(
                [self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object')
        # print("len data: {}".format(len(self.lr_paths)))
        # breakpoint()
        if self.normalize == 'standardize':
            if split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', downscale_method, 'flag')):
                all_residuals = np.zeros((len(self.lr_paths), 80, 80))
                all_hr = np.zeros((len(self.lr_paths), 80, 80))

                all_lr = np.zeros((len(self.lr_paths), 20, 20))
                all_upscaled_lr = np.zeros((len(self.lr_paths), 80, 80))
                print("Normalizing data...")
                for idx, (hr_path, lr_path, true_lr_path) in tqdm(enumerate(zip(self.hr_paths, self.upscaled_lr_paths, self.lr_paths)), total=len(self.lr_paths)):

                    hr_img = np.load(hr_path)
                    lr_img = np.load(lr_path)
                    true_lr_img = np.load(true_lr_path)
                    hr_img[hr_img > self.threshold_T] = self.threshold_T
                    lr_img[lr_img > self.threshold_T] = self.threshold_T
                    true_lr_img[true_lr_img >
                                self.threshold_T] = self.threshold_T
                    # np.save(os.path.join(residual_path, hr_path.split('/')[-1]), hr_img - lr_img)
                    all_residuals[idx] = (hr_img - lr_img)
                    all_lr[idx] = true_lr_img
                    all_hr[idx] = hr_img
                    all_upscaled_lr[idx] = lr_img

                self.std_lr = np.std(all_lr, axis=0)
                self.mean_lr = np.mean(all_lr, axis=0)
                self.mean_upscaled_lr = np.mean(all_upscaled_lr, axis=0)
                self.std_upscaled_lr = np.std(all_upscaled_lr, axis=0)
                self.std_resid = np.std(all_residuals, axis=0)
                self.mean_resid = np.mean(all_residuals, axis=0)
                self.std_hr = np.std(all_hr, axis=0)
                self.mean_hr = np.mean(all_hr, axis=0)
                self.stats_path = os.path.join(
                    self.root_folder, 'statistics', downscale_method)
                os.makedirs(os.path.join(self.root_folder,
                            'statistics', downscale_method), exist_ok=True)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_lr'), self.std_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_lr'), self.mean_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_resid'), self.mean_resid)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_resid'), self.std_resid)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_hr'), self.mean_hr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_hr'), self.std_hr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_upscaled_lr'), self.mean_upscaled_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_upscaled_lr'), self.std_upscaled_lr)
                np.savetxt(os.path.join(self.root_folder,
                           'statistics', downscale_method, 'flag'), np.array([0]))
            elif not split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', downscale_method, 'flag')):
                raise AttributeError('Initialize training set first')
            else:
                self.stats_path = os.path.join(
                    self.root_folder, 'statistics', downscale_method)
                self.std_lr = np.load(os.path.join(
                    self.stats_path, 'std_lr.npy'))
                self.mean_lr = np.load(os.path.join(
                    self.stats_path, 'mean_lr.npy'))
                self.std_resid = np.load(os.path.join(
                    self.stats_path, 'std_resid.npy'))
                self.mean_resid = np.load(os.path.join(
                    self.stats_path, 'mean_resid.npy'))
                self.std_hr = np.load(os.path.join(
                    self.stats_path, 'std_hr.npy'))
                self.mean_hr = np.load(os.path.join(
                    self.stats_path, 'mean_hr.npy'))
                self.std_upscaled_lr = np.load(os.path.join(
                    self.stats_path, 'std_upscaled_lr.npy'))
                self.mean_upscaled_lr = np.load(os.path.join(
                    self.stats_path, 'mean_upscaled_lr.npy'))

            self.std_lr[self.std_lr == 0] = 1
            self.std_hr[self.std_hr == 0] = 1
            self.std_upscaled_lr[self.std_upscaled_lr == 0] = 1
            self.std_resid[self.std_resid == 0] = 1
        elif self.normalize == 'rescaling':
            self.t_max = 5000
            self.t_min = 293

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, index):
        # ip = np.load(self.image_paths[index], allow_pickle = True)

            single_hr = np.load(self.hr_paths[index], allow_pickle=True)
            single_upscaled_lr = np.load(
                self.upscaled_lr_paths[index], allow_pickle=True)
            single_true_lr = np.load(self.lr_paths[index], allow_pickle=True)

            hr = np.empty((self.n_steps, single_hr.shape[0], single_hr.shape[1]))
            upscaled_lr = np.empty((self.n_steps, single_upscaled_lr.shape[0], single_upscaled_lr.shape[1]))
            true_lr = np.empty((self.n_steps, single_true_lr.shape[0], single_true_lr.shape[1]))
            hr[self.n_steps-1] = single_hr
            upscaled_lr[self.n_steps-1] = single_upscaled_lr
            true_lr[self.n_steps-1] = single_true_lr
            for step in reversed(range(2, self.n_steps + 1)):
                # print("HERE")
                if index - step > 0:
                    # print(self.n_steps-step, 'steps')
                    hr[self.n_steps - step] = np.load(self.hr_paths[index-  step], allow_pickle=True)
                    true_lr[self.n_steps - step] = np.load(self.lr_paths[index - step], allow_pickle=True)
                    upscaled_lr[self.n_steps - step] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)

                else:
                    hr[self.n_steps - step] = np.ones_like(single_hr)*293
                    true_lr[self.n_steps - step] = np.ones_like(single_true_lr)*293
                    upscaled_lr[self.n_steps - step] = np.ones_like(single_upscaled_lr)*293

            residual = hr - upscaled_lr
            hr[hr > self.threshold_T] = self.threshold_T
            residual[residual > self.threshold_T] = self.threshold_T
            true_lr[true_lr > self.threshold_T] = self.threshold_T
            upscaled_lr[upscaled_lr > self.threshold_T] = self.threshold_T
            power = int(self.lr_paths[index].split(
                'power')[1].split('velocity')[0])
            velocity = int(self.lr_paths[index].split('velocity')[-1].split('_')[0])
            time = 0.5 * \
                float(self.lr_paths[index].split('_1')[-1].split('.npy')[0])/100
            if self.normalize == 'standardize':
                res = (residual - self.mean_resid)/self.std_resid
                true_lr = (true_lr - self.mean_lr)/self.std_lr
                hr = (hr - self.mean_hr)/self.std_hr
                upscaled_lr = (upscaled_lr - self.mean_upscaled_lr) / \
                    self.std_upscaled_lr
            elif self.normalize == 'rescaling':
                hr = ((hr - self.t_min)/(self.t_max - self.t_min) - 0.5)*2
                true_lr = ((true_lr - self.t_min)/(self.t_max - self.t_min) - 0.5)*2
                upscaled_lr = ((upscaled_lr - self.t_min) / \
                    (self.t_max - self.t_min) -0.5) *2
                res = ((residual - self.t_min)/(self.t_max - self.t_min) - 0.5) * 2
            if self.return_info:
                info = torch.tensor([power, velocity, time])
                return res, hr, true_lr, upscaled_lr, info
            return res, hr, true_lr, upscaled_lr
    def testgetitem(self, index):

            single_hr = np.load(self.hr_paths[index], allow_pickle=True)
            single_upscaled_lr = np.load(
                self.upscaled_lr_paths[index], allow_pickle=True)
            single_true_lr = np.load(self.lr_paths[index], allow_pickle=True)

            hr = np.empty((self.n_steps, single_hr.shape[0], single_hr.shape[1]))
            upscaled_lr = np.empty((self.n_steps, single_upscaled_lr.shape[0], single_upscaled_lr.shape[1]))
            true_lr = np.empty((self.n_steps, single_true_lr.shape[0], single_true_lr.shape[1]))
            hr[self.n_steps-1] = single_hr
            upscaled_lr[self.n_steps-1] = single_upscaled_lr
            true_lr[self.n_steps-1] = single_true_lr
            for step in reversed(range(2, self.n_steps + 1)):
                # print("HERE")
                if index - step > 0:
                    # print(self.n_steps-step, 'steps')
                    hr[self.n_steps - step] = np.load(self.hr_paths[index-  step], allow_pickle=True)
                    true_lr[self.n_steps - step] = np.load(self.lr_paths[index - step], allow_pickle=True)
                    upscaled_lr[self.n_steps - step] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)

                else:
                    hr[self.n_steps - step] = np.ones_like(single_hr)*293
                    true_lr[self.n_steps - step] = np.ones_like(single_true_lr)*293
                    upscaled_lr[self.n_steps - step] = np.ones_like(single_upscaled_lr)*293

            residual = hr - upscaled_lr
            hr[hr > self.threshold_T] = self.threshold_T
            residual[residual > self.threshold_T] = self.threshold_T
            true_lr[true_lr > self.threshold_T] = self.threshold_T
            upscaled_lr[upscaled_lr > self.threshold_T] = self.threshold_T
            power = int(self.lr_paths[0].split(
                'power')[1].split('velocity')[0])
            velocity = int(self.lr_paths[0].split('velocity')[1].split('_')[0])
            time = 0.5 * \
                float(self.lr_paths[0].split('_1')[1].split('.npy')[0])/100
            if self.normalize == 'standardize':
                res = (residual - self.mean_resid)/self.std_resid
                true_lr = (true_lr - self.mean_lr)/self.std_lr
                hr = (hr - self.mean_hr)/self.std_hr
                upscaled_lr = (upscaled_lr - self.mean_upscaled_lr) / \
                    self.std_upscaled_lr
            elif self.normalize == 'rescaling':
                hr = (hr - self.t_min)/(self.t_max - self.t_min)
                true_lr = (true_lr - self.t_min)/(self.t_max - self.t_min)
                upscaled_lr = (upscaled_lr - self.t_min) / \
                    (self.t_max - self.t_min)
                res = (residual - self.t_min)/(self.t_max - self.t_min)
            if self.return_info:
                info = torch.tensor([power, velocity, time])
                return res, hr, true_lr, upscaled_lr, info
            return res, hr, true_lr, upscaled_lr
    def unscale_data(self, array, input_type):
        if self.normalize == 'standardize':
            if input_type == 'hr':
                unscaledarray = array*self.std_hr + self.mean_hr
            if input_type == 'lr':
                unscaledarray = array*self.std_lr + self.mean_lr
            if input_type == 'upscaled_lr':
                unscaledarray = array*self.std_upscaled_lr + self.mean_upscaled_lr
            if input_type == 'residual':
                unscaledarray = array*self.std_resid + self.mean_resid
            return unscaledarray
        elif self.normalize == 'rescaling':
            unscaledarray = (0.5 + array/2)*(self.t_max - self.t_min) + self.t_min
            return unscaledarray

    def rescale_data(self, array, input_type):
        if self.normalize == 'standardize':
            if input_type == 'hr':
                mean = self.mean_hr
                std = self.std_hr
            if input_type == 'lr':
                mean = self.mean_lr
                std = self.std_lr
            if input_type == 'upscaled_lr':
                mean = self.mean_upscaled_lr
                std = self.std_upscaled_lr
            if input_type == 'residual':
                mean = self.mean_resid
                std = self.std_resid
            scaled_array = (array - mean)/std
            return scaled_array
        elif self.normalize == 'rescaling':
            scaled_array = (array - self.t_min)/(self.t_max - self.t_min)



class SimulationXZDataset(Dataset):
    def __init__(self,
                 downscale_method,
                 root_folder='/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data',
                 split='train',
                 normalize='standardize',
                 return_info=False,
                 n_steps=1):
        # Use default HR paths, always the same
        # Use LR path as specified, may change
        # downscale_method: scaling downscale_method
        self.root_folder = root_folder
        self.split = split
        self.normalize = normalize
        self.hr_path = os.path.join(self.root_folder, split, 'HR') + '/'
        self.threshold_T = 8000
        self.powers = []
        self.velocities = []
        self.n_steps = n_steps
        self.times = []
        self.return_info = return_info
        self.lr_path = os.path.join(
            self.root_folder, split, 'LR', downscale_method, '1x')+'/'
        img_names = [f.split(self.lr_path)[-1] for f in glob.glob(os.path.join(self.lr_path, '**/*npy'), recursive=True)]#os.listdir(self.lr_path)
        targ_names =  [f.split(self.hr_path)[-1] for f in glob.glob(os.path.join(self.hr_path, '**/*npy'), recursive=True)] #os.listdir(self.hr_path)
        # breakpoint()

        intersection_list = list(set(img_names) & set(targ_names))
        # self.lr_path = os.path.join(
        #     self.root_folder, split, 'LR', downscale_method, '1x')+'/'
        # self.upscaled_lr_path = os.path.join(
        #     self.root_folder, split, 'LR', downscale_method, '4x')+'/'
        self.lr_paths = np.sort(np.array(
            [self.lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
        self.hr_paths = np.sort(np.array(
            [self.hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
        self.upscaled_lr_paths = self.hr_paths
        test_hr_shape = list(np.load(self.hr_paths[0]).shape)
        test_lr_shape = list(np.load(self.lr_paths[0]).shape)
        self.factor = int(test_hr_shape[0]/test_lr_shape[0]) 
        self.img_shape = int(test_hr_shape[0])
        n_channels = len(test_hr_shape)
        if n_channels < 3:
            self.num_fields = 1
        else:
            self.num_fields =test_hr_shape[-1]
        test_hr_shape.insert(0, len(self.lr_paths))
        test_lr_shape.insert(0, len(self.lr_paths))
            # breakpoint()
        # np.array(
        #     [self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object')
        # breakpoint()
        if self.normalize == 'standardize':
            if split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', downscale_method, 'flag')):
                all_residuals = np.zeros(tuple(test_hr_shape))#np.zeros((len(self.lr_paths), test_hr_shape[0], ))
                all_hr =  np.zeros(tuple(test_hr_shape)) #np.zeros((len(self.lr_paths), 80, 80))

                all_lr =  np.zeros(tuple(test_lr_shape))#np.zeros((len(self.lr_paths), 20, 20))
                all_upscaled_lr = np.zeros(tuple(test_hr_shape))#np.zeros((len(self.lr_paths), 80, 80))
                # breakpoint()
                # print("Normalizing data...")
                for idx, (hr_path, lr_path, true_lr_path) in tqdm(enumerate(zip(self.hr_paths, self.upscaled_lr_paths, self.lr_paths)), total=len(self.lr_paths)):

                    hr_img = np.load(hr_path)
                    lr_img = np.load(lr_path)
                    true_lr_img = np.load(true_lr_path)
                    hr_img[hr_img > self.threshold_T] = self.threshold_T
                    lr_img[lr_img > self.threshold_T] = self.threshold_T
                    true_lr_img[true_lr_img >
                                self.threshold_T] = self.threshold_T
                    # np.save(os.path.join(residual_path, hr_path.split('/')[-1]), hr_img - lr_img)
                    all_residuals[idx] = (hr_img - lr_img)
                    all_lr[idx] = true_lr_img
                    all_hr[idx] = hr_img
                    all_upscaled_lr[idx] = lr_img

                self.std_lr = np.std(all_lr, axis=0)
                self.mean_lr = np.mean(all_lr, axis=0)
                self.mean_upscaled_lr = np.mean(all_upscaled_lr, axis=0)
                self.std_upscaled_lr = np.std(all_upscaled_lr, axis=0)
                self.std_resid = np.std(all_residuals, axis=0)
                self.mean_resid = np.mean(all_residuals, axis=0)
                self.std_hr = np.std(all_hr, axis=0)
                self.mean_hr = np.mean(all_hr, axis=0)
                self.stats_path = os.path.join(
                    self.root_folder, 'statistics', downscale_method)
                os.makedirs(os.path.join(self.root_folder,
                            'statistics', downscale_method), exist_ok=True)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_lr'), self.std_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_lr'), self.mean_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_resid'), self.mean_resid)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_resid'), self.std_resid)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_hr'), self.mean_hr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_hr'), self.std_hr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'mean_upscaled_lr'), self.mean_upscaled_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        downscale_method, 'std_upscaled_lr'), self.std_upscaled_lr)
                np.savetxt(os.path.join(self.root_folder,
                           'statistics', downscale_method, 'flag'), np.array([0]))
            elif not split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', downscale_method, 'flag')):
                raise AttributeError('Initialize training set first')
            else:
                self.stats_path = os.path.join(
                    self.root_folder, 'statistics', downscale_method)
                self.std_lr = np.load(os.path.join(
                    self.stats_path, 'std_lr.npy'))
                self.mean_lr = np.load(os.path.join(
                    self.stats_path, 'mean_lr.npy'))
                self.std_resid = np.load(os.path.join(
                    self.stats_path, 'std_resid.npy'))
                self.mean_resid = np.load(os.path.join(
                    self.stats_path, 'mean_resid.npy'))
                self.std_hr = np.load(os.path.join(
                    self.stats_path, 'std_hr.npy'))
                self.mean_hr = np.load(os.path.join(
                    self.stats_path, 'mean_hr.npy'))
                self.std_upscaled_lr = np.load(os.path.join(
                    self.stats_path, 'std_upscaled_lr.npy'))
                self.mean_upscaled_lr = np.load(os.path.join(
                    self.stats_path, 'mean_upscaled_lr.npy'))

            self.std_lr[self.std_lr == 0] = 1
            self.std_hr[self.std_hr == 0] = 1
            self.std_upscaled_lr[self.std_upscaled_lr == 0] = 1
            self.std_resid[self.std_resid == 0] = 1
        elif self.normalize == 'rescaling':
            self.t_max = 5000
            self.t_min = 293

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, index):
        # ip = np.load(self.image_paths[index], allow_pickle = True)
            single_hr = np.load(self.hr_paths[index], allow_pickle=True)
            single_upscaled_lr = np.load(
                self.upscaled_lr_paths[index], allow_pickle=True)
            single_true_lr = np.load(self.lr_paths[index], allow_pickle=True)
            num_fields = self.num_fields
            hr = np.empty(( self.n_steps*num_fields,single_hr.shape[0], single_hr.shape[1],))
            upscaled_lr = np.empty(( self.n_steps*num_fields, single_upscaled_lr.shape[0], single_upscaled_lr.shape[1],))
            true_lr = np.empty((  self.n_steps*num_fields, single_true_lr.shape[0], single_true_lr.shape[1]))
            if num_fields > 1:
                hr = np.empty(( single_hr.shape[0], single_hr.shape[1], self.n_steps*num_fields))
                upscaled_lr = np.empty(( single_upscaled_lr.shape[0], single_upscaled_lr.shape[1], self.n_steps*num_fields))
                true_lr = np.empty(( single_true_lr.shape[0], single_true_lr.shape[1], self.n_steps*num_fields))
                hr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_hr
                upscaled_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_upscaled_lr
                true_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_true_lr
                
                for step in reversed(range(2, self.n_steps + 1)):
                    # print("HERE")
                    if index - step > 0:
                        # print(self.n_steps-step, 'steps')
                        hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.hr_paths[index-  step], allow_pickle=True)
                        true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.lr_paths[index - step], allow_pickle=True)
                        upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)

                    else:
                        hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_hr)*293
                        true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_true_lr)*293
                        upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_upscaled_lr)*293
            # breakpoint()
            else:
                hr[self.n_steps-1] = single_hr
                upscaled_lr[self.n_steps-1] = single_upscaled_lr
                true_lr[self.n_steps-1] = single_true_lr
                for step in reversed(range(2, self.n_steps + 1)):
                    # print("HERE")
                    if index - step > 0:
                        # print(self.n_steps-step, 'steps')
                        hr[self.n_steps - step] = np.load(self.hr_paths[index-  step], allow_pickle=True)
                        true_lr[self.n_steps - step] = np.load(self.lr_paths[index - step], allow_pickle=True)
                        upscaled_lr[self.n_steps - step] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)

                    else:
                        hr[self.n_steps - step] = np.ones_like(single_hr)*293
                        true_lr[self.n_steps - step] = np.ones_like(single_true_lr)*293
                        upscaled_lr[self.n_steps - step] = np.ones_like(single_upscaled_lr)*293

            # breakpoint()
            residual = hr - upscaled_lr
            hr[hr > self.threshold_T] = self.threshold_T
            residual[residual > self.threshold_T] = self.threshold_T
            true_lr[true_lr > self.threshold_T] = self.threshold_T
            upscaled_lr[upscaled_lr > self.threshold_T] = self.threshold_T
            power = int(self.lr_paths[index].split(
                'power')[-1].split('velocity')[0])
            # breakpoint()
            velocity = int(self.lr_paths[index].split('velocity')[-1].split('_')[0])
            time = 0.5 * \
                float(self.lr_paths[0].split('_1')[1].split('.npy')[0])/100
            if self.normalize == 'standardize':
                res = (residual - self.mean_resid)/self.std_resid
                true_lr = (true_lr - self.mean_lr)/self.std_lr
                hr = (hr - self.mean_hr)/self.std_hr
                upscaled_lr = (upscaled_lr - self.mean_upscaled_lr) / \
                    self.std_upscaled_lr
            elif self.normalize == 'rescaling':
                hr = (hr - self.t_min)/(self.t_max - self.t_min)
                true_lr = (true_lr - self.t_min)/(self.t_max - self.t_min)
                upscaled_lr = (upscaled_lr - self.t_min) / \
                    (self.t_max - self.t_min)
                res = (residual - self.t_min)/(self.t_max - self.t_min)
            if num_fields > 1:
                hr = np.moveaxis(hr, -1, 0)
                res = np.moveaxis(res, -1, 0)
    
                true_lr = np.moveaxis(true_lr, -1, 0)
                upscaled_lr = np.moveaxis(upscaled_lr, -1, 0)
            if self.return_info:
                info = torch.tensor([power, velocity, time])
                return res, hr, true_lr, upscaled_lr, info
            return res, hr, true_lr, upscaled_lr
    def testgetitem(self, index):

            single_hr = np.load(self.hr_paths[index], allow_pickle=True)
            single_upscaled_lr = np.load(
                self.upscaled_lr_paths[index], allow_pickle=True)
            single_true_lr = np.load(self.lr_paths[index], allow_pickle=True)
            n_channels = len(single_hr.shape)
            if n_channels < 3:
                num_fields = 1
            else:
                num_fields = single_hr.shape[-1]
            # breakpoint()
            hr = np.empty(( single_hr.shape[0], single_hr.shape[1], self.n_steps*num_fields))
            upscaled_lr = np.empty(( single_upscaled_lr.shape[0], single_upscaled_lr.shape[1], self.n_steps*num_fields))
            true_lr = np.empty(( single_true_lr.shape[0], single_true_lr.shape[1], self.n_steps*num_fields))
            hr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_hr
            upscaled_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_upscaled_lr
            true_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_true_lr
            
            for step in reversed(range(2, self.n_steps + 1)):
                # print("HERE")
                if index - step > 0:
                    # print(self.n_steps-step, 'steps')
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.hr_paths[index-  step], allow_pickle=True)
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.lr_paths[index - step], allow_pickle=True)
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)

                else:
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_hr)*293
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_true_lr)*293
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_upscaled_lr)*293
            # breakpoint()
           
            # breakpoint()
            residual = hr - upscaled_lr
            hr[hr > self.threshold_T] = self.threshold_T
            residual[residual > self.threshold_T] = self.threshold_T
            true_lr[true_lr > self.threshold_T] = self.threshold_T
            upscaled_lr[upscaled_lr > self.threshold_T] = self.threshold_T
            power = int(self.lr_paths[index].split(
                'power')[-1].split('velocity')[0])
            # breakpoint()
            velocity = int(self.lr_paths[index].split('velocity')[-1].split('_')[0])
            time = 0.5 * \
                float(self.lr_paths[0].split('_1')[1].split('.npy')[0])/100
            if self.normalize == 'standardize':
                res = (residual - self.mean_resid)/self.std_resid
                true_lr = (true_lr - self.mean_lr)/self.std_lr
                hr = (hr - self.mean_hr)/self.std_hr
                upscaled_lr = (upscaled_lr - self.mean_upscaled_lr) / \
                    self.std_upscaled_lr
            elif self.normalize == 'rescaling':
                hr = (hr - self.t_min)/(self.t_max - self.t_min)
                true_lr = (true_lr - self.t_min)/(self.t_max - self.t_min)
                upscaled_lr = (upscaled_lr - self.t_min) / \
                    (self.t_max - self.t_min)
                res = (residual - self.t_min)/(self.t_max - self.t_min)
            hr = np.moveaxis(hr, -1, 0)
            res = np.moveaxis(res, -1, 0)
            true_lr = np.moveaxis(true_lr, -1, 0)
            upscaled_lr = np.moveaxis(upscaled_lr, -1, 0)
            if self.return_info:
                info = torch.tensor([power, velocity, time])
                return res, hr, true_lr, upscaled_lr, info
            return res.float(), hr.float(), true_lr.float(), upscaled_lr.float()
    def unscale_data(self, array, input_type):
        if self.normalize == 'standardize':
            # breakpoint()
            if self.num_fields > 1:
                if torch.is_tensor(array):
                    array = torch.moveaxis(array, 0, -1)
                else:
                    array =  np.moveaxis(array, 0, -1)
            if input_type == 'hr':
                unscaledarray = array*self.std_hr + self.mean_hr
            if input_type == 'lr':
                unscaledarray = array*self.std_lr + self.mean_lr
            if input_type == 'upscaled_lr':
                unscaledarray = array*self.std_upscaled_lr + self.mean_upscaled_lr
            if input_type == 'residual':
                unscaledarray = array*self.std_resid + self.mean_resid
            if self.num_fields > 1:
                if torch.is_tensor(unscaledarray):
                    unscaledarray = torch.moveaxis(unscaledarray, -1,0 )
                else:
                    unscaledarray = np.moveaxis(unscaledarray, -1, 0)
            return unscaledarray
        elif self.normalize == 'rescaling':
            unscaledarray = (0.5 + array/2)*(self.t_max - self.t_min) + self.t_min
            return unscaledarray

    def rescale_data(self, array, input_type):
        if self.normalize == 'standardize':
            if input_type == 'hr':
                mean = self.mean_hr
                std = self.std_hr
            if input_type == 'lr':
                mean = self.mean_lr
                std = self.std_lr
            if input_type == 'upscaled_lr':
                mean = self.mean_upscaled_lr
                std = self.std_upscaled_lr
            if input_type == 'residual':
                mean = self.mean_resid
                std = self.std_resid
            scaled_array = (array - mean)/std
            return scaled_array
        elif self.normalize == 'rescaling':
            scaled_array = (array - self.t_min)/(self.t_max - self.t_min)


def main():
    dataset = SimulationXZDataset(downscale_method='direct', normalize='standardize', n_steps = 1, root_folder = '/media/cmu/DATA/francis/Melt_Pool_Data/superresdata_text/ss316l_all_laser_velocity_xz_cross_section_data')
    # dataset = SimulationXZDataset(downscale_method='direct', normalize='rescaling', n_steps = 1, root_folder = '/media/cmu/DATA/francis/Melt_Pool_Data/superresdata_text/simulation_basis_laser_velocity_xz_cross_section_data')
    
    # breakpoint()
    dataset.testgetitem(1)
    # breakpoint()
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, drop_last=True)
    res, hr, true_lr, upscaled_lr = iter(dataloader).next()
    print(res.shape, hr.shape, true_lr.shape)

    # dataset.unscale_data(res, input_type='residual')


if __name__ == '__main__':
    main()
