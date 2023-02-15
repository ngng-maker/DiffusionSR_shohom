
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
        print(f"Building dataset, downscale method: {downscale_method}...")

        # Use default HR paths, always the same
        # Use LR path as specified, may change
        # downscale_method: scaling downscale_method
        self.downscale_method = downscale_method
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
            if index - step > 0:
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
                if index - step > 0:
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
                 n_steps=1,  field_names = None):
        # Use default HR paths, always the same
        # Use LR path as specified, may change
        # downscale_method: scaling downscale_method
        self.downscale_method = downscale_method
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
        if 'ss316l_v2_all_laser_velocity_xz_cross_section_data' in self.root_folder:
            all_field_names = {'vx':0, 'temperature':1, 'pressure':2, 'vy':3, 'vz':4, 'liqlabel':5}
        else:
            all_field_names = {'temperature':0}
        if field_names is None:
            print("Using all {} fields".format(len(all_field_names.keys())))
            self.field_names = list(all_field_names.keys())
            self.field_idxs = np.arange(len(all_field_names)).astype('int')
        else:
            print("Using specific fields, ", field_names)
            self.field_names = field_names
        self.field_idxs = [all_field_names[key] for key in self.field_names]
        self.num_fields = len(self.field_names)

        self.field_idxs_steps = []
        for i in range(self.n_steps):
            for j in self.field_idxs:
                self.field_idxs_steps.append(j*(i+1))
        print("Processing dataset with {} fields".format(len(self.field_names)))
        self.lr_path = os.path.join(
            self.root_folder, split, 'LR', downscale_method, '1x')+'/'
        img_names = [f.split(self.lr_path)[-1] for f in glob.glob(os.path.join(self.lr_path, '**/*npy'), recursive=True)]#os.listdir(self.lr_path)
        targ_names =  [f.split(self.hr_path)[-1] for f in glob.glob(os.path.join(self.hr_path, '**/*npy'), recursive=True)] #os.listdir(self.hr_path)


        intersection_list = list(set(img_names) & set(targ_names))
        self.lr_paths = np.sort(np.array(
            [self.lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
        self.hr_paths = np.sort(np.array(
            [self.hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
       
        test_hr_shape = list(np.load(self.hr_paths[0]).shape)
        test_lr_shape = list(np.load(self.lr_paths[0]).shape)
        self.factor = int(test_hr_shape[0]/test_lr_shape[0]) 
        self.upscaled_lr_path = os.path.join(
            self.root_folder, split, 'LR', downscale_method, '{}x').format(self.factor)+'/'    
        self.upscaled_lr_paths = np.sort(np.array(
            [self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
        self.img_shape = int(test_hr_shape[0])
        n_channels = len(test_hr_shape)
        # if n_channels < 3:
        #     self.num_fields = 1
        # else:
        #     self.num_fields =test_hr_shape[-1]
        test_hr_shape.insert(0, len(self.lr_paths))
        test_lr_shape.insert(0, len(self.lr_paths))
        if self.normalize == 'standardize':
            if split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', downscale_method, 'flag')):
                all_residuals = np.zeros(tuple(test_hr_shape))#np.zeros((len(self.lr_paths), test_hr_shape[0], ))
                all_hr =  np.zeros(tuple(test_hr_shape)) #np.zeros((len(self.lr_paths), 80, 80))

                all_lr =  np.zeros(tuple(test_lr_shape))#np.zeros((len(self.lr_paths), 20, 20))
                all_upscaled_lr = np.zeros(tuple(test_hr_shape))#np.zeros((len(self.lr_paths), 80, 80))

                # print("Normalizing data...")
                for idx, (hr_path, lr_path, true_lr_path) in tqdm(enumerate(zip(self.hr_paths, self.upscaled_lr_paths, self.lr_paths)), total=len(self.lr_paths)):

                    hr_img = np.load(hr_path)
                    lr_img = np.load(lr_path)
                    true_lr_img = np.load(true_lr_path)
                    hr_img[hr_img > self.threshold_T] = self.threshold_T
                    lr_img[lr_img > self.threshold_T] = self.threshold_T
                    true_lr_img[true_lr_img >
                                self.threshold_T] = self.threshold_T
                    all_residuals[idx] = (hr_img - lr_img)
                    all_lr[idx] = true_lr_img
                    all_hr[idx] = hr_img
                    all_upscaled_lr[idx] = lr_img

                all_lr = np.moveaxis(all_lr, -1, 1)
                all_hr = np.moveaxis(all_hr, -1, 1)
                all_residuals = np.moveaxis(all_residuals, -1, 1)
                all_upscaled_lr = np.moveaxis(all_upscaled_lr, -1, 1)

                self.std_lr = np.std(all_lr, axis=0)
                self.mean_lr = np.mean(all_lr, axis=0)
                self.mean_upscaled_lr = np.mean(all_upscaled_lr, axis=0)
                self.std_upscaled_lr = np.std(all_upscaled_lr, axis=0)
                self.std_resid = np.std(all_residuals, axis=0)
                self.mean_resid = np.mean(all_residuals, axis=0)
                self.std_hr = np.std(all_hr, axis=0)
                self.mean_hr = np.mean(all_hr, axis=0)
                # breakpoint()
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
        # breakpoint()
    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, index):
        return self.testgetitem(index)

    def testgetitem(self, index):
        
        single_hr = np.load(self.hr_paths[index], allow_pickle=True)
        try:
            single_upscaled_lr = np.load(
                self.upscaled_lr_paths[index], allow_pickle=True)
        except:
            # breakpoint()
            print(f"Upscaled low resolution data not found for this dataset ({self.downscale_method}, {self.root_folder}), defaulting to HR instead")
            single_upscaled_lr =  np.load(
                self.hr_paths[index], allow_pickle=True)
        single_true_lr = np.load(self.lr_paths[index], allow_pickle=True)
        n_channels = len(single_hr.shape)
        if n_channels < 3:
            num_fields = 1
        else:
            num_fields = single_hr.shape[-1]

        hr = np.empty(( single_hr.shape[0], single_hr.shape[1], self.n_steps*num_fields))
        upscaled_lr = np.empty(( single_upscaled_lr.shape[0], single_upscaled_lr.shape[1], self.n_steps*num_fields))
        true_lr = np.empty(( single_true_lr.shape[0], single_true_lr.shape[1], self.n_steps*num_fields))
        if len(single_hr.shape) < 3:
            single_hr = single_hr[:,:, None]
            single_upscaled_lr = single_upscaled_lr[:,:, None]
            single_true_lr = single_true_lr[:,:, None]
        hr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_hr
        upscaled_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_upscaled_lr
        true_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_true_lr
        
        for step in reversed(range(2, self.n_steps + 1)):
            
                
            if index - step > 0:
                if len(np.load(self.hr_paths[index-  step], allow_pickle=True).shape) < 3:
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.hr_paths[index-  step], allow_pickle=True)[:, :, None]
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.lr_paths[index - step], allow_pickle=True)[:, :, None]
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)[:, :, None]
                else:
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.hr_paths[index-  step], allow_pickle=True)
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.lr_paths[index - step], allow_pickle=True)
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)
            else:
                hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_hr)*293
                true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_true_lr)*293
                upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.ones_like(single_upscaled_lr)*293
        hr = np.moveaxis(hr, -1, 0)


        true_lr = np.moveaxis(true_lr, -1, 0)
        upscaled_lr = np.moveaxis(upscaled_lr, -1, 0)


        residual = hr - upscaled_lr
        hr[hr > self.threshold_T] = self.threshold_T
        residual[residual > self.threshold_T] = self.threshold_T
        true_lr[true_lr > self.threshold_T] = self.threshold_T
        upscaled_lr[upscaled_lr > self.threshold_T] = self.threshold_T
        power = int(self.lr_paths[index].split(
            'power')[-1].split('velocity')[0])
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
        if self.return_info:
            info = torch.tensor([power, velocity, time])
            return res[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])], info
        return res[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])]
    def unscale_data(self, array, input_type):
        if self.normalize == 'standardize':
            # breakpoint()
            if self.num_fields > 1:
                if torch.is_tensor(array):

                    array = array.cpu().detach().numpy()
            if len(self.mean_lr.shape) < 3:   
                if input_type == 'hr':
                    unscaledarray = array*self.std_hr[None, :, :][tuple([self.field_idxs_steps])] + self.mean_hr[None, :, :][tuple([self.field_idxs_steps])]
                if input_type == 'lr':
                    unscaledarray = array*self.std_lr[None, :, :][tuple([self.field_idxs_steps])] + self.mean_lr[None, :, :][tuple([self.field_idxs_steps])]
                if input_type == 'upscaled_lr':
                    unscaledarray = array*self.std_upscaled_lr[None, :, :][tuple([self.field_idxs_steps])] + self.mean_upscaled_lr[None, :, :][tuple([self.field_idxs_steps])]
                if input_type == 'residual':
                    unscaledarray = array*self.std_resid[None, :, :][tuple([self.field_idxs_steps])] + self.mean_resid[None, :, :][tuple([self.field_idxs_steps])]
                return unscaledarray
            else:
                if input_type == 'hr':
                    unscaledarray = array*self.std_hr[tuple([self.field_idxs_steps])] + self.mean_hr[tuple([self.field_idxs_steps])]
                if input_type == 'lr':
                    unscaledarray = array*self.std_lr[tuple([self.field_idxs_steps])] + self.mean_lr[tuple([self.field_idxs_steps])]
                if input_type == 'upscaled_lr':
                    unscaledarray = array*self.std_upscaled_lr[tuple([self.field_idxs_steps])] + self.mean_upscaled_lr[tuple([self.field_idxs_steps])]
                if input_type == 'residual':
                    unscaledarray = array*self.std_resid[tuple([self.field_idxs_steps])] + self.mean_resid[tuple([self.field_idxs_steps])]
                return unscaledarray
        elif self.normalize == 'rescaling':
            unscaledarray = (0.5 + array/2)*(self.t_max - self.t_min) + self.t_min
            return unscaledarray

    def rescale_data(self, array, input_type):
       
        if self.normalize == 'standardize':
            if input_type == 'hr':
                mean = self.mean_hr[tuple([self.field_idxs_steps])]
                std = self.std_hr[tuple([self.field_idxs_steps])]
            if input_type == 'lr':
                mean = self.mean_lr[tuple([self.field_idxs_steps])]
                std = self.std_lr[tuple([self.field_idxs_steps])]
            if input_type == 'upscaled_lr':
                mean = self.mean_upscaled_lr[tuple([self.field_idxs_steps])]
                std = self.std_upscaled_lr[tuple([self.field_idxs_steps])]
            if input_type == 'residual':
                mean = self.mean_resid[tuple([self.field_idxs_steps])]
                std = self.std_resid[tuple([self.field_idxs_steps])]
            scaled_array = (array - mean)/std
            return scaled_array
        elif self.normalize == 'rescaling':
            scaled_array = (array - self.t_min)/(self.t_max - self.t_min)


def main():
    root_folder = '/home/oogoke/DiffusionSR/datasets/ss316l_v2_all_laser_velocity_xz_cross_section_data'
    print("Testing multiple field values")
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                      normalize='standardize',
                                      n_steps=1,
                                      root_folder=root_folder,
                                      split=split,
                                      field_names=None)
        print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    dataset.testgetitem(1)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, drop_last=True)
    res, hr, true_lr, upscaled_lr = iter(dataloader).next()
    lr_unscale= dataset.unscale_data(true_lr, input_type = 'lr')
    print(np.mean(lr_unscale, axis = (0,2, 3)))
    print(res.shape, hr.shape, true_lr.shape)
    print("Testing single field values")
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                      normalize='standardize',
                                      n_steps=1,
                                      root_folder=root_folder,
                                      split=split,
                                      field_names=['temperature'])
        print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    dataset.testgetitem(1)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, drop_last=True)
    res, hr, true_lr, upscaled_lr = iter(dataloader).next()
    lr_unscale= dataset.unscale_data(true_lr, input_type = 'lr')
    print(np.mean(lr_unscale.numpy(), axis = (0,2, 3)))
    print(res.shape, hr.shape, true_lr.shape)

    print("Testing 5 micron data, single field, organized")
    root_folder = '/home/oogoke/DiffusionSR/datasets/simulation_basis_laser_velocity_xz_cross_section_data'
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                      normalize='standardize',
                                      n_steps=1,
                                      root_folder=root_folder,
                                      split=split,
                                      field_names=['temperature'])
        print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    dataset.testgetitem(1)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, drop_last=True)
    res, hr, true_lr, upscaled_lr = iter(dataloader).next()
    lr_unscale= dataset.unscale_data(true_lr, input_type = 'lr')
    print(np.mean(lr_unscale.numpy(), axis = (0,2, 3)))
    print(res.shape, hr.shape, true_lr.shape)



    print("Testing 5 micron data, single field, unorganized")
    root_folder = '/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data'
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                      normalize='standardize',
                                      n_steps=3,
                                      root_folder=root_folder,
                                      split=split,
                                      field_names=['temperature'])
        print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    dataset.testgetitem(1)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, drop_last=True)
    res, hr, true_lr, upscaled_lr = iter(dataloader).next()
    lr_unscale= dataset.unscale_data(true_lr, input_type = 'lr')
    print(np.mean(lr_unscale.numpy(), axis = (0,2, 3)))
    print(res.shape, hr.shape, true_lr.shape)
if __name__ == '__main__':
    main()
