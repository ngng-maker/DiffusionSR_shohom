
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
import glob
import time
from matplotlib import pyplot as plt

def filter_data(array, thresholds, field_idxs, field_names):
    if len(array.shape) > 2:
        for i, field in zip(field_idxs, field_names):
            threshold = thresholds[field]
            if len(array.shape) > 2:

                array_channel = array[:, :, i]
            else:
                array_channel = array
            array_channel[array_channel > threshold] = threshold
            array_channel[array_channel < -threshold] = -threshold

            if len(array.shape) > 2:
                array[:,:, i] = array_channel
            else:
                array = array_channel

        # breakpoint()
    # pass
    return array

class SimulationXZDataset(Dataset):
    def __init__(self,
                 downscale_method,
                 root_folder='/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data',
                 split='train',
                 normalize='standardize',
                 return_info=False, cross_validate  = False,
                 n_steps=1,  field_names = None, out_steps = None):
        # Use default HR paths, always the same
        # Use LR path as specified, may change
        # downscale_method: scaling downscale_method
        self.downscale_method = downscale_method
        self.root_folder = root_folder
        self.split = split

        self.normalize = normalize
        print(f"Using normalize method: ... {self.normalize}")

        
        self.threshold_T = 8000
        self.powers = []
        self.velocities = []
        self.n_steps = n_steps
        if out_steps is None:
            self.out_steps = n_steps
        else:
            self.out_steps = out_steps
        self.times = []
        self.return_info = return_info
        if 'ss316l' in self.root_folder:
            # all_field_names = {'vx':0, 'temperature':1, 'pressure':2, 'vy':3, 'vz':4, 'liqlabel':5}
            # all_field_names = { 'temperature':0}
            all_field_names = { 'temperature':0, 'liqlabel':1}

        else:
            all_field_names = {'temperature':0}
        self.field_threshold = {'vx': 1000, 'temperature': self.threshold_T, 'pressure': 1e7, 'vy':1000, 'vz': 1000, 'liqlabel': 1 }
            
        # else:
            # all_field_names = {'temperature':0}
            # self.field_threshold={'temperature':self.threshold_T}
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

        
       
        # breakpoint()
        if not cross_validate:
            self.lr_path = os.path.join( self.root_folder, split, 'LR', downscale_method, '1x')+os.sep
            self.hr_path = os.path.join(self.root_folder, split, 'HR') + os.sep
            
            img_names = [f.split(self.lr_path)[-1] for f in glob.glob(os.path.join(self.lr_path, '**/*npy'), recursive=True)]
            targ_names =  [f.split(self.hr_path)[-1] for f in glob.glob(os.path.join(self.hr_path, '**/*npy'), recursive=True)]

            intersection_list = list(set(img_names) & set(targ_names))#[:300]


            

            self.lr_paths = np.sort(np.array(
                [self.lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
            self.hr_paths = np.sort(np.array(
                [self.hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
            # breakpoint()
            test_hr_shape = list(np.load(self.hr_paths[0]).shape)
            test_lr_shape = list(np.load(self.lr_paths[0]).shape)
            self.factor = int(test_hr_shape[0]/test_lr_shape[0]) 
            print("Downscale factor: ", self.factor)
            self.upscaled_lr_path = os.path.join(
                self.root_folder, split, 'LR', downscale_method, '{}x').format(self.factor)+os.sep

            self.upscaled_lr_paths = np.sort(np.array(
                [self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))


        if cross_validate:
            ### combine all splits into one list
            valid_split = 'train'

            lr_path = os.path.join( self.root_folder, valid_split, 'LR', downscale_method, '1x')+os.sep
            hr_path = os.path.join(self.root_folder, valid_split, 'HR') + os.sep
           
            img_names = [f.split(lr_path)[-1] for f in glob.glob(os.path.join(lr_path, '**/*npy'), recursive=True)]
            targ_names =  [f.split(hr_path)[-1] for f in glob.glob(os.path.join(hr_path, '**/*npy'), recursive=True)]

            intersection_list = list(set(img_names) & set(targ_names))


                
            self.lr_paths = np.sort(np.array(
                [lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
            self.hr_paths = np.sort(np.array(
                [hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
            test_hr_shape = list(np.load(self.hr_paths[0]).shape)
            test_lr_shape = list(np.load(self.lr_paths[0]).shape)
            self.factor = int(test_hr_shape[0]/test_lr_shape[0]) 
            upscaled_lr_path = os.path.join(
            self.root_folder, valid_split, 'LR', downscale_method, '{}x').format(self.factor)+os.sep

            self.upscaled_lr_paths = np.sort(np.array(
                [upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))

            valid_split = 'dev'


            
            lr_path = os.path.join( self.root_folder, valid_split, 'LR', downscale_method, '1x')+os.sep
            hr_path = os.path.join(self.root_folder, valid_split, 'HR') + os.sep
            upscaled_lr_path = os.path.join(
            self.root_folder, valid_split, 'LR', downscale_method, '{}x').format(self.factor)+os.sep
            
            img_names = [f.split(lr_path)[-1] for f in glob.glob(os.path.join(lr_path, '**/*npy'), recursive=True)]
            targ_names =  [f.split(hr_path)[-1] for f in glob.glob(os.path.join(hr_path, '**/*npy'), recursive=True)]

            intersection_list = list(set(img_names) & set(targ_names))


                

            self.upscaled_lr_paths = np.append(self.upscaled_lr_paths ,  np.sort(np.array(
                [upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object')))

            self.lr_paths = np.append(self.lr_paths, np.sort(np.array(
                [lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object')))
            self.hr_paths = np.append(self.hr_paths, np.sort(np.array(
                [hr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object')))
            indices = np.arange(self.lr_paths)
            np.random.shuffle(indices)
            train_idx = indices[:int(0.9*len(indices))]
            val_idx = indices[int(0.9*len(indices)):]
            



        if 'ss316l' in self.root_folder:
            self.baseline_hr  = np.dstack([np.ones(test_hr_shape[:-1])*0,
                        np.ones(test_hr_shape[:-1])*293, 
                        np.ones(test_hr_shape[:-1])*100000, 
                        np.ones(test_hr_shape[:-1])*0,
                        np.ones(test_hr_shape[:-1])*0,
                        np.ones(test_hr_shape[:-1])*0])
                        
            self.baseline_lr  = np.dstack([np.ones(test_lr_shape[:-1])*0,
                                    np.ones(test_lr_shape[:-1])*293, 
                                    np.ones(test_lr_shape[:-1])*100000, 
                                    np.ones(test_lr_shape[:-1])*0,
                                    np.ones(test_lr_shape[:-1])*0,
                                    np.ones(test_lr_shape[:-1])*0])
        else:
            self.baseline_hr  = np.dstack([np.ones(test_hr_shape)*293])
            self.baseline_lr = np.dstack([np.ones(test_lr_shape)*293])


        self.img_shape = int(test_hr_shape[0])
        n_channels = len(test_hr_shape)
        # if n_channels < 3:
        #     self.num_fields = 1
        # else:
        #     self.num_fields =test_hr_shape[-1]
        test_hr_shape.insert(0, len(self.lr_paths))
        test_lr_shape.insert(0, len(self.lr_paths))
       
        self.compute_statistics()
         
        self.t_max = 5000
        self.t_min = 293
        self.field_max = {'vx': 100, 'temperature': self.t_max, 'pressure': 1e8, 'vy':100, 'vz': 100, 'liqlabel': 1 }
        self.field_min = {'vx': -100, 'temperature': self.t_min, 'pressure': 1e6, 'vy':-100, 'vz': -100, 'liqlabel': 0 }
            

        # breakpoint()

    def compute_statistics(self):
        test_hr_shape = list(np.load(self.hr_paths[0]).shape)
        test_lr_shape = list(np.load(self.lr_paths[0]).shape)
        test_hr_shape.insert(0, len(self.lr_paths))
        test_lr_shape.insert(0, len(self.lr_paths))
        if self.split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', self.downscale_method, 'flag')):
                all_residuals = np.zeros(tuple(test_hr_shape))#np.zeros((len(self.lr_paths), test_hr_shape[0], ))
                all_hr =  np.zeros(tuple(test_hr_shape)) #np.zeros((len(self.lr_paths), 80, 80))

                all_lr =  np.zeros(tuple(test_lr_shape))#np.zeros((len(self.lr_paths), 20, 20))
                all_upscaled_lr = np.zeros(tuple(test_hr_shape))#np.zeros((len(self.lr_paths), 80, 80))

                # print("Normalizing data...")
                for idx, (hr_path, lr_path, true_lr_path) in tqdm(enumerate(zip(self.hr_paths, self.upscaled_lr_paths, self.lr_paths)), total=len(self.lr_paths)):
                    hr_img = filter_data(np.load(hr_path), thresholds=self.field_threshold, field_idxs = self.field_idxs, field_names = self.field_names)

                    lr_img =  filter_data(np.load(lr_path), thresholds=self.field_threshold, field_idxs = self.field_idxs, field_names = self.field_names)

                    true_lr_img = filter_data(np.load(true_lr_path), thresholds=self.field_threshold, field_idxs = self.field_idxs, field_names = self.field_names)

                    
                    # hr_img[hr_img > self.threshold_T] = self.threshold_T
                    # lr_img[lr_img > self.threshold_T] = self.threshold_T
                    # true_lr_img[true_lr_img >
                    #             self.threshold_T] = self.threshold_T
                    all_residuals[idx] = (hr_img - lr_img)
                    all_lr[idx] = true_lr_img
                    all_hr[idx] = hr_img
                    all_upscaled_lr[idx] = lr_img

                
                if len(all_lr.shape) > 3:
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
                    self.root_folder, 'statistics', self.downscale_method)
                os.makedirs(os.path.join(self.root_folder,
                            'statistics', self.downscale_method), exist_ok=True)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'std_lr'), self.std_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'mean_lr'), self.mean_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'mean_resid'), self.mean_resid)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'std_resid'), self.std_resid)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'mean_hr'), self.mean_hr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'std_hr'), self.std_hr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'mean_upscaled_lr'), self.mean_upscaled_lr)
                np.save(os.path.join(self.root_folder, 'statistics',
                        self.downscale_method, 'std_upscaled_lr'), self.std_upscaled_lr)
                np.savetxt(os.path.join(self.root_folder,
                            'statistics', self.downscale_method, 'flag'), np.array([0]))
        elif not self.split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', self.downscale_method, 'flag')):
            raise AttributeError('Initialize training set first')
        else:
            self.stats_path = os.path.join(
                self.root_folder, 'statistics', self.downscale_method)
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

    def __len__(self):
        return len(self.lr_paths)
    def load_file(self, folders, index):
        array = filter_data(np.load(folders[index], allow_pickle=True), thresholds=self.field_threshold, field_idxs = self.field_idxs, field_names = self.field_names)

        return array
    def __getitem__(self, index):
        return self.testgetitem(index)

    def testgetitem(self, index):
        single_hr = self.load_file(self.hr_paths, index = index)
        # breakpoint()
        # single_hr = filter_data(np.load(self.hr_paths[index], allow_pickle=True), thresholds=self.field_threshold, field_idxs = self.field_idxs, field_names = self.field_names)
        
        try:
            # single_upscaled_lr = np.load(
            #     self.upscaled_lr_paths[index], allow_pickle=True)
            single_upscaled_lr = self.load_file(self.upscaled_lr_paths, index = index)
        except:
            # breakpoint()
            print(f"Upscaled low resolution data not found for this dataset ({self.downscale_method}, {self.root_folder}), defaulting to HR instead")
            # single_upscaled_lr =  np.load(
            #     self.hr_paths[index], allow_pickle=True)
            single_upscaled_lr = self.load_file(self.hr_paths, index = index)
        # single_true_lr = np.load(self.lr_paths[index], allow_pickle=True)
        # if self.load_file(self.lr_paths, index=index)[:,:,4].max() > 1000:
        # breakpoint()
        single_true_lr = self.load_file(self.lr_paths, index=index)
        # print(single_true_lr[:,:, 4].max(), single_true_lr[:,:,4].min())
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
        # breakpoint()
        for step in reversed(range(2, self.n_steps + 1)): # For n_steps = 3, the values of step will be [3,2]. 
            
                
            if index - step > 0:
                if len(np.load(self.hr_paths[index-  step], allow_pickle=True).shape) < 3:
                    # hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.hr_paths[index-  step], allow_pickle=True)[:, :, None]
                    # true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.lr_paths[index - step], allow_pickle=True)[:, :, None]
                    # upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)[:, :, None]
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.hr_paths, index = index-step)[:,:, None]
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.lr_paths, index= index-step)[:, :, None]
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.upscaled_lr_paths, index = index-step)[:,:, None]

                else:
                    # hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.hr_paths[index-  step], allow_pickle=True)
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.hr_paths, index = index-step)[:,:, None]
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.lr_paths, index = index - step)[:,:, None]
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.upscaled_lr_paths, index = index - step)[:,:, None]
                    # true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.lr_paths[index - step], allow_pickle=True)
                    # upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = np.load(self.upscaled_lr_paths[index - step], allow_pickle=True)
                    
            else:
                hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.baseline_hr#np.ones_like(single_hr)*293
                true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] =self.baseline_lr #np.ones_like(single_true_lr)*293
                upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.baseline_hr#np.ones_like(single_upscaled_lr)*293
        #if len(hr.shape) > 2:
        # breakpoint()
        hr = np.moveaxis(hr, -1, 0)


        true_lr = np.moveaxis(true_lr, -1, 0)
        upscaled_lr = np.moveaxis(upscaled_lr, -1, 0)
        # breakpoint()
        residual = hr - upscaled_lr 
        
        hr[hr > self.threshold_T] = self.threshold_T
        residual[residual > self.threshold_T] = self.threshold_T
        true_lr[true_lr > self.threshold_T] = self.threshold_T
        upscaled_lr[upscaled_lr > self.threshold_T] = self.threshold_T
      
        power = int(self.lr_paths[index].split(
            'power')[-1].split('velocity')[0])

        velocity = int(self.lr_paths[index].split('velocity')[-1].split('_')[0].strip('/'))
        time = 0.5 * \
            float(self.lr_paths[0].split('_1')[1].split('.npy')[0])/100
        # true_lr = self.rescale_data(true_lr, input_type= 'lr')
        # hr = self.rescale_data(hr, input_type = 'hr')
        # upscaled_lr  = self.rescale_data(upscaled_lr, input_type = 'upscaled_lr')
        # res = self.rescale_data(residual, input_type = 'residual')
        if self.normalize == 'standardize':
            residual = (residual - self.mean_resid)/self.std_resid
            true_lr = (true_lr - self.mean_lr)/self.std_lr
            hr = (hr - self.mean_hr)/self.std_hr
            upscaled_lr = (upscaled_lr - self.mean_upscaled_lr) / \
                self.std_upscaled_lr
        elif self.normalize == 'rescaling':
            for i, field in enumerate(self.field_max.keys()):
                min = self.field_min[field]
                max = self.field_max[field]
                # print(min, max, hr[i].min(), hr[i].max(), field)

                hr[i] = 2*(hr[i] - min)/(max - min) - 1
                true_lr[i] = 2*(true_lr[i] - min)/(max - min) -1
                upscaled_lr[i] = 2*(upscaled_lr[i] - min) / \
                    (max- min) - 1
                residual[i] = 2*(residual[i] -min)/(max - min) - 1
                # print(min, max, hr[i].min(), hr[i].max(), field)
            # hr = 2*(hr - self.t_min)/(self.t_max - self.t_min) - 1
            # true_lr = 2*(true_lr - self.t_min)/(self.t_max - self.t_min) -1
            # upscaled_lr = 2*(upscaled_lr - self.t_min) / \
            #     (self.t_max - self.t_min) - 1
            # res = 2*(residual - self.t_min)/(self.t_max - self.t_min) - 1
            


        if self.return_info:
            info = torch.tensor([power, velocity, time])
            return residual[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])], info
        return residual[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])]
    def unscale_data(self, array, input_type, normalize  = None, maintain_torch = False):
        if normalize is None:
            normalize = self.normalize
        if normalize == 'standardize':
            # breakpoint()
            if self.num_fields > 1:
                if torch.is_tensor(array) and not maintain_torch:

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
                    std = self.std_hr[tuple([self.field_idxs_steps])] 
                    mean = self.mean_hr[tuple([self.field_idxs_steps])]
                    # unscaledarray = array*self.std_hr[tuple([self.field_idxs_steps])] + self.mean_hr[tuple([self.field_idxs_steps])]
                elif input_type == 'lr':
                    std = self.std_lr[tuple([self.field_idxs_steps])]
                    mean = self.mean_lr[tuple([self.field_idxs_steps])]
                    # unscaledarray = array*self.std_lr[tuple([self.field_idxs_steps])] + self.mean_lr[tuple([self.field_idxs_steps])]
                elif input_type == 'upscaled_lr':
                    mean = self.mean_upscaled_lr[tuple([self.field_idxs_steps])]
                    std =self.std_upscaled_lr[tuple([self.field_idxs_steps])]
                    # unscaledarray = array*self.std_upscaled_lr[tuple([self.field_idxs_steps])] + self.mean_upscaled_lr[tuple([self.field_idxs_steps])]
                elif input_type == 'residual':
                    mean = self.mean_resid[tuple([self.field_idxs_steps])]
                    std = self.std_resid[tuple([self.field_idxs_steps])] 
                    # unscaledarray = array*self.std_resid[tuple([self.field_idxs_steps])] + self.mean_resid[tuple([self.field_idxs_steps])]
                else:
                    raise Exception(f'Input type not found {input_type}')
                if torch.is_tensor(array) and maintain_torch:
                    mean = torch.Tensor(mean).to('cuda').float()

                    std = torch.Tensor(std).to('cuda').float()
                elif torch.is_tensor(array):
                    array = array.cpu().detach().numpy()
                unscaledarray = array*std + mean
                


                return unscaledarray
        elif normalize == 'rescaling':

            if torch.is_tensor(array) and maintain_torch:
                unscaledarray = torch.clone(array)
            elif torch.is_tensor(array):
                array = array.cpu().detach().numpy()
            if not torch.is_tensor(array) and not maintain_torch:
                unscaledarray = np.copy(array)
            for i, (idx, field) in enumerate(zip(self.field_idxs, self.field_names)):
                min = self.field_min[field]
                max = self.field_max[field]
                # print(min, max, hr[i].min(), hr[i].max(), field)
                if len(unscaledarray.shape) == 4:
                    unscaledarray[:,i] = (0.5 + array[:,i]/2)*(max - min) + min
                else:
                    unscaledarray[i] = (0.5 + array[i]/2)*(max - min) + min
            # unscaledarray = (0.5 + array/2)*(self.t_max - self.t_min) + self.t_min

            return unscaledarray

    def rescale_data(self, array, input_type, normalize = None, maintain_torch = False):
        if normalize is None:
            normalize = self.normalize
        if normalize == 'standardize':
            if input_type == 'hr':
                mean = self.mean_hr[tuple([self.field_idxs_steps])]
                std = self.std_hr[tuple([self.field_idxs_steps])]
            elif input_type == 'lr':
                mean = self.mean_lr[tuple([self.field_idxs_steps])]
                std = self.std_lr[tuple([self.field_idxs_steps])]
            elif input_type == 'upscaled_lr':
                mean = self.mean_upscaled_lr[tuple([self.field_idxs_steps])]
                std = self.std_upscaled_lr[tuple([self.field_idxs_steps])]
            elif input_type == 'residual':
                mean = self.mean_resid[tuple([self.field_idxs_steps])]
                std = self.std_resid[tuple([self.field_idxs_steps])]
            else:
                raise NotImplementedError
            if maintain_torch:
                mean = torch.tensor(mean).to('cuda')
                std = torch.tensor(std).to('cuda')
            scaled_array = (array - mean)/(std)
            
            return scaled_array
        elif normalize == 'rescaling':
            assert(len(array.shape) == 4)
            scaled_array = torch.clone(array)
            for i, (idx, field) in enumerate(zip(self.field_idxs, self.field_names)):
                min = self.field_min[field]
                max = self.field_max[field]
                # print(min, max, hr[i].min(), hr[i].max(), field)
                if len(scaled_array.shape) == 4:
                    scaled_array[:,i] = 2*(array[:,i] - min)/(max - min) - 1
                else:
                    scaled_array[i] = 2*(array[i] - min)/(max - min) - 1
                # print(min, max, hr[i].min(), hr[i].max(), field)

            return scaled_array
            # scaled_array = (array - self.t_min)/(self.t_max - self.t_min)
            # scaled_array = (scaled_array*2) - 1 # convert from [0,1] to [0,2] to [-1,1]


def main():
    root_folder = '../datasets/expanded_ss316l_all_laser_velocity_xz_cross_section_data_expanded_frame_fluid_fraction'
    example_dir = os.path.join(root_folder, 'example')
    os.makedirs(example_dir, exist_ok = True)
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                      normalize='standardize',
                                        n_steps=1,
                                        root_folder=root_folder,
                                        split=split,
                                        field_names=['temperature', 'liqlabel'])
        print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    sample = dataset.testgetitem(1)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, drop_last=True)
 
    res, hr, true_lr, upscaled_lr = iter(dataloader).next()

    for i in range(hr.shape[1]):
        plt.imshow(hr[0,i].T, origin = 'lower',cmap = 'jet')
        plt.colorbar()
        plt.savefig(os.path.join(example_dir, 'hr_initial{}.png'.format(i)))
        plt.clf()
    exit() # 
    root_folder = '/home/oogoke/DiffusionSR/datasets/expanded_ss316l_all_laser_velocity_xz_cross_section_data'#'/home/oogoke/DiffusionSR/datasets/ss316l_v2_all_laser_velocity_xz_cross_section_data'
    example_dir =os.path.join(root_folder, 'example')
    os.makedirs(example_dir, exist_ok = True)
    print("Testing multiple field values")
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                      normalize='rescaling',
                                      n_steps=1,
                                      root_folder=root_folder,
                                      split=split,
                                      field_names= ['vx', 'temperature',  'vy', 'vz', 'liqlabel'], cross_validate=True)
        print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    # dataset.testgetitem(1)

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, drop_last=True)
    # for res, hr, true_lr, upscaled_lr in dataloader:
    #     time.sleep(1)
    #     print(true_lr[:,3].max(), true_lr[:,3].min())
    #     print(dataset.unscale_data(true_lr, input_type='lr')[:,3].max(), dataset.unscale_data(true_lr, input_type='lr')[:,3].min())
    #     print('')
    res, hr, true_lr, upscaled_lr = iter(dataloader).next()
    for i in range(hr.shape[1]):
        plt.imshow(hr[0,i].T, origin = 'lower',cmap = 'jet')
        plt.colorbar()
        plt.savefig(os.path.join(example_dir, 'hr_initial{}.png'.format(i)))
        plt.clf()


        hr_scaled = dataset.unscale_data(hr, input_type= 'hr')


        plt.imshow(hr_scaled[0,i].T, origin = 'lower',cmap = 'jet')
        plt.savefig(os.path.join(example_dir, 'hr_unscaled{}.png'.format(i)))

        hr_standardized = dataset.rescale_data(hr_scaled, input_type = 'hr', normalize = 'standardize')


        plt.imshow(hr_standardized[0,i].T, origin = 'lower',cmap = 'jet')
        plt.savefig(os.path.join(example_dir, 'hr_standardized{}.png'.format(i)))

        hr_original_space = dataset.unscale_data(hr_standardized, input_type = 'hr', normalize = 'standardize')


        plt.imshow(hr_original_space[0,i].T, origin = 'lower',cmap = 'jet')
        plt.colorbar()
        plt.savefig(os.path.join(example_dir, 'hr_original_space{}.png'.format(i)))
        plt.clf()


    plt.imshow(hr[0,1].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'temp_hr_initial.png'))

    hr_scaled = dataset.unscale_data(hr, input_type= 'hr')


    plt.imshow(hr_scaled[0,1].T, origin = 'lower',cmap = 'jet')
    plt.colorbar()
    plt.savefig(os.path.join(example_dir, 'temp_hr_unscaled.png'))
    plt.clf()
    hr_standardized = dataset.rescale_data(hr_scaled, input_type = 'hr', normalize = 'standardize')


    plt.imshow(hr_standardized[0,1].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'temp_hr_standardized.png'))

    hr_original_space = dataset.unscale_data(hr_standardized, input_type = 'hr', normalize = 'standardize')


    plt.imshow(hr_original_space[0,1].T, origin = 'lower',cmap = 'jet')
    plt.colorbar()
    plt.savefig(os.path.join(example_dir, 'temp_hr_original_space.png'))
    plt.clf()


    # breakpoint()
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

    # print("Testing 5 micron data, single field, organized")
    # root_folder = '/home/oogoke/DiffusionSR/datasets/simulation_basis_laser_velocity_xz_cross_section_data'
    # for split in ['train', 'test', 'dev']:
    #     dataset = SimulationXZDataset(downscale_method='direct',
    #                                   normalize='standardize',
    #                                   n_steps=1,
    #                                   root_folder=root_folder,
    #                                   split=split,
    #                                   field_names=['temperature'])
    #     print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    # dataset.testgetitem(1)

    # dataloader = DataLoader(dataset, batch_size=4,
    #                         shuffle=True, drop_last=True)
    # res, hr, true_lr, upscaled_lr = iter(dataloader).next()
    # lr_unscale= dataset.unscale_data(true_lr, input_type = 'lr')
    # print(np.mean(lr_unscale.numpy(), axis = (0,2, 3)))
    # print(res.shape, hr.shape, true_lr.shape)



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

    example_dir =os.path.join(root_folder, 'example')
    os.makedirs(example_dir, exist_ok = True)
    from matplotlib import pyplot as plt
    plt.imshow(dataset.unscale_data(hr, input_type = 'hr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'hr.png'))
    plt.imshow(dataset.unscale_data(true_lr, input_type = 'lr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'lr.png'))
    plt.imshow(dataset.unscale_data(upscaled_lr, input_type = 'upscaled_lr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'upscaled_lr.png'))
    plt.imshow(hr[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'hr_normalized.png'))
    plt.imshow(true_lr[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'lr_normalized.png'))
    plt.imshow(dataset.mean_hr.T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'meanhr.png'))
    print("Testing 5 micron data, single field, organized by simulation")

    print("Testing 5 micron data, single field, unorganized")
    root_folder = '/home/oogoke/DiffusionSR/datasets/update_v2_laser_velocity_xz_cross_section_data'
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                        normalize='rescaling',
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
    print(np.mean(lr_unscale, axis = (0,2, 3)))
    print(res.shape, hr.shape, true_lr.shape)

    example_dir =os.path.join(root_folder, 'example')
    os.makedirs(example_dir, exist_ok = True)
    from matplotlib import pyplot as plt
    plt.imshow(dataset.unscale_data(hr, input_type = 'hr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'rescalinghr.png'))
    plt.imshow(dataset.unscale_data(true_lr, input_type = 'lr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'rescalinglr.png'))
    plt.imshow(dataset.unscale_data(upscaled_lr, input_type = 'upscaled_lr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'rescalingupscaled_lr.png'))
    plt.imshow(hr[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'rescalinghr_normalized.png'))
    plt.imshow(true_lr[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'rescalinglr_normalized.png'))
    print("Testing 5 micron data, single field, organized by simulation")
    root_folder = '/home/oogoke/DiffusionSR/datasets/simulation_basis_v3_laser_velocity_xz_cross_section_data'
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
    example_dir =os.path.join(root_folder, 'example')
    os.makedirs(example_dir, exist_ok = True)
    from matplotlib import pyplot as plt
    plt.imshow(dataset.unscale_data(hr, input_type = 'hr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'hr.png'))
    plt.imshow(dataset.unscale_data(true_lr, input_type = 'lr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'lr.png'))
    plt.imshow(dataset.unscale_data(upscaled_lr, input_type = 'upscaled_lr')[0,0].T, origin = 'lower', vmin = 293, vmax = 5000,cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'upscaled_lr.png'))
    plt.imshow(hr[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'hr_normalized.png'))
    plt.imshow(true_lr[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'lr_normalized.png'))
    plt.imshow(dataset.mean_hr.T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'meanhr.png'))
    lr_unscale= dataset.unscale_data(true_lr, input_type = 'lr')
    print(np.mean(lr_unscale.numpy(), axis = (0,2, 3)))
    print(res.shape, hr.shape, true_lr.shape)
if __name__ == '__main__':
    main()
