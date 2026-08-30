import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
from diffusionsr.datasets.dataset_utils import filter_data

class SimulationXZDataset(Dataset):
    def __init__(self,
                 downscale_method,
                 root_folder='../../data/update_v2_laser_velocity_xz_cross_section_data',
                 split='train',
                 normalize='standardize',
                 return_info=False, 
                 n_steps=1,
                 field_names = None, 
                 out_steps = None):
        
        # Use default HR paths, always the same
        # Use LR path as specified, may change
        # downscale_method: scaling downscale_method


        self.downscale_method = downscale_method
        self.root_folder = root_folder
        self.split = split
        self.normalize = normalize
        self.THRESHOLD_T = 8000 # threshold temperature at 8000 K
        self.n_steps = n_steps
        if out_steps is None:
            self.out_steps = n_steps
        else:
            self.out_steps = out_steps
        self.return_info = return_info
        print(f"Using normalize method: ... {self.normalize}")

        # Set thresholds for each field for plotting
        self.field_threshold = {'vx': 1000, 
                                'temperature': self.THRESHOLD_T, 
                                'pressure': 1e7, 'vy':1000, 'vz': 1000, 'liqlabel': 1 }
      
        field_metadata_path = os.path.join(self.root_folder, 'field_metadata.json')
        if os.path.exists(field_metadata_path):
            import json
            with open(field_metadata_path) as f:
                all_field_names = {name: i for i, name in enumerate(json.load(f)['fields'])}
        elif 'ss316l' in self.root_folder:
            all_field_names = {'temperature':0, 'liqlabel':1}
        else:
            all_field_names = {'temperature':0, 'liqlabel':1}



        if field_names is None:
            print(f"Using all {(len(all_field_names.keys()))} fields")
            self.field_names = list(all_field_names.keys())
            self.field_idxs = np.arange(len(all_field_names)).astype('int')
        else:
            print(f"Using specific fields, {field_names}")
            self.field_names = field_names
        self.field_idxs = [all_field_names[key] for key in self.field_names]
        self.num_fields = len(self.field_names)
        
        self.powers = []
        self.velocities = []
        self.times = []

        # Set indices for specific fields

        

      

        print(f"Processing dataset with {len(self.field_names)} fields")
        # Creating dataset splits
        # Low-fidelity
        self.lr_path = os.path.join(self.root_folder, 
                                    split, 
                                    'LR', 
                                    downscale_method, 
                                    '1x') + os.sep
        # High-fidelity
        self.hr_path = os.path.join(self.root_folder, 
                                    split, 
                                    'HR') + os.sep
        
        # Get all individual samples in each low-fidelity and high-fidelity

        img_names = [f.split(self.lr_path)[-1] for f in glob.glob(os.path.join(self.lr_path, '**/*npy'), recursive=True)]
        targ_names =  [f.split(self.hr_path)[-1] for f in glob.glob(os.path.join(self.hr_path, '**/*npy'), recursive=True)]

        # Ensure that the same samples are in both low-fidelity and high-fidelity
        intersection_list = list(set(img_names) & set(targ_names))

        self.lr_paths = np.sort(np.array(
            [os.path.join(self.lr_path, img_name) for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
        self.hr_paths = np.sort(np.array(
            [os.path.join(self.hr_path, img_name) for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))
        if len(self.lr_paths) == 0 or len(self.hr_paths) == 0:
            raise FileNotFoundError(f'No files found in specified directory: {self.lr_path}')
        
        # Identify how much the low-fidelity data has been downscaled
        test_hr_shape = list(np.load(self.hr_paths[0]).shape)
        test_lr_shape = list(np.load(self.lr_paths[0]).shape)
        self.factor = int(test_hr_shape[0]/test_lr_shape[0])
        self.num_fields_in_file = test_hr_shape[-1] if len(test_hr_shape) >= 3 else 1
        # field_idxs_steps indexes into the filter_data-reduced array (num_fields per step),
        # so the stride is num_fields, not num_fields_in_file.
        self.field_idxs_steps = list(range(self.n_steps * self.num_fields))
        print("Downscale factor: ", self.factor)


        self.num_fields_in_file = test_hr_shape[-1] if len(test_hr_shape) >= 3 else 1
        # field_idxs_steps indexes into the filter_data-reduced array (num_fields per step),
        # so the stride is num_fields, not num_fields_in_file.
        self.field_idxs_steps = list(range(self.n_steps * self.num_fields))



        # Load in bicubic upscaled low-fidelity data
        self.upscaled_lr_path = os.path.join(
            self.root_folder, split, 'LR', downscale_method, '{}x').format(self.factor)+os.sep
        self.upscaled_lr_paths = np.sort(np.array(
            [self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))

        # Add in extra fields
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
            # Build per-selected-field baseline with the right spatial shape.
            # Raw files may have more channels than we select, so use spatial dims only.
            _hr_2d = test_hr_shape[:-1] if len(test_hr_shape) >= 3 else test_hr_shape
            _lr_2d = test_lr_shape[:-1] if len(test_lr_shape) >= 3 else test_lr_shape
            _bvals = {'temperature': 293.0, 'liqlabel': 0.0}
            self.baseline_hr = np.dstack([np.ones(_hr_2d) * _bvals.get(f, 0.0) for f in self.field_names])
            self.baseline_lr = np.dstack([np.ones(_lr_2d) * _bvals.get(f, 0.0) for f in self.field_names])


        self.img_shape = int(test_hr_shape[0])
      
        test_hr_shape.insert(0, len(self.lr_paths))
        test_lr_shape.insert(0, len(self.lr_paths))
       
        self.compute_statistics()
         
        self.t_max = 5000
        self.t_min = 293
        self.field_max = {'vx': 100, 'temperature': self.t_max, 'pressure': 1e8, 'vy':100, 'vz': 100, 'liqlabel': 1 }
        self.field_min = {'vx': -100, 'temperature': self.t_min, 'pressure': 1e6, 'vy':-100, 'vz': -100, 'liqlabel': 0 }

    def compute_statistics(self):
        test_hr_shape = list(np.load(self.hr_paths[0]).shape)
        test_lr_shape = list(np.load(self.lr_paths[0]).shape)
        test_hr_shape.insert(0, len(self.lr_paths))
        test_lr_shape.insert(0, len(self.lr_paths))
        if self.split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', self.downscale_method, 'flag')):
            all_residuals = np.zeros(tuple(test_hr_shape)) # np.zeros((len(self.lr_paths), test_hr_shape[0], ))
            all_hr =  np.zeros(tuple(test_hr_shape)) # np.zeros((len(self.lr_paths), 80, 80))

            all_lr =  np.zeros(tuple(test_lr_shape)) # np.zeros((len(self.lr_paths), 20, 20))
            all_upscaled_lr = np.zeros(tuple(test_hr_shape)) # np.zeros((len(self.lr_paths), 80, 80))

            # print("Normalizing data...")
            for idx, (hr_path, lr_path, true_lr_path) in tqdm(enumerate(zip(self.hr_paths, self.upscaled_lr_paths, self.lr_paths)), total=len(self.lr_paths)):
                hr_img = filter_data(np.load(hr_path), 
                                        thresholds=self.field_threshold, 
                                        field_idxs = self.field_idxs,
                                        field_names = self.field_names)

                
                lr_img =  filter_data(np.load(lr_path), thresholds=self.field_threshold, field_idxs = self.field_idxs, field_names = self.field_names)

                true_lr_img = filter_data(np.load(true_lr_path), thresholds=self.field_threshold, field_idxs = self.field_idxs, field_names = self.field_names)

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

            self.stats_path = os.path.join(
                self.root_folder, 'statistics', self.downscale_method)
            

            # Create the statistics directory
            statistics_dir = os.path.join(self.root_folder, 'statistics', self.downscale_method)
            os.makedirs(statistics_dir, exist_ok=True)

            # Data to save and corresponding filenames
            data_to_save = {
                'std_lr': self.std_lr,
                'mean_lr': self.mean_lr,
                'mean_resid': self.mean_resid,
                'std_resid': self.std_resid,
                'mean_hr': self.mean_hr,
                'std_hr': self.std_hr,
                'mean_upscaled_lr': self.mean_upscaled_lr,
                'std_upscaled_lr': self.std_upscaled_lr
            }

            # Save each data array
            for name, data in data_to_save.items():
                np.save(os.path.join(statistics_dir, name), data)

            # Flag indicates that the statistics have been pre-computed
            np.savetxt(os.path.join(statistics_dir, 'flag'), np.array([0]))


        elif not self.split == 'train' and not os.path.exists(os.path.join(self.root_folder, 'statistics', self.downscale_method, 'flag')):
            raise AttributeError('Initialize training set first')
        else:
            self.stats_path = os.path.join(self.root_folder, 'statistics', self.downscale_method)

            # files to load and their corresponding attribute names
            stats_files = {
                'std_lr': 'std_lr',
                'mean_lr': 'mean_lr',
                'std_resid': 'std_resid',
                'mean_resid': 'mean_resid',
                'std_hr': 'std_hr',
                'mean_hr': 'mean_hr',
                'std_upscaled_lr': 'std_upscaled_lr',
                'mean_upscaled_lr': 'mean_upscaled_lr'
            }

            # load each file and assign it to the corresponding attribute
            for file_name, attr_name in stats_files.items():
                setattr(self, attr_name, np.load(os.path.join(self.stats_path, f'{file_name}.npy')))

            # Subset stats to the selected fields if the cache was computed with more fields.
            # Stats shape is (C, H, W); select field_idxs rows so it matches filter_data output.
            for attr in ['std_lr','mean_lr','std_resid','mean_resid',
                         'std_hr','mean_hr','std_upscaled_lr','mean_upscaled_lr']:
                stat = getattr(self, attr)
                if stat.ndim == 3 and stat.shape[0] > self.num_fields:
                    setattr(self, attr, stat[self.field_idxs])

        self.std_lr[self.std_lr == 0] = 1
        self.std_hr[self.std_hr == 0] = 1
        self.std_upscaled_lr[self.std_upscaled_lr == 0] = 1
        self.std_resid[self.std_resid == 0] = 1

        # Tile per-field stats across timesteps for n_steps > 1
        if self.n_steps > 1 and len(self.std_hr.shape) == 3:
            for attr in ['std_hr','mean_hr','std_lr','mean_lr',
                         'std_upscaled_lr','mean_upscaled_lr','std_resid','mean_resid']:
                setattr(self, attr, np.tile(getattr(self, attr), (self.n_steps, 1, 1)))


    def __len__(self):
        return len(self.lr_paths)
    
    def load_file(self, folders, index):
        
        array = filter_data(np.load(folders[index], allow_pickle=True), 
                            thresholds=self.field_threshold, 
                            field_idxs = self.field_idxs, 
                            field_names = self.field_names)
        
        return array
    
    def __getitem__(self, index):
        return self.testgetitem(index)

    def testgetitem(self, index):
        single_hr = self.load_file(self.hr_paths, index = index) 
        try:
            single_upscaled_lr = self.load_file(self.upscaled_lr_paths, index = index)
        except:
            print(f"Upscaled low resolution data not found for this dataset ({self.downscale_method}, {self.root_folder}), defaulting to HR instead")
            single_upscaled_lr = self.load_file(self.hr_paths, index = index)


        single_true_lr = self.load_file(self.lr_paths, index=index)
        n_channels = len(single_hr.shape)

        if n_channels < 3:
            num_fields = 1
        else:
            num_fields = single_hr.shape[-1]

        hr = np.empty((single_hr.shape[0], 
                       single_hr.shape[1],
                       self.n_steps*num_fields))


        upscaled_lr = np.empty((single_upscaled_lr.shape[0], 
                                single_upscaled_lr.shape[1], 
                                self.n_steps*num_fields))
        
        true_lr = np.empty((single_true_lr.shape[0], 
                            single_true_lr.shape[1], 
                            self.n_steps*num_fields))
        
        if len(single_hr.shape) < 3:
            single_hr = single_hr[:,:, None]
            single_upscaled_lr = single_upscaled_lr[:,:, None]
            single_true_lr = single_true_lr[:,:, None]

        hr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_hr
        upscaled_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_upscaled_lr
        true_lr[:,:,num_fields*(self.n_steps-1):num_fields*(self.n_steps)] = single_true_lr

        for step in reversed(range(2, self.n_steps + 1)): # For n_steps = 3, the values of step will be [3,2]. 
                
            if index - step > 0:
                if len(np.load(self.hr_paths[index-  step], allow_pickle=True).shape) < 3:
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.hr_paths, index = index-step)[:,:, None]
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.lr_paths, index= index-step)[:, :, None]
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.upscaled_lr_paths, index = index-step)[:,:, None]

                else:
                    hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.hr_paths, index = index-step)
                    true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.lr_paths, index = index - step)
                    upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.load_file(self.upscaled_lr_paths, index = index - step)
                    
            else:
                hr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.baseline_hr #np.ones_like(single_hr)*293
                true_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] =self.baseline_lr #np.ones_like(single_true_lr)*293
                upscaled_lr[:,:,num_fields*(self.n_steps-step):num_fields*(self.n_steps-step+1)] = self.baseline_hr#np.ones_like(single_upscaled_lr)*293

        hr = np.moveaxis(hr, -1, 0)
        true_lr = np.moveaxis(true_lr, -1, 0)
        upscaled_lr = np.moveaxis(upscaled_lr, -1, 0)
        residual = hr - upscaled_lr 
        
        hr[hr > self.THRESHOLD_T] = self.THRESHOLD_T
        residual[residual > self.THRESHOLD_T] = self.THRESHOLD_T
        true_lr[true_lr > self.THRESHOLD_T] = self.THRESHOLD_T
        upscaled_lr[upscaled_lr > self.THRESHOLD_T] = self.THRESHOLD_T
      
        power = int(self.lr_paths[index].split(
            'power')[-1].split('velocity')[0])

        velocity = int(self.lr_paths[index].split('velocity')[-1].split('_')[0].strip('/'))
        timestep = 0.5 * \
            float(self.lr_paths[0].split('_1')[1].split('.npy')[0])/100

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

                hr[i] = 2*(hr[i] - min)/(max - min) - 1
                true_lr[i] = 2*(true_lr[i] - min)/(max - min) -1
                upscaled_lr[i] = 2*(upscaled_lr[i] - min) / \
                    (max- min) - 1
                residual[i] = 2*(residual[i] -min)/(max - min) - 1

        if self.return_info:
            info = torch.tensor([power, velocity, timestep])
            return residual[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])], info
        return residual[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])]
    
    
    def unscale_data(self, array, input_type, normalize  = None, maintain_torch = False):
        if normalize is None:
            normalize = self.normalize
        if normalize == 'standardize':
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
                else:
                    raise Exception(f'Input type not found {input_type}')
                return unscaledarray
            else:
                if input_type == 'hr':
                    std = self.std_hr[tuple([self.field_idxs_steps])] 
                    mean = self.mean_hr[tuple([self.field_idxs_steps])]
                elif input_type == 'lr':
                    std = self.std_lr[tuple([self.field_idxs_steps])]
                    mean = self.mean_lr[tuple([self.field_idxs_steps])]
                elif input_type == 'upscaled_lr':
                    mean = self.mean_upscaled_lr[tuple([self.field_idxs_steps])]
                    std =self.std_upscaled_lr[tuple([self.field_idxs_steps])]
                elif input_type == 'residual':
                    mean = self.mean_resid[tuple([self.field_idxs_steps])]
                    std = self.std_resid[tuple([self.field_idxs_steps])] 
                else:
                    raise Exception(f'Input type not found {input_type}')
                if torch.is_tensor(array) and maintain_torch:
                    mean = torch.Tensor(mean).to(array.device).float()

                    std = torch.Tensor(std).to(array.device).float()
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
               
                if len(unscaledarray.shape) == 4:
                    unscaledarray[:,i] = (0.5 + array[:,i]/2)*(max - min) + min
                else:
                    unscaledarray[i] = (0.5 + array[i]/2)*(max - min) + min
            
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
                mean = torch.tensor(mean).to(array.device)
                std = torch.tensor(std).to(array.device)
            scaled_array = (array - mean)/(std)
            
            return scaled_array
        elif normalize == 'rescaling':
            assert(len(array.shape) == 4)
            scaled_array = torch.clone(array)
            for i, (idx, field) in enumerate(zip(self.field_idxs, self.field_names)):
                min = self.field_min[field]
                max = self.field_max[field]
              
                if len(scaled_array.shape) == 4:
                    scaled_array[:,i] = 2*(array[:,i] - min)/(max - min) - 1
                else:
                    scaled_array[i] = 2*(array[i] - min)/(max - min) - 1
            return scaled_array
           

def test_folder(root_folder, field_names = None, n_steps = 1, normalize = 'standardize'):

    if field_names is None:
        field_names = ['temperature']
    example_dir = os.path.join(root_folder, 'example')
    os.makedirs(example_dir, exist_ok = True)
    for split in ['train', 'test', 'dev']:
        dataset = SimulationXZDataset(downscale_method='direct',
                                      normalize=normalize,
                                        n_steps=n_steps,
                                        root_folder=root_folder,
                                        split=split,
                                        field_names=field_names)
        print(str(len(dataset.hr_paths)) + " samples available, in {} partition".format(split))
    sample = dataset.testgetitem(1)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, drop_last=True)
 
    res, hr, true_lr, upscaled_lr = next(iter(dataloader))

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
   
    plt.imshow(hr[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'temp_hr_initial.png'))

    hr_scaled = dataset.unscale_data(hr, input_type= 'hr')


    plt.imshow(hr_scaled[0,0].T, origin = 'lower',cmap = 'jet')
    plt.colorbar()
    plt.savefig(os.path.join(example_dir, 'temp_hr_unscaled.png'))
    plt.clf()
    hr_standardized = dataset.rescale_data(hr_scaled, input_type = 'hr', normalize = 'standardize')


    plt.imshow(hr_standardized[0,0].T, origin = 'lower',cmap = 'jet')
    plt.savefig(os.path.join(example_dir, 'temp_hr_standardized.png'))

    hr_original_space = dataset.unscale_data(hr_standardized, input_type = 'hr', normalize = 'standardize')


    plt.imshow(hr_original_space[0,0].T, origin = 'lower',cmap = 'jet')
    plt.colorbar()
    plt.savefig(os.path.join(example_dir, 'temp_hr_original_space.png'))
    plt.clf()

    return 


def main():
    # Testing single field value
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    data_folder = os.path.abspath(os.path.join(current_directory, '../../data'))
    test_folder(os.path.join(data_folder, 'expanded_ss316l_all_laser_velocity_xz_cross_section_data_expanded_frame'))
    print("Testing 5 micron data, single field, unorganized")


    # Testing multiple field values
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    data_folder = os.path.abspath(os.path.join(current_directory, '../../data'))
    test_folder(os.path.join(data_folder, 'expanded_ss316l_all_laser_velocity_xz_cross_section_data_expanded_frame'))
    print("Testing 5 micron data, single field, unorganized")



    folder_path = 'update_v2_laser_velocity_xz_cross_section_data'
    data_folder = os.path.abspath(os.path.join(current_directory, '../../data'))
    test_folder(os.path.join(data_folder, folder_path), n_steps = 3, normalize = 'standardize', field_names=['temperature'])

    folder_path = 'update_v2_laser_velocity_xz_cross_section_data'
    data_folder = os.path.abspath(os.path.join(current_directory, '../../data'))
    test_folder(os.path.join(data_folder, folder_path), n_steps = 3, normalize = 'rescaling', field_names=['temperature'])



    folder_path = 'simulation_basis_v3_laser_velocity_xz_cross_section_data'
    data_folder = os.path.abspath(os.path.join(current_directory, '../../data'))
    test_folder(os.path.join(data_folder, folder_path))


if __name__ == '__main__':
    main()
