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
                 out_steps = None,
                 inflate_dim=None,
                 inflate_method='repeat'):
        # note we are adding inflate_dim input
        
        
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
        self.inflate_dim = None if inflate_dim in [None, 0, 1] else int(inflate_dim)
        self.inflate_method = inflate_method
        self.return_info = return_info
        print(f"Using normalize method: ... {self.normalize}")

        # Set thresholds for each field for plotting/filtering. Ti64 multifield
        # arrays use final-axis order: temperature, liqlabel, meltregion.
        self.field_threshold = {'vx': 1000,
                                'temperature': self.THRESHOLD_T,
                                'pressure': 1e7, 'vy': 1000, 'vz': 1000,
                                'liqlabel': 1, 'meltregion': 1}

        field_aliases = {'melt_region': 'meltregion'}
        if 'ss316l' in self.root_folder:
            # Existing SS316L two-field splits store temperature, liqlabel.
            default_field_names = {'temperature': 0, 'liqlabel': 1}
        else:
            # Preserve the historical Ti64/default behavior: temperature-only
            # unless a config explicitly requests extra channels.
            default_field_names = {'temperature': 0}
        explicit_field_names = {'temperature': 0, 'liqlabel': 1, 'meltregion': 2}

        if field_names is None:
            print(f"Using all {(len(default_field_names.keys()))} fields")
            self.field_names = list(default_field_names.keys())
        else:
            print(f"Using specific fields, {field_names}")
            self.field_names = [field_aliases.get(field, field) for field in field_names]

        field_index_lookup = dict(default_field_names)
        field_index_lookup.update(explicit_field_names)
        unknown_fields = [field for field in self.field_names if field not in field_index_lookup]
        if unknown_fields:
            raise KeyError(f"Unknown field(s) {unknown_fields}. Known fields: {sorted(field_index_lookup)}")
        self.field_idxs = [field_index_lookup[key] for key in self.field_names]
        self.num_fields = len(self.field_names)
        
        self.powers = []
        self.velocities = []
        self.times = []


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
        raw_hr_shape = list(np.load(self.hr_paths[0]).shape)
        raw_lr_shape = list(np.load(self.lr_paths[0]).shape)
        self.factor = int(raw_hr_shape[0]/raw_lr_shape[0])
        print("Downscale factor: ", self.factor)

        # Load in bicubic upscaled low-fidelity data
        self.upscaled_lr_path = os.path.join(
            self.root_folder, split, 'LR', downscale_method, '{}x').format(self.factor)+os.sep
        self.upscaled_lr_paths = np.sort(np.array(
            [self.upscaled_lr_path+img_name for img_name in intersection_list if img_name.endswith('npy')], dtype='object'))

        sample_hr = self.load_file(self.hr_paths, index=0)
        sample_lr = self.load_file(self.lr_paths, index=0)
        test_hr_shape = list(sample_hr.shape)
        test_lr_shape = list(sample_lr.shape)
        self.is_true_volume = len(test_hr_shape) == 4
        if self.is_true_volume and self.inflate_dim is not None:
            print("Detected true 3D volume data; ignoring inflate_dim because depth is already explicit.")
            self.inflate_dim = None
        if len(test_hr_shape) < 3:
            self.num_fields = 1
        else:
            self.num_fields = test_hr_shape[-1]

        self.field_idxs_steps = list(range(self.n_steps * self.num_fields))

        # Add in extra fields
        self.baseline_hr = self.build_baseline(self._spatial_shape(sample_hr))
        self.baseline_lr = self.build_baseline(self._spatial_shape(sample_lr))


        self.img_shape = int(test_hr_shape[-3] if self.is_true_volume else test_hr_shape[0])

        self.compute_statistics()
         
        self.t_max = 5000
        self.t_min = 293
        self.field_max = {'vx': 100, 'temperature': self.t_max, 'pressure': 1e8, 'vy':100, 'vz': 100, 'liqlabel': 1, 'meltregion': 1 }
        self.field_min = {'vx': -100, 'temperature': self.t_min, 'pressure': 1e6, 'vy':-100, 'vz': -100, 'liqlabel': 0, 'meltregion': 0 }

    def compute_statistics(self):
        statistics_dir = os.path.join(self.root_folder, 'statistics', self.downscale_method)
        flag_path = os.path.join(statistics_dir, 'flag')
        if self.split == 'train' and not os.path.exists(flag_path):
            os.makedirs(statistics_dir, exist_ok=True)
            sums = {}
            sumsq = {}
            counts = {}
            fixed_shape_mode = not getattr(self, 'is_true_volume', False)
            sample_count = 0

            for idx in tqdm(range(len(self.lr_paths)), total=len(self.lr_paths), desc='dataset statistics'):
                residual, hr, true_lr, upscaled_lr = self._raw_arrays(index=idx)
                arrays = {
                    'lr': true_lr,
                    'hr': hr,
                    'resid': residual,
                    'upscaled_lr': upscaled_lr,
                }
                if not sums:
                    for key, array in arrays.items():
                        if fixed_shape_mode:
                            sums[key] = np.zeros_like(array, dtype=np.float64)
                            sumsq[key] = np.zeros_like(array, dtype=np.float64)
                            counts[key] = 0
                        else:
                            channels = int(array.shape[0])
                            sums[key] = np.zeros((channels,), dtype=np.float64)
                            sumsq[key] = np.zeros((channels,), dtype=np.float64)
                            counts[key] = np.zeros((channels,), dtype=np.float64)

                for key, array in arrays.items():
                    array64 = array.astype(np.float64, copy=False)
                    if fixed_shape_mode:
                        if array64.shape != sums[key].shape:
                            raise ValueError(
                                f'Variable shape detected for {key}: expected {sums[key].shape}, got {array64.shape}. '
                                'For variable-size data use true 3D arrays so channel-wise stats can broadcast.'
                            )
                        sums[key] += array64
                        sumsq[key] += array64 * array64
                        counts[key] += 1
                    else:
                        flattened = array64.reshape(array64.shape[0], -1)
                        sums[key] += flattened.sum(axis=1)
                        sumsq[key] += (flattened * flattened).sum(axis=1)
                        counts[key] += flattened.shape[1]
                sample_count += 1

            def finalize(key):
                if fixed_shape_mode:
                    mean = sums[key] / max(counts[key], 1)
                    variance = np.maximum(sumsq[key] / max(counts[key], 1) - mean * mean, 0.0)
                else:
                    safe_counts = np.maximum(counts[key], 1.0)
                    mean = sums[key] / safe_counts
                    variance = np.maximum(sumsq[key] / safe_counts - mean * mean, 0.0)
                    mean = mean.reshape((-1,) + (1,) * (arrays[key].ndim - 1))
                    variance = variance.reshape((-1,) + (1,) * (arrays[key].ndim - 1))
                return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

            self.mean_lr, self.std_lr = finalize('lr')
            self.mean_hr, self.std_hr = finalize('hr')
            self.mean_resid, self.std_resid = finalize('resid')
            self.mean_upscaled_lr, self.std_upscaled_lr = finalize('upscaled_lr')

            self.stats_path = statistics_dir
            data_to_save = {
                'std_lr': self.std_lr,
                'mean_lr': self.mean_lr,
                'mean_resid': self.mean_resid,
                'std_resid': self.std_resid,
                'mean_hr': self.mean_hr,
                'std_hr': self.std_hr,
                'mean_upscaled_lr': self.mean_upscaled_lr,
                'std_upscaled_lr': self.std_upscaled_lr,
            }
            for name, data in data_to_save.items():
                np.save(os.path.join(statistics_dir, name), data)
            np.savetxt(flag_path, np.array([0]))

        elif not self.split == 'train' and not os.path.exists(flag_path):
            raise AttributeError('Initialize training set first')
        else:
            self.stats_path = statistics_dir
            stats_files = {
                'std_lr': 'std_lr',
                'mean_lr': 'mean_lr',
                'std_resid': 'std_resid',
                'mean_resid': 'mean_resid',
                'std_hr': 'std_hr',
                'mean_hr': 'mean_hr',
                'std_upscaled_lr': 'std_upscaled_lr',
                'mean_upscaled_lr': 'mean_upscaled_lr',
            }
            for file_name, attr_name in stats_files.items():
                setattr(self, attr_name, np.load(os.path.join(self.stats_path, f'{file_name}.npy')))

        self.std_lr[self.std_lr == 0] = 1
        self.std_hr[self.std_hr == 0] = 1
        self.std_upscaled_lr[self.std_upscaled_lr == 0] = 1
        self.std_resid[self.std_resid == 0] = 1

    def __len__(self):
        return len(self.lr_paths)
    
    def load_file(self, folders, index):
        
        array = filter_data(np.load(folders[index], allow_pickle=True), 
                            thresholds=self.field_threshold, 
                            field_idxs = self.field_idxs, 
                            field_names = self.field_names)
        array = self.add_dim(self.inflate_dim, self.inflate_method, array)
        
        return array

    def build_baseline(self, spatial_shape):
        field_defaults = {
            'vx': 0,
            'temperature': 293,
            'pressure': 100000,
            'vy': 0,
            'vz': 0,
            'liqlabel': 0,
            'meltregion': 0,
        }
        baseline = np.stack(
            [np.ones(spatial_shape, dtype=np.float32) * field_defaults[field_name] for field_name in self.field_names],
            axis=-1,
        )
        return self.add_dim(self.inflate_dim, self.inflate_method, baseline)

    def _spatial_shape(self, array):
        array = self._ensure_field_last(array)
        return tuple(array.shape[:-1])

    def _ensure_field_last(self, array):
        if len(array.shape) == 2:
            return array[:, :, None]
        return array

    def _stack_temporal_channels(self, current, folders, index, baseline):
        current = self._ensure_field_last(current)
        num_fields = current.shape[-1]
        frames = [None] * self.n_steps
        frames[-1] = current
        for step in reversed(range(2, self.n_steps + 1)):
            frame_idx = self.n_steps - step
            if index - step > 0:
                previous = self._ensure_field_last(self.load_file(folders, index=index - step))
            else:
                previous = self._ensure_field_last(baseline)
            if previous.shape[:-1] != current.shape[:-1]:
                raise ValueError(
                    f'Previous timestep shape {previous.shape} does not match current shape {current.shape}'
                )
            if previous.shape[-1] != num_fields:
                raise ValueError(
                    f'Previous timestep has {previous.shape[-1]} fields, expected {num_fields}'
                )
            frames[frame_idx] = previous
        stacked = np.concatenate(frames, axis=-1)
        return np.moveaxis(stacked, -1, 0).astype(np.float32, copy=False)

    def _raw_arrays(self, index):
        single_hr = self.load_file(self.hr_paths, index=index)
        try:
            single_upscaled_lr = self.load_file(self.upscaled_lr_paths, index=index)
        except Exception:
            print(f"Upscaled low resolution data not found for this dataset ({self.downscale_method}, {self.root_folder}), defaulting to HR instead")
            single_upscaled_lr = self.load_file(self.hr_paths, index=index)
        single_true_lr = self.load_file(self.lr_paths, index=index)

        hr = self._stack_temporal_channels(single_hr, self.hr_paths, index, self.baseline_hr)
        upscaled_lr = self._stack_temporal_channels(single_upscaled_lr, self.upscaled_lr_paths, index, self.baseline_hr)
        true_lr = self._stack_temporal_channels(single_true_lr, self.lr_paths, index, self.baseline_lr)
        residual = hr - upscaled_lr

        hr[hr > self.THRESHOLD_T] = self.THRESHOLD_T
        residual[residual > self.THRESHOLD_T] = self.THRESHOLD_T
        true_lr[true_lr > self.THRESHOLD_T] = self.THRESHOLD_T
        upscaled_lr[upscaled_lr > self.THRESHOLD_T] = self.THRESHOLD_T
        return residual, hr, true_lr, upscaled_lr

    def __getitem__(self, index):
        return self.testgetitem(index)

    def testgetitem(self, index):
        residual, hr, true_lr, upscaled_lr = self._raw_arrays(index)

        power = int(self.lr_paths[index].split(
            'power')[-1].split('velocity')[0])

        velocity = int(self.lr_paths[index].split('velocity')[-1].split('_')[0].strip('/'))
        timestep = 0.5 * \
            float(self.lr_paths[0].split('_1')[1].split('.npy')[0])/100

        if self.normalize == 'standardize':
            residual = (residual - self.mean_resid)/self.std_resid
            true_lr = (true_lr - self.mean_lr)/self.std_lr
            hr = (hr - self.mean_hr)/self.std_hr
            upscaled_lr = (upscaled_lr - self.mean_upscaled_lr) / self.std_upscaled_lr
        elif self.normalize == 'rescaling':
            for channel_idx in range(hr.shape[0]):
                field = self.field_names[channel_idx % self.num_fields]
                min_value = self.field_min[field]
                max_value = self.field_max[field]
                hr[channel_idx] = 2*(hr[channel_idx] - min_value)/(max_value - min_value) - 1
                true_lr[channel_idx] = 2*(true_lr[channel_idx] - min_value)/(max_value - min_value) - 1
                upscaled_lr[channel_idx] = 2*(upscaled_lr[channel_idx] - min_value)/(max_value - min_value) - 1
                residual[channel_idx] = 2*(residual[channel_idx] - min_value)/(max_value - min_value) - 1

        if self.return_info:
            info = torch.tensor([power, velocity, timestep])
            return residual[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])], info
        return residual[tuple([self.field_idxs_steps])], hr[tuple([self.field_idxs_steps])], true_lr[tuple([self.field_idxs_steps])], upscaled_lr[tuple([self.field_idxs_steps])]

    def add_dim(self, inflate_dim, inflate_method, array):
        """Inflate the array to simulate a 3d array for each field, to simulate what diffusion would look like.
            if using repeat, repeat each field inflate_dim times, e.g. if we have 4 fields and inflate_dim = 4, we would repeat 
            each field 4 times to get a 16 channel array.
        inputs:
            inflate_dim: the dimension to inflate to, should be low to avoid memory issues, e.g. 4
            inflate_method: the method to inflate, either 'repeat' or 'maintain_width'
            array: the array to inflate, should be 2d or 3d (if 3d, will only inflate the last dimension)
        outputs:            the inflated array
        """
        if inflate_dim in [None, 0, 1]:
            return array
        if len(array.shape) == 4:
            return array
        if len(array.shape) == 2:
            array = array[:, :, None]
        if len(array.shape) != 3:
            return array
        if inflate_method not in ['repeat', 'maintain_width']:
            raise ValueError(f'Unknown inflate method: {inflate_method}')
        return np.repeat(array, repeats=inflate_dim, axis=-1)
    
    def unscale_data(self, array, input_type, normalize  = None, maintain_torch = False):
        if normalize is None:
            normalize = self.normalize
        if normalize == 'standardize':
            if self.num_fields > 1:
                if torch.is_tensor(array) and not maintain_torch:

                    array = array.cpu().detach().numpy()
            if len(self.mean_lr.shape) < 3:  
                #print(f"[{input_type}]")
                if input_type == 'hr':
                    unscaledarray = array*self.std_hr[None, :, :][tuple([self.field_idxs_steps])] + self.mean_hr[None, :, :][tuple([self.field_idxs_steps])]
                elif input_type == 'lr':
                    unscaledarray = array*self.std_lr[None, :, :][tuple([self.field_idxs_steps])] + self.mean_lr[None, :, :][tuple([self.field_idxs_steps])]
                elif input_type == 'upscaled_lr':
                    unscaledarray = array*self.std_upscaled_lr[None, :, :][tuple([self.field_idxs_steps])] + self.mean_upscaled_lr[None, :, :][tuple([self.field_idxs_steps])]
                elif input_type == 'residual':
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
