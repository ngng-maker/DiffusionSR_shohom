from pathlib import Path
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusionsr.models.diffusion_model_3d import Unet3D
from diffusionsr.runners.train_diffusion import DiffusionModel, forwardpass, num_to_groups


class DiffusionModel3D(DiffusionModel):
    def __init__(
        self,
        results_folder,
        lr_encoder_folder,
        train_dataset,
        dev_dataset,
        test_dataset,
        timesteps=200,
        conditioning='implicit',
        encoding=True,
        schedule='linear',
        device='cuda',
        enc_output=True,
        out_steps=None,
        transform_rescale=False,
    ):
        # The current 3D U-Net downsamples twice before the matching skip
        # connections are concatenated on the way back up, so every spatial
        # axis must be divisible by 4 to keep tensor sizes aligned.
        self.required_axis_divisor = 4
        self.results_folder = results_folder
        self.lr_encoder_folder = lr_encoder_folder
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.timesteps = timesteps
        self.transform_rescale = transform_rescale #this flag indicates whether to rescale the input volumes during transformation
        self.encoding = encoding
        self.conditioning = conditioning
        self.schedule = schedule
        self.enc_output = enc_output
        self.device = device
        self.base_channels = self.train_dataset.n_steps * len(self.train_dataset.field_names)
        self.channels = self.base_channels if out_steps is None else int(out_steps)
        self.volume_shape = self._infer_volume_shape(self.train_dataset)
        self.depth_size = self.volume_shape[1]
        self.spatial_shape = tuple(int(size) for size in self.volume_shape[1:])
        self.image_size = self.train_dataset.img_shape

        invalid_axes = []
        for axis_name, axis_size in zip(("depth", "height", "width"), self.spatial_shape):
            if axis_size % self.required_axis_divisor != 0:
                invalid_axes.append(f"{axis_name}={axis_size}")
        if invalid_axes:
            raise ValueError(
                "3D diffusion input axes must be divisible by "
                f"{self.required_axis_divisor} for the current U-Net skip connections; "
                f"got {', '.join(invalid_axes)}. "
                "Use true 3D volumes with fixed divisible axes, or an inflate_dim divisible by 4 "
                "(for example 4, 8, or 12) for fake-depth runs."
            )

        torch.manual_seed(0)
        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        if self.encoding:
            print(f"Loading encoder ... encoding = {encoding}")
            self.lr_enc = self.initialize_encoder()
        else:
            print(f'not loading encoder, ..., encoding = {encoding} ')

        init_dim = self.channels if self.enc_output else None
        self.model = Unet3D(
            dim=self.image_size,
            channels=self.channels,
            init_dim=init_dim,
            encoder_flag=self.encoding,
            dim_mults=(1, 2, 4),
            conditioning=conditioning,
            out_dim=self.base_channels,
        )
        self.initialize_variance_schedule()
        self.model.to(self.device)
        self.save_prefix = ''

    def _infer_volume_shape(self, dataset):
        """Return one HR sample shape as (channels, depth, height, width)."""
        sample = dataset[0][1]
        sample_shape = tuple(int(size) for size in sample.shape)
        if len(sample_shape) == 4:
            return sample_shape
        if len(sample_shape) != 3:
            raise ValueError(f'Expected HR sample with 3D or 4D shape, got {sample_shape}')

        channels, height, width = sample_shape
        depth_size = int(getattr(dataset, 'inflate_dim', None) or 1)
        if channels % depth_size != 0:
            raise ValueError(
                f'Channel count {channels} is not divisible by depth size {depth_size} for 3D reshaping.'
            )
        return (channels // depth_size, depth_size, height, width)

    def reshape_to_volume(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor[:, None, :, :]# dimension 0 is batch, dimension 1 is channels, dimensions 2 and 3 are height and width. We add a new dimension for depth
        if len(tensor.shape) == 5:# If the tensor is already 5D, we assume it's in the correct shape (batch, channels, depth, height, width) and return it as is.
            return tensor
        if len(tensor.shape) != 4:
            raise ValueError(f'Expected 4D or 5D tensor, got shape {tuple(tensor.shape)}')

        batch_size, channels, height, width = tensor.shape
        if channels % self.depth_size != 0:
            raise ValueError(
                f'Channel count {channels} is not divisible by depth size {self.depth_size} for 3D reshaping.'
            )
        return tensor.reshape(batch_size, channels // self.depth_size, self.depth_size, height, width)# Reshape to (batch, channels_per_depth, depth, height, width)
    #channels represent fields such as velocity, temp, liq_label

    def ddim_compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0).to(self.device)
        # note equation alpha_t = (1 - beta_1) * (1 - beta_2) * ... * (1 - beta_t) with alpha_0 = 1, so we shift the index by 1 and prepend a zero to beta
        return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)

    def _select_stats(self, dataset, input_type):
        if input_type == 'hr':
            mean = dataset.mean_hr
            std = dataset.std_hr
        elif input_type == 'lr':
            mean = dataset.mean_lr
            std = dataset.std_lr
        elif input_type == 'upscaled_lr':
            mean = dataset.mean_upscaled_lr
            std = dataset.std_upscaled_lr
        elif input_type == 'residual':
            mean = dataset.mean_resid
            std = dataset.std_resid
        else:
            raise ValueError(f'Unsupported input_type: {input_type}')
        return np.asarray(mean), np.asarray(std)

    def unscale_volume(self, dataset, volume, input_type):
        channels, depth, height, width = volume.shape
        flat = volume.reshape(channels * depth, height, width)
        mean, std = self._select_stats(dataset, input_type)
        #

        if mean.ndim == 2:
            mean = np.broadcast_to(mean, (channels * depth, height, width))
            std = np.broadcast_to(std, (channels * depth, height, width))
            return (flat * std + mean).reshape(channels, depth, height, width)

        if mean.ndim == 3:
            unscaled = np.asarray(dataset.unscale_data(flat, input_type=input_type))
            return unscaled.reshape(channels, depth, height, width)

        if mean.ndim == 4:
            return volume * std + mean

        raise ValueError(f'Unexpected stats rank for {input_type}: {mean.ndim}')

    def save_voxel_snapshot(
        self,
        epoch,
        sample_batch_idx=0,
        threshold=1800.0,
        voxel_channel=0,
        sampler='DDIM',
        skip=10,
    ):
        snapshot_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=False)
        selected_batch = None

        for batch_idx, (_, hr, true_lr, upscaled_lr) in enumerate(snapshot_loader):
            if batch_idx == sample_batch_idx:
                selected_batch = (hr, true_lr, upscaled_lr)
                break

        if selected_batch is None:
            raise ValueError(f'Batch index {sample_batch_idx} not found in test dataset')

        hr, true_lr, upscaled_lr = selected_batch
        batch = self.reshape_to_volume(hr.to(self.device).float())
        true_lr_2d = true_lr.to(self.device).float()

        if self.encoding:
            x_e_2d = forwardpass(
                self.lr_enc,
                true_lr_2d,
                factor=self.train_dataset.factor,
                output=self.enc_output,
                transform_rescale=self.transform_rescale,
                dataset=self.test_dataset,
            )
            x_e = self.reshape_to_volume(x_e_2d.to(self.device).float())
        else:
            x_e = self.reshape_to_volume(upscaled_lr.to(self.device).float())

        self.model.eval()
        with torch.no_grad():
            all_images = self.batch_sample(
                dataset=self.test_dataset,
                batch=batch,
                x_e=x_e,
                sampler=sampler,
                skip=skip,
            )

        pred_volume = all_images[-1, 0].detach().cpu().numpy()
        target_volume = batch[0].detach().cpu().numpy()
        input_volume = x_e[0].detach().cpu().numpy()

        pred_volume_unscaled = self.unscale_volume(self.test_dataset, pred_volume, input_type='hr')
        target_volume_unscaled = self.unscale_volume(self.test_dataset, target_volume, input_type='hr')
        input_volume_unscaled = self.unscale_volume(self.test_dataset, input_volume, input_type='upscaled_lr')

        if voxel_channel < 0 or voxel_channel >= pred_volume_unscaled.shape[0]:
            raise ValueError(
                f'voxel_channel {voxel_channel} is out of bounds for volume with {pred_volume_unscaled.shape[0]} channels'
            )

        scalar_volume = np.asarray(pred_volume_unscaled[voxel_channel])
        occupancy_grid = (scalar_volume >= threshold).astype(np.uint8)

        voxel_output_dir = self.results_folder / 'voxel_exports'
        voxel_output_dir.mkdir(parents=True, exist_ok=True)
        voxel_output_path = voxel_output_dir / (
            f'epoch_{epoch + 1:04d}_batch_{sample_batch_idx:03d}_{sampler.lower()}_skip_{skip}_channel_{voxel_channel}.npz'
        )

        np.savez_compressed(
            voxel_output_path,
            scalar_volume=scalar_volume,
            occupancy_grid=occupancy_grid,
            target_scalar_volume=np.asarray(target_volume_unscaled[voxel_channel]),
            input_scalar_volume=np.asarray(input_volume_unscaled[voxel_channel]),
            threshold=np.array(threshold, dtype=np.float32),
            channel=np.array(voxel_channel, dtype=np.int32),
            sampler=np.array(sampler),
            skip=np.array(skip, dtype=np.int32),
            epoch=np.array(epoch + 1, dtype=np.int32),
            batch_index=np.array(sample_batch_idx, dtype=np.int32),
            axes=np.array(['depth', 'height', 'width']),
        )

        if not voxel_output_path.exists() or voxel_output_path.stat().st_size == 0:
            raise IOError(f'Voxel export was not written correctly: {voxel_output_path}')

        with np.load(voxel_output_path, allow_pickle=False) as saved_voxels:
            saved_scalar_volume = saved_voxels['scalar_volume']
            saved_occupancy_grid = saved_voxels['occupancy_grid']

        if saved_scalar_volume.shape != scalar_volume.shape:
            raise ValueError(
                f'Saved scalar volume shape mismatch: expected {scalar_volume.shape}, got {saved_scalar_volume.shape}'
            )
        if not np.array_equal(saved_occupancy_grid, occupancy_grid):
            raise ValueError('Saved occupancy grid contents do not match the in-memory voxel mask')

        print(f'Saved voxel export for epoch {epoch + 1}: {voxel_output_path}')
        return voxel_output_path

    @torch.no_grad()
    def sample(self, model, x_e, image_size, timesteps, batch_size=16, channels=3):
        shape = (batch_size, channels, *self.spatial_shape)
        return self.p_sample_loop(model, x_e, timesteps=timesteps, shape=shape)

    def batch_sample(self, dataset, batch, x_e, sampler='DDPM', skip=None, **kwargs):
        timesteps = self.timesteps
        if sampler == 'DDPM':
            batch_size = batch.shape[0]
            batches = num_to_groups(1, batch_size)
            all_images_list = list(
                map(
                    lambda n: self.sample(
                        self.model,
                        timesteps=timesteps,
                        x_e=x_e,
                        image_size=dataset.img_shape,
                        batch_size=batch_size,
                        channels=self.channels,
                    ),
                    batches,
                )
            )[0]
        elif sampler == 'DDIM':
            shape = batch.shape
            if skip is None:
                skip = 1
            seq = range(0, timesteps, skip)
            with torch.no_grad():
                x = torch.randn(shape, device=self.device)
                n = x.size(0)
                seq_next = [-1] + list(seq[:-1])
                xs = [x]
                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = self.ddim_compute_alpha(self.betas, t.long())
                    at_next = self.ddim_compute_alpha(self.betas, next_t.long())
                    xt = xs[-1].to(self.device)
                    et = self.model(xt, t, x_e)
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    c1 = kwargs.get('eta', 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                    xs.append(xt_next.to(self.device))
            all_images_list = xs
        else:
            raise NotImplementedError(f'No such sampler, {sampler}')
        return torch.stack(all_images_list, dim=0)

    def train(
        self,
        epochs,
        restart=False,
        restart_dir='',
        additional_epochs=None,
        batch_size=8,
        learning_rate=1e-5,
        loss_type='l1',
        voxel_save_interval=0,
        voxel_sample_batch_idx=0,
        voxel_threshold=1800.0,
        voxel_channel=0,
        voxel_sampler='DDIM',
        voxel_skip=10,
    ):
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.restart = restart
        self.batch_size = batch_size
        self.restart_dir = restart_dir
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.start_epoch = 0
        self.step = 0

        if restart:
            print('Resuming training...')
            checkpoint = torch.load(os.path.join(self.restart_dir, 'ckpt.pth'))
            self.model.load_state_dict(checkpoint[0])
            self.optimizer.load_state_dict(checkpoint[1])
            self.start_epoch = checkpoint[2] + 1
            self.step = checkpoint[3]

        if additional_epochs is not None:
            self.epochs = self.start_epoch + int(additional_epochs)
        else:
            self.epochs = int(epochs)

        print(f'Training epoch range: start={self.start_epoch}, stop={self.epochs}')

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        all_train_losses = []
        unaveraged_train_losses = []
        all_test_losses = []
        unaveraged_test_losses = []

        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            losses = []
            self.model.train()

            for step, (_, hr, true_lr, upscaled_lr) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                batch = self.reshape_to_volume(hr.to(self.device).float())
                true_lr_2d = true_lr.to(self.device).float()
                self.optimizer.zero_grad()

                if self.encoding:
                    x_e_2d = forwardpass(
                        self.lr_enc,
                        true_lr_2d,
                        factor=self.train_dataset.factor,
                        output=self.enc_output,
                        transform_rescale=self.transform_rescale,
                        dataset=self.train_loader.dataset,
                    )
                    x_e = self.reshape_to_volume(x_e_2d.to(self.device).float())
                else:
                    x_e = self.reshape_to_volume(upscaled_lr.to(self.device).float())

                t = torch.randint(0, self.timesteps, (batch.shape[0],), device=self.device).long()
                loss = self.p_losses(self.model, batch, t, loss_type=self.loss_type, x_e=x_e)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

            mean_loss = np.mean(losses)
            all_train_losses.append(mean_loss)
            unaveraged_train_losses.extend(losses)
            np.savetxt(os.path.join(self.results_folder, self.save_prefix + 'loss_epoch.txt'), all_train_losses)
            np.savetxt(os.path.join(self.results_folder, self.save_prefix + 'loss_iterations.txt'), unaveraged_train_losses)

            self.model.eval()
            test_losses = []
            for _, (_, hr, true_lr, upscaled_lr) in enumerate(self.dev_loader):
                batch = self.reshape_to_volume(hr.to(self.device).float())
                true_lr_2d = true_lr.to(self.device).float()
                if self.encoding:
                    x_e_2d = forwardpass(
                        self.lr_enc,
                        true_lr_2d,
                        factor=self.train_dataset.factor,
                        output=self.enc_output,
                        transform_rescale=self.transform_rescale,
                        dataset=self.train_loader.dataset,
                    )
                    x_e = self.reshape_to_volume(x_e_2d.to(self.device).float())
                else:
                    x_e = self.reshape_to_volume(upscaled_lr.to(self.device).float())

                t = torch.randint(0, self.timesteps, (batch.shape[0],), device=self.device).long()
                loss = self.p_losses(self.model, batch, t, loss_type=self.loss_type, x_e=x_e)
                test_losses.append(loss.item())

            mean_test_loss = np.mean(test_losses)
            all_test_losses.append(mean_test_loss)
            unaveraged_test_losses.extend(test_losses)
            np.savetxt(os.path.join(self.results_folder, self.save_prefix + 'validation_loss_epoch.txt'), all_test_losses)
            np.savetxt(os.path.join(self.results_folder, self.save_prefix + 'validation_loss_iterations.txt'), unaveraged_test_losses)

            if voxel_save_interval and (epoch + 1) % int(voxel_save_interval) == 0:
                self.save_voxel_snapshot(
                    epoch=epoch,
                    sample_batch_idx=voxel_sample_batch_idx,
                    threshold=voxel_threshold,
                    voxel_channel=voxel_channel,
                    sampler=voxel_sampler,
                    skip=voxel_skip,
                )

            states = [self.model.state_dict(), self.optimizer.state_dict(), epoch, step]
            torch.save(states, os.path.join(self.results_folder, 'ckpt.pth'))