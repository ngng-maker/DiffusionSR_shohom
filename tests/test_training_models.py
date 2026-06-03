from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from diffusionsr.models.diffusion_model_3d import Unet3D
from diffusionsr.models.lr_encoder_model import rrdbnet_encoder
from diffusionsr.models.vae_model import VAE2D, VAE3D, vae_loss
from diffusionsr.runners.train_diffusion import DiffusionModel, forwardpass
from diffusionsr.runners.train_diffusion_3d import DiffusionModel3D
from diffusionsr.runners.train_vae import VAETrainer


class DummyDataset:
    def __init__(self, num_fields: int, img_shape: int = 16, n_steps: int = 1, factor: int = 2):
        self.num_fields = num_fields
        self.img_shape = img_shape
        self.n_steps = n_steps
        self.out_steps = n_steps
        self.factor = factor
        self.inflate_dim = None
        self.field_names = ["temperature"] if num_fields == 1 else ["temperature", "liqlabel"]


class TrainableDummyDataset(DummyDataset):
    def __init__(self, num_fields: int, img_shape: int = 16, n_steps: int = 1, factor: int = 2):
        super().__init__(num_fields=num_fields, img_shape=img_shape, n_steps=n_steps, factor=factor)
        self.mean_hr = np.zeros((img_shape, img_shape), dtype=np.float32)
        self.std_hr = np.ones((img_shape, img_shape), dtype=np.float32)
        self.mean_lr = np.zeros((img_shape // factor, img_shape // factor), dtype=np.float32)
        self.std_lr = np.ones((img_shape // factor, img_shape // factor), dtype=np.float32)
        self.mean_upscaled_lr = np.zeros((img_shape, img_shape), dtype=np.float32)
        self.std_upscaled_lr = np.ones((img_shape, img_shape), dtype=np.float32)
        self.mean_resid = np.zeros((img_shape, img_shape), dtype=np.float32)
        self.std_resid = np.ones((img_shape, img_shape), dtype=np.float32)
        self.field_idxs_steps = np.arange(self.num_fields)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        hr = torch.ones(self.n_steps * self.num_fields, self.img_shape, self.img_shape, dtype=torch.float32)
        true_lr = torch.ones(
            self.n_steps * self.num_fields,
            self.img_shape // self.factor,
            self.img_shape // self.factor,
            dtype=torch.float32,
        )
        upscaled_lr = torch.ones_like(hr)
        residual = torch.zeros_like(hr)
        return residual, hr, true_lr, upscaled_lr


def _make_diffusion_model(tmp_path: Path, dataset: DummyDataset, encoding: bool, monkeypatch):
    if encoding:
        def fake_initialize_encoder(self):
            model = rrdbnet_encoder(
                upscale_factor=self.train_dataset.factor,
                in_channels=self.train_dataset.n_steps * self.train_dataset.num_fields,
                out_channels=self.train_dataset.out_steps * self.train_dataset.num_fields,
            )
            model.eval()
            return model.to(self.device)

        monkeypatch.setattr(DiffusionModel, "initialize_encoder", fake_initialize_encoder)

    return DiffusionModel(
        results_folder=tmp_path / f"diffusion_{dataset.num_fields}_{'enc' if encoding else 'plain'}",
        lr_encoder_folder=tmp_path / "encoder",
        train_dataset=dataset,
        dev_dataset=dataset,
        test_dataset=dataset,
        timesteps=4,
        conditioning="implicit",
        encoding=encoding,
        schedule="linear",
        device="cpu",
        enc_output=True,
    )


def _make_diffusion_model_3d(tmp_path: Path, dataset: DummyDataset, encoding: bool, monkeypatch):
    dataset.inflate_dim = 4
    dataset.num_fields = len(dataset.field_names) * dataset.inflate_dim
    if encoding:
        def fake_initialize_encoder(self):
            model = rrdbnet_encoder(
                upscale_factor=self.train_dataset.factor,
                in_channels=self.train_dataset.n_steps * self.train_dataset.num_fields,
                out_channels=self.train_dataset.out_steps * self.train_dataset.num_fields,
            )
            model.eval()
            return model.to(self.device)

        monkeypatch.setattr(DiffusionModel3D, "initialize_encoder", fake_initialize_encoder)

    return DiffusionModel3D(
        results_folder=tmp_path / f"diffusion3d_{len(dataset.field_names)}_{'enc' if encoding else 'plain'}",
        lr_encoder_folder=tmp_path / "encoder3d",
        train_dataset=dataset,
        dev_dataset=dataset,
        test_dataset=dataset,
        timesteps=4,
        conditioning="implicit",
        encoding=encoding,
        schedule="linear",
        device="cpu",
        enc_output=True,
    )


@torch.no_grad()
def test_encoder_model_accepts_single_and_multi_field_inputs():
    for channels in [1, 2]:
        model = rrdbnet_encoder(upscale_factor=2, in_channels=channels, out_channels=channels)
        x = torch.randn(2, channels, 8, 8)
        y = model(x)
        assert y.shape == (2, channels, 16, 16)


def test_diffusion_trainer_smoke_single_and_multi_field(tmp_path: Path, monkeypatch):
    for num_fields in [1, 2]:
        dataset = DummyDataset(num_fields=num_fields)
        model = _make_diffusion_model(tmp_path, dataset, encoding=False, monkeypatch=monkeypatch)
        channels = dataset.n_steps * dataset.num_fields
        batch = torch.randn(2, channels, dataset.img_shape, dataset.img_shape)
        x_e = torch.randn_like(batch)
        t = torch.randint(0, model.timesteps, (batch.shape[0],), dtype=torch.long)
        loss = model.p_losses(model.model, batch, t, loss_type="l1", x_e=x_e)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_diffusion_trainer_encoder_conditioning_smoke_single_and_multi_field(tmp_path: Path, monkeypatch):
    for num_fields in [1, 2]:
        dataset = DummyDataset(num_fields=num_fields)
        model = _make_diffusion_model(tmp_path, dataset, encoding=True, monkeypatch=monkeypatch)
        channels = dataset.n_steps * dataset.num_fields
        true_lr = torch.randn(2, channels, dataset.img_shape // dataset.factor, dataset.img_shape // dataset.factor)
        batch = torch.randn(2, channels, dataset.img_shape, dataset.img_shape)
        x_e = forwardpass(model.lr_enc, true_lr, factor=dataset.factor, output=model.enc_output)
        t = torch.randint(0, model.timesteps, (batch.shape[0],), dtype=torch.long)
        loss = model.p_losses(model.model, batch, t, loss_type="l1", x_e=x_e)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_diffusion_train_step_updates_parameters_on_cpu(tmp_path: Path, monkeypatch):
    for num_fields in [1, 2]:
        dataset = DummyDataset(num_fields=num_fields)
        model = _make_diffusion_model(tmp_path, dataset, encoding=False, monkeypatch=monkeypatch)
        optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
        channels = dataset.n_steps * dataset.num_fields
        batch = torch.randn(2, channels, dataset.img_shape, dataset.img_shape)
        x_e = torch.randn_like(batch)
        t = torch.randint(0, model.timesteps, (batch.shape[0],), dtype=torch.long)
        before = next(model.model.parameters()).detach().clone()
        loss = model.p_losses(model.model, batch, t, loss_type="l1", x_e=x_e)
        loss.backward()
        optimizer.step()
        after = next(model.model.parameters()).detach()
        assert not torch.equal(before, after)


def test_diffusion3d_trainer_smoke_single_and_multi_field(tmp_path: Path, monkeypatch):
    for num_fields in [1, 2]:
        dataset = DummyDataset(num_fields=num_fields)
        model = _make_diffusion_model_3d(tmp_path, dataset, encoding=False, monkeypatch=monkeypatch)
        channels_2d = dataset.n_steps * dataset.num_fields
        batch_2d = torch.randn(2, channels_2d, dataset.img_shape, dataset.img_shape)
        batch = model.reshape_to_volume(batch_2d)
        x_e = torch.randn_like(batch)
        t = torch.randint(0, model.timesteps, (batch.shape[0],), dtype=torch.long)
        loss = model.p_losses(model.model, batch, t, loss_type="l1", x_e=x_e)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_diffusion3d_restart_uses_next_epoch_and_additional_epochs(tmp_path: Path, monkeypatch):
    dataset = TrainableDummyDataset(num_fields=1)
    model = _make_diffusion_model_3d(tmp_path, dataset, encoding=False, monkeypatch=monkeypatch)

    checkpoint_path = Path(model.results_folder) / "ckpt.pth"
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
    torch.save([model.model.state_dict(), optimizer.state_dict(), 4, 0], checkpoint_path)

    saved_epochs = []

    def fake_p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", x_e=None):
        return next(self.model.parameters()).sum() * 0

    def fake_save_voxel_snapshot(self, epoch, **kwargs):
        saved_epochs.append(epoch)
        return Path(self.results_folder) / "voxel_exports" / f"epoch_{epoch + 1:04d}.npz"

    monkeypatch.setattr(DiffusionModel3D, "p_losses", fake_p_losses)
    monkeypatch.setattr(DiffusionModel3D, "save_voxel_snapshot", fake_save_voxel_snapshot)

    model.train(
        epochs=999,
        restart=True,
        restart_dir=str(model.results_folder),
        additional_epochs=2,
        batch_size=1,
        learning_rate=1e-3,
        loss_type="l1",
        voxel_save_interval=1,
        voxel_sample_batch_idx=0,
    )

    assert model.start_epoch == 5
    assert model.epochs == 7
    assert saved_epochs == [5, 6]


def test_diffusion3d_voxel_snapshot_writes_archive(tmp_path: Path, monkeypatch):
    dataset = TrainableDummyDataset(num_fields=1)
    model = _make_diffusion_model_3d(tmp_path, dataset, encoding=False, monkeypatch=monkeypatch)

    def fake_batch_sample(self, dataset, batch, x_e, sampler="DDIM", skip=None, **kwargs):
        prediction = torch.full_like(batch, 2.0)
        return torch.stack([batch, prediction], dim=0)

    monkeypatch.setattr(DiffusionModel3D, "batch_sample", fake_batch_sample)

    output_path = model.save_voxel_snapshot(
        epoch=2,
        sample_batch_idx=0,
        threshold=1.5,
        voxel_channel=0,
        sampler="DDIM",
        skip=2,
    )

    assert output_path.exists()
    with np.load(output_path, allow_pickle=False) as saved_voxels:
        assert saved_voxels["scalar_volume"].shape == (4, 16, 16)
        assert int(saved_voxels["occupancy_grid"].sum()) == 4 * 16 * 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_diffusion_trainer_smoke_on_gpu(tmp_path: Path, monkeypatch):
    dataset = DummyDataset(num_fields=1)
    model = _make_diffusion_model(tmp_path, dataset, encoding=False, monkeypatch=monkeypatch)
    model.model = model.model.to("cuda")
    model.device = "cuda"
    batch = torch.randn(2, 1, dataset.img_shape, dataset.img_shape, device="cuda")
    x_e = torch.randn_like(batch)
    t = torch.randint(0, model.timesteps, (batch.shape[0],), dtype=torch.long, device="cuda")
    loss = model.p_losses(model.model, batch, t, loss_type="l1", x_e=x_e)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@torch.no_grad()
def test_unet3d_accepts_single_and_multi_field_inputs():
    for channels in [1, 2]:
        model = Unet3D(
            dim=8,
            encoder_flag=False,
            init_dim=channels,
            out_dim=channels,
            dim_mults=(1, 2),
            channels=channels,
            conditioning="implicit",
        )
        x = torch.randn(2, channels, 8, 8, 8)
        x_e = torch.randn_like(x)
        t = torch.randint(0, 4, (2,), dtype=torch.long)
        y = model(x, t, x_e=x_e)
        assert y.shape == x.shape

@torch.no_grad()
def test_vae_models_reconstruct_2d_and_3d_input_shapes():
    model_2d = VAE2D(input_channels=2, latent_channels=3, hidden_channels=4, channel_multipliers=(1,))
    x_2d = torch.randn(2, 2, 15, 17)
    out_2d = model_2d(x_2d, sample_posterior=False)
    assert out_2d.reconstruction.shape == x_2d.shape
    assert out_2d.mu.shape[1] == 3

    model_3d = VAE3D(input_channels=1, latent_channels=2, hidden_channels=4, channel_multipliers=(1,))
    x_3d = torch.randn(2, 1, 4, 16, 16)
    out_3d = model_3d(x_3d, sample_posterior=False)
    assert out_3d.reconstruction.shape == x_3d.shape
    assert out_3d.mu.shape[1] == 2

    losses = vae_loss(out_3d, x_3d, beta=1e-4, loss_type="l1")
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_vae_trainer_saves_best_checkpoint_on_cpu(tmp_path: Path):
    dataset = TrainableDummyDataset(num_fields=1, img_shape=8)
    trainer = VAETrainer(
        results_folder=tmp_path / "vae2d",
        train_dataset=dataset,
        dev_dataset=dataset,
        test_dataset=dataset,
        spatial_dims=2,
        latent_channels=2,
        hidden_channels=4,
        channel_multipliers=(1,),
        log_interval=0,
        sample_interval=0,
    )

    metrics = trainer.train(epochs=1, batch_size=1, learning_rate=1e-3)

    best_path = Path(trainer.results_folder) / "best_model.pth"
    latest_path = Path(trainer.results_folder) / "ckpt.pth"
    assert best_path.exists()
    assert latest_path.exists()
    assert "validation/loss" in metrics

    checkpoint = torch.load(best_path, map_location="cpu")
    assert checkpoint["model_config"]["spatial_dims"] == 2
    assert checkpoint["model_config"]["latent_channels"] == 2


def test_vae_trainer_formats_inflated_channels_as_3d_volume(tmp_path: Path):
    dataset = TrainableDummyDataset(num_fields=1, img_shape=8)
    dataset.inflate_dim = 4
    dataset.num_fields = 4
    trainer = VAETrainer(
        results_folder=tmp_path / "vae3d",
        train_dataset=dataset,
        dev_dataset=dataset,
        test_dataset=dataset,
        spatial_dims=3,
        latent_channels=2,
        hidden_channels=4,
        channel_multipliers=(1,),
        depth_size=4,
        log_interval=0,
        sample_interval=0,
    )

    batch = next(iter(DataLoader(dataset, batch_size=1)))
    target = trainer._select_target(batch)
    assert target.shape == (1, 1, 4, 8, 8)

    metrics = trainer.train(epochs=1, batch_size=1, learning_rate=1e-3)
    assert torch.isfinite(torch.tensor(metrics["validation/loss"]))
    assert (Path(trainer.results_folder) / "best_model.pth").exists()
