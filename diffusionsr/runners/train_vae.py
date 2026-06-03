import csv
import datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from diffusionsr.models.vae_model import VAE2D, VAE3D, vae_loss


_TARGET_INDEX = {
    "residual": 0,
    "hr": 1,
    "high_resolution": 1,
    "lr": 2,
    "true_lr": 2,
    "upscaled_lr": 3,
}


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def parse_channel_multipliers(value) -> Tuple[int, ...]:
    if value is None:
        return (1, 2, 4)
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, int):
        return (value,)
    return tuple(int(part) for part in value)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _wandb_log(payload: dict, step: Optional[int] = None) -> None:
    if wandb.run is not None:
        wandb.log(payload, step=step)


class VAETrainer:
    def __init__(
        self,
        results_folder,
        train_dataset,
        dev_dataset,
        test_dataset=None,
        spatial_dims: int = 2,
        target_type: str = "hr",
        input_type: Optional[str] = None,
        input_channels: Optional[int] = None,
        output_channels: Optional[int] = None,
        latent_channels: int = 4,
        hidden_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        output_activation: Optional[str] = None,
        beta: float = 1e-4,
        reconstruction_loss: str = "l1",
        kl_anneal_epochs: int = 0,
        num_workers: int = 0,
        log_interval: int = 50,
        sample_interval: int = 1,
        save_every: int = 0,
        grad_clip_norm: Optional[float] = None,
        depth_size: Optional[int] = None,
    ):
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        if target_type not in _TARGET_INDEX:
            raise ValueError(f"target_type must be one of {sorted(_TARGET_INDEX)}, got {target_type}")
        if input_type is None:
            input_type = target_type
        if input_type not in _TARGET_INDEX:
            raise ValueError(f"input_type must be one of {sorted(_TARGET_INDEX)}, got {input_type}")

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.spatial_dims = int(spatial_dims)
        self.target_type = target_type
        self.input_type = input_type
        self.depth_size = int(depth_size or getattr(train_dataset, "inflate_dim", None) or 1)
        self.beta = float(beta)
        self.reconstruction_loss = reconstruction_loss
        self.kl_anneal_epochs = int(kl_anneal_epochs or 0)
        self.num_workers = int(num_workers)
        self.log_interval = int(log_interval)
        self.sample_interval = int(sample_interval)
        self.save_every = int(save_every)
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_validation_loss = math.inf
        self.start_epoch = 0
        self.global_step = 0

        inferred_channels = self._infer_model_channels()
        self.input_channels = int(input_channels or inferred_channels)
        self.output_channels = int(output_channels or self.input_channels)
        self.channel_multipliers = parse_channel_multipliers(channel_multipliers)

        model_cls = VAE2D if self.spatial_dims == 2 else VAE3D
        self.model = model_cls(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            latent_channels=int(latent_channels),
            hidden_channels=int(hidden_channels),
            channel_multipliers=self.channel_multipliers,
            output_activation=output_activation,
        ).to(self.device)

        self.model_config = self.model.config()
        self.trainer_config = {
            "spatial_dims": self.spatial_dims,
            "input_type": self.input_type,
            "target_type": self.target_type,
            "depth_size": self.depth_size,
            "beta": self.beta,
            "reconstruction_loss": self.reconstruction_loss,
            "kl_anneal_epochs": self.kl_anneal_epochs,
            "num_workers": self.num_workers,
            "log_interval": self.log_interval,
            "sample_interval": self.sample_interval,
            "save_every": self.save_every,
            "grad_clip_norm": self.grad_clip_norm,
        }
        self._write_metadata()
        self._log_timestamp("initialized")

    def _infer_model_channels(self) -> int:
        n_steps = int(getattr(self.train_dataset, "n_steps", 1))
        num_fields = int(getattr(self.train_dataset, "num_fields", 1))
        total_channels = n_steps * num_fields
        if self.spatial_dims == 3:
            if total_channels % self.depth_size != 0:
                raise ValueError(
                    f"Cannot reshape {total_channels} channels into depth {self.depth_size}; "
                    "set inflate_dim/depth_size so channels are divisible by depth."
                )
            return total_channels // self.depth_size
        return total_channels

    def _write_metadata(self) -> None:
        metadata = {
            "created_at": now_iso(),
            "model_config": self.model_config,
            "trainer_config": self.trainer_config,
            "train_samples": len(self.train_dataset),
            "dev_samples": len(self.dev_dataset),
            "test_samples": len(self.test_dataset) if self.test_dataset is not None else None,
        }
        with open(self.results_folder / "vae_metadata.json", "w") as f:
            json.dump(_jsonable(metadata), f, indent=2)

    def _log_timestamp(self, event: str, **metadata) -> None:
        record = {"event": event, "timestamp": now_iso()}
        record.update(metadata)
        with open(self.results_folder / "timestamps.log", "a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")

    def _select_tensor(self, batch, data_type: str) -> torch.Tensor:
        tensor = batch[_TARGET_INDEX[data_type]]
        return self._format_tensor(tensor).to(self.device).float()

    def _select_input_and_target(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        model_input = self._select_tensor(batch, self.input_type)
        target = self._select_tensor(batch, self.target_type)
        return model_input, target

    def _select_target(self, batch) -> torch.Tensor:
        return self._select_tensor(batch, self.target_type)

    def _format_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2:
            if tensor.ndim == 3:
                return tensor.unsqueeze(1)
            if tensor.ndim != 4:
                raise ValueError(f"Expected a 4D 2D batch, got shape {tuple(tensor.shape)}")
            return tensor

        if tensor.ndim == 5:
            return tensor
        if tensor.ndim != 4:
            raise ValueError(f"Expected a 4D flattened-volume batch or 5D volume, got {tuple(tensor.shape)}")
        batch_size, channels, height, width = tensor.shape
        if self.depth_size <= 1:
            return tensor.reshape(batch_size, channels, 1, height, width)
        if channels % self.depth_size != 0:
            raise ValueError(f"Channel count {channels} is not divisible by depth size {self.depth_size}")
        return tensor.reshape(batch_size, channels // self.depth_size, self.depth_size, height, width)

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_anneal_epochs <= 0:
            return self.beta
        return self.beta * min(1.0, float(epoch + 1) / float(self.kl_anneal_epochs))

    def _make_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> Dict[str, float]:
        self.model.train(train)
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
        sample_count = 0
        split = "train" if train else "validation"
        iterator = tqdm(loader, desc=f"{split} epoch {epoch + 1}", leave=False)

        for step, batch in enumerate(iterator):
            model_input, target = self._select_input_and_target(batch)
            batch_size = target.shape[0]
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                output = self.model(
                    model_input,
                    output_shape=target.shape[-self.spatial_dims:],
                )
                losses = vae_loss(
                    output,
                    target,
                    beta=self._kl_weight(epoch),
                    loss_type=self.reconstruction_loss,
                )

            if train:
                losses["loss"].backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
                self.optimizer.step()
                self.global_step += 1

            for name in totals:
                totals[name] += losses[name].detach().item() * batch_size
            sample_count += batch_size
            iterator.set_postfix(loss=losses["loss"].detach().item())

            if train and self.log_interval > 0 and self.global_step % self.log_interval == 0:
                _wandb_log(
                    {f"{split}/{name}": losses[name].detach().item() for name in totals},
                    step=self.global_step,
                )

        divisor = max(1, sample_count)
        return {name: total / divisor for name, total in totals.items()}

    def _history_path(self) -> Path:
        return self.results_folder / "history.csv"

    def _append_history(self, row: dict) -> None:
        path = self._history_path()
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _checkpoint_payload(self, epoch: int, metrics: dict) -> dict:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "metrics": metrics,
            "model_config": self.model_config,
            "trainer_config": self.trainer_config,
            "saved_at": now_iso(),
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        payload = self._checkpoint_payload(epoch, metrics)
        latest_path = self.results_folder / "ckpt.pth"
        torch.save(payload, latest_path)

        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            torch.save(payload, self.results_folder / f"ckpt_epoch_{epoch + 1:04d}.pth")

        if is_best:
            best_path = self.results_folder / "best_model.pth"
            torch.save(payload, best_path)
            self._log_timestamp("best_model", epoch=epoch + 1, validation_loss=metrics["validation/loss"])
            self._log_artifact(best_path, aliases=["best", f"epoch-{epoch + 1}"], metadata=metrics)

    def _load_checkpoint(self, restart_dir: str) -> None:
        checkpoint_path = Path(restart_dir) / "ckpt.pth"
        if not checkpoint_path.exists():
            checkpoint_path = Path(restart_dir) / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No VAE checkpoint found in {restart_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self._log_timestamp("restart", checkpoint=str(checkpoint_path), start_epoch=self.start_epoch + 1)

    def _log_artifact(self, path: Path, aliases=None, metadata=None) -> None:
        if wandb.run is None:
            return
        aliases = aliases or ["latest"]
        artifact_name = f"vae_{self.spatial_dims}d_{Path(self.results_folder).name}".replace("/", "_")
        try:
            artifact = wandb.Artifact(artifact_name, type="model", metadata=_jsonable(metadata or {}))
            artifact.add_file(str(path))
            wandb.log_artifact(artifact, aliases=aliases)
        except Exception as exc:
            print(f"Could not log wandb artifact {path}: {exc}")

    def _save_reconstruction_plot(self, loader: DataLoader, epoch: int, split: str) -> Optional[Path]:
        try:
            batch = next(iter(loader))
        except StopIteration:
            return None

        self.model.eval()
        with torch.no_grad():
            model_input, target = self._select_input_and_target(batch)
            output = self.model(
                model_input,
                sample_posterior=False,
                output_shape=target.shape[-self.spatial_dims:],
            )

        target_np = target[0, 0].detach().cpu().numpy()
        recon_np = output.reconstruction[0, 0].detach().cpu().numpy()
        if self.spatial_dims == 3:
            center = target_np.shape[0] // 2
            target_np = target_np[center]
            recon_np = recon_np[center]
        error_np = np.abs(recon_np - target_np)

        vmin = float(min(target_np.min(), recon_np.min()))
        vmax = float(max(target_np.max(), recon_np.max()))
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=200)
        for ax, array, title in zip(axes, [target_np, recon_np, error_np], ["target", "reconstruction", "absolute error"]):
            if title == "absolute error":
                im = ax.imshow(array.T, origin="lower", cmap="magma")
            else:
                im = ax.imshow(array.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        image_dir = self.results_folder / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{split}_reconstruction_epoch_{epoch + 1:04d}.png"
        fig.savefig(path)
        plt.close(fig)
        _wandb_log({f"{split}/reconstruction": wandb.Image(str(path)), "epoch": epoch + 1}, step=self.global_step)
        return path

    def train(
        self,
        epochs: int,
        restart: bool = False,
        restart_dir: str = "",
        additional_epochs: Optional[int] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> dict:
        self.optimizer = Adam(self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        if restart:
            self._load_checkpoint(restart_dir)

        if restart and additional_epochs is not None:
            end_epoch = self.start_epoch + int(additional_epochs)
        else:
            end_epoch = int(epochs)

        train_loader = self._make_loader(self.train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = self._make_loader(self.dev_dataset, batch_size=batch_size, shuffle=False)

        self._log_timestamp(
            "train_start",
            start_epoch=self.start_epoch + 1,
            end_epoch=end_epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        train_losses = []
        validation_losses = []
        final_metrics = {}
        for epoch in tqdm(range(self.start_epoch, end_epoch), desc="vae epochs"):
            started = time.time()
            train_metrics = self._run_epoch(train_loader, epoch, train=True)
            with torch.no_grad():
                validation_metrics = self._run_epoch(dev_loader, epoch, train=False)
            duration = time.time() - started

            metrics = {
                **{f"train/{name}": value for name, value in train_metrics.items()},
                **{f"validation/{name}": value for name, value in validation_metrics.items()},
                "epoch": epoch + 1,
                "epoch_duration_seconds": duration,
                "kl_weight": self._kl_weight(epoch),
            }
            final_metrics = metrics
            train_losses.append(metrics["train/loss"])
            validation_losses.append(metrics["validation/loss"])
            np.savetxt(self.results_folder / "loss_epoch.txt", np.asarray(train_losses))
            np.savetxt(self.results_folder / "validation_loss_epoch.txt", np.asarray(validation_losses))

            is_best = metrics["validation/loss"] < self.best_validation_loss
            if is_best:
                self.best_validation_loss = metrics["validation/loss"]
            self._save_checkpoint(epoch, metrics, is_best=is_best)
            self._append_history({"timestamp": now_iso(), **metrics, "best_validation_loss": self.best_validation_loss})
            self._log_timestamp(
                "epoch_end",
                epoch=epoch + 1,
                train_loss=metrics["train/loss"],
                validation_loss=metrics["validation/loss"],
                duration_seconds=duration,
            )

            _wandb_log(metrics, step=self.global_step)
            if self.sample_interval > 0 and (epoch + 1) % self.sample_interval == 0:
                self._save_reconstruction_plot(dev_loader, epoch, split="validation")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            print(
                "Epoch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}, Time: {:.2f}s".format(
                    epoch + 1, metrics["train/loss"], metrics["validation/loss"], duration
                )
            )

        self._log_artifact(self.results_folder / "ckpt.pth", aliases=["latest"], metadata=final_metrics)
        self._log_timestamp("train_end", best_validation_loss=self.best_validation_loss)
        return final_metrics


__all__ = ["VAETrainer", "parse_channel_multipliers", "now_iso"]
