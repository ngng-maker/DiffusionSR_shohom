# Pipeline Standards

This project treats runners, models, logging, and artifacts as one contract. New training pipelines should be easy to launch from the same CLI shape, easy to resume, and easy to inspect later from only the run directory.

## CLI Contract

- Keep runnable entry points under `diffusionsr/runners/`.
- Preserve the common arguments: `--config`, `--gpu`, `--modeltype`, `--restart_dir`, and, for restartable trainers, `--additional_epochs`.
- Read YAML configs from `diffusionsr/configs/` when runners are launched from `diffusionsr/`, matching the existing diffusion runners.
- Route model families through `--modeltype` values such as `diffusion`, `vae2d`, and `vae3d` when they share a runner. Standalone downstream runners, such as `train_latentdiff.py`, should still preserve `--config`, `--gpu`, `--restart_dir`, and `--additional_epochs`.
- Use `getattr(new_config, "optional_key", default)` for optional config values. Avoid membership checks on `argparse.Namespace` objects.
- Keep run folders deterministic and timestamped: `runs/<downscale_method>/<model_family>/<timestamp>/<normalize_method>/n_steps_<n_steps>`.
- On restart, write into `--restart_dir` and load `ckpt.pth` first, falling back to `best_model.pth` only when appropriate.

## Config Keys

Common dataset/training keys should keep their current names: `root_folder`, `downscale_method`, `n_steps`, `fields`, `normalize_method`, `batch_size`, `epochs`, and `learning_rate`.

VAE-specific keys should use the `vae_` prefix:

- `vae_target`: one of `hr`, `residual`, `lr`, `true_lr`, or `upscaled_lr`.
- `vae_spatial_dims`: `2` or `3`.
- `vae_input_channels` and `vae_output_channels`: optional overrides; otherwise infer from the dataset.
- `vae_latent_channels`: latent channel count used by downstream latent diffusion.
- `vae_hidden_channels`: base channel width.
- `vae_channel_multipliers`: encoder/decoder depth, for example `[1, 2, 4]`.
- `vae_beta`: KL weight.
- `vae_reconstruction_loss`: `l1`, `l2`, or `huber`.
- `vae_kl_anneal_epochs`: optional KL warmup.
- `vae_log_interval`, `vae_sample_interval`, and `vae_save_every`: logging and checkpoint cadence.
- `vae_weight_decay`, `vae_grad_clip_norm`, and `vae_num_workers`: optimizer/runtime knobs.
- `vae_output_activation`: optional `tanh` or `sigmoid`; leave unset for standardized fields.

Latent-diffusion-specific keys should use the `latent_` prefix:

- `latent_input_encoder_dir` and `latent_target_encoder_dir`: VAE run directories or checkpoint files.
- `latent_input_encoder_checkpoint` and `latent_target_encoder_checkpoint`: usually `best_model.pth`.
- `latent_input_type` and `latent_target_type`: dataset tuple choices such as `lr`, `upscaled_lr`, `hr`, or `residual`.
- `latent_training_mode`: `diffusion` for the denoiser, or `nn` for direct latent mapping logic tests.
- `latent_timesteps`: forward diffusion/noising chain length used during training.
- `latent_schedule`: `linear` or `cosine`.
- `latent_sampler`: reverse sampler for previews/generation, `DDPM` or `DDIM`.
- `latent_sample_timesteps`: reverse sampling step count. Use the full `latent_timesteps` for exact DDPM; DDIM can use fewer steps for faster previews.
- `latent_ddim_eta`: `0.0` for deterministic DDIM; positive values add stochasticity.
- `latent_hidden_channels`, `latent_num_blocks`, `latent_loss_type`, and latent optimizer/logging keys should mirror the VAE naming pattern.

3D configs should also carry `inflate_dim` and `inflate_method` while the existing dataset represents fake depth by repeated channels. When real 3D data is wired in, prefer an explicit depth axis and keep the model input as `(batch, channels, depth, height, width)`.

## Dataset And Shape Contract

- 2D models consume tensors shaped `(batch, channels, height, width)`.
- 3D models consume tensors shaped `(batch, channels, depth, height, width)`.
- Legacy inflated-depth datasets may emit `(batch, channels * depth, height, width)`; the trainer may reshape this only when `depth_size` is explicit and divisible.
- Real 3D datasets should not hide depth inside channels. Emit 5D tensors directly, or expose a small adapter with a clearly named reshape step.
- Dataset tuple order should remain `(residual, hr, true_lr, upscaled_lr)` unless a new dataset wrapper documents a different contract.
- Field and timestep channel order should be stable across training, checkpoint loading, and inference notebooks.

## Model Wiring

- Export public models from `diffusionsr/models/__init__.py`.
- Model constructors should be parametric in input channels, output channels, latent channels, hidden width, and depth/multipliers.
- Prefer simple forward outputs with named fields. VAE forward output should expose `reconstruction`, `mu`, `logvar`, and `z`.
- Keep reusable loss functions next to the model when they are model-specific and small.
- Checkpoint payloads must include `model_config` so notebooks and downstream latent diffusion code can rebuild models without guessing architecture values.

## Logging And Artifacts

Every trainable pipeline should write these local files into its run directory:

- `configuration.yml`: copy of the launch config.
- `information.txt`: short human-readable run summary.
- `timestamps.log`: JSON lines for `initialized`, `train_start`, `epoch_end`, `best_model`, `restart`, and `train_end` events.
- `history.csv`: per-epoch metrics with timestamps.
- `loss_epoch.txt` and `validation_loss_epoch.txt`: simple numeric arrays for quick plotting.
- `ckpt.pth`: latest checkpoint.
- `best_model.pth`: best validation checkpoint for downstream tasks.
- `vae_metadata.json` or equivalent metadata for model/trainer configuration.
- `images/`: preview figures keyed by split and epoch.

WandB integration should follow the existing project pattern:

- Use project `Flow3D_SuperResolution` unless a config explicitly overrides it.
- Use `entity=os.getenv("WANDB_ENTITY")`.
- Pass the combined CLI and YAML config into `wandb.init`.
- Log scalar metrics with stable names such as `train/loss`, `validation/loss`, `train/reconstruction_loss`, and `validation/kl_loss`.
- Log best and latest model checkpoints as artifacts when wandb is active.
- Never make WandB required for local training; local files are the source of truth.

## Checkpoint Semantics

A checkpoint should be a dictionary with at least:

- `epoch`: zero-based epoch index saved.
- `global_step`: optimization step count.
- `model_state_dict` and `optimizer_state_dict`.
- `best_validation_loss`.
- `metrics`: latest epoch metrics.
- `model_config` and `trainer_config`.
- `saved_at`: ISO timestamp.

Resume logic should set `start_epoch = checkpoint["epoch"] + 1`. If `--additional_epochs N` is supplied, train through `start_epoch + N`; otherwise use the configured absolute `epochs` value.

## Inference Notebook Standards

- The notebook should be runnable from either repo root or `diffusionsr/notebooks/`.
- Require the user to set only `RUN_DIR`, `CHECKPOINT_NAME`, and basic display choices.
- Rebuild the model from `checkpoint["model_config"]`.
- Rebuild datasets from the copied `configuration.yml` when possible.
- Display target, reconstruction, absolute error, and latent summaries.
- Support both 2D and 3D outputs, showing a center depth slice for 3D by default.
- Keep export optional and explicit, for example `EXPORT_LATENTS = False`.

## Real 3D Data Migration

The VAE model is ready for true 3D volumes if the dataset emits 5D tensors. The current inflated-depth path is a compatibility bridge for existing fake-depth configs, not a long-term data model. When actual 3D data arrives:

- Add or adapt a dataset that reads volumes with an explicit depth axis.
- Preserve `(residual, hr, true_lr, upscaled_lr)` returns.
- Ensure normalization statistics are computed over real volume arrays, not repeated 2D slices.
- Validate that `input_channels`, `depth`, and field order match the saved `model_config`.
- Add one smoke test that loads a tiny true-volume fixture and confirms the trainer receives `(B, C, D, H, W)` without channel reshaping.
