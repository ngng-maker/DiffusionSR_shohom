# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for "Inexpensive high fidelity melt pool models in additive manufacturing using generative deep diffusion" (*Materials & Design* 245 (2024) 113181). A conditional DDPM, conditioned on a frozen RRDN CNN encoder, super-resolves coarse L-PBF melt-pool simulation cross-sections to high-fidelity ones. This is a research/paper-reproduction codebase, not a packaged application: there is no test suite, linter, or CI — correctness is judged by training/validation loss curves (logged to W&B) and visual/metric comparison of sampled outputs against ground truth.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is the day-to-day install (NumPy capped `<2.0`). `requirements_paper_exact.txt` pins the exact versions used for the paper (Python 3.9, CUDA 11.1, torch 1.10) and will not install on modern GPUs/Python — reference only.

Download the dataset with `bash download_data.sh` (pulls a zip from Google Drive via `gdown`, unpacks to `./data/`). Point a config's `root_folder` at the extracted directory under `./data/`.

W&B is used for experiment tracking. Set `WANDB_ENTITY` in the environment — code reads it via `os.getenv("WANDB_ENTITY")` and several scripts raise if it's unset.

## How code is run

The primary interface is notebooks. The modules under `diffusionsr/runners/` and `diffusionsr/analysis/` are designed to be imported as libraries:

```python
from diffusionsr.runners.train_rrdn_encoder import pretrain_encoder
from diffusionsr.runners.train_diffusion import DiffusionModel, forwardpass
```

Start from `diffusionsr/notebooks/00_demo.ipynb` to see the full load-model → sample → plot flow end to end. The numbered notebooks after it (`01_reproduce_ss316l_2x`, `02_reproduce_ti64_4x`, `03_zero_shot_transfer`, `04_keyhole_frequency_analysis`) reproduce specific paper figures/tables. `diffusionsr/notebooks/archive/` holds appendix-figure notebooks that are not maintained.

`train_srdiff.py` is an argparse CLI that dispatches to all three model types:

```bash
python -m diffusionsr.runners.train_srdiff \
  --config configs/conditioning_ablation/implicit_diffusion_direct_encoder.yml \
  --gpu 0 \
  --modeltype diffusion   # options: diffusion | encoder | mobilenet
```

It reads the config YAML, initializes datasets and DataLoaders, and calls the appropriate training function. Note that `FlowMatchingModel` is not wired into `train_srdiff.py` — it must be instantiated from a notebook or script directly.

`diffusionsr/analysis/streamlined_analysis.py` and `end_to_end_analysis.py` accept `--wandb_id`/`--wandb_entity`/`--wandb_project` and pull a run config from the W&B API rather than a local YAML.

To verify a change manually: run the relevant notebook cells (or `pretrain_encoder` / `DiffusionModel.train`) against a small dataset slice and confirm `*_loss_epoch.txt` decreases and the W&B-logged sample images look reasonable.

## Architecture

The pipeline is two stages, trained separately:

1. **Encoder pretraining** (`runners/train_rrdn_encoder.py: pretrain_encoder`) — an RRDB network (`models/lr_encoder_model.py: rrdbnet_encoder`, a `RRDBNet` with `upscale_factor` of 2/4/8) is trained with plain L1 loss to map the low-fidelity cross-section directly to an approximation of the high-fidelity field. Checkpoints are saved as `model_saved.pth` (running) and `bestmodel_saved.pth` (best val loss) under the run's results directory, and also logged as a W&B artifact.
2. **Conditional diffusion** (`runners/train_diffusion.py: DiffusionModel`) — loads the encoder from step 1, freezes it (`requires_grad_(False)`, `.eval()`), and trains a U-Net (`models/diffusion_model.py: Unet`) to denoise the high-fidelity field. The encoder's output is injected into the U-Net either by addition after the first conv (`conditioning='implicit'`) or by channel-concatenation before the first conv (`conditioning='explicit'`) — see `Unet.forward` in `diffusion_model.py`. `forwardpass()` in `train_diffusion.py` controls whether the encoder's *final* output or an intermediate feature map (`output=False`, used when `enc_output=False`) is passed as the conditioning tensor `x_e`.
3. **Flow matching** (`runners/train_flow_matching.py: FlowMatchingModel`) — a `DiffusionModel` subclass that replaces the DDPM objective with continuous-time flow matching. Inherits the encoder initialization, training loop, checkpointing, and W&B logging unchanged; only the forward process and sampler differ: linear interpolation path `x_t = (1-t)*x_0 + t*noise`, velocity-field prediction target `noise - x_0`, and Euler ODE integration at sampling time (`euler_sample`, from `t=1` noise to `t=0` data). The `timesteps` config key is repurposed as the time-embedding scale constant (`fm_timescale`), not a discretization count; the Euler step count is `fm_n_steps` (a separate config key). `self.betas = None` guards against accidental DDPM code paths. Conditioning design is identical to `DiffusionModel` (frozen RRDB encoder, `conditioning='implicit'`), deliberately isolating "DDPM vs. flow matching" as the only variable.

Other runners: `train_mobilenet.py` (MobileNetV2 SISR baseline), `train_rrdn_encoder_cross_entropy.py` (encoder variant with cross-entropy loss for the `liqlabel` field).

### Data (`diffusionsr/datasets/dataset.py: SimulationXZDataset`)

Each sample is a 2D (x, z) cross-section through the laser plane of travel, stored as paired `.npy` arrays in:
```
<root_folder>/<split>/HR/...                          # high-fidelity, ground truth
<root_folder>/<split>/LR/<downscale_method>/1x/...     # low-fidelity, native resolution
<root_folder>/<split>/LR/<downscale_method>/<factor>x/...  # low-fidelity, bicubic-upscaled to HR resolution
```
`split` is `train`/`dev`/`test`; `downscale_method` selects how the LR data was produced (e.g. `direct`, `lanczos` — see `configs/conditioning_ablation/`); the upscale `factor` is inferred from the HR/LR shape ratio. `__getitem__` returns `(residual, hr, true_lr, upscaled_lr)` tensors (plus `info` = `[power, velocity, timestep]` if `return_info=True`), all channel-first.

Per-channel mean/std (or min/max) normalization statistics are computed once from the **train** split and cached to `<root_folder>/statistics/<downscale_method>/*.npy` with a `flag` sentinel file; `dev`/`test` splits require this cache to already exist (`compute_statistics` raises otherwise). Two normalization modes are supported throughout (`dataset.normalize`, also threaded through model/runner code as `normalize`/`maintain_torch` args): `'standardize'` (zero mean/unit std) and `'rescaling'` (min/max to `[-1, 1]`). `unscale_data`/`rescale_data` convert between normalized and physical units and must be passed the correct `input_type` (`'hr'`, `'lr'`, `'upscaled_lr'`, or `'residual'`) since each has separately cached stats.

`n_steps`/`out_steps` support stacking multiple prior simulation timesteps as extra input channels (used by the time-series ablation notebook); for single-timestep configs these both default to 1. `field_names` selects which physical fields are used (`temperature`, and `liqlabel` for SS316L data); `THRESHOLD_T = 8000` K clips numerical noise spikes before normalization.

### Diffusion details

Beta/variance schedules (`linear`, `quadratic`, `sigmoid`, `cosine`) are implemented redundantly in `runners/train_diffusion.py`, `models/diffusion_model.py`, `analysis/sampling.py`, and inline inside several `analysis/analysis_functions.py` prediction functions — when changing a schedule's math, update all copies or you'll get train/inference mismatch. `timesteps` is typically 1000 for training.

Sampling supports both DDPM (`DiffusionModel.p_sample_loop`, one reverse step per timestep) and DDIM (`DiffusionModel.batch_sample(sampler='DDIM', skip=...)` and the standalone `predict_streamlined_ddim_diffusion` in `analysis/sampling.py`), where `skip` trades sampling speed for fidelity (paper reports up to 50x speedup with limited quality loss).

### Configs (`diffusionsr/configs/*.yml`)

Plain YAML, read by ad hoc notebook/script code (no schema/validation) — keys include `root_folder`, `downscale_method`, `conditioning` (`implicit`/`explicit`), `schedule`, `fields`, `timesteps`, `batch_size`, `epochs`, `learning_rate`, `encoder_results_dir`/`restart_dir`. Subdirectories group ablations: `conditioning_ablation/` (implicit vs explicit × direct vs lanczos downscaling × with/without encoder), `loss_type/` (l1/l2/huber), `simulation_organized/` (learning-rate/step-count variants).

Flow matching configs add `fm_n_steps` (Euler integration step count at sampling time); `timesteps` in those configs sets the time-embedding scale, not the diffusion chain length.

W&B run configs store absolute paths from the original training machine. `diffusionsr/utils.py: relocate_config_paths` rewrites `root_folder` to `<repo>/data/<basename>` and re-roots `encoder_results_dir`/`restart_dir` under `diffusionsr/` at the `runs/` segment — use this when loading an old W&B config on a new checkout instead of editing paths by hand.

### Analysis (`diffusionsr/analysis/`)

Given trained checkpoints, `analysis_functions.py` provides `load_diffusion`/`load_encoder`/`initialize_diffusion` to reconstruct models from a results directory, and `predict_*` wrappers (`predict_lrenc`, `predict_refactored_diffusion`, `predict_ddim_diffusion`, `predict_mobilenet`, ...) that all share the calling convention `(model[, lr_enc], res, hr, lr, upscaled_lr, dataset, ...) -> (input_in_physical_units, prediction_in_physical_units, target_in_physical_units)`. `get_profile()` extracts the melt-pool/keyhole depth profile (first/last pixel above 1900 K per column) used for the MP-MAE/keyhole metrics; `metrics.py` has `PSNR`/`SSIM`. `plotting_functions.py` holds shared matplotlib styling (`frame_tick`, `legend`, `savefig`).

`FlowMatchingModel.batch_sample()` is API-compatible with `DiffusionModel.batch_sample()` — existing analysis code works with both classes interchangeably; pass `sampler='euler'` for flow matching (only supported sampler; others raise `NotImplementedError`).

### Preprocessing (`diffusionsr/datasets/preprocessing/*.ipynb`)

Notebooks that build the paired LR/HR dataset from raw FLOW-3D simulation output (not needed if using the pre-built dataset from `download_data.sh`).
