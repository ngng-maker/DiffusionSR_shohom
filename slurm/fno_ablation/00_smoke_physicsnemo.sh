#!/bin/bash
#SBATCH --job-name=fno_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/fno_ablation/smoke_%j.log
#SBATCH --error=logs/fno_ablation/smoke_%j.err
#
# Milestone A0 — PhysicsNeMo compatibility spike.
#
# Answers three questions before any GPU-hours are spent on training:
#   1. Does physicsnemo import alongside this repo's pinned dependency stack?
#   2. Does the PhysicsNeMo-backed FNO encoder produce the [B, 64, 80, 80] conditioning tensor?
#   3. Does the RRDB regression gate still pass in this environment (i.e. is the baseline intact)?
#
# Submit with:
#   sbatch slurm/fno_ablation/00_smoke_physicsnemo.sh

set -eo pipefail  # Abort on any error and on failures inside pipelines (matches the other scripts here)

export WANDB_ENTITY=ngng-  # Several code paths raise if this is unset, even when W&B is unused

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom  # Repo root on the cluster filesystem
# Conda environment to use. Defaults to the existing project env; override at submit time with
#   sbatch --export=ALL,CONDA_ENV=/trace/group/forgelab/ngng/envs/diffusion_SR_fno <script>
CONDA_ENV="${CONDA_ENV:-/trace/group/forgelab/ngng/envs/diffusion_SR}"

cd "$REPO"                       # Run everything from the repo root so module imports resolve
# A non-interactive SLURM shell has no `conda` shell function until this profile script is sourced;
# calling `conda activate` without it fails with "shell has not been properly configured".
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
echo "Using conda env: $CONDA_ENV"

mkdir -p logs/fno_ablation       # SLURM does not create the log directory itself; missing it silently drops output

echo "=== Environment ==="
python -c "import sys, torch, numpy; print('python', sys.version.split()[0]); print('torch  ', torch.__version__); print('numpy  ', numpy.__version__); print('cuda   ', torch.cuda.is_available())"

echo
echo "=== [1/3] physicsnemo import ==="
# Report availability without aborting: the builtin backend is a supported fallback, so a missing
# physicsnemo is informative rather than fatal at this stage.
python -c "
from diffusionsr.models.fno_encoder_model import PHYSICSNEMO_AVAILABLE
print('PHYSICSNEMO_AVAILABLE =', PHYSICSNEMO_AVAILABLE)
if PHYSICSNEMO_AVAILABLE:
    import physicsnemo
    print('physicsnemo', getattr(physicsnemo, '__version__', 'unknown'))
"

echo
echo "=== [2/3] encoder construction + conditioning shape (both backends) ==="
python -c "
import torch
from diffusionsr.models.encoder_factory import build_encoder
from diffusionsr.models.fno_encoder_model import PHYSICSNEMO_AVAILABLE

# SS316L 4x geometry: 20x20 LR in, 80x80 HR out, single temperature field.
sample = torch.randn(2, 1, 20, 20).cuda()

backends = ['builtin'] + (['physicsnemo'] if PHYSICSNEMO_AVAILABLE else [])
for backend in backends:
    enc = build_encoder('fno', upscale_factor=4, in_channels=1, out_channels=1,
                        encoder_kwargs=dict(feature_channels=64, latent_channels=64,
                                            num_fno_layers=4, num_fno_modes=[14, 13],
                                            padding=8, upsample_mode='pre', backend=backend)).cuda().eval()
    with torch.no_grad():
        feats = enc.conditioning_features(sample)   # What the U-Net consumes as x_e
        pred  = enc(sample)                          # What the L1 pretraining loss targets
    n = sum(p.numel() for p in enc.parameters())
    assert feats.shape == (2, 64, 80, 80), f'bad conditioning shape {feats.shape}'
    assert pred.shape  == (2, 1, 80, 80),  f'bad output shape {pred.shape}'
    assert torch.isfinite(feats).all() and torch.isfinite(pred).all(), 'non-finite output'
    print(f'  {backend:<12} OK  params={n:,}  x_e={tuple(feats.shape)}  out={tuple(pred.shape)}')

rrdb = build_encoder('rrdb', upscale_factor=4, in_channels=1, out_channels=1).cuda().eval()
print(f'  {\"rrdb\":<12} OK  params={sum(p.numel() for p in rrdb.parameters()):,}')
"

echo
echo "=== [3/3] regression gate — RRDB baseline must be unchanged ==="
# The critical check. If this fails, the encoder refactor altered the conditioning path and every
# RRDB-vs-FNO comparison is invalid. -x stops at the first failure so it is obvious in the log.
python -m pytest tests/test_fno_encoder.py -v -x

echo
echo "=== A0 SMOKE TEST COMPLETE ==="
