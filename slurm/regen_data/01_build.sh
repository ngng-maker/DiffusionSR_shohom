#!/bin/bash
#SBATCH -J regen_build
#SBATCH -p batch
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH -t 08:00:00
#SBATCH -o logs/regen_data/build_%j.log
#
# Build the 2D LR/HR datasets from the 2026 SS316L regen HDF5 release (core family) and their train-split
# statistics, then sanity-check them. CPU only.
#   x4_40to10  HR 10 um 64x64, LR 40 um 16x16 (4x)
#   x2_40to20  HR 20 um 32x32, LR 40 um 16x16 (2x)   cascade stage A
#   x2_20to10  HR 10 um 64x64, LR 20 um 32x32 (2x)   cascade stage B
# One split by (P, V) condition, seed 20260518, shared by all three roots. See regen_2d/DATASET_INFO.md.
#
# Usage, from the repo root on TRACE:  sbatch slurm/regen_data/01_build.sh

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/regen_data

H5=/trace/group/forgelab/ngng/multifield/datasets/ss316l_flow3d_regen_2026-09-03/h5
OUT=/trace/group/forgelab/ngng/multifield/datasets/regen_2d
python -c "import h5py, skimage" || pip install -q h5py

python -m diffusionsr.datasets.preprocessing.build_regen_dataset --h5_root "$H5" --out "$OUT" --workers 8
python -m diffusionsr.datasets.preprocessing.verify_regen_dataset --out "$OUT" \
  --png "$OUT/preview_P490_V300_frame60.png"
du -sh "$OUT"
echo "=== regen_2d built ==="
