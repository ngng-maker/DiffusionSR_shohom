"""
Plot training and validation loss curves from a DiffusionSR run directory.

Reads whatever loss files exist and plots them together.  Works with all
runner types in this repo:

  DiffusionModel / FlowMatchingModel / UncondDiff
    {run_dir}/{prefix}loss_epoch.txt
    {run_dir}/{prefix}validation_loss_epoch.txt

  RRDB encoder  (train_rrdn_encoder.py)
    {run_dir}/train_loss.txt
    {run_dir}/test_loss.txt

  LDM VAE pretraining  (train_ldm.py pretrain_vae)
    {run_dir}/vae_train_loss.txt
    {run_dir}/vae_val_loss.txt

  LatentDiff / VAETrainer  (train_latentdiff.py / train_vae.py)
    {run_dir}/history.csv  (preferred)
    {run_dir}/loss_epoch.txt  (fallback)
    {run_dir}/validation_loss_epoch.txt

Usage (TRACE / any shell):
  cd /path/to/repo
  /path/to/envs/diffusion_SR/bin/python -m diffusionsr.runners.plot_training_curves \\
      --run_dir /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom/diffusionsr/runs/direct/diffusionimplicitencoded/cs_both_n3 \\
      --out loss_curves.png

  # Monitor a live run (poll every 60 s):
  while true; do python -m diffusionsr.runners.plot_training_curves --run_dir RUN_DIR --out curves.png; sleep 60; done
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_txt(path: Path):
    """Load a savetxt file; return 1-D float array or None."""
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(str(path))
        arr = arr.squeeze() if arr.ndim > 1 else arr
        return np.atleast_1d(arr)
    except Exception:
        return None


def _load_history_csv(path: Path):
    """Load history.csv written by LatentDiffusionTrainer / VAETrainer.

    Returns dict mapping column_name -> list of float values (one per epoch).
    """
    if not path.exists():
        return {}
    rows = []
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    out = {}
    for row in rows:
        for k, v in row.items():
            try:
                out.setdefault(k, []).append(float(v))
            except (ValueError, TypeError):
                pass
    return out


def collect_curves(run_dir: Path):
    """Return a list of (label, values_array) tuples for all loss series found."""
    curves = []

    # ── DiffusionModel / FlowMatching / UncondDiff ────────────────────────────
    for prefix in ("", "evaluation"):
        train_txt = run_dir / f"{prefix}loss_epoch.txt"
        val_txt   = run_dir / f"{prefix}validation_loss_epoch.txt"
        t = _load_txt(train_txt)
        v = _load_txt(val_txt)
        if t is not None:
            curves.append((f"Train loss{' (eval prefix)' if prefix else ''}", t))
        if v is not None:
            curves.append((f"Val loss{' (eval prefix)' if prefix else ''}", v))

    # ── RRDB encoder ──────────────────────────────────────────────────────────
    enc_train = _load_txt(run_dir / "train_loss.txt")
    enc_test  = _load_txt(run_dir / "test_loss.txt")
    if enc_train is not None:
        curves.append(("Encoder train loss", enc_train))
    if enc_test is not None:
        curves.append(("Encoder test loss", enc_test))

    # ── LDM VAE pretraining ────────────────────────────────────────────────────
    vae_train = _load_txt(run_dir / "vae_train_loss.txt")
    vae_val   = _load_txt(run_dir / "vae_val_loss.txt")
    if vae_train is not None:
        curves.append(("VAE train loss", vae_train))
    if vae_val is not None:
        curves.append(("VAE val loss", vae_val))

    # ── history.csv (LatentDiff / VAETrainer) ─────────────────────────────────
    history = _load_history_csv(run_dir / "history.csv")
    for key in ("train/loss", "validation/loss", "train/reconstruction_loss",
                "validation/reconstruction_loss", "train/kl_loss", "validation/kl_loss"):
        if key in history:
            curves.append((key, np.array(history[key])))

    return curves


# ── main ──────────────────────────────────────────────────────────────────────

def plot_curves(run_dir, out_path=None, log_scale=False, title=None):
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    curves = collect_curves(run_dir)
    if not curves:
        print(f"No loss files found in {run_dir}")
        return

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    colors = plt.cm.tab10.colors
    for i, (label, values) in enumerate(curves):
        epochs = np.arange(1, len(values) + 1)
        ls = "--" if "val" in label.lower() or "test" in label.lower() else "-"
        ax.plot(epochs, values, ls=ls, color=colors[i % len(colors)],
                label=label, linewidth=1.6, alpha=0.9)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss (log scale)" if log_scale else "Loss", fontsize=10)
    ax.set_title(title or f"Training curves — {run_dir.name}", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(out_path) if out_path else run_dir / "loss_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # Also print final values
    print("\nFinal loss values:")
    for label, values in curves:
        print(f"  {label:45s}  {values[-1]:.6f}  (epoch {len(values)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True,
                        help="Run directory containing loss .txt files / history.csv")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: <run_dir>/loss_curves.png)")
    parser.add_argument("--log", action="store_true",
                        help="Use log scale for the y-axis")
    parser.add_argument("--title", default=None, help="Plot title override")
    args = parser.parse_args()
    plot_curves(args.run_dir, out_path=args.out, log_scale=args.log, title=args.title)


if __name__ == "__main__":
    main()
