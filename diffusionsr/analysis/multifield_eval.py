"""
Multi-field physical-consistency evaluation and overlay generation.

Modes (--mode):
  eval      -- Run the full evaluation loop over the test set, computing IOU / MSE /
               Chamfer between Binary_T and Liqlabel for each (method, field_config).
               Resumes automatically from per-sample progress CSVs on SLURM restarts.
  overlays  -- Generate overlay plots for N_OVERLAY steady-state samples (middle
               timestep per (power, velocity) group) across all trained models.
               Also generates VAE latent-channel heatmaps and reconstruction overlays
               for LDM models.
  both      -- Run eval then overlays (default).

Output layout (one subdirectory per experiment label = method x field_config):
  OUT_DIR/
  ├── {label}/
  │   ├── per_sample.csv         per-sample IOU/MSE/Chamfer (written incrementally)
  │   ├── complete.flag          written when all test samples are done
  │   └── overlays/              prediction overlay PNGs for this experiment
  │       └── overlay_s{idx:04d}_pred.png
  ├── gt_overlays/               GT reference overlays (shared across experiments)
  │   └── overlay_gt_s{idx:04d}.png
  ├── bar_charts/                summary bar-chart PNGs
  │   └── bar_{metric}.png
  └── consistency_summary.csv    aggregated mean metrics table
"""
import argparse
import os
import traceback
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import KDTree

from diffusionsr.datasets.dataset import SimulationXZDataset
from diffusionsr.analysis.analysis_functions import load_encoder, initialize_diffusion
from diffusionsr.runners.train_flow_matching import FlowMatchingModel
from diffusionsr.runners.train_ldm import LDMModel

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ROOT      = "/trace/group/forgelab/ngng/multifield/data_fields"
RUNS_DIR  = "/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom/diffusionsr/runs/direct"
OUT_DIR   = "/trace/group/forgelab/ngng/multifield/eval_results"
T_LIQ     = 1700.0    # SS316L liquidus (K)
LIQ_THR   = 0.5       # liqlabel binarization threshold
N_OVERLAY = 5         # steady-state samples shown in overlay mode
FM_STEPS  = 1000      # Euler steps for FlowMatching
DEVICE    = "cuda"
N_STEPS   = 3

METHODS = {
    'DiffusionSR':  ('diffusionimplicitencoded',  'DDPM',  None),
    'FlowMatching': ('flowmatchingimplicitencoded','euler', FM_STEPS),
    'UncondDiff':   ('uncond_diffusion',           'DDPM',  None),
    'LDM':          ('ldmimplicitencoded',          'DDPM',  None),
}

FIELD_CONFIGS = {
    'temp':     ['temperature'],
    'liqlabel': ['liqlabel'],
    'both':     ['temperature', 'liqlabel'],
}

# ── PATH HELPERS ───────────────────────────────────────────────────────────────

def _enc_dir(tag):
    return os.path.join(RUNS_DIR, 'encoder', f'cs_{tag}_n3')

def _run_dir(subdir, tag):
    return os.path.join(RUNS_DIR, subdir, f'cs_{tag}_n3')

def _exp_dir(label):
    """Per-experiment output directory."""
    d = os.path.join(OUT_DIR, label)
    os.makedirs(d, exist_ok=True)
    return d

# ── DATASET BUILDER ───────────────────────────────────────────────────────────

def make_datasets(field_names):
    kw = dict(downscale_method='direct', root_folder=ROOT,
              normalize='standardize', n_steps=N_STEPS, field_names=field_names)
    return (SimulationXZDataset(split='train', **kw),
            SimulationXZDataset(split='dev',   **kw),
            SimulationXZDataset(split='test',  **kw))

# ── MODEL LOADERS ──────────────────────────────────────────────────────────────

def load_diffusionsr(tag, field_names, train_ds, dev_ds, test_ds):
    enc_dir  = _enc_dir(tag)
    diff_dir = _run_dir('diffusionimplicitencoded', tag)
    lr_enc   = load_encoder(enc_dir, train_ds)
    model    = initialize_diffusion(diff_dir, enc_dir, (train_ds, dev_ds, test_ds),
                                    timesteps=1000, conditioning='implicit',
                                    encoding=True, schedule='linear', device=DEVICE)
    return model, lr_enc

def load_flowmatch(tag, field_names, train_ds, dev_ds, test_ds):
    enc_dir  = _enc_dir(tag)
    diff_dir = _run_dir('flowmatchingimplicitencoded', tag)
    model = FlowMatchingModel(
        results_folder=diff_dir, lr_encoder_folder=enc_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        timesteps=1000, conditioning='implicit', encoding=True,
        schedule='linear', device=DEVICE, enc_output=False)
    model.load_saved_model()
    return model, model.lr_enc

def load_uncond(tag, field_names, train_ds, dev_ds, test_ds):
    from diffusionsr.runners.train_diffusion import DiffusionModel
    diff_dir = _run_dir('uncond_diffusion', tag)
    model = DiffusionModel(
        results_folder=diff_dir, lr_encoder_folder=diff_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        timesteps=1000, conditioning='none', encoding=False,
        schedule='linear', device=DEVICE, enc_output=False)
    model.load_saved_model()
    return model, None

def load_ldm(tag, field_names, train_ds, dev_ds, test_ds):
    enc_dir  = _enc_dir(tag)
    diff_dir = _run_dir('ldmimplicitencoded', tag)
    vae_dir  = _run_dir('vae', tag)
    model = LDMModel(
        vae_folder=vae_dir,
        results_folder=diff_dir, lr_encoder_folder=enc_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        timesteps=1000, conditioning='implicit', encoding=True,
        schedule='linear', device=DEVICE)
    model.load_saved_model()
    return model, model.lr_enc

_LOADERS = {
    'DiffusionSR':  load_diffusionsr,
    'FlowMatching': load_flowmatch,
    'UncondDiff':   load_uncond,
    'LDM':          load_ldm,
}

def model_checkpoint_exists(method_name, tag):
    _, subdir, _ = METHODS[method_name]
    ckpt = os.path.join(_run_dir(subdir, tag), 'ckpt.pth')
    if method_name == 'LDM':
        vae_ckpt = os.path.join(_run_dir('vae', tag), 'vae_best.pth')
        return os.path.exists(ckpt) and os.path.exists(vae_ckpt)
    return os.path.exists(ckpt)

# ── INFERENCE ─────────────────────────────────────────────────────────────────

def get_raw_gt(test_ds, idx):
    """Load raw HR npy, return (T_gt, liq_gt) in physical units."""
    raw = np.load(test_ds.hr_paths[idx], allow_pickle=True)
    if len(raw.shape) == 2:
        raw = raw[:, :, None]
    raw = np.clip(raw, None, 8000)
    T_gt   = raw[:, :, 0] if raw.shape[-1] >= 1 else None
    liq_gt = raw[:, :, 1] if raw.shape[-1] >= 2 else None
    return T_gt, liq_gt

def infer_predictions(model, lr_enc, test_ds, res, hr, true_lr, upscaled_lr, sampler, skip):
    """Run one inference step; return (T_pred, liq_pred) in physical units."""
    fn      = test_ds.field_names
    has_T   = 'temperature' in fn
    has_liq = 'liqlabel'    in fn
    x_e = model.compute_x_e(true_lr, upscaled_lr) if lr_enc is not None else None
    with torch.no_grad():
        if sampler == 'euler':
            samples = model.batch_sample(dataset=test_ds, batch=hr.to(DEVICE),
                                         x_e=x_e, sampler='euler', n_steps=skip)
        elif sampler == 'DDPM':
            samples = model.batch_sample(dataset=test_ds, batch=hr.to(DEVICE),
                                         x_e=x_e, sampler='DDPM')
        else:
            samples = model.batch_sample(dataset=test_ds, batch=hr.to(DEVICE),
                                         x_e=x_e, sampler='DDIM', skip=skip)
    pred_norm = samples[-1].cpu().numpy()
    pred_phys = test_ds.unscale_data(pred_norm[0], input_type='hr')
    T_pred   = pred_phys[fn.index('temperature')] if has_T   else None
    liq_pred = pred_phys[fn.index('liqlabel')]    if has_liq else None
    return T_pred, liq_pred

# ── METRIC HELPERS ─────────────────────────────────────────────────────────────

def boundary_pixels(binary_mask):
    eroded = binary_erosion(binary_mask, structure=np.ones((3, 3)))
    return np.argwhere(binary_mask & ~eroded)

def chamfer_distance(mask_a, mask_b):
    b_a, b_b = boundary_pixels(mask_a), boundary_pixels(mask_b)
    if len(b_a) == 0 or len(b_b) == 0:
        return float('nan')
    return float((KDTree(b_b).query(b_a)[0].mean() + KDTree(b_a).query(b_b)[0].mean()) / 2)

def consistency_metrics(bin_t, bin_liq):
    inter = (bin_t & bin_liq).sum()
    union = (bin_t | bin_liq).sum()
    iou   = inter / union if union > 0 else float('nan')
    mse   = float(np.mean((bin_t.astype(float) - bin_liq.astype(float)) ** 2))
    return iou, mse, chamfer_distance(bin_t, bin_liq)

# ── PLOT HELPERS ──────────────────────────────────────────────────────────────

def overlay_plot(T_bg, T_bin, liq_bin, title, fpath, smooth_sigma=1.5):
    """Temperature heatmap with smoothed Binary_T (red) and Liqlabel (blue) contours."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    if T_bg is not None:
        ax.imshow(T_bg.T, origin='lower', cmap='jet', vmin=293, vmax=5000, aspect='auto')
    ax.contour(gaussian_filter(T_bin.T.astype(float),   sigma=smooth_sigma),
               levels=[0.5], colors=['red'],  linewidths=[1.5], origin='lower')
    ax.contour(gaussian_filter(liq_bin.T.astype(float), sigma=smooth_sigma),
               levels=[0.5], colors=['blue'], linewidths=[1.5], origin='lower')
    ax.legend(handles=[
        plt.Line2D([0], [0], color='red',  lw=1.5, label='Binary-T boundary'),
        plt.Line2D([0], [0], color='blue', lw=1.5, label='Liqlabel boundary'),
    ], fontsize=6)
    ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()

# ── RESUME HELPERS ─────────────────────────────────────────────────────────────

def _load_progress(progress_path):
    if not os.path.exists(progress_path):
        return set(), [], [], [], [], [], []
    prog = pd.read_csv(progress_path)
    done = set(prog['idx'].tolist())
    return (done,
            prog['iou_pred'].tolist(), prog['mse_pred'].tolist(), prog['cham_pred'].tolist(),
            prog['iou_gt'].tolist(),   prog['mse_gt'].tolist(),   prog['cham_gt'].tolist())

def _append_progress(progress_path, idx, iou, mse, cham, g_iou, g_mse, g_cham):
    row = pd.DataFrame([{
        'idx': idx,
        'iou_pred': iou,  'mse_pred': mse,  'cham_pred': cham,
        'iou_gt':  g_iou, 'mse_gt':  g_mse, 'cham_gt':  g_cham,
    }])
    row.to_csv(progress_path, mode='a', header=not os.path.exists(progress_path), index=False)

# ── STEADY-STATE SAMPLE SELECTION (for overlays mode) ─────────────────────────

def select_steady_state_indices(n=N_OVERLAY):
    """
    Group test set by (power, velocity), pick the middle timestep from each group,
    then return n evenly-spaced (power, velocity) conditions for variety.
    """
    ref_ds = SimulationXZDataset(
        split='test', downscale_method='direct', root_folder=ROOT,
        normalize='standardize', n_steps=N_STEPS,
        field_names=['temperature', 'liqlabel'])

    groups = defaultdict(list)
    for i, path in enumerate(ref_ds.lr_paths):
        try:
            power = int(str(path).split('power')[-1].split('velocity')[0])
            vel   = int(str(path).split('velocity')[-1].split('_')[0].strip('/').strip('\\'))
        except (ValueError, IndexError):
            power, vel = 0, 0
        try:
            timestep = 0.5 * float(str(path).split('_1')[1].split('.npy')[0]) / 100
        except (IndexError, ValueError):
            timestep = 0.0
        groups[(power, vel)].append((timestep, i))

    pv_middle = {}
    for (p, v), items in sorted(groups.items()):
        mid = sorted(items)[len(items) // 2]
        pv_middle[(p, v)] = mid[1]

    pv_keys = sorted(pv_middle.keys())
    if len(pv_keys) >= n:
        step = len(pv_keys) // n
        selected = [pv_keys[i * step] for i in range(n)]
    else:
        selected = pv_keys
    indices = [pv_middle[pv] for pv in selected]
    print(f"Selected {len(indices)} steady-state indices: {indices}")
    print(f"  PV conditions: {selected}")
    return indices

# ── EVAL MODE ─────────────────────────────────────────────────────────────────

def run_eval_loop():
    """
    Full evaluation loop: IOU / MSE / Chamfer for every test sample.
    Saves per-sample CSV incrementally; writes complete.flag when done.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for method_name, (_, sampler, n_inf) in METHODS.items():
        loader_fn = _LOADERS[method_name]
        for cfg_tag, field_names in FIELD_CONFIGS.items():
            label         = f"{method_name}_{cfg_tag}"
            exp_dir       = _exp_dir(label)
            progress_path = os.path.join(exp_dir, 'per_sample.csv')
            complete_flag = os.path.join(exp_dir, 'complete.flag')
            print(f"\n{'='*60}\n{label}\n{'='*60}")

            if os.path.exists(complete_flag):
                print(f"  Already complete — loading from {progress_path}")
                done_set, iou_list, mse_list, cham_list, g_iou_list, g_mse_list, g_cham_list = \
                    _load_progress(progress_path)
                iou_list  = [v for v in iou_list  if not np.isnan(v)]
                mse_list  = [v for v in mse_list  if not np.isnan(v)]
                cham_list = [v for v in cham_list if not np.isnan(v)]
                g_iou_list  = [v for v in g_iou_list  if not np.isnan(v)]
                g_mse_list  = [v for v in g_mse_list  if not np.isnan(v)]
                g_cham_list = [v for v in g_cham_list if not np.isnan(v)]
                if iou_list:
                    rows.append({
                        'method': method_name, 'field_cfg': cfg_tag,
                        'IOU_pred':     np.nanmean(iou_list),
                        'MSE_pred':     np.nanmean(mse_list),
                        'Chamfer_pred': np.nanmean(cham_list),
                        'IOU_gt':       np.nanmean(g_iou_list)  if g_iou_list  else float('nan'),
                        'MSE_gt':       np.nanmean(g_mse_list)  if g_mse_list  else float('nan'),
                        'Chamfer_gt':   np.nanmean(g_cham_list) if g_cham_list else float('nan'),
                    })
                continue

            try:
                train_ds, dev_ds, test_ds = make_datasets(field_names)
                model, lr_enc = loader_fn(cfg_tag, field_names, train_ds, dev_ds, test_ds)
            except Exception as e:
                print(f"  SKIP (load failed): {e}")
                traceback.print_exc()
                continue

            indices = list(range(len(test_ds)))
            done_set, iou_list, mse_list, cham_list, g_iou_list, g_mse_list, g_cham_list = \
                _load_progress(progress_path)
            if done_set:
                print(f"  Resuming: {len(done_set)}/{len(indices)} samples already done")

            overlay_dir     = os.path.join(exp_dir, 'overlays')
            gt_overlay_dir  = os.path.join(OUT_DIR, 'gt_overlays')
            os.makedirs(overlay_dir, exist_ok=True)
            os.makedirs(gt_overlay_dir, exist_ok=True)
            gt_overlay_saved = bool(done_set)

            for i, idx in enumerate(indices):
                if idx in done_set:
                    continue

                batch = test_ds[idx]
                res, hr, true_lr, upscaled_lr = [
                    torch.tensor(x).unsqueeze(0) for x in batch[:4]]
                T_gt, liq_gt = get_raw_gt(test_ds, idx)

                try:
                    T_pred, liq_pred = infer_predictions(
                        model, lr_enc, test_ds, res, hr, true_lr, upscaled_lr, sampler, n_inf)
                except Exception as e:
                    print(f"  Inference error at sample {idx}: {e}")
                    traceback.print_exc()
                    continue

                T_for_bin   = T_pred   if T_pred   is not None else T_gt
                liq_for_bin = liq_pred if liq_pred is not None else liq_gt
                if T_for_bin is None or liq_for_bin is None:
                    continue

                bin_T_pred   = T_for_bin   > T_LIQ
                bin_liq_pred = liq_for_bin > LIQ_THR
                iou, mse, cham = consistency_metrics(bin_T_pred, bin_liq_pred)
                iou_list.append(iou); mse_list.append(mse); cham_list.append(cham)

                g_iou = g_mse = g_cham = float('nan')
                if T_gt is not None and liq_gt is not None:
                    bin_T_gt   = T_gt   > T_LIQ
                    bin_liq_gt = liq_gt > LIQ_THR
                    g_iou, g_mse, g_cham = consistency_metrics(bin_T_gt, bin_liq_gt)
                    g_iou_list.append(g_iou); g_mse_list.append(g_mse); g_cham_list.append(g_cham)

                _append_progress(progress_path, idx, iou, mse, cham, g_iou, g_mse, g_cham)
                done_set.add(idx)

                if i < N_OVERLAY:
                    T_bg_pred = T_pred if T_pred is not None else T_gt
                    overlay_plot(T_bg_pred, bin_T_pred, bin_liq_pred,
                                 f"{label} s{idx} (pred)",
                                 os.path.join(overlay_dir, f"overlay_s{idx:04d}_pred.png"))
                    if not gt_overlay_saved and T_gt is not None and liq_gt is not None:
                        overlay_plot(T_gt, bin_T_gt, bin_liq_gt,
                                     f"GT s{idx}",
                                     os.path.join(gt_overlay_dir, f"overlay_gt_s{idx:04d}.png"))
                        gt_overlay_saved = True

            iou_list  = [v for v in iou_list  if not np.isnan(v)]
            mse_list  = [v for v in mse_list  if not np.isnan(v)]
            cham_list = [v for v in cham_list if not np.isnan(v)]
            g_iou_list  = [v for v in g_iou_list  if not np.isnan(v)]
            g_mse_list  = [v for v in g_mse_list  if not np.isnan(v)]
            g_cham_list = [v for v in g_cham_list if not np.isnan(v)]

            if not iou_list:
                print("  No valid samples — skipping (all inference failed)")
                continue

            rows.append({
                'method': method_name, 'field_cfg': cfg_tag,
                'IOU_pred':     np.nanmean(iou_list),
                'MSE_pred':     np.nanmean(mse_list),
                'Chamfer_pred': np.nanmean(cham_list),
                'IOU_gt':       np.nanmean(g_iou_list)  if g_iou_list  else float('nan'),
                'MSE_gt':       np.nanmean(g_mse_list)  if g_mse_list  else float('nan'),
                'Chamfer_gt':   np.nanmean(g_cham_list) if g_cham_list else float('nan'),
            })
            print(f"  IOU={rows[-1]['IOU_pred']:.4f}  MSE={rows[-1]['MSE_pred']:.4f}  "
                  f"Chamfer={rows[-1]['Chamfer_pred']:.2f}  (n={len(iou_list)})")
            open(complete_flag, 'w').close()

    # Aggregate summary
    df       = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'consistency_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")
    print(df.to_string(index=False))

    # Bar charts
    bar_dir = os.path.join(OUT_DIR, 'bar_charts')
    os.makedirs(bar_dir, exist_ok=True)
    colors = ['#4472C4', '#ED7D31', '#A9D18E']
    for metric, mlabel in [
        ('IOU_pred',     'IOU (↑ = more consistent)'),
        ('MSE_pred',     'MSE (↓ = more consistent)'),
        ('Chamfer_pred', 'Chamfer distance [px] (↓ = more consistent)'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        x     = np.arange(len(METHODS))
        width = 0.25
        for i, cfg_tag in enumerate(FIELD_CONFIGS):
            vals = [
                df[(df.method == m) & (df.field_cfg == cfg_tag)][metric].values[0]
                if len(df[(df.method == m) & (df.field_cfg == cfg_tag)]) > 0 else float('nan')
                for m in METHODS
            ]
            ax.bar(x + (i - 1) * width, vals, width, label=cfg_tag,
                   color=colors[i], alpha=0.85)
        if metric == 'IOU_pred' and 'IOU_gt' in df.columns:
            gt_mean = df['IOU_gt'].dropna().mean()
            if not np.isnan(gt_mean):
                ax.axhline(gt_mean, color='black', ls='--', lw=1.5, label='GT reference')
        ax.set_xticks(x)
        ax.set_xticklabels(list(METHODS.keys()), rotation=15)
        if metric != 'IOU_pred':
            ax.set_yscale('log')
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_title(mlabel, fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(bar_dir, f"bar_{metric}.png"), bbox_inches='tight')
        plt.close()

    print(f"\nAll eval outputs saved to {OUT_DIR}/")

# ── OVERLAYS MODE ─────────────────────────────────────────────────────────────

def run_overlays():
    """
    Generate overlay PNGs for N_OVERLAY steady-state samples across all trained models.
    Also generates VAE latent-channel heatmaps and reconstruction overlays for LDM.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_indices = select_steady_state_indices()

    # GT overlays (using 'both' field dataset for raw GT)
    gt_overlay_dir = os.path.join(OUT_DIR, 'gt_overlays')
    os.makedirs(gt_overlay_dir, exist_ok=True)
    _, _, ref_test_ds = make_datasets(['temperature', 'liqlabel'])
    print("\nGenerating GT overlays...")
    for idx in sample_indices:
        T_gt, liq_gt = get_raw_gt(ref_test_ds, idx)
        if T_gt is not None and liq_gt is not None:
            overlay_plot(T_gt, T_gt > T_LIQ, liq_gt > LIQ_THR,
                         f"GT s{idx}",
                         os.path.join(gt_overlay_dir, f"overlay_gt_s{idx:04d}.png"))
            print(f"  Saved GT overlay s{idx}")

    # Prediction overlays
    for method_name, (_, sampler, n_inf) in METHODS.items():
        loader_fn = _LOADERS[method_name]
        for cfg_tag, field_names in FIELD_CONFIGS.items():
            label = f"{method_name}_{cfg_tag}"

            if not model_checkpoint_exists(method_name, cfg_tag):
                print(f"\nSKIP {label} (no checkpoint)")
                continue

            print(f"\n{'='*50}\n{label}\n{'='*50}")
            try:
                train_ds, dev_ds, test_ds = make_datasets(field_names)
                model, lr_enc = loader_fn(cfg_tag, field_names, train_ds, dev_ds, test_ds)
            except Exception as e:
                print(f"  SKIP (load failed): {e}")
                traceback.print_exc()
                continue

            overlay_dir = os.path.join(_exp_dir(label), 'overlays')
            os.makedirs(overlay_dir, exist_ok=True)

            for idx in sample_indices:
                batch = test_ds[idx]
                res, hr, true_lr, upscaled_lr = [
                    torch.tensor(x).unsqueeze(0) for x in batch[:4]]
                T_gt, liq_gt = get_raw_gt(test_ds, idx)

                try:
                    T_pred, liq_pred = infer_predictions(
                        model, lr_enc, test_ds, res, hr, true_lr, upscaled_lr, sampler, n_inf)
                except Exception as e:
                    print(f"  Inference error at sample {idx}: {e}")
                    traceback.print_exc()
                    continue

                T_for_bin   = T_pred   if T_pred   is not None else T_gt
                liq_for_bin = liq_pred if liq_pred is not None else liq_gt
                if T_for_bin is None or liq_for_bin is None:
                    continue

                fpath = os.path.join(overlay_dir, f"overlay_s{idx:04d}_pred.png")
                overlay_plot(T_pred if T_pred is not None else T_gt,
                             T_for_bin > T_LIQ, liq_for_bin > LIQ_THR,
                             f"{label} s{idx}", fpath)
                print(f"  Saved {os.path.basename(fpath)}")

    # VAE latent + reconstruction overlays (LDM only)
    print("\n" + "=" * 60)
    print("VAE analysis (LDM models only)")
    print("=" * 60)
    for cfg_tag, field_names in FIELD_CONFIGS.items():
        label = f"LDM_{cfg_tag}"
        if not model_checkpoint_exists('LDM', cfg_tag):
            print(f"\nSKIP {label} (no checkpoint)")
            continue

        print(f"\n{label}")
        try:
            train_ds, dev_ds, test_ds = make_datasets(field_names)
            model, _ = load_ldm(cfg_tag, field_names, train_ds, dev_ds, test_ds)
        except Exception as e:
            print(f"  SKIP (load failed): {e}")
            traceback.print_exc()
            continue

        fn      = test_ds.field_names
        has_T   = 'temperature' in fn
        has_liq = 'liqlabel'    in fn
        vae_dir = os.path.join(_exp_dir(label), 'vae')
        os.makedirs(vae_dir, exist_ok=True)

        for idx in sample_indices:
            batch = test_ds[idx]
            _, hr, _, _ = [torch.tensor(x).unsqueeze(0) for x in batch[:4]]
            T_gt, liq_gt = get_raw_gt(test_ds, idx)

            with torch.no_grad():
                mu, _ = model.vae.encode(hr.to(DEVICE).float())
                recon  = model.vae.decode(mu)

            # Latent channel heatmaps
            fig, axes = plt.subplots(1, 4, figsize=(14, 3), dpi=150)
            z = mu[0].cpu().numpy()
            for k, ax in enumerate(axes):
                ch   = z[k]
                vabs = max(abs(ch.min()), abs(ch.max()), 1e-6)
                im   = ax.imshow(ch.T, origin='lower', cmap='RdBu_r',
                                 vmin=-vabs, vmax=vabs, aspect='auto')
                ax.set_title(f'ch {k}', fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis('off')
            fig.suptitle(f'{label} — VAE latent channels  s{idx}', fontsize=9)
            plt.tight_layout()
            latent_path = os.path.join(vae_dir, f"latent_s{idx:04d}.png")
            plt.savefig(latent_path, bbox_inches='tight')
            plt.close()
            print(f"  Saved {os.path.basename(latent_path)}")

            # VAE reconstruction overlay
            recon_phys  = test_ds.unscale_data(recon[0].cpu().numpy(), input_type='hr')
            T_recon     = recon_phys[fn.index('temperature')] if has_T   else None
            liq_recon   = recon_phys[fn.index('liqlabel')]    if has_liq else None
            T_for_bin   = T_recon   if T_recon   is not None else T_gt
            liq_for_bin = liq_recon if liq_recon is not None else liq_gt
            if T_for_bin is not None and liq_for_bin is not None:
                recon_path = os.path.join(vae_dir, f"vae_recon_s{idx:04d}.png")
                overlay_plot(T_recon if T_recon is not None else T_gt,
                             T_for_bin > T_LIQ, liq_for_bin > LIQ_THR,
                             f"{label} VAE recon s{idx}", recon_path)
                print(f"  Saved {os.path.basename(recon_path)}")

    print(f"\nAll overlay outputs saved to {OUT_DIR}/")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['eval', 'overlays', 'both'], default='both',
                        help='eval: full metric sweep; overlays: steady-state overlay PNGs')
    args = parser.parse_args()

    if args.mode in ('eval', 'both'):
        run_eval_loop()
    if args.mode in ('overlays', 'both'):
        run_overlays()
