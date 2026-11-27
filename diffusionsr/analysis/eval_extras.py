"""
Extra evaluation outputs for the SI multi-field / new-dataset / FM studies
(docs/plans/2026-11_si_multifield_fm_plan.md). Everything here is ADDED next to an eval's results.json;
no existing metric is changed, so notebooks 22/24 stay comparable.

  pv_metrics(...)          calibration, spread-skill, CRPS and depth errors per (power, velocity) test condition
  field_metrics(...)       per-field MAE and, for a liquid-label channel, IoU of the liquid region
  model_size(...)          parameter counts (U-Net, encoder) and checkpoint sizes
  benchmark_sampling(...)  timed sampling on one fixed batch (s/sample, samples/s, nfe), after warm-up
  trajectory_gif(...)      one test frame from the start point to the final sample, step by step
  simtime_gif(...)         final predictions over consecutive timesteps of one test condition

All model-facing helpers use the shared sampling interface of the SI and FM models:
    model.sample(upscaled_lr, x_e, lr=..., solver=..., n_steps=..., eps=..., generator=..., callback=...)
        -> (x1 in HR-standardised space, nfe)
with callback(step_index, t, x_state) called along the trajectory.
"""

from __future__ import annotations

import io
import json
import os
import re
import time

import numpy as np

THRESHOLD_K = 1900.0


# ----------------------------------------------------------------------------------------------
# Test-set bookkeeping
# ----------------------------------------------------------------------------------------------

def frame_info(test_ds) -> list[dict]:
    """(power, velocity, timestep, condition) for every test frame, in dataset order.

    Parsed from the LR paths exactly as SimulationXZDataset.__getitem__ does (power<P>velocity<V>/_<t>.npy).
    """
    out = []
    for p in test_ds.lr_paths:
        P = int(p.split('power')[-1].split('velocity')[0])
        V = int(p.split('velocity')[-1].split('_')[0].strip('/').split('/')[0])
        m = re.search(r'_(\d+)\.npy$', os.path.basename(p))
        out.append(dict(P=P, V=V, t=int(m.group(1)) if m else -1, cond=f'P{P}_V{V}'))
    return out


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------

def _pixel_depths(preds, gts, px_um, ch=0):
    from diffusionsr.analysis.evaluate_coupled import _mp_geometry
    N, K = preds.shape[:2]
    dp = np.array([[_mp_geometry(np.asarray(preds[n, k, ch]), px_um)['mp_depth'] for k in range(K)] for n in range(N)])
    dg = np.array([_mp_geometry(np.asarray(gts[n, ch]), px_um)['mp_depth'] for n in range(N)])
    return dp, dg


def _calib(dp, dg, melt):
    if not melt.any():
        return float('nan'), float('nan')
    mu, sd = dp[melt].mean(1), dp[melt].std(1)
    err = np.abs(dg[melt] - mu)
    return float(np.mean(err <= 2 * sd)), float(np.mean(err <= sd))


def pv_metrics(preds, gts, info: list[dict], px_um: float, ch: int = 0) -> dict:
    """Per-(P, V) condition metrics. preds (N, K, C, H, W), gts (N, C, H, W), physical units.

    For each condition: frames, melted frames, mean true MP depth, ensemble MAE, per-sample MP-depth MAE,
    calibration (pixel and sub-pixel, 2 sigma and 1 sigma; same rule as calibration_2sigma), and the
    melt-pool spread, spread-skill ratio and CRPS of evaluate_si_schedules.ensemble_skill.
    """
    from diffusionsr.analysis.add_depth_metrics import mp_depth_subpixel
    from diffusionsr.analysis.evaluate_si_schedules import ensemble_skill, ensemble_spread
    N, K = preds.shape[:2]
    assert len(info) >= N, (len(info), N)
    info = info[:N]
    dp_px, dg_px = _pixel_depths(preds, gts, px_um, ch)
    dp_sp = np.zeros((N, K)); dg_sp = np.zeros(N); melt = np.zeros(N, bool)
    for s in range(0, N, 64):
        e = min(N, s + 64)
        dp_sp[s:e] = mp_depth_subpixel(np.asarray(preds[s:e, :, ch]), px_um)[0]
        dg_sp[s:e], _, melt[s:e] = mp_depth_subpixel(np.asarray(gts[s:e, ch]), px_um)
    conds = sorted({(d['P'], d['V']) for d in info})
    rows = []
    for P, V in conds:
        idx = np.array([i for i, d in enumerate(info) if d['P'] == P and d['V'] == V])
        m = melt[idx]
        c2_px, c1_px = _calib(dp_px[idx], dg_px[idx], m)
        c2_sp, c1_sp = _calib(dp_sp[idx], dg_sp[idx], m)
        p_sub, g_sub = np.asarray(preds[idx]), np.asarray(gts[idx])
        sk = ensemble_skill(p_sub, g_sub, ch=ch)
        sk.update(ensemble_spread(p_sub, g_sub, ch=ch))
        rows.append(dict(
            P=int(P), V=int(V), energy_density_J_per_m=float(P / (V * 1e-3)) if V else float('nan'),
            n_frames=int(len(idx)), n_melt=int(m.sum()),
            true_mp_depth_um=float(dg_sp[idx][m].mean()) if m.any() else 0.0,
            ens_mae_K=float(np.abs(p_sub[:, :, ch].mean(1) - g_sub[:, ch]).mean()),
            mp_mae_px_um=float(np.abs(dp_px[idx] - dg_px[idx][:, None]).mean()),
            mp_mae_subpx_um=float(np.abs(dp_sp[idx] - dg_sp[idx][:, None]).mean()),
            calibration_2sigma=c2_px, calibration_1sigma=c1_px,
            calibration_2sigma_subpx=c2_sp, calibration_1sigma_subpx=c1_sp,
            spread_melt_K=float(sk['spread_melt_px_K']),
            ssr_melt=float(sk['ssr_melt']), crps_melt_K=float(sk['crps_melt_K']),
            ssr_all=float(sk['ssr_all']), crps_all_K=float(sk['crps_all_K'])))
    return dict(threshold_K=THRESHOLD_K, px_um=px_um, n_conditions=len(rows), per_condition=rows)


def field_metrics(preds, gts, field_names: list[str], liquid_field: str | None = None,
                  liquid_rule=None) -> dict:
    """Per-field MAE (per sample and of the ensemble mean). If liquid_field is given, also the IoU of the
    predicted liquid region against the truth, using liquid_rule(array) -> bool mask (set once the
    channel's convention is known: binary label > 0.5, or the sign of a signed distance)."""
    out = {}
    pm = np.asarray(preds).mean(1)
    for c, name in enumerate(field_names):
        out[f'MAE_{name}'] = float(np.abs(np.asarray(preds[:, :, c]) - np.asarray(gts[:, None, c])).mean())
        out[f'MAE_ensemble_mean_{name}'] = float(np.abs(pm[:, c] - np.asarray(gts[:, c])).mean())
    if liquid_field and liquid_field in field_names:
        c = field_names.index(liquid_field)
        rule = liquid_rule or (lambda a: a > 0.5)
        g = rule(np.asarray(gts[:, c]))[:, None]
        p = rule(np.asarray(preds[:, :, c]))
        inter = (p & g).sum(axis=(-1, -2)).astype(float)
        union = (p | g).sum(axis=(-1, -2)).astype(float)
        has = union > 0
        out[f'IoU_liquid_{liquid_field}'] = float((inter[has] / union[has]).mean()) if has.any() else float('nan')
        pe = rule(pm[:, c]); ge = g[:, 0]
        i2, u2 = (pe & ge).sum(axis=(-1, -2)).astype(float), (pe | ge).sum(axis=(-1, -2)).astype(float)
        out[f'IoU_liquid_ensemble_mean_{liquid_field}'] = float((i2[u2 > 0] / u2[u2 > 0]).mean()) if (u2 > 0).any() else float('nan')
    return out


def model_size(model, ckpt_path: str | None = None, enc_ckpt_path: str | None = None) -> dict:
    """Parameter counts of the generative network and the frozen encoder, and checkpoint sizes (MB)."""
    def n_params(m):
        return int(sum(p.numel() for p in m.parameters())) if m is not None else 0
    net = getattr(model, 'model', None)
    enc = getattr(model, 'lr_enc', None)
    out = dict(params_generator=n_params(net), params_encoder=n_params(enc))
    out['params_total'] = out['params_generator'] + out['params_encoder']
    for key, path in (('ckpt_MB', ckpt_path), ('encoder_ckpt_MB', enc_ckpt_path)):
        out[key] = float(os.path.getsize(path) / 2 ** 20) if path and os.path.exists(path) else float('nan')
    return out


def benchmark_sampling(model, test_ds, solver: str, n_steps: int, eps: float, k: int = 10,
                       frames: int = 4, device: str = 'cuda', warmup: int = 1, reps: int = 3,
                       sample_fn=None) -> dict:
    """Time sampling of `frames` test frames x k samples on one fixed batch, after `warmup` runs.

    sample_fn(lr, ulr, x_e, generator) -> (x1, nfe) overrides the default model.sample call (used for the
    DDPM baselines and the cascade). Returns s per sample, s per frame (k samples), samples/s and nfe.
    """
    import torch
    batch = [test_ds[i] for i in range(min(frames, len(test_ds)))]
    hr = torch.stack([b[1] for b in batch]).to(device).float()
    lr = torch.stack([b[2] for b in batch]).to(device).float()
    ulr = torch.stack([b[3] for b in batch]).to(device).float()
    B = hr.shape[0]

    def run(seed):
        g = torch.Generator(device=device).manual_seed(seed)
        if sample_fn is not None:
            return sample_fn(lr, ulr, None, g)
        x_e = model.compute_x_e(lr, ulr)
        rep = lambda x: None if x is None else x.repeat_interleave(k, dim=0)
        return model.sample(rep(ulr), rep(x_e), lr=rep(lr), solver=solver, n_steps=n_steps, eps=eps, generator=g)

    on_gpu = str(device).startswith('cuda') and torch.cuda.is_available()
    sync = torch.cuda.synchronize if on_gpu else (lambda: None)
    with torch.no_grad():
        for w in range(warmup):
            run(w)
        sync()
        times, nfe = [], 0
        for r in range(reps):
            t0 = time.perf_counter()
            _, nfe = run(100 + r)
            sync()
            times.append(time.perf_counter() - t0)
    t = float(np.median(times))
    n = B * k
    return dict(bench_frames=B, bench_k=k, bench_reps=reps, bench_batch_s=t,
                sec_per_sample=t / n, sec_per_frame_k=t / B, samples_per_sec=n / t,
                bench_nfe=int(nfe), device_name=torch.cuda.get_device_name() if on_gpu else 'cpu')


# ----------------------------------------------------------------------------------------------
# GIFs
# ----------------------------------------------------------------------------------------------

def _render_row(panels, titles, scales, suptitle):
    """One GIF frame: a row of temperature-style panels -> PIL image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0), dpi=100)
    for ax, a, ti, (vmin, vmax, cmap) in zip(axes, panels, titles, scales):
        im = ax.imshow(np.asarray(a).T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(ti, fontsize=9); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('P', palette=Image.ADAPTIVE)


def _save_gif(frames, path, ms=250, hold_last=6):
    frames = list(frames) + [frames[-1]] * hold_last
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    return path


def _scales(gt, final_mean, final_std, t_scale=(293.0, 5000.0), err_max=None, std_max=None):
    err_max = err_max or float(np.percentile(np.abs(final_mean - gt), 99)) or 1.0
    std_max = std_max or float(np.percentile(final_std, 99)) or 1.0
    return [(t_scale[0], t_scale[1], 'jet')] * 3 + [(0.0, err_max, 'magma'), (0.0, std_max, 'hot')]


def trajectory_gif(model, test_ds, frame_idx: int, out_path: str, solver: str = 'heun', n_steps: int = 20,
                   eps: float = 0.0, k: int = 10, seed: int = 0, field: int = 0, device: str = 'cuda',
                   t_scale=(293.0, 5000.0), err_max=None, std_max=None, label: str = '',
                   max_gif_frames: int = 61) -> str:
    """Animate one test frame from the start point to the final sample, one GIF frame per solver step.

    Columns: truth | sample 1 | sample 2 | |ensemble mean - truth| | ensemble std (k samples), physical
    units, fixed temperature scale; the error/std scales come from the final state unless given, so pass
    shared values to compare models frame by frame.
    """
    import torch
    _, hr, lr, ulr = test_ds[frame_idx][:4]
    hr, lr, ulr = (x[None].to(device).float() for x in (hr, lr, ulr))
    states = []
    with torch.no_grad():
        x_e = model.compute_x_e(lr, ulr)
        rep = lambda x: None if x is None else x.repeat_interleave(k, dim=0)
        g = torch.Generator(device=device).manual_seed(seed)
        cb = lambda i, t, x: states.append((int(i), float(t), x.detach().float().cpu().numpy()))
        x1, _ = model.sample(rep(ulr), rep(x_e), lr=rep(lr), solver=solver, n_steps=n_steps, eps=eps,
                             generator=g, callback=cb)
    states.append((len(states), 1.0, x1.detach().float().cpu().numpy()))
    if len(states) > max_gif_frames:            # e.g. ancestral DDPM: 1000 steps -> evenly spaced subset
        keep = np.unique(np.linspace(0, len(states) - 1, max_gif_frames).round().astype(int))
        states = [states[j] for j in keep]
    unscale = lambda a: np.stack([test_ds.unscale_data(s, input_type='hr') for s in a])
    gt_all = test_ds.unscale_data(hr[0].cpu().numpy(), input_type='hr')
    # `field` is the channel of frame t in an n_steps stack; a model that returns only the last timestep has
    # n_fields channels, where the same field sits at field % n_fields.
    fs = lambda a, c: c if c < a.shape[0] else c % a.shape[0]
    gt = gt_all[fs(gt_all, field)]
    fin_all = unscale(states[-1][2])
    final = fin_all[:, fs(fin_all[0], field)]
    scales = _scales(gt, final.mean(0), final.std(0), t_scale, err_max, std_max)
    frames = []
    for i, t, x in states:
        s_all = unscale(x)
        s = s_all[:, fs(s_all[0], field)]
        mean, std = s.mean(0), s.std(0)
        frames.append(_render_row([gt, s[0], s[min(1, len(s) - 1)], np.abs(mean - gt), std],
                                  ['Truth', 'Sample 1', 'Sample 2', '|mean - truth|', 'Ensemble std'],
                                  scales, f'{label}  step {i}  t = {t:.3f}  (frame {frame_idx})'.strip()))
    return _save_gif(frames, out_path)


def simtime_gif(preds, gts, info: list[dict], out_path: str, condition: str | None = None, field: int = 0,
                px_um: float = 10.0, t_scale=(293.0, 5000.0), label: str = '', max_frames: int = 120) -> str:
    """Animate the final predictions over consecutive timesteps of one test condition.

    Columns: truth | sample 1 | ensemble mean | |mean - truth| | std. Default condition: the deepest mean
    true melt pool in the test set (sub-pixel depth), i.e. the hardest keyhole case.
    """
    from diffusionsr.analysis.add_depth_metrics import mp_depth_subpixel
    N = preds.shape[0]
    info = info[:N]
    if condition is None:
        conds = sorted({d['cond'] for d in info})
        depth = {c: float(np.mean(mp_depth_subpixel(np.asarray(gts[[i for i, d in enumerate(info) if d['cond'] == c], field]), px_um)[0]))
                 for c in conds}
        condition = max(depth, key=depth.get)
    idx = sorted((i for i, d in enumerate(info) if d['cond'] == condition), key=lambda i: info[i]['t'])[:max_frames]
    P = np.asarray(preds[idx, :, field]); G = np.asarray(gts[idx, field])
    mean, std = P.mean(1), P.std(1)
    scales = _scales(G, mean, std, t_scale)
    scales[2] = scales[0]
    frames = [_render_row([G[j], P[j, 0], mean[j], np.abs(mean[j] - G[j]), std[j]],
                          ['Truth', 'Sample 1', 'Ensemble mean', '|mean - truth|', 'Ensemble std'],
                          scales, f'{label}  {condition}  timestep {info[i]["t"]}'.strip())
              for j, i in enumerate(idx)]
    return _save_gif(frames, out_path, ms=150, hold_last=3)


def current_frame(n_steps: int, n_fields: int) -> slice:
    """Channels of the current frame t in an n_steps-stacked tensor (channel = step * n_fields + field, oldest
    step first). Metrics, GIFs and saved predictions use only these; for n_steps = 1 it is every channel."""
    n_steps = int(n_steps)
    return slice((n_steps - 1) * n_fields, n_steps * n_fields)


def to_current_frame(preds, gts, ulrs, n_steps: int, n_fields: int):
    """Keep frame t only. Models that return all n_steps stacked frames (n_steps * F channels) are sliced; models
    that already return the last timestep (F channels, e.g. analysis/ablation_adapter.py) pass unchanged."""
    C = preds.shape[2]
    if C == n_fields:
        return preds, gts, ulrs
    if C != int(n_steps) * n_fields:
        raise ValueError(f'{C} prediction channels: expected {n_fields} or {int(n_steps) * n_fields}')
    cur = current_frame(n_steps, n_fields)
    return preds[:, :, cur], gts[:, cur], ulrs[:, cur]


# ---------------------------------------------------------------------------------------------------------------
# predictions.npz formats (docs/plans/2026-11_si_multi_eval_storage.md)
#   'full'   pred float32 (N, K, C, H, W), gt float32 (N, C, H, W), ens_std float32, field_names  (all older evals)
#   'q16'    pred stored as uint16 q with pred = q_offset[c] + q * q_step[c]; values outside the uint16 range (or
#            not finite) are kept EXACTLY in a float32 side list (out_idx = flat index into pred, out_val). Every
#            other value is within q_step / 2 of the float32 prediction (0.125 K for temperature). gt stays float32;
#            ens_std is not stored (load_predictions recomputes it on request). format = 'q16_v1'.
# Metrics are always computed inside the eval job from the float32 arrays, before saving.
# ---------------------------------------------------------------------------------------------------------------
Q16_FORMAT = 'q16_v1'
# (offset, step) per field. Temperature: 0.25 K steps over -4096 .. 12287.75 K. Labels: 2^-13 over -4 .. 4.
# Both steps are powers of two, so offset + q * step is exact in float32.
Q16_PARAMS = {'temperature': (-4096.0, 0.25), 'liqlabel': (-4.0, 2.0 ** -13), 'sdfliqlabel': (-4.0, 2.0 ** -13)}
Q16_MAX_OUTLIER_FRAC = 0.05     # more values outside the range than this: save float32 ('full') instead


def save_predictions(path: str, preds, gts, field_names, fmt: str = 'full', chunk: int = 64) -> str:
    """Write predictions.npz atomically in format 'full' or 'q16'; returns the format actually written ('q16'
    falls back to 'full' for unknown fields or too many out-of-range values)."""
    names = list(field_names)
    tmp = path + '.tmp.npz'
    if fmt == 'q16' and all(f in Q16_PARAMS for f in names):
        N, K, C, H, W = preds.shape
        if C != len(names):
            raise ValueError(f'{C} prediction channels for fields {names}')
        offs = np.array([Q16_PARAMS[f][0] for f in names], np.float64)[None, None, :, None, None]
        steps = np.array([Q16_PARAMS[f][1] for f in names], np.float64)[None, None, :, None, None]
        q = np.empty((N, K, C, H, W), np.uint16)
        idx, val = [], []
        per_frame = K * C * H * W
        for s in range(0, N, chunk):
            x = np.asarray(preds[s:s + chunk], dtype=np.float32)
            z = (x.astype(np.float64) - offs) / steps
            ok = np.isfinite(z) & (z > -0.5) & (z < 65535.5)
            q[s:s + len(x)] = np.where(ok, np.rint(np.where(ok, z, 0.0)), 0.0).astype(np.uint16)
            bad = np.flatnonzero(~ok)
            if bad.size:
                idx.append(bad.astype(np.int64) + s * per_frame)
                val.append(x.reshape(-1)[bad])
        n_out = int(sum(len(i) for i in idx))
        if n_out <= Q16_MAX_OUTLIER_FRAC * q.size:
            np.savez_compressed(tmp, format=np.array(Q16_FORMAT), pred_q=q,
                                q_offset=offs.reshape(-1), q_step=steps.reshape(-1),
                                out_idx=np.concatenate(idx) if idx else np.zeros(0, np.int64),
                                out_val=np.concatenate(val) if val else np.zeros(0, np.float32),
                                gt=np.asarray(gts, dtype=np.float32), field_names=np.array(names, dtype=object))
            os.replace(tmp, path)
            return 'q16'
        print(f'save_predictions: {n_out} of {q.size} values outside the 16-bit range; saving float32 instead')
        del q
    elif fmt == 'q16':
        print(f'save_predictions: no 16-bit parameters for fields {names}; saving float32 instead')
    elif fmt != 'full':
        raise ValueError(f'unknown predictions format {fmt!r}')
    p = np.asarray(preds, dtype=np.float32)
    np.savez_compressed(tmp, pred=p, gt=np.asarray(gts, dtype=np.float32), ens_std=p.std(axis=1),
                        field_names=np.array(names, dtype=object))
    os.replace(tmp, path)
    return 'full'


def load_predictions(path: str, with_std: bool = False) -> dict:
    """Read predictions.npz in either format: dict(pred float32 (N, K, C, H, W), gt, field_names, format[, ens_std])."""
    with np.load(path, allow_pickle=True) as z:
        names = [str(f) for f in z['field_names']] if 'field_names' in z.files else []
        gt = np.asarray(z['gt'], dtype=np.float32)
        if 'format' in z.files and str(z['format']) == Q16_FORMAT:
            q = z['pred_q']
            step = z['q_step'].astype(np.float32)[None, None, :, None, None]
            off = z['q_offset'].astype(np.float32)[None, None, :, None, None]
            pred = q.astype(np.float32)
            pred *= step
            pred += off
            pred.reshape(-1)[z['out_idx']] = z['out_val']
            out = dict(pred=pred, gt=gt, field_names=names, format='q16')
        else:
            pred = np.asarray(z['pred'], dtype=np.float32)
            out = dict(pred=pred, gt=gt, field_names=names, format='full')
            if with_std and 'ens_std' in z.files:
                out['ens_std'] = np.asarray(z['ens_std'], dtype=np.float32)
    if with_std and 'ens_std' not in out:
        out['ens_std'] = pred.std(axis=1)
    return out


def write_json(path: str, obj: dict):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
