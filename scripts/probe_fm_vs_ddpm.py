"""
Why does Flow-Matching make fewer melt-pool boundary mistakes than DiffusionSR? Probes on the existing
ablation_v2 encoder checkpoints (no training). For each field/norm config, loads diffusion_enc_* and fm_enc_*.

A. One-step denoising at matched noise levels (the trained network alone, no sampler).
   Same GT frame, same Gaussian noise eps, same noise-to-signal ratio r = sigma/alpha:
     DiffusionSR: x_t = sqrt(abar) x0 + sqrt(1-abar) eps at the integer t whose r_t is closest to r;
                  x0_hat = (x_t - sqrt(1-abar) eps_hat) / sqrt(abar)
     FM:          t = r/(1+r);  x_t = (1-t) x0 + t eps;  x0_hat = x_t - t v_hat
   Saves x0_hat (temperature, physical units) for every frame x draw x noise level.

B. Sampling trajectories from the same initial noise: DDIM-50 (eta=0), ancestral DDPM-1000 and FM Euler-50.
   Records the network's clean-image estimate x0_hat at every step (DDPM: every 20th step), so one can see
   at which noise level a stray pixel first appears and whether it is later corrected.

Output: <out_dir>/<field>_<norm>__probeA.npz and __probeB.npz (float16 to keep them small); metrics offline.
  python scripts/probe_fm_vs_ddpm.py --out_dir diffusionsr/runs/ablation_v2/_diagnostics_fm_vs_ddpm
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
_spec = importlib.util.spec_from_file_location('pv', SCRIPT_DIR / 'precompute_ablation_v2_preview.py')
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)

R_GRID = np.logspace(-2, 2, 17)  # noise-to-signal ratios for probe A


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--fields', nargs='*', default=pv.FIELDS)
    p.add_argument('--norms', nargs='*', default=pv.NORMS)
    p.add_argument('--n_qss', type=int, default=5)
    p.add_argument('--n_draws', type=int, default=4, help='noise draws per frame for probe A')
    p.add_argument('--n_samples', type=int, default=5, help='trajectories per frame for probe B')
    return p.parse_args()


def load(m, f, n, device):
    variant = f'{m}_enc_{f}_{n}'
    run_dir = pv.RUNS_ROOT / variant
    cfg, _ = pv.load_run_config(variant, run_dir)
    settings = pv.sampling_settings(m, cfg, 20)
    tr, dv, te = pv.load_datasets(f, n)
    model = pv.load_model(m, 'enc', f, n, run_dir, tr, dv, te, device, cfg, settings)
    if model is None:
        raise RuntimeError(f'{variant}: model load failed')
    return model, te, settings


def phys_T(ds, x, n_ch):
    """(B, n_steps*n_ch, H, W) normalised -> (B, H, W) temperature of the last timestep, physical, float16."""
    x = x.detach().float().cpu().numpy()
    return np.stack([ds.unscale_data(xi, input_type='hr')[-n_ch] for xi in x]).astype(np.float16)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for f in args.fields:
        for n in args.norms:
            dm, ds, _ = load('diffusion', f, n, device)
            fm, ds_fm, fm_set = load('fm', f, n, device)
            n_ch = len(pv.FIELD_NAMES[f])
            abar = torch.cumprod(1 - dm.betas.float().to(device), 0)        # (T,)
            r_t = ((1 - abar) / abar).sqrt().cpu().numpy()                   # noise/signal per DDPM t
            t_idx = np.array([int(np.argmin(np.abs(r_t - r))) for r in R_GRID])
            r_used = r_t[t_idx]
            qss = list(range(len(ds) - args.n_qss, len(ds)))
            frames = [ds[i] for i in qss]
            gts = np.stack([ds.unscale_data(fr[1].numpy(), input_type='hr')[-n_ch] for fr in frames]).astype(np.float32)
            print(f'\n=== {f}_{n}: T={dm.timesteps}, fm_timescale={fm.fm_timescale}, r grid {r_used.round(3).tolist()}',
                  flush=True)

            # ---------------- probe A ----------------
            A_dm = np.zeros((len(frames), args.n_draws, len(R_GRID), 60, 60), np.float16); A_fm = A_dm.copy()
            with torch.no_grad():
                for k, (res, hr, lr, ulr) in enumerate(frames):
                    lr_t, ulr_t = lr.unsqueeze(0).to(device).float(), ulr.unsqueeze(0).to(device).float()
                    xe_d = dm.compute_x_e(lr_t, ulr_t); xe_f = fm.compute_x_e(lr_t, ulr_t)
                    x0 = hr.unsqueeze(0).to(device).float()
                    g = torch.Generator(device=device).manual_seed(100 + k)
                    for d in range(args.n_draws):
                        eps = torch.randn(x0.shape, device=device, generator=g)
                        B = len(R_GRID)
                        x0b, epsb = x0.repeat(B, 1, 1, 1), eps.repeat(B, 1, 1, 1)
                        # DiffusionSR
                        tt = torch.tensor(t_idx, device=device)
                        a = abar[tt].view(-1, 1, 1, 1)
                        xt = a.sqrt() * x0b + (1 - a).sqrt() * epsb
                        eps_hat = dm.model(xt, tt.float(), xe_d.repeat(B, *[1] * (xe_d.ndim - 1)))
                        A_dm[k, d] = phys_T(ds, (xt - (1 - a).sqrt() * eps_hat) / a.sqrt(), n_ch)
                        # FM at the same noise-to-signal ratio
                        tf = torch.tensor(r_used / (1 + r_used), device=device, dtype=torch.float32)
                        tv = tf.view(-1, 1, 1, 1)
                        xt = (1 - tv) * x0b + tv * epsb
                        v_hat = fm.model(xt, tf * fm.fm_timescale, xe_f.repeat(B, *[1] * (xe_f.ndim - 1)))
                        A_fm[k, d] = phys_T(ds, xt - tv * v_hat, n_ch)
            np.savez_compressed(out_dir / f'{f}_{n}__probeA.npz', dm=A_dm, fm=A_fm, gts=gts, r=r_used, t_idx=t_idx)
            print('  probe A saved', flush=True)

            # ---------------- probe B ----------------
            N = args.n_samples
            seq = list(range(0, dm.timesteps, 20))  # DDIM-50 grid
            trajs = {'ddim': [], 'ddpm': [], 'fm': []}
            rs = {'ddim': [], 'ddpm': [], 'fm': []}
            finals = {'ddim': [], 'ddpm': [], 'fm': []}
            with torch.no_grad():
                for k, (res, hr, lr, ulr) in enumerate(frames):
                    lr_t, ulr_t = lr.unsqueeze(0).to(device).float(), ulr.unsqueeze(0).to(device).float()
                    xe_d = dm.compute_x_e(lr_t, ulr_t).repeat(N, *[1] * 3)
                    xe_f = fm.compute_x_e(lr_t, ulr_t).repeat(N, *[1] * 3)
                    shape = (N,) + tuple(hr.shape)
                    g = torch.Generator(device=device).manual_seed(1000 + k)
                    x_T = torch.randn(shape, device=device, generator=g)

                    # DDIM-50, eta=0 (same update as DiffusionModel.batch_sample)
                    x, tr, rr = x_T.clone(), [], []
                    for i, j in zip(reversed(seq), reversed([-1] + seq[:-1])):
                        at = abar[i]; an = abar[j] if j >= 0 else torch.tensor(1.0, device=device)
                        e = dm.model(x, torch.full((N,), float(i), device=device), xe_d)
                        x0h = (x - (1 - at).sqrt() * e) / at.sqrt()
                        tr.append(phys_T(ds, x0h, n_ch)); rr.append(float(((1 - at) / at).sqrt()))
                        x = an.sqrt() * x0h + (1 - an).sqrt() * e
                    trajs['ddim'].append(np.stack(tr)); rs['ddim'] = rr; finals['ddim'].append(phys_T(ds, x, n_ch))

                    # ancestral DDPM-1000 (same update as DiffusionModel.p_sample), x0_hat kept every 20 steps
                    x, tr, rr = x_T.clone(), [], []
                    gen = torch.Generator(device=device).manual_seed(5000 + k)
                    for i in reversed(range(dm.timesteps)):
                        tt = torch.full((N,), i, device=device, dtype=torch.long)
                        e = dm.model(x, tt, xe_d)
                        at = abar[i]
                        if i % 20 == 0:
                            tr.append(phys_T(ds, (x - (1 - at).sqrt() * e) / at.sqrt(), n_ch))
                            rr.append(float(((1 - at) / at).sqrt()))
                        mean = dm.sqrt_recip_alphas[i].to(device) * (x - dm.betas[i].to(device) * e
                                                                      / dm.sqrt_one_minus_alphas_cumprod[i].to(device))
                        if i > 0:
                            x = mean + dm.posterior_variance[i].to(device).sqrt() * torch.randn(shape, device=device, generator=gen)
                        else:
                            x = mean
                    trajs['ddpm'].append(np.stack(tr)); rs['ddpm'] = rr; finals['ddpm'].append(phys_T(ds, x, n_ch))

                    # FM Euler-50 (same update as FlowMatchingModel.euler_sample), started from the same x_T
                    x, tr, rr = x_T.clone(), [], []
                    nst = fm_set['n_sampling_steps']; dt = 1.0 / nst
                    for i in reversed(range(nst)):
                        t = (i + 1) / nst
                        v = fm.model(x, torch.full((N,), t * fm.fm_timescale, device=device), xe_f)
                        tr.append(phys_T(ds, x - t * v, n_ch)); rr.append(t / (1 - t) if t < 1 else np.inf)
                        x = x - v * dt
                    trajs['fm'].append(np.stack(tr)); rs['fm'] = rr; finals['fm'].append(phys_T(ds, x, n_ch))
            np.savez_compressed(out_dir / f'{f}_{n}__probeB.npz', gts=gts,
                                **{f'traj_{s}': np.stack(v) for s, v in trajs.items()},      # (F, S, N, H, W)
                                **{f'final_{s}': np.stack(v) for s, v in finals.items()},    # (F, N, H, W)
                                **{f'r_{s}': np.array(v) for s, v in rs.items()})
            print('  probe B saved', flush=True)
            del dm, fm
            if device == 'cuda':
                torch.cuda.empty_cache()
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
