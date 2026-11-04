"""
Sampler / LDM diagnostics for the ablation_v2 encoder runs (uses the existing checkpoints, no training).

Question 1 -- is DiffusionSR's pixel speckle a sampler effect?
  diffusion_enc_*: the same 5 QSS frames x 5 samples as the notebook-18 preview, sampled with
    ddim50   (DDIM, skip=20; the preview setting)   ddim200 (skip=5)   ddim1000 (skip=1)
    ddpm1000 (ancestral DDPM, 1000 stochastic steps; the sampler the model was trained for)
  The DDIM settings share the same initial noise per sample (eta=0), so differences are the step grid only.
  fm_enc_*: euler50 (preview setting) and euler200, as a control.

Question 2 -- why is LDM off: the VAE, the latent U-Net, or the sampler?
  ldm_enc_*: ddim50, ddim1000, ddpm1000, plus
    vae_recon_gt   decode(encode(GT))           -- the best any latent model with this VAE can do
    vae_recon_enc  decode(encode(RRDB(LR)))     -- what the conditioning carries, back in pixel space
    latent stats   per-channel mean/std of the VAE latent over 200 train frames (DDPM assumes ~unit variance)
  For every encoder run: enc_only = the frozen RRDB encoder's own pixel output (no diffusion at all).

Writes <out_dir>/<variant>__<setting>.npz with samples (N_QSS, N_S, C, H, W) in physical units, gts, ulrs,
seconds; and <out_dir>/<variant>__latent_stats.json for LDM. Metrics are computed offline from these files.

Run (CPU is fine: ~1-3 h; set threads with --threads):
  python scripts/diag_ablation_v2_sampling.py --out_dir diffusionsr/runs/ablation_v2/_diagnostics
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TQDM_DISABLE', '1')  # p_sample_loop wraps 1000 steps in tqdm

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the preview script's dataset/config/model loading so the models are built exactly as in the preview.
_spec = importlib.util.spec_from_file_location('pv', SCRIPT_DIR / 'precompute_ablation_v2_preview.py')
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)

SETTINGS = {
    'diffusion': [('ddim50', 'ddim', 20), ('ddim200', 'ddim', 5), ('ddim1000', 'ddim', 1), ('ddpm1000', 'ddpm', None)],
    'fm':        [('euler50', 'euler', 50), ('euler200', 'euler', 200)],
    'ldm':       [('ddim50', 'ddim', 20), ('ddim1000', 'ddim', 1), ('ddpm1000', 'ddpm', None)],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--models', nargs='*', default=['diffusion', 'fm', 'ldm'])
    p.add_argument('--fields', nargs='*', default=pv.FIELDS)
    p.add_argument('--norms', nargs='*', default=pv.NORMS)
    p.add_argument('--settings', nargs='*', default=None, help='Subset of setting names (default: all).')
    p.add_argument('--n_qss', type=int, default=5)
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def unscale(ds, x, n_ch, input_type='hr'):
    """(B, n_steps*n_ch, H, W) normalised tensor -> (B, n_ch, H, W) physical numpy, last timestep."""
    x = x.detach().cpu().numpy()
    return np.stack([ds.unscale_data(xi, input_type=input_type)[-n_ch:] for xi in x])


def sample(model, m, kind, arg, hr_b, x_e_b, ds):
    if kind == 'euler':
        out = model.batch_sample(ds, hr_b, x_e_b, sampler='euler', n_steps=arg)
    elif kind == 'ddim':
        out = model.batch_sample(ds, hr_b, x_e_b, sampler='DDIM', skip=arg)
    elif m == 'ldm':  # LDMModel.batch_sample handles the latent shape and decodes
        out = model.batch_sample(ds, hr_b, x_e_b, sampler='DDPM')
    else:
        out = model.p_sample_loop(model.model, x_e_b, shape=tuple(hr_b.shape), timesteps=model.timesteps)
    return out[-1] if (isinstance(out, list) or out.ndim == 5) else out


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device {device}, threads {torch.get_num_threads()}, out {out_dir}', flush=True)

    for m in args.models:
        for f in args.fields:
            for n in args.norms:
                variant = f'{m}_enc_{f}_{n}'
                run_dir = pv.RUNS_ROOT / variant
                cfg, cfg_path = pv.load_run_config(variant, run_dir)
                settings = pv.sampling_settings(m, cfg, 20)
                train_ds, dev_ds, test_ds = pv.load_datasets(f, n)
                model = pv.load_model(m, 'enc', f, n, run_dir, train_ds, dev_ds, test_ds, device, cfg, settings)
                if model is None:
                    raise RuntimeError(f'{variant}: model load failed')
                n_ch = len(pv.FIELD_NAMES[f])
                qss_idx = list(range(len(test_ds) - args.n_qss, len(test_ds)))
                frames = [test_ds[i] for i in qss_idx]
                gts = np.stack([test_ds.unscale_data(fr[1].numpy(), input_type='hr')[-n_ch:] for fr in frames])
                ulrs = np.stack([test_ds.unscale_data(fr[3].numpy(), input_type='upscaled_lr')[-n_ch:] for fr in frames])
                print(f'\n=== {variant} (T={model.timesteps}, cfg {cfg_path.name}) ===', flush=True)

                def save(setting, samples, secs, **extra):
                    np.savez_compressed(out_dir / f'{variant}__{setting}.npz', samples=samples, gts=gts, ulrs=ulrs,
                                        seconds=np.array(secs), **extra)
                    print(f'  {setting:14s} {secs:7.1f} s', flush=True)

                with torch.no_grad():
                    # Encoder alone: RRDB pixel output, the field the diffusion stage is conditioned on.
                    t0 = time.time()
                    enc_px = [pv_forward_output(model, train_ds, fr[2], device) for fr in frames]
                    save('enc_only', np.stack([unscale(test_ds, e, n_ch) for e in enc_px]), time.time() - t0)

                    for name, kind, arg in SETTINGS[m]:
                        if args.settings and name not in args.settings:
                            continue
                        if (out_dir / f'{variant}__{name}.npz').exists() and not args.force:
                            print(f'  {name:14s} exists, skipped')
                            continue
                        t0 = time.time()
                        out = []
                        for k, (res, hr, lr, ulr) in enumerate(frames):
                            x_e = model.compute_x_e(lr.unsqueeze(0).to(device).float(),
                                                    ulr.unsqueeze(0).to(device).float())
                            x_e_b = x_e.repeat(args.n_samples, *[1] * (x_e.ndim - 1))
                            hr_b = hr.unsqueeze(0).repeat(args.n_samples, 1, 1, 1).to(device).float()
                            torch.manual_seed(1000 + k)  # same initial noise for every setting of a frame
                            out.append(unscale(test_ds, sample(model, m, kind, arg, hr_b, x_e_b, test_ds), n_ch))
                        save(name, np.stack(out), time.time() - t0)

                    if m == 'ldm':
                        t0 = time.time()
                        rec_gt, rec_enc = [], []
                        for (res, hr, lr, ulr), e in zip(frames, enc_px):
                            mu, _ = model.vae.encode(hr.unsqueeze(0).to(device).float())
                            rec_gt.append(unscale(test_ds, model.vae.decode(mu), n_ch))
                            mu_e, _ = model.vae.encode(e.to(device).float())
                            rec_enc.append(unscale(test_ds, model.vae.decode(mu_e), n_ch))
                        save('vae_recon_gt', np.stack(rec_gt), time.time() - t0)
                        save('vae_recon_enc', np.stack(rec_enc), time.time() - t0)
                        rng = np.random.default_rng(0)
                        idx = rng.choice(len(train_ds), size=min(200, len(train_ds)), replace=False)
                        mus = torch.cat([model.vae.encode(train_ds[int(i)][1].unsqueeze(0).to(device).float())[0].cpu()
                                         for i in idx])  # (200, 4, h, w)
                        stats = {'n_frames': int(len(idx)), 'latent_shape': list(mus.shape[1:]),
                                 'per_channel_mean': mus.mean((0, 2, 3)).tolist(),
                                 'per_channel_std': mus.std((0, 2, 3)).tolist(),
                                 'overall_std': float(mus.std()), 'abs_max': float(mus.abs().max()),
                                 'unet_params': int(sum(p.numel() for p in model.model.parameters())),
                                 'vae_params': int(sum(p.numel() for p in model.vae.parameters()))}
                        (out_dir / f'{variant}__latent_stats.json').write_text(json.dumps(stats, indent=1))
                        print(f'  latent stats: {stats}', flush=True)
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
    print('\nDone.', flush=True)


def pv_forward_output(model, train_ds, lr, device):
    """RRDB encoder pixel output for one LR frame, in HR-normalised units (as LDMModel.compute_x_e uses it)."""
    from diffusionsr.runners.train_diffusion import forwardpass
    return forwardpass(model.lr_enc, lr.unsqueeze(0).to(device).float(), factor=train_ds.factor, output=True,
                       transform_rescale=model.transform_rescale, dataset=train_ds)


if __name__ == '__main__':
    main()
