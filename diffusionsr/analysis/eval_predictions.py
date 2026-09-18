"""
Evaluate one ablation run: sample the full test set and upload predictions + GT to W&B.

For each test sample, runs model.batch_sample() to obtain the final prediction,
unscales both prediction and ground truth to physical units, and saves everything
as a single .npz artifact on the experiment's W&B run.

LDM runs also include VAE reconstructions (key: 'vae_recon') for notebook-side
vae_mae computation.

Usage (FM + encoder + SDF):
    python -m diffusionsr.analysis.eval_predictions \
        --run_name abl_fm_enc_sdf \
        --wandb_run_name 1_Sep_2026_fm_enc_sdf \
        --model_type flow_matching \
        --model_dir /path/to/abl_fm_enc_sdf \
        --enc_dir   /path/to/enc_sdf \
        --data_root /path/to/data_fields \
        --field_names temperature liqlabel

Usage (LDM + no-encoder + temp):
    python -m diffusionsr.analysis.eval_predictions \
        --run_name abl_ldm_noenc_temp \
        --wandb_run_name 1_Sep_2026_ldm_noenc_temp \
        --model_type ldm \
        --model_dir /path/to/abl_ldm_noenc_temp \
        --vae_dir   /path/to/vae_temp \
        --data_root /path/to/data_fields \
        --field_names temperature
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader


def _find_root():
    s = Path(__file__).resolve()
    for p in [s, *s.parents]:
        if (p / 'setup.py').exists() and (p / 'diffusionsr').exists():
            return p
    raise RuntimeError('cannot find project root')


PROJECT_ROOT = _find_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',         required=True,  help='short experiment id, e.g. abl_fm_enc_sdf')
    parser.add_argument('--wandb_run_name',   required=True,  help='W&B display name of the training run')
    parser.add_argument('--model_type',       required=True,  choices=['flow_matching', 'ldm'])
    parser.add_argument('--model_dir',        required=True,  help='results_folder for the trained model')
    parser.add_argument('--enc_dir',          default=None,   help='RRDB encoder folder; omit for no-encoder runs')
    parser.add_argument('--vae_dir',          default=None,   help='VAE folder; required for LDM runs')
    parser.add_argument('--data_root',        required=True)
    parser.add_argument('--field_names',      nargs='+',      default=['temperature'])
    parser.add_argument('--n_steps',          type=int,       default=3)
    parser.add_argument('--fm_n_steps',       type=int,       default=50,   help='Euler steps; FM only')
    parser.add_argument('--timesteps',        type=int,       default=1000)
    parser.add_argument('--normalize',        default='standardize')
    parser.add_argument('--downscale_method', default='direct')
    parser.add_argument('--conditioning',     default='implicit')
    parser.add_argument('--schedule',         default='linear')
    parser.add_argument('--batch_size',       type=int,       default=4)
    parser.add_argument('--device',           default='cuda')
    parser.add_argument('--wandb_entity',     default=os.getenv('WANDB_ENTITY', ''))
    parser.add_argument('--wandb_project',    default='Flow3D_SuperResolution')
    args = parser.parse_args()

    if args.model_type == 'ldm' and args.vae_dir is None:
        parser.error('--vae_dir is required for model_type=ldm')

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    encoding = args.enc_dir is not None

    # ── Datasets ──────────────────────────────────────────────────────────────
    from diffusionsr.datasets.dataset import SimulationXZDataset
    kw = dict(downscale_method=args.downscale_method, root_folder=args.data_root,
              normalize=args.normalize, n_steps=args.n_steps, field_names=args.field_names)
    train_ds, dev_ds, test_ds = (SimulationXZDataset(split=s, **kw) for s in ['train', 'dev', 'test'])
    print(f'Fields: {train_ds.field_names}  |  n_test={len(test_ds)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_type == 'flow_matching':
        from diffusionsr.runners.train_flow_matching import FlowMatchingModel
        model = FlowMatchingModel(
            results_folder=args.model_dir, lr_encoder_folder=args.enc_dir,
            train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
            timesteps=args.timesteps, conditioning=args.conditioning,
            encoding=encoding, schedule=args.schedule, device=device, enc_output=False,
        )
    else:
        from diffusionsr.runners.train_ldm import LDMModel
        model = LDMModel(
            vae_folder=args.vae_dir, results_folder=args.model_dir,
            lr_encoder_folder=args.enc_dir,
            train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
            timesteps=args.timesteps, conditioning=args.conditioning,
            encoding=encoding, schedule=args.schedule, device=device, enc_output=False,
        )
    model.load_saved_model()
    print(f'Model loaded from {args.model_dir}')

    # ── Sample full test set ──────────────────────────────────────────────────
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    all_preds, all_gts, all_vae = [], [], []
    is_ldm = args.model_type == 'ldm'

    for i, batch in enumerate(test_loader):
        _, hr_b, lr_b, ul_b = batch[:4]
        xe = model.compute_x_e(lr_b, ul_b) if encoding else None

        with torch.no_grad():
            if is_ldm:
                samps = model.batch_sample(dataset=test_ds, batch=hr_b.to(device),
                                           x_e=xe, sampler='DDPM')
                pred = samps[-1].cpu().numpy()
                mu, _ = model.vae.encode(hr_b.to(device).float())
                vr = model.vae.decode(mu).cpu().numpy()
                all_vae.extend(
                    test_ds.unscale_data(vr[s], input_type='hr') for s in range(hr_b.shape[0])
                )
            else:
                samps = model.batch_sample(dataset=test_ds, batch=hr_b.to(device),
                                           x_e=xe, sampler='euler', n_steps=args.fm_n_steps)
                pred = samps[-1].cpu().numpy()
                del samps; torch.cuda.empty_cache()

        for s in range(hr_b.shape[0]):
            all_preds.append(test_ds.unscale_data(pred[s],              input_type='hr'))
            all_gts.append(  test_ds.unscale_data(hr_b.numpy()[s],      input_type='hr'))

        print(f'  batch {i+1}/{len(test_loader)}  ({hr_b.shape[0]} samples)', flush=True)

    print(f'Sampling complete: {len(all_preds)} total samples')

    # ── Build .npz payload ────────────────────────────────────────────────────
    save_dict = dict(
        pred=np.array(all_preds, dtype=np.float32),   # (N, C, H, W) physical units
        gt=np.array(all_gts,   dtype=np.float32),     # (N, C, H, W) physical units
        field_names=np.array(args.field_names, dtype=object),
        n_steps=np.array([args.n_steps]),
    )
    if all_vae:
        save_dict['vae_recon'] = np.array(all_vae, dtype=np.float32)

    # ── Upload artifact to existing W&B training run ──────────────────────────
    api = wandb.Api()
    existing = list(api.runs(
        f'{args.wandb_entity}/{args.wandb_project}',
        filters={'display_name': args.wandb_run_name},
    ))
    if existing:
        run = wandb.init(id=existing[0].id, resume='allow',
                         project=args.wandb_project, entity=args.wandb_entity)
    else:
        print(f'Warning: no W&B run with display_name={args.wandb_run_name!r}; creating eval run.')
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=f'{args.run_name}_eval', job_type='eval')

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, 'predictions.npz')
        np.savez(npz_path, **save_dict)

        artifact = wandb.Artifact(
            name=f'{args.run_name}_eval_predictions',
            type='eval_predictions',
            description=f'Test-set predictions for {args.run_name} ({len(all_preds)} samples)',
        )
        artifact.add_file(npz_path, name='predictions.npz')
        run.log_artifact(artifact, aliases=['latest'])

    run.finish()
    print('Artifact uploaded. Done.')


if __name__ == '__main__':
    main()
