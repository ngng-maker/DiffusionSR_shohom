"""Score the DiffusionSR (DDPM + encoder) baseline with the SAME metrics as the SI arms.

`evaluate_coupled.py` computes the metrics the stochastic-interpolant study is judged on -- K-sample
ensemble MAE/RMSE, MP-MAE, keyhole depth MAE, calibration coverage, per-pixel ensemble std -- but
its `main()` only ever builds a `CoupledInterpolantModel`, so the DDPM baseline cannot be put
through it. Evaluating the baseline with `eval_predictions.py` instead produces a different artifact
with a different (smaller) metric set, which makes the two arms not directly comparable.

This module closes that gap WITHOUT modifying `evaluate_coupled.py`: it imports that module's
`evaluate()` function verbatim and simply hands it a `DiffusionModel` instead. Identical metric code,
identical artifact layout, so the numbers line up arm for arm.

It works because `evaluate()` only touches `model.compute_x_e(...)` and
`model.batch_sample(dataset=..., batch=..., x_e=..., sampler=..., n_steps=..., upscaled_lr=...,
eps0=...)`, and `DiffusionModel.batch_sample` has a `**kwargs` tail that absorbs the arguments
meaningful only to the interpolant (`upscaled_lr`, `eps0`, `n_steps`).

Usage (on TRACE):
  python -m diffusionsr.analysis.evaluate_baseline_matched \\
      --run_name   baseline_diffusionsr_temp \\
      --model_dir  /path/to/runs/.../baseline_diffusionsr_temp \\
      --enc_dir    /path/to/runs/ablation_20260901/enc_temp \\
      --data_root  /trace/group/forgelab/ngng/multifield/data_fields \\
      --field_names temperature \\
      --skip       50 \\
      --k_samples  10 \\
      --out_dir    /path/to/eval_outputs/baseline \\
      --wandb_entity $WANDB_ENTITY
"""

import argparse  # Command-line parsing
import json  # Writing the results file
import os  # Path handling and existence checks
import sys  # Used by the project-root bootstrap below
import tempfile  # Scratch directory for the W&B artifact
from pathlib import Path  # Walking up to find the project root

import numpy as np  # Array handling for the saved artifact
import torch  # Checkpoint loading and device selection
import wandb  # Artifact upload
from torch.utils.data import DataLoader  # Test-set iteration


def _find_root():
    """Walk up from this file until the directory containing setup.py and diffusionsr/ is found."""
    start = Path(__file__).resolve()
    for candidate in [start, *start.parents]:  # Include the file itself so a flat layout still works
        if (candidate / 'setup.py').exists() and (candidate / 'diffusionsr').exists():
            return candidate
    raise RuntimeError('cannot find project root')


PROJECT_ROOT = _find_root()
if str(PROJECT_ROOT) not in sys.path:
    # Prepending lets `python -m diffusionsr.analysis...` work from any cwd, matching the
    # bootstrap the other analysis modules use.
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_name', required=True, help='short experiment id, e.g. baseline_diffusionsr_temp')
    parser.add_argument('--wandb_run_name', default=None,
                        help='W&B display name of the training run; omit to skip the artifact upload')
    parser.add_argument('--model_dir', required=True, help='results_folder of the trained DDPM')
    parser.add_argument('--enc_dir', required=True,
                        help='frozen encoder folder. Required: DiffusionSR is encoder-conditioned, '
                             'so there is no meaningful no-encoder baseline to compare against.')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--field_names', nargs='+', default=['temperature'])
    parser.add_argument('--downscale', default='direct')
    parser.add_argument('--timesteps', type=int, default=1000, help='length of the DDPM chain used in training')
    parser.add_argument('--schedule', default='linear', help='beta schedule used in training; must match')
    parser.add_argument('--conditioning', default='implicit', help='implicit (added) or explicit (concatenated)')
    # The SI arms are integrated for --n_steps; the DDPM is strided by --skip. 1000/50 = 20 steps,
    # so skip=50 is the DDIM setting that matches n_steps=20 on the interpolant side.
    parser.add_argument('--skip', type=int, default=50,
                        help='DDIM stride. The effective step count is timesteps/skip, so pick skip '
                             'to match the SI arms n_steps for a like-for-like sampling budget.')
    parser.add_argument('--k_samples', type=int, default=10,
                        help='ensemble size. Must match the SI arms or the diversity metrics are '
                             'not comparable.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--plot_frame_idx', type=int, default=70)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--wandb_entity', default=os.getenv('WANDB_ENTITY', ''))
    parser.add_argument('--wandb_project', default='Flow3D_SuperResolution')
    parser.add_argument('--artifact_name', default=None)
    # Which training harness produced this checkpoint. The two baselines are NOT interchangeable:
    #   'coupled' -> trained by DDPMMatchedHarnessModel through CoupledInterpolantModel.train(),
    #                so the checkpoint carries EMA weights and the model must be rebuilt as that
    #                subclass. This is the harness-matched baseline, the attributable comparison.
    #   'plain'   -> trained by DiffusionModel.train(). The as-published DiffusionSR, kept as a
    #                reference point; it differs from the SI arms by four harness features as well
    #                as by the generative process.
    parser.add_argument('--harness', default='coupled', choices=['coupled', 'plain'],
                        help="which training loop produced the checkpoint; determines the class "
                             "rebuilt here and whether EMA weights are loaded")
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='load the EMA weights (harness=coupled only); EMA is standard practice '
                             'and the SI arms are evaluated the same way')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # The metric code, imported rather than reimplemented. Any change the other session makes to
    # these metrics automatically applies to the baseline too, which is the point of importing.
    from diffusionsr.analysis.evaluate_coupled import evaluate

    # FlippedDataset with p_flip=0.0 is used (not SimulationXZDataset) so the baseline sees exactly
    # the same data pipeline as the SI arms -- the flip augmentation is disabled at eval for both.
    from diffusionsr.datasets.flipped_dataset import FlippedDataset
    dataset_kwargs = dict(downscale_method=args.downscale, root_folder=args.data_root,
                          normalize='standardize', n_steps=1, field_names=args.field_names, p_flip=0.0)
    train_ds = FlippedDataset(split='train', **dataset_kwargs)
    dev_ds = FlippedDataset(split='dev', **dataset_kwargs)
    test_ds = FlippedDataset(split='test', **dataset_kwargs)
    print(f'Fields: {test_ds.field_names}  factor={test_ds.factor}  n_test={len(test_ds)}')

    # Recover the U-Net's `dim` from the checkpoint rather than from the dataset. DiffusionModel
    # derives dim from dataset.img_shape, which can differ between the training and eval
    # environments; reading it back guarantees the architecture matches the weights. Same trick
    # evaluate_coupled.py uses, and the same reason.
    best_path = os.path.join(args.model_dir, 'bestmodel_saved.pth')
    ckpt_path = os.path.join(args.model_dir, 'ckpt.pth')
    load_path = best_path if os.path.exists(best_path) else ckpt_path
    if not os.path.exists(load_path):
        raise FileNotFoundError(f'no checkpoint found in {args.model_dir} (looked for bestmodel_saved.pth and ckpt.pth)')
    # weights_only=False: plain-DiffusionModel checkpoints hold a numpy scalar (best_val_loss) that
    # torch>=2.6's weights_only=True default refuses. These are this repo's own checkpoints.
    peek = torch.load(load_path, map_location='cpu', weights_only=False)
    # time_mlp.1 is Linear(dim, 4*dim), so shape[1] is the true dim. init_conv.weight.shape[0] is
    # init_dim (which defaults to 64) and would give the wrong answer.
    unet_dim = int(peek[0]['time_mlp.1.weight'].shape[1])
    del peek  # Free the checkpoint before building the model, which allocates again
    print(f'Recovered U-Net dim={unet_dim} from checkpoint')

    common = dict(
        results_folder=args.model_dir,
        lr_encoder_folder=args.enc_dir,
        train_dataset=train_ds, dev_dataset=dev_ds, test_dataset=test_ds,
        timesteps=args.timesteps,  # Must match training: it sets the beta schedule's length
        conditioning=args.conditioning,
        encoding=True,  # DiffusionSR is encoder-conditioned by definition
        schedule=args.schedule,  # Must match training or the reverse process is mismatched
        device=device,
        enc_output=False,  # Production path: condition on the encoder's 64-channel intermediate
        image_size_override=unet_dim,  # Use the checkpoint's dim, not the dataset's
    )

    if args.harness == 'coupled':
        # The harness-matched baseline. Rebuilt as the same subclass that trained it, so the
        # restored weights land in the architecture they were trained in.
        from diffusionsr.runners.train_ddpm_matched import DDPMMatchedHarnessModel
        model = DDPMMatchedHarnessModel(sigma=0.1, gamma_c=0.0, si_cond='enc_implicit',
                                        fm_timescale=1000.0, **common)
    else:
        from diffusionsr.runners.train_diffusion import DiffusionModel
        model = DiffusionModel(**common)

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    if args.harness == 'coupled' and args.use_ema and len(checkpoint) > 5 and checkpoint[5] is not None:
        # CoupledInterpolantModel.train() stores the EMA shadow weights at index 5. The SI arms are
        # evaluated from their EMA weights, so the baseline must be too or the comparison is unfair
        # in the baseline's disfavour.
        from diffusionsr.utils import remove_module_prefix
        model.model.load_state_dict(remove_module_prefix(checkpoint[5]))
        print(f'Loaded EMA weights from {os.path.basename(load_path)}')
    else:
        model.model.load_state_dict(checkpoint[0])  # Element 0 is the raw model state dict
        print(f'Loaded raw weights from {os.path.basename(load_path)}')
    model.model.eval()  # Disable dropout/batchnorm updates during sampling

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # The identical metric function the SI arms go through. `sampler='DDIM'` selects the strided
    # sampler; `n_steps` is passed through only to satisfy the shared signature and is absorbed by
    # DiffusionModel.batch_sample's **kwargs -- the DDPM's step count is set by `skip` instead,
    # which is forwarded via the same kwargs tail.
    results, preds, gts, ens_std, overlay_path = evaluate(
        model, test_ds, test_loader,
        k_samples=args.k_samples,
        sampler='DDIM',
        n_steps=args.skip,  # reaches batch_sample as a kwarg; `skip` below is what DDIM actually reads
        eps0=0.0,  # meaningless for a DDPM; present for signature parity
        device=device,
        out_dir=args.out_dir,
        plot_frame_idx=args.plot_frame_idx,
    )

    print('\n=== RESULTS ===')
    print(json.dumps(results, indent=2))
    json_path = os.path.join(args.out_dir, f'{args.run_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    if args.wandb_run_name is None:
        print('--wandb_run_name not provided; skipping W&B artifact upload.')
        return

    # Attach the artifact to the existing training run so the eval sits alongside the loss curves.
    api = wandb.Api()
    existing = list(api.runs(f'{args.wandb_entity}/{args.wandb_project}',
                             filters={'display_name': args.wandb_run_name}))
    if existing:
        run = wandb.init(id=existing[0].id, resume='allow',
                         project=args.wandb_project, entity=args.wandb_entity)
    else:
        print(f'Warning: no W&B run named {args.wandb_run_name!r}; creating a separate eval run.')
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=f'{args.run_name}_eval', job_type='eval')

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, 'predictions.npz')
        # Same keys and dtypes as evaluate_coupled writes, so one comparison notebook reads both.
        np.savez(npz_path,
                 pred=preds.astype(np.float32),
                 gt=gts.astype(np.float32),
                 ens_std=ens_std.astype(np.float32),
                 field_names=np.array(args.field_names, dtype=object))
        artifact_name = args.artifact_name or f'{args.run_name}_eval_predictions'
        artifact = wandb.Artifact(name=artifact_name, type='eval_predictions',
                                  description=f'DiffusionSR baseline eval (SI-matched metrics) - {args.run_name}')
        artifact.add_file(npz_path, name='predictions.npz')
        artifact.add_file(json_path, name='results.json')
        if overlay_path and os.path.exists(overlay_path):
            artifact.add_file(overlay_path, name='overlay.png')
        run.log_artifact(artifact, aliases=['latest'])
        run.finish()
    print('Artifact uploaded.')


if __name__ == '__main__':
    main()
