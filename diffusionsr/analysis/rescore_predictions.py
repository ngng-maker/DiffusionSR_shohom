"""
Re-score finished evals from their saved predictions, without re-sampling.

Each eval uploads predictions.npz (physical-unit pred (N, K, C, H, W) and gt (N, C, H, W)) as the W&B
artifact `<run_name>_eval_predictions`. This downloads it, recomputes every metric with
evaluate_coupled.compute_metrics -- the exact code an eval run uses -- and rewrites the local
results.json, keeping the previous file as *_results.before_rescore.json.

Written after the melt-pool metrics turned out to be constant (MP/KH/depth/area MAE always 0,
calibration always 1). Re-scoring the saved samples gives corrected metrics on the IDENTICAL
predictions, whereas re-running the eval would re-sample (the base noise is random) and shift every
number slightly.

As a check that the artifact holds the predictions the existing results came from, MAE_mean and
MAE_ensemble_mean must reproduce (to float32 precision) before anything is written.

    python -m diffusionsr.analysis.rescore_predictions --results_root eval_results/interpolant \
        [--arms si_si_opt_a_sigma010_enc_temp ...] [--factor 2]
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import wandb

from diffusionsr.analysis.evaluate_coupled import _px_um, compute_metrics


def rescore(arm_dir: str, entity: str, project: str, factor: int) -> dict:
    name = os.path.basename(os.path.normpath(arm_dir))
    json_path = os.path.join(arm_dir, f'{name}_results.json')
    old = json.load(open(json_path))

    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        art = wandb.Api().artifact(f'{entity}/{project}/{name}_eval_predictions:latest')
        art.download(root=tmp)
        d = np.load(os.path.join(tmp, 'predictions.npz'), allow_pickle=True)
        pred, gt = d['pred'].astype(np.float64), d['gt'].astype(np.float64)

    new = compute_metrics(pred, gt, _px_um(factor))
    for k in ('MAE_mean', 'MAE_ensemble_mean'):
        if not np.isclose(new[k], old[k], rtol=1e-4):
            raise RuntimeError(f'{name}: {k} {new[k]:.6f} does not reproduce the stored {old[k]:.6f}; '
                               'the artifact does not hold these predictions. Nothing written.')

    shutil.copy2(json_path, json_path.replace('_results.json', '_results.before_rescore.json'))
    with open(json_path, 'w') as f:
        json.dump(new, f, indent=2)
    return {'name': name, 'old': old, 'new': new}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--arms', nargs='*', default=None, help='arm dir names; default: every arm')
    ap.add_argument('--factor', type=int, default=2, help='HR/LR factor (sets the pixel size)')
    ap.add_argument('--wandb_entity', default=os.getenv('WANDB_ENTITY', ''))
    ap.add_argument('--wandb_project', default='Flow3D_SuperResolution')
    args = ap.parse_args()

    arms = args.arms or sorted(
        a for a in os.listdir(args.results_root)
        if os.path.exists(os.path.join(args.results_root, a, f'{a}_results.json')))
    keys = ('MP_MAE_mean', 'KH_MAE_mean', 'depth_mae_um', 'area_mae_um2', 'calibration_2sigma')
    for a in arms:
        r = rescore(os.path.join(args.results_root, a), args.wandb_entity, args.wandb_project,
                    args.factor)
        print(f"{a}: MAE reproduced ({r['new']['MAE_mean']:.2f}); melt frames {r['new']['n_melt_frames']}")
        for k in keys:
            print(f"    {k:20s} {r['old'].get(k)!s:>10} -> {r['new'][k]:.4g}")


if __name__ == '__main__':
    main()
