"""
Add the SI-noise study's ensemble-verification scores (fair CRPS, spread-skill ratio, rank histogram;
see evaluate_si_schedules.ensemble_skill) to evaluations that were run BEFORE those scores existed:
the 128 SI-schedules evals and the notebook-21 baselines.

Each directory given must hold either predictions.npz (pred, gt) or notebook 21/22's cache pair
pred.npy + gt.npy. The scores are written to <dir>/skill.json; the original results.json is never
touched, so the published metrics of notebooks 21/22 stay exactly as they were.

    python -m diffusionsr.analysis.add_skill_metrics eval_results/si_schedules/*/ \
        eval_results/_predictions_cache/si_baseline_ddpm_matched_temp
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from diffusionsr.analysis.evaluate_si_schedules import ensemble_skill


def load_pair(d: str):
    npz = os.path.join(d, 'predictions.npz')
    if os.path.exists(npz):
        with np.load(npz, allow_pickle=True) as z:
            return np.asarray(z['pred']), np.asarray(z['gt'])
    p, g = os.path.join(d, 'pred.npy'), os.path.join(d, 'gt.npy')
    if os.path.exists(p) and os.path.exists(g):
        return np.load(p, mmap_mode='r'), np.load(g, mmap_mode='r')
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--force', action='store_true', help='recompute even if skill.json exists')
    args = ap.parse_args(argv)
    bad = 0
    for d in args.dirs:
        out = os.path.join(d, 'skill.json')
        if os.path.exists(out) and not args.force:
            print(f'{d}: skill.json exists, skipping'); continue
        pair = load_pair(d)
        if pair is None:
            print(f'{d}: no predictions.npz or pred.npy/gt.npy -- SKIPPED'); bad += 1; continue
        pred, gt = pair
        if pred.ndim == 4:                       # (N, C, H, W) point prediction -> K = 1
            pred = pred[:, None]
        s = ensemble_skill(pred, gt)
        s.update(K=int(pred.shape[1]), N=int(pred.shape[0]))
        with open(out + '.tmp', 'w') as f:
            json.dump(s, f, indent=2)
        os.replace(out + '.tmp', out)
        print(f"{d}: K={s['K']} CRPS all {s['crps_all_K']:.2f} melt {s['crps_melt_K']:.1f} | "
              f"SSR all {s['ssr_all']:.3f} melt {s['ssr_melt']:.3f}", flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
