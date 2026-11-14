"""
Build the train-split statistics and sanity-check the regen_2d roots made by build_regen_dataset.py.

For each root: load train/dev/test through FlippedDataset (the train load naming all six channels writes
statistics/direct/ over ALL channels first, so any field subset loads later, exactly as data_fields'
6-channel cache), then reload with ['temperature'] and ['temperature', 'sdfliqlabel'] and print shapes,
factor, counts and value ranges; report melt-pool geometry on HR test frames and how often the LR frame
melts when the HR frame does; save one PNG of a matched 40/20/10 um frame.

    python -m diffusionsr.datasets.preprocessing.verify_regen_dataset --out <regen_2d> --png <path>
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from diffusionsr.datasets.flipped_dataset import FlippedDataset
from diffusionsr.analysis.evaluate_coupled import _mp_geometry

ALL_FIELDS = ['temperature', 'liqlabel', 'pressure', 'vx', 'vy', 'vz']
TASKS = {'x4_40to10': (10, 40), 'x2_40to20': (20, 40), 'x2_20to10': (10, 20)}


def kw(root, fields):
    return dict(downscale_method='direct', root_folder=root, normalize='standardize', n_steps=1,
                field_names=fields, p_flip=0.0)


def check_root(root, hr_um, lr_um):
    print(f'\n=== {root}')
    # field_names=None would add the dataset's 'sdfliqlabel' alias as a 7th name (duplicate column) and break
    # compute_statistics, so name all six file channels explicitly.
    FlippedDataset(split='train', **kw(root, ALL_FIELDS))                 # writes statistics over all channels
    for fields in (['temperature'], ['temperature', 'sdfliqlabel']):
        for split in ('train', 'dev', 'test'):
            ds = FlippedDataset(split=split, **kw(root, fields))
            res, hr, lr, ulr = ds[0]
            n_cond = len({os.path.basename(os.path.dirname(p)) for p in ds.hr_paths})
            print(f'  {fields} {split}: n={len(ds)} conditions={n_cond} factor={ds.factor} '
                  f'hr{tuple(hr.shape)} lr{tuple(lr.shape)} ulr{tuple(ulr.shape)} '
                  f'hr[min,max]=({float(hr.min()):.2f},{float(hr.max()):.2f}) mean_hr{ds.mean_hr.shape}')
    # Physical-unit checks on the raw test files
    ds = FlippedDataset(split='test', **kw(root, ['temperature']))
    mp, melt_hr, melt_both, tmax = [], 0, 0, 0.0
    for p_hr, p_lr in zip(ds.hr_paths[::25], ds.lr_paths[::25]):
        T, Tl = np.load(p_hr)[..., 0], np.load(p_lr)[..., 0]
        tmax = max(tmax, float(T.max()))
        g = _mp_geometry(T, float(hr_um))
        if g['mp_depth'] > 0:
            melt_hr += 1
            mp.append(g['mp_depth'])
            melt_both += bool((Tl > 1900).any())
    print(f'  test HR frames sampled: {len(ds.hr_paths[::25])}, melted {melt_hr}, LR also melted {melt_both}; '
          f'MP depth um median {np.median(mp):.0f} p90 {np.percentile(mp, 90):.0f} max {max(mp):.0f}; '
          f'max T {tmax:.0f} K')


def preview_png(out, path, cond='power490velocity300', frame='_1060.npy'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    found = [s for s in ('train', 'dev', 'test') if os.path.exists(os.path.join(out, 'x4_40to10', s, 'HR', cond, frame))]
    if found:
        split = found[0]
    else:                                   # smoke builds may not contain the default condition
        split = 'test'
        cond = sorted(os.listdir(os.path.join(out, 'x4_40to10', split, 'HR')))[0]
    arrs = {'10 um (HR, x4_40to10)': np.load(os.path.join(out, 'x4_40to10', split, 'HR', cond, frame)),
            '20 um (HR, x2_40to20)': np.load(os.path.join(out, 'x2_40to20', split, 'HR', cond, frame)),
            '40 um (LR 1x)': np.load(os.path.join(out, 'x4_40to10', split, 'LR', 'direct', '1x', cond, frame)),
            '40 um upsampled (LR 4x)': np.load(os.path.join(out, 'x4_40to10', split, 'LR', 'direct', '4x', cond, frame))}
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=110)
    for j, (name, a) in enumerate(arrs.items()):
        ext = [0, 640, 0, 640]
        ax[0, j].imshow(a[..., 0].T, origin='lower', cmap='jet', vmin=293, vmax=5000, extent=ext)
        ax[0, j].contour(np.linspace(0, 640, a.shape[0]), np.linspace(0, 640, a.shape[1]), a[..., 0].T,
                         levels=[1900], colors='w', linewidths=0.8)
        ax[0, j].set_title(f'T  {name}', fontsize=9)
        ax[1, j].imshow(a[..., 1].T, origin='lower', cmap='gray', vmin=0, vmax=1, extent=ext)
        ax[1, j].set_title(f'liqlabel  {name}', fontsize=9)
        for r in (0, 1):
            ax[r, j].set_xlabel('x (um, window)'); ax[r, j].set_ylabel('z (um)')
    fig.suptitle(f'regen_2d {cond} {frame} ({split}); white = 1900 K')
    fig.tight_layout()
    fig.savefig(path)
    print('PNG', path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--png', required=True)
    args = ap.parse_args(argv)
    for task, (hr_um, lr_um) in TASKS.items():
        check_root(os.path.join(args.out, task), hr_um, lr_um)
    preview_png(args.out, args.png)


if __name__ == '__main__':
    main()
