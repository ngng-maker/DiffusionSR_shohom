"""
Build 2D LR/HR super-resolution datasets from the 2026 SS316L FLOW-3D regen HDF5 release.

The output reproduces the on-disk layout and conventions of the 2022 `data_fields` set exactly, so
`SimulationXZDataset` / `FlippedDataset` load it unchanged:

    <root>/<split>/HR/power{P}velocity{V}/_{1000+frame}.npy            (H, W, C) float32
    <root>/<split>/LR/direct/1x/power{P}velocity{V}/_{1000+frame}.npy  (h, w, C) float32, true LR simulation
    <root>/<split>/LR/direct/{f}x/power{P}velocity{V}/_{...}.npy       (H, W, C) LR upsampled to the HR grid
    <root>/field_metadata.json, split_metadata.json, split_manifest.csv, split_manifest_seed{S}.npy,
    <root>/{train,dev,test}_conditions_seed{S}.npy
    (statistics/direct/ is created by the dataset class on the first train-split load)

Conventions copied from data_fields (checked against its files, see DATASET_INFO.md):
  * channels (temperature, liqlabel, pressure, vx, vy, vz); liqlabel is a binary 0/1 liquid indicator
    (the dataset's 'sdfliqlabel' is an alias of this same column); CGS units (K, dyn/cm^2, cm/s).
  * array axes (x, z): x along the scan direction, z vertical with the free surface at HIGH z, so
    evaluate_coupled._mp_geometry's temp.T has rows = z.
  * xz plane through the beam centre line; window centred on the laser's x position; full domain depth.
  * frames 1..100 of the 101 written (t = 0, before the laser acts, is dropped), named _1001.._1100.
  * no temperature clipping in the files (the dataset clips at THRESHOLD_T = 8000 K on load).
  * upsampled LR = skimage.transform.resize, order 1 (bilinear) for continuous fields and order 0 (nearest)
    for liqlabel. This reproduces data_fields' LR/direct/2x bit for bit (the old notebook's cv2 cubic does not).

Choices specific to this release:
  * beam plane: y = 0.32 mm is a cell FACE on every mesh, so the plane is the mean of the two cells either
    side of it (liqlabel: liquid if either cell is liquid, i.e. mean >= 0.5). Symmetric about the beam on
    every mesh, unlike picking one cell (which would sit 5, 10 or 20 um off the line depending on the mesh).
  * void masking: cells with fluidfrac < 0.5 are void (README 4.3). FLOW-3D 12 leaves hot values in some
    of them (notably at 40 um, above the surface), whereas in data_fields gas and the keyhole cavity are at
    ambient. Void cells are therefore set to the void state (299.15 K, 1 atm, v = 0, liqlabel 0) BEFORE
    the two planes are averaged.
  * window: 640 x 640 um (HR 64x64 at 10 um, 32x32 at 20 um, LR 16x16 at 40 um). Its left edge is the
    laser position minus 320 um, snapped to the 40 um grid, so every mesh crops the IDENTICAL physical
    region (the 2022 notebook rounded each mesh separately, leaving LR and HR up to a cell apart). The
    laser position uses the nominal frame time k * 5 us (actual solver times differ by < 0.3 us, < 0.5 um
    of laser travel; both are recorded in the manifest). Cells outside the 1.28 mm domain are padded with
    the void state, as the 2022 notebook padded with its pad values.
  * ghost planes (README 4.4) are trimmed: arrays[:-1, :-1, :-1].

Usage (TRACE):
    python -m diffusionsr.datasets.preprocessing.build_regen_dataset \
        --h5_root /trace/group/forgelab/ngng/multifield/datasets/ss316l_flow3d_regen_2026-09-03/h5 \
        --out /trace/group/forgelab/ngng/multifield/datasets/regen_2d --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

FIELDS = ['temperature', 'liqlabel', 'pressure', 'vx', 'vy', 'vz']      # data_fields channel order
VOID = {'temperature': 299.15, 'liqlabel': 0.0, 'pressure': 1.013e6, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0}
MESHES_UM = (10, 20, 40)
WINDOW_UM = 640
SNAP_UM = 40                     # coarsest mesh: every crop edge lands on a cell boundary of all meshes
LASER_X0_UM = 400.0              # README 2.4: beam on at x = 0.40 mm, t = 0, moving +x
BEAM_Y_UM = 320.0                # README 2.3: centre line y = 0.32 mm
FRAME_DT_S = 5e-6
FRAMES = range(1, 101)           # drop frame 0 (t = 0), keep 100 frames -> _1001.._1100
SEED = 20260518                  # same seed as data_fields
RATIOS = {'train': 0.8, 'dev': 0.1, 'test': 0.1}

# task name -> (HR mesh um, LR mesh um)
TASKS = {'x4_40to10': (10, 40), 'x2_40to20': (20, 40), 'x2_20to10': (10, 20)}


def cond_name(P, V):
    return f'power{P}velocity{V}'


def h5_path(h5_root, mesh, P, V):
    return os.path.join(h5_root, f'{mesh}um', f'ss316l_{mesh}um_core_P{P}_V{V}.h5')


def crop_planes(h5_file, mesh_um, V, frames):
    """Return {frame: (W, W, C) array}, plus actual times, for one mesh of one condition."""
    import h5py
    n = WINDOW_UM // mesh_um
    out, times = {}, {}
    with h5py.File(h5_file, 'r') as f:
        g = f['output_5us']
        t_all = g['time'][:]
        y = f['mesh/y'][:-1] * 1e4                             # cm -> um, interior cells only
        j_hi = int(np.searchsorted(y, BEAM_Y_UM))               # first cell centre above the beam line
        j_lo = j_hi - 1
        assert abs((y[j_lo] + y[j_hi]) / 2 - BEAM_Y_UM) < 1e-3, (h5_file, y[j_lo], y[j_hi])
        dx = float(f['mesh/x'][1] - f['mesh/x'][0]) * 1e4
        assert abs(dx - mesh_um) < 1e-3, (h5_file, dx)
        nx = f['mesh/x'].shape[0] - 1
        nz = f['mesh/z'].shape[0] - 1
        assert nz * mesh_um == WINDOW_UM, (h5_file, nz)          # full domain depth = window height
        dsets = {k: g['fields'][k] for k in FIELDS + ['fluidfrac']}
        for fr in frames:
            # Read only the two beam-adjacent y planes (one chunk per frame, so this is cheap).
            raw = {k: dsets[k][fr, :nx, j_lo:j_hi + 1, :nz].astype(np.float64) for k in dsets}  # (nx, 2, nz)
            void = raw['fluidfrac'] < 0.5
            planes = []
            for k in FIELDS:
                a = raw[k]
                if k == 'liqlabel':
                    a = (a > 0).astype(np.float64)
                a = np.where(void, VOID[k], a)
                a = a.mean(axis=1)                                 # (nx, nz), plane at y = 0.32 mm
                if k == 'liqlabel':
                    a = (a >= 0.5).astype(np.float64)
                planes.append(a)
            plane = np.stack(planes, axis=-1)                      # (nx, nz, C)
            x0 = crop_x0_um(V, fr)
            i0 = int(round(x0 / mesh_um))
            crop = np.empty((n, nz, len(FIELDS)))
            for c, k in enumerate(FIELDS):
                crop[..., c] = VOID[k]
            src_lo, src_hi = max(i0, 0), min(i0 + n, nx)
            if src_hi > src_lo:
                crop[src_lo - i0:src_hi - i0] = plane[src_lo:src_hi]
            out[fr] = crop.astype(np.float32)
            times[fr] = float(t_all[fr])
    return out, times


def crop_x0_um(V, frame):
    """Left edge of the crop (um): laser x - 320 um snapped to the 40 um grid, identical on every mesh."""
    x_laser = LASER_X0_UM + V * 1e3 * (frame * FRAME_DT_S)     # V mm/s -> um/s
    return SNAP_UM * round((x_laser - WINDOW_UM / 2) / SNAP_UM)


def upsample(lr, shape):
    from skimage.transform import resize
    up = resize(lr, (*shape, lr.shape[-1]), order=1, preserve_range=True, anti_aliasing=False)
    up[..., FIELDS.index('liqlabel')] = resize(lr[..., FIELDS.index('liqlabel')], shape, order=0,
                                               preserve_range=True, anti_aliasing=False)
    return up.astype(np.float32)


def _link_or_save(path, arr, cache):
    """Save once per (mesh, condition, frame) array and hard-link the identical file into other roots."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = id(arr)
    if key in cache and os.path.exists(cache[key]):
        if os.path.exists(path):
            os.remove(path)
        try:
            os.link(cache[key], path)
            return
        except OSError:
            pass
    np.save(path, arr)
    cache[key] = path


def process_condition(job):
    h5_root, out, P, V, split = job
    name = cond_name(P, V)
    data, times = {}, {}
    for m in MESHES_UM:
        data[m], times[m] = crop_planes(h5_path(h5_root, m, P, V), m, V, FRAMES)
    rows, cache = [], {}
    ups = {}
    for task, (hr_m, lr_m) in TASKS.items():
        factor = lr_m // hr_m
        root = os.path.join(out, task)
        for fr in FRAMES:
            fn = f'_{1000 + fr}.npy'
            hr, lr = data[hr_m][fr], data[lr_m][fr]
            if (lr_m, hr_m, fr) not in ups:
                ups[(lr_m, hr_m, fr)] = upsample(lr, hr.shape[:2])
            up = ups[(lr_m, hr_m, fr)]
            rel = {'hr': f'{split}/HR/{name}/{fn}', 'lr': f'{split}/LR/direct/1x/{name}/{fn}',
                   'upscaled_lr': f'{split}/LR/direct/{factor}x/{name}/{fn}'}
            _link_or_save(os.path.join(root, rel['hr']), hr, cache)
            _link_or_save(os.path.join(root, rel['lr']), lr, cache)
            _link_or_save(os.path.join(root, rel['upscaled_lr']), up, cache)
            rows.append(dict(task=task, kind='2d', split=split, condition=name, frame=fr, filename=fn,
                             fields=','.join(FIELDS), hr_shape=hr.shape, lr_shape=lr.shape,
                             upscaled_lr_shape=up.shape, **rel,
                             source_hr=h5_path(h5_root, hr_m, P, V), source_lr=h5_path(h5_root, lr_m, P, V),
                             source_upscaled_lr=f'resize(order=1; liqlabel order=0) of {rel["lr"]}',
                             time_hr_s=times[hr_m][fr], time_lr_s=times[lr_m][fr],
                             time_nominal_s=fr * FRAME_DT_S, window_x0_um=crop_x0_um(V, fr),
                             laser_x_um=LASER_X0_UM + V * 1e3 * fr * FRAME_DT_S))
    return rows


def core_conditions(h5_root):
    with open(os.path.join(h5_root, 'manifest.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['family'] == 'core']
    have = {}
    for r in rows:
        have.setdefault((int(r['power_W']), int(r['velocity_mm_s'])), set()).add(int(r['dx_um']))
    return sorted(pv for pv, ms in have.items() if set(MESHES_UM) <= ms)


def make_split(conds, seed):
    rng = np.random.default_rng(seed)
    perm = [conds[i] for i in rng.permutation(len(conds))]
    n_test = int(round(RATIOS['test'] * len(conds)))
    n_dev = int(round(RATIOS['dev'] * len(conds)))
    split = {'test': perm[:n_test], 'dev': perm[n_test:n_test + n_dev], 'train': perm[n_test + n_dev:]}
    return {s: sorted(v) for s, v in split.items()}


MANIFEST_COLS = ['condition', 'fields', 'filename', 'frame', 'hr', 'hr_shape', 'kind', 'lr', 'lr_shape',
                 'source_hr', 'source_lr', 'source_upscaled_lr', 'split', 'upscaled_lr', 'upscaled_lr_shape',
                 'time_hr_s', 'time_lr_s', 'time_nominal_s', 'window_x0_um', 'laser_x_um']


def write_metadata(out, split, rows, seed):
    for task in TASKS:
        root = os.path.join(out, task)
        os.makedirs(root, exist_ok=True)
        trows = sorted((r for r in rows if r['task'] == task),
                       key=lambda r: (('train', 'dev', 'test').index(r['split']), r['condition'], r['frame']))
        names = {s: [cond_name(P, V) for P, V in split[s]] for s in split}
        with open(os.path.join(root, 'field_metadata.json'), 'w') as f:
            json.dump({'fields': FIELDS}, f)
        with open(os.path.join(root, 'split_metadata.json'), 'w') as f:
            json.dump({'seed': seed, 'ratios': RATIOS, 'split_conditions': names, 'sample_count': len(trows)},
                      f, indent=2)
        with open(os.path.join(root, 'split_manifest.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction='ignore')
            w.writeheader()
            w.writerows(trows)
        np.save(os.path.join(root, f'split_manifest_seed{seed}.npy'),
                np.array([{k: r[k] for k in MANIFEST_COLS} for r in trows], dtype=object), allow_pickle=True)
        for s in ('train', 'dev', 'test'):
            np.save(os.path.join(root, f'{s}_conditions_seed{seed}.npy'), np.array(names[s]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--h5_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='smoke runs: first N conditions per split')
    args = ap.parse_args(argv)

    conds = core_conditions(args.h5_root)
    split = make_split(conds, args.seed)
    print({s: len(v) for s, v in split.items()}, 'conditions', flush=True)
    jobs = [(args.h5_root, args.out, P, V, s) for s in ('train', 'dev', 'test')
            for (P, V) in (split[s][:args.limit] if args.limit else split[s])]
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_condition, jobs, chunksize=1)):
            rows.extend(r)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f'  {i + 1}/{len(jobs)} conditions', flush=True)
    if args.limit:
        split = {s: v[:args.limit] for s, v in split.items()}
    write_metadata(args.out, split, rows, args.seed)
    print('done', {t: sum(r['task'] == t for r in rows) for t in TASKS})


if __name__ == '__main__':
    main()
