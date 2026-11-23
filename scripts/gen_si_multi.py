"""
Generate every config and job list of the SI multi-field / regen-data / FM study
(docs/plans/2026-11_si_multifield_fm_plan.md).

Writes
  diffusionsr/configs/si_multi/<setting>/<name>.yml        training (and eval) configs
  diffusionsr/configs/si_multi/encoders/enc_<setting>.yml  frozen-encoder pretraining configs
  slurm/si_multi/train.tsv    one row per training run  (kind, name, modeltype, config, run_dir, enc_dir, phase)
  slurm/si_multi/eval.tsv     one row per evaluation     (name, config, model_dir, enc_dir, data_root, solver, eps,
                                                          n_steps, k, weights, out_root, extra, phase)
  slurm/si_multi/cascade.tsv  one row per chained eval   (chain, cfgA, runA, encA, rootA, cfgB, runB, encB, rootB,
                                                          solver, eps, n_steps, out_root, phase)
Paths in the files are relative to the TRACE repo root ($CL_REPO) unless absolute.

Settings
  o1   old data, temperature,                 20 -> 10 um   (retrained: the SI-schedules runs used n_steps 1)
  o2   old data, temperature + sdfliqlabel,   20 -> 10 um
  n1   regen,    temperature,                 40 -> 10 um direct
  n2   regen,    temperature + sdfliqlabel,   40 -> 10 um direct
  n3a/n3b  regen, temperature,                40 -> 20, 20 -> 10 um (cascade stages)
  n4a/n4b  regen, temperature + sdfliqlabel,  cascade stages

Every model uses n_steps = 5 (the current frame plus the 4 before it, stacked as channels) with
history = 'consecutive' (frames t-4..t of the same simulation, ambient before its first frame; the dataset's
'legacy' stacking skipped t-1 and crossed into the previous simulation). Metrics score frame t only.

Re-run after any change; the output is deterministic.
"""

from __future__ import annotations

import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = '/trace/group/forgelab/ngng/multifield'
TRACE_REPO = f'{TRACE}/DiffusionSR_shohom'
OLD = f'{TRACE}/data_fields'
REGEN = f'{TRACE}/datasets/regen_2d'
CFG_DIR = 'diffusionsr/configs/si_multi'
RUNS = 'diffusionsr/runs/si_multi'
EVALS = 'eval_results/si_multi'

# Liquid-region rule for the sdfliqlabel IoU (eval_extras.field_metrics); set from the dataset build.
# liqlabel is a binary 0/1 label in both datasets ('sdfliqlabel' is only an alias of that column).
LIQUID_RULE = os.getenv('SI_MULTI_LIQUID_RULE', 'gt0.5')

SETTINGS = {
    #  id      data root                  fields                      hr_px  study
    'o1':  dict(root=OLD,                      fields='temperature',             px=10, data='old'),
    'o2':  dict(root=OLD,                      fields='temperature_sdfliqlabel', px=10, data='old'),
    'n1':  dict(root=f'{REGEN}/x4_40to10',     fields='temperature',             px=10, data='regen'),
    'n2':  dict(root=f'{REGEN}/x4_40to10',     fields='temperature_sdfliqlabel', px=10, data='regen'),
    'n3a': dict(root=f'{REGEN}/x2_40to20',     fields='temperature',             px=20, data='regen'),
    'n3b': dict(root=f'{REGEN}/x2_20to10',     fields='temperature',             px=10, data='regen'),
    'n4a': dict(root=f'{REGEN}/x2_40to20',     fields='temperature_sdfliqlabel', px=20, data='regen'),
    'n4b': dict(root=f'{REGEN}/x2_20to10',     fields='temperature_sdfliqlabel', px=10, data='regen'),
}
CASCADES = {'n3': ('n3a', 'n3b'), 'n4': ('n4a', 'n4b')}
SCHEDULES = ('linear', 'trig', 'vp', 'bridge')
BASES = ('enc', 'ulr')
EPS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
FM_SETTINGS = ('o1', 'o2', 'n1', 'n2')
STEP_SWEEP = (5, 10, 20, 50, 100)


# Training hyper-parameters: byte-identical to the SI-schedules study (and tasks 0 / 11 / 12).
N_STEPS = '5'
HISTORY = 'consecutive'
COMMON = dict(downscale_method='direct', n_steps=N_STEPS, history=HISTORY, residual_flag=False, normalize_method='standardize',
              schedule='linear', timesteps=1000, batch_size=40, epochs=300, learning_rate=2e-4,
              use_pretrained=False, loss_type='l2', encoding=True, conditioning='implicit')
HARNESS = dict(fm_timescale=1000.0, ema_decay=0.999, grad_clip=1.0, p_flip=0.2, ckpt_every=5)
# SI and FM models are scored with their EMA weights (checkpoint[5]), matching the DDPM-matched EMA headline row;
# DDPM matched is also scored raw as a reference, DiffusionSR has no EMA (plain loop). The cascade job does the same.
SI_WEIGHTS = 'ema'
# Encoder pretraining: identical to fno_ablation/enc_rrdb (the O1 encoder).
ENC = dict(downscale_method='direct', n_steps=N_STEPS, history=HISTORY, encoding=True, residual_flag=False, conditioning='implicit',
           normalize_method='standardize', schedule='linear', timesteps=1000, batch_size=8, epochs=300,
           learning_rate=5e-5, use_pretrained=False, loss_type='huber', encoder_type='rrdb')


def enc_dir(s):
    return f'{TRACE_REPO}/{RUNS}/encoders/enc_{s}'


def write_cfg(rel, cfg, header):
    path = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n') as f:
        f.write(''.join(f'# {h}\n' for h in header))
        yaml.safe_dump(cfg, f, sort_keys=False)
    return rel


def base_cfg(s):
    st = SETTINGS[s]
    cfg = dict(root_folder=st['root'], **COMMON, fields=st['fields'], encoder_results_dir=enc_dir(s),
               hr_px_um=st['px'])
    if LIQUID_RULE and 'sdfliqlabel' in st['fields']:
        cfg['liquid_rule'] = LIQUID_RULE
    return cfg


def main():
    train, evals, casc = [], [], []

    def tsv(rows, path, head):
        with open(os.path.join(REPO, path), 'w', newline='\n') as f:
            f.write('\t'.join(head) + '\n')
            for r in rows:
                f.write('\t'.join(str(x) for x in r) + '\n')

    def grid_evals(s, name, cfg_rel, run_rel, out_root, phase):
        for solver in ('euler', 'heun'):
            for e in EPS:
                evals.append((name, cfg_rel, run_rel, enc_dir(s), SETTINGS[s]['root'], solver, e, 20, 10, SI_WEIGHTS,
                              out_root, '-', phase))
        for solver in ('rk4', 'dopri5'):
            evals.append((name, cfg_rel, run_rel, enc_dir(s), SETTINGS[s]['root'], solver, 0.0, 20, 10, SI_WEIGHTS,
                          out_root, '-', phase))

    # ---------------- encoders (phase 1) ----------------
    for s in SETTINGS:
        st = SETTINGS[s]
        rel = write_cfg(f'{CFG_DIR}/encoders/enc_{s}.yml',
                        dict(root_folder=st['root'], **ENC, fields=st['fields']),
                        [f'Frozen RRDB encoder for setting {s}: recipe of fno_ablation/enc_rrdb, with n_steps 5',
                         'Run: train_srdiff --modeltype encoder --force_enc_dir <enc dir>'])
        train.append(('encoder', f'enc_{s}', 'encoder', rel, enc_dir(s), enc_dir(s), 1))

    # ---------------- SI grids (phase 2) ----------------
    for s in SETTINGS:
        phase = 2 if s in ('o1', 'o2') else (3 if s in ('n1', 'n2') else 4)
        for sch in SCHEDULES:
            for b in BASES:
                name = f'sim_{s}_{sch}_{b}'
                cfg = base_cfg(s)
                cfg.update(HARNESS, si_sigma=0.1, si_gamma_c=0.0, si_cond='enc_implicit', si_process='scheduled',
                           si_schedule=sch, si_base=b)
                rel = write_cfg(f'{CFG_DIR}/{s}/{name}.yml', cfg,
                                [f'SI {sch} schedule, {b} start, setting {s} ({SETTINGS[s]["fields"]})',
                                 'All other keys identical to configs/si_schedules (harness, sigma, encoder conditioning)'])
                run = f'{RUNS}/{s}/{name}'
                train.append(('si', name, 'coupled', rel, run, enc_dir(s), phase))
                grid_evals(s, name, rel, run, f'{EVALS}/{s}', phase)

    # ---------------- baselines (encoder alone, DDPM matched, DiffusionSR) ----------------
    for s in SETTINGS:
        phase = 2 if s in ('o1', 'o2') else (3 if s in ('n1', 'n2') else 4)
        cfg = base_cfg(s)
        cfg['model_kind'] = 'encoder_only'
        rel = write_cfg(f'{CFG_DIR}/{s}/encoder_only.yml', cfg, [f'Frozen encoder alone, setting {s} (K = 1)'])
        evals.append((f'sim_{s}_encoder_only', rel, '-', enc_dir(s), SETTINGS[s]['root'], 'encoder', 0.0, 1, 1,
                      'raw', f'{EVALS}/{s}', '-', phase))
        # DDPM in the SI harness (task-0 style); O1 is retrained because task 0's checkpoint is gone.
        cfg = base_cfg(s)
        cfg.update(HARNESS, si_sigma=0.1, si_gamma_c=0.0, si_cond='enc_implicit', si_process='ddpm',
                   model_kind='ddpm_matched')
        name = f'sim_{s}_ddpm_matched'
        rel = write_cfg(f'{CFG_DIR}/{s}/{name}.yml', cfg, [f'DDPM in the SI harness (task-0 recipe), setting {s}'])
        run = f'{RUNS}/{s}/{name}'
        train.append(('ddpm_matched', name, 'coupled', rel, run, enc_dir(s), phase))
        for w in ('ema', 'raw'):
            evals.append((f'{name}_{w}', rel, run, enc_dir(s), SETTINGS[s]['root'], 'ddpm', 0.0, 1000, 10, w,
                          f'{EVALS}/{s}', '-', phase))
        # DiffusionSR as published (plain loop; no EMA in its checkpoint)
        cfg = base_cfg(s)
        cfg['model_kind'] = 'diffusionsr'
        name = f'sim_{s}_diffusionsr'
        rel = write_cfg(f'{CFG_DIR}/{s}/{name}.yml', cfg, [f'DiffusionSR as published (task-11 recipe), setting {s}'])
        run = f'{RUNS}/{s}/{name}'
        train.append(('diffusionsr', name, 'diffusion', rel, run, enc_dir(s), phase))
        evals.append((f'{name}_raw', rel, run, enc_dir(s), SETTINGS[s]['root'], 'ddpm', 0.0, 1000, 10, 'raw',
                      f'{EVALS}/{s}', '-', phase))

    # ---------------- cascades: matched pairs (phase 4) ----------------
    for c, (a, b) in CASCADES.items():
        for sch in SCHEDULES:
            for base in BASES:
                na, nb = f'sim_{a}_{sch}_{base}', f'sim_{b}_{sch}_{base}'
                rows = [(s_, e_) for s_ in ('euler', 'heun') for e_ in EPS] + [('rk4', 0.0), ('dopri5', 0.0)]
                for solver, e in rows:
                    casc.append((f'sim_{c}_{sch}_{base}_cascade', f'{CFG_DIR}/{a}/{na}.yml', f'{RUNS}/{a}/{na}',
                                 enc_dir(a), SETTINGS[a]['root'], f'{CFG_DIR}/{b}/{nb}.yml', f'{RUNS}/{b}/{nb}',
                                 enc_dir(b), SETTINGS[b]['root'], solver, e, 20, f'{EVALS}/{c}_cascade', 4))
        # baseline chains: DDPM matched (EMA and raw) and DiffusionSR, ancestral sampling
        for kind, ws in (('ddpm_matched', ('ema', 'raw')), ('diffusionsr', ('raw',))):
            na, nb = f'sim_{a}_{kind}', f'sim_{b}_{kind}'
            for w in ws:
                casc.append((f'sim_{c}_{kind}_{w}_cascade', f'{CFG_DIR}/{a}/{na}.yml', f'{RUNS}/{a}/{na}', enc_dir(a),
                             SETTINGS[a]['root'], f'{CFG_DIR}/{b}/{nb}.yml', f'{RUNS}/{b}/{nb}', enc_dir(b),
                             SETTINGS[b]['root'], f'ddpm:{w}', 0.0, 1000, f'{EVALS}/{c}_cascade', 4))

    # ---------------- FM study (Experiment 2) ----------------
    for s in FM_SETTINGS:
        phase = 2 if s in ('o1', 'o2') else 3
        si_name = f'sim_{s}_linear_enc'
        si_cfg = f'{CFG_DIR}/{s}/{si_name}.yml'
        si_run = f'{RUNS}/{s}/{si_name}'
        si_out = f'{EVALS}/{s}'
        models = [(si_name, si_cfg, si_run, si_out, ('euler', 'heun', 'rk4'), True)]
        for src in ('noise', 'paired'):
            name = f'sim_{s}_fm_{src}'
            cfg = base_cfg(s)
            cfg.update(HARNESS, si_sigma=0.1, si_gamma_c=0.0, si_cond='enc_implicit', si_base='enc',
                       si_process='fm_library', model_kind='fm_library', fm_source=src)
            rel = write_cfg(f'{CFG_DIR}/{s}/{name}.yml', cfg,
                            [f'Standard flow matching (facebookresearch/flow_matching, CondOT path, single velocity head)',
                             f'start: {"pure Gaussian noise (independent coupling)" if src == "noise" else "encoder prediction + sigma*zeta (SI paired coupling)"}; setting {s}'])
            run = f'{RUNS}/{s}/{name}'
            train.append(('fm', name, 'coupled', rel, run, enc_dir(s), phase))
            models.append((name, rel, run, f'eval_results/fm_study/{s}', ('euler', 'midpoint', 'rk4'), False))
        for name, rel, run, out, solvers, is_si in models:
            for solver in solvers:
                for n in STEP_SWEEP:
                    if is_si and n == 20:
                        continue                 # already in the SI grid (same eval name and out_root)
                    evals.append((name, rel, run, enc_dir(s), SETTINGS[s]['root'], solver, 0.0, n, 10, SI_WEIGHTS,
                                  out, '-', phase))
            if not is_si:                        # SI's dopri5 is already in its grid
                evals.append((name, rel, run, enc_dir(s), SETTINGS[s]['root'], 'dopri5', 0.0, 20, 10, SI_WEIGHTS,
                              out, '-', phase))

    os.makedirs(os.path.join(REPO, 'slurm/si_multi'), exist_ok=True)
    tsv(train, 'slurm/si_multi/train.tsv', ('kind', 'name', 'modeltype', 'config', 'run_dir', 'enc_dir', 'phase'))
    tsv(evals, 'slurm/si_multi/eval.tsv', ('name', 'config', 'model_dir', 'enc_dir', 'data_root', 'solver', 'eps',
                                           'n_steps', 'k', 'weights', 'out_root', 'extra', 'phase'))
    tsv(casc, 'slurm/si_multi/cascade.tsv', ('chain', 'cfg_a', 'run_a', 'enc_a', 'root_a', 'cfg_b', 'run_b', 'enc_b',
                                             'root_b', 'solver', 'eps', 'n_steps', 'out_root', 'phase'))
    by = lambda rows, i: {p: sum(1 for r in rows if r[i] == p) for p in sorted({r[i] for r in rows})}
    print(f'train rows {len(train)} by phase {by(train, 6)}')
    print(f'eval rows {len(evals)} by phase {by(evals, 12)}')
    print(f'cascade rows {len(casc)}')
    if not LIQUID_RULE:
        print('NOTE: SI_MULTI_LIQUID_RULE unset -> no liquid IoU for sdfliqlabel until the rule is known', file=sys.stderr)


if __name__ == '__main__':
    main()
