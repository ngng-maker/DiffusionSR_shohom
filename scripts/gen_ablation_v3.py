"""
Ablation v3: the ablation-v2 architecture comparison (DiffusionSR / Flow-Matching / LDM x encoder / no encoder x
temperature / temperature+sdfliqlabel x pixel-wise / global standardisation) redone on the 2026 regen data
(regen_2d/x2_20to10: 20 -> 10 um, 32 -> 64 px), plus seed repeats and an LDM latent U-Net size sweep.

Differences from v2, decided by the user (2026-11-14):
  * ONE shared frozen RRDB encoder per (field, normalisation), 600 epochs, used by every model of that pair, so
    model differences are not encoder luck (v2's per-run encoders confounded the comparison).
  * ONE shared VAE per (field, normalisation) for the LDM runs, trained before them.
  * seed 1 repeats of the 8 DSR/FM encoder arms (model seed only; the encoder is shared).
  * LDM latent U-Net width 16 / 48 / 96 / 192 / 256 (mults (1,): 0.07 / 0.51 / 1.9 / 7.4 / 13.1 M params) on
    temp/std with the encoder, seeds 0 and 1.
  * evaluation on the FULL test split with K = 10 through evaluate_si_schedules (SI metrics, P-V maps, timing,
    model size, trajectory + simulation-time GIFs); DSR and LDM ancestral DDPM-1000, FM Euler-50.
  * 12 LR-conditioned no-encoder arms (added 2026-11-23, appended so existing array indices are unchanged):
    the 'noenc' arms are unconditional, so 'lrcond' (as v2) separates what the encoder adds from having the input.
  * history: consecutive (dataset.py): the 5 stacked frames are t-4..t of the same simulation, ambient before
    frame 1. v2's legacy stacking skipped t-1 and took early frames' history from the previous simulation.
Everything else as v2: n_steps 5, 600 epochs, batch 40, lr 5e-5, Huber loss, linear schedule, T = 1000.

Writes diffusionsr/configs/ablation_v3/{encoders,vaes,models,evals}/*.yml and slurm/ablation_v3/{train,eval}.tsv.
    python scripts/gen_ablation_v3.py            # full study
    python scripts/gen_ablation_v3.py --smoke    # 1-epoch configs + TSVs under *_smoke for slurm/ablation_v3/00_smoke.sh
"""
import argparse
import os

REPO = '/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom'
DATA = '/trace/group/forgelab/ngng/multifield/datasets/regen_2d/x2_20to10'
FIELDS = {'temp': 'temperature', 'multi': 'temperature_sdfliqlabel'}
NORMS = {'std': 'standardize', 'gstd': 'global_standardize'}
MODELS = {'dsr': ('diffusion', 'abl_dsr', 'DiffusionSR (pixel-space DDPM)'),
          'fm': ('flow_matching', 'abl_fm', 'Flow-Matching'),
          'ldm': ('ldm', 'abl_ldm', 'LDM (latent DDPM)')}
LDM_WIDTHS = {16: '0.07M', 48: '0.51M', 96: '1.9M', 192: '7.4M', 256: '13.1M'}   # mults (1,), measured


def base(field, norm, epochs):
    return dict(root_folder=DATA, downscale_method='direct', n_steps="'5'", residual_flag='false',
                schedule='linear', timesteps=1000, batch_size=40, epochs=epochs, learning_rate='5.0e-05',
                use_pretrained='false', loss_type='huber', normalize_method=NORMS[norm], fields=FIELDS[field],
                hr_px_um=10, history='consecutive')


def dump(path, header, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n') as f:
        f.write(''.join(f'# {h}\n' for h in header))
        for k, v in d.items():
            f.write(f'{k}: {v}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    tag = 'ablation_v3_smoke' if a.smoke else 'ablation_v3'
    epochs = 1 if a.smoke else 600
    cfg_root = f'diffusionsr/configs/{tag}'
    runs = f'{REPO}/diffusionsr/runs/{tag}'
    train_rows, eval_rows = [], []
    enc_dir = lambda f, n: f'{runs}/encoders/enc_{f}_{n}'
    vae_dir = lambda f, n: f'{runs}/vaes/vae_{f}_{n}'
    out_root = f'eval_results/{tag}'

    # Phase 1: shared encoders and VAEs (4 + 4)
    for f in FIELDS:
        for n in NORMS:
            name = f'abl3_enc_{f}_{n}'
            p = f'{cfg_root}/encoders/{name}.yml'
            dump(p, [f'Ablation v3 shared frozen RRDB encoder | {FIELDS[f]} | {NORMS[n]} | regen x2 20->10 um',
                     'Run: train_srdiff --modeltype encoder --force_enc_dir <enc dir>'],
                 {**base(f, n, epochs), 'encoding': 'true', 'conditioning': 'implicit', 'encoder_type': 'rrdb'})
            train_rows.append(('encoder', name, 'encoder', p, enc_dir(f, n), enc_dir(f, n), 1))
            pe = f'{cfg_root}/evals/{name}.yml'      # the encoder alone, scored like the models
            dump(pe, ['Ablation v3: shared encoder alone (eval only)'],
                 {**base(f, n, epochs), 'encoding': 'true', 'conditioning': 'implicit', 'model_kind': 'abl_enc'})
            eval_rows.append((name, pe, '-', enc_dir(f, n), DATA, 'encoder', 0.0, 1, 1, 'raw', out_root, '-', 3))

            name = f'abl3_vae_{f}_{n}'
            p = f'{cfg_root}/vaes/{name}.yml'
            dump(p, [f'Ablation v3 shared LDM VAE | {FIELDS[f]} | {NORMS[n]} | regen x2 20->10 um',
                     'Run: train_srdiff --modeltype vae (trains vae_results_dir only)'],
                 {**base(f, n, epochs), 'encoding': 'false', 'conditioning': 'none', 'vae_epochs': epochs,
                  'vae_results_dir': vae_dir(f, n)})
            train_rows.append(('vae', name, 'vae', p, vae_dir(f, n), '-', 1))

    # Phase 2: models
    def model(m, enc, f, n, seed=0, width=None, lrcond=False):
        mtype, kind, label = MODELS[m]
        arm = 'enc' if enc else ('lrcond' if lrcond else 'noenc')
        name = f'abl3_{m}_{arm}_{f}_{n}' + (f'_w{width}' if width else '') + (f'_s{seed}' if seed else '')
        # enc: frozen encoder output added after the first conv. noenc: unconditional (no LR input at all).
        # lrcond (as ablation v2): no encoder, conditioned on the upscaled LR, concatenated (DSR/FM) or, for the
        # LDM, through the frozen VAE latent of the upscaled LR re-normalised to HR stats (LDMModel.compute_x_e).
        cond = {'enc': 'implicit', 'noenc': 'none', 'lrcond': 'implicit' if m == 'ldm' else 'explicit'}[arm]
        d = {**base(f, n, epochs), 'model_kind': kind, 'seed': seed,
             'encoding': 'true' if enc else 'false', 'conditioning': cond}
        if m == 'fm':
            d.update(fm_timescale=1000, fm_n_steps=50)
        if m == 'ldm':
            d.update(vae_epochs=epochs, vae_results_dir=vae_dir(f, n))
            if width:
                d.update(latent_unet_dim=width, latent_unet_mults='[1]')
        arm_label = {'enc': 'shared RRDB encoder', 'noenc': 'no encoder, unconditional',
                     'lrcond': 'no encoder, conditioned on the upscaled LR'}[arm]
        hdr = [f'Ablation v3 | {label} | {arm_label} | {FIELDS[f]} | {NORMS[n]}'
               f' | seed {seed}' + (f' | latent U-Net width {width} ({LDM_WIDTHS[width]} params)' if width else ''),
               'regen x2 20->10 um; n_steps 5; generated by scripts/gen_ablation_v3.py']
        p = f'{cfg_root}/models/{name}.yml'
        dump(p, hdr, d)
        ed = enc_dir(f, n) if enc else '-'
        train_rows.append((m, name, mtype, p, f'{runs}/{name}', ed, 2))
        solver, steps = ('euler', 50) if m == 'fm' else ('ddpm', 1000)
        eval_rows.append((name, p, f'{runs}/{name}', ed, DATA, solver, 0.0, steps, 10, 'raw', out_root, '-', 3))

    for m in MODELS:                                  # the 24 v2 arms
        for enc in (True, False):
            for f in FIELDS:
                for n in NORMS:
                    model(m, enc, f, n)
    for m in ('dsr', 'fm'):                           # seed repeats of the DSR/FM encoder arms
        for f in FIELDS:
            for n in NORMS:
                model(m, True, f, n, seed=1)
    model('ldm', True, 'temp', 'std', seed=1)         # LDM size sweep: width 16 is the default arm
    for w in (48, 96, 192, 256):
        for s in (0, 1):
            model('ldm', True, 'temp', 'std', seed=s, width=w)
    for m in MODELS:                                  # LR-conditioned no-encoder arms (added 2026-11-23)
        for f in FIELDS:
            for n in NORMS:
                model(m, False, f, n, lrcond=True)

    if a.smoke:   # one of each kind is enough to exercise every code path
        keep = {'abl3_enc_temp_std', 'abl3_enc_multi_gstd', 'abl3_vae_temp_std', 'abl3_dsr_enc_temp_std',
                'abl3_fm_noenc_multi_gstd', 'abl3_fm_enc_multi_gstd', 'abl3_ldm_enc_temp_std_w48',
                'abl3_dsr_noenc_temp_std', 'abl3_dsr_lrcond_temp_std', 'abl3_fm_lrcond_multi_gstd',
                'abl3_ldm_lrcond_temp_std'}
        train_rows = [r for r in train_rows if r[1] in keep]
        eval_rows = [r for r in eval_rows if r[0] in keep]

    sdir = 'slurm/ablation_v3'
    os.makedirs(sdir, exist_ok=True)
    sfx = '_smoke' if a.smoke else ''
    with open(f'{sdir}/train{sfx}.tsv', 'w', newline='\n') as fh:
        fh.write('kind\tname\tmodeltype\tconfig\trun_dir\tenc_dir\tphase\n')
        fh.writelines('\t'.join(map(str, r)) + '\n' for r in train_rows)
    with open(f'{sdir}/eval{sfx}.tsv', 'w', newline='\n') as fh:
        fh.write('name\tconfig\tmodel_dir\tenc_dir\tdata_root\tsolver\teps\tn_steps\tk\tweights\tout_root\textra\tphase\n')
        fh.writelines('\t'.join(map(str, r)) + '\n' for r in eval_rows)
    print(f'{tag}: {len(train_rows)} trainings, {len(eval_rows)} evals')


if __name__ == '__main__':
    main()
