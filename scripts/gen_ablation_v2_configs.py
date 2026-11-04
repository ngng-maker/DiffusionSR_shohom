"""Generate the 36 ablation_v2 configs: 3 models × 3 conditioning × 2 fields × 2 normalize.

Conditioning arms: enc (frozen RRDB encoder), noenc (unconditional), lrcond (bicubic-upscaled LR,
no encoder; added 2026-11 to match the paper's no-encoder ablation).
"""
from pathlib import Path
import yaml

OUT = Path(__file__).parent.parent / 'diffusionsr' / 'configs' / 'ablation_v2'
OUT.mkdir(parents=True, exist_ok=True)

TRACE = '/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom'
DATA  = '/trace/group/forgelab/ngng/multifield/data_fields'
ABL   = f'{TRACE}/diffusionsr/runs/ablation_20260901'
V2    = f'{TRACE}/diffusionsr/runs/ablation_v2'

BASE = dict(
    root_folder       = DATA,
    downscale_method  = 'direct',
    n_steps           = '5',
    residual_flag     = False,
    schedule          = 'linear',
    timesteps         = 1000,
    batch_size        = 40,
    epochs            = 600,
    learning_rate     = 5e-5,
    use_pretrained    = False,
    loss_type         = 'huber',
)

FIELDS = {
    'temp':  'temperature',
    'multi': 'temperature_sdfliqlabel',
}

ENC_DIRS = {
    ('temp',  'std'):  f'{ABL}/enc_temp',
    ('temp',  'gstd'): f'{ABL}/enc_temp_global',
    ('multi', 'std'):  f'{ABL}/enc_multifield',
    ('multi', 'gstd'): f'{ABL}/enc_multifield_global',
}

NORM_METHOD = {
    'std':  'standardize',
    'gstd': 'global_standardize',
}

FM_EXTRA  = dict(fm_timescale=1000, fm_n_steps=50)

def ldm_extra(field_key, norm_key):
    return dict(
        vae_epochs   = 600,
        vae_results_dir = f'{V2}/vae_{field_key}_{norm_key}',
    )

# Conditioning for the encoder-free, LR-conditioned arm ('lrcond'): the paper's no-encoder ablation
# (configs/conditioning_ablation/*_noencoder.yml) conditions on the bicubic-upscaled LR rather than
# nothing. Pixel-space models concatenate it with x_t ('explicit'): implicit addition would need the
# conditioning to have init_dim (64) channels, but upscaled LR has n_steps*n_fields. LDM encodes it with
# its frozen VAE and adds the latent, exactly like its encoder arm ('implicit', see LDMModel.compute_x_e).
LRCOND_CONDITIONING = {'diffusion': 'explicit', 'fm': 'explicit', 'ldm': 'implicit'}

for model in ('diffusion', 'fm', 'ldm'):
    for enc in ('enc', 'noenc', 'lrcond'):
        for field_key in ('temp', 'multi'):
            for norm_key in ('std', 'gstd'):
                cfg = dict(BASE)
                cfg['normalize_method'] = NORM_METHOD[norm_key]
                cfg['fields']           = FIELDS[field_key]

                if enc == 'enc':
                    cfg['encoding']           = True
                    cfg['conditioning']       = 'implicit'
                    cfg['encoder_results_dir'] = ENC_DIRS[(field_key, norm_key)]
                elif enc == 'lrcond':
                    cfg['encoding']     = False
                    cfg['conditioning'] = LRCOND_CONDITIONING[model]
                else:
                    cfg['encoding']    = False
                    cfg['conditioning'] = 'none'

                if model == 'fm':
                    cfg.update(FM_EXTRA)
                elif model == 'ldm':
                    cfg.update(ldm_extra(field_key, norm_key))

                # header comment
                model_label = {'diffusion': 'DiffusionSR (pixel-space DDPM)',
                               'fm':        'Flow-Matching',
                               'ldm':       'LDM (latent DDPM)'}[model]
                enc_label   = {'enc':    'RRDB encoder',
                               'noenc':  'no encoder',
                               'lrcond': 'no encoder, conditioned on bicubic-upscaled LR'}[enc]
                field_label = 'temperature only' if field_key == 'temp' else 'temperature + sdfliqlabel'
                norm_label  = NORM_METHOD[norm_key]
                comment = (
                    f'# {model_label} | {enc_label} | {field_label} | {norm_label}\n'
                    f'# Ablation v2: n_steps=5, epochs=600, batch_size=40\n'
                )

                fname = OUT / f'{model}_{enc}_{field_key}_{norm_key}.yml'
                with open(fname, 'w') as f:
                    f.write(comment)
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                print(f'wrote {fname.name}')

print(f'\nDone — {len(list(OUT.glob("*.yml")))} configs in {OUT}')
