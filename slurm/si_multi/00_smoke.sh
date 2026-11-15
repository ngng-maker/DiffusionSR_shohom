#!/bin/bash
#SBATCH -J simulti_smoke
#SBATCH -p batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/si_multi/smoke_%j.log
#
# End-to-end smoke test of the SI multi-field / regen / FM pipeline on real data, 1 epoch per model:
#   0. dataset checks: cascade upscaling == stored LR/direct/2x; every root loads (1 and 2 fields)
#   1. encoders (1 epoch) for n2 (40->10, 2 fields), n3a (40->20) and n3b (20->10)
#   2. trainings (1 epoch): SI linear-enc, SI bridge-ulr, DDPM matched, DiffusionSR, FM noise, FM paired on n2;
#      SI linear-enc on n3a and n3b
#   3. evals on 8 test frames, K = 3, with all extras: every model above, FM midpoint/dopri5 + a 5-step run,
#      ancestral DDPM (EMA and raw), encoder alone, and one O1 extras_only back-fill on a copy
#   4. cascade n3a -> n3b, heun eps 0.1 and euler
# Everything is written under *_smoke paths with names smoke_simulti_*; their W&B runs and artifacts are
# deleted at the end. Usage, from the repo root on TRACE:  sbatch slurm/si_multi/00_smoke.sh

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_multi
export SI_WANDB_GROUP=si_multi_smoke
S=diffusionsr/runs/si_multi_smoke; E=eval_results/si_multi_smoke; C=diffusionsr/configs/si_multi_smoke
rm -rf "$S" "$E" "$C"; mkdir -p "$S" "$E" "$C"
FAIL=0
step() { echo; echo "######## $* ########"; }
check() { if "$@"; then echo "PASS: $*"; else echo "FAIL: $*"; FAIL=1; fi; }

step "0. dataset checks"
check python - <<'PY'
import numpy as np, torch
from diffusionsr.analysis.evaluate_cascade import upscale_like_dataset
from diffusionsr.datasets.flipped_dataset import FlippedDataset
R = '/trace/group/forgelab/ngng/multifield/datasets/regen_2d'
FIELDS = ['temperature', 'liqlabel', 'pressure', 'vx', 'vy', 'vz']
import glob, os
worst = 0.0
for root, f in (('x2_20to10', 2), ('x2_40to20', 2), ('x4_40to10', 4)):
    p1 = sorted(glob.glob(f'{R}/{root}/test/LR/direct/1x/*/_*.npy'))[::400][:10]
    for p in p1:
        lr = np.load(p); up_ref = np.load(p.replace('/1x/', f'/{f}x/'))
        x = torch.tensor(lr).permute(2, 0, 1)[None].double()
        up = upscale_like_dataset(x, up_ref.shape[:2], FIELDS)[0].permute(1, 2, 0).numpy()
        d = np.abs(up - up_ref).max(axis=(0, 1)) / (np.abs(up_ref).max(axis=(0, 1)) + 1e-9)
        worst = max(worst, float(d.max()))
print('max relative upscale mismatch', worst)
for root in ('x4_40to10', 'x2_40to20', 'x2_20to10'):
    for fields in (['temperature'], ['temperature', 'sdfliqlabel']):
        ds = FlippedDataset(split='test', downscale_method='direct', root_folder=f'{R}/{root}', normalize='standardize',
                            n_steps=1, field_names=fields, p_flip=0.0)
        r, hr, lr, ulr = ds[0][:4]
        print(root, fields, len(ds), ds.factor, tuple(hr.shape), tuple(lr.shape), tuple(ulr.shape))
assert worst < 1e-4, worst
# n_steps 5 with consecutive history: channels = frames t-4..t of the same simulation, ambient before frame 1
import numpy as np, os
for root, hist in (('/trace/group/forgelab/ngng/multifield/data_fields', 'consecutive'), (f'{R}/x4_40to10', 'consecutive')):
    ds = FlippedDataset(split='test', downscale_method='direct', root_folder=root, normalize='standardize',
                        n_steps=5, field_names=['temperature'], p_flip=0.0, history=hist)
    P = list(ds.hr_paths)
    cond = lambda p: os.path.dirname(p)
    first = next(i for i in range(1, len(P)) if cond(P[i]) != cond(P[i - 1]))
    for i, want in ((first + 6, [first + 2, first + 3, first + 4, first + 5, first + 6]),
                    (first + 1, [None, None, None, first, first + 1])):
        un = ds.unscale_data(np.asarray(ds[i][1]), input_type='hr')
        for ch, k in enumerate(want):
            ref = np.full(un[ch].shape, float(ds.baseline_hr[..., 0].mean())) if k is None else np.minimum(np.load(P[k])[..., 0], 8000)
            ok = bool(np.allclose(un[ch], ref, atol=0.5))
            print(os.path.basename(root), 'frame', i, 'slot', ch, '<-', 'ambient' if k is None else os.path.basename(P[k]), ok)
            assert ok
PY

step "1-2. smoke configs, encoders and trainings (1 epoch)"
python - <<'PY'
import yaml, os
src, dst, runs = 'diffusionsr/configs/si_multi', 'diffusionsr/configs/si_multi_smoke', 'diffusionsr/runs/si_multi_smoke'
full = os.path.abspath(runs)
def put(setting, name, new=None, enc=True):
    cfg = yaml.safe_load(open(f'{src}/{setting}/{name}.yml' if setting != 'encoders' else f'{src}/encoders/{name}.yml'))
    if enc and setting != 'encoders':
        cfg['encoder_results_dir'] = f'{full}/enc_{setting}'
    cfg['epochs'] = 1
    yaml.safe_dump(cfg, open(f'{dst}/{new or name}.yml', 'w'), sort_keys=False)
for s in ('n2', 'n3a', 'n3b'):
    put('encoders', f'enc_{s}')
for n in ('linear_enc', 'bridge_ulr', 'ddpm_matched', 'diffusionsr', 'fm_noise', 'fm_paired'):
    put('n2', f'sim_n2_{n}')
put('n2', 'encoder_only')
for s in ('n3a', 'n3b'):
    put(s, f'sim_{s}_linear_enc')
print(sorted(os.listdir(dst)))
PY
DATA_OF() { python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['root_folder'])" "$1"; }
for s in n2 n3a n3b; do
  check python -m diffusionsr.runners.train_srdiff --data_root "$(DATA_OF $C/enc_$s.yml)" --config "$C/enc_$s.yml" \
    --modeltype encoder --gpu 0 --force_enc_dir "$PWD/$S/enc_$s" --wandb_run_name "smoke_simulti_enc_$s" --epochs 1
done
train() {   # name setting modeltype
  local RESUME="--wandb_run_id smoke-simulti-$1 --ckpt_artifact smoke_simulti_$1_ckpt"
  case "$1" in *ddpm_matched*|*diffusionsr*) RESUME="--resume_from_wandb smoke_simulti_$1" ;; esac
  check python -m diffusionsr.runners.train_srdiff --data_root "$(DATA_OF $C/$1.yml)" --config "$C/$1.yml" \
    --modeltype "$3" --gpu 0 --force_enc_dir "$PWD/$S/enc_$2" --force_run_dir "$PWD/$S/$1" \
    --wandb_run_name "smoke_simulti_$1" $RESUME --epochs 1
}
for n in linear_enc bridge_ulr ddpm_matched fm_noise fm_paired; do train sim_n2_$n n2 coupled; done
train sim_n2_diffusionsr n2 diffusion
train sim_n3a_linear_enc n3a coupled
train sim_n3b_linear_enc n3b coupled

step "3. evals: 8 frames, K = 3, all extras"
R2=/trace/group/forgelab/ngng/multifield/datasets/regen_2d/x4_40to10
ev() {   # name solver eps n_steps weights k [extra args]
  local n=$1; shift
  check python -m diffusionsr.analysis.evaluate_si_schedules --train_name "smoke_simulti_$n${WTAG:-}" --config "$C/$n.yml" \
    --model_dir "$S/$n" --enc_dir "$S/enc_n2" --data_root "$R2" --solver "$1" --eps "$2" --n_steps "$3" \
    --weights "$4" --k_samples "$5" --max_frames 8 --frames_per_batch 4 --plot_frame_idx -1 --out_root "$E" \
    --no_upload "${@:6}"
}
ev sim_n2_linear_enc heun 0.1 20 raw 3
ev sim_n2_linear_enc dopri5 0 20 raw 3
ev sim_n2_bridge_ulr euler 0.5 20 raw 3
ev sim_n2_fm_noise midpoint 0 20 raw 3
ev sim_n2_fm_paired euler 0 5 raw 3
ev sim_n2_fm_paired dopri5 0 20 ema 3
WTAG=_ema ev sim_n2_ddpm_matched ddpm 0 1000 ema 3
WTAG=_raw ev sim_n2_ddpm_matched ddpm 0 1000 raw 3
ev sim_n2_diffusionsr ddpm 0 1000 raw 3
check python -m diffusionsr.analysis.evaluate_si_schedules --train_name smoke_simulti_encoder_only --config "$C/encoder_only.yml" \
  --model_dir "$S/_none" --enc_dir "$S/enc_n2" --data_root "$R2" --solver encoder --n_steps 1 --k_samples 1 \
  --max_frames 8 --plot_frame_idx -1 --out_root "$E" --no_upload
# O1 extras_only back-fill on a COPY of one existing SI-schedules eval (the original is not touched)
mkdir -p "$E/o1copy" && cp -r eval_results/si_schedules/si_sched_linear_enc_temp__euler_eps0 "$E/o1copy/"
check python -m diffusionsr.analysis.evaluate_si_schedules --train_name si_sched_linear_enc_temp \
  --config diffusionsr/configs/si_schedules/si_sched_linear_enc_temp.yml \
  --model_dir diffusionsr/runs/si_schedules/si_sched_linear_enc_temp --enc_dir "$CL_ENC_DIR" --data_root "$CL_DATA" \
  --solver euler --eps 0 --out_root "$E/o1copy" --extras_only --no_upload

step "4. cascade n3a -> n3b"
for args in "heun 0.1" "euler 0"; do
  set -- $args
  check python -m diffusionsr.analysis.evaluate_cascade --chain_name smoke_simulti_n3_linear_enc_cascade \
    --a_config "$C/sim_n3a_linear_enc.yml" --a_model_dir "$S/sim_n3a_linear_enc" --a_enc_dir "$S/enc_n3a" \
    --a_data_root /trace/group/forgelab/ngng/multifield/datasets/regen_2d/x2_40to20 \
    --b_config "$C/sim_n3b_linear_enc.yml" --b_model_dir "$S/sim_n3b_linear_enc" --b_enc_dir "$S/enc_n3b" \
    --b_data_root /trace/group/forgelab/ngng/multifield/datasets/regen_2d/x2_20to10 \
    --solver "$1" --eps "$2" --k_samples 3 --max_frames 8 --out_root "$E"
done

step "summary"
python - <<'PY'
import json, glob, os
for f in sorted(glob.glob('eval_results/si_multi_smoke/**/*_results.json', recursive=True)):
    r = json.load(open(f))
    d = os.path.dirname(f)
    have = [x for x in ('pv_metrics.json', 'depth_subpx.json', 'trajectory.gif', 'simtime.gif') if os.path.exists(os.path.join(d, x))]
    print(os.path.basename(d)[:48].ljust(48), 'ensMAE %7.1f' % r.get('MAE_ensemble_mean', float('nan')),
          'nfe', r.get('nfe_per_sample'), 's/sample %.4f' % r.get('sec_per_sample', float('nan')),
          'params', r.get('params_total'), 'IoU', r.get('IoU_liquid_sdfliqlabel'), have,
          'ERR' if r.get('extras_errors') else '', r.get('extras_errors', ''))
PY

step "cleanup W&B smoke runs/artifacts"
python - <<'PY'
import os, wandb
api = wandb.Api(timeout=60); ent = os.environ.get('WANDB_ENTITY'); proj = 'Flow3D_SuperResolution'
n = 0
for r in api.runs(f'{ent}/{proj}', filters={'display_name': {'$regex': '^smoke_simulti'}}):
    for a in r.logged_artifacts():
        try:
            a.delete(delete_aliases=True); n += 1
        except Exception as e:
            print('artifact', a.name, e)
    r.delete(); n += 1
print('deleted', n, 'W&B objects')
PY
echo; [ $FAIL = 0 ] && echo "SMOKE PASSED" || echo "SMOKE FAILED (see FAIL lines)"
