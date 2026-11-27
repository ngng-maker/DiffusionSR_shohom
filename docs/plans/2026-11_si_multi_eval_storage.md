# SI multi-field + FM study: evaluation storage plan (draft, 2026-11-26)

Status: **draft, awaiting your decision.** Nothing below is implemented or submitted yet.

## Problem

The evaluations as designed (1,232 single-stage rows in `slurm/si_multi/eval.tsv` + 262 cascade chains in
`cascade.tsv`) would write about **970 GB** of `predictions.npz` to `/trace/group/forgelab`. That disk is 1.2 TB,
shared by the whole group, has **122 GB free** today, and filled up completely on 2026-11-23 (which killed every
running training). Every eval also uploads its `predictions.npz` to W&B, so W&B would receive the same ~1 TB.

There is a second, hidden problem: **temporary files while sampling.** `sample_resumable`
(`analysis/evaluate_si_schedules.py`) writes uncompressed float32 memory maps into `<out_dir>/partial/` holding all
5 history frames (the model output has `n_steps x F` channels; only frame t is kept at the end). That is
**4–8 GB per running eval** (e.g. n2: pred 6.7 GB + gt/ulr 1.3 GB). With the array throttle at 20, the temporary
files alone would reach **80–160 GB**, more than is free.

## Measured sizes

From a real eval on TRACE (`si_noise_g0p5_enc_temp__euler_eps0p1`, old data, temperature, N = 3,829 frames,
K = 10, 60x60): `predictions.npz` = 439 MB, of which pred 381 MB, ens_std 47 MB (recomputable from pred), gt 11 MB.

| setting | fields | HR px | MB per eval (float32, as now) | eval rows | total GB |
|---|---|---|---|---|---|
| o1 | T | 60 | 440 | 176 | 77 |
| o2 | T + liq | 60 | ~850 | 176 | 150 |
| n1 | T | 64 | ~530 | 176 | 93 |
| n2 | T + liq | 64 | ~1,050 | 176 | 185 |
| n3a / n3b | T | 32 / 64 | ~130 / ~530 | 132 / 132 | 17 / 70 |
| n4a / n4b | T + liq | 32 / 64 | ~260 / ~1,050 | 132 / 132 | 34 / 139 |
| n3 / n4 cascade | T / T + liq | 64 | ~530 / ~1,050 | 131 / 131 | 69 / 138 |
| **all** | | | | **1,494** | **~970** |

(o1 is measured; the rest are scaled from it by pixels x fields x frames.)

## What the analysis actually needs

Every number the results notebooks report is already computed **inside the eval job** from the float32
predictions and written to small files: `*_results.json` (MAE/RMSE, MP/KH depth, area, calibration, spread,
CRPS, spread–skill, rank histogram, NFE, wall clock, model size), `pv_metrics.json` (per P/V), `depth_subpx.json`,
the overlay PNG, and the two GIFs. Together these are about 1–3 MB per eval.

The full prediction arrays are only needed for (a) figures not planned yet and (b) re-scoring with a metric added
later. Both are only realistic for the headline configurations. Also, **any eval can be regenerated exactly**: each
batch is seeded from (seed, batch index), so re-running an eval from its checkpoint reproduces identical
samples. The cost of a deleted prediction file is GPU time, not data.

## Proposal (recommended: option C)

**1. Temporary files: keep frame t only (required for any option).** Slice the model output to the current frame
before writing to `partial/`, so the memory maps are F channels instead of 5F. Peak temporary use drops from
80–160 GB to about **16 GB** at 20 concurrent evals. This is an opt-in flag passed only by the si_multi job
scripts, so the parallel session's ablation v3 evals (same script, possibly with partial files on disk) are
unaffected.

**2. Keep prediction arrays only for headline evals (126 of 1,494).** Headline = per trained model its default
setting, plus every baseline:

- each SI model (4 schedules x 2 starts) and each FM model at Euler, eps = 0, 20 steps;
- encoder alone, DDPM matched (EMA and raw, ancestral 1,000 steps), DiffusionSR;
- each cascade chain at the same default, plus the cascade baselines.

That is 12 per SI setting x 8 settings + 8 FM + 22 cascade = **126 evals**. Every other eval keeps the small files
above (all metrics, P/V tables, overlay, GIFs) but no `predictions.npz`, and its W&B artifact holds only those
small files.

**3. Store the headline arrays compactly (option C only).** Drop `ens_std` (recomputable), store `gt` once per
setting, and store `pred` as 16-bit integers at 0.25 K steps with an offset (range −4,096 to 12,287 K), plus an
exact float32 side list for the few values outside that range. Predictions do contain such outliers: the test file
above has 464 values above 16,383 K (max 278,000 K) and 326,000 below 0 K, so plain clipping would change the
metrics (ensemble-mean MAE 43.78 vs 42.73 K). With the side list nothing is clipped, and every value is within
0.125 K of float32. Tested on that file: pred shrinks from 381 MB to 108 MB (3.5x). A small loader
(`eval_extras.load_predictions`) returns float32 arrays from either format, so notebooks read old and new files
the same way. Metrics are still computed from the float32 arrays inside the job, so they are unaffected.

| option | what is kept | on disk | on W&B |
|---|---|---|---|
| A. as designed | float32 predictions for all 1,494 evals | ~970 GB | ~970 GB |
| B. headline only, float32 | arrays for 126 evals, small files for all | ~85 GB | ~85 GB |
| **C. headline only, 16-bit (recommended)** | same, compact format | **~25 GB** | **~25 GB** |

Peak usage under C: ~25 GB of headline arrays + ~3 GB of small files + ~16 GB of temporary files ≈ **45 GB**,
against 122 GB free minus the ~40 GB the running trainings still need. B (~85 GB + 16 GB temporary) would not
leave a safe margin on the shared disk.

## Code changes (for C)

1. `evaluate_si_schedules.sample_resumable`: optional slice to frame t before writing `partial/` (new flag).
2. `--save_predictions {full,none}` and the compact format in `evaluate_si_schedules.py` and
   `evaluate_cascade.py`; the W&B artifact includes `predictions.npz` only when it was saved.
3. The "already done, skip sampling" check currently requires `predictions.npz`; switch it to the results JSON
   plus a `done` marker so metrics-only evals are not resampled after a requeue.
4. `gen_si_multi.py`: a column marking headline rows; regenerate `eval.tsv` / `cascade.tsv` (row order unchanged).
5. `eval_extras.load_predictions` (reads float32 and 16-bit files).
6. One smoke eval on TRACE (small n3a row, one headline and one metrics-only) before submitting anything.

## Optional, separate decision

Older studies' local prediction files: `eval_results/si_schedules` (57 GB) and `eval_results/si_noise` (31 GB)
were uploaded to W&B (one artifact per eval; not yet verified). After checking that each artifact exists, the local copies could
be removed to give the group more room. This is a permanent deletion, so it waits for your explicit OK.

## Decisions needed

1. Option C (recommended), B, or something else.
2. Is the headline set right? The alternative is to add Heun eps = 0 per SI model (+64 evals, ~+12 GB under C).
