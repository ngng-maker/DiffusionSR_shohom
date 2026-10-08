# Flip / Normalization Order Audit

> **Status: code-analysis complete — quantitative figures pending TRACE run.**
> Do not modify any training code until this report is reviewed and a go/no-go decision is made.

---

## 1. Order of Operations

### CNN Encoder (`train_rrdn_encoder.py`)

`SimulationXZDataset.__getitem__` applies pixel-wise standardization **before** the
tensor is returned.  The training loop then applies the flip on the already-normalized
tensor:

```python
# train_rrdn_encoder.py  lines 143–150
_, hr, lr, upscaled_lr = batch
flip = False
if np.random.uniform() < 0.2:           # p = 0.2, NOT 0.25
    hr = torch.flip(hr, dims=[2])        # W-axis flip on NORMALIZED hr
    lr = torch.flip(lr, dims=[2])        # W-axis flip on NORMALIZED lr
    flip = True
# upscaled_lr and residual are NEVER flipped
```

**Correct order** would be: load raw → upsample → flip → normalize.
Because normalization statistics are pixel-wise `(C, H, W)` arrays (each spatial cell
has its own mean and std), swapping the order introduces a spatially-varying bias.
Specifically, after a W-axis flip the pixel at `[:, i, j]` now carries the statistics of
`[:, i, W-1-j]`, not `[:, i, j]`.

### Diffusion / Flow Matching / LDM (`train_diffusion.py`, `train_flow_matching.py`, `train_ldm.py`)

**No flip augmentation anywhere.**  Confirmed by `grep -r "torch.flip" diffusionsr/runners/`.

---

## 2. Pair Consistency

Inside the encoder training loop:

| Tensor | Flipped? |
|--------|----------|
| `hr`   | ✓ (`dims=[2]`) |
| `lr`   | ✓ (`dims=[2]`) |
| `upscaled_lr` | ✗ |
| `residual` (`hr - upscaled_lr`) | ✗ — but re-derived from the now-inconsistent `hr` and un-flipped `upscaled_lr` |

`upscaled_lr` is the bicubic-upsampled version of `lr`.  When the loop flips `hr` and
`lr` but not `upscaled_lr`, the residual passed to `x_e` during training is computed
from mismatched sources.

---

## 3. Flip Axis Clarification

The flip is `torch.flip(hr, dims=[2])` on a tensor of shape `(C, H, W)`:
- `H` = x-length (laser scan direction, plotted horizontally)
- `W` = z-depth (depth into substrate, plotted vertically with `origin='lower'`)

`dims=[2]` therefore flips the **z / depth axis**, not the laser-direction axis.
A depth-flip mirrors the melt pool upside-down (surface ↔ deep solid), which
is physically implausible as a data-augmentation strategy.
*(A laser-direction flip — `dims=[1]` — would be the physically motivated choice.)*

---

## 4. Statistical Asymmetry (requires TRACE run)

Section populated by `diffusionsr/analysis/audit_flip_norm.py`.

**Metric reported:**  For the HF-mean statistics array `mean_hr` of shape `(C, H, W)`:

```
asymmetry_K   = |mean_hr - flip_W(mean_hr)|    # (C, H, W)  in Kelvin
asymmetry_rel = asymmetry_K / std_hr           # fraction of one std
```

Because the melt-pool sits near the top surface (small z) and temperature decays
monotonically with z, `mean_hr` is **highly asymmetric along W**.  We therefore expect
`asymmetry_K` to be large (O(hundreds of K)) near the pool and ~0 far from it.

*Figures saved to artifact `audit_flip_norm_single_field` / `audit_flip_norm_multifield`
on W&B run `audit_flip_norm_<tag>`.*

---

## 5. Effect Size (requires TRACE run)

Section populated by `diffusionsr/analysis/audit_flip_norm.py`.

**Metric:**  For each of N=50 training frames, compute the mean per-pixel absolute
difference (in normalized units and Kelvin) between:
- `FtN` — flip then normalize (correct)
- `NtF` — normalize then flip (code behaviour)

```
err = | (flip(x) - mean) / std  -  flip((x - mean) / std) |
    = | flip(x)/std - mean/std  -  flip(x)/std + flip(mean)/std |
    = | (flip(mean) - mean) / std |
    = asymmetry_rel
```

So the per-pixel effect size in normalized units **equals the statistical asymmetry**.
The error is deterministic (does not depend on the sample), only on the spatial position
of the flipped pixel relative to the statistics map.

*Results reported in JSON and figures in the W&B artifact.*

---

## 6. Evaluation Path

Evaluation (all notebooks NB12–NB15, `eval_predictions.py`) calls
`model.batch_sample()` on test-set batches obtained from `SimulationXZDataset(split='test')`.

- **No flip applied during evaluation** — confirmed.
- **Statistics used** are those cached from the training split, loaded by the dataset
  constructor regardless of split.  The same stats are therefore used at train and test time.
- Because the diffusion model receives no flip at all (Section 1), train/eval are
  consistent for the diffusion models.
- For the CNN encoder: the encoder was trained with a post-normalization flip, so
  the learned statistics are slightly mismatched, but evaluation is still flip-free.

---

## 7. Which Models Are Affected?

| Model | Flip augmentation? | Ordering bug? | Missing-pair bug (upscaled_lr not flipped)? |
|---|---|---|---|
| RRDB CNN encoder | ✓ p=0.2 | ✓ flip post-norm | ✓ residual inconsistent |
| Conditional DDPM | ✗ | n/a | n/a |
| Flow Matching | ✗ | n/a | n/a |
| LDM | ✗ | n/a | n/a |
| MobileNet baseline | *check `train_mobilenet.py`* | — | — |

---

## 8. Recommendations (pending quantitative results)

**a) Should baselines be retrained?**

If the effect size from Section 5 is small (mean error ≪ 1 std ≈ O(10 K) in
background, O(100 K) near pool), the existing checkpoints are usable.  If large
(mean error > 0.5 std), retrain the encoder with the corrected ordering before
drawing conclusions about conditioning vs. no-conditioning comparisons.

**b) Proposed fix — coupled flip before normalization**

In `train_rrdn_encoder.py`, move the flip to operate on **raw Kelvin tensors** before
the normalization step.  Because `SimulationXZDataset.__getitem__` normalizes on
return, the cleanest fix is to add a `return_raw` mode to the dataset, or to add
a pre-normalization transform hook.  An alternative is to invert the normalization
inline (using the cached `mean_hr`, `std_hr` accessible via `dataset.mean_hr`), apply
the flip, and re-normalize — functionally equivalent.

**c) Flip axis correction**

Change `dims=[2]` → `dims=[1]` to flip the laser-direction axis (x) rather than the
depth axis (z).  This makes the augmentation physically meaningful (symmetric laser
scan) and is less likely to corrupt depth-dependent statistics.

**d) Diffusion models**

Consider adding a coupled flip augmentation to `train_diffusion.py` so the diffusion
model also benefits from the augmentation.  This must use the corrected order (raw →
flip → normalize) and flip all four tensors (`hr`, `lr`, `upscaled_lr`, `res`) jointly.

---

## Appendix: Script and Notebook

| File | Purpose |
|------|---------|
| `diffusionsr/analysis/audit_flip_norm.py` | Runs on TRACE; produces figures and JSON |
| `diffusionsr/notebooks/16_audit_flip_norm.ipynb` | Downloads W&B artifact and renders results |

TRACE commands:

```bash
# Single-field
python -m diffusionsr.analysis.audit_flip_norm \
    --data_root  /trace/group/forgelab/ngng/multifield/data_fields \
    --downscale  direct \
    --field_names temperature \
    --tag         single_field \
    --out_dir    /trace/group/forgelab/ngng/multifield/eval_outputs/audit_flip_norm \
    --wandb_entity $WANDB_ENTITY \
    --n_samples  50

# Multifield
python -m diffusionsr.analysis.audit_flip_norm \
    --data_root  /trace/group/forgelab/ngng/multifield/data_fields \
    --downscale  direct \
    --field_names temperature sdfliqlabel \
    --tag         multifield \
    --out_dir    /trace/group/forgelab/ngng/multifield/eval_outputs/audit_flip_norm \
    --wandb_entity $WANDB_ENTITY \
    --n_samples  50
```
