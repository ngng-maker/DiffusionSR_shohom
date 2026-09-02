---
name: slurm-job
description: Whenever creating or submitting a SLURM job for TRACE, produce complete PowerShell commands covering SSH connection, cd, conda activate, the sbatch/bash command, and job verification — never give a partial command sequence.
---

## Behaviour

Whenever the user asks to create, submit, run, or requeue a SLURM job on TRACE, you MUST output the **complete, copy-pasteable PowerShell command sequence** below. Never output only the `sbatch` line; always include every step from opening the SSH connection to confirming the job is in the queue.

---

## Required command sequence (PowerShell on Windows)

### Step 1 — SSH into TRACE

```powershell
ssh ngng@trace
```

All subsequent commands are run **inside that SSH session** (i.e., on the TRACE login node). Present them in a single code block so the user can paste them all at once after connecting:

### Steps 2–5 — Inside the SSH session (bash)

```bash
# 2. Go to the repo
cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom

# 3. Activate the conda environment
conda activate diffusion_SR

# 4. Create the log directory (if it doesn't exist yet for this experiment)
mkdir -p logs/<experiment_folder>

# 5. Submit the job
#    — Use bash ... for a master submit script that chains dependencies
bash slurm/<path_to_submit_script>.sh
#    — OR use sbatch directly for a single script
sbatch slurm/<path_to_script>.sh
```

### Step 6 — Verify the job is in the queue

```bash
squeue -u ngng
```

Expected output: one or more lines with your job IDs, job names, partition `batch`, and status `PD` (pending) or `R` (running). If the output is empty, the submission failed — check the script for errors with `cat logs/<experiment_folder>/setup_<JID>.log`.

---

## Rules

1. **Always provide ALL six steps** — Steps 1 through 6, in order, every time. Never omit steps 1–3 even if the user "just wants the sbatch command".
2. **Fill in real paths** — Replace `<path_to_submit_script>` and `<experiment_folder>` with the actual paths relevant to this session's context (e.g., `slurm/ablation_20260901/submit_all.sh`, `logs/ablation_20260901`).
3. **Add `--resume_from_wandb` for requeues** — If the user is resubmitting a preempted job, add `--resume_from_wandb <WANDB_RUN_NAME>` to the relevant `sbatch` line and explain which task IDs to target with `--array=<id>`.
4. **For job arrays** — After `squeue -u ngng`, also show how to watch a specific task: `squeue -u ngng -j <ARRAY_JID>`.
5. **Provide a single contiguous SSH-session block** — Steps 2–6 should appear inside one bash code block (after the initial `ssh` command) so the user can paste them into the terminal at once without switching blocks mid-session.
6. **Note the login-node restriction** — Remind the user that `trace-login05` has **no GPU**; never run training or inference directly on the login node.

---

## Template (copy this structure every time)

```
# From Windows PowerShell:
ssh ngng@trace

# Then inside the SSH session:
cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
conda activate diffusion_SR
mkdir -p logs/<experiment_folder>
bash slurm/<experiment_folder>/submit_all.sh   # or sbatch slurm/.../<script>.sh
squeue -u ngng
```

Replace the bracketed placeholders with the actual values for the current task. If submitting individual array tasks instead of a master script, use `sbatch` and include the `--dependency` and `--array` flags as appropriate.
