# AutoDev — project instructions for Claude

## Standing rules
- Hyak: work ONLY under /gscratch/raivn/andy132; never touch others' files; at most 6 GPUs running at once (count with `squeue -u andy132 -t R`).
- Results must be real: every counted hit replays from reset (verified fitness); racy-era files (`*_racy*`) never enter tables.
- Frozen protocol: autodev/protocol.py (splits, seeds 100-104 eval / 0-2 fitness, budgets, exclusions). Tables regenerate via scripts/make_tables.py, methods_table.py, breadth_table.py, make_bug_table.py, paired_ci.py.
- Git: continuous trailer-free commit+push to origin main is authorized for THIS repo only (user grant, Sep 1 2026).

## Environments
- Laptop: `.venv` (all substrates incl. jumanji run on CPU).
- Hyak env A `conda_envs/autodev` = everything sourced via scripts/slurm/_common.sh (fitness servers, eval_programs, scans). Env B `conda_envs/autodev_multi` = eval_substrate/eval_craftax jobs. Install with the env's ABSOLUTE pip and verify with its absolute python before submitting.

## Sbatch pitfalls (each cost real GPU-hours once)
- `${PROGRAMS-default}`: pass `PROGRAMS=` (set-but-empty) or the built-ins default runs too.
- Check every env var you pass is consumed (header comment lists them); eval_ppo needed a patch for OBJECTIVES.
- SEEDS colon-lists: only eval_server*.sbatch and eval_programs.sbatch convert `0:1:2`; scan sbatches take `--seeds` as an INT.
- TAG concatenates into `--out` patterns — check the pattern before choosing TAG.
- Never touch a queue's work/ while a server is alive; requeue only when no server runs on that queue.
- Result-writing scripts must load-and-merge their out-file (never `res = {}` over a shared JSON).

## Code rules
- Every string replace in an edit script carries an assert on the match (or use a whitespace-tolerant regex).
- Programs slicing the Kinetix feature layout need `initial=` identities on reductions to survive other substrates; exported evolution candidates inherit their run's launch-time seed code — re-apply guards to exports.
- LOG.md rows take their time from `date`, never estimates.
