# AutoDev — an autonomous loop that discovers game-testing programs and finds real bugs in Kinetix levels

Target: NeurIPS 2026 AutoMLR workshop (paper due Sep 4, 5 PM PDT). See `proposal/PROPOSAL.md` and `LOG.md` (research ledger).

**Pipeline**
1. `autodev/defects.py` plants 5 bug classes into Kinetix levels (OOB, TUNNEL, SHORTCUT, ANOMALY, DISARMED); `autodev/oracles.py` gives each a witness state.
2. `autodev/tester.py` runs a *tester program* (hooks: restore / act / archive over save-restorable states) on a level with B workers; every first hit carries an action chain replayable from reset (`chain_actions`, `replay_chain`).
3. `scripts/build_pool.py` + `scripts/make_split.py` build the planted-bug pool and the search / held-out-levels / held-out-class split.
4. `evolve/` — ShinkaEvolve searches over the program in `evolve/initial.py` with fitness = speed of witnessing planted bugs on the search split (`autodev/search_eval.py`).
5. `scripts/scan_clean.py` runs testers on CLEAN levels with generic engine oracles (`autodev/anomalies.py`: OOB, PENETRATE, JOINT_TEAR, BLOWUP) → real-bug candidates, replay-verified.
6. PPO + intrinsic-reward testers (`autodev/ppo.py`, `autodev/objectives.py`) are hand-designed baselines.

**Local dev**: `python3 -m venv .venv && . .venv/bin/activate && pip install "jax[cpu]" kinetix-env optax shinka-evolve`, then `python tests/test_tester.py`.
**Hyak**: `scripts/hyak_setup.sh`, then `sbatch scripts/slurm/{throughput,pool,scan,evolve}.sbatch` (fill `scripts/slurm/_common.sh`).

## Remote fitness (GPU search without an LLM on the cluster)

`scripts/eval_server.py` (SLURM: `scripts/slurm/eval_server.sbatch`) watches `runs/hyak/evalq/inbox/` on Hyak and scores each
tester program on the size-l search split. On the laptop, `AUTODEV_REMOTE_EVAL=1 python evolve/run_evolution.py ...` makes
`evaluate_hooks` ship every candidate through `autodev/remote_eval.py` (scp + poll) instead of simulating locally, so the LLM
proposals stay on the laptop and the fitness runs on the GPU (see `scripts/launch_run4.sh`).
