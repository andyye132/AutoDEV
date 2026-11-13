# Born, Not Trained

Code, data and paper source for **"Born, Not Trained: Zero-Training Programs Out-Explore Reinforcement Learning
for Game Testing at Equal Budget"** (Workshop for Autonomous ML Research, NeurIPS 2026).

The paper: [`paper/PAPER_camera_ready.pdf`](paper/PAPER_camera_ready.pdf)

## What this is

An LLM-driven evolutionary search starts from a simple Go-Explore program and improves it over 36 rounds. The best
program it found is then run, unchanged, on four games it never saw (DoorKey, FourRooms, Breakout, Sokoban). With the
same number of game steps for every method it reaches 91% of the targets, against 54% for PPO trained per game. PPO
needs 4 to 16 times as many steps to catch up. The games are small, the program relies on saving and reloading game
states, and the gain on actual bug finding (planted bugs in Kinetix) is modest. The paper also describes a bug in our
own evaluation code that inflated an early result, and how replaying detections exposed it.

## Where things are

| Path | Contents |
|---|---|
| `programs/` | The programs the paper refers to: the best program, the Go-Explore base, and three others |
| `paper/` | Paper source, the compiled PDF, and the script that makes the figures |
| `autodev/` | The test harness, game adapters, planted-bug benchmark, replay check, PPO baseline |
| `evolve/` | The evolutionary search setup (ShinkaEvolve), the base program, hand-written programs |
| `scripts/` | Evaluation and table scripts, cluster job files |
| `runs/` | Logged results: per-target outcomes for every method, and the main search database |
| `results/` | Summary tables and analysis outputs built from `runs/` |
| `tests/` | Tests for the harness, the planted bugs and PPO |
| `LOG.md` | The timestamped research log the driver agent kept during the project |
| `notes/` | Planning and hand-off notes from the project, kept as written |
| `archive/` | Superseded paper drafts and side experiments |

## Reproducing the figures

Figures 2 and 3 are built from the logged results. No simulator is needed.

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy matplotlib
python paper/figs/make_figs.py        # writes paper/figs/fig_results.pdf, fig_evolution.pdf, stats.json
```

`paper/figs/stats.json` then holds every number quoted from those figures, with its interval. Figure 1 is
`paper/figs/pipeline.tex` (`pdflatex pipeline.tex`). To build the paper: `cd paper && latexmk -pdf neurips_2026.tex`.

Result files used:

| Result | Files |
|---|---|
| Figure 2, evolved program and simple baselines | `runs/hyak/eval_xmini_doorkey.json`, `eval_xmini_fourrooms.json`, `eval_gymnax_breakout.json`, `eval_jumanji_sokoban.json` |
| Figure 2, PPO at 1x, 4x, 16x | `runs/ppo_<game>.json`, `runs/ppo_<game>_tuned.json`, `runs/ppo_<game>_4x_tuned.json`, `runs/ppo_<game>_16x_tuned.json` |
| Figure 3, the search | `runs/evolve_run7_multi/programs.sqlite` |
| Table 2, planted bugs in Kinetix | `runs/hyak/eval_fixed_merged.json`, summarised in `results/tables/table_transfer_fixed.tex` |
| Craftax, MountainCar, DeepSea | `runs/hyak/eval_craftax_pilot2.json`, `eval_gymnax_mc.json`, `eval_gymnax_ds24.json`, `runs/ppo_craftax*.json` |

## Re-running the experiments

This needs JAX and the simulators (`pip install -r requirements.txt`). The small games run on a CPU; the Kinetix
benchmark was run on GPUs.

Run the best program and the simple baselines on one of the unseen games:

```bash
AUTODEV_SPLIT=runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json \
python scripts/eval_baselines.py --substrate xminigrid:MiniGrid-DoorKey-8x8 \
  --programs random,repeat_random,goexplore --program_files programs/best_program.py \
  --splits search,heldout_levels_instances --seeds 0,1,2,3,4 \
  --workers 64 --steps 2048 --slots 128 --chunk 8 --out runs/my_doorkey.json
```

Run PPO on the same game with the same number of steps:

```bash
python scripts/eval_ppo_substrate.py --substrate xminigrid:MiniGrid-DoorKey-8x8 \
  --split runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json --which search,heldout_levels_instances \
  --objectives count,rnd,icm,curiosity_proximity,zero --seeds 100,101,102,103,104 \
  --num_updates 32 --lr 0.0001 --ent_coef 0.2 --out runs/my_ppo_doorkey.json
```

The other games use `xminigrid:MiniGrid-FourRooms`, `gymnax:Breakout-MinAtar` and `jumanji:Sokoban-v0`, with the
matching split file in `runs/substrates/`. The search itself is launched by `scripts/launch_run7.sh`; it calls the
Claude models through ShinkaEvolve and scores programs on a GPU server (`scripts/eval_server.py`).

The evaluation rules for the Kinetix benchmark (levels, seeds, step counts, exclusions) are fixed in
`autodev/protocol.py`. The replay check is `verify_chain_harness` in `autodev/tester.py`.

## Notes

- Paths in some scripts and job files refer to the cluster and laptop the project was run on.
- `LOG.md`, `notes/` and `archive/` are working records. They include plans and intermediate claims that were later
  corrected; the paper is the authoritative account.
