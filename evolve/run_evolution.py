"""Launch ShinkaEvolve on the tester-program search. Run from the repo root.
   python evolve/run_evolution.py --generations 100 --results_dir runs/evolve_x [--models claude-sonnet-5]
Needs ANTHROPIC_API_KEY (or OPENAI_API_KEY for gpt models, or 'headless/claude' with the claude CLI installed)."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AUTODEV_ROOT", ROOT)
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

TASK_SYS_MSG = """You are improving a game-testing program for 2D physics puzzle games (Kinetix, JAX). A batch of B workers
explores one level; the harness keeps K archive slots of saved states. Your program defines four pure-JAX hooks:
init(rng, F, n_act, K, B) -> mem; restore(mem, feats, arch_feats, arch_valid, arch_visits, rng) -> (do (B,), slot (B,), mem);
act(mem, feats, obs, rng) -> (action (B,), mem); archive(mem, feats, done, arch_feats, arch_valid, rng) -> (do (B,), slot (B,), mem).
The fitness is how quickly the workers reach states that expose planted defects (leaving the arena, passing through a solid,
winning via an unintended goal, runaway physics, touching a disarmed hazard) across many levels — measured as
1 - steps_to_first_witness/budget, averaged. A held-out defect class and held-out levels are NOT in the fitness; programs
that overfit the visible classes will not transfer, so prefer general exploration strategies (systematic coverage of
position/velocity space, reaching extreme states, high-energy interactions, varied action patterns, archive diversity).
Rules: only jax/jnp; all shapes fixed (no Python control flow on traced values); keep mem a dict of arrays with fixed shapes;
actions are integers in [0, n_act); never index arch arrays out of range; the program must run in a few seconds per level.
Only modify code between the EVOLVE-BLOCK markers.

Scoring: your fitness is the mean over targets of (your normalized first-hit score) / (the best score any known program achieved
on that target), capped at 2.0 - so points come from the targets in the feedback list where you are far below best-known, not from
polishing already-fast targets. The feedback line also gives your paired delta vs the reference program on identical tasks and seeds;
treat a delta within its +- interval as noise, not signal. Design hint from the last run: the single mutation that produced a real,
transfer-verified gain replaced open-loop action holds with CLOSED-LOOP feedback control (bang-bang on state features). Feedback on
state beats blind schedules; consider what the program can measure and react to."""

ap = argparse.ArgumentParser()
ap.add_argument("--generations", type=int, default=100); ap.add_argument("--results_dir", default="runs/evolve")
ap.add_argument("--models", default="headless/claude@claude-sonnet-5?effort=medium"); ap.add_argument("--eval_jobs", type=int, default=2); ap.add_argument("--proposal_jobs", type=int, default=4)
ap.add_argument("--islands", type=int, default=2); ap.add_argument("--archive_size", type=int, default=40); ap.add_argument("--max_cost", type=float, default=60.0)
args = ap.parse_args()

evo = EvolutionConfig(task_sys_msg=TASK_SYS_MSG, num_generations=args.generations, job_type="local", language="python",
                      llm_models=args.models.split(","), init_program_path=os.environ.get("AUTODEV_INIT", os.path.join(ROOT, "evolve", "initial.py")),
                      results_dir=args.results_dir, max_api_costs=args.max_cost, embedding_model=None, meta_rec_interval=10,
                      use_text_feedback=os.environ.get("AUTODEV_TEXT_FEEDBACK", "0") == "1",
                      novelty_llm_models=(["headless/claude@claude-sonnet-5?effort=low"] if os.environ.get("AUTODEV_NOVELTY", "0") == "1" else None))
job = LocalJobConfig(eval_program_path=os.path.join(ROOT, "evolve", "evaluate.py"), python_executable=sys.executable)
db = DatabaseConfig(num_islands=args.islands, archive_size=args.archive_size, elite_selection_ratio=0.3, num_archive_inspirations=2, num_top_k_inspirations=2,
                    migration_interval=10, migration_rate=0.1, parent_selection_strategy="weighted")
runner = ShinkaEvolveRunner(evo_config=evo, job_config=job, db_config=db, max_evaluation_jobs=args.eval_jobs, max_proposal_jobs=args.proposal_jobs, verbose=True)
runner.run()
