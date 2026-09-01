"""Best-so-far search fitness by generation for several evolution runs, read from each run's ShinkaEvolve programs.sqlite.
Scores are on each run's own fitness (different splits/budgets), so curves are shown relative to the run's seed program.

    python scripts/fig_search_runs.py --runs "run 1 (local, OOB held out)=runs/evolve_overnight;run 3 (local, ANOMALY held out)=runs/evolve_run3_rot;run 4 (GPU fitness)=runs/evolve_run4_gpu" --out figures/search_runs.png
"""
import os, sys, argparse, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="run 1 (local fitness, OOB held out)=runs/evolve_overnight;run 3 (local fitness, ANOMALY held out)=runs/evolve_run3_rot;run 4 (GPU fitness, from run-1 best)=runs/evolve_run4_gpu;run 5 (GPU fitness, from movertarget)=runs/evolve_run5_mover")
ap.add_argument("--out", default="figures/search_runs.png"); ap.add_argument("--relative", action="store_true", help="plot score minus the seed's score")
a = ap.parse_args()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for spec in a.runs.split(";"):
    label, run = spec.split("=", 1); db = os.path.join(run, "programs.sqlite")
    if not os.path.exists(db): print("skip", run); continue
    rows = sqlite3.connect(db).execute("select generation, combined_score from programs where combined_score is not null order by generation, timestamp").fetchall()
    if not rows: continue
    gens = sorted({g for g, _ in rows}); seed = rows[0][1]
    best = []; cur = -1e9
    for g in gens:
        cur = max(cur, max(s for gg, s in rows if gg == g)); best.append(cur - seed if a.relative else cur)
    ax.step(gens, best, where="post", label=f"{label}: seed {seed:.3f} → best {max(s for _, s in rows):.3f} ({len(rows)} programs)")
ax.set_xlabel("generation"); ax.set_ylabel("best search fitness so far" + (" (minus seed)" if a.relative else "")); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=160); print("wrote", a.out)
