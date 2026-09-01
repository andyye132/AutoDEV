"""Rival discovery methods at equal fitness: best score vs evaluations for the evolutionary runs (ShinkaEvolve DBs), the agentic
loop and tree-search scaffolds (nodes.jsonl), one-shot expert designs (points: their equal-fitness score from the fitness-server
outbox, at evaluation 1), and constant tuning / restarts as horizontal bands. All on the same GPU fitness (size-l search split,
3 seeds, 128x1024) — only runs scored on that fitness are drawn; local-fitness runs (1-3) are excluded by default.

    python scripts/fig_scaffolds.py --out figures/scaffolds.png
"""
import os, json, glob, sqlite3, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--evo", default="evolutionary run 4 (seed: run-1 best)=runs/evolve_run4_gpu;evolutionary run 5 (seed: movertarget)=runs/evolve_run5_mover;evolutionary run 6 (reachable split)=runs/evolve_run6_reach")
ap.add_argument("--scaffolds", default="agentic loop=runs/scaffold_loop_pilot;tree search=runs/scaffold_tree_pilot;agentic loop (full)=runs/scaffold_loop_full;tree search (full)=runs/scaffold_tree_full")
ap.add_argument("--outbox", default="runs/hyak/evalq/outbox", help="fitness-server results; design_*.json are the one-shot designs")
ap.add_argument("--tuning", default="runs/hyak/tuning_cand1_gen037_0.6994.json"); ap.add_argument("--out", default="figures/scaffolds.png")
a = ap.parse_args()
fig, ax = plt.subplots(figsize=(7.2, 4.0))


def best_curve(scores):
    b, cur = [], -1e9
    for s in scores: cur = max(cur, s); b.append(cur)
    return b


for spec in a.evo.split(";"):
    label, run = spec.split("=", 1); db = os.path.join(run, "programs.sqlite")
    if not os.path.exists(db): continue
    rows = sqlite3.connect(db).execute("select combined_score from programs where combined_score is not null order by timestamp").fetchall()
    sc = [r[0] for r in rows]
    if len(sc) < 2: continue
    ax.step(range(len(sc)), best_curve(sc), where="post", label=f"{label}: {sc[0]:.3f} → {max(sc):.3f} ({len(sc)} evals)")
for spec in a.scaffolds.split(";"):
    label, run = spec.split("=", 1); f = os.path.join(run, "nodes.jsonl")
    if not os.path.exists(f): continue
    nodes = [json.loads(l) for l in open(f)]
    sc = [n["result"]["score"] if n.get("result") else np.nan for n in nodes]
    valid = [s for s in sc if not np.isnan(s)]
    if len(valid) < 2: continue
    filled = []; cur = -1e9
    for s in sc:
        if not np.isnan(s): cur = max(cur, s)
        filled.append(cur if cur > -1e8 else np.nan)
    ax.step(range(len(sc)), filled, where="post", linestyle="--", label=f"{label}: {valid[0]:.3f} → {max(valid):.3f} ({len(sc)-1} evals)")
pts = []
for f in sorted(glob.glob(os.path.join(a.outbox, "*.json"))):
    try:
        j = json.load(open(f)); name = j.get("program", os.path.basename(f))
        if name.startswith("design_") and "score" in j and not j.get("error"): pts.append((name[len("design_"):], j["score"]))
    except Exception: pass
if pts:
    ax.scatter([1] * len(pts), [p[1] for p in pts], marker="x", color="k", zorder=5, label="one-shot expert designs (1 eval each)")
    for n, s in pts: ax.annotate(n, (1, s), fontsize=6, xytext=(4, 0), textcoords="offset points")
if os.path.exists(a.tuning):
    T = json.load(open(a.tuning)); sc = [r["score"] for r in T["results"][1:]]
    if sc: ax.axhspan(min(sc), max(sc), color="grey", alpha=0.15, label=f"constant tuning of the run-1 best ({len(sc)} trials, 2-seed fitness)")
ax.set_xlabel("fitness evaluations"); ax.set_ylabel("best search fitness so far (GPU fitness)"); ax.grid(alpha=0.3); ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=160); print("wrote", a.out, "| designs:", pts)
