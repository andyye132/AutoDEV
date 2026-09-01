"""Breadth heatmap: suites (rows) x programs (columns), cell = fraction of targets reached within budget (from results/breadth.json
written by scripts/breadth_table.py). Missing evaluations are hatched. Runs on partial data.

    python scripts/fig_breadth.py --out figures/breadth.png
"""
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="results/breadth.json"); ap.add_argument("--out", default="figures/breadth.png")
ap.add_argument("--programs", default="random,repeat_random,goexplore,cand1_gen037_0.6994.py,cand4_gen035_0.6934.py,macrosweep.py,elites.py,ppo_count,ppo_rnd")
ap.add_argument("--names", default="random=Random,repeat_random=Sticky random,goexplore=Go-Explore,cand1_gen037_0.6994.py=Discovered (best),cand4_gen035_0.6934.py=Discovered (g35),macrosweep.py=Macrosweep,elites.py=Elites,ppo_count=PPO+count,ppo_rnd=PPO+RND")
a = ap.parse_args(); names = dict(kv.split("=", 1) for kv in a.names.split(",") if kv); progs = a.programs.split(",")
M = json.load(open(a.data)); SPL = {"search": "search", "heldout_levels_instances": "held-out lvl", "heldout_class_instances": "held-out cls"}
rows = [(s, spl) for s in M for spl in SPL if spl in M[s]]
Z = np.full((len(rows), len(progs)), np.nan)
for i, (s, spl) in enumerate(rows):
    for j, p in enumerate(progs):
        if p in M[s][spl]: Z[i, j] = M[s][spl][p]["detected"]
fig, ax = plt.subplots(figsize=(1.1 * len(progs) + 2.5, 0.45 * len(rows) + 1.5))
im = ax.imshow(np.where(np.isnan(Z), 0, Z), cmap="viridis", vmin=0, vmax=1, aspect="auto")
for i in range(len(rows)):
    for j in range(len(progs)):
        if np.isnan(Z[i, j]): ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="grey", linewidth=0))
        else: ax.text(j, i, f"{100*Z[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if Z[i, j] < 0.55 else "black")
ax.set_xticks(range(len(progs))); ax.set_xticklabels([names.get(p, p) for p in progs], rotation=35, ha="right", fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{s} / {SPL[spl]}" for s, spl in rows], fontsize=7)
ax.set_title("Targets reached within budget (%) — same frozen programs on every suite", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02); fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=160); print("wrote", a.out, Z.shape)
