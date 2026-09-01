"""Figures for the paper: (1) search fitness vs generation, (2) transfer bars (search / held-out levels / held-out class),
(3) detection curves (fraction of planted bugs witnessed vs env steps) per tester and split, (4) per-class detection."""
import sys, os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--eval", default="runs/hyak/eval_programs.json"); ap.add_argument("--curve", default="runs/evolve_overnight/exports/curve.json")
ap.add_argument("--out", default="figures"); ap.add_argument("--names", default="", help="comma-separated display-name mapping old=new")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rename = dict(x.split("=") for x in a.names.split(",") if "=" in x)
def nm(k): return rename.get(k, k.replace(".py", ""))
SPLITS = [("search", "search (seen classes, seen levels)"), ("heldout_levels_instances", "held-out levels"), ("heldout_class_instances", "held-out class (OOB)")]
if os.path.exists(a.curve):
    c = json.load(open(a.curve))["curve"]
    plt.figure(figsize=(5, 3.2)); gens = [x["gen"] for x in c]; best = [x["best"] for x in c]
    allpts = [(x["gen"], s) for x in c for s in x["gen_scores"] if s is not None]
    if allpts: plt.scatter([p[0] for p in allpts], [p[1] for p in allpts], s=10, alpha=0.4, label="candidates")
    plt.plot(gens, best, lw=2, label="best so far"); plt.xlabel("generation"); plt.ylabel("search fitness (1 − steps/budget)"); plt.legend(); plt.tight_layout(); plt.savefig(f"{a.out}/fitness_curve.png", dpi=200); plt.close()
    print("fitness_curve.png")
if os.path.exists(a.eval):
    ev = json.load(open(a.eval)); progs = list(ev.keys())
    # transfer bars with 95% CI over seeds (mean of per-instance normalized score per seed)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (sp, title) in zip(axes, SPLITS):
        means, cis = [], []
        for p in progs:
            r = ev[p].get(sp)
            if not r: means.append(np.nan); cis.append(0); continue
            fh = np.array(r["first_hit"]); budget = r["budget"]
            per_seed = 1 - np.minimum(fh, budget).mean(0) / budget
            means.append(per_seed.mean()); cis.append(1.96 * per_seed.std(ddof=1) / np.sqrt(len(per_seed)) if len(per_seed) > 1 else 0)
        ax.bar(range(len(progs)), means, yerr=cis, capsize=3); ax.set_xticks(range(len(progs))); ax.set_xticklabels([nm(p) for p in progs], rotation=30, ha="right", fontsize=8); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("score (1 − steps to witness / budget)"); plt.tight_layout(); plt.savefig(f"{a.out}/transfer_bars.png", dpi=200); plt.close(); print("transfer_bars.png")
    # detection curves
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (sp, title) in zip(axes, SPLITS):
        for p in progs:
            r = ev[p].get(sp)
            if not r: continue
            fh = np.array(r["first_hit"]).reshape(-1); budget = r["budget"]; grid = np.linspace(0, budget, 200)
            ax.plot(grid, [(fh < g).mean() for g in grid], label=nm(p))
        ax.set_title(title, fontsize=9); ax.set_xlabel("env steps"); 
    axes[0].set_ylabel("fraction of planted bugs witnessed"); axes[-1].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{a.out}/detection_curves.png", dpi=200); plt.close(); print("detection_curves.png")
    # per-class detection on search + held-out levels
    classes = sorted({c for p in progs for sp, _ in SPLITS for c in (ev[p].get(sp) or {}).get("detected_per_class", {})})
    fig, ax = plt.subplots(figsize=(7, 3.2)); w = 0.8 / len(progs)
    for i, p in enumerate(progs):
        vals = []
        for c in classes:
            v = [ev[p][sp]["detected_per_class"].get(c) for sp, _ in SPLITS if ev[p].get(sp) and c in ev[p][sp]["detected_per_class"]]
            vals.append(np.mean(v) if v else np.nan)
        ax.bar(np.arange(len(classes)) + i * w, vals, w, label=nm(p))
    ax.set_xticks(np.arange(len(classes)) + 0.4); ax.set_xticklabels(classes); ax.set_ylabel("fraction detected"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{a.out}/per_class_detection.png", dpi=200); plt.close(); print("per_class_detection.png")
