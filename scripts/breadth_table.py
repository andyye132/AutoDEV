"""Accuracy across breadth: one matrix of suites x programs from every evaluation file (Kinetix K-Defect / K-Goal / K-Fresh /
rotated, Gymnax, XLand-MiniGrid, Craftax pilot). Detection within budget (mean over seeds) with a 95% CI over seeds, per split.
Writes results/breadth.json and results/tables/table_breadth.tex. Runs on whatever files exist.

    python scripts/breadth_table.py
"""
import os, json, glob, argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.exclusions import load_exclusions, mask_rows
EXCL, _ = load_exclusions()

ap = argparse.ArgumentParser()
ap.add_argument("--files", default="runs/hyak_eval_merged.json=K-Defect,runs/hyak/eval_goal.json=K-Goal,runs/hyak/eval_ppo_goal.json=K-Goal,runs/hyak/eval_fresh.json=K-Fresh,runs/hyak/eval_run3_rot.json=K-Defect-rot,runs/hyak/eval_baselines_rot.json=K-Defect-rot,"
                                   "runs/hyak/eval_gymnax_mc.json=G-MountainCar,runs/hyak/eval_gymnax_ac.json=G-Acrobot,runs/hyak/eval_gymnax_ds.json=G-DeepSea8,runs/hyak/eval_gymnax_ds24.json=G-DeepSea24,"
                                   "runs/hyak/eval_xmini_doorkey.json=X-DoorKey,runs/hyak/eval_xmini_fourrooms.json=X-FourRooms,runs/hyak/eval_xmini_unlock.json=X-Unlock,runs/hyak/eval_craftax_pilot2.json=C-Ach")
ap.add_argument("--programs", default="random,repeat_random,goexplore,cand1_gen037_0.6994.py,cand4_gen035_0.6934.py,macrosweep.py,elites.py,ppo_count,ppo_rnd")
ap.add_argument("--names", default="random=Random,repeat_random=Action-repeat,goexplore=Go-Explore,cand1_gen037_0.6994.py=Discovered (r1 best),cand4_gen035_0.6934.py=Discovered (r1 g35),macrosweep.py=Opus macrosweep,elites.py=Elites,ppo_count=PPO+count,ppo_rnd=PPO+RND")
ap.add_argument("--out", default="results/breadth.json"); ap.add_argument("--tex", default="results/tables/table_breadth.tex")
a = ap.parse_args(); names = dict(kv.split("=", 1) for kv in a.names.split(",") if kv); progs = a.programs.split(",")
M = {}   # suite -> split -> program -> (det, ci, n)
for spec in a.files.split(","):
    f, suite = spec.split("=", 1)
    if not os.path.exists(f): continue
    E = json.load(open(f))
    for p, splits in E.items():
        if "first_hit" in splits: splits = {"search": splits}      # flat files (substrate pilots) hold one split
        for spl, r in splits.items():
            if "first_hit" not in r: continue
            fh, _nd = mask_rows(r, EXCL); b = r["budget"]; det = (fh < b).mean(0)
            ci = 1.96 * det.std(ddof=1) / np.sqrt(len(det)) if len(det) > 1 else 0.0
            M.setdefault(suite, {}).setdefault(spl, {})[p] = dict(detected=float(det.mean()), ci=float(ci), n=int(fh.shape[0]), seeds=int(fh.shape[1]), budget=b)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(M, open(a.out, "w"), indent=1)
SPL_ORDER = ["search", "heldout_levels_instances", "heldout_class_instances"]; SPL_SHORT = {"search": "srch", "heldout_levels_instances": "h-lvl", "heldout_class_instances": "h-cls"}
rows = [(suite, spl) for suite in M for spl in SPL_ORDER if spl in M[suite]]
print(f"{'suite/split':26s} " + " ".join(f"{names.get(p, p)[:12]:>12s}" for p in progs))
for suite, spl in rows:
    d = M[suite][spl]; print(f"{suite + '/' + SPL_SHORT[spl]:26s} " + " ".join(f"{(str(round(100*d[p]['detected'])) + '%') if p in d else '--':>12s}" for p in progs) + f"   (n={next(iter(d.values()))['n']})")
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{l" + "c" * len(progs) + "}", "\\toprule", "Suite / split & " + " & ".join(names.get(p, p) for p in progs) + " \\\\", "\\midrule"]
for suite, spl in rows:
    d = M[suite][spl]; L.append(f"{suite} / {SPL_SHORT[spl]} (n={next(iter(d.values()))['n']}) & " + " & ".join(f"{100*d[p]['detected']:.0f}$\\pm${100*d[p]['ci']:.0f}" if p in d else "--" for p in progs) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\caption{Accuracy across breadth: fraction of targets reached within the suite's budget (\\%, mean $\\pm$ 95\\% CI over seeds) for the same frozen programs on every suite; -- = not evaluated.}", "\\label{tab:breadth}", "\\end{table}"]
os.makedirs(os.path.dirname(a.tex), exist_ok=True); open(a.tex, "w").write("\n".join(L) + "\n"); print("wrote", a.out, a.tex)
