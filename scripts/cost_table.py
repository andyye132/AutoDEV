"""Cost per target (C5.2): for each tester and split, the median simulator steps to first witness over the instances that
ALL compared testers find (so the comparison is on common ground), the fraction found, and the harness wall-clock per instance
from the evaluation logs. Writes results/cost.json and results/tables/table_cost.tex.

    python scripts/cost_table.py --evals runs/hyak_eval_merged.json --programs repeat_random,goexplore,cand1_gen037_0.6994.py,macrosweep.py,ppo_count,ppo_rnd
"""
import os, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--evals", default="runs/hyak_eval_merged.json"); ap.add_argument("--programs", default="repeat_random,goexplore,cand1_gen037_0.6994.py,macrosweep.py,ppo_count,ppo_rnd")
ap.add_argument("--names", default="repeat_random=Action-repeat random,goexplore=Go-Explore (seed),cand1_gen037_0.6994.py=Discovered (run 1 best),macrosweep.py=Opus: macrosweep,ppo_count=PPO+count,ppo_rnd=PPO+RND")
ap.add_argument("--out", default="results/cost.json"); ap.add_argument("--tex", default="results/tables/table_cost.tex")
a = ap.parse_args(); names = dict(kv.split("=", 1) for kv in a.names.split(",") if kv)
E = json.load(open(a.evals)); progs = [p for p in a.programs.split(",") if p in E]
SPL = {"search": "Search", "heldout_levels_instances": "Held-out levels", "heldout_class_instances": "Held-out class"}
out = {}
for spl, title in SPL.items():
    have = [p for p in progs if spl in E[p]]
    if not have: continue
    # common instances: found (in every seed's mean sense: found in >= half of seeds) by all compared testers
    found = {}
    for p in have:
        fh = np.array(E[p][spl]["first_hit"]); b = E[p][spl]["budget"]; found[p] = (fh < b).mean(1) >= 0.5
    common = np.logical_and.reduce([found[p] for p in have]); out[spl] = {"n_common": int(common.sum()), "programs": {}}
    for p in have:
        fh = np.array(E[p][spl]["first_hit"]); b = E[p][spl]["budget"]
        med = float(np.median(np.where(fh[common] < b, fh[common], np.nan)[~np.isnan(np.where(fh[common] < b, fh[common], np.nan))])) if common.any() else float("nan")
        wall = E[p][spl].get("wall"); n = fh.shape[0] * fh.shape[1]
        out[spl]["programs"][p] = {"median_steps_common": med, "detected": float((fh < b).mean()), "wall_s_per_instance_seed": (wall / n) if wall else None, "budget": b}
        print(f"{title:16s} {names.get(p, p):26s} median steps (common {int(common.sum())}) {med:9.0f}  detected {float((fh < b).mean()):.2f}  wall/inst-seed {((wall / n) if wall else float('nan')):.1f}s")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
spls = [s for s in SPL if s in out]
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{l" + "rr" * len(spls) + "}", "\\toprule",
     "Tester & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SPL[s]} (common $n$={out[s]['n_common']})}}" for s in spls) + " \\\\",
     " & " + " & ".join("median steps & found" for _ in spls) + " \\\\", "\\midrule"]
for p in progs:
    cells = []
    for s in spls:
        v = out[s]["programs"].get(p); cells += [f"{v['median_steps_common']/1000:.1f}k", f"{v['detected']*100:.0f}\\%"] if v else ["--", "--"]
    L.append(f"{names.get(p, p)} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\caption{Cost per target: median simulator steps to the first witness over the instances every compared tester finds, and the fraction of all instances found, per split (5 seeds; RL testers 3 seeds).}", "\\label{tab:cost}", "\\end{table}"]
os.makedirs(os.path.dirname(a.tex), exist_ok=True); open(a.tex, "w").write("\n".join(L) + "\n"); print("wrote", a.out, a.tex)
