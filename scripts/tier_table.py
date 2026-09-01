"""Detection by hardness tier. Tiers from results/structural.json: 'constant' = a single held input witnesses the defect (the
constant-input envelope probe), 'explore' = witnessable but no single held input reaches it, plus the excluded categories
(unwitnessable by construction, free hit). Reads evaluation files (program -> split -> first_hit/names) and prints, per program
and split, detection within each tier, so that gains on the tier that needs exploration are visible separately.

    python scripts/tier_table.py --evals runs/hyak/eval_refix1.json,runs/hyak/eval_refix2.json --splits search,heldout_class_instances
"""
import os, sys, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--evals", required=True); ap.add_argument("--structural", default="results/structural.json")
ap.add_argument("--splits", default="search,heldout_levels_instances,heldout_class_instances"); ap.add_argument("--tex", default=""); a = ap.parse_args()
S = json.load(open(a.structural))["instances"]
def tier(n):
    r = S.get(n, {})
    if not r.get("feasible", True): return "unwitnessable"
    if r.get("free_hit_step", -1) >= 0: return "free"
    return "constant" if r.get("constant_input_witness") else "explore"
E = {}
for f in a.evals.split(","):
    if os.path.exists(f):
        J = json.load(open(f)); E.update(J["programs"] if "programs" in J and "cfg" in J else J)     # bench files hold programs under "programs"
rows = []
for p, splits in E.items():
    for spl in a.splits.split(","):
        r = splits.get(spl)
        if not r or "first_hit" not in r or not r.get("names"): continue
        fh = np.array(r["first_hit"]); b = r["budget"]; det = (fh < b).mean(1); names = r["names"]
        out = {}
        for t in ("constant", "explore"):
            m = np.array([tier(n) == t for n in names]); out[t] = (float(det[m].mean()) if m.any() else float("nan"), int(m.sum()))
        rows.append((p, spl, out)); print(f"{p:28s} {spl[:8]:8s} constant-input tier {100*out['constant'][0]:5.1f}% (n={out['constant'][1]})   exploration tier {100*out['explore'][0]:5.1f}% (n={out['explore'][1]})")
if a.tex:
    L = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{llcc}", "\\toprule", "Tester & split & constant-input tier & exploration tier \\\\", "\\midrule"]
    for p, spl, out in rows: L.append(f"{p.replace('_', chr(92) + '_').replace('.py', '')} & {spl.split('_')[0]} & {100*out['constant'][0]:.0f}\\% ({out['constant'][1]}) & {100*out['explore'][0]:.0f}\\% ({out['explore'][1]}) \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\caption{Detection by hardness tier: targets a single held input reaches vs.\\ targets that need exploration (unwitnessable and free-hit instances excluded).}", "\\label{tab:tiers}", "\\end{table}"]
    open(a.tex, "w").write("\n".join(L) + "\n"); print("wrote", a.tex)
