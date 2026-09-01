"""LaTeX transfer table from merged eval JSONs: score ± 95% CI over seeds and detection per split, per program."""
import sys, os, json, argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.exclusions import load_exclusions, mask_rows
ap = argparse.ArgumentParser(); ap.add_argument("--evals", default="runs/hyak_eval_merged.json"); ap.add_argument("--out", default="paper/table_transfer.tex")
ap.add_argument("--only", default="", help="ordered subset of program keys; use '|' between keys to insert a midrule")
ap.add_argument("--label", default="tab:transfer"); ap.add_argument("--caption_extra", default="")
ap.add_argument("--exclusions", default="results/structural.json", help="structural file; unwitnessable + free-hit instances are dropped from denominators ('' = keep all)")
ap.add_argument("--names", default="random=Random,repeat_random=Action-repeat random,goexplore=Go-Explore (seed),initial.py=Go-Explore (seed),best_upto_gen014.py=Evolved (run 1)")
a = ap.parse_args(); rename = dict(x.split("=", 1) for x in a.names.split(",") if "=" in x)
EXCL, _ = load_exclusions(a.exclusions) if a.exclusions else (set(), {}); DROPPED = {}
ev = {}
for f in a.evals.split(","): ev.update(json.load(open(f)))
SP = [("search", "Search"), ("heldout_levels_instances", "Held-out levels"), ("heldout_class_instances", "Held-out class (OOB)")]
def cell(r):
    if not r: return "--"
    fh, nd = mask_rows(r, EXCL); DROPPED[len(r["first_hit"])] = nd; b = r["budget"]; ps = 1 - np.minimum(fh, b).mean(0) / b; det = (fh < b).mean(0)
    ci = 1.96 * ps.std(ddof=1) / np.sqrt(len(ps)) if len(ps) > 1 else 0
    return f"{ps.mean():.3f}$\\pm${ci:.3f} ({100*det.mean():.0f}\\%)"
rows = []
order = [k for k in a.only.replace("|", ",|,").split(",") if k] if a.only else [k for k in ev if not (k == "initial.py" and "goexplore" in ev)]
for p in order:
    if p == "|": rows.append(("|", None)); continue
    if p not in ev: print("missing", p); continue
    rows.append((rename.get(p, p.replace('.py', '').replace('_', '\\_')), [cell(ev[p].get(sp)) for sp, _ in SP]))
lines = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{l" + "c" * len(SP) + "}", "\\toprule", "Tester & " + " & ".join(t for _, t in SP) + " \\\\", "\\midrule"]
for name, cells in rows: lines.append("\\midrule" if name == "|" else f"{name} & " + " & ".join(cells) + " \\\\")
excl_note = (" Instances unwitnessable by construction or witnessed by no-ops alone are excluded from the denominators (" + ", ".join(f"{v} of {k}" for k, v in sorted(DROPPED.items(), reverse=True)) + ").") if any(DROPPED.values()) else ""
lines += ["\\bottomrule", "\\end{tabular}", "\\caption{Score $=1-\\min(\\text{steps},\\text{budget})/\\text{budget}$ (mean $\\pm$ 95\\% CI over 5 seeds) and fraction of planted defects witnessed, per split; budget $128\\times2048$ env steps per instance; size-l configuration, 74 shipped levels." + excl_note + a.caption_extra + "}", f"\\label{{{a.label}}}", "\\end{table}"]
open(a.out, "w").write("\n".join(lines)); print("\n".join(lines[:12]))
