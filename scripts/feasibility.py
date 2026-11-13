"""Feasibility-normalized detection. The oracle-guided probe (scripts/eval_guided.py) says which planted instances are reachable
within budget when the tester is told where the witness is; detection is then reported over that feasible set, per program and
split, next to the raw detection. Also reports how many feasible instances no evaluated tester ever finds.

    python scripts/feasibility.py --guided runs/hyak/guided_ceiling.json --evals runs/hyak_eval_merged.json \
        --out results/feasibility.json --tex results/tables/table_feasibility.tex
"""
import os, sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--guided", default="runs/hyak/guided_ceiling.json"); ap.add_argument("--evals", default="runs/hyak_eval_merged.json")
ap.add_argument("--out", default="results/feasibility.json"); ap.add_argument("--tex", default="results/tables/table_feasibility.tex")
ap.add_argument("--names", default="random=Random,repeat_random=Action-repeat random,goexplore=Go-Explore (seed),cand1_gen037_0.6994.py=Evolved run 1 gen 37,"
                                   "macrosweep.py=Opus: macrosweep,ppo_count=PPO+count")
ap.add_argument("--programs", default="", help="comma-separated subset for the LaTeX table (default: those in --names)")
ap.add_argument("--structural", default="results/structural.json", help="per-instance structural feasibility (scripts/structural_feasibility.py); unwitnessable instances are reported per split")
ap.add_argument("--union", action="store_true", help="feasible = reached by the guided probe OR by any evaluated tester in any seed (the guided probe alone is not an upper bound)")
a = ap.parse_args()
G = json.load(open(a.guided)); E = json.load(open(a.evals))
STRUCT = json.load(open(a.structural))["instances"] if a.structural and os.path.exists(a.structural) else {}
names = dict(kv.split("=", 1) for kv in a.names.split(",") if kv)
progs = [p for p in (a.programs.split(",") if a.programs else names) if p in E]
SPL = {"search": "Search", "heldout_levels_instances": "Held-out levels", "heldout_class_instances": "Held-out class (OOB)"}
out = {}
for spl, g in G.items():
    gfh = np.array(g["first_hit"]); budget = g["budget"]; gnames = g["names"]
    feasible = (gfh < budget).any(1)                       # reachable in any guided seed
    pos = {n: i for i, n in enumerate(gnames)}
    if a.union:
        for p, splits in E.items():
            if spl not in splits or not splits[spl].get("names"): continue
            fh = np.array(splits[spl]["first_hit"]); found = (fh < splits[spl]["budget"]).any(1)
            for n, f in zip(splits[spl]["names"], found):
                if f and n in pos: feasible[pos[n]] = True
    row = {"n": len(gnames), "n_feasible": int(feasible.sum()), "feasible_frac": float(feasible.mean()), "programs": {}}
    union = np.zeros(len(gnames), bool)
    for p, splits in E.items():
        if spl not in splits: continue
        r = splits[spl]; fh = np.array(r["first_hit"]); b = r["budget"]
        rn = r.get("names")
        if rn and set(rn) == set(gnames): idx = np.array([pos[n] for n in rn])
        elif fh.shape[0] == len(gnames): idx = np.arange(len(gnames))
        else: print(f"skip {p}/{spl}: {fh.shape[0]} instances vs {len(gnames)} in guided file"); continue
        found_any = np.zeros(len(gnames), bool); found_any[idx] = (fh < b).any(1)
        found_seed = np.zeros(len(gnames)); found_seed[idx] = (fh < b).mean(1)
        union |= found_any
        row["programs"][p] = {"detected": float(found_seed.mean()), "detected_feasible": float(found_seed[feasible].mean()) if feasible.any() else None,
                              "detected_any_seed_feasible": float(found_any[feasible].mean()) if feasible.any() else None,
                              "found_infeasible_any_seed": int((found_any & ~feasible).sum())}
    row["union_any_tester_any_seed"] = float(union.mean())
    row["feasible_never_found"] = int((feasible & ~union).sum())
    unwit = np.array([not STRUCT.get(n, {}).get("feasible", True) for n in gnames])
    row["unwitnessable_by_construction"] = int(unwit.sum()); row["unwitnessable_names"] = [n for n, u in zip(gnames, unwit) if u]
    row["unreached_not_provably_infeasible"] = int((~union & ~unwit).sum())
    assert not (unwit & union).any(), "an unwitnessable instance was reached: structural rule is wrong"
    row["found_but_guided_missed"] = int((union & ~feasible).sum())
    out[spl] = row
    print(f"{SPL.get(spl, spl):22s} n={row['n']:3d} feasible {row['feasible_frac']:.2f} ({row['n_feasible']}) | union of testers {row['union_any_tester_any_seed']:.2f} | "
          f"feasible-never-found {row['feasible_never_found']} | found-but-guided-missed {row['found_but_guided_missed']} | unwitnessable-by-construction {row['unwitnessable_by_construction']} | unreached-not-provably-infeasible {row['unreached_not_provably_infeasible']}")
    for p, v in row["programs"].items():
        if p in progs: print(f"    {names.get(p, p):26s} detected {v['detected']:.2f} -> feasible-normalized {v['detected_feasible']:.2f}")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
spls = [s for s in SPL if s in out]
L = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{l" + "cc" * len(spls) + "}", "\\toprule",
     "Tester & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SPL[s]}}}" for s in spls) + " \\\\",
     " & " + " & ".join("raw & feasible" for _ in spls) + " \\\\", "\\midrule",
     ("Feasible within budget (guided probe or any tester) & " if a.union else "Feasible within budget (guided probe) & ") + " & ".join(f"\\multicolumn{{2}}{{c}}{{{out[s]['feasible_frac']*100:.0f}\\% ({out[s]['n_feasible']}/{out[s]['n']})}}" for s in spls) + " \\\\",
     "\\midrule"]
for p in progs:
    cells = []
    for s in spls:
        v = out[s]["programs"].get(p)
        cells += [f"{v['detected']*100:.0f}\\%", f"{v['detected_feasible']*100:.0f}\\%"] if v and v["detected_feasible"] is not None else ["--", "--"]
    L.append(f"{names.get(p, p)} & " + " & ".join(cells) + " \\\\")
L += ["\\midrule", "Union of all testers, any seed & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{out[s]['union_any_tester_any_seed']*100:.0f}\\%}}" for s in spls) + " \\\\",
      "Unwitnessable by construction & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{out[s]['unwitnessable_by_construction']} of {out[s]['n']}}}" for s in spls) + " \\\\",
      "\\bottomrule", "\\end{tabular}",
      "\\caption{Detection within budget, raw and normalized by the feasible set (instances the oracle-guided probe" + (" or any evaluated tester" if a.union else "") + " reaches in any seed). "
      "Feasible-but-never-found: " + ", ".join(f"{out[s]['feasible_never_found']} ({SPL[s].lower()})" for s in spls) + ".}", "\\label{tab:feasibility}", "\\end{table}"]
os.makedirs(os.path.dirname(a.tex), exist_ok=True); open(a.tex, "w").write("\n".join(L) + "\n"); print("wrote", a.out, a.tex)
