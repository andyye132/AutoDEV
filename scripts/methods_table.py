"""LaTeX table of the exploration procedures on the method bench: easy subset (constant-input tier) and exploration-tier
hard bench, from results/bench/*.json (scripts/bench_methods.py output; fixed harness, verified hits).

    python scripts/methods_table.py --easy results/bench/gpu_round0.json,results/bench/gpu_round1.json,results/bench/gpu_round2.json \
        --hard results/bench/gpu_hard0.json,results/bench/gpu_hard1.json --out paper/table_methods.tex
"""
import json, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--easy", required=True); ap.add_argument("--hard", required=True)
ap.add_argument("--out", default="paper/table_methods.tex"); a = ap.parse_args()

NAMES = {  # program key -> (display name, family)
 "repeat_random": ("Action-repeat random (hand-designed)", "baseline"), "initial": ("Go-Explore seed", "baseline"),
 "cand1_gen037_0.6994": ("Run-1 program (exploit era)", "baseline"), "pairladder": ("Pair-ladder (LLM design)", "baseline"),
 "macrosweep": ("Macrosweep (LLM design)", "baseline"),
 "zeta_greedy": ("$\\zeta$-greedy holds", "coherence"), "splitting": ("Rare-event splitting", "rare-event"),
 "cem_plans": ("CEM over plans", "optimisation"), "kpiece": ("KPIECE archive", "planning"), "directed": ("Directed power schedule", "fuzzing"),
 "goexplore_zeta": ("Go-Explore + $\\zeta$", "hybrid"), "ucb_tree": ("UCB tree", "planning"), "map_elites": ("MAP-Elites", "QD"),
 "velmax": ("Velocity maximisation", "energy"), "contact_branch": ("Contact branching", "contact"), "goexplore_plr": ("Go-Explore + PLR", "hybrid"),
 "split_pump": ("Split-and-pump", "hybrid"), "goexplore_yield": ("Go-Explore + yield credit", "hybrid"), "cem_root": ("Root CEM", "optimisation"),
 "deathaware": ("Death-aware checkpointing", "termination"), "velalign": ("Velocity-aligned drive", "energy"), "steer": ("Learned effects + steering", "planning"),
 "resonance": ("Resonance sweep", "energy"), "saturate": ("Actuator saturation", "coherence"),
 "split_pump_vel": ("Split-and-pump + velocity", "hybrid$^2$"), "cem_root_ratchet": ("Root CEM + ratchet", "hybrid$^2$"),
}
def load(paths):
    P, cfg = {}, None
    for p in paths.split(","):
        try: d = json.load(open(p))
        except FileNotFoundError: continue
        cfg = cfg or d["cfg"]; P.update(d["programs"])
    return P, cfg
E, ec = load(a.easy); H, hc = load(a.hard)
keys = [k for k in NAMES if k in E or k in H]
def cell(P, k, split, det=True):
    r = P.get(k, {}).get(split)
    if r is None: return "--"
    return f"{r['score']:.3f}\\,/\\,{100*r['detected']:.0f}" if det else f"{100*r['detected']:.0f}"
rows = sorted(keys, key=lambda k: -(H.get(k, {}).get("search", {}).get("score", -1)))
L = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabular}{llrrrrr}\\toprule",
     " & & \\multicolumn{2}{c}{Constant-input tier (48)} & \\multicolumn{3}{c}{Exploration tier (43/12/16)} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
     "Procedure & family & search & sealed cls & search & held-out lvl & sealed cls \\\\ \\midrule"]
for k in rows:
    nm, fam = NAMES[k]
    L.append(f"{nm} & {fam} & {cell(E,k,'search')} & {cell(E,k,'heldout_class_instances',False)} & {cell(H,k,'search')} & {cell(H,k,'heldout_levels_instances',False)} & {cell(H,k,'heldout_class_instances',False)} \\\\")
L += ["\\bottomrule\\end{tabular}",
      f"\\caption{{Exploration procedures on the method bench (fixed harness; every counted hit replays through the harness). Constant-input tier: the stratified 48-target subset of the size-l splits ({ec['workers']}$\\times${ec['steps']} steps, seeds {ec['seeds']}); "
      f"exploration tier: the {len(next(iter(H.values()))['search']['names'])} / 12 / 16 targets no held input reaches ({hc['workers']}$\\times${hc['steps']}, seeds {hc['seeds']}). "
      "Cells: score\\,/\\,\\% detected (score $=1-\\text{first hit}/\\text{budget}$, 0 when missed); sealed-class and held-out-level cells give \\% detected. "
      "Sorted by exploration-tier search score. On the exploration tier the union over all procedures reaches 24/43 search targets against the best single procedure's 22: the residual is a common barrier (\\S\\ref{sec:barrier}).}",
      "\\label{tab:methods}\\end{table}"]
open(a.out, "w").write("\n".join(L) + "\n"); print("wrote", a.out, len(rows), "rows")
