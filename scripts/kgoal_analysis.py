"""K-Goal (C5.1): which levels does the discovered procedure solve within budget that no PPO objective solves? Reads the
program evaluation (eval_goal.json) and the PPO evaluations at 1x (eval_ppo_goal.json) and, if present, 4x budget
(eval_ppo_goal4x.json). Writes results/kgoal.json and prints the solved/unsolved level lists with steps.

    python scripts/kgoal_analysis.py --programs runs/hyak/eval_goal.json --ppo runs/hyak/eval_ppo_goal.json --ppo4x runs/hyak/eval_ppo_goal4x.json
"""
import os, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--programs", default="runs/hyak/eval_goal.json"); ap.add_argument("--ppo", default="runs/hyak/eval_ppo_goal.json"); ap.add_argument("--ppo4x", default="runs/hyak/eval_ppo_goal4x.json")
ap.add_argument("--procedure", default="cand1_gen037_0.6994.py"); ap.add_argument("--out", default="results/kgoal.json")
a = ap.parse_args()
P = json.load(open(a.programs)); PPO = json.load(open(a.ppo)) if os.path.exists(a.ppo) else {}; PPO4 = json.load(open(a.ppo4x)) if os.path.exists(a.ppo4x) else {}
res = {}
for spl in ("search", "heldout_levels_instances"):
    if spl not in P.get(a.procedure, {}): continue
    r = P[a.procedure][spl]; names = r["names"]; b = r["budget"]; fh = np.array(r["first_hit"])
    solved = (fh < b).mean(1) >= 0.5                      # solved in at least half the seeds
    def solved_by(E):
        m = np.zeros(len(names), bool)
        for p, splits in E.items():
            if spl not in splits: continue
            rr = splits[spl]; idx = {n: i for i, n in enumerate(rr["names"])}
            f = np.array(rr["first_hit"]); s = (f < rr["budget"]).mean(1) >= 0.5
            for n in names:
                if n in idx and s[idx[n]]: m[names.index(n)] = True
        return m
    ppo_any = solved_by(PPO); ppo4_any = solved_by(PPO4)
    only_proc = solved & ~ppo_any; only_proc4 = solved & ~ppo4_any if PPO4 else None
    base = {}
    for p, splits in P.items():
        if spl in splits: base[p] = float(((np.array(splits[spl]["first_hit"]) < splits[spl]["budget"]).mean(1) >= 0.5).mean())
    res[spl] = dict(n=len(names), procedure_solved=int(solved.sum()), ppo_any_solved=int(ppo_any.sum()), ppo4x_any_solved=int(ppo4_any.sum()) if PPO4 else None,
                    solved_by_procedure_not_ppo=[dict(level=names[i], median_steps=float(np.median(fh[i][fh[i] < b]))) for i in np.where(only_proc)[0]],
                    solved_by_procedure_not_ppo4x=[names[i] for i in np.where(only_proc4)[0]] if PPO4 else None,
                    solved_by_ppo_not_procedure=[names[i] for i in np.where(ppo_any & ~solved)[0]], solve_rate_by_program=base)
    print(f"{spl}: n={len(names)} procedure solves {int(solved.sum())}, any PPO@1x {int(ppo_any.sum())}, any PPO@4x {int(ppo4_any.sum()) if PPO4 else 'n/a'}; "
          f"procedure-only vs 1x: {int(only_proc.sum())}; vs 4x: {int(only_proc4.sum()) if PPO4 else 'n/a'}; PPO-only: {int((ppo_any & ~solved).sum())}")
    for d in res[spl]["solved_by_procedure_not_ppo"][:12]: print(f"   {d['level']:40s} median steps {d['median_steps']:.0f}")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)
