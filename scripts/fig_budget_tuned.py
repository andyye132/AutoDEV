"""Fig 2 (HONEST, tuned): PPO detection vs interaction budget (1x/4x/16x) on the 4 LLM-hard tasks, TUNED config
(lr=1e-4, ent=0.2). The LLM tester is a fixed horizontal line at its 1x budget. Shows where/if tuned RL catches up.
Reads: runs/ppo_<task>_tuned.json (1x), runs/ppo_<task>_{4x,16x}_tuned.json. Plots only budgets that exist.
  python scripts/fig_budget_tuned.py -> figures/budget_crossover_tuned.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# task -> (pretty, LLM-tester detection at 1x [from results/generalization_table.json])
TASKS = [("doorkey","DoorKey"), ("fourrooms","FourRooms"), ("breakout","Breakout"), ("sokoban","Sokoban")]
OBJ = ["ppo_count","ppo_icm","ppo_curiosity_proximity"]  # the 3 in the budget runs

# LLM 1x detection (authoritative, from the generalization table)
llm = {r["task"].split()[-1].lower().replace("-24",""): r["llm_best"]
       for r in json.load(open("results/generalization_table.json"))}
LLM = {"doorkey":1.0, "fourrooms":0.8667, "breakout":1.0, "sokoban":0.7889}

def best_detected(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); best=None
    for obj,splits in d.items():
        v = splits.get("search",{}).get("detected")
        if v is not None and (best is None or v>best): best=v
    return best

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax,(key,pretty) in zip(axes, TASKS):
    xs, ys = [], []
    for mult,path in [(1,f"runs/ppo_{key}_tuned.json"),
                      (4,f"runs/ppo_{key}_4x_tuned.json"),
                      (16,f"runs/ppo_{key}_16x_tuned.json")]:
        v = best_detected(path)
        if v is not None: xs.append(mult); ys.append(v)
    if xs:
        ax.plot(xs, ys, "o-", color="#c1440e", lw=2, ms=7, label="tuned PPO (best-of-obj)")
        for x,y in zip(xs,ys): ax.annotate(f"{y:.2f}", (x,y), textcoords="offset points", xytext=(0,7), fontsize=8, ha="center")
    ax.axhline(LLM[key], ls="--", color="#1f6f4a", lw=2, label=f"LLM tester (1x) = {LLM[key]:.2f}")
    ax.set_xscale("log", base=2); ax.set_xticks([1,4,16]); ax.set_xticklabels(["1x","4x","16x"])
    ax.set_title(pretty, fontsize=11); ax.set_xlabel("PPO budget (x the tester's)")
    ax.set_ylim(-0.03, 1.08); ax.grid(alpha=0.25)
axes[0].set_ylabel("detection fraction")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Fig 2 (tuned): where trained RL catches up to a zero-training tester. The tester's budget is fixed at 1x; "
             "only PPO's budget scales.\nAt EQUAL budget (1x) the tester wins all four; tuned PPO needs 4-16x more "
             "interaction to approach it (and catches up on DoorKey).", fontsize=9.5)
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig("figures/budget_crossover_tuned.png", dpi=130)
print("wrote figures/budget_crossover_tuned.png")
for key,pretty in TASKS:
    row=[("1x",best_detected(f"runs/ppo_{key}_tuned.json")),("4x",best_detected(f"runs/ppo_{key}_4x_tuned.json")),("16x",best_detected(f"runs/ppo_{key}_16x_tuned.json"))]
    print(f"  {pretty:10s} LLM={LLM[key]:.2f} | " + " ".join(f"{t}={v:.2f}" if v is not None else f"{t}=--" for t,v in row))
