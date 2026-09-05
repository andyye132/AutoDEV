"""Fig: detection vs compute (interaction budget). Our zero-training tester reaches its score at 1x; trained PPO needs
~16x more compute to approach it, and no baseline comes close. Avg over the 4 sparse-nav tasks (DoorKey/FourRooms/
Breakout/Sokoban) where PPO budget curves exist. Okabe-Ito CVD-safe palette. -> budget_curve.pdf"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT="."; TASKS=["doorkey","fourrooms","breakout","sokoban"]
def best_det(path):
    import os
    if not os.path.exists(path): return None
    d=json.load(open(path)); return max((s.get("search",{}).get("detected",-1) for s in d.values()))
def avg_ppo(tag):
    vals=[best_det(f"{ROOT}/runs/ppo_{t}{tag}.json") for t in TASKS]; vals=[v for v in vals if v is not None]
    return np.mean(vals) if vals else None
ppo={1:avg_ppo("_tuned"),4:avg_ppo("_4x_tuned"),16:avg_ppo("_16x_tuned")}
# fixed-at-1x methods (avg over the same 4 tasks), from generalization_table.json + the go-explore skeptic table
gt={r["task"].split()[-1].lower().replace("-8x8",""):r for r in json.load(open(f"{ROOT}/results/generalization_table.json"))}
name={"doorkey":"MiniGrid DoorKey","fourrooms":"MiniGrid FourRooms","breakout":"MinAtar Breakout","sokoban":"Jumanji Sokoban"}
rows={r["task"]:r for r in json.load(open(f"{ROOT}/results/generalization_table.json"))}
ours=np.mean([rows[name[t]]["llm_best"] for t in TASKS])
arep=np.mean([rows[name[t]]["action_repeat"] for t in TASKS])
rand=np.mean([rows[name[t]]["random"] for t in TASKS])
goexp=np.mean([0.10,0.40,0.33,0.51])  # DoorKey/FourRooms/Breakout/Sokoban plain go-explore (skeptic-defense table)

OI={"ppo":"#D55E00","ours":"#009E73","arep":"#56B4E9","goexp":"#CC79A7","rand":"#999999"}
fig,ax=plt.subplots(figsize=(6.2,4.0))
xs=[1,4,16]; yp=[ppo[m] for m in xs]
ax.plot(xs,yp,"o-",color=OI["ppo"],lw=2.4,ms=8,label="PPO (trained RL)",zorder=5)
for x,y in zip(xs,yp): ax.annotate(f"{y:.2f}",(x,y),textcoords="offset points",xytext=(0,-14),fontsize=8,color=OI["ppo"],ha="center")
# our tester: reached at 1x, drawn as the target line
ax.axhline(ours,ls="--",color=OI["ours"],lw=2.6,zorder=4)
ax.plot([1],[ours],"D",color=OI["ours"],ms=10,zorder=6,label="LLM tester (ours, zero-training, 1x)")
ax.annotate(f"ours = {ours:.2f} at 1x",(1,ours),textcoords="offset points",xytext=(6,8),fontsize=9,color="#0b3d2e",fontweight="bold")
for val,key,lab in [(arep,"arep","Action-repeat"),(goexp,"goexp","Go-Explore"),(rand,"rand","Random")]:
    ax.axhline(val,ls=":",color=OI[key],lw=1.8,zorder=2)
    ax.annotate(f"{lab} {val:.2f}",(1.05,val),textcoords="offset points",xytext=(0,4),fontsize=7.5,color=OI[key],ha="left",va="bottom")
# annotate the efficiency gap
ax.text(3.0,0.66,"PPO needs ~16x more\ncompute to reach\nour 1x result",fontsize=8.2,color=OI["ppo"],ha="center",va="center")
ax.set_xscale("log",base=2); ax.set_xticks(xs); ax.set_xticklabels(["1x","4x","16x"])
ax.set_xlabel("interaction budget / compute  (x the tester's budget; only PPO scales)",fontsize=9.5)
ax.set_ylabel("detection (avg over 4 sparse-nav tasks)",fontsize=9.5)
ax.set_ylim(0.25,1.0); ax.set_xlim(0.8,26)
ax.set_title("At equal budget the zero-training tester dominates; trained RL needs\n~16x the compute to catch up, baselines never do",fontsize=10.5,pad=8)
ax.legend(fontsize=8.5,loc="lower right",framealpha=0.95,bbox_to_anchor=(1.0,0.02))
ax.grid(alpha=0.25); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{ROOT}/paper/figs/budget_curve.pdf",bbox_inches="tight"); fig.savefig(f"{ROOT}/paper/figs/budget_curve.png",dpi=150,bbox_inches="tight")
print(f"ours={ours:.3f} ppo 1x/4x/16x={ppo[1]:.3f}/{ppo[4]:.3f}/{ppo[16]:.3f} arep={arep:.3f} goexp={goexp:.3f} rand={rand:.3f}")
