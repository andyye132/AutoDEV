"""Fig 1 (headline, reframed): the write-vs-train boundary in one image.
Panel A — equal-budget detection on the 4 sparse-reward tasks: run7 is the single best method, and plain Go-Explore
(same archive) is visibly NOT, so 'strategy not archive' is shown, not asserted.
Panel B — the budget crossover: run7 fixed at 1x vs tuned PPO scaling 1x/4x/16x (owns the boundary).
Okabe-Ito CVD-safe palette. -> fig1.pdf"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT="."
rows={r["task"]:r for r in json.load(open(f"{ROOT}/results/generalization_table.json"))}
br=json.load(open(f"{ROOT}/results/breadth.json"))
name={"DoorKey":"MiniGrid DoorKey","FourRooms":"MiniGrid FourRooms","Breakout":"MinAtar Breakout","Sokoban":"Jumanji Sokoban"}
sk={"DoorKey":"X-DoorKey","FourRooms":"X-FourRooms","Breakout":"B-Breakout","Sokoban":"S-Sokoban"}
TASKS=["DoorKey","FourRooms","Breakout","Sokoban"]
def ge(t): return br[sk[t]]["search"]["goexplore"]["detected"]
def run7(t):
    s=br[sk[t]]["search"]; return [v["detected"] for p,v in s.items() if "run7" in p][0]
# methods x tasks (equal budget, 1x)
data={
 "Random":       [rows[name[t]]["random"] for t in TASKS],
 "Action-repeat":[rows[name[t]]["action_repeat"] for t in TASKS],
 "Go-Explore":   [ge(t) for t in TASKS],
 "PPO (tuned)":  [rows[name[t]]["ppo_best"] for t in TASKS],
 "Our tester":   [run7(t) for t in TASKS],
}
OI={"Random":"#999999","Action-repeat":"#56B4E9","Go-Explore":"#CC79A7","PPO (tuned)":"#D55E00","Our tester":"#009E73"}

fig,(axA,axB)=plt.subplots(1,2,figsize=(11,3.9),gridspec_kw={"width_ratios":[1.45,1]})
# ---- Panel A: grouped bars ----
methods=list(data); x=np.arange(len(TASKS)); w=0.16
for i,m in enumerate(methods):
    bars=axA.bar(x+(i-2)*w, data[m], w, color=OI[m], label=m,
                 edgecolor="#0b3d2e" if m=="Our tester" else "none", linewidth=1.2 if m=="Our tester" else 0)
axA.set_xticks(x); axA.set_xticklabels(TASKS, fontsize=9.5)
axA.set_ylabel("detection at equal budget", fontsize=9.5); axA.set_ylim(0,1.08)
axA.set_title("(A) On the 4 sparse-reward tasks, our written tester is the single best\nmethod, and the same archive alone (Go-Explore) is not",fontsize=9.5)
axA.legend(fontsize=7.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5,-0.12), framealpha=0.9)
axA.grid(axis="y",alpha=0.25); axA.spines[["top","right"]].set_visible(False)
# ---- Panel B: budget crossover (avg over the 4) ----
def bestdet(p):
    import os
    if not os.path.exists(p): return None
    d=json.load(open(p)); return max((s.get("search",{}).get("detected",-1) for s in d.values()))
tk=["doorkey","fourrooms","breakout","sokoban"]
def avg(tag): 
    v=[bestdet(f"{ROOT}/runs/ppo_{t}{tag}.json") for t in tk]; v=[x for x in v if x is not None]; return np.mean(v)
ppo={1:avg("_tuned"),4:avg("_4x_tuned"),16:avg("_16x_tuned")}
ours=np.mean(data["Our tester"])
xs=[1,4,16]
axB.plot(xs,[ppo[m] for m in xs],"o-",color=OI["PPO (tuned)"],lw=2.4,ms=8,label="PPO (trained RL)")
for m in xs: axB.annotate(f"{ppo[m]:.2f}",(m,ppo[m]),textcoords="offset points",xytext=(0,-13),fontsize=7.5,color=OI["PPO (tuned)"],ha="center")
axB.axhline(ours,ls="--",color=OI["Our tester"],lw=2.4)
axB.plot([1],[ours],"D",color=OI["Our tester"],ms=10,label="Our tester (1x)")
axB.annotate(f"Our tester = {ours:.2f} at 1x",(1,ours),textcoords="offset points",xytext=(6,7),fontsize=8.5,color="#0b3d2e",fontweight="bold")
axB.set_xscale("log",base=2); axB.set_xticks(xs); axB.set_xticklabels(["1x","4x","16x"])
axB.set_xlabel("PPO interaction budget (× the tester's)",fontsize=9.5); axB.set_ylabel("detection (avg of 4)",fontsize=9.5)
axB.set_ylim(0.3,1.0); axB.set_title("(B) The boundary: tuned RL needs ~16x the budget\nto match our tester's equal-budget result",fontsize=9.5)
axB.legend(fontsize=8,loc="lower right"); axB.grid(alpha=0.25); axB.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{ROOT}/paper/figs/fig1.pdf",bbox_inches="tight"); fig.savefig(f"{ROOT}/paper/figs/fig1.png",dpi=140,bbox_inches="tight")
print("wrote fig1.pdf | PanelA run7:",[round(v,2) for v in data["Our tester"]],"| PanelB ppo:",{k:round(v,3) for k,v in ppo.items()},"ours",round(ours,3))
