"""Fig 1: robustness heatmap — detection (fraction of target/defect states reached) for each method x task.
Sequential single-hue ramp (magnitude); our column is the only one with no cold cells. Direct-labeled. -> heatmap.pdf"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT="."
rows=json.load(open(f"{ROOT}/results/generalization_table.json"))
TASKS=[("MiniGrid DoorKey","grid+key"),("MiniGrid FourRooms","grid nav"),("MinAtar Breakout","arcade"),
       ("Jumanji Sokoban","puzzle"),("Gymnax MountainCar","control"),("Gymnax DeepSea-24","deep-explore"),
       ("Craftax","crafting")]
label={r["task"]:r for r in rows}
data=[]; ylabels=[]
for name,fam in TASKS:
    r=label[name]; data.append([r["random"],r["action_repeat"],r["ppo_best"],r["llm_best"]]); ylabels.append(name.replace("Gymnax ","").replace("MiniGrid ","").replace("MinAtar ","").replace("Jumanji ",""))
# Kinetix K-Defect row from breadth.json (protocol numbers)
br=json.load(open(f"{ROOT}/results/breadth.json"))["K-Defect"]["search"]
kd_r=br["random"]["detected"]; kd_ar=br["repeat_random"]["detected"]
kd_ppo=max(d["detected"] for p,d in br.items() if p.startswith("ppo_"))
kd_llm=max(d["detected"] for p,d in br.items() if p.endswith(".py"))
data.append([kd_r,kd_ar,kd_ppo,kd_llm]); ylabels.append("Kinetix K-Defect")
data=np.array(data)
methods=["Random","Action-\nrepeat","PPO\n(best of 5)","LLM tester\n(ours)"]

fig,ax=plt.subplots(figsize=(6.4,4.2))
im=ax.imshow(data, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
avgs0=data.mean(0) if False else None  # placeholder
ax.set_xticks(range(4))
ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=9)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v=data[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8.5,
                             color="white" if v>0.62 else "#222", fontweight="bold" if j==3 else "normal")
# emphasize our column
ax.add_patch(Rectangle((3-0.5,-0.5),1,data.shape[0],fill=False,edgecolor="#0b3d2e",lw=2.4))
avgs=data.mean(0)
xlabs=[f"{m}\n(avg {a:.2f})" for m,a in zip(methods,avgs)]
ax.set_xticklabels(xlabs, fontsize=8.5)
ax.set_title("Detection across 8 environments: the LLM-written tester is the only\nmethod with no cold cells (best or tied on every task)",fontsize=10.5,pad=8)
cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.03); cb.set_label("detection (fraction of targets reached)",fontsize=8.5); cb.ax.tick_params(labelsize=8)
ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,len(ylabels),1),minor=True)
ax.grid(which="minor",color="white",lw=1.5); ax.tick_params(which="minor",length=0)
fig.tight_layout()
fig.savefig(f"{ROOT}/paper/figs/heatmap.pdf",bbox_inches="tight"); fig.savefig(f"{ROOT}/paper/figs/heatmap.png",dpi=150,bbox_inches="tight")
print("wrote paper/figs/heatmap.pdf; column avgs:", {m.replace(chr(10),' '):round(a,3) for m,a in zip(methods,avgs)})
