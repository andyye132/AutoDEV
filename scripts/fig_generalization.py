"""Figure: LLM-written testers vs trained PPO vs random, at equal budget, across diverse RL tasks.
Reads results/generalization_table.json (+ K-Goal and Craftax) -> figures/generalization.png."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

rows = json.load(open("results/generalization_table.json"))  # DoorKey, FourRooms, Breakout, Sokoban, MountainCar, DeepSea-24
data = [(r["task"].replace("MiniGrid ", "").replace("Gymnax ", "").replace("MinAtar ", "").replace("Jumanji ", ""),
         r["random"], r["ppo_best"], r["llm_best"]) for r in rows]

# Kinetix K-Goal (verified: results/RESULTS_GENERALIZATION.md)
data.append(("Kinetix K-Goal", 0.362, 0.558, 0.673))

# Craftax if available
cx = "runs/ppo_craftax.json"
if os.path.exists(cx):
    d = json.load(open(cx))
    ppo = max((v["search"]["detected"] for v in d.values() if "search" in v), default=None)
    br = json.load(open("results/breadth.json")).get("C-Ach", {}).get("search", {})
    BASE = {"random", "repeat_random", "goexplore"}
    llm = max((v["detected"] for k, v in br.items() if k not in BASE and "ppo" not in k), default=None)
    rnd = br.get("random", {}).get("detected")
    if ppo is not None and llm is not None:
        data.append(("Craftax", rnd, ppo, llm))

labels = [d[0] for d in data]
rnd = [d[1] for d in data]; ppo = [d[2] for d in data]; llm = [d[3] for d in data]
n = len(data); x = np.arange(n); w = 0.26

fig, ax = plt.subplots(figsize=(max(8, n * 1.25), 4.6))
c_rnd, c_ppo, c_llm = "#9aa4ad", "#d98b3a", "#0e6e62"
ax.bar(x - w, rnd, w, label="random", color=c_rnd)
ax.bar(x, ppo, w, label="best PPO (5 objectives, trained)", color=c_ppo)
ax.bar(x + w, llm, w, label="best LLM tester (zero training)", color=c_llm)

for i in range(n):
    ax.text(x[i] + w, llm[i] + 0.02, f"{llm[i]:.2f}", ha="center", va="bottom", fontsize=8, color=c_llm, fontweight="bold")

ax.set_ylabel("fraction of targets reached\n(detection, equal budget)")
ax.set_ylim(0, 1.15)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_title("At equal interaction budget, a zero-training LLM-written tester beats trained PPO\non every RL task — grid, arcade, puzzle, control, deep-exploration, physics, survival", fontsize=10.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=8.5, framealpha=0, columnspacing=1.4)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.axhline(0, color="#333", lw=0.6)
fig.tight_layout()
fig.savefig("figures/generalization.png", dpi=170, bbox_inches="tight")
print(f"wrote figures/generalization.png ({n} tasks). LLM beats PPO on {sum(1 for i in range(n) if llm[i] > ppo[i] + 0.02)}/{n}.")
