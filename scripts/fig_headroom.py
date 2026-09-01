"""Stacked headroom decomposition per split: leader-found / stealable / frontier (nobody). From scripts/headroom.py (Aug 31)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
splits = ["Search\n(119)", "Held-out levels\n(58)", "Sealed class\n(148)"]
found = np.array([82, 36, 60]); steal = np.array([4, 1, 6]); frontier = np.array([33, 21, 82])
tot = found + steal + frontier
fig, ax = plt.subplots(figsize=(6.4, 4.6))
x = np.arange(3)
p1 = ax.bar(x, 100*found/tot, color="#2a9d8f", label="found by the leader (run 11)")
p2 = ax.bar(x, 100*steal/tot, bottom=100*found/tot, color="#e9c46a", label="found only by another program (stealable)")
p3 = ax.bar(x, 100*frontier/tot, bottom=100*(found+steal)/tot, color="#9aa5ab", label="reached by nobody (mostly physically blocked; T2-confirmed)")
for i in range(3):
    ax.text(i, 100*found[i]/tot[i]/2, f"{found[i]}", ha="center", color="white", fontsize=10, weight="bold")
    ax.text(i, 100*(found[i]+steal[i]+frontier[i]/2)/tot[i], f"{frontier[i]}", ha="center", color="white", fontsize=10, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("% of instances")
ax.set_title("Where the remaining points live", fontsize=11)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, framealpha=0.95)
fig.tight_layout(); fig.savefig("figures/headroom.png", dpi=160); print("wrote figures/headroom.png")
