"""Two-panel budget/cost figure: (left) K-Goal win targets - PPO climbs with training budget vs the best untrained procedure;
(right) K-Defect - procedures saturate with simulator budget. Data hardcoded from the evaluation files (see LOG.md Aug 30-31)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6))
# left: K-Goal search % detected vs training budget multiple
a.plot([4, 16], [63, 71], "-o", color="C0", label="PPO+count")
a.plot([4, 16], [72, 76], "-s", color="C1", label="PPO+curiosity-prox")
a.plot([16], [77], "d", color="C2", label="PPO+RND (16x)")
a.axhline(59, ls="--", color="C3"); a.text(1.05, 60, "best procedure, zero training (59%)", fontsize=8, color="C3")
a.set_xscale("log", base=2); a.set_xticks([1, 4, 16]); a.set_xticklabels(["1x", "4x", "16x"])
a.set_xlabel("training budget (multiples of the procedures' simulator budget)"); a.set_ylabel("K-Goal search targets reached (%)")
a.set_title("Win targets: RL climbs with budget", fontsize=10); a.legend(fontsize=7, loc="lower right"); a.grid(alpha=0.3)
# right: K-Defect detection vs simulator budget for the leader procedure
b.plot([1, 2, 4], [66, 65, 66], "-o", label="search split")
b.plot([1, 2, 4], [36, 37, 38], "-s", label="sealed class (OOB)")
b.axhline(45, ls=":", color="grey"); b.text(1.02, 45.7, "witnessable ceiling (sealed): 45%", fontsize=8, color="grey")
b.set_xscale("log", base=2); b.set_xticks([1, 2, 4]); b.set_xticklabels(["1x", "2x", "4x"])
b.set_xlabel("simulator budget (multiples of 128x2048)"); b.set_ylabel("K-Defect targets reached (%)")
b.set_title("Defect targets: procedures saturate", fontsize=10); b.legend(fontsize=8, loc="center right"); b.grid(alpha=0.3)
fig.suptitle("Two sides of the compute axis: zero-training procedures vs trained RL", fontsize=11)
fig.tight_layout(); fig.savefig("figures/cost_curve.png", dpi=160); print("wrote figures/cost_curve.png")
