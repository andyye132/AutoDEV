"""Analyze + plot the graded bug-finding benchmark: detection accuracy vs bug difficulty (ANOMALY speed threshold),
per method. Reads runs/graded/eval_testers.json (+ eval_ppo_default/tuned.json) and runs/graded/pool.json.
-> figures/graded_bench.png + results/graded_bench.json"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

pool = {p["name"]: p for p in json.load(open("runs/graded/pool.json"))["instances"]}
THRS = sorted({p["difficulty"]["speed_thr"] for p in pool.values()})


def detected_by_instance(splitdata):
    """Return {instance_name: detected_bool} from an eval split dict (handles first_hit+names or detected list)."""
    d = splitdata
    names = d.get("names")
    if names and d.get("first_hit") is not None:
        fh = np.asarray(d["first_hit"], float); B = d.get("budget", fh.max() + 1)
        if fh.ndim == 2:  # [inst, seed] or [seed, inst]
            if fh.shape[0] != len(names): fh = fh.T
            det = (fh < B).mean(1)
        else:
            det = (fh < B).astype(float)
        return {names[i]: float(det[i]) for i in range(len(names))}
    return {}


def load_methods():
    methods = {}
    # testers
    if os.path.exists("runs/graded/eval_testers.json"):
        for prog, splits in json.load(open("runs/graded/eval_testers.json")).items():
            if "search" in splits:
                methods[os.path.basename(prog).replace(".py", "").replace("run7_best_gen035_0.5888", "run7")] = detected_by_instance(splits["search"])
    # PPO (best over objectives of default+tuned, per instance)
    ppo_inst = {}
    for f in ["runs/graded/eval_ppo_default.json", "runs/graded/eval_ppo_tuned.json"]:
        if os.path.exists(f):
            for lbl, splits in json.load(open(f)).items():
                if "search" in splits:
                    di = detected_by_instance(splits["search"])
                    for n, v in di.items():
                        ppo_inst[n] = max(ppo_inst.get(n, 0.0), v)
    if ppo_inst:
        methods["PPO (best-of, tuned)"] = ppo_inst
    return methods


methods = load_methods()
# per-difficulty detection per method
curves = {}
for m, inst in methods.items():
    y = []
    for thr in THRS:
        names = [n for n, p in pool.items() if p["difficulty"]["speed_thr"] == thr and n in inst]
        y.append(np.mean([inst[n] for n in names]) if names else np.nan)
    curves[m] = y

# order/emphasis: LLM testers teal-ish, PPO orange, random grey
COLOR = {"run7": "#0e6e62", "saturate": "#1a9e8f", "split_pump": "#3bb9a8", "macrosweep": "#7fc9bd",
         "PPO (best-of, tuned)": "#d98b3a", "random": "#9aa4ad", "repeat_random": "#b7bec4", "goexplore": "#6b7580"}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for m, y in sorted(curves.items(), key=lambda kv: -np.nanmean(kv[1])):
    ax.plot(THRS, y, marker="o", lw=2 if "run7" in m or "PPO" in m else 1.4,
            color=COLOR.get(m, "#888"), label=m, alpha=0.95)
ax.set_xlabel("bug difficulty — ANOMALY speed threshold (m/s), higher = harder to trigger")
ax.set_ylabel("detection rate (equal budget)")
ax.set_ylim(-0.03, 1.05)
ax.set_title("Detection accuracy vs bug difficulty: which methods find harder bugs?", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/graded_bench.png", dpi=170, bbox_inches="tight")
json.dump({"thresholds": THRS, "curves": {m: [None if np.isnan(v) else round(v, 3) for v in y] for m, y in curves.items()}},
          open("results/graded_bench.json", "w"), indent=1)
print("wrote figures/graded_bench.png + results/graded_bench.json")
for m, y in sorted(curves.items(), key=lambda kv: -np.nanmean(kv[1])):
    print(f"  {m:22s} " + " ".join(f"{('%.2f'%v) if not np.isnan(v) else ' -- '}" for v in y))
