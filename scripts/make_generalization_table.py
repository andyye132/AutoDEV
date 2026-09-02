"""Generalization table: per diverse RL task, does the LLM-written tester beat trained PPO at EQUAL budget?
Reads results/breadth.json (LLM/evolved testers + baselines, `detected` at budget 131072) and runs/ppo_<task>.json
(PPO objectives from eval_ppo_substrate). Metric = detection fraction on the search split, all at 131072 env steps.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BASE = {"random", "repeat_random", "goexplore"}
# suite in breadth.json -> (ppo json, pretty task name, family)
TASKS = [
    ("X-DoorKey", "runs/ppo_doorkey.json", "MiniGrid DoorKey", "grid nav + subgoal"),
    ("X-FourRooms", "runs/ppo_fourrooms.json", "MiniGrid FourRooms", "grid navigation"),
    ("B-Breakout", "runs/ppo_breakout.json", "MinAtar Breakout", "arcade"),
    ("S-Sokoban", "runs/ppo_sokoban.json", "Jumanji Sokoban", "puzzle / planning"),
    ("G-MountainCar", "runs/ppo_mountaincar.json", "Gymnax MountainCar", "continuous control"),
]
br = json.load(open("results/breadth.json"))


def best_ppo(path):
    if not os.path.exists(path):
        return None, None
    d = json.load(open(path))
    best, who = -1, None
    for obj, splits in d.items():
        sc = splits.get("search", {}).get("detected")
        if sc is not None and sc > best:
            best, who = sc, obj
    return best, who


rows = []
for suite, ppo_path, pretty, family in TASKS:
    sch = br.get(suite, {}).get("search", {})
    rnd = sch.get("random", {}).get("detected")
    arep = sch.get("repeat_random", {}).get("detected")
    llm = {k: v["detected"] for k, v in sch.items() if k not in BASE}
    llm_best = max(llm.values(), default=None); llm_who = max(llm, key=llm.get) if llm else None
    ppo_best, ppo_who = best_ppo(ppo_path)
    rows.append(dict(task=pretty, family=family, random=rnd, action_repeat=arep,
                     ppo_best=ppo_best, ppo_who=ppo_who, llm_best=llm_best, llm_who=llm_who))

# print
print(f"{'Task':22s} {'family':20s} {'random':>7s} {'a-rep':>7s} {'PPO*':>7s} {'LLM*':>7s}  verdict")
print("-" * 90)
llm_wins = ppo_wins = ties = 0
for r in rows:
    def f(x): return f"{x:.2f}" if isinstance(x, float) else " -- "
    v = "n/a"
    if r["llm_best"] is not None and r["ppo_best"] is not None:
        if r["llm_best"] > r["ppo_best"] + 0.02: v = "LLM > PPO"; llm_wins += 1
        elif r["ppo_best"] > r["llm_best"] + 0.02: v = "PPO > LLM"; ppo_wins += 1
        else: v = "tie"; ties += 1
    print(f"{r['task']:22s} {r['family']:20s} {f(r['random']):>7s} {f(r['action_repeat']):>7s} "
          f"{f(r['ppo_best']):>7s} {f(r['llm_best']):>7s}  {v}  (PPO={r['ppo_who']}, LLM={r['llm_who']})")
print("-" * 90)
print(f"LLM-tester beats PPO on {llm_wins}/{llm_wins+ppo_wins+ties} tasks (ties {ties}, PPO wins {ppo_wins}); "
      f"all at 131,072 env steps, zero training for the LLM tester.")
json.dump(rows, open("results/generalization_table.json", "w"), indent=1)
