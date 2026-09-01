"""Write the human-anchor session bundle: runs/human/README.md with per-instance commands and the session protocol, from
runs/human/instances.json (selected by class). Players run the commands on this machine (or a copy of the repo + venv).

    python scripts/human_bundle.py --player alice
"""
import os, json, argparse

ap = argparse.ArgumentParser(); ap.add_argument("--player", default="PLAYER"); ap.add_argument("--minutes", type=float, default=5.0); ap.add_argument("--budget", type=int, default=2048)
a = ap.parse_args()
sel = json.load(open("runs/human/instances.json"))
order = ["ANOMALY", "DISARMED", "SHORTCUT", "TUNNEL", "OOB"]
desc = {"OOB": "One boundary wall of the arena has been removed. Get any moving shape out of the 5 m arena.",
        "TUNNEL": "One static obstacle is no longer solid. Find it: get a moving shape's centre inside a solid-looking obstacle.",
        "SHORTCUT": "A neutral (grey) shape now counts as the goal. Win the level by touching the wrong shape, away from the real goal.",
        "ANOMALY": "Some shapes are super-bouncy. Make something move unreasonably fast.",
        "DISARMED": "One red hazard is harmless. Touch a red shape with a green one and survive."}
lines = [f"# Human anchor session — player `{a.player}`", "",
         f"Ten planted-defect levels, {a.minutes:.0f} minutes each (or until the defect is witnessed). The same witness oracle the automated testers use runs every step and logs when you trigger it.",
         "Controls: arrows/WASD = motors 1–2 (±), Q/E = motors 3–4, Z/C = motors 5–6, 1/2 = thrusters, R = reset the level, ESC = give up. No hints beyond the class description.", "",
         "```", "cd /Users/andy132/projects/research/ACT/AutoDev && . .venv/bin/activate", "```", ""]
for c in order:
    for x in sel.get(c, []):
        name = x[0]
        lines += [f"## {c}: `{name}`", desc[c], "", "```", f"python scripts/play_instance.py --instance runs/hyak/instances/{name}.json --meta runs/hyak/pool.json --size l --budget {a.budget} --minutes {a.minutes} --player {a.player} --log runs/human/{a.player}.jsonl", "```", ""]
lines += ["When done, send `runs/human/<player>.jsonl`. Each line: instance, class, steps, resets, wall_s, first_witness_step (null if not found), reason."]
os.makedirs("runs/human", exist_ok=True); open("runs/human/README.md", "w").write("\n".join(lines)); print("wrote runs/human/README.md for", a.player, "with", sum(len(v) for v in sel.values()), "instances")
