"""Task splits for the non-Kinetix substrates (same JSON shape the Substrate.tasks() loaders read).

  Craftax (C-Ach):  worlds x achievements; held-out = a disjoint set of worlds; the achievement list can also hold out a tier.
  Gymnax:           initial-state seeds x one target per env; held-out = disjoint seeds.

    python scripts/make_substrate_splits.py --out_dir runs/substrates
"""
import os, json, argparse

ap = argparse.ArgumentParser(); ap.add_argument("--out_dir", default="runs/substrates")
ap.add_argument("--craftax_worlds", type=int, default=8); ap.add_argument("--craftax_achievements", default="0,1,5,9,11,20,29,61,28,49")
ap.add_argument("--craftax_heldout_achievements", default="61,28", help="held-out achievement 'class' (never in the fitness)")
ap.add_argument("--gymnax_seeds", type=int, default=8)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
try:
    from craftax.craftax.constants import Achievement as ACH
    names = {m.value: m.name for m in ACH}
except Exception:
    names = {}
achs = [int(x) for x in a.craftax_achievements.split(",")]; held_ach = {int(x) for x in a.craftax_heldout_achievements.split(",") if x}
W = a.craftax_worlds; held_worlds = set(range(W - max(2, W // 4), W))
search, hl, hc = [], [], []
for w in range(W):
    for k in achs:
        row = dict(name=f"world{w}__{names.get(k, k)}", world=w, achievement=k, cls=names.get(k, str(k)))
        if k in held_ach: hc.append(row)
        elif w in held_worlds: hl.append(row)
        else: search.append(row)
json.dump(dict(n_worlds=W, base_seed=0, search=search, heldout_levels_instances=hl, heldout_class_instances=hc,
               note="C-Ach: worlds x achievements; held-out worlds; held-out achievements as the sealed class"), open(os.path.join(a.out_dir, "craftax_split.json"), "w"), indent=1)
print(f"craftax: search {len(search)} / held-out worlds {len(hl)} / held-out achievements {len(hc)}")
targets = {"MountainCar-v0": [0.5], "Acrobot-v1": [1.0], "DeepSea-bsuite": [0.0], "Breakout-MinAtar": [3.0, 8.0, 15.0]}
for env_name, tgt in targets.items():
    S = a.gymnax_seeds; held = set(range(S - max(2, S // 4), S))
    rows = ([dict(name=f"{env_name}__seed{s}", seed=s, target=tgt, cls=env_name) for s in range(S)] if len(tgt) == 1 else
            [dict(name=f"{env_name}__thr{int(th)}_seed{s}", seed=s, target=[th], cls=env_name) for th in tgt for s in range(S)])
    json.dump(dict(search=[r for r in rows if r["seed"] not in held], heldout_levels_instances=[r for r in rows if r["seed"] in held], heldout_class_instances=[]),
              open(os.path.join(a.out_dir, f"gymnax_{env_name}_split.json"), "w"), indent=1)
    print(f"gymnax {env_name}: search {S - len(held)} / held-out seeds {len(held)}")
