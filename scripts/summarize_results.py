"""Collect headline numbers into results/summary.json: per-program scores with 95% CIs per split, per-class detection,
relative gains vs the strongest hand-designed baseline, and bug counts from scans."""
import sys, os, json, argparse, glob
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.exclusions import load_exclusions, mask_rows
EXCL, _ = load_exclusions()
ap = argparse.ArgumentParser(); ap.add_argument("--eval", default="runs/hyak/eval_programs.json"); ap.add_argument("--scans", default="")
ap.add_argument("--design", default="runs/hyak/design_checks.json"); ap.add_argument("--out", default="results/summary.json")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
S = {}
if os.path.exists(a.eval):
    ev = json.load(open(a.eval)); S["eval"] = {}
    for p, splits in ev.items():
        S["eval"][p] = {}
        for sp, r in splits.items():
            fh, n_dropped = mask_rows(r, EXCL); budget = r["budget"]
            per_seed = 1 - np.minimum(fh, budget).mean(0) / budget
            det_seed = (fh < budget).mean(0)
            S["eval"][p][sp] = dict(n_instances=int(fh.shape[0]), n_excluded=int(n_dropped), n_seeds=int(fh.shape[1]), budget=budget,
                                    score=float(per_seed.mean()), score_ci95=float(1.96 * per_seed.std(ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else None,
                                    detected=float(det_seed.mean()), detected_ci95=float(1.96 * det_seed.std(ddof=1) / np.sqrt(len(det_seed))) if len(det_seed) > 1 else None,
                                    median_steps_when_found=float(np.median(fh[fh < budget])) if (fh < budget).any() else None,
                                    detected_per_class=r["detected_per_class"], per_class=r["per_class"])
    base = [p for p in ev if p in ("random", "repeat_random", "goexplore")]
    S["strongest_baseline"] = {}
    for sp in next(iter(ev.values())).keys():
        best = max([b for b in base if sp in S["eval"][b]], key=lambda p: S["eval"][p][sp]["score"]) if base else None
        S["strongest_baseline"][sp] = best
        for p in ev:
            if best and p not in base and sp in S["eval"][p]:
                S["eval"][p][sp]["gain_vs_strongest_baseline"] = S["eval"][p][sp]["score"] - S["eval"][best][sp]["score"]
                S["eval"][p][sp]["detected_gain_vs_strongest_baseline"] = S["eval"][p][sp]["detected"] - S["eval"][best][sp]["detected"]
for path in [x for x in a.scans.split(",") if x]:
    if not os.path.exists(path): continue
    scan = json.load(open(path)); key = os.path.basename(path).replace(".json", "")
    rows = {}
    for r in scan["levels"]:
        for p, hs in r["programs"].items():
            for h in hs:
                if h["found"] and h.get("replay_verified"):
                    rows.setdefault(p, set()).add((r["level"], h["flag"]))
    S.setdefault("bugs", {})[key] = dict(n_levels=len(scan["levels"]), budget=scan["cfg"]["budget"],
        distinct_level_anomaly_pairs_per_program={p: len(v) for p, v in rows.items()},
        union_pairs=len(set().union(*rows.values())) if rows else 0,
        by_flag={fl: len({lv for p in rows for (lv, f) in rows[p] if f == fl}) for fl in {f for p in rows for (_, f) in rows[p]}},
        pairs={p: sorted(v) for p, v in rows.items()})
if os.path.exists(a.design):
    d = json.load(open(a.design)); S["design"] = dict(n_levels=len(d), trivial_win=[r["level"] for r in d if "TRIVIAL_WIN" in r["flags"]], noop_death=sum(1 for r in d if "NOOP_DEATH" in r["flags"]))
json.dump(S, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in S.items() if k != "eval"}, indent=1)[:2500])
if "eval" in S:
    for p in S["eval"]:
        print(p, {sp: f"{v['score']:.3f}±{(v['score_ci95'] or 0):.3f} det {v['detected']:.2f}" for sp, v in S["eval"][p].items()})
