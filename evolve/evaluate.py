"""ShinkaEvolve evaluation entry point: python evaluate.py --program_path <file> --results_dir <dir>"""
import argparse, os, json, sys
from shinka.core import run_shinka_eval

ROOT = os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
SEEDS = tuple(int(s) for s in os.environ.get("AUTODEV_SEEDS", "0").split(","))
STEPS = int(os.environ.get("AUTODEV_STEPS", "1024")); WORKERS = int(os.environ.get("AUTODEV_WORKERS", "64"))


def get_kwargs(run_idx):
    return dict(split_path=os.environ.get("AUTODEV_SPLIT", "runs/split.json"), which="search", size=os.environ.get("AUTODEV_SIZE", "m"), seeds=SEEDS, workers=WORKERS, steps=STEPS)


def aggregate(results):
    import os, json as _json
    import numpy as _np
    r = results[0]
    pub = {"detected": r["detected"], **{f"cls_{k}": v for k, v in r["per_class"].items()},
           **{f"det_{k}": v for k, v in r["detected_per_class"].items()}, "wall_s": r.get("wall", 0.0)}
    score = float(r["score"]); fb_lines = []
    names = r.get("names"); fh = r.get("first_hit"); B = r.get("budget")
    if names and fh is not None and B:
        fh = _np.asarray(fh, float)
        if fh.ndim == 1: fh = fh[:, None]
        if fh.shape[0] != len(names): fh = fh.T
        s = _np.clip(1.0 - _np.minimum(fh, B) / B, 0.0, 1.0).mean(1)          # per-target score, seed-mean
        # per-target regret normalization vs best-known (env AUTODEV_TARGET_NORM: {name: {"s": best_score_at_this_budget}})
        npth = os.environ.get("AUTODEV_TARGET_NORM")
        if npth and os.path.exists(npth):
            N = _json.load(open(npth))
            sstar = _np.array([max(float(N.get(n, {"s": 0.05}).get("s", 0.05)), 0.05) for n in names])
            ratio = _np.minimum(s / sstar, 2.0)
            score = float(ratio.mean())
            worst = _np.argsort(ratio)[:5]
            fb_lines.append("worst-vs-best targets (your score / best-known): " + "; ".join(
                f"{names[i]} {s[i]:.2f}/{sstar[i]:.2f}" for i in worst))
        # paired delta vs a frozen reference on identical tasks+seeds (env AUTODEV_REF_RESULT: a server outbox JSON)
        rpth = os.environ.get("AUTODEV_REF_RESULT")
        if rpth and os.path.exists(rpth):
            R = _json.load(open(rpth)); rfh = _np.asarray(R["first_hit"], float); rB = R["budget"]
            if rfh.ndim == 1: rfh = rfh[:, None]
            if rfh.shape[0] != len(R["names"]): rfh = rfh.T
            idx = {n: i for i, n in enumerate(R["names"]) }
            keep = [k for k, n in enumerate(names) if n in idx]
            rs = _np.clip(1.0 - _np.minimum(_np.stack([rfh[idx[names[k]]] for k in keep]), rB) / rB, 0, 1).mean(1)
            d = s[keep] - rs
            ci = 1.96 * d.std(ddof=1) / max(len(d), 1) ** 0.5
            pub["paired_delta_vs_ref"] = float(d.mean()); pub["paired_ci"] = float(ci)
            fb_lines.append(f"paired delta vs the reference program on identical tasks+seeds: {d.mean():+.4f} +- {ci:.4f} "
                            f"(better on {(d > 0.005).sum()} targets, worse on {(d < -0.005).sum()})")
        found = fh.min(1) < B * 0.999
        fb_lines.append(f"found {int(found.sum())}/{len(names)} targets; mean first-hit of found = {fh.min(1)[found].mean():,.0f} of {B:,} budget")
    fb_lines.append("per-class detection: " + ", ".join(f"{k}={v:.2f}" for k, v in r["detected_per_class"].items()))
    return {"combined_score": score, "public": pub, "private": {},
            "text_feedback": " | ".join(fb_lines)}


def validate(r):
    ok = isinstance(r, dict) and "score" in r and r["score"] == r["score"]
    return ok, None if ok else "bad result"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--program_path", required=True); ap.add_argument("--results_dir", required=True)
    a = ap.parse_args()
    os.environ["AUTODEV_PROGRAM_PATH"] = os.path.abspath(a.program_path)   # lets evaluate_hooks ship the file to the cluster (AUTODEV_REMOTE_EVAL=1)
    metrics, correct, err = run_shinka_eval(program_path=a.program_path, results_dir=a.results_dir, experiment_fn_name="run_experiment",
                                            num_runs=1, get_experiment_kwargs=get_kwargs, aggregate_metrics_fn=aggregate, validate_fn=validate)
    print(json.dumps(metrics, indent=1)); print("correct:", correct, err)
