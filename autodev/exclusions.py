"""Instances excluded from paper denominators: unwitnessable by construction (collision-matrix exclusion of every dynamic body) and
free hits (the witness fires within a few no-op steps from reset). Both lists come from scripts/structural_feasibility.py
(results/structural.json). Tables report them as rows rather than silently dropping them; analysis scripts call `mask_rows`."""
import os, json
import numpy as np

def load_exclusions(path="results/structural.json"):
    """Returns (excluded_names: set, reasons: dict name -> 'unwitnessable' | 'free_hit')."""
    if not os.path.exists(path): return set(), {}
    R = json.load(open(path))["instances"]; reasons = {}
    for n, r in R.items():
        if not r.get("feasible", True): reasons[n] = "unwitnessable"
        elif r.get("free_hit_step", -1) >= 0: reasons[n] = "free_hit"
    return set(reasons), reasons

def mask_rows(r, excluded):
    """Given an evaluation record r (with 'names' and 'first_hit' [n, seeds]) return (first_hit_kept, n_dropped)."""
    fh = np.array(r["first_hit"]); names = r.get("names")
    if not names or not excluded: return fh, 0
    keep = np.array([n not in excluded for n in names]); return fh[keep], int((~keep).sum())
