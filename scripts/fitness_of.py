"""Equal-fitness scores for a list of tester programs through a fitness-server queue (same split/seeds/budget as the search
that used that queue), so hand designs and evolved candidates can be placed on the same fitness scale. Updates a JSON
name -> result dict; already-scored names are skipped.

    AUTODEV_REMOTE_TIMEOUT=3600 python scripts/fitness_of.py --queue /gscratch/raivn/andy132/AutoDev/runs/hyak/evalq5 \
        --out runs/fitness_evalq5.json evolve/proposals2/pairladder.py runs/evolve_run4_gpu/candidates/cand1_gen030_0.6460.py
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--queue", required=True); ap.add_argument("--out", required=True); ap.add_argument("programs", nargs="+")
a = ap.parse_args(); os.environ["AUTODEV_REMOTE_QUEUE"] = a.queue
from autodev.remote_eval import remote_evaluate
R = json.load(open(a.out)) if os.path.exists(a.out) else {}
for p in a.programs:
    key = os.path.basename(p)
    if key in R and R[key].get("score") is not None: print("skip", key, R[key]["score"]); continue
    try:
        r = remote_evaluate(p, log=print); r["path"] = p; R[key] = r
    except Exception as e:
        R[key] = {"error": str(e)[-800:], "path": p}; print("ERR", key, str(e)[-200:])
    json.dump(R, open(a.out, "w"), indent=1); print("scored", key, R[key].get("score"), R[key].get("detected"), flush=True)
