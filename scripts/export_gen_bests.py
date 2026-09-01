"""Export from a ShinkaEvolve run: (a) a lineage table (generation, island, score, patch name, parent), (b) the best-so-far
program at each generation checkpoint as a standalone .py (for transfer/overfitting analysis)."""
import sys, os, json, sqlite3, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/evolve_overnight"); ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()
con = sqlite3.connect(os.path.join(a.run, "programs.sqlite")); cur = con.cursor()
rows = cur.execute("select id, generation, island_idx, combined_score, correct, parent_id, metadata, code, public_metrics from programs order by generation, timestamp").fetchall()
lineage = []
for pid, gen, isl, score, correct, parent, meta, code, pub in rows:
    m = json.loads(meta) if meta else {}
    lineage.append(dict(id=pid, gen=gen, island=isl, score=score, correct=bool(correct), parent=parent, patch=m.get("patch_name") or m.get("name") or "", public=json.loads(pub) if pub else {}))
out = os.path.join(a.run, "exports"); os.makedirs(out, exist_ok=True)
json.dump(lineage, open(os.path.join(out, "lineage.json"), "w"), indent=1)
best_so_far, curve, exported = None, [], []
max_gen = max(r["gen"] for r in lineage)
by_gen = {}
for r in lineage:
    by_gen.setdefault(r["gen"], []).append(r)
for g in range(max_gen + 1):
    for r in by_gen.get(g, []):
        if r["correct"] and r["score"] is not None and (best_so_far is None or r["score"] > best_so_far["score"]): best_so_far = r
    curve.append(dict(gen=g, best=best_so_far["score"] if best_so_far else None, best_id=best_so_far["id"] if best_so_far else None,
                      gen_scores=[r["score"] for r in by_gen.get(g, []) if r["correct"]]))
    if best_so_far and (g % a.every == 0 or g == max_gen):
        code = next(c for (pid, *_rest, c, _p) in [(x[0], x[7], x[8]) for x in rows] if pid == best_so_far["id"]) if False else [x[7] for x in rows if x[0] == best_so_far["id"]][0]
        path = os.path.join(out, f"best_upto_gen{g:03d}.py"); open(path, "w").write(code); exported.append(dict(gen=g, path=path, id=best_so_far["id"], score=best_so_far["score"], patch=best_so_far["patch"]))
json.dump(dict(curve=curve, exported=exported), open(os.path.join(out, "curve.json"), "w"), indent=1)
print(f"{len(lineage)} programs, {max_gen+1} generations; best {best_so_far['score']:.4f} ({best_so_far['patch']}, gen {best_so_far['gen']}, island {best_so_far['island']})")
for e in exported: print(f"  gen {e['gen']:3d}: {e['score']:.4f}  {e['patch'][:50]}  -> {e['path']}")
