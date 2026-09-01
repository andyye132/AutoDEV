"""Pick the top-k distinct evolved programs by SEARCH fitness only (never by held-out results) and export them as files."""
import sys, os, json, sqlite3, hashlib, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/evolve_overnight"); ap.add_argument("--k", type=int, default=3); ap.add_argument("--out", default=None)
a = ap.parse_args(); out = a.out or os.path.join(a.run, "candidates"); os.makedirs(out, exist_ok=True)
con = sqlite3.connect(os.path.join(a.run, "programs.sqlite")); cur = con.cursor()
rows = cur.execute("select id, generation, island_idx, combined_score, code, metadata from programs where correct=1 and combined_score is not null order by combined_score desc").fetchall()
seen, picked = set(), []
for pid, gen, isl, score, code, meta in rows:
    h = hashlib.sha256(code.encode()).hexdigest()[:12]
    if h in seen: continue
    seen.add(h); m = json.loads(meta) if meta else {}
    name = f"cand{len(picked)+1}_gen{gen:03d}_{score:.4f}"; path = os.path.join(out, name + ".py"); open(path, "w").write(code)
    picked.append(dict(name=name, path=path, id=pid, gen=gen, island=isl, score=score, patch=m.get("patch_name", ""), code_hash=h))
    if len(picked) >= a.k: break
json.dump(picked, open(os.path.join(out, "candidates.json"), "w"), indent=1)
for p in picked: print(f"{p['name']}  island {p['island']}  {p['patch'][:50]}")
