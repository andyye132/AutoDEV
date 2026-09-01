"""Real-bug table for Part 1 from classified scan rows (scripts/classify_hits.py output). Distinct (level, anomaly) pairs of
severity A/B, which testers reached them and after how many steps; spawn-time hits (t<=2) are level-generation defects and are
counted separately. Several classified files (hand-designed and evolved testers, shipped and fresh levels) are merged.

    python scripts/make_bug_table.py --classified runs/hyak_scan_shipped.classified.json,runs/hyak_scan_fresh.classified.json \
        --tex paper/table_bugs.tex --out results/bugs.json
"""
import os, json, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--classified", default="runs/hyak_scan_shipped.classified.json,runs/hyak_scan_fresh.classified.json")
ap.add_argument("--tex", default="paper/table_bugs.tex"); ap.add_argument("--out", default="results/bugs.json")
ap.add_argument("--names", default="random=Random,repeat_random=Action-repeat,goexplore=Go-Explore,cand1_gen037_0.6994.py=Evolved,macrosweep.py=Macrosweep,"
                                   "wallbreaker.py=Wallbreaker,resonator.py=Resonator,contact_ucb.py=Contact-UCB")
ap.add_argument("--max_rows", type=int, default=14)
a = ap.parse_args()
names = dict(kv.split("=", 1) for kv in a.names.split(",") if kv)
names.update({k[:-3]: v for k, v in list(names.items()) if k.endswith(".py")})   # classified rows carry program names without .py
def _nm(p): return names.get(p, p.replace("_", "\\_"))
rows = []
for f in a.classified.split(","):
    if os.path.exists(f): rows += json.load(open(f))
    else: print("missing", f)


def short_level(lv):
    b = os.path.basename(lv).replace(".json", "")
    return ("fresh/" + b.split("_")[-1]) if "fresh" in lv else (lv.split("/")[0] + "/" + b if "/" in lv else b)


def corpus(lv): return "fresh" if "fresh" in lv else "shipped"


pairs = defaultdict(list)                       # (level, flag) -> rows
for r in rows:
    if r["severity"] != "X": pairs[(r["level"], r["flag"])].append(r)     # X = not reproduced in the classifier replay
table, spawn_pairs, per_program = [], set(), defaultdict(set)
for (lv, fl), rs in pairs.items():
    sev = min(r["severity"] for r in rs)          # A < B < C
    for r in rs: r["spawn"] = bool(r.get("spawn", r.get("t", 99) <= 2))
    if all(r["spawn"] for r in rs):
        if sev in "AB": spawn_pairs.add((lv, fl))
        continue
    rs = [r for r in rs if not r["spawn"]]
    if sev not in "AB": continue
    by_prog = defaultdict(list)
    for r in rs: by_prog[r["program"]].append(r)
    for p in by_prog: per_program[p].add((lv, fl))
    finders = {p: (min(x["first_hit_steps"] for x in v), len({x["seed"] for x in v})) for p, v in by_prog.items()}
    depth = max((r["depth"] or 0) for r in rs); dur = max(r["duration"] for r in rs); later = any(r["oob_later"] for r in rs)
    table.append(dict(level=lv, corpus=corpus(lv), flag=fl, severity=sev, depth=depth, duration=dur, oob_later=later, finders=finders))
table.sort(key=lambda t: (t["corpus"] != "shipped", t["severity"], -len(t["finders"]), t["level"]))
summary = dict(n_rows=len(table), by_corpus={c: sum(1 for t in table if t["corpus"] == c) for c in ("shipped", "fresh")},
               by_severity={s: sum(1 for t in table if t["severity"] == s) for s in "AB"},
               spawn_time_generation_defects=sorted(f"{short_level(l)}:{f}" for l, f in spawn_pairs),
               per_program={p: dict(distinct_AB_pairs=len(v), shipped=sum(1 for l, _ in v if corpus(l) == "shipped"), fresh=sum(1 for l, _ in v if corpus(l) == "fresh"))
                            for p, v in per_program.items()},
               union_pairs=len(set().union(*per_program.values())) if per_program else 0)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(dict(summary=summary, rows=table), open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))


def evidence(t):
    if t["flag"] == "OOB": return ((f"leaves arena, $\\ge$41 steps (replay-window end)" if t["duration"] >= 40 else f"leaves arena, {t['duration']} steps") if t["severity"] == "A" else f"crosses wall face, ejected after {t['duration']} step" + ("s" if t["duration"] != 1 else ""))
    if t["flag"] == "JOINT_TEAR": return "joint anchors separate"
    if t["flag"] == "PENETRATE": return f"{t['depth']*100:.1f}\\,cm deep, {t['duration']} steps" + ("; then crosses a wall face" if t["oob_later"] else "")
    if t["flag"] == "BLOWUP": return f"runaway speed/spin, {t['duration']} steps" + ("; then crosses a wall face" if t["oob_later"] else "")
    return t["flag"]


L = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{llllp{5.6cm}}", "\\toprule",
     "Level & Anomaly & Sev. & Evidence (replay) & Found by (min.\\ steps, seeds) \\\\", "\\midrule"]
for t in table[:a.max_rows]:
    finders = ", ".join(f"{_nm(p)} ({s/1000:.0f}k, {n})" for p, (s, n) in sorted(t["finders"].items(), key=lambda kv: kv[1][0]))
    L.append(f"{short_level(t['level']).replace('_', '\\_')} & {t['flag'].replace('_', '\\_')} & {t['severity']} & {evidence(t)} & {finders} \\\\")
if len(table) > a.max_rows: L.append(f"\\multicolumn{{5}}{{l}}{{\\ldots and {len(table) - a.max_rows} more (results/bugs.json)}} \\\\")
L += ["\\midrule"]
L.append("Distinct A/B pairs per tester & \\multicolumn{4}{l}{" + "; ".join(f"{_nm(p)} {v['distinct_AB_pairs']} ({v['shipped']} shipped, {v['fresh']} fresh)"
        for p, v in sorted(summary["per_program"].items(), key=lambda kv: -kv[1]["distinct_AB_pairs"])) + "} \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      f"\\caption{{Replay-verified engine anomalies of severity A (sustained escape or joint tear) or B (momentary wall-face crossing, deep or persistent penetration, long runaway) on unmodified levels: "
      f"{summary['by_corpus']['shipped']} distinct level--anomaly pairs on the 74 shipped levels and {summary['by_corpus']['fresh']} on 64 generated levels "
      f"({len(spawn_pairs)} further spawn-time pairs are level-generation defects, excluded). Each pair lists the testers that reached it, the fewest steps, and in how many seeds.}}",
      "\\label{tab:bugs}", "\\end{table}"]
os.makedirs(os.path.dirname(a.tex), exist_ok=True); open(a.tex, "w").write("\n".join(L) + "\n"); print("wrote", a.tex, "rows", len(table))
