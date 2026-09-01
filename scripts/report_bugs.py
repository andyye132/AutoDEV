"""Aggregate scan results (+ design checks) into a bug table with severity tiers and render strips for verified hits."""
import sys, os, json, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--scans", default="runs/scan_clean_v2.json"); ap.add_argument("--design", default="runs/design_checks.json")
ap.add_argument("--out", default="runs/bug_report.md"); ap.add_argument("--render", action="store_true"); ap.add_argument("--size", default="m")
a = ap.parse_args()
SEVERITY = {"OOB": "A (escapes arena)", "JOINT_TEAR": "A (joint torn)", "PENETRATE": "B (persistent penetration)", "BLOWUP": "B (persistent runaway physics)"}
rows = []
for scan_path in a.scans.split(","):
    scan = json.load(open(scan_path)); budget = scan["cfg"]["budget"]
    cls_path = scan_path.replace(".json", ".classified.json")
    cls = {(x["level"], x["program"], x["seed"]): x for x in json.load(open(cls_path))} if os.path.exists(cls_path) else {}
    for r in scan["levels"]:
        for p, hs in r["programs"].items():
            for h in hs:
                if h["found"]:
                    c = cls.get((r["level"], p, h["seed"]), {})
                    rows.append(dict(level=r["level"], program=p, seed=h["seed"], flag=h["flag"], first_hit=int(h["first_hit"][h["flag"]]), budget=budget, verified=h["replay_verified"], chain=h["chain_len"], scan=scan_path,
                                     severity=c.get("severity", "?"), duration=c.get("duration"), depth=c.get("depth"), spawn=c.get("spawn", False)))
levels = sorted({x["level"] for x in rows}); by_flag = {}
for x in rows: by_flag.setdefault((x["level"], x["flag"]), []).append(x)
lines = ["# Bug report (automatic)", "", f"Scans: {a.scans}", "", "## Engine anomalies (replay-verified, persistence-filtered)", "",
         "Severity: A = escapes the arena / joint torn; B = deep (≥5 cm) or long (≥10 steps) penetration, or ≥10-step runaway physics; C = minor transient overlap/spike.", "",
         "| Level | Anomaly | Severity | Found by (program:seed → env steps) | Duration / depth | Verified |", "|---|---|---|---|---|---|"]
def sev_of(xs):
    s_ = [x.get("severity", "?") for x in xs if x.get("verified")]
    return min(s_) if s_ else "?"
for (lv, fl), xs in sorted(by_flag.items(), key=lambda kv: (sev_of(kv[1]), kv[0])):
    found = ", ".join(f"{x['program']}:{x['seed']}→{x['first_hit']}" for x in sorted(xs, key=lambda x: x['first_hit']))
    dd = "; ".join(f"{x['duration']}st" + (f"/{x['depth']:.3f}m" if x.get('depth') is not None else "") for x in xs[:3] if x.get("duration") is not None)
    ver = [x for x in xs if x.get("verified")]
    tag = " (spawn-time: level defect)" if ver and all(x.get("spawn") for x in ver) else ""
    lines.append(f"| {lv} | {fl}{tag} | {sev_of(xs)} | {found} | {dd} | {all(x['verified'] for x in xs)} |")
ab = {k for k, xs in by_flag.items() if sev_of(xs) in ("A", "B")}
lines += ["", f"Levels with at least one verified anomaly: {len(levels)}; distinct (level, anomaly) pairs: {len(by_flag)}; of which severity A/B: {len(ab)}.", ""]
prog_counts = {}
for (lv, fl), xs in by_flag.items():
    for x in xs:
        if x.get("verified") and x.get("severity") in ("A", "B") and not x.get("spawn"): prog_counts.setdefault(x["program"], set()).add((lv, fl))
lines += ["Distinct A/B (level, anomaly) pairs found per tester (excluding spawn-time level defects): " + ", ".join(f"{p}: {len(v)}" for p, v in sorted(prog_counts.items())), ""]
if os.path.exists(a.design):
    d = json.load(open(a.design))
    lines += ["## Design checks (automatic)", "", "| Level | No-input outcome | Random-play win rate | Median win step | Flags |", "|---|---|---|---|---|"]
    for r in d:
        outcome = "wins" if r["noop_win"] > 0.5 else ("dies" if r["noop_death"] > 0.5 else "survives")
        lines.append(f"| {r['level']} | {outcome} | {r['random_win']:.2f} | {r['random_win_median_t']} | {', '.join(r['flags'])} |")
    lines += ["", "NOOP_DEATH = the level is lost with no input (usually the intended challenge, reported as a statistic); TRIVIAL_WIN = random play wins quickly (design flaw candidate).", ""]
open(a.out, "w").write("\n".join(lines)); print("\n".join(lines[:40])); print(f"... written to {a.out}")
if a.render:
    for (lv, fl), xs in sorted(by_flag.items()):
        if sev_of(xs) not in ("A", "B"): continue
        best = min([x for x in xs if x.get("verified")] or xs, key=lambda x: x["first_hit"])
        subprocess.run([sys.executable, "scripts/render_bug.py", "--scan", best["scan"], "--level", lv, "--program", best["program"], "--size", a.size, "--out", os.path.join(os.path.dirname(a.out), "bugs_ab")])
