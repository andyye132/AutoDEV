"""Fitness server for the evolution loop. Watches an inbox directory for tester programs (<name>.py), evaluates each on the GPU
with the search-split fitness (autodev.search_eval.evaluate_hooks) and writes <name>.json to the outbox. The Mac-side loop
(evolve/evaluate.py with AUTODEV_REMOTE_EVAL=1) ships programs here through autodev/remote_eval.py, so the LLM stays on the
laptop and only the simulation runs on the cluster. Several servers may share one queue: a program is claimed by an atomic
rename into work/.

    python scripts/eval_server.py --queue runs/hyak/evalq --seeds 0,1,2 --workers 128 --steps 1024 --size l --split runs/hyak/split.json
"""
import os, sys, time, json, glob, shutil, traceback, argparse, importlib.util

ROOT = os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="runs/hyak/evalq"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--workers", type=int, default=128); ap.add_argument("--steps", type=int, default=1024)
ap.add_argument("--size", default="l"); ap.add_argument("--split", default="runs/hyak/split.json"); ap.add_argument("--which", default="search")
ap.add_argument("--max_hours", type=float, default=11.5); ap.add_argument("--poll", type=float, default=5.0)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--substrates", default="", help="multi-engine fitness: 'kind[:arg]|split|weight[|workers|steps|slots|chunk];...' (weighted mean of per-substrate search-split scores); empty = Kinetix only")
a = ap.parse_args()

Q = os.path.join(ROOT, a.queue)
DIRS = {d: os.path.join(Q, d) for d in ("inbox", "work", "outbox", "done")}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)
seeds = tuple(int(s) for s in a.seeds.split(","))
tag = f"{os.environ.get('SLURM_JOB_ID', os.getpid())}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_program(path):
    spec = importlib.util.spec_from_file_location(f"cand_{int(time.time() * 1000)}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f)
    os.replace(tmp, path)


SUBS = []
if a.substrates:
    from autodev.substrate import make_substrate
    for spec in a.substrates.split(";"):
        f = spec.split("|"); kind, split, weight = f[0], f[1], float(f[2]) if len(f) > 2 else 1.0
        cfg = dict(workers=int(f[3]) if len(f) > 3 else a.workers, steps=int(f[4]) if len(f) > 4 else a.steps, slots=int(f[5]) if len(f) > 5 else 256, chunk=int(f[6]) if len(f) > 6 else a.chunk)
        SUBS.append(dict(kind=kind, split=split, weight=weight, cfg=cfg, sub=make_substrate(kind)))
    log("substrates: " + ", ".join(f"{x['kind']} w={x['weight']} {x['cfg']}" for x in SUBS))


def evaluate(path):
    from autodev.search_eval import evaluate_hooks
    mod = load_program(path)
    if not SUBS:
        return evaluate_hooks(mod.init, mod.restore, mod.act, mod.archive, split_path=a.split, which=a.which, size=a.size, seeds=seeds,
                              workers=a.workers, steps=a.steps, chunk=a.chunk, log=log)
    per, wsum, total = {}, 0.0, 0.0
    for x in SUBS:
        r = evaluate_hooks(mod.init, mod.restore, mod.act, mod.archive, split_path=x["split"], which=a.which, seeds=seeds, substrate=x["sub"],
                           workers=x["cfg"]["workers"], steps=x["cfg"]["steps"], slots=x["cfg"]["slots"], chunk=x["cfg"]["chunk"], log=log)
        per[x["kind"]] = dict(score=r["score"], detected=r["detected"], per_class=r["per_class"], detected_per_class=r["detected_per_class"], wall=r["wall"])
        total += x["weight"] * r["score"]; wsum += x["weight"]
        log(f"  {x['kind']}: score {r['score']:.4f} detected {r['detected']:.3f} ({r['wall']:.0f}s)")
    return dict(score=total / max(wsum, 1e-9), detected=float(sum(v["detected"] for v in per.values()) / len(per)),
                per_class={f"{k}:{c}": v for k, d in per.items() for c, v in d["per_class"].items()},
                detected_per_class={f"{k}:{c}": v for k, d in per.items() for c, v in d["detected_per_class"].items()},
                per_substrate=per, wall=float(sum(v["wall"] for v in per.values())))


log(f"server {tag} up: queue={Q} seeds={seeds} workers={a.workers} steps={a.steps} size={a.size} split={a.split} which={a.which} "
    f"hard_weight={os.environ.get('AUTODEV_HARD_WEIGHT', '0')}")
t_start = time.time(); n_done = 0
while time.time() - t_start < a.max_hours * 3600:
    pending = sorted(glob.glob(os.path.join(DIRS["inbox"], "*.py")))
    if not pending:
        time.sleep(a.poll); continue
    src = pending[0]; name = os.path.basename(src)[:-3]
    work = os.path.join(DIRS["work"], f"{name}.py")
    try:
        os.rename(src, work)                      # atomic claim; another server may have taken it
    except FileNotFoundError:
        continue
    log(f"evaluating {name}")
    t0 = time.time()
    try:
        res = evaluate(work)
        res["server"] = tag; res["program"] = name
        write_json(os.path.join(DIRS["outbox"], f"{name}.json"), res)
        log(f"done {name}: score {res['score']:.4f} detected {res['detected']:.3f} ({time.time() - t0:.0f}s)")
    except Exception:
        err = traceback.format_exc()
        write_json(os.path.join(DIRS["outbox"], f"{name}.json"), {"score": 0.0, "detected": 0.0, "per_class": {}, "detected_per_class": {},
                                                                  "wall": time.time() - t0, "error": err[-4000:], "server": tag, "program": name})
        log(f"FAILED {name}: {err.strip().splitlines()[-1][:200]}")
    shutil.move(work, os.path.join(DIRS["done"], f"{name}.py"))
    n_done += 1
log(f"server {tag} exiting after {n_done} programs ({(time.time() - t_start) / 3600:.1f} h)")
