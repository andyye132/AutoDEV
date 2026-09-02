"""Best-of-N ablation (the arXiv:2602.16805 comparison): does the EVOLUTIONARY loop beat non-evolutionary
i.i.d. sampling from the same LLM at equal LLM-call budget?

Each of N draws is INDEPENDENT: conditioned only on (task system message, the seed program) — no parent-climbing,
no inspirations, no feedback, no cross-sample information. This is exactly the "IID sampling from an LLM prompted
to solve the problem, conditioned on the seed" baseline. We reuse ShinkaEvolve's own headless CLI invocation, the
identical run-11 TASK_SYS_MSG, the split-and-pump seed run 11 was seeded from, and run 11's 25/15 sonnet/opus mix.
Each returned program is scored on the SAME fitness server run 11 used (verified harness, run-11 split, 3 seeds).

best-of-N vs run 11's evolved best (0.7917 fitness, +0.009 over the 0.7827 seed) at matched N=40 tests interchangeability.

Usage:
  AUTODEV_REMOTE_QUEUE=/gscratch/raivn/andy132/AutoDev/runs/hyak/evalqf \
  SHINKA_HEADLESS_COMMAND="npx -y @roberttlange/headless" \
  python scripts/best_of_n.py --n 40 --seed_prog runs/evolve_run11_hard/exports/best_upto_gen000.py \
    --out results/best_of_n.json --workdir runs/best_of_n
"""
import argparse, json, os, re, subprocess, sys, time, uuid, shlex, threading, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AUTODEV_ROOT", ROOT)

# Byte-identical to evolve/run_evolution.py's TASK_SYS_MSG at run-11 launch (the "system instructions").
TASK_SYS_MSG = """You are improving a game-testing program for 2D physics puzzle games (Kinetix, JAX). A batch of B workers
explores one level; the harness keeps K archive slots of saved states. Your program defines four pure-JAX hooks:
init(rng, F, n_act, K, B) -> mem; restore(mem, feats, arch_feats, arch_valid, arch_visits, rng) -> (do (B,), slot (B,), mem);
act(mem, feats, obs, rng) -> (action (B,), mem); archive(mem, feats, done, arch_feats, arch_valid, rng) -> (do (B,), slot (B,), mem).
The fitness is how quickly the workers reach states that expose planted defects (leaving the arena, passing through a solid,
winning via an unintended goal, runaway physics, touching a disarmed hazard) across many levels — measured as
1 - steps_to_first_witness/budget, averaged. A held-out defect class and held-out levels are NOT in the fitness; programs
that overfit the visible classes will not transfer, so prefer general exploration strategies (systematic coverage of
position/velocity space, reaching extreme states, high-energy interactions, varied action patterns, archive diversity).
Rules: only jax/jnp; all shapes fixed (no Python control flow on traced values); keep mem a dict of arrays with fixed shapes;
actions are integers in [0, n_act); never index arch arrays out of range; the program must run in a few seconds per level.
Only modify code between the EVOLVE-BLOCK markers."""

USER_REQUEST = """Here is the CURRENT program (the part you may change is between the EVOLVE-BLOCK-START and EVOLVE-BLOCK-END
markers). Propose ONE improved version of the program that reaches planted-defect states faster, on average, across many levels.
Think about the mechanism, then commit to a concrete change.

Return the COMPLETE replacement for the EVOLVE-BLOCK region — everything that goes between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END (do not include the marker lines themselves) — inside a single fenced block exactly like:

<PROGRAM>
...your complete EVOLVE-BLOCK code here...
</PROGRAM>

Constraints (must hold or the program scores zero): only jax/jnp; fixed shapes, no Python control flow on traced values;
mem is a dict of fixed-shape arrays; actions are integers in [0, n_act); never index the arch arrays out of range;
it must define init, restore, act, archive with the signatures above and run in a few seconds per level.

CURRENT PROGRAM:
```python
{seed}
```"""

HEADLESS_CMD = shlex.split(os.environ.get("SHINKA_HEADLESS_COMMAND", "npx -y @roberttlange/headless"))
QUEUE = os.environ.get("AUTODEV_REMOTE_QUEUE", "/gscratch/raivn/andy132/AutoDev/runs/hyak/evalqf")
HOST = os.environ.get("AUTODEV_REMOTE_HOST", "hyak")
SSH = ["-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15"]


def split_seed(seed_path):
    txt = open(seed_path).read()
    a = txt.index("# EVOLVE-BLOCK-START")
    b = txt.index("# EVOLVE-BLOCK-END")
    head = txt[:a] + "# EVOLVE-BLOCK-START\n"
    tail = "# EVOLVE-BLOCK-END" + txt[b + len("# EVOLVE-BLOCK-END"):]
    block = txt[a + len("# EVOLVE-BLOCK-START"):b].strip("\n")
    return head, block, tail


def draw_program(model, effort, seed_block_program, workdir, idx):
    """One independent LLM draw. Returns the returned EVOLVE-BLOCK text, or None on failure."""
    prompt_dir = os.path.join(workdir, "prompts"); os.makedirs(prompt_dir, exist_ok=True)
    prompt_path = os.path.join(prompt_dir, f"prompt_{idx:03d}_{uuid.uuid4().hex[:6]}.md")
    prompt = ("# System Instructions\n\n" + TASK_SYS_MSG + "\n\n# User Request\n\n"
              + USER_REQUEST.format(seed=seed_block_program))
    open(prompt_path, "w").write(prompt)
    cmd = [*HEADLESS_CMD, "claude", "--prompt-file", prompt_path, "--work-dir", os.path.abspath(workdir),
           "--allow", "read-only", "--usage", "--model", model, "--reasoning-effort", effort]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    out = r.stdout or ""
    m = re.search(r"<PROGRAM>\s*(.*?)\s*</PROGRAM>", out, re.DOTALL)
    if not m:
        # fall back to a fenced python block if the model ignored the tag
        m = re.search(r"```(?:python)?\s*(.*?)```", out, re.DOTALL)
    if not m:
        return None, "no_program_in_output"
    return m.group(1), None


def score_remote(program_path, poll=20, timeout_s=3000):
    name = f"bon-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"
    tmp, final, outp = f"{QUEUE}/inbox/{name}.py.tmp", f"{QUEUE}/inbox/{name}.py", f"{QUEUE}/outbox/{name}.json"
    subprocess.run(["scp", *SSH, program_path, f"{HOST}:{tmp}"], capture_output=True, text=True, timeout=120)
    subprocess.run(["ssh", *SSH, HOST, f"mv {tmp} {final}"], capture_output=True, text=True, timeout=60)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll)
        r = subprocess.run(["ssh", *SSH, HOST, f"cat {outp} 2>/dev/null"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            try:
                res = json.loads(r.stdout)
            except json.JSONDecodeError:
                continue
            return res
    return {"error": "timeout", "score": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed_prog", default="runs/evolve_run11_hard/exports/best_upto_gen000.py")
    ap.add_argument("--out", default="results/best_of_n.json")
    ap.add_argument("--workdir", default="runs/best_of_n")
    ap.add_argument("--parallel", type=int, default=3)  # match the 3 fixed servers on the queue
    ap.add_argument("--n_sonnet", type=int, default=25)  # run-11 mix: 25 sonnet-medium
    ap.add_argument("--n_opus", type=int, default=15)    #             15 opus-high
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    head, seed_block, tail = split_seed(a.seed_prog)
    seed_full_program = open(a.seed_prog).read()

    # build the model schedule: n_sonnet + n_opus, shuffled deterministically (idx-based, no wall-clock RNG)
    schedule = ([("claude-sonnet-5", "medium")] * a.n_sonnet + [("claude-opus-5", "high")] * a.n_opus)[:a.n]
    rng = random.Random(11)  # run-11
    rng.shuffle(schedule)

    results = [None] * a.n
    lock = threading.Lock()
    sem = threading.Semaphore(a.parallel)

    def work(i):
        model, effort = schedule[i]
        with sem:
            block, err = draw_program(model, effort, seed_full_program, a.workdir, i)
        rec = {"idx": i, "model": model, "effort": effort, "gen_error": err}
        if block is None:
            rec["score"] = None; rec["status"] = f"gen_fail:{err}"
        else:
            prog = head + block.strip("\n") + "\n" + tail
            ppath = os.path.join(a.workdir, f"cand_{i:03d}.py"); open(ppath, "w").write(prog)
            res = score_remote(ppath)
            rec["score"] = res.get("score"); rec["detected"] = res.get("detected")
            rec["status"] = "error:" + str(res.get("error"))[:80] if res.get("error") else "ok"
        with lock:
            results[i] = rec
            done = sum(1 for r in results if r is not None)
            sc = rec["score"]
            print(f"[{done}/{a.n}] draw {i:2d} {model.split('-')[1]}/{effort}: "
                  f"{'FAIL '+str(rec['status']) if sc is None else f'{sc:.4f}'}", flush=True)
            json.dump({"config": vars(a), "seed_fitness": 0.7827, "run11_best_fitness": 0.7917,
                       "draws": [r for r in results if r is not None]}, open(a.out, "w"), indent=1)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(a.n)]
    for t in threads: t.start()
    for t in threads: t.join()

    scored = [r["score"] for r in results if r and r["score"] is not None]
    valid = len(scored)
    best = max(scored) if scored else None
    # best-of-k curve (ordered by draw index = arrival, matches evolution's sequential budget)
    curve, run_best = [], None
    for r in results:
        if r and r["score"] is not None:
            run_best = r["score"] if run_best is None else max(run_best, r["score"])
        curve.append(run_best)
    summary = {"n": a.n, "valid": valid, "best_of_n": best, "seed_fitness": 0.7827,
               "run11_best_fitness": 0.7917, "run11_generations": 40,
               "best_of_n_curve": curve,
               "verdict": None}
    if best is not None:
        summary["verdict"] = ("best-of-N >= run-11 evolved (interchangeable)" if best >= 0.7917 - 0.005
                              else "evolution ahead of best-of-N")
    out = {"config": vars(a), "summary": summary, "draws": [r for r in results if r is not None]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n=== SUMMARY ===")
    print(f"valid draws: {valid}/{a.n}   best-of-N: {best}   seed: 0.7827   run-11 evolved best: 0.7917")
    print(f"verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
