"""Rival discovery scaffolds on the same problem, LLM and fitness as the evolutionary run (ShinkaEvolve):

  loop  agentic hill-climb (the "autoresearch" pattern): one complete candidate program per iteration, feedback = score,
        per-class detection and any error; the best program is kept and shown back to the model.
  tree  AIDE-style solution tree: nodes are programs; a parent is chosen by score with some exploration; the operator is
        `debug` if the parent failed, `improve` otherwise, and occasionally `draft` (from the seed) for diversity.

Fitness: autodev.search_eval.evaluate_hooks — locally, or on the Hyak fitness server when AUTODEV_REMOTE_EVAL=1 (the same
server/queue the evolutionary run uses, so all scaffolds are scored identically). Every evaluation is appended to
<results_dir>/nodes.jsonl (program text, score, parent, operator, tokens, cost) and the best program is written to
<results_dir>/best.py.

    AUTODEV_REMOTE_EVAL=1 python evolve/scaffolds.py --strategy loop --budget 30 --results_dir runs/scaffold_loop
    AUTODEV_REMOTE_EVAL=1 python evolve/scaffolds.py --strategy tree --budget 30 --results_dir runs/scaffold_tree
"""
import os, sys, re, json, time, argparse, random, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.environ.setdefault("AUTODEV_ROOT", ROOT)
import ast as _ast
_tree = _ast.parse(open(os.path.join(ROOT, 'evolve', 'run_evolution.py')).read())    # same task description as the evolutionary run,
TASK_SYS_MSG = next(n.value.value for n in _ast.walk(_tree) if isinstance(n, _ast.Assign) and any(getattr(t, 'id', '') == 'TASK_SYS_MSG' for t in n.targets))   # read without importing (its argparse runs at import)

ap = argparse.ArgumentParser()
ap.add_argument("--strategy", choices=["loop", "tree"], default="loop"); ap.add_argument("--budget", type=int, default=30, help="number of fitness evaluations")
ap.add_argument("--results_dir", default="runs/scaffold_loop"); ap.add_argument("--seed_program", default=os.environ.get("AUTODEV_INIT", os.path.join(ROOT, "evolve", "initial.py")))
ap.add_argument("--model", default="headless/claude@claude-sonnet-5?effort=medium"); ap.add_argument("--max_tokens", type=int, default=20000)
ap.add_argument("--history", type=int, default=4, help="loop: how many previous attempts are summarised in the prompt")
ap.add_argument("--p_best", type=float, default=0.7, help="tree: probability of expanding the current best node")
ap.add_argument("--p_draft", type=float, default=0.15, help="tree: probability of drafting from the seed instead of expanding")
ap.add_argument("--rng", type=int, default=0)
a = ap.parse_args(); random.seed(a.rng)
os.makedirs(a.results_dir, exist_ok=True)
NODES = os.path.join(a.results_dir, "nodes.jsonl")

SYSTEM = TASK_SYS_MSG + """

You are one of several research scaffolds being compared on this problem. Respond with ONE complete Python program file in a
single ```python fenced block: keep the module docstring, the `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers, the four
hooks `init`, `restore`, `act`, `archive` (pure JAX, fixed shapes, jit/vmap-compatible), and the `run_experiment` function
exactly as in the program you are shown. Before the code block, write at most 8 lines explaining the change and why it should
reach witness states faster. No other prose after the code block."""


def extract_program(text):
    blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.S)
    if not blocks: return None
    src = blocks[-1]
    if not all(f"def {h}(" in src for h in ("init", "restore", "act", "archive")): return None
    if "def run_experiment" not in src:                      # re-attach the harness entry point verbatim
        seed = open(a.seed_program).read(); src = src.rstrip() + "\n\n\n" + seed[seed.index("def run_experiment"):]
    return src


def evaluate(path):
    """Return (score dict or None, error string or None)."""
    try:
        if os.environ.get("AUTODEV_REMOTE_EVAL") == "1":
            from autodev.remote_eval import remote_evaluate
            r = remote_evaluate(os.path.abspath(path), log=lambda m: print("   ", m, flush=True))
        else:
            from autodev.search_eval import evaluate_hooks
            ns = {}; exec(open(path).read(), ns)
            r = evaluate_hooks(ns["init"], ns["restore"], ns["act"], ns["archive"],
                               seeds=tuple(int(s) for s in os.environ.get("AUTODEV_SEEDS", "0").split(",")),
                               workers=int(os.environ.get("AUTODEV_WORKERS", "64")), steps=int(os.environ.get("AUTODEV_STEPS", "1024")),
                               size=os.environ.get("AUTODEV_SIZE", "m"))
        return {"score": float(r["score"]), "detected": float(r["detected"]), "per_class": r.get("per_class", {}),
                "detected_per_class": r.get("detected_per_class", {})}, None
    except Exception:
        return None, traceback.format_exc()[-1500:]


def feedback(node):
    if node["error"]: return f"FAILED with error:\n{node['error']}"
    r = node["result"]
    return (f"score {r['score']:.4f} (detected {r['detected']:.3f}); per-class score " + ", ".join(f"{k} {v:.3f}" for k, v in r["per_class"].items())
            + "; per-class detected " + ", ".join(f"{k} {v:.2f}" for k, v in r["detected_per_class"].items()))


class LLM:
    def __init__(self):
        from shinka.llm import LLMClient
        self.client = LLMClient(model_names=[a.model], temperatures=0.7, max_tokens=a.max_tokens, verbose=False,
                                headless_work_dir=os.path.join(a.results_dir, "headless"))
        self.tokens_in = self.tokens_out = 0; self.cost = 0.0
    def ask(self, msg):
        res = self.client.query(msg, SYSTEM)
        if res is None: return None
        self.tokens_in += res.input_tokens; self.tokens_out += res.output_tokens; self.cost += float(res.cost or 0.0)
        return res.content


nodes = []
def record(node):
    nodes.append(node)
    with open(NODES, "a") as f: f.write(json.dumps(node) + "\n")


def run_candidate(src, parent, operator, note):
    k = len(nodes); path = os.path.join(a.results_dir, f"cand_{k:03d}.py"); open(path, "w").write(src)
    t0 = time.time(); result, err = evaluate(path)
    node = dict(id=k, path=path, parent=parent, operator=operator, note=note[:600], result=result, error=err, wall=time.time() - t0, program=src)
    record(node)
    best = max((n for n in nodes if n["result"]), key=lambda n: n["result"]["score"], default=None)
    print(f"[{k:3d}] {operator:8s} parent={parent} -> {feedback(node)[:160]} | best {best['result']['score']:.4f} (node {best['id']})" if best else f"[{k:3d}] {operator} -> failed", flush=True)
    if best and best["id"] == k: open(os.path.join(a.results_dir, "best.py"), "w").write(src)
    return node


def program_block(src): return "```python\n" + src.rstrip() + "\n```"


def main():
    llm = LLM()
    seed_src = open(a.seed_program).read()
    print(f"strategy={a.strategy} budget={a.budget} model={a.model} remote={os.environ.get('AUTODEV_REMOTE_EVAL')=='1'}", flush=True)
    root = run_candidate(seed_src, None, "seed", "seed program")
    best = root
    def dump():
        json.dump(dict(strategy=a.strategy, evaluations=len(nodes) - 1, best_score=best["result"]["score"] if best["result"] else None, best_node=best["id"],
                       seed_score=root["result"]["score"] if root["result"] else None, tokens_in=llm.tokens_in, tokens_out=llm.tokens_out, cost=llm.cost, model=a.model),
                  open(os.path.join(a.results_dir, "summary.json"), "w"), indent=1)
    dump()
    while len(nodes) < a.budget + 1:
        if a.strategy == "loop":
            hist = [n for n in nodes[1:]][-a.history:]
            msg = (f"CURRENT BEST PROGRAM (node {best['id']}, {feedback(best)}):\n{program_block(best['program'])}\n\n"
                   + ("PREVIOUS ATTEMPTS (most recent last):\n" + "\n".join(f"- node {n['id']} ({n['operator']}): {n['note'][:200]} -> {feedback(n)[:220]}" for n in hist) + "\n\n" if hist else "")
                   + "Propose the next improvement: one complete program. Prefer structural changes over constant tweaks when the last attempts moved the score by less than 0.01.")
            parent, op = best, "improve"
        else:
            r = random.random()
            if r < a.p_draft:
                parent, op = root, "draft"
                msg = (f"SEED PROGRAM ({feedback(root)}):\n{program_block(root['program'])}\n\nThe current best scores {best['result']['score']:.4f}. "
                       f"Draft a structurally DIFFERENT program from the seed (different archive keying or restore policy or action model), complete file.")
            else:
                cands = [n for n in nodes if n["id"] != 0] or [root]
                parent = best if random.random() < a.p_best else random.choice(cands)
                if parent["error"]:
                    op = "debug"; msg = f"PROGRAM (node {parent['id']}) FAILED:\n{parent['error']}\n\n{program_block(parent['program'])}\n\nFix the error with the smallest change that keeps the idea; complete file."
                else:
                    op = "improve"; msg = (f"PROGRAM (node {parent['id']}, {feedback(parent)}):\n{program_block(parent['program'])}\n\n"
                                          f"Best so far: {best['result']['score']:.4f} (node {best['id']}). Improve this program; complete file.")
        text = llm.ask(msg)
        if text is None:
            record(dict(id=len(nodes), path=None, parent=parent["id"], operator=op, note="LLM returned nothing", result=None, error="no response", wall=0.0, program="")); continue
        src = extract_program(text)
        note = text.split("```")[0].strip()
        if src is None:
            record(dict(id=len(nodes), path=None, parent=parent["id"], operator=op, note=note, result=None, error="no valid program block", wall=0.0, program="")); continue
        node = run_candidate(src, parent["id"], op, note)
        if node["result"] and node["result"]["score"] > best["result"]["score"]: best = node
        dump()
    print("done:", json.load(open(os.path.join(a.results_dir, "summary.json"))))


if __name__ == "__main__":
    main()
