"""Fitness evaluation of a tester program on the planted-bug SEARCH split (used by ShinkaEvolve's evaluate.py and by
baselines). Instances come from runs/pool.json + runs/split.json."""
from __future__ import annotations
import json, os, time, functools
import numpy as np, jax, jax.numpy as jnp
from .tester import TesterConfig, make_tester, SourceProgram
from .protocol import PROTOCOL_HASH


def stack(xs): return jax.tree.map(lambda *a: jnp.stack(a), *xs)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HookProgram:
    name = "hooks"
    def __init__(self, init, restore, act, archive): self._i, self._r, self._a, self._s = init, restore, act, archive
    def init(self, rng, F, n_act, K, B): return self._i(rng, F, n_act, K, B)
    def restore(self, mem, feats, af, av, avis, rng): return self._r(mem, feats, af, av, avis, rng)
    def act(self, mem, feats, obs, rng): return self._a(mem, feats, obs, rng)
    def archive(self, mem, feats, done, af, av, rng): return self._s(mem, feats, done, af, av, rng)


@functools.lru_cache(maxsize=None)
def load_split(split_path: str, which: str, size: str):
    from .evaluate import meta_from_json, load_instance_level
    split = json.load(open(split_path))
    pool = {p["name"]: p for p in json.load(open(os.path.join(ROOT, split["pool"])))["instances"]}
    names = split[which]
    levels, metas, info = [], [], []
    for n in names:
        p = pool[n]
        st, sp, ep = load_instance_level(os.path.join(ROOT, p["path"]), size)
        probes = p.get("probes", {}); n_prog = max(1, len(probes))
        hard = 1.0 + sum(1 for v in probes.values() if v is None) / n_prog     # 1 (all baselines find it) .. 2 (none does)
        levels.append(st); metas.append(meta_from_json(p["meta"])); info.append(dict(name=n, cls=p["cls"], level=p["level"], hard_weight=hard))
    return levels, metas, info


def evaluate_hooks(init, restore, act, archive, split_path=None, which="search", size="m", seeds=(0,),
                   workers=64, steps=1024, slots=256, chunk=16, log=None, substrate=None, verify_hits=None, chain_log=16384):
    """Returns dict(score, detected, per_class, per_instance...). score = mean over instances/seeds of (1 - first_hit/budget).
    `substrate` (autodev.substrate.Substrate) selects the engine; default Kinetix at `size`."""
    if os.environ.get("AUTODEV_REMOTE_EVAL") == "1" and os.environ.get("AUTODEV_PROGRAM_PATH"):
        # Fitness on the cluster GPU (scripts/eval_server.py); the split/budget/seeds are the server's, not the kwargs here.
        from autodev.remote_eval import remote_evaluate
        return remote_evaluate(os.environ["AUTODEV_PROGRAM_PATH"], log=log or (lambda m: print(m, flush=True)))
    split_path = split_path or os.environ.get("AUTODEV_SPLIT", "runs/split.json")
    if substrate is None:
        from autodev.substrate import KinetixSubstrate
        substrate = KinetixSubstrate(size=size)
    levels, metas, info = substrate.tasks(split_path, which)
    env, ep, sp, feat_fn, oracle = substrate.env, substrate.ep, getattr(substrate, "sp", None), substrate.feat_fn, substrate.oracle
    # Verified fitness (Aug 30): a hit counts only if the action chain that produced it replays from a level reset and raises
    # the witness again. Makes evaluator exploits (e.g. the archive race) worthless to a program search by construction.
    if verify_hits is None:   # default on for Kinetix (deterministic step); off for stochastic engines whose replay cannot reuse the step keys
        from autodev.substrate import KinetixSubstrate
        verify_hits = os.environ.get("AUTODEV_VERIFY_HITS", "1") == "1" and isinstance(substrate, KinetixSubstrate)
    cfg = TesterConfig(num_workers=workers, steps=steps, slots=slots, chain_log=int(chain_log) if verify_hits else 0)
    run = jax.jit(jax.vmap(make_tester(env, ep, sp, feat_fn, HookProgram(init, restore, act, archive), cfg, oracle, 1), in_axes=(0, 0, 0)))
    n = len(levels); fh = np.full((n, len(seeds)), float(cfg.budget)); cov = np.zeros((n, len(seeds))); n_unverified = 0; n_hits = 0
    from autodev.tester import chain_actions, verify_chain_harness
    t0 = time.time()
    for si, seed in enumerate(seeds):
        for c0 in range(0, n, chunk):
            idx = list(range(c0, min(n, c0 + chunk)))
            out = run(jax.random.split(jax.random.PRNGKey(7919 * seed + c0), len(idx)), stack([levels[i] for i in idx]), stack([metas[i] for i in idx]))
            jax.block_until_ready(out)
            fh[idx, si] = np.array(out["first_hit"][:, 0]); cov[idx, si] = np.array(out["coverage"])
            if verify_hits:
                found = np.array(out["found"]); js = [j for j in range(len(idx)) if found[j]]
                if js:
                    hits = {j: jax.tree.map(lambda x: np.array(x[j]), out["hit"]) for j in js}
                    for j in js:
                        n_hits += 1
                        ok = verify_chain_harness(env, ep, sp, feat_fn, levels[idx[j]], chain_actions(hits[j]), oracle, metas[idx[j]], B=workers)
                        if not ok:
                            n_unverified += 1; fh[idx[j], si] = float(cfg.budget)      # not reproducible from reset -> not found
                            if os.environ.get("AUTODEV_UNVERIFIED_LOG"):
                                h = hits[j]; rec = dict(name=info[idx[j]]["name"], seed=int(seed), t=int(h["t"]), worker=int(h["worker"]), chain=chain_actions(h), origin=int(h["origin"]), buf_len=int(h["len"]), log_size=int(h["log_size"]))
                                with open(os.environ["AUTODEV_UNVERIFIED_LOG"], "a") as f: f.write(json.dumps(rec) + "\n")
            if log: log(f"  seed {seed} {idx[0]}-{idx[-1]} found {(fh[idx, si] < cfg.budget).mean():.2f} ({time.time()-t0:.0f}s)")
    norm = 1.0 - np.minimum(fh, cfg.budget) / cfg.budget          # 1 = found immediately, 0 = never
    if os.environ.get("AUTODEV_HARD_WEIGHT", "0") == "1":
        w = np.array([x["hard_weight"] for x in info])[:, None]
        norm_score = float((norm * w).sum() / (w.sum() * fh.shape[1]))
    else:
        norm_score = float(norm.mean())
    classes = sorted({x["cls"] for x in info}, key=lambda c: (c not in ["OOB", "TUNNEL", "SHORTCUT", "ANOMALY", "DISARMED", "GOAL"], c))
    per_class = {c: float(np.mean([norm[i].mean() for i in range(n) if info[i]["cls"] == c])) for c in classes}
    det_class = {c: float(np.mean([(fh[i] < cfg.budget).mean() for i in range(n) if info[i]["cls"] == c])) for c in per_class}
    return dict(score=norm_score, unweighted_score=float(norm.mean()), detected=float((fh < cfg.budget).mean()), per_class=per_class, detected_per_class=det_class,
                first_hit=fh.tolist(), coverage=cov.tolist(), names=[x["name"] for x in info], budget=cfg.budget, wall=time.time() - t0,
                verified=bool(verify_hits), n_hits=int(n_hits), n_unverified=int(n_unverified), protocol=PROTOCOL_HASH)
