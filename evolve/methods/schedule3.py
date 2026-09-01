"""Time-sliced schedule of the three complementary leaders: actuator saturation (constant-input tier + ANOMALY), velocity-aligned
drive (sealed-class leader), learned-effects steering (held-out-level leader). Unlike the worker-split portfolios (flat) and the
mechanism fusions (below their parents), each phase keeps ALL workers and the phases share one archive, so phase p can restore
from states phase p-1 discovered. Phase length 341 steps -> the cycle repeats ~2x at the 2048-step evaluation budget.
"""
import os, importlib.util
import jax, jax.numpy as jnp

def _load(p):
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3] + "_sub", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
_A = _load("evolve/methods/saturate.py"); _B = _load("evolve/methods/velalign.py"); _C = _load("evolve/methods/steer.py")
PHASE = 341

def _sel(p, a, b, c): return jnp.where(p == 0, a, jnp.where(p == 1, b, c))

def init(rng, F, n_act, K, B):
    k1, k2, k3 = jax.random.split(rng, 3)
    return {"t": jnp.int32(0), "a": _A.init(k1, F, n_act, K, B), "b": _B.init(k2, F, n_act, K, B), "c": _C.init(k3, F, n_act, K, B)}

def restore(mem, feats, af, av, avis, rng):
    da, sa, ma = _A.restore(mem["a"], feats, af, av, avis, rng)
    db, sb, mb = _B.restore(mem["b"], feats, af, av, avis, rng)
    dc, sc, mc = _C.restore(mem["c"], feats, af, av, avis, rng)
    p = (mem["t"] // PHASE) % 3
    return _sel(p, da, db, dc), _sel(p, sa, sb, sc), {**mem, "a": ma, "b": mb, "c": mc}

def act(mem, feats, obs, rng):
    aa, ma = _A.act(mem["a"], feats, obs, rng)
    ab, mb = _B.act(mem["b"], feats, obs, rng)
    ac, mc = _C.act(mem["c"], feats, obs, rng)
    p = (mem["t"] // PHASE) % 3
    return _sel(p, aa, ab, ac).astype(jnp.int32), {**mem, "t": mem["t"] + 1, "a": ma, "b": mb, "c": mc}

def archive(mem, feats, done, af, av, rng):
    da, sa, ma = _A.archive(mem["a"], feats, done, af, av, rng)
    db, sb, mb = _B.archive(mem["b"], feats, done, af, av, rng)
    dc, sc, mc = _C.archive(mem["c"], feats, done, af, av, rng)
    p = (mem["t"] // PHASE) % 3
    return _sel(p, da, db, dc), _sel(p, sa, sb, sc), {**mem, "a": ma, "b": mb, "c": mc}

def run_experiment(**kwargs):
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
