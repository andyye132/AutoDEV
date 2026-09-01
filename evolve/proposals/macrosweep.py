"""Checkpointed action-macro sweep ("macrosweep") -- a breadth-first tree search over structured action programs.

Idea: sticky-random actions almost never produce the *coherent* input programs a human tester tries first: hold one
motor to the stop for a second, reverse it rhythmically, fire a thruster continuously, pulse it, sweep every motor in
turn. This program enumerates such macros deterministically instead of sampling them.
(1) act() replays a fixed library of ~36 macros (sustained holds, fast/slow reversals, thruster holds and pulse
    trains, thruster+motor interleaves, all-motor sweeps, a settle macro, and short-burst variants for finer
    branching); each worker advances to the next macro in the library every time one finishes, so the batch sweeps
    the whole library rather than re-sampling the same few behaviours.
(2) The archive is the search *frontier*: a macro endpoint is stored only if it is farther than NOV_EPS from every
    stored state in a scaled position/velocity metric, plus any mid-macro state that gets unusually fast.
(3) restore() fires exactly when a macro finishes and sends each worker to a *different* least-expanded frontier node
    (ranked by the harness's visit counts, with noise for ties). The result is a breadth-first expansion of the
    (state x macro) product: depth comes from chaining macros through checkpoints, breadth from the library.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
MACRO_L = 20        # length of a full macro
SHORT_L = 6         # length of the short-burst macros (finer branching factor)
NOV_EPS = 0.30      # novelty radius (scaled feature units) required to store an endpoint
NOV_DIMS = (0, 1, 2, 3, 4, 5, 6, 9, 10)                    # dyn pos/vel/speed, green pos, blue pos
NOV_SCALE = (1.0, 1.0, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0)
EPS_RAND = 0.03     # tiny action noise, breaks symmetry between workers on identical checkpoints
SPD_HOT = 4.0       # mid-macro store threshold on dyn_max_speed
VIS_NOISE = 0.9     # tie-break noise on the frontier visit ranking


def _macros(n_act):
    """Deterministic macro library: (M, MACRO_L) action table + per-macro length."""
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    ms = []
    def add(pat, L): ms.append(([int(pat[i % len(pat)]) for i in range(MACRO_L)], L))
    for m in range(nm):
        add([2 * m], MACRO_L); add([2 * m + 1], MACRO_L)                    # sustained pushes
        add([2 * m, 2 * m, 2 * m + 1, 2 * m + 1], MACRO_L)                  # fast reversal
        add([2 * m] * 5 + [2 * m + 1] * 5, MACRO_L)                         # slow reversal
    for t in thr:
        add([t], MACRO_L); add([t, t, noop, noop], MACRO_L)                 # thruster hold / pulse train
    for m in range(nm):
        add([thr[m % len(thr)], 2 * m], MACRO_L)                            # thrust + motor interleave
    add([2 * m for m in range(nm)], MACRO_L); add([2 * m + 1 for m in range(nm)], MACRO_L)
    add([noop], MACRO_L)                                                    # settle
    for m in range(nm):
        add([2 * m], SHORT_L); add([2 * m + 1], SHORT_L)                    # short bursts
    add([thr[0], noop], SHORT_L)
    return [p for p, _ in ms], [L for _, L in ms]


def init(rng, F, n_act, K, B):
    tab, ln = _macros(n_act)
    M = len(tab)
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "M": jnp.int32(M),
            "table": jnp.array(tab, jnp.int32), "mlen": jnp.array(ln, jnp.int32),
            "nov": jnp.array(NOV_DIMS, jnp.int32), "sc": jnp.array(NOV_SCALE, jnp.float32),
            "mid": jnp.arange(B, dtype=jnp.int32) % jnp.int32(M), "ph": jnp.zeros(B, jnp.int32),
            "at_end": jnp.zeros(B, bool), "next_free": jnp.int32(0)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    key = jnp.where(arch_valid, arch_visits.astype(jnp.float32) + VIS_NOISE * jax.random.uniform(rng, (K,)), 1e6)
    slot = jnp.argsort(key)[jnp.arange(B) % K]           # distinct least-expanded frontier nodes
    return mem["at_end"] & arch_valid.any(), slot, mem


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    ph = jnp.clip(mem["ph"], 0, mem["table"].shape[1] - 1)
    a = mem["table"][mem["mid"], ph]
    a = jnp.where(jax.random.uniform(k1, (B,)) < EPS_RAND, jax.random.randint(k2, (B,), 0, mem["n_act"]), a)
    nxt = mem["ph"] + 1
    end = nxt >= mem["mlen"][mem["mid"]]
    return a.astype(jnp.int32), {**mem, "ph": jnp.where(end, 0, nxt),
                                 "mid": jnp.where(end, (mem["mid"] + 1) % mem["M"], mem["mid"]), "at_end": end}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    x = feats[:, mem["nov"]] * mem["sc"]
    y = arch_feats[:, mem["nov"]] * mem["sc"]
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    mind = jnp.min(jnp.where(arch_valid[None, :], d2, 1e12), axis=1)
    store = (~done) & (mem["at_end"] | (feats[:, 4] > SPD_HOT)) & (mind > NOV_EPS ** 2)
    slot = (mem["next_free"] + jnp.cumsum(store) - 1) % mem["K"]
    return store, slot, {**mem, "next_free": (mem["next_free"] + store.sum()) % mem["K"]}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
