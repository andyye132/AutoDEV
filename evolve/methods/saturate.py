"""M16 — actuator saturation / self-folding (miss diagnosis M7, human QA "drive everything to its stops"): with one binding
per action, a worker cycles through ALL motor bindings in the same direction, one step each, for a whole bout — the linkage
folds toward its joint limits (walkers, hoppers, cartpoles overlap their own recoloured links); other bouts hold a single
thruster or use zeta random holds. Coverage archive with 1/sqrt(visits) restores."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12; CELL_DIMS = (0, 1, 2, 3, 5, 6); CELL_RES = (0.5, 0.5, 1.0, 1.0, 0.5, 0.5)
P_RESTORE = 0.05; BOUT = 96; NM = 4
MODE_P = (0.5, 0.2, 0.3)          # fold / thruster / zeta-random
MU = 2.0; N_MAX = 120
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], jnp.int32)
    return {"n_act": jnp.int32(n_act), "t": jnp.int32(0), "mode": jnp.zeros(B, jnp.int32), "sign": jnp.zeros(B, jnp.int32), "thr": jnp.zeros(B, jnp.int32),
            "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult, "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k1, (B, K)), axis=1)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    return do, slot, {**mem, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; t = mem["t"]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    new_bout = (t % BOUT) == 0
    mode = jnp.where(new_bout, jax.random.choice(k1, 3, (B,), p=jnp.array(MODE_P)), mem["mode"])
    sign = jnp.where(new_bout, jax.random.randint(k2, (B,), 0, 2), mem["sign"])
    thr = jnp.where(new_bout, jax.random.randint(k3, (B,), 0, jnp.maximum(mem["n_act"] - 2 * NM - 1, 1)), mem["thr"])
    fold = 2 * ((t + jnp.arange(B)) % NM) + sign                              # cycle all motor bindings, same direction
    thr_action = jnp.where(mem["n_act"] > 2 * NM + 1, 2 * NM + 1 + thr, fold)
    new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k4, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k4, (B,))); left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    action = jnp.where(mode == 0, fold, jnp.where(mode == 1, thr_action, cur))
    return action, {**mem, "t": t + 1, "mode": mode, "sign": sign, "thr": thr, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    cell = _cell(mem, feats); live = ~done
    is_new = (mem["cell_slot"][cell] < 0) & live
    first = jnp.full(1 << BITS, B, jnp.int32).at[jnp.where(is_new, cell, 0)].min(jnp.where(is_new, jnp.arange(B), B))
    is_new = is_new & (first[cell] == jnp.arange(B))
    order = jnp.cumsum(is_new) - 1; slot = (mem["next_free"] + order) % K
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    return is_new, jnp.where(is_new, slot, 0), {**mem, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
