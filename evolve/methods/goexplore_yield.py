"""M10 — Go-Explore archive with downstream-novelty credit on restore slots (AFLFast, Bohme et al. CCS 2016; Entropic, FSE 2020):
every worker remembers the slot it was last restored from; when it discovers a new cell, that source slot's yield increases.
Restore priority = log((1 + yield) / (1 + selections)) - 0.5 log(1 + visits): slots whose expansions keep producing novelty get
more expansions — credit assignment through restore, which the plain coverage archive lacks. Zeta holds."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12          # hash bits for archive cells
P_RESTORE = 0.05   # per-step probability that a worker jumps to an archived state
MU = 2.0           # zeta exponent of the hold-duration distribution (heavy-tailed; Dabney et al. 2021)
N_MAX = 120        # truncation of hold durations
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)   # feature dims used to define cells
CELL_RES = 0.5     # coarse grid resolution for cells


def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
            "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS)), "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "next_free": jnp.int32(0),
            "yield": jnp.zeros(K), "sel": jnp.zeros(K), "src": jnp.full(B, -1, jnp.int32),
            "cdf": jnp.cumsum((jnp.arange(1, N_MAX + 1, dtype=jnp.float32) ** (-MU)) / (jnp.arange(1, N_MAX + 1, dtype=jnp.float32) ** (-MU)).sum())}


def _cell(mem, feats):
    x = jnp.round(feats[:, jnp.array(CELL_DIMS)] / CELL_RES) * CELL_RES
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    do = jax.random.uniform(k1, (B,)) < P_RESTORE
    power = jnp.log((1.0 + mem["yield"]) / (1.0 + mem["sel"]))
    logits = jnp.where(arch_valid, power - 0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k2, (B, arch_valid.shape[0])), axis=1)
    do = do & arch_valid.any()
    sel = mem["sel"].at[jnp.where(do, slot, 0)].add(do.astype(jnp.float32)); src = jnp.where(do, slot, mem["src"])
    return do, slot, {**mem, "sel": sel, "src": src}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))            # n ~ zeta(MU)
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    cell = _cell(mem, feats)
    is_new = (mem["cell_slot"][cell] < 0) & ~done
    order = jnp.cumsum(is_new) - 1
    slot = (mem["next_free"] + order) % mem["K"]
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    credit = is_new & (mem["src"] >= 0)
    yld = mem["yield"].at[jnp.where(credit, mem["src"], 0)].add(credit.astype(jnp.float32))
    yld = yld.at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, 0.0, yld[0])); sel = mem["sel"].at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, 0.0, mem["sel"][0]))
    src = jnp.where(done, -1, mem["src"])
    return is_new, slot, {**mem, "cell_slot": cell_slot, "yield": yld, "sel": sel, "src": src, "next_free": (mem["next_free"] + is_new.sum()) % mem["K"]}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
