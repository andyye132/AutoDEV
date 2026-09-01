"""H2 — the survey's suggested hybrid: Go-Explore backbone with (i) cells = position x contact-bitmask hash x speed band
(contacts are where planted defects live), (ii) PLR-style restore selection (Jiang et al., ICML 2021):
priority = (1 - RHO) * score + RHO * staleness, where score = the best importance improvement produced by expansions from
the slot and staleness = steps since the slot was last selected, (iii) a 20% reserve of workers that never restore (PBT-style
population split), (iv) heavy-tailed (zeta) holds. Importance is class-agnostic (travel, speed, boundary and goal proximity)."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12
CELL_RES_POS, CELL_RES_VEL, N_SPEED = 0.5, 1.0, 4
P_RESTORE = 0.05
RHO = 0.1                 # staleness weight in the PLR priority
RESERVE_FRAC = 0.2        # workers that never restore
MU = 2.0; N_MAX = 120
WORLD = 5.0
W_DIST, W_SPEED, W_BOUND, W_GOAL = 1.0, 0.6, 0.8, 0.4
LAYOUT = {}
def _layout(F):
    """Per-shape blocks from the feature width (features.py): 11 head scalars; poly pos/vel/rot/active/role (7 Np); circle
    pos/vel/active/role (6 Nc); contact flags (poly-poly pairs minus the 6 wall-wall pairs, circle-poly, circle-circle)."""
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0)
    C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _importance(feats, start):
    d_travel = jnp.linalg.norm(feats[:, 0:2] - start[:, 0:2], axis=-1) / WORLD
    speed = jnp.minimum(feats[:, 4] / 8.0, 1.5)
    edge = jnp.minimum(jnp.minimum(feats[:, 0], WORLD - feats[:, 0]), jnp.minimum(feats[:, 1], WORLD - feats[:, 1]))
    near_bound = 1.0 - jnp.clip(edge / (0.5 * WORLD), 0.0, 1.0)
    goal = 1.0 - jnp.clip(feats[:, 7] / WORLD, 0.0, 1.0)
    return W_DIST * d_travel + W_SPEED * speed + W_BOUND * near_bound + W_GOAL * goal
def _cell(mem, feats):
    pos = jnp.floor(feats[:, jnp.array([0, 1, 5, 6])] / CELL_RES_POS).astype(jnp.int32)
    vel = jnp.floor(feats[:, jnp.array([2, 3])] / CELL_RES_VEL).astype(jnp.int32)
    spd = jnp.clip((feats[:, 4] / 8.0 * N_SPEED).astype(jnp.int32), 0, N_SPEED - 1)[:, None]
    contact = (feats[:, _layout(feats.shape[1])["cfrom"]:] > 0.5).astype(jnp.int32)
    n = contact.shape[1]; cm = ((jnp.arange(n, dtype=jnp.int32) * 40503 + 97) % 1000003).astype(jnp.int32)
    chash = (contact * cm).sum(-1)[:, None]
    x = jnp.concatenate([pos, vel, spd, chash], axis=1)
    return ((x * mem["mult"][: x.shape[1]]).sum(-1)) & ((1 << BITS) - 1)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863, 32452843, 67867967], jnp.int32)
    cmult = jnp.array([(i * 2654435761 + 97) % 1000003 for i in range(64)], jnp.int32)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult, "cmult": cmult,
            "t": jnp.int32(0), "start": jnp.zeros((B, F)), "have_start": jnp.bool_(False), "best": jnp.zeros(B), "base": jnp.zeros(B),
            "parent": jnp.full(B, -1, jnp.int32), "score": jnp.zeros(K), "last_sel": jnp.zeros(K, jnp.int32),
            "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    start = jnp.where(mem["have_start"], mem["start"], feats)
    staleness = (mem["t"] - mem["last_sel"]).astype(jnp.float32) / 256.0
    prio = (1 - RHO) * mem["score"] + RHO * staleness
    logits = jnp.where(arch_valid, prio, -jnp.inf)
    slot = jnp.argmax(logits[None, :] + 0.3 * jax.random.gumbel(k1, (B, K)), axis=1)
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any() & ~reserve
    last_sel = mem["last_sel"].at[jnp.where(do, slot, 0)].set(jnp.where(do, mem["t"], mem["last_sel"][0]))
    base = jnp.where(do, arch_feats[slot, 4] * 0.0 + _importance(arch_feats[slot], start), mem["base"])   # importance at the restore point
    parent = jnp.where(do, slot, mem["parent"]); best = jnp.where(do, 0.0, mem["best"])
    return do, slot, {**mem, "start": start, "have_start": jnp.bool_(True), "last_sel": last_sel, "base": base, "parent": parent, "best": best, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left, "t": mem["t"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    imp = _importance(feats, mem["start"]); gain = imp - mem["base"]
    has_parent = mem["parent"] >= 0
    score = mem["score"].at[jnp.where(has_parent, mem["parent"], 0)].max(jnp.where(has_parent, gain, -jnp.inf))
    score = jnp.maximum(score, 0.0)
    cell = _cell(mem, feats); live = ~done
    is_new = (mem["cell_slot"][cell] < 0) & live
    first = jnp.full(1 << BITS, B, jnp.int32).at[jnp.where(is_new, cell, 0)].min(jnp.where(is_new, jnp.arange(B), B))
    is_new = is_new & (first[cell] == jnp.arange(B))
    order = jnp.cumsum(is_new) - 1; slot = (mem["next_free"] + order) % K
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    score = score.at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, 0.0, score[0]))
    last_sel = mem["last_sel"].at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, mem["t"], mem["last_sel"][0]))
    return is_new, jnp.where(is_new, slot, 0), {**mem, "score": score, "last_sel": last_sel, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
