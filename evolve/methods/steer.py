"""M14 — learned action effects + directed steering (miss diagnosis M6/M8/M9): each worker estimates, per action, the
displacement direction it produces (EMA of the change of the moving bodies' centroid during holds of that action). With that
2-vector per action it can STEER: choose the action whose effect vector best aligns with a desired direction. Desired directions
cycle through target hypotheses computable from features: toward the nearest static non-wall shape (contact / tunnelling /
disarmed hazards), toward the nearest arena face (escapes), and away from the goal after the level was won once (goal shunning),
plus a random-direction hypothesis. Each hypothesis is pursued for a bout of BOUT steps with heavy-tailed holds inside the bout;
a coverage archive with depth-aware restores keeps breadth."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12; CELL_DIMS = (0, 1, 5, 6); CELL_RES = (0.5, 0.5, 0.5, 0.5)
P_RESTORE = 0.05
BOUT = 96                   # steps per steering bout
P_EXPLORE = 0.3             # probability of a random action at a hold boundary (keeps the effect estimates fresh)
LR = 0.3
MU = 2.0; N_MAX = 96
WORLD = 5.0; NW = 4
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _shapes(feats):
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pos = jnp.concatenate([feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2), feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2)], 1)
    vel = jnp.concatenate([feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2), feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)], 1)
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    return pos, jnp.linalg.norm(vel, axis=-1), act
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def _desired(mem, feats):
    """(B, 2) unit direction for the worker's current hypothesis: 0 = nearest static non-wall shape, 1 = nearest arena face, 2 = away from goal, 3 = random."""
    pos, spd, act = _shapes(feats); N = spd.shape[1]
    dyn = mem["dyn"][:N] & act; stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)
    me = feats[:, 0:2]                                                       # moving-bodies centroid
    d = jnp.linalg.norm(pos - me[:, None, :], axis=-1)
    j = jnp.argmin(jnp.where(stat, d, 9.0), axis=1); to_shape = pos[jnp.arange(pos.shape[0]), j] - me
    faces = jnp.stack([-me[:, 0], WORLD - me[:, 0], -me[:, 1], WORLD - me[:, 1]], 1)          # signed distances to left/right/floor/ceiling
    f = jnp.argmin(jnp.abs(faces), axis=1)
    to_face = jnp.stack([jnp.where(f == 0, -1.0, jnp.where(f == 1, 1.0, 0.0)), jnp.where(f == 2, -1.0, jnp.where(f == 3, 1.0, 0.0))], 1)
    away_goal = feats[:, 5:7] - feats[:, 9:11]
    dirs = jnp.stack([to_shape, to_face, away_goal, mem["rand_dir"]], 1)     # (B, 4, 2)
    dsel = dirs[jnp.arange(dirs.shape[0]), mem["hyp"]]
    return dsel / (jnp.linalg.norm(dsel, axis=-1, keepdims=True) + 1e-6)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687], jnp.int32); k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult,
            "effect": jnp.zeros((B, n_act, 2)), "p0": jnp.zeros((B, 2)), "hyp": jax.random.randint(k1, (B,), 0, 4), "rand_dir": jax.random.normal(k2, (B, 2)),
            "t": jnp.int32(0), "dyn": jnp.zeros(64, bool), "won": jnp.bool_(False), "prev_dist": jnp.full(B, 9.0),
            "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k1, (B, K)), axis=1)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    return do, slot, {**mem, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4); new = mem["left"] <= 0
    # update the effect estimate of the action that was just held
    disp = feats[:, 0:2] - mem["p0"]
    effect = mem["effect"].at[jnp.arange(B), mem["cur"]].set(jnp.where(new[:, None], (1 - LR) * mem["effect"][jnp.arange(B), mem["cur"]] + LR * disp, mem["effect"][jnp.arange(B), mem["cur"]]))
    # new bout every BOUT steps: rotate the hypothesis (skip 'away from goal' until the level has been won once)
    new_bout = (mem["t"] % BOUT) == 0
    hyp = jnp.where(new_bout, (mem["hyp"] + 1) % 4, mem["hyp"]); hyp = jnp.where((hyp == 2) & ~mem["won"], 3, hyp)
    rand_dir = jnp.where(new_bout, jax.random.normal(k4, (B, 2)), mem["rand_dir"])          # new_bout is a scalar
    mem2 = {**mem, "hyp": hyp, "rand_dir": rand_dir}
    want = _desired(mem2, feats)
    score = (effect * want[:, None, :]).sum(-1) + 0.05 * jax.random.gumbel(k1, effect.shape[:2])
    steer = jnp.argmax(score, axis=1); rand = jax.random.randint(k2, (B,), 0, mem["n_act"])
    pick = jnp.where(jax.random.uniform(k3, (B,)) < P_EXPLORE, rand, steer)
    cur = jnp.where(new, pick, mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k4, (B,))); left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    p0 = jnp.where(new[:, None], feats[:, 0:2], mem["p0"])
    return cur, {**mem2, "cur": cur, "left": left, "effect": effect, "p0": p0, "t": mem["t"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, act = _shapes(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    won = mem["won"] | (done & (feats[:, 7] < 0.6)).any()          # an episode that ended next to the goal counts as a win
    cell = _cell(mem, feats); live = ~done
    is_new = (mem["cell_slot"][cell] < 0) & live
    first = jnp.full(1 << BITS, B, jnp.int32).at[jnp.where(is_new, cell, 0)].min(jnp.where(is_new, jnp.arange(B), B))
    is_new = is_new & (first[cell] == jnp.arange(B))
    order = jnp.cumsum(is_new) - 1; slot = (mem["next_free"] + order) % K
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    return is_new, jnp.where(is_new, slot, 0), {**mem, "dyn": dyn, "won": won, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
