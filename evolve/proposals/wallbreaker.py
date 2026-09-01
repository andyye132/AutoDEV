"""Boundary-seeking escape search ("wallbreaker").

Idea: most planted defects that matter live at the *edges* of the reachable set, so this program optimises a single
scalar -- how far the outermost movable shape is from the arena centre, plus how fast it is still moving outward.
(1) A self-calibrating mobility mask learns online which shape indices ever move, so static walls (which sit outside
    the box and would dominate any extremeness statistic) are excluded automatically without any oracle knowledge.
(2) act() is a per-worker (1+1) hill-climber on the action *distribution*: each worker holds a preference logit vector,
    perturbs it every HOLD steps, and keeps the perturbation only if the batch of steps made progress along the
    worker's assigned compass direction; stalled workers pick a new direction. This finds sustained directional pushes
    that sticky-random actions almost never produce.
(3) The archive is an elite pool over the boundary score: each step the weakest slots are offered to the workers and
    overwritten only by strictly better states, so restores branch from the current record-holders (plus a small
    random-slot channel for diversity). Escaping the box (OOB), ramming a wall hard enough to pass through a
    non-solid obstacle (TUNNEL) and extreme sustained pushes (ANOMALY) all fall out of the same objective.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
CENTER = 2.5        # centre of the 5x5 arena; scores are measured relative to it
BETA_VEL = 0.5      # weight of the outward radial velocity in the boundary score
MOB_THR = 0.03      # speed above which a shape index is considered movable (learned online)
P_BASE = 0.02       # baseline per-step restore probability
P_STALL = 0.30      # restore probability once a worker has stalled
STALL_T = 20        # steps without a new outward record before a worker counts as stalled
REDIR_T = 60        # steps stalled before a worker draws a new escape direction
TEMP = 1.5          # inverse temperature on archive boundary scores
HOLD = 8            # length of one hill-climbing trial block
SIGMA = 1.5         # perturbation scale of the action-preference logits
STICKY = 0.55       # probability of repeating the previous action inside a block
EPS_RAND = 0.10     # uniform-random action rate
MARGIN = 0.01       # improvement required to evict an elite slot
P_DIVERSE = 0.07    # probability of storing into a random slot regardless of score
DIRS = ((1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0), (-0.7071, 0.7071),
        (-1.0, 0.0), (-0.7071, -0.7071), (0.0, -1.0), (0.7071, -0.7071))


def _idx(F):
    """Recover the per-shape feature blocks from the feature width (unique for every Kinetix size)."""
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41)
            if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    if len(sols) != 1:
        return dict(use=0.0, px=[0], py=[1], vx=[2], vy=[3], ai=[8])
    Np, Nc = sols[0]
    P = 11; PV = P + 2 * Np; PA = PV + 2 * Np + Np; C = PA + 2 * Np; CV = C + 2 * Nc; CA = CV + 2 * Nc
    return dict(use=1.0,
                px=[P + 2 * i for i in range(Np)] + [C + 2 * i for i in range(Nc)],
                py=[P + 2 * i + 1 for i in range(Np)] + [C + 2 * i + 1 for i in range(Nc)],
                vx=[PV + 2 * i for i in range(Np)] + [CV + 2 * i for i in range(Nc)],
                vy=[PV + 2 * i + 1 for i in range(Np)] + [CV + 2 * i + 1 for i in range(Nc)],
                ai=[PA + i for i in range(Np)] + [CA + i for i in range(Nc)])


def _w(mem, X):
    """Weights of shapes that are active AND have been seen to move (walls therefore get weight 0)."""
    return X[:, mem["ai"]] * (mem["mob"] > MOB_THR).astype(jnp.float32) * mem["use"]


def _out_score(mem, X):
    """How close to / far past the box boundary the outermost movable shape is (+ outward momentum). X: (N, F)."""
    dx = X[:, mem["px"]] - CENTER; dy = X[:, mem["py"]] - CENTER
    vx = X[:, mem["vx"]]; vy = X[:, mem["vy"]]
    rad = (dx * vx + dy * vy) / (jnp.sqrt(dx * dx + dy * dy) + 1e-3)
    s = jnp.maximum(jnp.abs(dx), jnp.abs(dy)) + BETA_VEL * jnp.tanh(0.3 * rad)
    s_shape = jnp.max(jnp.where(_w(mem, X) > 0.5, s, -1e6), axis=-1)
    cx = X[:, 0] - CENTER; cy = X[:, 1] - CENTER
    crad = (cx * X[:, 2] + cy * X[:, 3]) / (jnp.sqrt(cx * cx + cy * cy) + 1e-3)
    s_cent = jnp.maximum(jnp.abs(cx), jnp.abs(cy)) + BETA_VEL * jnp.tanh(0.3 * crad)
    return jnp.maximum(s_shape, s_cent)


def _progress(mem, X, d):
    """Signed extent of the movable shapes along each row's escape direction d: (N, 2)."""
    px = X[:, mem["px"]] - CENTER; py = X[:, mem["py"]] - CENTER
    proj = px * d[:, 0:1] + py * d[:, 1:2]
    p_shape = jnp.max(jnp.where(_w(mem, X) > 0.5, proj, -1e6), axis=-1)
    return jnp.maximum(p_shape, (X[:, 0] - CENTER) * d[:, 0] + (X[:, 1] - CENTER) * d[:, 1])


def init(rng, F, n_act, K, B):
    ix = _idx(F); k1, k2 = jax.random.split(rng)
    dirs = jnp.array(DIRS, jnp.float32)
    return {"n_act": jnp.int32(n_act),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "use": jnp.float32(ix["use"]),
            "mob": jnp.zeros(len(ix["px"]), jnp.float32),
            "dirs": dirs, "dir": dirs[jax.random.randint(k1, (B,), 0, len(DIRS))],
            "base": jnp.zeros((B, n_act), jnp.float32),
            "trial": SIGMA * jax.random.normal(k2, (B, n_act), jnp.float32),
            "best": jnp.full((B,), -1e6, jnp.float32), "p0": jnp.zeros(B, jnp.float32),
            "pmax": jnp.full((B,), -1e6, jnp.float32), "stall": jnp.zeros(B, jnp.int32),
            "left": jnp.zeros(B, jnp.int32), "cur": jnp.zeros(B, jnp.int32),
            "sscore": jnp.full(K + 1, -1e6, jnp.float32)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    s = _out_score(mem, arch_feats) - 0.15 * jnp.log1p(arch_visits.astype(jnp.float32))
    logits = jnp.where(arch_valid, TEMP * s, -1e9)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k2, (B, K), jnp.float32), axis=1)
    p = jnp.where(mem["stall"] > STALL_T, P_STALL, P_BASE)
    do = (jax.random.uniform(k1, (B,)) < p) & arch_valid.any()
    return do, slot, {**mem, "stall": jnp.where(do, 0, mem["stall"]),
                      "pmax": jnp.where(do, jnp.float32(-1e6), mem["pmax"])}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    prog = _progress(mem, feats, mem["dir"])
    newb = mem["left"] <= 0
    better = (prog - mem["p0"]) > mem["best"]
    base = jnp.where((newb & better)[:, None], mem["trial"], mem["base"])
    upd = jnp.maximum(jnp.where(better, prog - mem["p0"], mem["best"] * 0.9), 0.0)
    best = jnp.where(newb, upd, mem["best"])
    trial = jnp.where(newb[:, None], base + SIGMA * jax.random.normal(k1, base.shape, jnp.float32), mem["trial"])
    a = jax.random.categorical(k2, trial)
    a = jnp.where(jax.random.uniform(k3, (B,)) < EPS_RAND, jax.random.randint(k4, (B,), 0, mem["n_act"]), a)
    a = jnp.where((jax.random.uniform(k5, (B,)) < STICKY) & ~newb, mem["cur"], a).astype(jnp.int32)
    stall = jnp.where(prog > mem["pmax"] + 1e-3, 0, mem["stall"] + 1)
    pmax = jnp.maximum(mem["pmax"], prog)
    redir = stall > REDIR_T
    d = jnp.where(redir[:, None], mem["dirs"][jax.random.randint(k6, (B,), 0, len(DIRS))], mem["dir"])
    return a, {**mem, "base": base, "trial": trial, "best": best, "cur": a,
               "p0": jnp.where(newb, prog, mem["p0"]), "left": jnp.where(newb, HOLD, mem["left"]) - 1,
               "pmax": jnp.where(redir, prog, pmax), "stall": jnp.where(redir, 0, stall), "dir": d}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    spd = jnp.sqrt(feats[:, mem["vx"]] ** 2 + feats[:, mem["vy"]] ** 2) * feats[:, mem["ai"]]
    mob = jnp.maximum(mem["mob"], spd.max(0) * mem["use"])
    sc = _out_score({**mem, "mob": mob}, feats)
    eff = jnp.where(arch_valid, mem["sscore"][:K], jnp.float32(-1e6))
    tgt = jnp.argsort(eff)[jnp.arange(B) % K]
    div = jax.random.uniform(k1, (B,)) < P_DIVERSE
    slot = jnp.where(div, jax.random.randint(k2, (B,), 0, K), tgt)
    store = (~done) & (div | (sc > eff[tgt] + MARGIN))
    sscore = mem["sscore"].at[jnp.where(store, slot, K)].set(jnp.where(store, sc, mem["sscore"][K]))
    return store, slot, {**mem, "mob": mob, "sscore": sscore}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
