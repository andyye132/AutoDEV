"""Per-target distance ladders swept by a macro library ("pairladder").

A macro sweep explores well but its novelty frontier has no notion that a tester's job is to bring shapes *together*,
so here the frontier metric becomes the set of (mover class, target shape) centre distances.
(1) Half the archive is one ladder per coordinate j = (class in {non-green, green}, non-wall target); rung b holds the
    best state whose class-to-target distance falls in the b-th log band, making that half a staircase of ever-closer
    approaches to every shape. The other half stays a plain novelty frontier, supplying the breadth a ladder lacks.
(2) act() replays a deterministic macro library -- sustained motor holds, fast and slow reversals, thruster holds and
    pulse trains, all-motor sweeps, settle, short bursts -- the coherent input programs a human tester tries first.
(3) restore() fires exactly when a macro finishes: with probability LAD_P the worker moves to the coordinate whose
    ladder is *least descended* (damped by a round-robin counter so every shape keeps getting attempts) and restarts
    from one of its occupied rungs, preferring low rungs that have been expanded least; otherwise it restarts from the
    least-expanded novelty node. Closing a gap by one band creates a new rung to launch from next time.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
MACRO_L = 18        # length of a full macro
SHORT_L = 6         # length of the short-burst macros
LAD_P = 0.6         # probability that a macro is launched from a ladder rung (else from the novelty frontier)
RUNG_W = 0.8        # preference for low (close) rungs when choosing where on a ladder to restart
VIS_W = 0.7         # least-expanded preference among rungs / frontier nodes
NOV_EPS = 0.30      # novelty radius (scaled feature units) required to store a frontier state
NOV_DIMS = (0, 1, 2, 3, 4, 5, 6, 9, 10)
NOV_SCALE = (1.0, 1.0, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0)
P_STORE = 0.02      # probability of refreshing a rung with a non-record state
MARGIN = 0.01       # improvement required to overwrite a rung
BAND_W = 0.5        # priority for coordinates whose ladder has not descended far
CNT_W = 1.5         # round-robin damping of that priority
JGS = 1.0           # gumbel scale on the coordinate choice
DMAX = 8.0          # distance cap used to lay out the ladder bands
EPS_RAND = 0.04     # tiny action noise, breaks symmetry between workers on identical checkpoints
MOB_THR = 0.05      # speed above which a shape index counts as a mover (learned online)
NW = 4              # boundary walls: the first 4 polygons


def _idx(F):
    """Per-shape position / velocity / active / role blocks from the (size-unique) feature width."""
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41)
            if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    if len(sols) != 1:
        return dict(use=0.0, N=1, px=[0], py=[1], vx=[2], vy=[3], ai=[8], ri=[8], wall=[1.0])
    Np, Nc = sols[0]
    P = 11; PV = P + 2 * Np; PA = P + 5 * Np; PL = P + 6 * Np
    C = P + 7 * Np; CV = C + 2 * Nc; CA = C + 4 * Nc; CL = CA + Nc
    return dict(use=1.0, N=Np + Nc,
                px=[P + 2 * i for i in range(Np)] + [C + 2 * i for i in range(Nc)],
                py=[P + 2 * i + 1 for i in range(Np)] + [C + 2 * i + 1 for i in range(Nc)],
                vx=[PV + 2 * i for i in range(Np)] + [CV + 2 * i for i in range(Nc)],
                vy=[PV + 2 * i + 1 for i in range(Np)] + [CV + 2 * i + 1 for i in range(Nc)],
                ai=[PA + i for i in range(Np)] + [CA + i for i in range(Nc)],
                ri=[PL + i for i in range(Np)] + [CL + i for i in range(Nc)],
                wall=[1.0 if i < NW else 0.0 for i in range(Np)] + [0.0] * Nc)


def _macros(n_act):
    """Deterministic macro library: (M, MACRO_L) action table + per-macro length."""
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    ms = []
    def add(pat, L): ms.append(([int(pat[i % len(pat)]) for i in range(MACRO_L)], L))
    for m in range(nm):
        add([2 * m], MACRO_L); add([2 * m + 1], MACRO_L)
        add([2 * m, 2 * m, 2 * m + 1, 2 * m + 1], MACRO_L)
        add([2 * m] * 5 + [2 * m + 1] * 5, MACRO_L)
    for t in thr:
        add([t], MACRO_L); add([t, t, noop, noop], MACRO_L)
    for m in range(nm):
        add([thr[m % len(thr)], 2 * m], MACRO_L)
    add([2 * m for m in range(nm)], MACRO_L); add([2 * m + 1 for m in range(nm)], MACRO_L)
    add([noop], MACRO_L)
    for m in range(nm):
        add([2 * m], SHORT_L); add([2 * m + 1], SHORT_L)
    add([thr[0], noop], SHORT_L)
    return [p for p, _ in ms], [L for _, L in ms]


def _known(mem):
    return jnp.where(mem["mob"].max() > MOB_THR, mem["mob"] > MOB_THR, jnp.ones_like(mem["mob"], bool))


def _dsc(mem, X):
    """(R, 2N): closest approach of any non-green / green mover to each target shape, flattened."""
    px = X[:, mem["px"]]; py = X[:, mem["py"]]
    act = X[:, mem["ai"]] > 0.5
    role = X[:, mem["ri"]]
    N = px.shape[1]
    d = jnp.sqrt((px[:, :, None] - px[:, None, :]) ** 2 + (py[:, :, None] - py[:, None, :]) ** 2 + 1e-9)
    d = jnp.where(jnp.eye(N, dtype=bool)[None], jnp.inf, d)
    nowall = mem["wall"] < 0.5
    green = act & (jnp.abs(role - 1.0) < 0.5)
    mover = act & nowall[None, :] & _known(mem)[None, :]
    cls = jnp.stack([mover & ~green, mover & green], axis=1)                    # (R,2,N)
    dd = jnp.where(cls[:, :, :, None], d[:, None, :, :], jnp.inf)
    dd = jnp.where((act & nowall[None, :])[:, None, None, :], dd, jnp.inf)
    return jnp.minimum(jnp.min(dd, axis=2), DMAX).reshape(px.shape[0], 2 * N)


def _ok(mem, X):
    """(2N,) coordinates worth working on: an existing mover class and an active non-wall target."""
    act = (X[:, mem["ai"]] > 0.5).any(0)
    role = X[:, mem["ri"]].max(0)
    green = act & (jnp.abs(role - 1.0) < 0.5)
    mover = act & (mem["wall"] < 0.5) & _known(mem)
    tgt = act & (mem["wall"] < 0.5)
    has = jnp.stack([(mover & ~green).any(), (mover & green).any()])
    return ((has[:, None] & tgt[None, :]).reshape(-1)) | (mem["use"] < 0.5)


def _band(mem, d):
    S = mem["occ"].shape[1]
    return jnp.clip((mem["bpo"] * jnp.log2(1.0 + jnp.maximum(d, 0.0))).astype(jnp.int32), 0, S - 1)


def init(rng, F, n_act, K, B):
    import math
    ix = _idx(F); N = ix["N"]; J = 2 * N
    S = max(2, (K // 2) // J); tab, ln = _macros(n_act); M = len(tab); KL = min(J * S, K - 1)
    return {"n_act": jnp.int32(n_act), "M": jnp.int32(M), "use": jnp.float32(ix["use"]),
            "bpo": jnp.float32((S - 1) / math.log2(1.0 + DMAX)),
            "wall": jnp.array(ix["wall"], jnp.float32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "mob": jnp.zeros(N, jnp.float32), "KL": jnp.int32(KL), "KN": jnp.int32(max(1, K - KL)),
            "nov": jnp.array(NOV_DIMS, jnp.int32), "sc": jnp.array(NOV_SCALE, jnp.float32),
            "next_free": jnp.int32(0),
            "best": jnp.full((J, S), jnp.float32(1e6)), "occ": jnp.zeros((J, S), bool),
            "table": jnp.array(tab, jnp.int32), "mlen": jnp.array(ln, jnp.int32),
            "jcnt": jnp.zeros(J, jnp.float32),
            "j": jnp.arange(B, dtype=jnp.int32) % jnp.int32(J), "mid": jnp.arange(B, dtype=jnp.int32) % jnp.int32(M),
            "ph": jnp.zeros(B, jnp.int32), "at_end": jnp.zeros(B, bool)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; J, S = mem["occ"].shape
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    ok = _ok(mem, feats)
    lowb = jnp.argmin(jnp.where(mem["occ"], jnp.arange(S)[None, :], S + 10), axis=1)
    lowb = jnp.where(mem["occ"].any(1), lowb, S).astype(jnp.float32)   # how far this ladder still has to descend
    jsc = jnp.where(ok, BAND_W * lowb - CNT_W * jnp.log1p(mem["jcnt"]), -1e9)
    jp = jnp.argmax(jsc[None, :] + JGS * jax.random.gumbel(k1, (B, J), jnp.float32), axis=1)
    jn = jnp.where(mem["at_end"], jp, mem["j"])
    jcnt = mem["jcnt"].at[jnp.where(mem["at_end"], jp, 0)].add(jnp.where(mem["at_end"], 1.0, 0.0))
    # ---- pick a rung of this coordinate's ladder: low band, least expanded ----
    idx = jnp.clip(jn[:, None] * S + jnp.arange(S)[None, :], 0, K - 1)
    occ = mem["occ"][jn]
    rsc = jnp.where(occ, -RUNG_W * jnp.arange(S)[None, :]
                    - VIS_W * jnp.log1p(arch_visits[idx].astype(jnp.float32)), -1e9)
    band = jnp.argmax(rsc + jax.random.gumbel(k2, (B, S), jnp.float32), axis=1)
    lad_slot = jnp.clip(jn * S + band, 0, K - 1)
    # ---- or the least-expanded novelty frontier node ----
    fsc = jnp.where(arch_valid & (jnp.arange(K) >= mem["KL"]),
                    -VIS_W * jnp.log1p(arch_visits.astype(jnp.float32)), -1e9)
    nov_slot = jnp.argmax(fsc[None, :] + jax.random.gumbel(k3, (B, K), jnp.float32), axis=1)
    use_lad = (jax.random.uniform(k4, (B,)) < LAD_P) & occ.any(1)
    slot = jnp.where(use_lad, lad_slot, nov_slot)
    do = mem["at_end"] & (use_lad | (fsc.max() > -1e8)) & arch_valid.any()
    return do, slot, {**mem, "j": jn, "jcnt": jcnt, "at_end": jnp.zeros(B, bool)}


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
    B = feats.shape[0]; K = arch_valid.shape[0]; J, S = mem["occ"].shape
    ar = jnp.arange(B)
    spd = jnp.sqrt(feats[:, mem["vx"]] ** 2 + feats[:, mem["vy"]] ** 2) * (feats[:, mem["ai"]] > 0.5)
    mob = jnp.maximum(mem["mob"], spd.max(0) * mem["use"])
    d = _dsc({**mem, "mob": mob}, feats)[ar, mem["j"]]
    b = _band({**mem, "mob": mob}, d)
    lad = (~done) & (mem["use"] > 0.5) & ((d < mem["best"][mem["j"], b] - MARGIN)
                                          | (~mem["occ"][mem["j"], b])
                                          | (jax.random.uniform(rng, (B,)) < P_STORE))
    # novelty frontier: keep a macro endpoint only if it is far from everything already stored
    x = feats[:, mem["nov"]] * mem["sc"]; y = arch_feats[:, mem["nov"]] * mem["sc"]
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    mind = jnp.min(jnp.where(arch_valid[None, :], d2, 1e12), axis=1)
    nov = (~done) & (~lad) & mem["at_end"] & (mind > NOV_EPS ** 2)
    nov_slot = mem["KL"] + (mem["next_free"] + jnp.cumsum(nov) - 1) % mem["KN"]
    slot = jnp.clip(jnp.where(lad, mem["j"] * S + b, nov_slot), 0, K - 1)
    best = mem["best"].at[jnp.where(lad, mem["j"], 0), jnp.where(lad, b, 0)].min(jnp.where(lad, d, jnp.inf))
    occ = mem["occ"].at[jnp.where(lad, mem["j"], 0), jnp.where(lad, b, 0)].max(lad)
    return lad | nov, slot, {**mem, "mob": mob, "best": best, "occ": occ,
                             "next_free": (mem["next_free"] + nov.sum()) % mem["KN"]}


# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
