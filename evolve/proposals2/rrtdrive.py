"""Goal-biased RRT over the arena with a learned action model and pump/aim extensions ("rrtdrive").

Reaching a particular shape that may sit anywhere -- behind an obstacle, on a ledge, across the level -- is a
motion-planning problem, so this program grows a rapidly-exploring tree whose nodes are archived states.
(1) A worker samples a goal: with probability GOAL_BIAS the live position of the least-visited non-wall shape, read
    from the per-shape position blocks so it tracks moving targets, otherwise a random arena point. It then restores
    to the archived node whose driven body is nearest that goal -- RRT's "extend from the nearest node", with a visit
    penalty so the tree keeps branching.
(2) An extension is energy pumping (a resonant motor square wave or a thruster hold, to buy reach and height) followed
    by aiming: each worker keeps an online per-action model of how the body it drives accelerates and picks the action
    whose learnt acceleration best aligns with the goal direction. Two bodies are driven in turn, the whole moving
    assembly and the green shapes, because the green shape is often a passenger rather than the controlled vehicle.
(3) The archive is a coverage roadmap: an occupancy grid over each body's position keeps the fastest state seen in
    every cell, plus a random reservoir tail, so the tree always has well-spread, momentum-carrying nodes to grow from.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
GOAL_BIAS = 0.72    # probability that an extension aims at a shape rather than a random arena point
PUMP_MAX = 26       # longest energy-pumping prefix of an extension
PUMP_MIN = 0        # shortest (0 = pure aiming)
AIM_L = 22          # length of the aiming suffix of an extension
LR = 0.30           # EMA rate of the per-worker action -> body-acceleration model
MDECAY = 0.6        # how much of the model survives a restore
TAU = 0.9           # gumbel temperature on the aiming action choice
EPS_RAND = 0.12     # uniform-random action rate while aiming
P_BASE = 0.01       # per-step chance of an unscheduled re-plan
VIS_W = 0.05        # visit penalty in the nearest-node ranking
NOISE = 0.10        # tie-break noise in the nearest-node ranking
WORLD = 5.0         # arena side
P_REF = 0.02        # chance of refreshing a roadmap cell with a non-record state
P_RES = 0.05        # chance of dropping a state into the random reservoir tail
Q_EPS = 0.05        # speed margin needed to replace a roadmap cell
PERIODS = (2, 3, 4, 6, 9, 14)
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


def _bodies(mem, X):
    """Position and velocity of the two driven bodies: (R,2,2) each -- [0] whole assembly, [1] green shapes."""
    green = ((X[:, mem["ai"]] > 0.5) & (jnp.abs(X[:, mem["ri"]] - 1.0) < 0.5)).astype(jnp.float32)
    w = green / jnp.maximum(green.sum(-1, keepdims=True), 1.0)
    gv = jnp.stack([(X[:, mem["vx"]] * w).sum(-1), (X[:, mem["vy"]] * w).sum(-1)], axis=-1)
    pos = jnp.stack([X[:, 0:2], X[:, 5:7]], axis=1)
    vel = jnp.stack([X[:, 2:4], gv], axis=1)
    return pos, vel


def init(rng, F, n_act, K, B):
    ix = _idx(F); N = ix["N"]
    R = int(max(4, min(10, (max(K, 8) * 0.5 / 2) ** 0.5))); G = 2 * R * R
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    return {"n_act": jnp.int32(n_act), "N": jnp.int32(N), "R": jnp.int32(R), "G": jnp.int32(G),
            "nm": jnp.int32(nm), "noop": jnp.int32(noop), "thr": jnp.array(thr, jnp.int32),
            "per": jnp.array(PERIODS, jnp.int32), "use": jnp.float32(ix["use"]),
            "wall": jnp.array(ix["wall"], jnp.float32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "tcnt": jnp.zeros(N, jnp.float32), "qbest": jnp.full(K, -1e6, jnp.float32),
            "occ": jnp.zeros(K, bool), "res": jnp.int32(0),
            "M": jnp.zeros((B, 2, n_act, 2), jnp.float32), "pv": jnp.zeros((B, 2, 2), jnp.float32),
            "chan": jax.random.randint(k1, (B,), 0, 2), "gsel": jnp.full(B, N, jnp.int32),
            "gx": jax.random.uniform(k2, (B,), minval=0.3, maxval=WORLD - 0.3),
            "gy": jax.random.uniform(k3, (B,), minval=0.3, maxval=WORLD - 0.3),
            "cur": jnp.zeros(B, jnp.int32), "phase": jnp.zeros(B, jnp.int32), "ph": jnp.zeros(B, jnp.int32),
            "mot": 2 * jax.random.randint(k4, (B,), 0, max(1, nm)),
            "prd": jnp.array(PERIODS, jnp.int32)[jax.random.randint(k5, (B,), 0, len(PERIODS))],
            "tha": jnp.array(thr, jnp.int32)[jnp.zeros(B, jnp.int32)], "pmode": jnp.zeros(B, jnp.int32),
            "at_end": jnp.ones(B, bool), "fresh": jnp.ones(B, bool)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; N = mem["tcnt"].shape[0]
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(rng, 9)
    plan = mem["at_end"] | (jax.random.uniform(k1, (B,)) < P_BASE)
    # ---- sample a goal: least-visited non-wall shape, or a random arena point ----
    act = (feats[:, mem["ai"]] > 0.5).any(0)
    okt = act & (mem["wall"] < 0.5)
    tsc = jnp.where(okt, -jnp.log1p(mem["tcnt"]), -1e9)
    pt = jnp.argmax(tsc[None, :] + jax.random.gumbel(k2, (B, N), jnp.float32), axis=1)
    use_shape = (jax.random.uniform(k3, (B,)) < GOAL_BIAS) & okt.any()
    gsel = jnp.where(plan, jnp.where(use_shape, pt, N), mem["gsel"])
    gx = jnp.where(plan, jax.random.uniform(k4, (B,), minval=0.3, maxval=WORLD - 0.3), mem["gx"])
    gy = jnp.where(plan, jax.random.uniform(k5, (B,), minval=0.3, maxval=WORLD - 0.3), mem["gy"])
    chan = jnp.where(plan, jax.random.randint(k6, (B,), 0, 2), mem["chan"])
    tcnt = mem["tcnt"].at[jnp.where(plan & use_shape, pt, 0)].add(jnp.where(plan & use_shape, 1.0, 0.0))
    # ---- extend from the archived node whose driven body is nearest that goal ----
    apos = jnp.stack([arch_feats[:, 0:2], arch_feats[:, 5:7]], axis=1)            # (K,2,2)
    ap = jnp.transpose(apos, (1, 0, 2))[chan]                                     # (B,K,2)
    ci = jnp.minimum(gsel, N - 1)
    agx = jnp.transpose(arch_feats[:, mem["px"]])[ci]; agy = jnp.transpose(arch_feats[:, mem["py"]])[ci]
    tx = jnp.where((gsel < N)[:, None], agx, gx[:, None]); ty = jnp.where((gsel < N)[:, None], agy, gy[:, None])
    dd = jnp.sqrt((ap[:, :, 0] - tx) ** 2 + (ap[:, :, 1] - ty) ** 2 + 1e-9)
    key = dd + VIS_W * jnp.log1p(arch_visits.astype(jnp.float32))[None, :] + NOISE * jax.random.exponential(k7, (B, K))
    slot = jnp.argmin(jnp.where(arch_valid[None, :], key, 1e9), axis=1)
    do = plan & arch_valid.any()
    pl = jax.random.randint(k8, (B,), PUMP_MIN, PUMP_MAX + 1)
    k10, k11, k12 = jax.random.split(k9, 3)
    return do, slot, {**mem, "gsel": gsel, "gx": gx, "gy": gy, "chan": chan, "tcnt": tcnt,
                      "phase": jnp.where(plan, (pl <= 0).astype(jnp.int32), mem["phase"]),
                      "ph": jnp.where(plan, jnp.where(pl > 0, pl, AIM_L), mem["ph"]),
                      "mot": jnp.where(plan, 2 * jax.random.randint(k10, (B,), 0, jnp.maximum(mem["nm"], 1)), mem["mot"]),
                      "prd": jnp.where(plan, mem["per"][jax.random.randint(k11, (B,), 0, mem["per"].shape[0])], mem["prd"]),
                      "tha": jnp.where(plan, mem["thr"][jax.random.randint(k12, (B,), 0, mem["thr"].shape[0])], mem["tha"]),
                      "pmode": jnp.where(plan, jax.random.randint(k8, (B,), 0, 2), mem["pmode"]),
                      "M": jnp.where(do[:, None, None, None], MDECAY * mem["M"], mem["M"]),
                      "at_end": jnp.zeros(B, bool), "fresh": mem["fresh"] | do}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; N = mem["tcnt"].shape[0]; ar = jnp.arange(B)
    k1, k2, k3 = jax.random.split(rng, 3)
    pos, vel = _bodies(mem, feats)
    # ---- online per-action acceleration model of each driven body ----
    dv = vel - mem["pv"]
    oh = (jnp.arange(mem["M"].shape[2]) == mem["cur"][:, None]).astype(jnp.float32)      # (B,n_act)
    tgtM = dv[:, :, None, :]                                                             # (B,2,1,2)
    upd = (~mem["fresh"])[:, None, None, None] * oh[:, None, :, None]
    M = mem["M"] + LR * upd * (tgtM - mem["M"])
    # ---- aim: the action whose learnt acceleration best aligns with the goal direction ----
    p = pos[ar, mem["chan"]]
    sx = feats[:, mem["px"]][ar, jnp.minimum(mem["gsel"], N - 1)]
    sy = feats[:, mem["py"]][ar, jnp.minimum(mem["gsel"], N - 1)]
    gx = jnp.where(mem["gsel"] < N, sx, mem["gx"]); gy = jnp.where(mem["gsel"] < N, sy, mem["gy"])
    ux = gx - p[:, 0]; uy = gy - p[:, 1]; nn = jnp.sqrt(ux * ux + uy * uy) + 1e-6
    Mc = M[ar, mem["chan"]]                                                              # (B,n_act,2)
    score = (Mc[:, :, 0] * (ux / nn)[:, None] + Mc[:, :, 1] * (uy / nn)[:, None])
    a_aim = jnp.argmax(score / TAU + jax.random.gumbel(k1, score.shape, jnp.float32), axis=1)
    a_aim = jnp.where(jax.random.uniform(k2, (B,)) < EPS_RAND, jax.random.randint(k3, (B,), 0, mem["n_act"]), a_aim)
    # ---- pump: resonant motor square wave or thruster hold ----
    a_pump = jnp.where(mem["pmode"] == 0, mem["mot"] + (mem["ph"] // jnp.maximum(mem["prd"], 1)) % 2, mem["tha"])
    a = jnp.where(mem["phase"] == 0, a_pump, a_aim).astype(jnp.int32)
    nph = mem["ph"] - 1
    to_aim = (nph <= 0) & (mem["phase"] == 0)
    done_ext = (nph <= 0) & (mem["phase"] == 1)
    return a, {**mem, "M": M, "pv": vel, "cur": a, "fresh": jnp.zeros(B, bool),
               "phase": jnp.where(to_aim, 1, mem["phase"]),
               "ph": jnp.where(to_aim, AIM_L, nph), "at_end": done_ext}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    k1, k2, k3 = jax.random.split(rng, 3)
    pos, vel = _bodies(mem, feats)
    p = pos[jnp.arange(B), mem["chan"]]
    cw = WORLD / mem["R"].astype(jnp.float32)
    cx = jnp.clip((p[:, 0] / cw).astype(jnp.int32), 0, mem["R"] - 1)
    cy = jnp.clip((p[:, 1] / cw).astype(jnp.int32), 0, mem["R"] - 1)
    g_slot = jnp.clip(mem["chan"] * mem["R"] * mem["R"] + cy * mem["R"] + cx, 0, K - 1)
    q = feats[:, 4]
    fresh_cell = ~mem["occ"][g_slot]
    beats = q > mem["qbest"][g_slot] - Q_EPS
    grid = fresh_cell | beats | (jax.random.uniform(k1, (B,)) < P_REF)
    ntail = jnp.maximum(K - mem["G"], 1)
    r_slot = mem["G"] + (mem["res"] + jnp.arange(B)) % ntail
    use_res = (jax.random.uniform(k2, (B,)) < P_RES) & (K > mem["G"])
    slot = jnp.where(use_res, jnp.clip(r_slot, 0, K - 1), g_slot)
    store = (~done) & (grid | use_res)
    upd = store & ~use_res
    qbest = mem["qbest"].at[jnp.where(upd, g_slot, 0)].max(jnp.where(upd, q, -1e6))
    occ = mem["occ"].at[jnp.where(store, slot, 0)].max(store)
    return store, slot, {**mem, "qbest": qbest, "occ": occ, "res": (mem["res"] + use_res.sum()) % ntail}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
