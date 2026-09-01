"""Systematic (mover, target) visitation with restore-and-approach ("targetcycle").

SHORTCUT, TUNNEL and DISARMED are all "shape A must end up on shape B" bugs, so this program treats testing as a
covering problem over the (moving shape, non-wall target shape) grid instead of a search over action sequences.
(1) The contact-flag tail of `feats` is decoded back into shape pairs -- the engine's pair lists are a deterministic
    function of the shape counts -- giving an online "which pair has ever touched" matrix. Each worker takes the
    least-attempted *untouched* valid pair, mildly favouring goal-coloured or immobile-neutral targets and green
    movers, so effort drains away from pairs that already collide; a de-solidified obstacle can never register a
    contact, so it keeps drawing attempts.
(2) act() is a per-worker bandit over actions whose reward is the best closing achieved during a whole hold (start
    distance minus the minimum distance reached) plus a bonus for touching the pair, which credits ballistic
    launches that a greedy per-step closing rate would reject.
(3) The archive keeps the closest approach ever recorded for each (mover class, target) coordinate plus a hash-cell
    coverage pool; a worker that stops closing jumps back to its coordinate's closest-approach checkpoint and
    re-attacks, and gives up on the pair after a few failed attempts.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
HOLD = 9            # steps one action is held; the bandit is scored on the whole hold
LONG_P = 0.25       # fraction of holds that are extended (energy / long launches)
LONG_M = 3          # multiplier for a long hold
LR = 0.35           # EMA rate of the per-worker action -> hold-progress table
INV_T = 3.0         # inverse temperature on that table
UCB_C = 0.5         # optimism bonus for rarely tried actions
EPS_RAND = 0.15     # uniform-random action rate
TOUCH_BONUS = 1.5   # bandit reward for actually contacting the assigned pair during a hold
STALL_T = 3         # failed holds before jumping back to the pair's best approach
GIVEUP = 3          # failed approach attempts before the worker takes a new pair
P_BASE = 0.02       # baseline per-step restore probability (coverage pool)
BIG = 12.0          # distance cap / "unreachable" value
MOB_THR = 0.05      # speed above which a shape index counts as a mover (learned online)
W_GOAL = 0.8        # pair priority: target carries the goal colour
W_STAT = 0.8        # pair priority: target is an immobile neutral obstacle
W_GREEN = 0.6       # pair priority: mover is a green (agent-side) shape
SOLVED_W = 2.5      # priority penalty for a pair that has already been seen to touch
CNT_W = 1.2         # weight of the round-robin attempt counter in pair selection
GSC = 0.7           # gumbel scale on pair selection
REC_INFL = 0.001    # per-step inflation of the closest-approach records (keeps slots replaceable)
MARGIN = 0.02       # improvement required to overwrite a record slot
ELITE_B = 0.3       # restore-logit bonus for the closest-approach slots
BITS = 11           # hash bits for the coverage pool
CELL_RES = 0.5      # coarse grid resolution of the coverage cells
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)
NW = 4              # boundary walls: the first 4 polygons


def _idx(F):
    """Per-shape blocks and the contact-flag -> shape-pair map, from the (size-unique) feature width."""
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41)
            if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    if len(sols) != 1:
        return dict(use=0.0, N=1, px=[0], py=[1], vx=[2], vy=[3], ai=[8], ri=[8], wall=[1.0],
                    cbase=0, nf=1, fidx=[[-1]])
    Np, Nc = sols[0]
    P = 11; PV = P + 2 * Np; PA = P + 5 * Np; PL = P + 6 * Np
    C = P + 7 * Np; CV = C + 2 * Nc; CA = C + 4 * Nc; CL = CA + Nc
    pp = [(i, j) for i in range(Np) for j in range(Np) if i < j and not (i < NW and j < NW)]
    cp = [(Np + c, r) for c in range(Nc) for r in range(Np)]
    cc = [(Np + i, Np + j) for i in range(Nc) for j in range(Nc) if i < j]
    prs = pp + cp + cc
    N = Np + Nc
    fidx = [[-1] * N for _ in range(N)]
    for k, (i, j) in enumerate(prs):
        fidx[i][j] = k; fidx[j][i] = k
    return dict(use=1.0, N=N, cbase=CL + Nc, nf=len(prs), fidx=fidx,
                px=[P + 2 * i for i in range(Np)] + [C + 2 * i for i in range(Nc)],
                py=[P + 2 * i + 1 for i in range(Np)] + [C + 2 * i + 1 for i in range(Nc)],
                vx=[PV + 2 * i for i in range(Np)] + [CV + 2 * i for i in range(Nc)],
                vy=[PV + 2 * i + 1 for i in range(Np)] + [CV + 2 * i + 1 for i in range(Nc)],
                ai=[PA + i for i in range(Np)] + [CA + i for i in range(Nc)],
                ri=[PL + i for i in range(Np)] + [CL + i for i in range(Nc)],
                wall=[1.0 if i < NW else 0.0 for i in range(Np)] + [0.0] * Nc)


def _geo(mem, X):
    """Centre-to-centre distances (R,N,N) plus mover / target / green masks (R,N) for the rows of X."""
    px = X[:, mem["px"]]; py = X[:, mem["py"]]
    act = X[:, mem["ai"]] > 0.5
    role = X[:, mem["ri"]]
    d = jnp.sqrt((px[:, :, None] - px[:, None, :]) ** 2 + (py[:, :, None] - py[:, None, :]) ** 2 + 1e-9)
    nowall = mem["wall"] < 0.5
    known = jnp.where(mem["mob"].max() > MOB_THR, mem["mob"] > MOB_THR, jnp.ones_like(mem["mob"], bool))
    green = act & (jnp.abs(role - 1.0) < 0.5)
    return d, act & nowall[None, :] & known[None, :], act & nowall[None, :], green


def _dmin(mem, X):
    """(R, 2, N): closest approach of any non-green / green mover to each target shape."""
    d, mover, tgt, green = _geo(mem, X)
    N = d.shape[-1]
    d = jnp.where(jnp.eye(N, dtype=bool)[None], jnp.inf, d)
    cls = jnp.stack([mover & ~green, mover & green], axis=1)
    dd = jnp.where(cls[:, :, :, None], d[:, None, :, :], jnp.inf)
    dd = jnp.where(tgt[:, None, None, :], dd, jnp.inf)
    return jnp.minimum(jnp.min(dd, axis=2), BIG)


def _cell(mem, feats):
    x = jnp.round(feats[:, jnp.array(CELL_DIMS)] / CELL_RES) * CELL_RES
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)


def init(rng, F, n_act, K, B):
    ix = _idx(F); N = ix["N"]; E = min(2 * N, max(2, K // 2)); KC = max(1, K - E)
    return {"n_act": jnp.int32(n_act), "N": jnp.int32(N), "E": jnp.int32(E), "KC": jnp.int32(KC),
            "use": jnp.float32(ix["use"]), "wall": jnp.array(ix["wall"], jnp.float32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "fidx": jnp.array(ix["fidx"], jnp.int32), "cbase": jnp.int32(ix["cbase"]),
            "cf": jnp.array(list(range(ix["cbase"], ix["cbase"] + ix["nf"])), jnp.int32),
            "tflag": jnp.zeros(ix["nf"], jnp.float32),
            "mob": jnp.zeros(N, jnp.float32), "pcnt": jnp.zeros(N * N, jnp.float32),
            "rec": jnp.full(E, BIG, jnp.float32),
            "mov": jnp.zeros(B, jnp.int32), "tgt": jnp.zeros(B, jnp.int32), "cls": jnp.zeros(B, jnp.int32),
            "Q": jnp.zeros((B, n_act), jnp.float32), "Qn": jnp.zeros((B, n_act), jnp.float32),
            "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
            "d0": jnp.full(B, BIG, jnp.float32), "hmin": jnp.full(B, BIG, jnp.float32),
            "hcon": jnp.zeros(B, jnp.float32), "bd": jnp.full(B, BIG, jnp.float32),
            "stall": jnp.zeros(B, jnp.int32), "tries": jnp.full(B, GIVEUP + 1, jnp.int32),
            "fresh": jnp.ones(B, bool), "jump": jnp.zeros(B, bool),
            "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS)),
            "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "next_free": jnp.int32(0)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    e_slot = jnp.clip(mem["cls"] * mem["N"] + mem["tgt"], 0, mem["E"] - 1)
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32))
                       + ELITE_B * (jnp.arange(K) < mem["E"]).astype(jnp.float32), -1e9)
    cov = jnp.argmax(logits[None, :] + jax.random.gumbel(k2, (B, K), jnp.float32), axis=1)
    jump = mem["jump"] & arch_valid[e_slot]
    slot = jnp.where(jump, e_slot, cov)
    do = (jump | (jax.random.uniform(k1, (B,)) < P_BASE)) & arch_valid.any()
    return do, slot, {**mem, "fresh": do, "jump": jnp.zeros(B, bool),
                      "stall": jnp.where(mem["jump"], 0, mem["stall"]),
                      "tries": jnp.where(mem["jump"], mem["tries"] + 1, mem["tries"]),
                      "bd": jnp.where(do, jnp.float32(BIG), mem["bd"]),
                      "left": jnp.where(do, 0, mem["left"])}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; N = mem["N"]; ar = jnp.arange(B)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    d, mover, tgt_ok, green = _geo(mem, feats)
    dp = jnp.minimum(d[ar, mem["mov"], mem["tgt"]], BIG)
    fl = mem["fidx"][mem["mov"], mem["tgt"]]
    con = jnp.where(fl >= 0, feats[:, mem["cf"]][ar, jnp.maximum(fl, 0)], 0.0)
    hmin = jnp.minimum(mem["hmin"], dp); hcon = jnp.maximum(mem["hcon"], con)
    end = mem["left"] <= 0
    # ---- score the finished hold: best closing achieved + contact bonus ----
    r = (mem["d0"] - hmin) + TOUCH_BONUS * hcon
    oh = (jnp.arange(mem["Q"].shape[1]) == mem["cur"][:, None]).astype(jnp.float32)
    upd = end & (~mem["fresh"])
    Q = jnp.where(upd[:, None], mem["Q"] + LR * oh * (r[:, None] - mem["Q"]), mem["Q"])
    Qn = mem["Qn"] + upd[:, None] * oh
    good = r > 1e-3
    stall = jnp.where(upd & ~good, mem["stall"] + 1, jnp.where(upd, 0, mem["stall"]))
    bd = jnp.minimum(mem["bd"], dp)
    switch = mem["tries"] > GIVEUP
    # ---- least-attempted, still-untouched valid (mover, target) pair ----
    mv_b = mover.any(0); tg_b = tgt_ok.any(0); gr_b = green.any(0)
    role_b = feats[:, mem["ri"]].max(0)
    solved = jnp.where(mem["fidx"] >= 0, mem["tflag"][jnp.maximum(mem["fidx"], 0)], 1e3) > 0.5
    prio_t = W_GOAL * (jnp.abs(role_b - 2.0) < 0.5) + W_STAT * ((jnp.abs(role_b) < 0.5) & (mem["mob"] <= MOB_THR))
    prio = (W_GREEN * gr_b[:, None].astype(jnp.float32) + prio_t[None, :]
            - SOLVED_W * solved.astype(jnp.float32) - CNT_W * jnp.log1p(mem["pcnt"].reshape(solved.shape)))
    valid = mv_b[:, None] & tg_b[None, :] & ~jnp.eye(role_b.shape[0], dtype=bool)
    sc = jnp.where(valid, prio, -1e9).reshape(-1)
    pick = jnp.argmax(sc[None, :] + GSC * jax.random.gumbel(k1, (B, sc.shape[0]), jnp.float32), axis=1)
    mov = jnp.where(switch, pick // N, mem["mov"]); tgt = jnp.where(switch, pick % N, mem["tgt"])
    cls = jnp.where(switch, gr_b[mov].astype(jnp.int32), mem["cls"])
    pcnt = mem["pcnt"].at[jnp.where(switch, pick, 0)].add(jnp.where(switch, 1.0, 0.0))
    Q = jnp.where(switch[:, None], 0.0, Q); Qn = jnp.where(switch[:, None], 0.0, Qn)
    stall = jnp.where(switch, 0, stall); tries = jnp.where(switch, 0, mem["tries"])
    bd = jnp.where(switch, jnp.float32(BIG), bd)
    dp = jnp.where(switch, jnp.minimum(d[ar, mov, tgt], BIG), dp)
    # ---- action for the next hold ----
    new = end | switch
    score = INV_T * Q + UCB_C * jax.lax.rsqrt(1.0 + Qn)
    a_g = jnp.argmax(score + jax.random.gumbel(k2, score.shape, jnp.float32), axis=1)
    a_r = jax.random.randint(k3, (B,), 0, mem["n_act"])
    a_new = jnp.where(jax.random.uniform(k4, (B,)) < EPS_RAND, a_r, a_g)
    L = jnp.where(jax.random.uniform(k5, (B,)) < LONG_P, HOLD * LONG_M, HOLD)
    cur = jnp.where(new, a_new, mem["cur"]).astype(jnp.int32)
    left = jnp.where(new, L, mem["left"]) - 1
    return cur, {**mem, "Q": Q, "Qn": Qn, "cur": cur, "left": left, "mov": mov, "tgt": tgt, "cls": cls,
                 "pcnt": pcnt, "bd": bd, "stall": stall, "tries": tries, "fresh": jnp.zeros(B, bool),
                 "d0": jnp.where(new, dp, mem["d0"]), "hmin": jnp.where(new, dp, hmin),
                 "hcon": jnp.where(new, 0.0, hcon), "jump": stall > STALL_T}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]
    spd = jnp.sqrt(feats[:, mem["vx"]] ** 2 + feats[:, mem["vy"]] ** 2) * (feats[:, mem["ai"]] > 0.5)
    mob = jnp.maximum(mem["mob"], spd.max(0) * mem["use"])
    tflag = mem["tflag"] + feats[:, mem["cf"]].sum(0)
    dsc = _dmin({**mem, "mob": mob}, feats).reshape(B, -1)[:, :mem["rec"].shape[0]]
    rec = jnp.minimum(mem["rec"] + REC_INFL, BIG)
    imp = rec[None, :] - dsc
    j = jnp.argmax(imp + 0.35 * jax.random.gumbel(rng, imp.shape, jnp.float32), axis=1)
    gain = imp[jnp.arange(B), j]
    store_e = (gain > MARGIN) & (~done) & (mem["use"] > 0.5)
    cell = _cell(mem, feats)
    is_new = (mem["cell_slot"][cell] < 0) & (~done) & (~store_e)
    cov = mem["E"] + (mem["next_free"] + jnp.cumsum(is_new) - 1) % mem["KC"]
    slot = jnp.where(store_e, j, cov)
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, cov, mem["cell_slot"][0]))
    rec = rec.at[jnp.where(store_e, j, 0)].min(jnp.where(store_e, dsc[jnp.arange(B), j], jnp.inf))
    return store_e | is_new, slot, {**mem, "mob": mob, "tflag": tflag, "rec": rec, "cell_slot": cell_slot,
                                    "next_free": (mem["next_free"] + is_new.sum()) % mem["KC"]}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
