"""Static-obstacle hunter ("obstaclehunter") -- drive a moving shape *into* each immobile obstacle in turn.

TUNNEL is the one planted class that coverage- and contact-driven testers never reach: its witness is a moving
shape's centroid *inside* a de-solidified static obstacle, and a de-solidified obstacle can never register a
contact, so every objective built on contacts, novelty or speed actively steers away from it. Obstacles are
attractors, not repellers, and the defective one is unknown, so the tester must visit each of them in turn.
(1) The candidate set is reconstructed online from `feats`: active, non-wall, neutral-role shapes whose speed and
    displacement have stayed at zero for the whole run -- exactly the set the TUNNEL injector draws from. Each
    hunter worker takes one obstacle for ATT_L steps, chosen round-robin with a penalty for obstacles that have
    already registered a *contact* (they are still solid, hence not the defect) or been entered.
(2) Aiming is velocity feedback: a per-worker EMA of how each action changes the dynamic centroid's velocity,
    plus a per-worker EMA of the closing rate each action actually achieved, scored against the unit direction
    from the *nearest mover* to the obstacle; the best-scoring action is held for HOLD steps, with occasional
    energy-pumping holds so the body has the momentum to cross the arena. Steering and archiving use the same
    nearest-mover distance, because the witness needs one shape's centre inside the obstacle -- the assembly's
    mean position can sit on it while every individual mover is elsewhere.
(3) The archive is a per-obstacle distance ladder: rung (target, floor(distance/BIN_W)) keeps the closest state
    ever recorded in that band, and a worker starting a new attempt (or stalling) restores to its obstacle's
    lowest occupied rung -- so approaches compose instead of restarting from the level reset. A quarter of the
    workers and the archive tail stay on generic coverage / speed / goal-proximity pools, which is what the other
    four defect classes need.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
NW = 4                # boundary walls: the first num_static_fixated_polys (=4) polygons of every Kinetix config
HUNT_FRAC = 0.75      # fraction of workers that hunt obstacles; the rest do generic energy pumping
ATT_L = 72            # steps of one attempt on an obstacle before the worker cycles to the next
HOLD = 8              # steps one aiming action is held
PUMP_P = 0.30         # chance that a hunter's hold is an energy-pumping hold instead of an aiming hold
LR = 0.30             # EMA rate of the per-worker action -> dyn-centroid acceleration model
LR_OBS = 0.35         # EMA rate of the observed per-action closing rate (distance reduction per step)
W_OBS = 3.0           # weight of the observed closing table relative to the predicted acceleration model
TAU = 0.35            # gumbel temperature on the (per-worker normalised) aiming score
EPS_RAND = 0.10       # uniform-random action rate while aiming
MDECAY = 0.6          # how much of the action models survives a restore / target switch
P_RESUME = 0.12       # per-hold chance a hunter jumps back to its obstacle's closest approach
P_GEN_R = 0.05        # per-step restore probability for the generic (pumping) workers
CNT_W = 1.0           # round-robin weight in obstacle selection
TOUCH_PEN = 2.5       # selection penalty for an obstacle that has registered a contact (=> still solid)
ENTER_PEN = 6.0       # selection penalty for an obstacle already entered
ENTER_R = 0.15        # centre distance that counts as "entered"
GSC = 0.6             # gumbel scale on obstacle selection
BIN_W = 0.30          # width of one distance-to-target ladder rung
BIN_MAX = 24          # cap on rungs per obstacle
MARGIN = 0.02         # improvement required to overwrite a rung
VIS_W = 0.35          # visit penalty when choosing which rung to resume from
GEN_FRAC = 0.30       # fraction of archive slots kept for the generic coverage / speed / proximity pools
BITS = 12             # hash bits for the generic coverage cells
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)   # feature dims used to define coverage cells
CELL_RES = 0.5        # coarse grid resolution for coverage cells
STAT_V = 0.05         # speed under which a shape still counts as immobile
STAT_D = 0.05         # displacement under which a shape still counts as immobile
BIG = 12.0            # "no record" distance
MODE_LO = (1, 15, 6, 41, 2)     # generic hold lengths: short random / long motor / thruster / extreme motor / oscillate
MODE_HI = (5, 40, 14, 70, 6)


def _idx(F):
    """Per-shape blocks and the contact-flag tail, recovered from the (size-unique) feature width.
    Layout (features.py): 11 scalars, then poly pos/vel/rot/active/role (7*Np), circle pos/vel/active/role (6*Nc),
    then the contact flags for the engine's poly-poly / circle-poly / circle-circle pair lists."""
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41)
            if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    if len(sols) != 1:
        return dict(use=0.0, N=1, px=[0], py=[1], vx=[2], vy=[3], ai=[8], ri=[8], wall=[1.0],
                    ci=[0], inc=[[0.0]])
    Np, Nc = sols[0]
    P = 11; PV = P + 2 * Np; PA = P + 5 * Np; PL = P + 6 * Np
    C = P + 7 * Np; CV = C + 2 * Nc; CA = C + 4 * Nc; CL = CA + Nc; CB = CL + Nc
    # engine pair lists are a deterministic function of (Np, Nc): wall-wall pairs are dropped, then lexicographic
    pp = [(i, j) for i in range(Np) for j in range(i + 1, Np) if not (i < NW and j < NW)]
    cp = [(a, b) for a in range(Nc) for b in range(Np)]
    cc = [(i, j) for i in range(Nc) for j in range(i + 1, Nc)]
    ncon = len(pp) + len(cp) + len(cc)
    inc = [[0.0] * ncon for _ in range(Np + Nc)]          # (N, ncon) shape-to-contact-pair incidence
    k = 0
    for i, j in pp:
        inc[i][k] = 1.0; inc[j][k] = 1.0; k += 1
    for a, b in cp:
        inc[Np + a][k] = 1.0; inc[b][k] = 1.0; k += 1
    for i, j in cc:
        inc[Np + i][k] = 1.0; inc[Np + j][k] = 1.0; k += 1
    return dict(use=1.0, N=Np + Nc,
                px=[P + 2 * i for i in range(Np)] + [C + 2 * i for i in range(Nc)],
                py=[P + 2 * i + 1 for i in range(Np)] + [C + 2 * i + 1 for i in range(Nc)],
                vx=[PV + 2 * i for i in range(Np)] + [CV + 2 * i for i in range(Nc)],
                vy=[PV + 2 * i + 1 for i in range(Np)] + [CV + 2 * i + 1 for i in range(Nc)],
                ai=[PA + i for i in range(Np)] + [CA + i for i in range(Nc)],
                ri=[PL + i for i in range(Np)] + [CL + i for i in range(Nc)],
                wall=[1.0 if i < NW else 0.0 for i in range(Np)] + [0.0] * Nc,
                ci=list(range(CB, CB + ncon)), inc=inc)


def _layout(K, N):
    """Archive layout: N distance ladders of `nbin` rungs, then coverage / speed / proximity pools."""
    gen = max(3, int(round(GEN_FRAC * K)))
    nbin = max(1, min(BIN_MAX, (K - gen) // max(N, 1)))
    while N * nbin > K - 3 and nbin > 1:
        nbin -= 1
    tb = min(N * nbin, max(1, K - 3))
    rest = K - tb
    nspd = max(1, int(0.25 * rest)); npx = max(1, int(0.25 * rest)); ncov = max(1, rest - nspd - npx)
    nspd = max(1, rest - ncov - npx)
    return nbin, tb, ncov, [0] * tb + [1] * ncov + [2] * nspd + [3] * (K - tb - ncov - nspd)


def init(rng, F, n_act, K, B):
    ix = _idx(F); N = ix["N"]
    nbin, tb, ncov, pool = _layout(K, N)
    assert len(pool) == K
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "Kt": jnp.int32(K), "use": jnp.float32(ix["use"]),
            "nbin": jnp.int32(nbin), "G0": jnp.int32(tb), "ncov": jnp.int32(ncov), "TB": jnp.int32(tb),
            "brange": jnp.arange(nbin, dtype=jnp.int32), "pool": jnp.array(pool, jnp.int32),
            "nm": jnp.int32(nm), "noop": jnp.int32(noop), "thr": jnp.array(thr, jnp.int32),
            "lo": jnp.array(MODE_LO, jnp.int32), "hi": jnp.array(MODE_HI, jnp.int32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "ci": jnp.array(ix["ci"], jnp.int32), "inc": jnp.array(ix["inc"], jnp.float32),
            "wall": jnp.array(ix["wall"], jnp.float32),
            # ---- global per-shape statistics (shared by the batch) ----
            "smax": jnp.zeros(N, jnp.float32), "disp": jnp.zeros(N, jnp.float32),
            "touch": jnp.zeros(N, jnp.float32), "tent": jnp.zeros(N, jnp.float32),
            "tsel": jnp.zeros(N, jnp.float32), "tok": jnp.zeros(N, bool),
            "refx": jnp.zeros(N, jnp.float32), "refy": jnp.zeros(N, jnp.float32), "seen": jnp.zeros((), bool),
            # ---- per-worker state ----
            "hunt": (jnp.arange(B) < max(1, int(HUNT_FRAC * B))) & (ix["use"] > 0.5),
            "tgt": (jnp.arange(B) % N).astype(jnp.int32), "att": jnp.zeros(B, jnp.int32),
            "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "mode": jnp.zeros(B, jnp.int32),
            "osc_a": jnp.zeros(B, jnp.int32), "osc_b": jnp.zeros(B, jnp.int32),
            "aim": jnp.zeros(B, bool), "fresh": jnp.ones(B, bool),
            "M": jnp.zeros((B, n_act, 2), jnp.float32), "R": jnp.zeros((B, n_act), jnp.float32),
            "pv": jnp.zeros((B, 2), jnp.float32), "pd": jnp.full(B, BIG, jnp.float32),
            # ---- archive bookkeeping ----
            "best": jnp.full(K + 1, BIG, jnp.float32), "nf": jnp.int32(0),
            "proj": jax.random.normal(k1, (len(CELL_DIMS), BITS)),
            "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "cdims": jnp.array(CELL_DIMS, jnp.int32)}


def _cell(mem, feats):
    x = jnp.round(feats[:, mem["cdims"]] / CELL_RES) * CELL_RES
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)


def _dist(mem, feats, tgt):
    """The quantity this program both steers on and archives on: the distance from the *nearest mover* to the
    worker's obstacle, since the witness needs one moving shape's centre inside it -- not the assembly's mean
    position, which can sit on the obstacle while every individual mover is elsewhere. Movers are the shapes seen
    to move so far; before any has been identified this degrades to the dynamic centroid. Returns
    (distance, target xy, aim-origin xy); steering pushes the aim origin along (target - origin)."""
    px = feats[:, mem["px"]]; py = feats[:, mem["py"]]
    tx = jnp.take_along_axis(px, tgt[:, None], 1)[:, 0]
    ty = jnp.take_along_axis(py, tgt[:, None], 1)[:, 0]
    dc = jnp.sqrt((feats[:, 0] - tx) ** 2 + (feats[:, 1] - ty) ** 2 + 1e-12)
    mob = (mem["smax"] > STAT_V) | (mem["disp"] > STAT_D)
    ds = jnp.where(mob[None, :], jnp.sqrt((px - tx[:, None]) ** 2 + (py - ty[:, None]) ** 2 + 1e-12), BIG)
    j = jnp.argmin(ds, axis=1)
    dmin = jnp.take_along_axis(ds, j[:, None], 1)[:, 0]
    use = dmin < dc
    ax = jnp.where(use, jnp.take_along_axis(px, j[:, None], 1)[:, 0], feats[:, 0])
    ay = jnp.where(use, jnp.take_along_axis(py, j[:, None], 1)[:, 0], feats[:, 1])
    return jnp.minimum(dc, dmin), tx, ty, ax, ay


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; N = mem["px"].shape[0]
    k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    # ---- online shape statistics: immobility, contacts, activity, role ----
    px = feats[:, mem["px"]]; py = feats[:, mem["py"]]
    spd = jnp.sqrt(feats[:, mem["vx"]] ** 2 + feats[:, mem["vy"]] ** 2 + 1e-12)
    refx = jnp.where(mem["seen"], mem["refx"], px[0]); refy = jnp.where(mem["seen"], mem["refy"], py[0])
    disp = jnp.maximum(mem["disp"], jnp.sqrt((px - refx[None, :]) ** 2 + (py - refy[None, :]) ** 2).max(0))
    smax = jnp.maximum(mem["smax"], spd.max(0))
    touch = jnp.maximum(mem["touch"], (feats[:, mem["ci"]] @ mem["inc"].T).max(0))
    act_m = feats[:, mem["ai"]].max(0) > 0.5
    role = feats[:, mem["ri"]].max(0)
    free = act_m & (mem["wall"] < 0.5)
    # tier 1: the TUNNEL candidate set -- immobile, neutral, non-wall.  Fall back if a level has none.
    t1 = free & (jnp.abs(role) < 0.5) & (smax < STAT_V) & (disp < STAT_D)
    t2 = free & (jnp.abs(role) < 0.5)
    t3 = free & (jnp.abs(role - 1.0) > 0.5)
    ok = jnp.where(t1.any(), t1, jnp.where(t2.any(), t2, t3))
    # ---- cycle to the next obstacle when the attempt budget runs out ----
    plan = mem["att"] <= 0
    sc = -CNT_W * jnp.log1p(mem["tsel"]) - TOUCH_PEN * touch - ENTER_PEN * mem["tent"]
    sc = jnp.where(ok, sc, -1e9)
    newt = jnp.argmax(sc[None, :] + GSC * jax.random.gumbel(k1, (B, N)), axis=1).astype(jnp.int32)
    sel = plan & mem["hunt"] & ok.any()
    tgt = jnp.where(sel, newt, mem["tgt"])
    tsel = mem["tsel"].at[jnp.where(sel, newt, 0)].add(jnp.where(sel, 1.0, 0.0))
    d = _dist({**mem, "smax": smax, "disp": disp}, feats, tgt)[0]
    tent = mem["tent"].at[tgt].max(jnp.where(mem["hunt"] & (d < ENTER_R), 1.0, 0.0))
    # ---- resume the closest approach recorded for this obstacle ----
    idx = jnp.clip(tgt[:, None] * mem["nbin"] + mem["brange"][None, :], 0, mem["TB"] - 1)
    rung_ok = arch_valid[idx] & (mem["best"][idx] < BIG)
    key = mem["brange"][None, :].astype(jnp.float32) + VIS_W * jnp.log1p(arch_visits[idx].astype(jnp.float32)) \
        + 0.3 * jax.random.exponential(k2, (B, mem["brange"].shape[0]))
    pick = jnp.argmin(jnp.where(rung_ok, key, 1e9), axis=1)
    slot_l = jnp.take_along_axis(idx, pick[:, None], 1)[:, 0]
    use_l = mem["hunt"] & (plan | (jax.random.uniform(k3, (B,)) < P_RESUME)) & rung_ok.any(1)
    # ---- generic workers browse the coverage / speed / proximity pools ----
    gen_ok = arch_valid & (mem["pool"] > 0)
    glog = jnp.where(gen_ok, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot_g = jnp.argmax(glog[None, :] + jax.random.gumbel(k4, (B, K)), axis=1)
    use_g = (~use_l) & (~mem["hunt"]) & (jax.random.uniform(k5, (B,)) < P_GEN_R) & gen_ok.any()
    do = use_l | use_g
    slot = jnp.where(use_l, slot_l, slot_g)
    return do, slot, {**mem, "smax": smax, "disp": disp, "touch": touch, "tent": tent, "tsel": tsel,
                      "tok": ok, "refx": refx, "refy": refy, "seen": jnp.ones((), bool),
                      "tgt": tgt,
                      # jittered attempt lengths keep the batch from cycling target in lockstep after step 0
                      "att": jnp.where(plan, ATT_L // 2 + jax.random.randint(k6, (B,), 0, ATT_L), mem["att"]),
                      "left": jnp.where(do | plan, 0, mem["left"]),
                      "M": jnp.where((do | sel)[:, None, None], MDECAY * mem["M"], mem["M"]),
                      "R": jnp.where(sel[:, None], MDECAY * mem["R"], mem["R"]), "fresh": mem["fresh"] | do}


def _pump(mem, mode, k1, k2):
    """Generic hold: short random / long motor hold / thruster hold / extreme motor hold / motor oscillation."""
    B = mode.shape[0]
    rand_a = jax.random.randint(k1, (B,), 0, mem["n_act"])
    motor = jax.random.randint(k1, (B,), 0, 2 * mem["nm"])
    thr_a = mem["thr"][jax.random.randint(k2, (B,), 0, mem["thr"].shape[0])]
    osc_m = jax.random.randint(k2, (B,), 0, mem["nm"])
    a = jnp.where(mode == 0, rand_a, jnp.where(mode == 2, thr_a, motor))
    return a.astype(jnp.int32), (2 * osc_m).astype(jnp.int32), (2 * osc_m + 1).astype(jnp.int32)


def act(mem, feats, obs, rng):
    B = feats.shape[0]; n_act = mem["M"].shape[1]
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 7)
    d, tx, ty, ax, ay = _dist(mem, feats, mem["tgt"])
    v = feats[:, 2:4]
    # ---- velocity feedback: what each action does to the dynamic centroid, and what it actually closed ----
    oh = (jnp.arange(n_act)[None, :] == mem["cur"][:, None]).astype(jnp.float32)
    live = (~mem["fresh"]).astype(jnp.float32)[:, None]
    M = mem["M"] + LR * (live * oh)[:, :, None] * ((v - mem["pv"])[:, None, :] - mem["M"])
    closed = jnp.clip(mem["pd"] - d, -1.0, 1.0) * (mem["pd"] < BIG)
    R = mem["R"] + LR_OBS * live * oh * (closed[:, None] - mem["R"])
    ux = tx - ax; uy = ty - ay; nn = jnp.sqrt(ux * ux + uy * uy) + 1e-6
    s = (M[:, :, 0] * (ux / nn)[:, None] + M[:, :, 1] * (uy / nn)[:, None]) + W_OBS * R
    s = s / (jnp.abs(s).max(-1, keepdims=True) + 1e-6)
    a_aim = jnp.argmax(s / TAU + jax.random.gumbel(k1, s.shape), axis=1).astype(jnp.int32)
    a_aim = jnp.where(jax.random.uniform(k2, (B,)) < EPS_RAND, jax.random.randint(k3, (B,), 0, mem["n_act"]), a_aim)
    # ---- hold scheduling: aim holds for hunters, generic macro holds otherwise ----
    new = mem["left"] <= 0
    mode = jnp.where(new, jax.random.randint(k4, (B,), 0, mem["lo"].shape[0]), mem["mode"])
    pa, oa, ob = _pump(mem, mode, k5, k6)
    lo = mem["lo"][mode]; hi = mem["hi"][mode]
    plen = lo + jax.random.randint(k7, (B,), 0, jnp.maximum(hi - lo + 1, 1))
    aim = jnp.where(new, mem["hunt"] & (jax.random.uniform(k5, (B,)) > PUMP_P), mem["aim"])
    cur = jnp.where(new, jnp.where(aim, a_aim, pa), mem["cur"])
    left = jnp.where(new, jnp.where(aim, jnp.int32(HOLD), plen), mem["left"]) - 1
    osc_a = jnp.where(new, oa, mem["osc_a"]); osc_b = jnp.where(new, ob, mem["osc_b"])
    cur = jnp.where((~aim) & (mode == 4), jnp.where(left % 2 == 0, osc_a, osc_b), cur)
    return cur.astype(jnp.int32), {**mem, "M": M, "R": R, "pv": v, "pd": d, "cur": cur.astype(jnp.int32),
                                   "left": left, "mode": mode, "osc_a": osc_a, "osc_b": osc_b, "aim": aim,
                                   "att": mem["att"] - 1, "fresh": jnp.zeros(B, bool)}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    live = ~done
    # ---- pool 0: one distance ladder per obstacle; rung = coarse distance band ----
    d = _dist(mem, feats, mem["tgt"])[0]
    b = jnp.clip((d / BIN_W).astype(jnp.int32), 0, mem["nbin"] - 1)
    lslot = jnp.clip(mem["tgt"] * mem["nbin"] + b, 0, mem["TB"] - 1)
    lad = mem["hunt"] & live & mem["tok"][mem["tgt"]] & ((~arch_valid[lslot]) | (d < mem["best"][lslot] - MARGIN))
    # ---- pool 1: coverage hash cells ----
    cell = _cell(mem, feats)
    is_new = (mem["cell_slot"][cell] < 0) & live & (~lad)
    cov_slot = mem["G0"] + (mem["nf"] + jnp.cumsum(is_new) - 1) % jnp.maximum(mem["ncov"], 1)
    # ---- pool 2: fastest states (ANOMALY) / pool 3: closest green-blue states (SHORTCUT) ----
    skey = jnp.where(mem["pool"] == 2, jnp.where(arch_valid, arch_feats[:, 4], -1e9), 1e9)
    s_slot = jnp.argmin(skey); use_s = (feats[:, 4] > skey[s_slot]) & live & (~lad) & (~is_new)
    pkey = jnp.where(mem["pool"] == 3, jnp.where(arch_valid, arch_feats[:, 7], 1e9), -1e9)
    p_slot = jnp.argmax(pkey); use_p = (feats[:, 7] < pkey[p_slot]) & live & (~lad) & (~is_new) & (~use_s)
    do = lad | is_new | use_s | use_p
    slot = jnp.where(lad, lslot, jnp.where(is_new, cov_slot, jnp.where(use_s, s_slot, p_slot)))
    best = mem["best"].at[jnp.where(do, slot, mem["Kt"])].set(jnp.where(lad, d, BIG))
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, cov_slot, mem["cell_slot"][0]))
    return do, slot, {**mem, "best": best, "cell_slot": cell_slot,
                      "nf": (mem["nf"] + is_new.sum()) % jnp.maximum(mem["ncov"], 1)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
