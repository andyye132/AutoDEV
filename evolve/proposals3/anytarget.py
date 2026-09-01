"""Pair-targeting with a closest-approach bandit ("anytarget") -- every non-green shape is something to be reached.

TUNNEL, SHORTCUT and DISARMED are all "get body A onto shape B" bugs, and the two bodies a Kinetix level lets you
drive are the whole moving assembly and the green (agent) shapes; so the search space is the (mover, target) grid,
and the only progress signal that works for *all* of them is how close the pair has ever come. Contact cannot be the
reward: a de-solidified obstacle never registers one, which is exactly why contact-driven testers abandon it.
(1) Arms = (mover channel in {moving assembly, green shapes}) x (active, non-wall, non-green shape). A worker holds
    one arm for ~ATT_L steps, then the arm is scored by *closest-approach improvement* -- how much the attempt
    beat the global record distance for that pair -- and a UCB1 bandit shared by the whole batch picks the next arm. A
    solid obstacle's record bottoms out on its surface and its reward decays to zero; a non-solid one keeps yielding
    improvement all the way through its interior, so it keeps drawing budget.
(2) Steering is velocity feedback on the selected mover: a per-worker EMA of each action's effect on that body's
    velocity, plus a per-worker EMA of the closing rate each action actually achieved, scored against the unit
    direction to the target; the winner is held for HOLD steps, interleaved with energy-pumping holds for reach.
(3) The archive is one distance ladder per arm -- rung (arm, sqrt-warped distance band) keeps the closest state seen
    in that band -- so a new attempt restores to the pair's closest approach rather than to the level reset. The
    archive tail keeps generic coverage / top-speed / goal-proximity pools so ANOMALY and OOB are not given up.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
NW = 4                # boundary walls: the first num_static_fixated_polys (=4) polygons of every Kinetix config
ATT_L = 64            # steps of one attempt on an arm before it is scored and a new arm is drawn
HOLD = 8              # steps one aiming action is held
PUMP_P = 0.35         # chance that a hold is an energy-pumping hold instead of an aiming hold
UCB_C = 0.7           # UCB1 optimism constant over the (mover, target) arms
DISC = 0.93           # per-attempt discount on an arm's statistics (~14-attempt sliding mean, keeps it adaptive)
RCAP = 2.0            # closest-approach improvement (world units) that counts as reward 1.0
GSC = 0.5             # gumbel scale on arm selection (keeps the batch spread over arms)
LR = 0.30             # EMA rate of the per-worker action -> mover-acceleration model
LR_OBS = 0.35         # EMA rate of the observed per-action closing rate
W_OBS = 3.0           # weight of the observed closing table relative to the predicted acceleration model
TAU = 0.35            # gumbel temperature on the (per-worker normalised) aiming score
EPS_RAND = 0.10       # uniform-random action rate while aiming
MDECAY = 0.6          # how much of the action models survives a restore / arm switch
P_RESUME = 0.14       # per-step chance a worker jumps back to its arm's closest approach
DMAX = 7.5            # distance that maps to the top ladder rung (arena diagonal)
BIN_MAX = 16          # cap on rungs per arm
MARGIN = 0.02         # improvement required to overwrite a rung
VIS_W = 0.35          # visit penalty when choosing which rung to resume from
GEN_FRAC = 0.28       # fraction of archive slots kept for the generic coverage / speed / proximity pools
BITS = 12             # hash bits for the generic coverage cells
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)   # feature dims used to define coverage cells
CELL_RES = 0.5        # coarse grid resolution for coverage cells
BIG = 12.0            # "no record" distance
MODE_LO = (1, 15, 6, 41, 2)     # pumping hold lengths: short random / long motor / thruster / extreme motor / oscillate
MODE_HI = (5, 40, 14, 70, 6)


def _idx(F):
    """Per-shape position / velocity / active / role blocks from the (size-unique) feature width.
    Layout (features.py): 11 scalars, then poly pos/vel/rot/active/role (7*Np), circle pos/vel/active/role (6*Nc),
    then the contact-flag tail (unused here -- this program deliberately never rewards contact)."""
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


def _layout(K, A):
    """Archive layout: A distance ladders of `nbin` rungs, then coverage / speed / proximity pools."""
    gen = max(3, int(round(GEN_FRAC * K)))
    nbin = max(1, min(BIN_MAX, (K - gen) // max(A, 1)))
    while A * nbin > K - 3 and nbin > 1:
        nbin -= 1
    tb = min(A * nbin, max(1, K - 3))
    rest = K - tb
    nspd = max(1, int(0.25 * rest)); npx = max(1, int(0.25 * rest)); ncov = max(1, rest - nspd - npx)
    nspd = max(1, rest - ncov - npx)
    return nbin, tb, ncov, [0] * tb + [1] * ncov + [2] * nspd + [3] * (K - tb - ncov - nspd)


def init(rng, F, n_act, K, B):
    ix = _idx(F); N = ix["N"]; A = 2 * N
    nbin, tb, ncov, pool = _layout(K, A)
    assert len(pool) == K
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "Kt": jnp.int32(K), "use": jnp.float32(ix["use"]),
            "N": jnp.int32(N), "nbin": jnp.int32(nbin), "nbinf": jnp.float32(nbin), "TB": jnp.int32(tb),
            "G0": jnp.int32(tb), "ncov": jnp.int32(ncov),
            "brange": jnp.arange(nbin, dtype=jnp.int32), "pool": jnp.array(pool, jnp.int32),
            "nm": jnp.int32(nm), "noop": jnp.int32(noop), "thr": jnp.array(thr, jnp.int32),
            "lo": jnp.array(MODE_LO, jnp.int32), "hi": jnp.array(MODE_HI, jnp.int32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "wall": jnp.array(ix["wall"], jnp.float32),
            # ---- global bandit over the A = 2N (mover, target) arms ----
            "bn": jnp.zeros(A, jnp.float32), "bsum": jnp.zeros(A, jnp.float32),
            "brec": jnp.full(A, BIG, jnp.float32), "btot": jnp.zeros((), jnp.float32),
            "aok": jnp.zeros(A, bool),
            # ---- per-worker state ----
            "arm": ((jnp.arange(B) % max(A, 1)) * jnp.int32(ix["use"] > 0.5)).astype(jnp.int32),
            "aleft": jnp.zeros(B, jnp.int32), "amin": jnp.full(B, BIG, jnp.float32),
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


def _body(mem, X, chan):
    """Position / velocity of the mover selected by `chan`: 0 = whole moving assembly, 1 = the green shapes."""
    green = ((X[:, mem["ai"]] > 0.5) & (jnp.abs(X[:, mem["ri"]] - 1.0) < 0.5)).astype(jnp.float32)
    w = green / jnp.maximum(green.sum(-1, keepdims=True), 1.0)
    gv = jnp.stack([(X[:, mem["vx"]] * w).sum(-1), (X[:, mem["vy"]] * w).sum(-1)], axis=-1)
    z = (chan == 0)[:, None]
    return jnp.where(z, X[:, 0:2], X[:, 5:7]), jnp.where(z, X[:, 2:4], gv)


def _aim(mem, feats, arm):
    """(distance, target x, target y, mover position, mover velocity) for each worker's arm."""
    t = jnp.mod(arm, jnp.maximum(mem["N"], 1))
    tx = jnp.take_along_axis(feats[:, mem["px"]], t[:, None], 1)[:, 0]
    ty = jnp.take_along_axis(feats[:, mem["py"]], t[:, None], 1)[:, 0]
    p, v = _body(mem, feats, arm // jnp.maximum(mem["N"], 1))
    d = jnp.sqrt((p[:, 0] - tx) ** 2 + (p[:, 1] - ty) ** 2 + 1e-12)
    return d, tx, ty, p, v


def _bin(mem, d):
    """Square-root-warped distance band: fine rungs close in, coarse ones far away."""
    u = jnp.sqrt(jnp.clip(d, 0.0, DMAX) / DMAX)
    return jnp.clip((mem["nbinf"] * u).astype(jnp.int32), 0, mem["nbin"] - 1)


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; N = mem["px"].shape[0]; A = mem["bn"].shape[0]
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    # ---- which arms exist on this level: mover channel x (active, non-wall, non-green) target ----
    act_m = feats[:, mem["ai"]].max(0) > 0.5
    role = feats[:, mem["ri"]].max(0)
    green_any = (act_m & (jnp.abs(role - 1.0) < 0.5)).any()
    tok = act_m & (mem["wall"] < 0.5) & (jnp.abs(role - 1.0) > 0.5) & (mem["use"] > 0.5)
    aok = jnp.concatenate([tok, tok & green_any])
    # ---- score the finished attempts by closest-approach improvement, then draw new arms with UCB1 ----
    plan = mem["aleft"] <= 0
    r = jnp.clip(mem["brec"][mem["arm"]] - mem["amin"], 0.0, RCAP) / RCAP
    hot = jnp.where(plan, 1.0, 0.0)
    # Discounted sample mean rather than an EMA: several workers finish an attempt on the same step, and a
    # scatter-added EMA step would be applied once per worker and overshoot.  Here each arm is decayed exactly once
    # per attempt scored *on that arm* (DISC ** events), so bsum/bn is a ~1/(1-DISC)-attempt sliding mean.
    cnt = jnp.zeros_like(mem["bn"]).at[mem["arm"]].add(hot)
    rsum = jnp.zeros_like(mem["bsum"]).at[mem["arm"]].add(hot * r)
    dsc = DISC ** cnt
    bn = dsc * mem["bn"] + cnt
    bsum = dsc * mem["bsum"] + rsum
    bmu = bsum / jnp.maximum(bn, 1.0)
    brec = mem["brec"].at[mem["arm"]].min(jnp.where(plan, mem["amin"], BIG))
    btot = mem["btot"] + hot.sum()
    ucb = bmu + UCB_C * jnp.sqrt(jnp.log(1.0 + btot) / (1.0 + bn))
    ucb = jnp.where(aok, ucb, -1e9)
    newa = jnp.argmax(ucb[None, :] + GSC * jax.random.gumbel(k1, (B, A)), axis=1).astype(jnp.int32)
    sel = plan & aok.any()
    arm = jnp.where(sel, newa, mem["arm"])
    # ---- resume this arm's closest approach ----
    idx = jnp.clip(arm[:, None] * mem["nbin"] + mem["brange"][None, :], 0, mem["TB"] - 1)
    rung_ok = arch_valid[idx] & (mem["best"][idx] < BIG)
    key = mem["brange"][None, :].astype(jnp.float32) + VIS_W * jnp.log1p(arch_visits[idx].astype(jnp.float32)) \
        + 0.3 * jax.random.exponential(k2, (B, mem["brange"].shape[0]))
    pick = jnp.argmin(jnp.where(rung_ok, key, 1e9), axis=1)
    slot_l = jnp.take_along_axis(idx, pick[:, None], 1)[:, 0]
    use_l = (plan | (jax.random.uniform(k3, (B,)) < P_RESUME)) & rung_ok.any(1)
    # ---- otherwise a worker with no ladder yet browses the generic pools ----
    gen_ok = arch_valid & (mem["pool"] > 0)
    glog = jnp.where(gen_ok, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot_g = jnp.argmax(glog[None, :] + jax.random.gumbel(k4, (B, K)), axis=1)
    use_g = (~use_l) & plan & gen_ok.any()
    do = use_l | use_g
    slot = jnp.where(use_l, slot_l, slot_g)
    return do, slot, {**mem, "bn": bn, "bsum": bsum, "brec": brec, "btot": btot, "aok": aok, "arm": arm,
                      # jittered attempt lengths keep the batch from scoring every arm on the same step
                      "aleft": jnp.where(plan, ATT_L // 2 + jax.random.randint(k5, (B,), 0, ATT_L), mem["aleft"]),
                      "amin": jnp.where(plan, jnp.float32(BIG), mem["amin"]),
                      "left": jnp.where(do | plan, 0, mem["left"]),
                      "M": jnp.where((do | sel)[:, None, None], MDECAY * mem["M"], mem["M"]),
                      "R": jnp.where(sel[:, None], MDECAY * mem["R"], mem["R"]), "fresh": mem["fresh"] | do}


def _pump(mem, mode, k1, k2):
    """Pumping hold: short random / long motor hold / thruster hold / extreme motor hold / motor oscillation."""
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
    d, tx, ty, p, v = _aim(mem, feats, mem["arm"])
    # ---- velocity feedback: predicted effect of each action, plus the closing it actually achieved ----
    oh = (jnp.arange(n_act)[None, :] == mem["cur"][:, None]).astype(jnp.float32)
    live = (~mem["fresh"]).astype(jnp.float32)[:, None]
    M = mem["M"] + LR * (live * oh)[:, :, None] * ((v - mem["pv"])[:, None, :] - mem["M"])
    closed = jnp.clip(mem["pd"] - d, -1.0, 1.0) * (mem["pd"] < BIG)
    R = mem["R"] + LR_OBS * live * oh * (closed[:, None] - mem["R"])
    ux = tx - p[:, 0]; uy = ty - p[:, 1]; nn = jnp.sqrt(ux * ux + uy * uy) + 1e-6
    s = (M[:, :, 0] * (ux / nn)[:, None] + M[:, :, 1] * (uy / nn)[:, None]) + W_OBS * R
    s = s / (jnp.abs(s).max(-1, keepdims=True) + 1e-6)
    a_aim = jnp.argmax(s / TAU + jax.random.gumbel(k1, s.shape), axis=1).astype(jnp.int32)
    a_aim = jnp.where(jax.random.uniform(k2, (B,)) < EPS_RAND, jax.random.randint(k3, (B,), 0, mem["n_act"]), a_aim)
    # ---- hold scheduling ----
    new = mem["left"] <= 0
    mode = jnp.where(new, jax.random.randint(k4, (B,), 0, mem["lo"].shape[0]), mem["mode"])
    pa, oa, ob = _pump(mem, mode, k5, k6)
    lo = mem["lo"][mode]; hi = mem["hi"][mode]
    plen = lo + jax.random.randint(k7, (B,), 0, jnp.maximum(hi - lo + 1, 1))
    aim = jnp.where(new, mem["aok"].any() & (jax.random.uniform(k5, (B,)) > PUMP_P), mem["aim"])
    cur = jnp.where(new, jnp.where(aim, a_aim, pa), mem["cur"])
    left = jnp.where(new, jnp.where(aim, jnp.int32(HOLD), plen), mem["left"]) - 1
    osc_a = jnp.where(new, oa, mem["osc_a"]); osc_b = jnp.where(new, ob, mem["osc_b"])
    cur = jnp.where((~aim) & (mode == 4), jnp.where(left % 2 == 0, osc_a, osc_b), cur)
    return cur.astype(jnp.int32), {**mem, "M": M, "R": R, "pv": v, "pd": d, "cur": cur.astype(jnp.int32),
                                   "left": left, "mode": mode, "osc_a": osc_a, "osc_b": osc_b, "aim": aim,
                                   "aleft": mem["aleft"] - 1, "fresh": jnp.zeros(B, bool)}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    live = ~done
    # ---- pool 0: one distance ladder per (mover, target) arm ----
    d, _, _, _, _ = _aim(mem, feats, mem["arm"])
    lslot = jnp.clip(mem["arm"] * mem["nbin"] + _bin(mem, d), 0, mem["TB"] - 1)
    lad = live & mem["aok"][mem["arm"]] & ((~arch_valid[lslot]) | (d < mem["best"][lslot] - MARGIN))
    # ---- pool 1: coverage hash cells ----
    cell = _cell(mem, feats)
    is_new = (mem["cell_slot"][cell] < 0) & live & (~lad)
    cov_slot = mem["G0"] + (mem["nf"] + jnp.cumsum(is_new) - 1) % jnp.maximum(mem["ncov"], 1)
    # ---- pool 2: fastest states (ANOMALY) / pool 3: closest green-blue states ----
    skey = jnp.where(mem["pool"] == 2, jnp.where(arch_valid, arch_feats[:, 4], -1e9), 1e9)
    s_slot = jnp.argmin(skey); use_s = (feats[:, 4] > skey[s_slot]) & live & (~lad) & (~is_new)
    pkey = jnp.where(mem["pool"] == 3, jnp.where(arch_valid, arch_feats[:, 7], 1e9), -1e9)
    p_slot = jnp.argmax(pkey); use_p = (feats[:, 7] < pkey[p_slot]) & live & (~lad) & (~is_new) & (~use_s)
    do = lad | is_new | use_s | use_p
    slot = jnp.where(lad, lslot, jnp.where(is_new, cov_slot, jnp.where(use_s, s_slot, p_slot)))
    best = mem["best"].at[jnp.where(do, slot, mem["Kt"])].set(jnp.where(lad, d, BIG))
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, cov_slot, mem["cell_slot"][0]))
    return do, slot, {**mem, "best": best, "cell_slot": cell_slot, "amin": jnp.minimum(mem["amin"], d),
                      "nf": (mem["nf"] + is_new.sum()) % jnp.maximum(mem["ncov"], 1)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
