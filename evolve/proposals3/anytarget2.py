"""Standoff-normalised pair targeting ("anytarget2") -- measure every approach in units of the target's own size.

Two measurements drive this program. (a) Control authority is the scarce resource: only two bodies in a Kinetix
level respond to actions, the whole moving assembly and the green shapes, so those stay the two mover channels
rather than being opened up into a bandit dimension. (b) A distance ladder in absolute world units has no
level-invariant success band: on the TUNNEL instances the injected obstacle's inradius spans 0.13 to 0.61, so
"within 0.1 m" is far stricter than needed on every one of them, and a tester tuned to drive centre distance to
zero over-invests on wide obstacles and under-invests on narrow ones.
(1) The target's scale is estimated online from the contact-flag block: whenever a contact pair involving target j
    is active, the mover-channel distance to j at that instant is a sample of j's contact standoff, kept as a
    running minimum (default DEF_STAND until one is seen). No new features are needed -- only the pair lists, which
    are a deterministic function of the shape counts.
(2) The ladder and the reward are then scale-free: rung = floor(d / (BAND_FRAC * standoff[j])), and an attempt is
    scored by closest-approach improvement measured in those same units, so one rung means the same thing on a
    narrow obstacle as on a wide one and the discounted-UCB1 arm statistics become comparable across targets.
(3) The absence of a standoff is itself the TUNNEL signal: a solid shape stops approaching movers at its surface, a
    de-solidified one does not. Targets with zero contacts accumulate `approaches[j]` -- attempts that closed to
    within APPROACH_K * DEF_STAND and still never touched -- and earn a UCB bonus growing as log1p(approaches),
    capped at TUN_CAP so the goal- and hazard-coloured targets that SHORTCUT and DISARMED need are never starved.
(4) Steering is unchanged velocity feedback on the selected channel; holds are interleaved with energy pumping, and
    the archive tail keeps generic coverage / top-speed / goal-proximity pools so ANOMALY and OOB survive.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
NW = 4                # boundary walls: the first num_static_fixated_polys (=4) polygons of every Kinetix config
ATT_L = 64            # steps of one attempt on an arm before it is scored and a new arm is drawn
HOLD = 8              # steps one aiming action is held
PUMP_P = 0.35         # chance that a hold is an energy-pumping hold instead of an aiming hold
UCB_C = 0.7           # UCB1 optimism constant over the (mover channel, target) arms
DISC = 0.93           # per-attempt discount on an arm's statistics (~14-attempt sliding mean, keeps it adaptive)
RCAP = 3.0            # closest-approach improvement, in ladder rungs, that counts as reward 1.0
GSC = 0.5             # gumbel scale on arm selection (keeps the batch spread over arms)
DEF_STAND = 0.35      # assumed contact standoff before one has been observed (median TUNNEL target inradius)
MIN_STAND = 0.06      # floor on the estimate, so the rung unit can never collapse
BAND_FRAC = 0.5       # one ladder rung = this many estimated standoffs
APPROACH_K = 1.5      # an attempt closing inside this many default standoffs with no contact counts as an approach
TUN_W = 0.5           # UCB bonus per log1p(approaches) for a target that has never registered a contact
TUN_CAP = 0.8         # cap on that bonus, so contact-capable targets keep drawing budget
LR = 0.30             # EMA rate of the per-worker action -> mover-acceleration model
LR_OBS = 0.35         # EMA rate of the observed per-action closing rate
W_OBS = 3.0           # weight of the observed closing table relative to the predicted acceleration model
TAU = 0.35            # gumbel temperature on the (per-worker normalised) aiming score
EPS_RAND = 0.10       # uniform-random action rate while aiming
MDECAY = 0.6          # how much of the action models survives a restore / arm switch
P_RESUME = 0.14       # per-step chance a worker jumps back to its arm's closest approach
MARGIN_F = 0.05       # improvement, in rung units, required to overwrite a rung
VIS_W = 0.35          # visit penalty when choosing which rung to resume from
GEN_FRAC = 0.28       # fraction of archive slots kept for the generic coverage / speed / proximity pools
BITS = 12             # hash bits for the generic coverage cells
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)   # feature dims used to define coverage cells
CELL_RES = 0.5        # coarse grid resolution for coverage cells
BIG = 12.0            # "no record" distance
MODE_LO = (1, 15, 6, 41, 2)     # pumping hold lengths: short random / long motor / thruster / extreme motor / oscillate
MODE_HI = (5, 40, 14, 70, 6)


def _idx(F):
    """Per-shape blocks and the contact-flag tail, recovered from the (size-unique) feature width.
    Layout (features.py): 11 scalars, then poly pos/vel/rot/active/role (7*Np), circle pos/vel/active/role (6*Nc),
    then the contact flags for the engine's poly-poly / circle-poly / circle-circle pair lists."""
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41)
            if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    if len(sols) != 1:
        return dict(use=0.0, N=1, px=[0], py=[1], vx=[2], vy=[3], ai=[8], ri=[8], wall=[1.0], ci=[0], inc=[[0.0]])
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


def _layout(K, A):
    """Archive layout: A distance ladders of `nbin` rungs, then coverage / speed / proximity pools."""
    gen = max(3, int(round(GEN_FRAC * K)))
    nbin = max(1, (K - gen) // max(A, 1))
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
            "nbin": jnp.int32(nbin), "TB": jnp.int32(tb), "G0": jnp.int32(tb), "ncov": jnp.int32(ncov),
            "brange": jnp.arange(nbin, dtype=jnp.int32), "pool": jnp.array(pool, jnp.int32),
            "nm": jnp.int32(nm), "noop": jnp.int32(noop), "thr": jnp.array(thr, jnp.int32),
            "lo": jnp.array(MODE_LO, jnp.int32), "hi": jnp.array(MODE_HI, jnp.int32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "ci": jnp.array(ix["ci"], jnp.int32), "inc": jnp.array(ix["inc"], jnp.float32),
            "wall": jnp.array(ix["wall"], jnp.float32),
            # ---- online per-target scale: contact standoff, contact history, contact-free approaches ----
            "stand": jnp.full(N, DEF_STAND, jnp.float32), "ncon": jnp.zeros(N, jnp.float32),
            "appr": jnp.zeros(N, jnp.float32),
            # ---- global bandit over the A = 2N (mover channel, target) arms ----
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


def _channels(mem, X):
    """Position / velocity of the two controllable movers: 0 = whole moving assembly, 1 = the green shapes."""
    green = ((X[:, mem["ai"]] > 0.5) & (jnp.abs(X[:, mem["ri"]] - 1.0) < 0.5)).astype(jnp.float32)
    w = green / jnp.maximum(green.sum(-1, keepdims=True), 1.0)
    gv = jnp.stack([(X[:, mem["vx"]] * w).sum(-1), (X[:, mem["vy"]] * w).sum(-1)], axis=-1)
    return jnp.stack([X[:, 0:2], X[:, 5:7]], 1), jnp.stack([X[:, 2:4], gv], 1)      # (B,2,2) each


def _dists(mem, X):
    """Distance from each mover channel to every shape: (B, 2, N)."""
    p, _ = _channels(mem, X)
    px = X[:, mem["px"]]; py = X[:, mem["py"]]
    return jnp.sqrt((px[:, None, :] - p[:, :, 0:1]) ** 2 + (py[:, None, :] - p[:, :, 1:2]) ** 2 + 1e-12)


def _aim(mem, X, arm):
    """(distance, target xy, mover xy, mover velocity) for each worker's arm."""
    N = mem["px"].shape[0]; ar = jnp.arange(arm.shape[0])
    ch = arm // N; t = jnp.mod(arm, N)
    p, v = _channels(mem, X)
    tx = jnp.take_along_axis(X[:, mem["px"]], t[:, None], 1)[:, 0]
    ty = jnp.take_along_axis(X[:, mem["py"]], t[:, None], 1)[:, 0]
    return _dists(mem, X)[ar, ch, t], tx, ty, p[ar, ch], v[ar, ch]


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; N = mem["px"].shape[0]; A = mem["bn"].shape[0]
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    act_m = feats[:, mem["ai"]].max(0) > 0.5
    role = feats[:, mem["ri"]].max(0)
    green_any = (act_m & (jnp.abs(role - 1.0) < 0.5)).any()
    tok = act_m & (mem["wall"] < 0.5) & (jnp.abs(role - 1.0) > 0.5) & (mem["use"] > 0.5)
    aok = jnp.concatenate([tok, tok & green_any])
    # ---- online scale: a contact involving shape j samples j's standoff at the current channel distance ----
    dch = _dists(mem, feats)                                     # (B,2,N)
    dmin = dch.min(1)                                            # (B,N) nearest mover channel
    touching = (feats[:, mem["ci"]] @ mem["inc"].T) > 0.5         # (B,N)
    stand = jnp.clip(jnp.minimum(mem["stand"], jnp.where(touching, dmin, BIG).min(0)), MIN_STAND, BIG)
    ncon = jnp.maximum(mem["ncon"], touching.max(0).astype(jnp.float32))
    unit = BAND_FRAC * stand                                     # (N,) one ladder rung, in world units
    # ---- score finished attempts by closest-approach improvement measured in rungs ----
    plan = mem["aleft"] <= 0
    tj = jnp.mod(mem["arm"], N)
    r = jnp.clip((mem["brec"][mem["arm"]] - mem["amin"]) / unit[tj], 0.0, RCAP) / RCAP
    hot = jnp.where(plan, 1.0, 0.0)
    # each arm is decayed exactly once per attempt scored on it, so concurrent scatter-adds cannot overshoot
    cnt = jnp.zeros_like(mem["bn"]).at[mem["arm"]].add(hot)
    rsum = jnp.zeros_like(mem["bsum"]).at[mem["arm"]].add(hot * r)
    dsc = DISC ** cnt
    bn = dsc * mem["bn"] + cnt
    bsum = dsc * mem["bsum"] + rsum
    bmu = bsum / jnp.maximum(bn, 1.0)
    brec = mem["brec"].at[mem["arm"]].min(jnp.where(plan, mem["amin"], BIG))
    btot = mem["btot"] + hot.sum()
    # ---- a target that keeps being approached and never touches is the tunnelling candidate ----
    got_close = plan & (mem["amin"] < APPROACH_K * DEF_STAND) & (ncon[tj] < 0.5)
    appr = mem["appr"].at[tj].add(jnp.where(got_close, 1.0, 0.0))
    tun = jnp.minimum(TUN_CAP, TUN_W * jnp.log1p(appr)) * (ncon < 0.5)
    ucb = bmu + jnp.concatenate([tun, tun]) + UCB_C * jnp.sqrt(jnp.log(1.0 + btot) / (1.0 + bn))
    newa = jnp.argmax(jnp.where(aok, ucb, -1e9)[None, :] + GSC * jax.random.gumbel(k1, (B, A)), axis=1).astype(jnp.int32)
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
    # ---- an arm with no ladder yet browses the generic pools instead ----
    gen_ok = arch_valid & (mem["pool"] > 0)
    glog = jnp.where(gen_ok, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot_g = jnp.argmax(glog[None, :] + jax.random.gumbel(k4, (B, K)), axis=1)
    use_g = (~use_l) & plan & gen_ok.any()
    do = use_l | use_g
    slot = jnp.where(use_l, slot_l, slot_g)
    return do, slot, {**mem, "stand": stand, "ncon": ncon, "appr": appr,
                      "bn": bn, "bsum": bsum, "brec": brec, "btot": btot, "aok": aok, "arm": arm,
                      # jittered attempt lengths keep the batch from scoring every arm on the same step
                      "aleft": jnp.where(plan, ATT_L // 2 + jax.random.randint(k5, (B,), 0, ATT_L), mem["aleft"]),
                      "amin": jnp.where(plan, jnp.float32(BIG), mem["amin"]),
                      "left": jnp.where(do | plan, 0, mem["left"]),
                      "M": jnp.where((do | sel)[:, None, None], MDECAY * mem["M"], mem["M"]),
                      "R": jnp.where(sel[:, None], MDECAY * mem["R"], mem["R"]),
                      # a restore *or* an arm switch invalidates the stored velocity / distance the EMAs difference
                      "fresh": mem["fresh"] | do | sel}


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
    B = feats.shape[0]; K = arch_valid.shape[0]; N = mem["px"].shape[0]
    live = ~done
    # ---- pool 0: one ladder per arm, rungs measured in estimated standoffs of that arm's target ----
    d = _aim(mem, feats, mem["arm"])[0]
    unit = BAND_FRAC * mem["stand"][jnp.mod(mem["arm"], N)]
    band = jnp.clip((d / unit).astype(jnp.int32), 0, mem["nbin"] - 1)
    lslot = jnp.clip(mem["arm"] * mem["nbin"] + band, 0, mem["TB"] - 1)
    lad = live & mem["aok"][mem["arm"]] & ((~arch_valid[lslot]) | (d < mem["best"][lslot] - MARGIN_F * unit))
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
