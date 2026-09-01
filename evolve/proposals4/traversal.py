"""Traversal objective ("traversal") -- score an attempt by how FAST a mover crosses a target, never by how close it parks.

Five measurements shape this program.
(a) The TUNNEL witness is one *individual* dynamic body's centroid inside the de-solidified obstacle, not the assembly
    centroid, so every geometric quantity here is per-body: the nearest active dynamic body to a target supplies the
    distance, the speed and the crossing sign. Bodies are split static/dynamic online (a body that is ever seen with
    non-zero speed, or displaced from its reset position, is dynamic), which also gives the "static non-wall" target set.
(b) The one TUNNEL instance that separates targeted testers from random ones is a thin slab that has to be swept
    *through* at speed; a closest-approach objective parks beside it. The per-attempt reward is therefore
    (max speed while inside a SHELL_K*R shell of the target) + (sign changes of the mover->target vector along the
    attempt's approach axis), with closest approach demoted to a small tie-break that only orders zero-reward arms.
(c) Target inradius spans 0.13-0.61 m across the instances, so nothing is in absolute units: the shell is measured in
    an online standoff estimate R (contact-flag samples, running min, DEF_STAND until one is seen) and speed is
    normalised by the running level maximum, so rungs and arm statistics are comparable across levels and targets.
(d) The held-out class needs sustained outward motion at speed, so the three removable arena faces are targets too:
    distance to a face is 1-D along its normal and its target point sits on the boundary, so a "pass" through a face
    target *is* the OOB witness and the same reward drives it.
(e) Only the assembly and the green shapes are actuated, so the bandit is over targets alone (never over passenger
    bodies) and the steered channel is simply whichever of the two is the right one for the nearest body.
Archive rungs are keyed by (target, side of the approach axis, speed band) and hold the closest state seen in each, so
a restore resumes a *fast* approach rather than the parked closest approach. GEN_W of the workers never target
anything: they run the energy-pumping mode mix into coverage / top-speed / outward pools for the other classes.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
NW = 4                # boundary walls: the first num_static_fixated_polys (=4) polygons of every Kinetix config
WORLD = 5.0           # Kinetix arena side, in metres
FACES = ((0, 0.0, -1.0), (0, 5.0, 1.0), (1, 5.0, 1.0))   # (axis, boundary value, outward sign): left, right, ceiling
FACE_R = 0.70         # shell unit of a face target (a face has no contact standoff to estimate)
GEN_W = 0.28          # fraction of workers that never target: pure energy pumping for the other defect classes
ATT_L = 56            # steps in one attempt on a target before it is scored and a new target is drawn
UCB_C = 0.75          # UCB1 optimism over the A target arms
DISC = 0.92           # per-attempt discount on an arm's statistics (~12-attempt sliding mean)
GSC = 0.35            # gumbel scale on arm selection (keeps the batch spread over arms)
SPD_W = 0.45          # weight of the in-shell speed term in the attempt reward
PASS_W = 1.00         # weight of the crossing (sign-change) term
PASS_CAP = 2.0        # crossings in one attempt that already count as full crossing reward
TIE_W = 0.15          # tie-break on closest approach, in shells: orders arms that never crossed anything
SHELL_K = 2.0         # the scoring shell is this many R around the target
STORE_K = 5.0         # states further than this many R from the target are not worth a rung
FREEZE_K = 1.6        # inside this many R the aim is frozen: never brake in the middle of a sweep
PASS_K = 4.0          # sign changes only count this near the target
SGN_K = 0.25          # dead zone on the approach-axis coordinate, in R, before a sign change counts
VREF = 1.5            # floor on the running level max speed used to normalise speeds
DEF_STAND = 0.35      # assumed contact standoff before one is observed (median TUNNEL target inradius)
MIN_STAND = 0.06      # floor on the estimate, so the shell can never collapse
MAX_STAND = 1.10      # ceiling, so one bad sample cannot make the shell the whole arena
VEPS = 1e-3           # speed above which a body is evidence-dynamic
PEPS = 0.02           # displacement from the reset position that also marks a body dynamic
TUN_W = 0.5           # UCB bonus per log1p(approaches) for a target that has never registered a contact
TUN_CAP = 0.8         # cap on that bonus, so contact-capable targets keep drawing budget
APPROACH_K = 2.0      # an attempt closing inside this many R with no contact counts as an approach
NS = 4                # speed bands per (target, side) rung
VBANDS = (0.20, 0.45, 0.75)   # band edges as fractions of the running level max speed (level-invariant)
LR = 0.30             # EMA rate of the per-worker action -> channel-acceleration model
LR_OBS = 0.35         # EMA rate of the observed per-action closing rate
W_OBS = 3.0           # weight of the observed closing table relative to the predicted acceleration model
TAU = 0.35            # gumbel temperature on the (per-worker normalised) aiming score
EPS_RAND = 0.10       # uniform-random action rate while aiming
MDECAY = 0.6          # how much of the action models survives a restore / arm switch
HOLD = 6              # steps one aiming action is held
PUMP_P = 0.30         # chance a targeted worker's hold is a pumping hold instead of an aiming hold
P_RESUME = 0.12       # per-step chance a targeted worker jumps back to one of its arm's rungs
P_GEN_R = 0.06        # per-step chance a worker browses the generic pools instead
P_PLAN_G = 0.35       # chance a worker with no rung for its new arm is teleported to a generic state
P_PLAN_L = 0.55       # chance a finished attempt resumes a rung at all (the rest keep the momentum they have)
GEN_PREF = 0.70       # restore preference for the top-speed / outward pools over plain coverage cells
BAND_W = 1.00         # preference for the fastest rung when resuming (per band)
SIDE_W = 0.30         # extra preference for resuming on the approach side of the axis
VIS_W = 0.35          # visit penalty when choosing which rung to resume from
MARGIN_F = 0.05       # improvement, in R, required to overwrite a rung
GEN_FRAC = 0.30       # fraction of archive slots kept for the generic coverage / speed / outward pools
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
    pp = [(i, j) for i in range(Np) for j in range(i + 1, Np) if not (i < NW and j < NW)]
    cp = [(a, b) for a in range(Nc) for b in range(Np)]
    cc = [(i, j) for i in range(Nc) for j in range(i + 1, Nc)]
    ncon = len(pp) + len(cp) + len(cc)
    # (N, ncon) shape-to-contact-pair incidence, counting ONLY contacts whose partner is not a boundary wall: the four
    # walls carry collision_mode 2 ("collides with everything"), so they keep touching a de-solidified obstacle and
    # would otherwise wipe out the one signal that identifies it -- a shape movers approach and never touch.
    inc = [[0.0] * ncon for _ in range(Np + Nc)]
    def _pair(u, v, k):
        if v >= Np or v >= NW:
            inc[u][k] = 1.0
        if u >= Np or u >= NW:
            inc[v][k] = 1.0
    k = 0
    for i, j in pp:
        _pair(i, j, k); k += 1
    for a, b in cp:
        _pair(Np + a, b, k); k += 1
    for i, j in cc:
        _pair(Np + i, Np + j, k); k += 1
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
    """Archive layout: A * 2 * ns rungs (target x side x speed band), then coverage / top-speed / outward pools."""
    gen = max(6, int(round(GEN_FRAC * K)))
    ns = NS
    while A * 2 * ns > max(1, K - gen) and ns > 1:
        ns -= 1
    TB = min(A * 2 * ns, max(1, K - 6))
    rest = K - TB
    q = max(1, rest // 4)
    ncov = max(1, rest - 2 * q); nspd = max(1, (rest - ncov) // 2); nout = rest - ncov - nspd
    return ns, TB, ncov, [0] * TB + [1] * ncov + [2] * nspd + [3] * nout


def init(rng, F, n_act, K, B):
    ix = _idx(F); N = ix["N"]; NF = len(FACES); A = N + NF
    ns, TB, ncov, pool = _layout(K, A)
    assert len(pool) == K
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    ngen = max(1, int(round(GEN_W * B)))
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "Kt": jnp.int32(K), "use": jnp.float32(ix["use"]),
            "TB": jnp.int32(TB), "G0": jnp.int32(TB), "ncov": jnp.int32(ncov), "pool": jnp.array(pool, jnp.int32),
            "nsr": jnp.int32(ns), "A2": jnp.int32(2 * ns), "brange": jnp.arange(2 * ns, dtype=jnp.int32),
            "bband": jnp.array([i % ns for i in range(2 * ns)], jnp.float32),
            "bside": jnp.array([i // ns for i in range(2 * ns)], jnp.float32),
            "vband": jnp.array(VBANDS, jnp.float32),
            "nm": jnp.int32(nm), "noop": jnp.int32(noop), "thr": jnp.array(thr, jnp.int32),
            "lo": jnp.array(MODE_LO, jnp.int32), "hi": jnp.array(MODE_HI, jnp.int32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "ci": jnp.array(ix["ci"], jnp.int32), "inc": jnp.array(ix["inc"], jnp.float32),
            "wall": jnp.array(ix["wall"], jnp.float32),
            "fax": jnp.array([f[0] for f in FACES], jnp.float32), "fval": jnp.array([f[1] for f in FACES], jnp.float32),
            # ---- online level model: which bodies move, target standoffs, contact history, top speed ----
            "moved": jnp.zeros(N, jnp.float32), "p0": jnp.zeros((N, 2), jnp.float32), "p0set": jnp.float32(0.0),
            "stand": jnp.full(N, DEF_STAND, jnp.float32), "ncon": jnp.zeros(N, jnp.float32),
            "appr": jnp.zeros(A, jnp.float32), "vglob": jnp.float32(VREF),
            # ---- discounted-UCB bandit over the A targets ----
            "bn": jnp.zeros(A, jnp.float32), "bsum": jnp.zeros(A, jnp.float32), "btot": jnp.zeros((), jnp.float32),
            "aok": jnp.zeros(A, bool),
            # ---- per-worker state ----
            "gen": jnp.arange(B) < ngen,
            "arm": ((jnp.arange(B) % max(A, 1)) * jnp.int32(ix["use"] > 0.5)).astype(jnp.int32),
            "ch": jnp.zeros(B, jnp.int32), "aleft": jnp.zeros(B, jnp.int32),
            "u0": jnp.zeros((B, 2), jnp.float32).at[:, 0].set(1.0), "psign": jnp.ones(B, jnp.float32),
            "vmax": jnp.zeros(B, jnp.float32), "passes": jnp.zeros(B, jnp.float32), "amin": jnp.full(B, BIG, jnp.float32),
            "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "mode": jnp.zeros(B, jnp.int32),
            "osc_a": jnp.zeros(B, jnp.int32), "osc_b": jnp.zeros(B, jnp.int32),
            "aim": jnp.zeros(B, bool), "fresh": jnp.ones(B, bool), "reax": jnp.ones(B, bool),
            "bi": jnp.full(B, -1, jnp.int32),
            "M": jnp.zeros((B, n_act, 2), jnp.float32), "R": jnp.zeros((B, n_act), jnp.float32),
            "pv": jnp.zeros((B, 2), jnp.float32), "pd": jnp.full(B, BIG, jnp.float32),
            # ---- archive bookkeeping ----
            "best": jnp.full(K + 1, BIG, jnp.float32), "nf": jnp.int32(0),
            "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32),
            "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS)), "cdims": jnp.array(CELL_DIMS, jnp.int32)}


def _cell(mem, feats):
    x = jnp.round(feats[:, mem["cdims"]] / CELL_RES) * CELL_RES
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)


def _dynm(mem):
    """Estimated dynamic bodies; before any motion has been seen, every non-wall shape counts."""
    return jnp.where(mem["moved"].sum() > 0.5, mem["moved"], 1.0 - mem["wall"])


def _outk(mem, X):
    """Chebyshev distance of the most outlying active dynamic body from the arena centre (the OOB direction)."""
    px = X[:, mem["px"]]; py = X[:, mem["py"]]
    ok = (X[:, mem["ai"]] > 0.5) & (_dynm(mem)[None, :] > 0.5)
    d = jnp.maximum(jnp.abs(px - 0.5 * WORLD), jnp.abs(py - 0.5 * WORLD))
    return jnp.where(ok, d, -1.0).max(-1)


def _geo(mem, X):
    """Per-target geometry against the nearest active dynamic body.
    Returns d (B,A), bi (B,A), bp (B,A,2), bv (B,A,2), bs (B,A), tp (B,A,2): distance, body index, body position,
    body velocity, body speed, target point. Shape targets use the shape centroid; face targets use a 1-D distance along the face normal and a
    target point that slides along the face, so the approach axis is the outward normal."""
    N = mem["px"].shape[0]; NFc = mem["fax"].shape[0]
    p = jnp.stack([X[:, mem["px"]], X[:, mem["py"]]], -1)                       # (B,N,2)
    v = jnp.stack([X[:, mem["vx"]], X[:, mem["vy"]]], -1)
    sp = jnp.sqrt((v * v).sum(-1) + 1e-12)                                      # (B,N)
    ok = (X[:, mem["ai"]] > 0.5) & (_dynm(mem)[None, :] > 0.5)                  # (B,N)
    tx = X[:, mem["px"]]; ty = X[:, mem["py"]]                                  # shape target points
    ds = jnp.sqrt((p[:, None, :, 0] - tx[:, :, None]) ** 2 + (p[:, None, :, 1] - ty[:, :, None]) ** 2 + 1e-12)
    coord = jnp.where(mem["fax"][None, :, None] < 0.5, p[:, None, :, 0], p[:, None, :, 1])   # (B,NF,N)
    df = jnp.abs(coord - mem["fval"][None, :, None])
    d = jnp.where(ok[:, None, :], jnp.concatenate([ds, df], 1), BIG)            # (B,A,N)
    bi = jnp.argmin(d, -1)
    oh = (jnp.arange(N)[None, None, :] == bi[:, :, None]).astype(jnp.float32)   # (B,A,N)
    dm = (oh * d).sum(-1)
    bp = jnp.einsum("ban,bnd->bad", oh, p)
    bv = jnp.einsum("ban,bnd->bad", oh, v)
    bs = (oh * sp[:, None, :]).sum(-1)
    bpf = bp[:, N:, :]
    tpf = jnp.stack([jnp.where(mem["fax"][None, :] < 0.5, mem["fval"][None, :], bpf[..., 0]),
                     jnp.where(mem["fax"][None, :] > 0.5, mem["fval"][None, :], bpf[..., 1])], -1)
    tp = jnp.concatenate([jnp.stack([tx, ty], -1), tpf], 1)                     # (B,A,2)
    return dm, bi, bp, bv, bs, tp


def _pick(X, arm):
    """Gather the selected arm's column out of a (B,A) or (B,A,d) array."""
    if X.ndim == 2:
        return jnp.take_along_axis(X, arm[:, None], 1)[:, 0]
    return jnp.take_along_axis(X, arm[:, None, None], 1)[:, 0]


def _rad(mem):
    """Shell unit R per target: the online contact standoff for shapes, a constant for faces."""
    return jnp.concatenate([mem["stand"], jnp.full(mem["fax"].shape[0], FACE_R, jnp.float32)])


def _chan(mem, X):
    """The two controllable channels: 0 = whole moving assembly, 1 = the green shapes."""
    green = ((X[:, mem["ai"]] > 0.5) & (jnp.abs(X[:, mem["ri"]] - 1.0) < 0.5)).astype(jnp.float32)
    w = green / jnp.maximum(green.sum(-1, keepdims=True), 1.0)
    gv = jnp.stack([(X[:, mem["vx"]] * w).sum(-1), (X[:, mem["vy"]] * w).sum(-1)], -1)
    return jnp.stack([X[:, 2:4], gv], 1)                                        # (B,2,2) velocities


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; N = mem["px"].shape[0]; A = mem["bn"].shape[0]
    k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    # ---- online level model ----
    p = jnp.stack([feats[:, mem["px"]], feats[:, mem["py"]]], -1)
    v = jnp.stack([feats[:, mem["vx"]], feats[:, mem["vy"]]], -1)
    spd = jnp.sqrt((v * v).sum(-1) + 1e-12)
    p0 = jnp.where(mem["p0set"] > 0.5, mem["p0"], p[0])
    ev = ((spd > VEPS) | (jnp.abs(p - p0[None]).max(-1) > PEPS)).any(0).astype(jnp.float32)
    moved = jnp.maximum(mem["moved"], ev)
    vglob = jnp.maximum(mem["vglob"], feats[:, 4].max())
    mem = {**mem, "moved": moved, "p0": p0, "p0set": jnp.float32(1.0), "vglob": vglob}
    dm, bi, bp, bv, bs, tp = _geo(mem, feats)
    # ---- contact standoff: a contact involving shape j samples j's standoff at the current nearest-body distance ----
    touching = (feats[:, mem["ci"]] @ mem["inc"].T) > 0.5                        # (B,N)
    stand = jnp.clip(jnp.minimum(mem["stand"], jnp.where(touching, dm[:, :N], BIG).min(0)), MIN_STAND, MAX_STAND)
    ncon = jnp.maximum(mem["ncon"], touching.max(0).astype(jnp.float32))
    ncon_a = jnp.concatenate([ncon, jnp.ones(mem["fax"].shape[0], jnp.float32)])  # a face is always "solid"
    Rt = jnp.concatenate([stand, jnp.full(mem["fax"].shape[0], FACE_R, jnp.float32)])
    # ---- valid arms: active, non-green, static-or-wall shapes, plus every arena face ----
    act_m = feats[:, mem["ai"]].max(0) > 0.5
    role = feats[:, mem["ri"]].max(0)
    is_green = jnp.abs(role - 1.0) < 0.5
    static = _dynm({**mem, "moved": moved}) < 0.5
    tok = act_m & (~is_green) & (static | (mem["wall"] > 0.5)) & (mem["use"] > 0.5)
    aok = jnp.concatenate([tok, jnp.full(mem["fax"].shape[0], mem["use"] > 0.5)])
    # ---- score finished attempts: crossings first, in-shell speed second, closest approach only as a tie-break ----
    plan = (mem["aleft"] <= 0) & (~mem["gen"])
    tj = mem["arm"]; Rj = Rt[tj]
    r = (PASS_W * jnp.minimum(1.0, mem["passes"] / PASS_CAP)
         + SPD_W * jnp.minimum(1.0, mem["vmax"])
         + TIE_W * jnp.clip(1.0 - mem["amin"] / (2.0 * SHELL_K * Rj), 0.0, 1.0)) / (PASS_W + SPD_W + TIE_W)
    hot = jnp.where(plan, 1.0, 0.0)
    cnt = jnp.zeros_like(mem["bn"]).at[tj].add(hot)                              # each arm decays once per scored attempt
    rsum = jnp.zeros_like(mem["bsum"]).at[tj].add(hot * r)
    dsc = DISC ** cnt
    bn = dsc * mem["bn"] + cnt; bsum = dsc * mem["bsum"] + rsum
    bmu = bsum / jnp.maximum(bn, 1.0); btot = mem["btot"] + hot.sum()
    got_close = plan & (mem["amin"] < APPROACH_K * Rj) & (ncon_a[tj] < 0.5)
    appr = mem["appr"].at[tj].add(jnp.where(got_close, 1.0, 0.0))
    tun = jnp.minimum(TUN_CAP, TUN_W * jnp.log1p(appr)) * (ncon_a < 0.5)         # approached and never touched = tunnelling
    ucb = bmu + tun + UCB_C * jnp.sqrt(jnp.log(1.0 + btot) / (1.0 + bn))
    newa = jnp.argmax(jnp.where(aok, ucb, -1e9)[None, :] + GSC * jax.random.gumbel(k1, (B, A)), axis=1).astype(jnp.int32)
    sel = plan & aok.any()
    arm = jnp.where(sel, newa, mem["arm"])
    # steered channel: the green channel when the body nearest this target is green, else the whole assembly
    bi_a = _pick(bi, arm)
    role_b = jnp.take_along_axis(feats[:, mem["ri"]], bi_a[:, None], 1)[:, 0]
    ch = jnp.where(sel, ((jnp.abs(role_b - 1.0) < 0.5) & is_green.any()).astype(jnp.int32), mem["ch"])
    # ---- resume a rung of this arm: fastest band first, then the approach side, then least visited ----
    idx = jnp.clip(arm[:, None] * mem["A2"] + mem["brange"][None, :], 0, mem["TB"] - 1)
    rung_ok = arch_valid[idx] & (mem["best"][idx] < BIG)
    key = (-BAND_W * mem["bband"][None, :] + SIDE_W * mem["bside"][None, :]
           + VIS_W * jnp.log1p(arch_visits[idx].astype(jnp.float32)) + 0.3 * jax.random.exponential(k2, idx.shape))
    pick = jnp.argmin(jnp.where(rung_ok, key, 1e9), axis=1)
    slot_l = jnp.take_along_axis(idx, pick[:, None], 1)[:, 0]
    ul = jax.random.uniform(k3, (B, 2))
    use_l = (~mem["gen"]) & ((plan & (ul[:, 0] < P_PLAN_L)) | (ul[:, 1] < P_RESUME)) & rung_ok.any(1)
    # ---- generic pools for the pumping workers (and for arms with no rung yet) ----
    # the pumping workers exist for ANOMALY (top speed) and OOB (outward), so their restores prefer those two pools
    gen_ok = arch_valid & (mem["pool"] > 0)
    glog = jnp.where(gen_ok, GEN_PREF * (mem["pool"] > 1) - 0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot_g = jnp.argmax(glog[None, :] + jax.random.gumbel(k4, (B, K)), axis=1)
    use_g = (~use_l) & ((jax.random.uniform(k5, (B,)) < P_GEN_R)
                        | (plan & (jax.random.uniform(k2, (B,)) < P_PLAN_G))) & gen_ok.any()
    do = use_l | use_g
    slot = jnp.where(use_l, slot_l, slot_g).astype(jnp.int32)
    fresh = mem["fresh"] | do | sel
    return do, slot, {**mem, "stand": stand, "ncon": ncon, "appr": appr, "bn": bn, "bsum": bsum, "btot": btot,
                      "aok": aok, "arm": arm, "ch": ch,
                      "aleft": jnp.where(plan, ATT_L // 2 + jax.random.randint(k6, (B,), 0, ATT_L), mem["aleft"]),
                      "vmax": jnp.where(plan, 0.0, mem["vmax"]), "passes": jnp.where(plan, 0.0, mem["passes"]),
                      "amin": jnp.where(plan, jnp.float32(BIG), mem["amin"]),
                      "left": jnp.where(do | plan, 0, mem["left"]),
                      "M": jnp.where((do | sel)[:, None, None], MDECAY * mem["M"], mem["M"]),
                      "R": jnp.where(sel[:, None], MDECAY * mem["R"], mem["R"]),
                      "fresh": fresh, "reax": mem["reax"] | do | sel}


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
    dm, bi, bp, bv, bs, tp = _geo(mem, feats)
    arm = mem["arm"]; Rj = _rad(mem)[arm]
    d = _pick(dm, arm); tpa = _pick(tp, arm); bpa = _pick(bp, arm); bva = _pick(bv, arm)
    bi_a = _pick(bi, arm)
    u = tpa - bpa; nu = jnp.sqrt((u * u).sum(-1, keepdims=True)) + 1e-6; uh = u / nu
    # the approach axis is fixed for the attempt (re-set after a restore or an arm switch), so "past the target" is signed
    u0 = jnp.where(mem["reax"][:, None], uh, mem["u0"])
    psign = jnp.where(mem["reax"], 1.0, mem["psign"])
    # ---- velocity feedback on the steered channel, aimed body -> target ----
    cv = _chan(mem, feats); cvel = cv[jnp.arange(B), mem["ch"]]
    oh = (jnp.arange(n_act)[None, :] == mem["cur"][:, None]).astype(jnp.float32)
    live = (~mem["fresh"]).astype(jnp.float32)[:, None]
    M = mem["M"] + LR * (live * oh)[:, :, None] * ((cvel - mem["pv"])[:, None, :] - mem["M"])
    # the closing table is only credited while the nearest body is the same one it was last step
    closed = jnp.clip(mem["pd"] - d, -1.0, 1.0) * (mem["pd"] < BIG) * (bi_a == mem["bi"])
    Robs = mem["R"] + LR_OBS * live * oh * (closed[:, None] - mem["R"])
    s = (M[:, :, 0] * uh[:, 0:1] + M[:, :, 1] * uh[:, 1:2]) + W_OBS * Robs
    s = s / (jnp.abs(s).max(-1, keepdims=True) + 1e-6)
    a_aim = jnp.argmax(s / TAU + jax.random.gumbel(k1, s.shape), axis=1).astype(jnp.int32)
    a_aim = jnp.where(jax.random.uniform(k2, (B,)) < EPS_RAND, jax.random.randint(k3, (B,), 0, mem["n_act"]), a_aim)
    # ---- hold scheduling; inside the freeze shell the aim is held so the sweep is never braked mid-crossing ----
    frozen = mem["aim"] & (~mem["gen"]) & (d < FREEZE_K * Rj) & ((bva * uh).sum(-1) > 0.0)
    new = (mem["left"] <= 0) & (~frozen)
    mode = jnp.where(new, jax.random.randint(k4, (B,), 0, mem["lo"].shape[0]), mem["mode"])
    pa, oa, ob = _pump(mem, mode, k5, k6)
    lo = mem["lo"][mode]; hi = mem["hi"][mode]
    plen = lo + jax.random.randint(k7, (B,), 0, jnp.maximum(hi - lo + 1, 1))
    aim = jnp.where(new, (~mem["gen"]) & mem["aok"][arm] & (jax.random.uniform(k5, (B,)) > PUMP_P), mem["aim"])
    cur = jnp.where(new, jnp.where(aim, a_aim, pa), mem["cur"])
    left = jnp.where(new, jnp.where(aim, jnp.int32(HOLD), plen), mem["left"]) - 1
    osc_a = jnp.where(new, oa, mem["osc_a"]); osc_b = jnp.where(new, ob, mem["osc_b"])
    cur = jnp.where((~aim) & (mode == 4), jnp.where(left % 2 == 0, osc_a, osc_b), cur)
    return cur.astype(jnp.int32), {**mem, "M": M, "R": Robs, "pv": cvel, "pd": d, "cur": cur.astype(jnp.int32),
                                   "left": left, "mode": mode, "osc_a": osc_a, "osc_b": osc_b, "aim": aim,
                                   "u0": u0, "psign": psign, "reax": jnp.zeros(B, bool), "bi": bi_a,
                                   "aleft": mem["aleft"] - 1, "fresh": jnp.zeros(B, bool)}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    live = ~done
    dm, bi, bp, bv, bs, tp = _geo(mem, feats)
    arm = mem["arm"]; Rj = _rad(mem)[arm]
    d = _pick(dm, arm); tpa = _pick(tp, arm); bpa = _pick(bp, arm); vsel = _pick(bs, arm)
    # ---- attempt statistics: crossings of the approach axis and top speed inside the shell ----
    sg = ((tpa - bpa) * mem["u0"]).sum(-1)
    big_s = jnp.abs(sg) > SGN_K * Rj
    cross = live & big_s & (d < PASS_K * Rj) & (sg * mem["psign"] < 0.0)
    passes = mem["passes"] + cross.astype(jnp.float32)
    psign = jnp.where(big_s, jnp.sign(sg), mem["psign"])
    # the shell term is speed x closeness inside the shell: a flat shell gives no gradient on a 0.27 m slab
    insh = d < SHELL_K * Rj
    q = (jnp.minimum(vsel / jnp.maximum(mem["vglob"], VREF), 1.0)
         * jnp.clip(1.0 - d / (SHELL_K * Rj), 0.0, 1.0))
    vmax = jnp.maximum(mem["vmax"], jnp.where(insh & live, q, 0.0))
    amin = jnp.minimum(mem["amin"], jnp.where(live, d, BIG))
    # ---- rung: (target, side of the approach axis, speed band); each rung keeps its closest state ----
    band = (vsel[:, None] > mem["vband"][None, :] * jnp.maximum(mem["vglob"], VREF)).sum(-1).astype(jnp.int32)
    band = jnp.minimum(band, mem["nsr"] - 1)
    side = (sg < 0.0).astype(jnp.int32)
    lslot = jnp.clip(arm * mem["A2"] + side * mem["nsr"] + band, 0, mem["TB"] - 1)
    lad = (live & (~mem["gen"]) & mem["aok"][arm] & (d < STORE_K * Rj)
           & ((~arch_valid[lslot]) | (d < mem["best"][lslot] - MARGIN_F * Rj)))
    # ---- generic pools: coverage cells / top speed / most outlying body ----
    cell = _cell(mem, feats)
    is_new = (mem["cell_slot"][cell] < 0) & live & (~lad)
    cov_slot = mem["G0"] + (mem["nf"] + jnp.cumsum(is_new) - 1) % jnp.maximum(mem["ncov"], 1)
    skey = jnp.where(mem["pool"] == 2, jnp.where(arch_valid, arch_feats[:, 4], -1e9), 1e9)
    s_slot = jnp.argmin(skey); use_s = (feats[:, 4] > skey[s_slot]) & live & (~lad) & (~is_new)
    okey = jnp.where(mem["pool"] == 3, jnp.where(arch_valid, _outk(mem, arch_feats), -1e9), 1e9)
    o_slot = jnp.argmin(okey); use_o = (_outk(mem, feats) > okey[o_slot]) & live & (~lad) & (~is_new) & (~use_s)
    do = lad | is_new | use_s | use_o
    slot = jnp.where(lad, lslot, jnp.where(is_new, cov_slot, jnp.where(use_s, s_slot, o_slot))).astype(jnp.int32)
    best = mem["best"].at[jnp.where(do, slot, mem["Kt"])].set(jnp.where(lad, d, BIG))
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, cov_slot, mem["cell_slot"][0]))
    return do, slot, {**mem, "best": best, "cell_slot": cell_slot, "vmax": vmax, "passes": passes, "psign": psign,
                      "amin": amin, "nf": (mem["nf"] + is_new.sum()) % jnp.maximum(mem["ncov"], 1)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
