"""Per-BODY extremes archive ("elites2"). The witnesses this search is after are per-body facts, not aggregate ones:
OOB fires when *one* moving shape's centroid leaves the arena box and TUNNEL when *one* moving shape's centroid is
inside the de-solidified obstacle. An archive of extremes over the aggregate feature head (dyn_pos, dyn_vel, ...)
therefore records the wrong thing -- the assembly centroid can sit still while a passenger body is flung out of the
level. So the frontier here is per body: for every active dynamic body (polygon and circle blocks, split from the
static ones online by ever having been seen moving) the archive keeps the states of its own largest x, smallest x,
largest y, smallest y and largest speed -- 5 frontier slots per body, masked to the bodies that exist and move.

A second frontier holds the traversal key of the companion program: per target (every static/wall shape plus the
three removable arena faces), the state with the highest speed of the body nearest that target while it was inside a
SHELL_K * R shell of it, with R an online contact-standoff estimate (DEF_STAND until a contact is observed) so the
shell is not in absolute world units. That is "the fastest pass-through seen at this target", the state worth
resuming to sweep through it again.

Restores pick a frontier slot of a uniformly chosen (body, dimension) -- uniform over pairs so no body's frontier is
starved by another's -- with a slow-decaying bonus for frontiers extended recently, and every frontier slot also
remembers the action that was being held when it was set: a restore resumes that action for a long hold, so the
archive is a ratchet that keeps pushing each body further in the direction that last worked. Actions are otherwise
long single-direction motor / thruster holds (energy pumping) mixed with short random ones.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
NW = 4                # boundary walls: the first num_static_fixated_polys (=4) polygons of every Kinetix config
ND = 5                # frontier dimensions per body: +x, -x, +y, -y, speed
WORLD = 5.0           # Kinetix arena side, in metres
FACES = ((0, 0.0), (0, 5.0), (1, 5.0))   # (axis, boundary value): left, right, ceiling -- the removable faces
FACE_R = 0.70         # shell unit of a face target
P_RESTORE = 0.08      # per-step probability that a worker jumps to an archived state
P_BODY = 0.62         # of those, fraction going to a per-body frontier slot
P_PASS = 0.15         # ... to the per-target fastest-pass frontier (rest: the coverage pool)
RECENT_DECAY = 0.98   # decay of the "recently extended" bonus, per step
RESUME_HOLD = 14      # steps the stored direction is re-held after a frontier restore
P_DIR = 0.55          # fraction of restores that re-hold the stored direction (the rest draw a fresh hold)
MAX_SHORT = 6         # short random holds
MAX_LONG = 48         # long single-direction holds
MIN_LONG = 10
P_LONG = 0.55         # probability a fresh hold is a long single-direction hold
P_THR = 0.25          # of long holds, fraction that use a thruster binding instead of a motor
SHELL_K = 2.0         # the pass shell is this many R around the target
DEF_STAND = 0.35      # assumed contact standoff before one is observed (median TUNNEL target inradius)
MIN_STAND = 0.06
MAX_STAND = 1.10
VEPS = 1e-3           # speed above which a body is evidence-dynamic
PEPS = 0.02           # displacement from the reset position that also marks a body dynamic
BITS = 12             # coverage-pool hash bits
CELL_DIMS = (0, 1, 2, 3, 5, 6)
CELL_RES = 0.5
BIG = 12.0


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


def init(rng, F, n_act, K, B):
    ix = _idx(F); N = ix["N"]; NFc = len(FACES); A = N + NFc
    nf = ND * N                      # per-body frontier slots
    nfa = nf + A                     # + per-target fastest-pass slots
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "use": jnp.float32(ix["use"]),
            "G0": jnp.int32(min(nfa, K - 1)), "npool": jnp.int32(max(1, K - nfa)), "next_free": jnp.int32(0),
            "nm": jnp.int32(nm), "noop": jnp.int32(noop), "thr": jnp.array(thr, jnp.int32),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "vx": jnp.array(ix["vx"], jnp.int32), "vy": jnp.array(ix["vy"], jnp.int32),
            "ai": jnp.array(ix["ai"], jnp.int32), "ri": jnp.array(ix["ri"], jnp.int32),
            "ci": jnp.array(ix["ci"], jnp.int32), "inc": jnp.array(ix["inc"], jnp.float32),
            "wall": jnp.array(ix["wall"], jnp.float32),
            "fax": jnp.array([f[0] for f in FACES], jnp.float32), "fval": jnp.array([f[1] for f in FACES], jnp.float32),
            # ---- online level model ----
            "moved": jnp.zeros(N, jnp.float32), "p0": jnp.zeros((N, 2), jnp.float32), "p0set": jnp.float32(0.0),
            "stand": jnp.full(N, DEF_STAND, jnp.float32),
            # ---- frontiers: ND per body then one per target; the array shapes carry the split ----
            "eb_val": jnp.full(nf, -jnp.inf, jnp.float32), "eb_rec": jnp.zeros(nf, jnp.float32),
            "ep_val": jnp.full(A, -jnp.inf, jnp.float32), "ep_rec": jnp.zeros(A, jnp.float32),
            "slot_act": jnp.zeros(K + 1, jnp.int32),
            "cell_slot": jnp.full(1 << BITS, -1, jnp.int32),
            # ---- per-worker ----
            "hold": jnp.zeros(B, jnp.int32), "act": jnp.zeros(B, jnp.int32),
            "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS)), "cdims": jnp.array(CELL_DIMS, jnp.int32)}


def _cell(mem, feats):
    x = feats[:, mem["cdims"]] / CELL_RES
    return (((x @ mem["proj"]) > 0) * (1 << jnp.arange(BITS))).sum(-1).astype(jnp.int32)


def _dynm(mem):
    """Estimated dynamic bodies; before any motion has been seen, every non-wall shape counts."""
    return jnp.where(mem["moved"].sum() > 0.5, mem["moved"], 1.0 - mem["wall"])


def _bodies(mem, X):
    """Per-body position (B,N,2), speed (B,N) and an active-dynamic mask (B,N)."""
    p = jnp.stack([X[:, mem["px"]], X[:, mem["py"]]], -1)
    v = jnp.stack([X[:, mem["vx"]], X[:, mem["vy"]]], -1)
    sp = jnp.sqrt((v * v).sum(-1) + 1e-12)
    ok = (X[:, mem["ai"]] > 0.5) & (_dynm(mem)[None, :] > 0.5)
    return p, sp, ok


def _pass_key(mem, X):
    """Per target: the speed of the body nearest it while that body is inside the shell, else -inf. (B, A)."""
    N = mem["px"].shape[0]
    p, sp, ok = _bodies(mem, X)
    tx = X[:, mem["px"]]; ty = X[:, mem["py"]]
    ds = jnp.sqrt((p[:, None, :, 0] - tx[:, :, None]) ** 2 + (p[:, None, :, 1] - ty[:, :, None]) ** 2 + 1e-12)
    coord = jnp.where(mem["fax"][None, :, None] < 0.5, p[:, None, :, 0], p[:, None, :, 1])
    df = jnp.abs(coord - mem["fval"][None, :, None])
    d = jnp.where(ok[:, None, :], jnp.concatenate([ds, df], 1), BIG)             # (B,A,N)
    i = jnp.argmin(d, -1)
    oh = (jnp.arange(N)[None, None, :] == i[:, :, None]).astype(jnp.float32)
    dm = (oh * d).sum(-1)                                                        # (B,A)
    bs = (oh * sp[:, None, :]).sum(-1)
    R = jnp.concatenate([mem["stand"], jnp.full(mem["fax"].shape[0], FACE_R, jnp.float32)])
    return jnp.where(dm < SHELL_K * R[None, :], bs, -jnp.inf), dm


def _keys(mem, X):
    """The nfa frontier keys per worker, all "larger is better": ND per body, then one per target. (B, nfa)."""
    p, sp, ok = _bodies(mem, X)
    B, N = sp.shape
    k = jnp.stack([p[..., 0], -p[..., 0], p[..., 1], -p[..., 1], sp], -1)        # (B,N,ND)
    k = jnp.where(ok[:, :, None], k, -jnp.inf).reshape(B, N * ND)
    pk, _ = _pass_key(mem, X)
    return jnp.concatenate([k, pk], 1)


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; N = mem["px"].shape[0]
    nf = mem["eb_val"].shape[0]; npass = mem["ep_val"].shape[0]
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    # ---- online level model: which bodies move, and each target's contact standoff ----
    p, sp, _ = _bodies(mem, feats)
    p0 = jnp.where(mem["p0set"] > 0.5, mem["p0"], p[0])
    ev = ((sp > VEPS) | (jnp.abs(p - p0[None]).max(-1) > PEPS)).any(0).astype(jnp.float32)
    moved = jnp.maximum(mem["moved"], ev)
    mem = {**mem, "moved": moved, "p0": p0, "p0set": jnp.float32(1.0)}
    _, dm = _pass_key(mem, feats)
    touching = (feats[:, mem["ci"]] @ mem["inc"].T) > 0.5
    stand = jnp.clip(jnp.minimum(mem["stand"], jnp.where(touching, dm[:, :N], BIG).min(0)), MIN_STAND, MAX_STAND)
    # ---- frontier choice: uniform over (body, dimension) pairs, weighted by (1 + recency), masked to live bodies ----
    live_b = (feats[:, mem["ai"]].max(0) > 0.5) & (_dynm(mem) > 0.5)
    bok = jnp.repeat(live_b.astype(jnp.float32), ND)[:nf]                        # (nf,)
    av_b = arch_valid[jnp.minimum(jnp.arange(nf), K - 1)]
    wb = (1.0 + mem["eb_rec"]) * bok * av_b
    wb = jnp.where(wb.sum() > 0, wb / jnp.maximum(wb.sum(), 1e-6), jnp.ones(nf) / nf)
    bslot = jax.random.choice(k1, nf, (B,), p=wb)
    av_p = arch_valid[jnp.minimum(nf + jnp.arange(npass), K - 1)]
    wp = (1.0 + mem["ep_rec"]) * av_p
    wp = jnp.where(wp.sum() > 0, wp / jnp.maximum(wp.sum(), 1e-6), jnp.ones(npass) / npass)
    pslot = nf + jax.random.choice(k2, npass, (B,), p=wp)
    # ---- coverage pool: least-visited valid slot, with noise ----
    pool_ok = arch_valid & (jnp.arange(K) >= mem["G0"])
    score = jnp.where(pool_ok, -jnp.log1p(arch_visits.astype(jnp.float32)) + 0.3 * jax.random.gumbel(k3, (K,)), -jnp.inf)
    cslot = jnp.argmax(score)
    u = jax.random.uniform(k4, (B,))
    slot = jnp.where(u < P_BODY, bslot, jnp.where(u < P_BODY + P_PASS, pslot, cslot)).astype(jnp.int32)
    slot = jnp.clip(slot, 0, K - 1)
    ur = jax.random.uniform(k5, (B, 2))
    do = (ur[:, 0] < P_RESTORE) & arch_valid[slot]
    # a restore resumes the direction that set that frontier: the archive ratchets each body further the same way;
    # the rest of the restores draw a fresh hold, so the ratchet never crowds out ordinary exploration
    ratchet = do & (ur[:, 1] < P_DIR)
    act = jnp.where(ratchet, mem["slot_act"][slot], mem["act"])
    hold = jnp.where(ratchet, jnp.int32(RESUME_HOLD), jnp.where(do, 0, mem["hold"]))
    return do, slot, {**mem, "stand": stand, "act": act.astype(jnp.int32), "hold": hold}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    n_act = mem["n_act"]
    new = mem["hold"] <= 0
    long = jax.random.uniform(k1, (B,)) < P_LONG
    length = jnp.where(long, jax.random.randint(k2, (B,), MIN_LONG, MAX_LONG + 1),
                       jax.random.randint(k2, (B,), 1, MAX_SHORT + 1))
    # a long hold is a single motor or thruster binding held down (energy pumping); a short hold is uniform
    motor = jax.random.randint(k3, (B,), 0, 2 * mem["nm"])
    thr = mem["thr"][jax.random.randint(k4, (B,), 0, mem["thr"].shape[0])]
    a_long = jnp.where(jax.random.uniform(k5, (B,)) < P_THR, thr, motor)
    a_new = jnp.where(long, a_long, jax.random.randint(k3, (B,), 0, n_act))
    action = jnp.where(new, a_new, mem["act"])
    hold = jnp.where(new, length, mem["hold"]) - 1
    return action.astype(jnp.int32), {**mem, "hold": hold, "act": action.astype(jnp.int32)}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    ev0 = jnp.concatenate([mem["eb_val"], mem["ep_val"]]); nfa = ev0.shape[0]
    val = jnp.where(done[:, None], -jnp.inf, _keys(mem, feats))                  # (B, nfa)
    bw = jnp.argmax(val, 0)                                                      # best worker per frontier
    bval = val[bw, jnp.arange(nfa)]
    ext = (bval > ev0) & jnp.isfinite(bval)
    # one store per worker: the lowest-indexed frontier it extends (bodies before pass slots)
    match = (bw[None, :] == jnp.arange(B)[:, None]) & ext[None, :]               # (B, nfa)
    has_f = match.any(1)
    fslot = jnp.where(has_f, jnp.argmax(match, 1), 0).astype(jnp.int32)
    # a frontier's value only advances if the state that set it was actually stored
    stored = ext & (fslot[bw] == jnp.arange(nfa)) & has_f[bw]
    ext_val = jnp.where(stored, bval, ev0)
    ext_rec = jnp.concatenate([mem["eb_rec"], mem["ep_rec"]]) * RECENT_DECAY + stored.astype(jnp.float32)
    nb = mem["eb_val"].shape[0]
    # ---- coverage pool for everyone else ----
    cell = _cell(mem, feats)
    is_new = (mem["cell_slot"][cell] < 0) & (~done) & (~has_f)
    order = jnp.cumsum(is_new) - 1
    cslot = mem["G0"] + (mem["next_free"] + order) % jnp.maximum(mem["npool"], 1)
    do = has_f | is_new
    slot = jnp.where(has_f, fslot, cslot).astype(jnp.int32)
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, cslot, mem["cell_slot"][0]))
    slot_act = mem["slot_act"].at[jnp.where(do, jnp.clip(slot, 0, K - 1), K)].set(mem["act"])
    return do, jnp.clip(slot, 0, K - 1), {**mem, "eb_val": ext_val[:nb], "ep_val": ext_val[nb:],
                      "eb_rec": ext_rec[:nb], "ep_rec": ext_rec[nb:], "cell_slot": cell_slot, "slot_act": slot_act,
                      "next_free": (mem["next_free"] + is_new.sum()) % jnp.maximum(mem["npool"], 1)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
