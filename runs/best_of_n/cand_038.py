"""H3 — "split-and-pump" (the planning/rare-event survey's ranked hybrid). Three parts:
(1) act: zeta-greedy holds with a per-worker exponent mu in {1.5, 2, 2.5, 3} (a portfolio of exploration timescales) and three
    action families — motor holds, thruster holds, and resonant oscillation (a motor whose sign flips with a heavy-tailed period),
    the energy-pumping behaviour the evolved program had hand-capped at 60 steps;
(2) archive: multi-hypothesis ratchet pools. A target-blind importance VECTOR phi is computed per worker — escape (max wall
    distance of any moving body), kinetic energy, displacement, per-static-shape proximity of the nearest moving body, per-shape
    reach by the green body, new contacts — and the K slots are split into one pool per coordinate; a worker's state enters pool c
    when phi_c beats the pool's current minimum (a ratchet on that hypothesis). Since every planted witness is a threshold on one
    such coordinate, the pool for the true hypothesis always exists;
(3) restore: adaptive multilevel splitting per pool (Cerou & Guyader 2007): each pool keeps a level = running quantile of its
    members' phi; a worker assigned to pool c is restored onto a random pool member when its own phi_c is below the level (it is
    "killed and split") or after H steps; pools are assigned to workers by a bandit on downstream yield (new ratchet entries
    produced from that pool per selection), so hypotheses that keep producing progress get more workers."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
```python
MUS = (1.5, 2.0, 2.5, 3.0); N_MAX = 256
FAM_P = (0.4, 0.2, 0.4)             # motor / thruster / oscillate
H = 160                             # max steps between splits
Q = 0.5                             # AMS level = this quantile of the pool
LVL_EMA = 0.2
P_RES0, P_RES1 = 0.05, 0.25         # restore rate ramps with time (score is time-discounted)
WORLD = 5.0
RESERVE_FRAC = 0.15                 # workers that never restore
NW = 4                              # boundary walls are the first polygons
WIN = 48                            # controller hill-climb / stuck-detector window
EPS = 0.25
BIG = 9.0
NH = 10                             # dense hypothesis vector (see _phi)
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pext=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np,
                     cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _bodies(feats):
    """positions (B,N,2), speeds (B,N), active, role, spin (polygon angular channel), polygons first (walls 0..NW-1)."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    spin = jnp.concatenate([feats[:, L["pext"]:L["pext"] + Np], jnp.zeros((B, Nc))], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), act, role, spin, L
def _phi(feats, mem):
    """importance vector (B, NH). Ten dense, target-blind extremal coordinates:
    0 escape (moving body furthest toward/past a wall)      5 tunnel  (fast body AT a static solid: spd/dist)
    1 max speed                                             6 wallhit (fast body AT the boundary: spd/dist)
    2 total kinetic energy (runaway of the whole scene)     7 reach   (green's closest approach to any solid)
    3 max displacement from the start layout                8 gpen    (green arriving at a solid WITH speed)
    4 max |spin| (runaway rotation)                         9 new contacts (novel interaction pairs)
    Every planted witness is a threshold on one of these, and each is a max-ratchet, so the pools are 'reach an
    extreme state' in ten different directions -- the speed-weighted ones (5,6,8) replace 32 per-shape proximity
    coordinates that diluted the archive and said nothing about whether a contact would actually penetrate."""
    pos, spd, act, role, spin, L = _bodies(feats); B, N = spd.shape
    mov = mem["dyn"][:N] & act
    mov = jnp.where(mov.any(1, keepdims=True), mov, act)                    # before anything has moved, use all active
    dwall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)                    # Chebyshev distance from arena centre (2.5 = at a wall)
    mspd = jnp.where(mov, spd, 0.0)
    escape = jnp.max(jnp.where(mov, dwall, -BIG), axis=1, initial=-BIG) / (0.5 * WORLD)
    vmax = jnp.max(mspd, axis=1, initial=0.0) / 8.0
    ke = jnp.sum(mspd ** 2, axis=1) / 64.0
    disp = jnp.max(jnp.where(mov, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), axis=1, initial=0.0) / WORLD
    rot = jnp.max(jnp.where(mov, jnp.abs(spin), 0.0), axis=1, initial=0.0) / 8.0
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1) + jnp.eye(N)[None] * BIG
    solid = act & (jnp.arange(N)[None, :] >= NW)                            # non-wall shapes
    stat = solid & ~mov                                                     # static solids: tunnel / disarmed-hazard targets
    pen = spd[:, :, None] / (d + EPS)                                       # (B,N,N) penetration potential of i into j
    tunnel = jnp.max(jnp.where(mov[:, :, None] & stat[:, None, :], pen, -BIG), axis=(1, 2), initial=-BIG) / 8.0
    towall = jnp.max(jnp.where(mov, mspd / jnp.clip(0.5 * WORLD - dwall, EPS, None), -BIG), axis=1, initial=-BIG) / 8.0
    green = (role == 1) & act
    green = jnp.where(green.any(1, keepdims=True), green, mov)
    gm = green[:, :, None] & solid[:, None, :] & ~green[:, None, :]
    reach = -jnp.min(jnp.where(gm, d, BIG), axis=(1, 2), initial=BIG) / WORLD
    gpen = jnp.max(jnp.where(gm, pen, -BIG), axis=(1, 2), initial=-BIG) / 8.0
    contacts = feats[:, L["cfrom"]:] > 0.5
    newc = jnp.sum(contacts & ~mem["seen_c"][None, : contacts.shape[1]], axis=1).astype(jnp.float32)
    phi = jnp.stack([escape, vmax, ke, disp, rot, tunnel, towall, reach, gpen, newc], 1)
    return phi, contacts, spd, act
def init(rng, F, n_act, K, B):
    ks = jax.random.split(rng, 3)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
            "sign": jnp.ones(B, jnp.int32), "per": jnp.full(B, 8, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(ks[0], (B,), 0, len(MUS))],
            "fam": jax.random.choice(ks[1], 3, (B,), p=jnp.array(FAM_P)),
            "greedy": (jnp.arange(B) % 3) != 0,                             # 1/3 stay pure heavy-tailed random
            "ebest": jnp.zeros(B), "pchk": jnp.zeros((B, 64, 2)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, 64, 2)), "dyn": jnp.zeros(64, bool),
            "seen_c": jnp.zeros(256, bool), "have_start": jnp.bool_(False),
            "pool": jnp.arange(B) % NH, "age": jnp.zeros(B, jnp.int32),
            "slot_pool": jnp.arange(K) % NH, "slot_phi": jnp.full(K, -jnp.inf),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH),
            "src": jnp.full(B, -1, jnp.int32)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    phi, _, _, _ = _phi(feats, mem)
    my = phi[jnp.arange(B), mem["pool"]]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 1500.0, 0.0, 1.0)
    pool_ok = (mem["slot_pool"][None, :] == mem["pool"][:, None]) & arch_valid[None, :]        # (B,K)
    pool_ok = jnp.where(pool_ok.any(1, keepdims=True), pool_ok, arch_valid[None, :])           # empty pool: fall back to anything
    # AMS: split onto a SURVIVING member (phi above the pool level), not an arbitrary one
    alive = pool_ok & (mem["slot_phi"][None, :] >= mem["level"][mem["pool"]][:, None])
    cand = jnp.where(alive.any(1, keepdims=True), alive, pool_ok)
    valid_phi = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pmax = jnp.full(NH, -jnp.inf).at[mem["slot_pool"]].max(valid_phi)
    pmin = jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(jnp.where(arch_valid, mem["slot_phi"], jnp.inf))
    span = pmax[mem["slot_pool"]] - pmin[mem["slot_pool"]]
    z = (mem["slot_phi"] - pmin[mem["slot_pool"]]) / jnp.where(span > 1e-6, span, 1.0)
    z = jnp.clip(jnp.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    # frontier-biased, visit-discounted slot choice (arch_visits was previously ignored)
    score = jax.random.gumbel(k1, (B, K)) + 1.0 * z[None, :] - 0.7 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(cand, score, -jnp.inf), axis=1)
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    below = my < mem["level"][mem["pool"]]
    p = P_RES0 + (P_RES1 - P_RES0) * tau
    do = ((below & (jax.random.uniform(k2, (B,)) < p)) | (mem["age"] >= H)) & cand.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    w = (1.0 + mem["yield"]) / (1.0 + sel); w = w / jnp.sum(w)              # bandit on downstream ratchet yield
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, mem["pool"])
    return do, slot, {**mem, "pool": pool, "age": jnp.where(do, 0, mem["age"]), "sel": sel,
                      "src": jnp.where(do, slot, mem["src"]), "left": jnp.where(do, 0, mem["left"]),
                      "ebest": jnp.where(do, 0.0, mem["ebest"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    n_act = mem["n_act"]; NM = 4
    pos, spd, actm, role, spin, L = _bodies(feats); N = spd.shape[1]
    e = jnp.max(jnp.where(actm, spd, 0.0), axis=1, initial=0.0)
    chk = (mem["t"] % WIN) == 0
    mv = jnp.max(jnp.linalg.norm(pos - mem["pchk"][:, :N], axis=-1), axis=1, initial=0.0)
    # stuck detector: a heavy-tailed hold on a controller that moves nothing burns up to N_MAX discounted steps
    stuck = chk & mem["greedy"] & ((mv < 0.06) | (e < 0.15 * mem["ebest"]))
    ebest = jnp.where(chk, jnp.maximum(e, 0.995 * mem["ebest"]), mem["ebest"])
    pchk = jnp.where(chk, mem["pchk"].at[:, :N].set(pos), mem["pchk"])
    new = (mem["left"] <= 0) | stuck
    fam = jnp.where(stuck, jax.random.choice(k5, 3, (B,), p=jnp.array(FAM_P)), mem["fam"])
    mmax = jnp.maximum(jnp.minimum(2 * NM, n_act), 1)
    motor = jax.random.randint(k1, (B,), 0, mmax)
    tmin = jnp.minimum(2 * NM + 1, jnp.maximum(n_act - 1, 0))
    thr = jnp.where(n_act > 2 * NM + 1, tmin + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - tmin, 1)), motor)
    base = jnp.where(fam == 1, thr, motor)                                   # family 2 (oscillate) drives a motor and flips its sign
    cur = jnp.where(new, base, mem["cur"])
    hold = _hold(mem, k3, B); left = jnp.where(new, hold, mem["left"]) - 1
    lg = jax.random.uniform(k4, (B,))                                        # period log-uniform in [2,64], decoupled from the hold
    per = jnp.where(new, jnp.clip(jnp.round(2.0 * jnp.exp(lg * jnp.log(32.0))), 2, 64).astype(jnp.int32), mem["per"])
    osc = fam == 2; tick = (mem["left"] % jnp.maximum(per, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    action = jnp.where(osc, (cur // 2) * 2 + (sign < 0).astype(jnp.int32), cur)
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0))
    return action, {**mem, "cur": cur, "left": left, "per": per, "sign": sign, "fam": fam,
                    "ebest": ebest, "pchk": pchk, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, actm, role, spin, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True)}
    phi, contacts, _, _ = _phi(feats, mem2)
    pools = jnp.arange(NH)
    slot_score = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    owns = mem["slot_pool"][None, :] == pools[:, None]                                       # (NH,K)
    pool_min = jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(slot_score)
    pool_min = jnp.where(jnp.isinf(pool_min), -jnp.inf, pool_min)                            # empty / unfilled pool: anything qualifies
    cand = (phi > pool_min[None, :]) & ~done[:, None]                                        # ratchet per hypothesis
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0)                              # (NH,) best worker per pool
    has = cand.any(0) & owns.any(1)
    pool_slot = jnp.argmin(jnp.where(owns, slot_score[None, :], jnp.inf), axis=1)            # pool's weakest (or empty) slot
    # a worker that wins several pools may only store once: give it to the highest-index claimant so
    # slot_phi bookkeeping matches what is actually written
    owner = jnp.full(B, -1, jnp.int32).at[best_w].max(jnp.where(has, pools, -1))
    ok = has & (owner[best_w] == pools)
    do = jnp.zeros(B, jnp.int32).at[best_w].max(ok.astype(jnp.int32)) > 0
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(ok, pool_slot, 0))
    slot_phi = mem["slot_phi"].at[pool_slot].set(jnp.where(ok, phi[best_w, pools], mem["slot_phi"][pool_slot]))
    members = jnp.where(arch_valid[None, :] & owns, slot_phi[None, :], jnp.nan)
    q = jnp.nan_to_num(jnp.nanquantile(members, Q, axis=1), nan=-jnp.inf)
    fin = q > -1e30
    level = jnp.where(fin, (1 - LVL_EMA) * jnp.where(mem["level"] > -1e30, mem["level"], q) + LVL_EMA * q, mem["level"])
    src_pool = jnp.where(mem["src"] >= 0, mem["slot_pool"][jnp.clip(mem["src"], 0, K - 1)], 0)
    yld = (mem["yield"] * 0.999).at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    seen_c = mem["seen_c"].at[: contacts.shape[1]].set(mem["seen_c"][: contacts.shape[1]] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld,
                      "sel": mem["sel"] * 0.999, "seen_c": seen_c}
```
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
