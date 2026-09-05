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
MUS = (1.5, 2.0, 2.5, 3.0); N_MAX = 192
FAM_P = (0.38, 0.20, 0.26, 0.16)    # motor hold / thruster hold / resonant oscillation / systematic action sweep
H = 72                              # forced split interval: AMS needs MANY short generations under a time-discounted score
P0, P1, T_RAMP = 0.06, 0.35, 512.0  # level-triggered split probability, ramped
Q = 0.5                             # AMS level = this quantile of the pool
LVL_EMA = 0.25
DECAY = 0.998                       # bandit forgetting
WORLD = 5.0
RESERVE_FRAC = 0.2                  # workers that never restore (long-horizon explorers)
MAX_SHAPES = 12
NW = 4                              # boundary walls are the first polygons
NB = 96                             # max bodies tracked for start-state memory
NC_MAX = 512
NH = 7 + 2 * MAX_SHAPES             # global coords + per-shape proximity + per-shape green-reach
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if sols else (max(4, min(40, (F - 11) // 7)), 0)
    C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pspin=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np,
                     cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _bodies(feats):
    """pos (B,N,2), speed (B,N), |angular velocity| (B,N), active (B,N), role (B,N); polygons first, walls are 0..NW-1."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    spin = jnp.concatenate([feats[:, L["pspin"]:L["pspin"] + Np], jnp.zeros((B, Nc))], 1)   # polygons carry an extra scalar: angular velocity
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), jnp.abs(spin), act, role, L
def _phi(feats, mem):
    """importance VECTOR (B,NH) + the live-coordinate mask (NH,) + the contact bitmap.
    coords: 0 escape, 1 speed, 2 spin, 3 displacement, 4 spread, 5 compression (penetration proxy),
            6 new contacts, 7..7+M-1 proximity of any body to shape j, 7+M.. green's reach to shape j."""
    pos, spd, spin, act, role, L = _bodies(feats)
    B, N = spd.shape; nb = min(N, NB); ns = max(0, min(MAX_SHAPES, N - NW)); ctr = 0.5 * WORLD
    escape = jnp.max(jnp.where(act, jnp.max(jnp.abs(pos - ctr), axis=-1), 0.0), axis=1) / ctr
    speed = jnp.max(jnp.where(act, spd, 0.0), axis=1) / 8.0
    swirl = jnp.max(jnp.where(act, spin, 0.0), axis=1) / 8.0
    disp = jnp.max(jnp.where(act[:, :nb], jnp.linalg.norm(pos[:, :nb] - mem["p0"][:, :nb], axis=-1), 0.0), axis=1) / WORLD
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)                       # (B,N,N)
    pair = act[:, :, None] & act[:, None, :] & ~jnp.eye(N, dtype=bool)[None]
    spread = jnp.max(jnp.where(pair, d, 0.0), axis=(1, 2)) / WORLD
    d0 = jnp.linalg.norm(mem["p0"][:, :nb, None, :] - mem["p0"][:, None, :nb, :], axis=-1)      # rest-state separations
    comp = -jnp.min(jnp.where(pair[:, :nb, :nb], d[:, :nb, :nb] - d0, 9.0), axis=(1, 2)) / WORLD  # shape-agnostic tunnel/penetration signal
    prox = -jnp.min(jnp.where(pair[:, :, NW:NW + ns], d[:, :, NW:NW + ns], 9.0), axis=1)        # (B,ns) nearest body to each non-wall shape
    green = (role == 1) & act
    reach = -jnp.min(jnp.where(green[:, :, None] & pair, d, 9.0), axis=1)[:, NW:NW + ns]        # (B,ns) green body to each shape
    pad = lambda x: jnp.concatenate([x, jnp.full((B, MAX_SHAPES - x.shape[1]), -9.0)], 1)
    con = (feats[:, L["cfrom"]:] > 0.5)[:, :NC_MAX]; nc = con.shape[1]
    newc = jnp.sum(con & ~mem["seen_c"][None, :nc], axis=1).astype(jnp.float32)
    phi = jnp.concatenate([escape[:, None], speed[:, None], swirl[:, None], disp[:, None], spread[:, None], comp[:, None],
                           newc[:, None], pad(prox) / WORLD, pad(reach) / WORLD], 1)
    # liveness: a per-shape coordinate is only a hypothesis if the level actually instantiates that shape
    sact = act[:, NW:NW + ns].any(0)
    padb = lambda x: jnp.concatenate([x, jnp.zeros(MAX_SHAPES - x.shape[0], bool)])
    live = jnp.concatenate([jnp.ones(7, bool), padb(sact), padb(sact & ~green[:, NW:NW + ns].any(0)) & green.any()])
    return phi, live, con
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32), "flip": jnp.zeros(B, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 4, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, NB, 2)), "seen_c": jnp.zeros(NC_MAX, bool), "have_start": jnp.bool_(False),
            "pool": jnp.arange(B) % NH, "age": jnp.zeros(B, jnp.int32), "slot_pool": jnp.arange(K) % NH, "slot_phi": jnp.full(K, -jnp.inf),
            "live": jnp.ones(NH, bool), "has_slot": jnp.ones(NH, bool),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    phi = _phi(feats, mem)[0]; pool = mem["pool"]
    my = phi[jnp.arange(B), pool]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / T_RAMP, 0.0, 1.0)
    below = my < mem["level"][pool]                                      # AMS: killed when it falls under its pool's level
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == pool[:, None]) & arch_valid[None, :]          # (B,K)
    own = (mem["slot_pool"][None, :] == jnp.arange(NH)[:, None]) & arch_valid[None, :]    # (NH,K)
    best_slot = jnp.argmax(jnp.where(own, mem["slot_phi"][None, :], -jnp.inf), axis=1)
    top = jnp.zeros(K).at[best_slot].max(own.any(1).astype(jnp.float32))                  # the frontier state of each pool
    score = jax.random.gumbel(k1, (B, K)) + 1.5 * top[None, :] - 0.35 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(pool_ok, score, -jnp.inf), axis=1)
    do = ((below & (jax.random.uniform(k2, (B,)) < P0 + (P1 - P0) * tau)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[pool].add(do.astype(jnp.float32))
    # pool bandit on downstream yield, restricted to pools that actually own archive slots
    w = ((1.0 + mem["yield"]) / (1.0 + sel)) * mem["has_slot"].astype(jnp.float32)
    w = jnp.where(w.sum() > 0, w, jnp.ones(NH)); w = w / w.sum()
    swap = do & (jax.random.uniform(k4, (B,)) < 0.4)
    pool = jnp.where(swap, jax.random.choice(k3, NH, (B,), p=w), pool)
    # a split also resamples the policy, so one archived state is retried at several exploration timescales
    mu = jnp.where(do, jnp.array(MUS)[jax.random.randint(k5, (B,), 0, len(MUS))], mem["mu"])
    fam = jnp.where(do, jax.random.choice(k6, 4, (B,), p=jnp.array(FAM_P)), mem["fam"])
    return do, slot, {**mem, "pool": pool, "age": jnp.where(do, 0, mem["age"]), "sel": sel, "mu": mu, "fam": fam,
                      "src": jnp.where(do, slot, mem["src"]), "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3 = jax.random.split(rng, 3); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    motor = jax.random.randint(k1, (B,), 0, jnp.maximum(jnp.minimum(2 * NM, n_act), 1))
    lo = jnp.clip(jnp.minimum(2 * NM + 1, n_act - 1), 0, jnp.maximum(n_act - 1, 0))
    thr = lo + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - lo, 1))
    fam = mem["fam"]
    cur = jnp.where(new, jnp.where(fam == 1, thr, motor), mem["cur"])
    hold = _hold(mem, k3, B); left = jnp.where(new, hold, mem["left"]) - 1
    flip = jnp.where(new, jnp.clip(hold // 4, 2, 40), mem["flip"])
    osc = fam == 2; tick = (mem["left"] % jnp.maximum(flip, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    a_osc = (cur // 2) * 2 + (sign < 0).astype(jnp.int32)                                  # resonant pumping
    a_swp = (cur + mem["t"] // jnp.maximum(flip // 2, 1)) % jnp.maximum(n_act, 1)           # systematic coverage of the action set
    action = jnp.where(fam == 3, a_swp, jnp.where(osc, a_osc, cur))
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0))
    return action, {**mem, "cur": cur, "left": left, "flip": flip, "sign": sign, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos = _bodies(feats)[0]; N = pos.shape[1]; nb = min(N, NB)
    first = ~mem["have_start"]
    p0 = jnp.where(first, mem["p0"].at[:, :nb].set(pos[:, :nb]), mem["p0"])
    mem2 = {**mem, "p0": p0}
    phi, live_now, con = _phi(feats, mem2)
    live = jnp.where(first, live_now, mem["live"])
    # hand archive slots round-robin to LIVE pools only (dead per-shape coordinates get none)
    rank = jnp.cumsum(live) - 1; nl = jnp.maximum(live.sum(), 1)
    sp = jnp.argmax(live[None, :] & (rank[None, :] == (jnp.arange(K) % nl)[:, None]), axis=1)
    slot_pool = jnp.where(first, sp, mem["slot_pool"])
    has_slot = jnp.zeros(NH, bool).at[slot_pool].set(True)
    own = slot_pool[None, :] == jnp.arange(NH)[:, None]                                    # (NH,K)
    ov = own & arch_valid[None, :]
    pool_min = jnp.min(jnp.where(ov, mem["slot_phi"][None, :], jnp.inf), axis=1)
    thr = jnp.where(ov.sum(1) < own.sum(1), -jnp.inf, pool_min)                            # empty slots accept anything
    sc = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_slot = jnp.argmin(jnp.where(own, sc[None, :], jnp.inf), axis=1)                   # the pool's weakest (or empty) slot
    cand = (phi > thr[None, :]) & ~done[:, None] & (has_slot & live)[None, :]
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0); has = cand.any(0)
    # one store per worker per step: a worker winning several pools commits to exactly one, so slot_phi never desynchronises
    which = jnp.full(B, -1, jnp.int32).at[best_w].max(jnp.where(has, jnp.arange(NH), -1))
    eff = has & (which[best_w] == jnp.arange(NH))
    do = jnp.zeros(B).at[best_w].max(eff.astype(jnp.float32)) > 0.5
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(eff, pool_slot, 0))
    sm = (jnp.arange(K)[:, None] == pool_slot[None, :]) & eff[None, :]                     # (K,NH) collision-free write
    slot_phi = jnp.where(sm.any(1), jnp.max(jnp.where(sm, phi[best_w, jnp.arange(NH)][None, :], -jnp.inf), axis=1), mem["slot_phi"])
    members = jnp.where(own & arch_valid[None, :], slot_phi[None, :], jnp.nan)
    q = jnp.nanquantile(members, Q, axis=1)
    level = jnp.where(jnp.isnan(q), mem["level"], (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], jnp.nan_to_num(q)) + LVL_EMA * jnp.nan_to_num(q))
    src_pool = jnp.where(mem["src"] >= 0, slot_pool[jnp.clip(mem["src"], 0, K - 1)], 0)
    yld = (mem["yield"] * DECAY).at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    nc = con.shape[1]
    seen_c = mem["seen_c"].at[:nc].set(mem["seen_c"][:nc] | con.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "slot_pool": slot_pool, "live": live, "has_slot": has_slot,
                      "level": level, "yield": yld, "sel": mem["sel"] * DECAY, "seen_c": seen_c, "have_start": jnp.bool_(True)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
