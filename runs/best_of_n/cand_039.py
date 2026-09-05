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
MUS = (1.25, 1.6, 2.0, 2.5, 3.0); N_MAX = 256
FAM_P = (0.45, 0.25, 0.30)         # motor / thruster / oscillate
FLIPS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)   # resonance period sweep (matching a natural frequency is a search, not a formula)
H = 160                             # max steps between splits
LVL_EMA = 0.2
P_RESTORE_MIN = 0.03; P_RESTORE_MAX = 0.15    # early phase: restores rare (the score is time-discounted)
KEEP_P = 0.75                       # prob of re-using an action that just improved the local score
RESAMPLE_FAM = 0.25
WORLD = 5.0
RESERVE_FRAC = 0.15                 # workers that never restore
MAX_SHAPES = 6                      # per-shape hypotheses (walls excluded)
NG = 6                              # global hypotheses
NW = 4                              # boundary walls are the first polygons
NCAP = 64
NH = NG + 2 * MAX_SHAPES            # 18 importance coordinates = 18 ratchet pools
DECAY = 0.999
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _bodies(feats):
    """positions (B, N, 2), speeds (B, N), active (B, N), role (B, N) for all shapes, polygons first (walls are 0..NW-1)."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), act, role, L
def _local(feats, mem):
    """cheap target-blind progress scalar used by act's greedy hold-extension (escape + speed)."""
    pos, spd, act, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"][:N] & act
    e = jnp.max(jnp.where(dyn, jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1), 0.0), axis=1, initial=0.0) / (0.5 * WORLD)
    s = jnp.max(jnp.where(dyn, spd, 0.0), axis=1, initial=0.0) / 8.0
    return e + s
def _phi(feats, mem):
    """importance vector (B, NH): [escape, energy, displacement, new-contacts, closest-moving-pair, spread,
    proximity of nearest moving body to non-wall shape j (MAX_SHAPES), green's reach to shape j (MAX_SHAPES)]"""
    pos, spd, act, role, L = _bodies(feats); B, N = spd.shape; n = min(N, NCAP)
    dyn = mem["dyn"][:N] & act                                             # (B, N) moving & active
    dist_wall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)               # Chebyshev distance from the arena centre (2.5 = at a wall)
    escape = jnp.max(jnp.where(dyn, dist_wall, 0.0), axis=1, initial=0.0) / (0.5 * WORLD)
    energy = jnp.max(jnp.where(dyn, spd, 0.0), axis=1, initial=0.0) ** 2 / 64.0
    dsp = jnp.linalg.norm(pos[:, :n] - mem["p0"][:, :n], axis=-1)
    disp = jnp.max(jnp.where(dyn[:, :n], dsp, 0.0), axis=1, initial=0.0) / WORLD
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)   # (B, N, N)
    mm = dyn[:, :, None] & dyn[:, None, :] & ~jnp.eye(N, dtype=bool)[None]  # distinct moving pairs
    spread = jnp.max(jnp.where(mm, d, 0.0), axis=(1, 2), initial=0.0) / WORLD          # coverage of position space
    close = -jnp.min(jnp.where(mm, d, 9.0), axis=(1, 2), initial=9.0) / WORLD          # high-energy interaction / interpenetration
    stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)                     # static non-wall shapes: tunnel/contact targets
    prox = -jnp.min(jnp.where(dyn[:, :, None] & stat[:, None, :], d, 9.0), axis=1, initial=9.0)   # (B, N) per target shape: nearest moving body
    green = (role == 1) & act
    reach = -jnp.min(jnp.where(green[:, :, None] & act[:, None, :] & ~green[:, None, :], d, 9.0), axis=1, initial=9.0)  # (B, N) green to each other shape
    pad = lambda x: jnp.concatenate([x[:, NW:], jnp.full((B, MAX_SHAPES), -9.0)], 1)[:, :MAX_SHAPES]   # walls are never targets
    nc = min(max(feats.shape[1] - L["cfrom"], 0), 256)
    contacts = feats[:, L["cfrom"]:L["cfrom"] + nc] > 0.5
    newc = jnp.sum(contacts & ~mem["seen_c"][None, :nc], axis=1).astype(jnp.float32)
    glob = jnp.stack([escape, energy, disp, newc, close, spread], axis=1)
    return jnp.concatenate([glob, pad(prox) / WORLD, pad(reach) / WORLD], 1), contacts, spd, act
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng); NP = min(NH, K)                        # no hypothesis without archive slots
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32), "flip": jnp.full(B, 8, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 3, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, NCAP, 2)), "dyn": jnp.zeros(NCAP, bool), "seen_c": jnp.zeros(256, bool), "have_start": jnp.bool_(False),
            "g0": jnp.zeros(B), "pool": jnp.arange(B) % NP, "age": jnp.zeros(B, jnp.int32), "slot_pool": jnp.arange(K) % NP, "slot_phi": jnp.full(K, -jnp.inf),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mu, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mu - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; NP = min(NH, K)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    phi, contacts, spd, act = _phi(feats, mem)
    my = phi[jnp.arange(B), mem["pool"]]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    below = my < mem["level"][mem["pool"]]
    reserve = jnp.arange(B) < max(1, int(RESERVE_FRAC * B))
    pool_ok = (mem["slot_pool"][None, :] == mem["pool"][:, None]) & arch_valid[None, :]           # (B, K)
    spf = jnp.where(jnp.isfinite(mem["slot_phi"]), mem["slot_phi"], -2.0)
    key = 2.0 * spf[None, :] - 0.5 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :] + jax.random.gumbel(k1, (B, K))
    slot = jnp.argmax(jnp.where(pool_ok, key, -jnp.inf), axis=1)           # stochastic-elitist, visit-penalised member of the pool
    p_res = P_RESTORE_MIN + (P_RESTORE_MAX - P_RESTORE_MIN) * tau
    do = ((below & (jax.random.uniform(k2, (B,)) < p_res)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    # pool bandit: re-assign a restoring worker to a pool with probability ∝ (1 + yield) / (1 + sel)
    w = jnp.where(jnp.arange(NH) < NP, (1.0 + mem["yield"]) / (1.0 + sel), 0.0); w = w / jnp.maximum(w.sum(), 1e-9)
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, mem["pool"])
    mu = jnp.where(do, jnp.array(MUS)[jax.random.randint(k5, (B,), 0, len(MUS))], mem["mu"])      # fresh timescale on a fresh branch
    fam = jnp.where(do, jax.random.choice(k5, 3, (B,), p=jnp.array(FAM_P)), mem["fam"])
    age = jnp.where(do, 0, mem["age"]); src = jnp.where(do, slot, mem["src"])
    return do, slot, {**mem, "pool": pool, "age": age, "sel": sel, "src": src, "mu": mu, "fam": fam,
                      "left": jnp.where(do, 0, mem["left"]), "g0": jnp.where(do, 0.0, mem["g0"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 7); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    g = _local(feats, mem); improved = g > mem["g0"] + 1e-3                # did the last hold pump energy / push outward?
    motor = jax.random.randint(k1, (B,), 0, jnp.maximum(jnp.minimum(2 * NM, n_act), 1))
    lo = jnp.minimum(2 * NM + 1, jnp.maximum(n_act - 1, 0))
    thr = jnp.where(n_act > 2 * NM + 1, lo + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - lo, 1)), motor)
    fam = jnp.where(new & (jax.random.uniform(k6, (B,)) < RESAMPLE_FAM), jax.random.choice(k7, 3, (B,), p=jnp.array(FAM_P)), mem["fam"])
    base = jnp.where(fam == 1, thr, motor)                    # family 2 (oscillate) uses a motor and flips its sign
    keep = new & improved & (jax.random.uniform(k4, (B,)) < KEEP_P)        # greedy persistence on a productive actuator
    cur = jnp.where(new, jnp.where(keep, mem["cur"], base), mem["cur"])
    hold = _hold(mem["mu"], k3, B); hold = jnp.where(keep, jnp.minimum(hold * 2, N_MAX), hold)
    left = jnp.where(new, hold, mem["left"]) - 1
    flip = jnp.where(new, jnp.array(FLIPS)[jax.random.randint(k5, (B,), 0, len(FLIPS))], mem["flip"])
    osc = (fam == 2); tick = (mem["left"] % jnp.maximum(flip, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    action = jnp.where(osc, (cur // 2) * 2 + (sign < 0).astype(jnp.int32), cur)
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0))
    return action, {**mem, "cur": cur, "left": left, "flip": flip, "sign": sign, "fam": fam,
                    "g0": jnp.where(new, g, mem["g0"]), "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; NP = min(NH, K)
    pos, spd, act, role, L = _bodies(feats); N = spd.shape[1]; n = min(N, NCAP)
    dyn = mem["dyn"].at[:n].set(mem["dyn"][:n] | (spd[:, :n] > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :n].set(pos[:, :n]))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True)}
    phi, contacts, _, _ = _phi(feats, mem2)
    active = jnp.arange(NH) < NP
    sp = jnp.where(arch_valid & jnp.isfinite(mem["slot_phi"]), mem["slot_phi"], -jnp.inf)          # invalid slot = free
    # ratchet: for each pool, the worker with the highest phi_c replaces the pool's weakest slot if it beats it
    pool_min = jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(sp)
    pool_min = jnp.where(pool_min == jnp.inf, -jnp.inf, pool_min)                                  # pool with no slot: masked out below
    cand = (phi > pool_min[None, :] + 1e-4) & ~done[:, None] & active[None, :]                     # (B, NH)
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0)                                    # (NH,) best worker per pool
    has = cand.any(0)
    pool_slot = jnp.argmin(jnp.where(mem["slot_pool"][None, :] == jnp.arange(NH)[:, None], sp[None, :], jnp.inf), axis=1)  # (NH,)
    val = phi[best_w, jnp.arange(NH)]
    pm = jnp.where(jnp.isfinite(pool_min), pool_min, -2.0)
    gain = jnp.where(has, val - pm, -jnp.inf) - 1e-6 * jnp.arange(NH)                              # epsilon breaks ties uniquely
    best_gain = jnp.full(B, -jnp.inf).at[best_w].max(gain)
    accept = has & (gain >= best_gain[best_w])                                                     # exactly one pool per worker
    do = jnp.zeros(B, bool).at[best_w].max(accept)
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(accept, pool_slot, 0))
    slot_phi = sp.at[pool_slot].max(jnp.where(accept, val, -jnp.inf))                              # only real stores move the ratchet
    # AMS levels: EMA toward the mean phi of the pool's members
    msk = arch_valid[None, :] & (mem["slot_pool"][None, :] == jnp.arange(NH)[:, None])
    fin = jnp.where(jnp.isfinite(slot_phi), slot_phi, -2.0)
    cnt = msk.sum(1); mval = jnp.sum(jnp.where(msk, fin[None, :], 0.0), axis=1) / jnp.maximum(cnt, 1)
    level = jnp.where((cnt > 0) & active, (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], mval) + LVL_EMA * mval, mem["level"])
    # credit: a store made by a worker that was restored from pool p counts as yield for p
    src_pool = mem["slot_pool"][jnp.clip(mem["src"], 0, K - 1)]
    yld = (mem["yield"] * DECAY).at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    nc = contacts.shape[1]
    seen_c = mem["seen_c"].at[:nc].set(mem["seen_c"][:nc] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld, "sel": mem["sel"] * DECAY, "seen_c": seen_c}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
