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
MUS = (1.3, 1.7, 2.2, 2.8)          # zeta exponents: portfolio of hold timescales (heavier tails than before)
PER = (2, 3, 4, 6, 8, 11, 16, 23, 32, 45)   # geometric ladder of oscillation periods = systematic frequency sweep
N_MAX = 256
FAM_P = (0.34, 0.26, 0.26, 0.14)    # motor / thruster / oscillate / sweep
H = 160                              # max steps between splits
IDLE_H = 40                          # steps of zero motion before declaring the action dead
Q = 0.5                              # AMS level = this quantile of the pool
LVL_EMA = 0.2
P_RESTORE_MIN = 0.02                 # early phase: restores rare (the score is time-discounted)
P_RESTORE_MAX = 0.14
WORLD = 5.0
RESERVE_FRAC = 0.15                  # workers that never restore
NW = 4                               # boundary walls are the first polygons
NB = 128                             # max bodies tracked
G = 8; SB = 3; NCELL = G * G * SB    # occupancy grid over position x speed-bucket
CW = 256                             # max contact pairs tracked
MARGIN = 0.004                       # per-step loosening of a stale pool's admission threshold
NH = 8                               # hypotheses / archive pools
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
    act_m = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), act_m, role, L
def _phi(feats, mem):
    """importance vector (B, NH): [escape, energy, displacement, tunnel-proximity, green-reach, new-contacts, new-cells, spread].
    All eight are target-blind and shape-index-free, so they mean the same thing on every level."""
    pos, spd, act_m, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act_m                                            # (B, N) moving & active
    dwall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)                    # Chebyshev distance from the arena centre
    escape = jnp.max(jnp.where(dyn, dwall, 0.0), 1) / (0.5 * WORLD)
    energy = jnp.max(jnp.where(dyn, spd, 0.0), 1) ** 2 / 64.0
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), 1) / WORLD
    stat = act_m & ~dyn & (jnp.arange(N)[None, :] >= NW)                    # static non-wall shapes: tunnel/contact targets
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)   # (B, N, N)
    prox = -jnp.min(jnp.where(dyn[:, :, None] & stat[:, None, :], d, 9.0), axis=(1, 2)) / WORLD
    green = (role == 1) & act_m
    reach = -jnp.min(jnp.where(green[:, :, None] & act_m[:, None, :] & ~green[:, None, :], d, 9.0), axis=(1, 2)) / WORLD
    spread = jnp.max(jnp.where(dyn[:, :, None] & dyn[:, None, :], d, 0.0), axis=(1, 2)) / WORLD
    cw = min(feats.shape[1] - L["cfrom"], CW)
    contacts = feats[:, L["cfrom"]:L["cfrom"] + cw] > 0.5
    newc = jnp.sum(contacts & ~mem["seen_c"][None, :cw], axis=1).astype(jnp.float32)
    px = jnp.clip((pos[..., 0] / WORLD * G).astype(jnp.int32), 0, G - 1)
    py = jnp.clip((pos[..., 1] / WORLD * G).astype(jnp.int32), 0, G - 1)
    sb = (spd > 1.0).astype(jnp.int32) + (spd > 4.0).astype(jnp.int32)      # coverage of velocity space, not just position
    cells = (py * G + px) + (G * G) * sb
    novel = jnp.sum((~mem["vis"][cells]) & dyn, axis=1).astype(jnp.float32)
    phi = jnp.stack([escape, energy, disp, prox, reach, newc, novel, spread], axis=1)
    return phi, contacts, cells, dyn
def init(rng, F, n_act, K, B):
    idx = jnp.arange(B)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32),
            "per": jnp.array(PER)[(idx * 3) % len(PER)],                     # stratified portfolio instead of iid draws
            "mu": jnp.array(MUS)[(idx // 4) % len(MUS)], "fam": (idx % 4).astype(jnp.int32),
            "t": jnp.int32(0), "idle": jnp.zeros(B, jnp.int32), "p0": jnp.zeros((B, NB, 2)), "dyn": jnp.zeros(NB, bool),
            "seen_c": jnp.zeros(CW, bool), "vis": jnp.zeros(NCELL, bool), "have_start": jnp.bool_(False),
            "pool": (idx % NH).astype(jnp.int32), "age": jnp.zeros(B, jnp.int32), "slot_pool": (jnp.arange(K) % NH).astype(jnp.int32),
            "slot_phi": jnp.full(K, -jnp.inf), "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH),
            "stale": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mu, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mu - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; ks = jax.random.split(rng, 8)
    phi, _, _, _ = _phi(feats, mem)
    my = phi[jnp.arange(B), mem["pool"]]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    below = my < mem["level"][mem["pool"]]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == mem["pool"][:, None]) & arch_valid[None, :]           # (B, K)
    sc = jax.random.gumbel(ks[0], (B, K)) - 0.5 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(pool_ok, sc, -jnp.inf), axis=1).astype(jnp.int32)
    p = P_RESTORE_MIN + (P_RESTORE_MAX - P_RESTORE_MIN) * tau
    do = ((below & (jax.random.uniform(ks[1], (B,)) < p)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    # pool bandit: re-assign a restoring worker to a pool with probability ~ (1 + yield) / (1 + sel)
    w = (1.0 + mem["yield"]) / (1.0 + sel); w = w / jnp.sum(w)
    newpool = jax.random.choice(ks[2], NH, (B,), p=w).astype(jnp.int32)
    pool = jnp.where(do & (jax.random.uniform(ks[3], (B,)) < 0.5), newpool, mem["pool"])
    # diversify on split: offspring must not re-run the parent's behaviour, or the split buys nothing
    fam = jnp.where(do, jax.random.choice(ks[4], 4, (B,), p=jnp.array(FAM_P)).astype(jnp.int32), mem["fam"])
    mu = jnp.where(do, jnp.array(MUS)[jax.random.randint(ks[5], (B,), 0, len(MUS))], mem["mu"])
    per = jnp.where(do, jnp.array(PER)[jax.random.randint(ks[6], (B,), 0, len(PER))], mem["per"])
    return do, slot, {**mem, "pool": pool, "age": jnp.where(do, 0, mem["age"]), "sel": sel, "src": jnp.where(do, slot, mem["src"]),
                      "left": jnp.where(do, 0, mem["left"]), "idle": jnp.where(do, 0, mem["idle"]), "sign": jnp.where(do, 1, mem["sign"]),
                      "fam": fam, "mu": mu, "per": per}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; ks = jax.random.split(rng, 6); n_act = mem["n_act"]; NM = 4
    _, spd, act_m, _, _ = _bodies(feats)
    mx = jnp.max(jnp.where(act_m, spd, 0.0), axis=1)
    idle = jnp.where(mx < 1e-2, mem["idle"] + 1, 0)
    stuck = idle >= IDLE_H                                   # the held action drives nothing: abandon it now, not in 256 steps
    fam = jnp.where(stuck, jax.random.randint(ks[0], (B,), 0, 4), mem["fam"])
    mu = jnp.where(stuck, jnp.array(MUS)[jax.random.randint(ks[1], (B,), 0, len(MUS))], mem["mu"])
    per = jnp.where(stuck, jnp.array(PER)[jax.random.randint(ks[2], (B,), 0, len(PER))], mem["per"])
    idle = jnp.where(stuck, 0, idle)
    new = (mem["left"] <= 0) | stuck
    motor = jax.random.randint(ks[3], (B,), 0, jnp.maximum(jnp.minimum(2 * NM, n_act), 1))
    nthr = jnp.maximum(n_act - (2 * NM + 1), 1)
    thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + 1 + jax.random.randint(ks[4], (B,), 0, nthr), motor)
    base = jnp.where(fam == 1, thr, motor)                   # families 2/3 build on a motor index
    cur = jnp.where(new, base, mem["cur"])
    hold = _hold(mu, ks[5], B); left = jnp.where(new, hold, mem["left"]) - 1
    osc = fam == 2; tick = (left % jnp.maximum(per, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    a_osc = (cur // 2) * 2 + (sign < 0).astype(jnp.int32)     # resonant pumping at a laddered period
    a_swp = (cur + (mem["t"] // 6)) % jnp.maximum(n_act, 1)   # systematic sweep of the whole action set
    action = jnp.where(osc, a_osc, jnp.where(fam == 3, a_swp, cur))
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0)).astype(jnp.int32)
    return action, {**mem, "cur": cur, "left": left, "sign": sign, "fam": fam, "mu": mu, "per": per, "idle": idle,
                    "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, act_m, role, L = _bodies(feats); N = spd.shape[1]
    dyn_g = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn_g, "p0": p0, "have_start": jnp.bool_(True)}
    phi, contacts, cells, dyn = _phi(feats, mem2)
    ispool = mem["slot_pool"][None, :] == jnp.arange(NH)[:, None]                         # (NH, K), pools own disjoint slots
    slot_score = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_min = jnp.min(jnp.where(ispool, slot_score[None, :], jnp.inf), axis=1)            # inf only if the pool owns no slot
    pool_slot = jnp.argmin(jnp.where(ispool, slot_score[None, :], jnp.inf), axis=1)
    thresh = pool_min - MARGIN * mem["stale"]                                              # a stale pool reopens; its elites are never touched
    # one store per pool per step, each by a DISTINCT worker, and slot_phi written only where a store really happened
    do = jnp.zeros(B, bool); slot = jnp.zeros(B, jnp.int32); avail = ~done
    slot_phi = mem["slot_phi"]; stale = mem["stale"] + 1.0
    for c in range(NH):
        sc = jnp.where(avail & (phi[:, c] > thresh[c]), phi[:, c], -jnp.inf)
        w = jnp.argmax(sc); ok = sc[w] > -jnp.inf
        do = do.at[w].set(do[w] | ok)
        slot = slot.at[w].set(jnp.where(ok, pool_slot[c], slot[w]))
        slot_phi = slot_phi.at[pool_slot[c]].set(jnp.where(ok, phi[w, c], slot_phi[pool_slot[c]]))
        avail = avail.at[w].set(avail[w] & ~ok)
        stale = stale.at[c].set(jnp.where(ok, 0.0, stale[c]))
    # AMS levels: EMA toward the pool's Q-quantile of member phi
    members = jnp.where(arch_valid[None, :] & ispool, slot_phi[None, :], jnp.nan)
    q = jnp.nan_to_num(jnp.nanquantile(members, Q, axis=1), nan=-jnp.inf, posinf=jnp.inf, neginf=-jnp.inf)
    level = jnp.where(jnp.isfinite(q), (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], q) + LVL_EMA * q, mem["level"])
    # credit: a store made by a worker that was restored from pool p counts as yield for p
    src_pool = jnp.where(mem["src"] >= 0, mem["slot_pool"][jnp.maximum(mem["src"], 0)], 0)
    yld = mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    cw = contacts.shape[1]
    seen_c = mem["seen_c"].at[:cw].set(mem["seen_c"][:cw] | contacts.any(0))
    hit = jnp.zeros(NCELL, jnp.int32).at[cells.reshape(-1)].max(dyn.reshape(-1).astype(jnp.int32))
    vis = mem["vis"] | (hit > 0)
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld, "seen_c": seen_c, "vis": vis, "stale": stale}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
