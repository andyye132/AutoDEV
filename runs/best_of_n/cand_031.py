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
MUS = (1.6, 2.0, 2.5, 3.0); N_MAX = 256
FLIPS = (2, 3, 4, 6, 8, 12, 16, 24)          # bang-bang / oscillator control periods
FAM_PAT = (0, 1, 0, 1, 2, 3, 0, 1, 2, 3)     # 30% hold, 30% bang-bang, 20% oscillate, 20% sweep
H = 192                                       # max steps between splits
Q = 0.5                                       # AMS level = this quantile of the pool
LVL_EMA = 0.2
P_RESTORE_MIN = 0.02                          # early phase: restores rare (the score is time-discounted)
P_RESTORE_MAX = 0.12
WORLD = 5.0
RESERVE_FRAC = 0.15                           # workers that never restore
NW = 4                                        # boundary walls are the first polygons
NB = 128                                      # max bodies tracked
NCON = 4096                                   # max contact pairs tracked
S_STAT = 6                                    # per-static-shape proximity coordinates
S_REACH = 6                                   # per-shape green-reach coordinates
NH = 4 + S_STAT + S_REACH                     # 0 escape, 1 energy, 2 disp, 3 new-contacts, then prox, then reach
BDEC = 0.999                                  # bandit / yield decay
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    cand = []
    for p in range(4, 41):
        for c in range(0, 41):
            f = 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2
            if f <= F: cand.append((F - f, p, c))
    cand.sort()
    _, Np, Nc = cand[0] if cand else (0, 4, 0)          # exact fit first; else nearest fit that stays in bounds
    C = 11 + 7 * Np
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
def _pick(mask, S, N):
    """body indices of the first S set entries of a (N,) mask (padded/clamped so the shape is always S)."""
    m = jnp.concatenate([mask, jnp.zeros((S,), bool)])
    order = jnp.argsort(jnp.where(m, 0, 1).astype(jnp.int32))[:S]
    return jnp.minimum(order, N - 1), m[order]
def _phi(feats, mem):
    """importance vector (B, NH) + which coordinates are live on this level."""
    pos, spd, act, role, L = _bodies(feats); B, N = spd.shape; idx = jnp.arange(N)
    dyn = mem["dyn"][:N]; dynm = dyn[None, :] & act                        # (B, N) has-moved & active
    dist_wall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)               # Chebyshev distance from the arena centre (2.5 = at a wall)
    escape = jnp.max(jnp.where(dynm, dist_wall, 0.0), axis=1) / (0.5 * WORLD)
    energy = jnp.max(jnp.where(dynm, spd, 0.0), axis=1) ** 2 / 64.0
    disp = jnp.max(jnp.where(dynm, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), axis=1) / WORLD
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)   # (B, N, N)
    stat_m = mem["seen"][:N] & ~dyn & (idx >= NW)                          # static non-wall shapes: tunnel/contact targets
    tgt_m = mem["seen"][:N] & (idx >= NW) & ~mem["grn"][:N]
    si, sok = _pick(stat_m, S_STAT, N); ti, tok = _pick(tgt_m, S_REACH, N)
    prox = -jnp.min(jnp.where(dynm[:, :, None], d, 9.0), axis=1)           # (B, N) per shape: nearest moving body
    prox = jnp.where(sok[None, :], prox[:, si], -9.0) / WORLD
    green = mem["grn"][:N][None, :] & act
    reach = -jnp.min(jnp.where(green[:, :, None], d, 9.0), axis=1)         # (B, N) green body to each shape
    reach = jnp.where(tok[None, :], reach[:, ti], -9.0) / WORLD
    contacts = feats[:, L["cfrom"]:] > 0.5; nc = min(contacts.shape[1], NCON)
    newc = jnp.sum(contacts[:, :nc] & ~mem["seen_c"][None, :nc], axis=1).astype(jnp.float32)
    phi = jnp.concatenate([escape[:, None], energy[:, None], disp[:, None], newc[:, None], prox, reach], 1)
    live = jnp.concatenate([jnp.ones((4,), bool), sok, tok])
    return phi, live, contacts[:, :nc]
def _pools(live, K, P):
    """map the K slots onto the live coordinates (compact, >=1 slot each); own[c] = pool c holds slots."""
    ordr = jnp.argsort(jnp.where(live, 0, 1).astype(jnp.int32))            # live coordinate ids first, priority order
    nl = jnp.clip(jnp.sum(live.astype(jnp.int32)), 1, jnp.maximum(P, 1))
    slot_pool = ordr[jnp.mod(jnp.arange(K), nl)]
    return slot_pool, jnp.zeros((NH,), bool).at[slot_pool].set(True)
def _guide(phi, pool):
    """per-worker drive: own hypothesis + a general escape/energy/displacement term (dense even when phi_c is flat)."""
    gen = phi[:, 0] + phi[:, 1] + 0.5 * phi[:, 2]
    return jnp.take_along_axis(phi, pool[:, None], 1)[:, 0] + gen
def init(rng, F, n_act, K, B):
    A = int(n_act); P = max(1, min(NH, K // 2)); nf = len(FAM_PAT); w = jnp.arange(B)
    return {"n_act": jnp.int32(A), "P": jnp.int32(P),
            "cur": (w % A).astype(jnp.int32), "last_a": (w % A).astype(jnp.int32),          # t=0: the batch holds every primitive action
            "left": (4 + (w % 16) * 8).astype(jnp.int32), "sign": jnp.ones(B, jnp.int32),
            "flip": jnp.array(FLIPS, jnp.int32)[(w // 2) % len(FLIPS)],
            "stride": (1 + (w % max(1, A - 1))).astype(jnp.int32), "fcnt": jnp.zeros(B, jnp.int32),
            "gref": jnp.zeros(B), "gprev": jnp.zeros(B),
            "mu": jnp.array(MUS)[(w // nf) % len(MUS)], "fam": jnp.array(FAM_PAT, jnp.int32)[w % nf],
            "wins": jnp.zeros(A), "cnt": jnp.zeros(A),
            "t": jnp.int32(0), "p0": jnp.zeros((B, NB, 2)), "dyn": jnp.zeros(NB, bool), "seen": jnp.zeros(NB, bool),
            "grn": jnp.zeros(NB, bool), "seen_c": jnp.zeros(NCON, bool), "have_start": jnp.bool_(False),
            "pool": (w % 4).astype(jnp.int32), "age": jnp.zeros(B, jnp.int32), "slot_phi": jnp.full(K, -jnp.inf),
            "live": jnp.concatenate([jnp.ones((4,), bool), jnp.zeros((NH - 4,), bool)]),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    phi, _, _ = _phi(feats, mem)
    slot_pool, own = _pools(mem["live"], K, mem["P"])
    my = jnp.take_along_axis(phi, mem["pool"][:, None], 1)[:, 0]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    below = my < mem["level"][mem["pool"]]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (slot_pool[None, :] == mem["pool"][:, None]) & arch_valid[None, :]                    # (B, K)
    elite = (mem["slot_phi"] > mem["level"][slot_pool]).astype(jnp.float32)                          # AMS: split onto members above the level
    slot = jnp.argmax(jnp.where(pool_ok, 1.5 * elite[None, :] + jax.random.gumbel(k1, (B, K)), -jnp.inf), axis=1)
    do = ((below & (jax.random.uniform(k2, (B,)) < P_RESTORE_MIN + P_RESTORE_MAX * tau)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    # pool bandit over LIVE, slot-owning pools: weight ∝ (1 + yield) / (1 + selections)
    w = jnp.where(own, (1.0 + mem["yield"]) / (1.0 + sel), 0.0); w = w / jnp.maximum(jnp.sum(w), 1e-9)
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, mem["pool"])
    return do, slot, {**mem, "pool": pool, "age": jnp.where(do, 0, mem["age"]), "sel": sel,
                      "src": jnp.where(do, slot, mem["src"]), "left": jnp.where(do, 0, mem["left"]),
                      "fcnt": jnp.where(do, 0, mem["fcnt"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng, 2); A = mem["wins"].shape[0]
    new = mem["left"] <= 0
    p = (1.0 + mem["wins"]) / (2.0 + mem["cnt"]); p = p / jnp.sum(p)          # Beta-mean bandit over primitives
    samp = jax.random.choice(k1, A, (B,), p=p)
    swept = jnp.mod(mem["cur"] + mem["stride"], A)                            # family 3: systematic coprime sweep
    base = jnp.where(mem["fam"] == 3, swept, samp)
    cur = jnp.where(new, base, mem["cur"])                                    # family 1's cur is steered by the bang-bang loop in archive
    hold = _hold(mem, k2, B); left = jnp.where(new, hold, mem["left"]) - 1
    osc = mem["fam"] == 2; tick = jnp.mod(jnp.maximum(left, 0), jnp.maximum(mem["flip"], 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    action = jnp.clip(jnp.where(osc, (cur // 2) * 2 + (sign < 0).astype(jnp.int32), cur), 0, A - 1).astype(jnp.int32)
    return action, {**mem, "cur": cur, "last_a": action, "left": left, "sign": sign, "t": mem["t"] + 1,
                    "age": mem["age"] + 1, "fcnt": jnp.where(new, 0, mem["fcnt"])}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; A = mem["wins"].shape[0]; ka, kb = jax.random.split(rng, 2)
    pos, spd, actm, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    seen = mem["seen"].at[:N].set(mem["seen"][:N] | actm.any(0))
    grn = mem["grn"].at[:N].set(mem["grn"][:N] | ((role == 1) & actm).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn, "seen": seen, "grn": grn, "p0": p0, "have_start": jnp.bool_(True)}
    phi, live_now, contacts = _phi(feats, mem2)
    live = mem["live"] | live_now                                             # monotone: the slot map stays stable
    slot_pool, own = _pools(live, K, mem["P"])
    # ratchet: per pool, the best worker replaces that pool's weakest slot if it beats it
    valid_phi = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_min = jnp.full(NH, jnp.inf).at[slot_pool].min(valid_phi)
    pool_min = jnp.where(own, jnp.where(jnp.isinf(pool_min), -jnp.inf, pool_min), jnp.inf)   # pools with no slots never store
    cand = (phi > pool_min[None, :]) & ~done[:, None] & live[None, :]
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0); has = cand.any(0); bw = jnp.where(has, best_w, 0)
    mine = slot_pool[None, :] == jnp.arange(NH)[:, None]
    pool_slot = jnp.argmin(jnp.where(mine, valid_phi[None, :], jnp.inf), axis=1)             # (NH,)
    do = jnp.zeros(B, jnp.int32).at[bw].max(has.astype(jnp.int32)) > 0                       # one store per pool, deduped by worker
    slot = jnp.zeros(B, jnp.int32).at[bw].max(jnp.where(has, pool_slot, 0))
    which = jnp.full(B, -1, jnp.int32).at[bw].max(jnp.where(has, jnp.arange(NH), -1))
    eff = has & (which[bw] == jnp.arange(NH)) & (slot[bw] == pool_slot)                      # only the pool that actually won the worker
    slot_phi = mem["slot_phi"].at[pool_slot].max(jnp.where(eff, phi[bw, jnp.arange(NH)], -jnp.inf))
    # AMS levels: EMA toward the pool's Q-quantile of member phi
    members = jnp.where(arch_valid[None, :] & mine, slot_phi[None, :], jnp.nan)
    q = jnp.nan_to_num(jnp.nanquantile(members, Q, axis=1), nan=-jnp.inf)
    level = jnp.where(jnp.isfinite(q), (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], q) + LVL_EMA * q, mem["level"])
    # bang-bang extremum seeker: keep the polarity that raises the drive, flip it when it stops paying
    g = _guide(phi, mem["pool"]); fresh = mem["fcnt"] == 0
    gref = jnp.where(fresh, g, mem["gref"]); due = (mem["fcnt"] >= mem["flip"]) & ~fresh
    stuck = (mem["fam"] == 1) & due & (g <= gref + 1e-5)
    alt = jnp.where(jax.random.uniform(ka, (B,)) < 0.75, jnp.bitwise_xor(mem["cur"], 1), jax.random.randint(kb, (B,), 0, A))
    cur = jnp.where(stuck, jnp.clip(alt, 0, A - 1), mem["cur"])
    fcnt = jnp.where(due, 0, mem["fcnt"] + 1); gref = jnp.where(due, g, gref)
    # primitive-action bandit: credit the executed action when the drive rose (no-ops and inert motors fade out)
    ok = (mem["age"] > 1).astype(jnp.float32)
    wins = mem["wins"] * BDEC + jnp.zeros(A).at[mem["last_a"]].add(ok * (g > mem["gprev"] + 1e-6).astype(jnp.float32))
    cnt = mem["cnt"] * BDEC + jnp.zeros(A).at[mem["last_a"]].add(ok)
    # credit: a store made by a worker restored from pool p counts as yield for p
    src_pool = jnp.where(mem["src"] >= 0, slot_pool[jnp.clip(mem["src"], 0, K - 1)], 0)
    yld = (mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))) * BDEC
    nc = contacts.shape[1]
    seen_c = mem["seen_c"].at[:nc].set(mem["seen_c"][:nc] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "live": live, "yield": yld, "sel": mem["sel"] * BDEC,
                      "seen_c": seen_c, "cur": cur, "fcnt": fcnt, "gref": gref, "gprev": g, "wins": wins, "cnt": cnt}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
