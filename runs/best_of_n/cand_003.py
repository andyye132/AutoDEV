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
MUS = (1.5, 2.0, 2.5, 3.0); N_MAX = 256
FAM_P = (0.5, 0.25, 0.25)          # motor / thruster / oscillate
H = 192                             # max steps between splits
Q = 0.5                             # AMS level = this quantile of the pool
LVL_EMA = 0.2
P_RESTORE_MIN = 0.02                # early phase: restores rare (the score is time-discounted)
WORLD = 5.0
RESERVE_FRAC = 0.15                 # workers that never restore
MAX_SHAPES = 16
NW = 4                              # boundary walls are the first polygons
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
def _seg_point_dist(a, b, p):
    """distance from point p (B,M,2) to segment [a,b] (B,N,2 each) -> (B,N,M)."""
    ab = b - a; ab2 = jnp.sum(ab * ab, axis=-1) + 1e-9                       # (B,N)
    diff = p[:, None, :, :] - a[:, :, None, :]                               # (B,N,M,2)
    t = jnp.clip(jnp.sum(diff * ab[:, :, None, :], axis=-1) / ab2[:, :, None], 0.0, 1.0)   # (B,N,M)
    closest = a[:, :, None, :] + t[..., None] * ab[:, :, None, :]            # (B,N,M,2)
    return jnp.linalg.norm(p[:, None, :, :] - closest, axis=-1)              # (B,N,M)
def _phi(feats, mem):
    """importance vector (B, NH): [escape, energy, displacement, proximity-to-shape j (MAX_SHAPES), reach-to-shape j (MAX_SHAPES), new-contacts]"""
    pos, spd, act, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act                                             # (B, N) moving & active
    dist_wall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)               # Chebyshev distance from the arena centre (2.5 = at a wall)
    escape = jnp.max(jnp.where(dyn, dist_wall, 0.0), axis=1) / (0.5 * WORLD)
    energy = jnp.max(jnp.where(dyn, spd, 0.0), axis=1) ** 2 / 64.0
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), axis=1) / WORLD
    stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)                     # static non-wall shapes: tunnel/contact targets
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)   # (B, N, N) point-to-point
    prev = jnp.where(mem["have_prev"], mem["prev_pos"][:, :N], pos)        # (B, N, 2): last step's position (or current, pre-warmup)
    dseg = _seg_point_dist(prev, pos, pos)                                  # (B, N, N): swept path of body i to point of body j
    deff = jnp.where(mem["have_prev"], jnp.minimum(d, dseg), d)             # segment distance is <= endpoint distance; guard warmup
    prox = -jnp.min(jnp.where(dyn[:, :, None] & stat[:, None, :], deff, 9.0), axis=1)   # (B, N) per target shape: nearest moving-body path
    green = (role == 1) & act
    reach = -jnp.min(jnp.where(green[:, :, None] & act[:, None, :] & ~green[:, None, :], d, 9.0), axis=1)   # (B, N) green to each other shape
    pad = lambda x: jnp.concatenate([x, jnp.full((B, MAX_SHAPES), -9.0)], 1)[:, :MAX_SHAPES]
    contacts = (feats[:, L["cfrom"]:] > 0.5)
    newc = jnp.sum(contacts & ~mem["seen_c"][None, : contacts.shape[1]], axis=1).astype(jnp.float32)
    return jnp.concatenate([escape[:, None], energy[:, None], disp[:, None], pad(prox) / WORLD, pad(reach) / WORLD, newc[:, None]], 1), contacts, spd, act
NH = 3 + 2 * MAX_SHAPES + 1
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32), "flip": jnp.zeros(B, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 3, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, 64, 2)), "dyn": jnp.zeros(64, bool), "seen_c": jnp.zeros(256, bool), "have_start": jnp.bool_(False),
            "prev_pos": jnp.zeros((B, 64, 2)), "have_prev": jnp.bool_(False),
            "pool": jnp.arange(B) % NH, "age": jnp.zeros(B, jnp.int32), "slot_pool": jnp.arange(K) % NH, "slot_phi": jnp.full(K, -jnp.inf),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3 = jax.random.split(rng, 3)
    phi, contacts, spd, act = _phi(feats, mem)
    my = phi[jnp.arange(B), mem["pool"]]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    below = my < mem["level"][mem["pool"]]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == mem["pool"][:, None]) & arch_valid[None, :]           # (B, K)
    slot = jnp.argmax(jnp.where(pool_ok, jax.random.gumbel(k1, (B, K)), -jnp.inf), axis=1)
    do = ((below & (jax.random.uniform(k2, (B,)) < P_RESTORE_MIN + 0.1 * tau)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    # pool bandit: re-assign a restoring worker to a pool with probability ∝ (1 + yield) / (1 + sel)
    w = (1.0 + mem["yield"]) / (1.0 + sel); w = w / w.sum()
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k1, (B,)) < 0.5), newpool, mem["pool"])
    age = jnp.where(do, 0, mem["age"]); src = jnp.where(do, slot, mem["src"])
    return do, slot, {**mem, "pool": pool, "age": age, "sel": sel, "src": src, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    motor = jax.random.randint(k1, (B,), 0, 2 * NM); thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + 1 + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - 2 * NM - 1, 1)), motor)
    fam = mem["fam"]
    base = jnp.where(fam == 1, thr, motor)                    # family 2 (oscillate) uses a motor and flips its sign
    cur = jnp.where(new, base, mem["cur"])
    hold = _hold(mem, k3, B); left = jnp.where(new, hold, mem["left"]) - 1
    flip = jnp.where(new, jnp.clip(hold // 4, 2, 40), mem["flip"])
    osc = (fam == 2); tick = (mem["left"] % jnp.maximum(flip, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    action = jnp.where(osc, (cur // 2) * 2 + (sign < 0).astype(jnp.int32), cur)
    return action, {**mem, "cur": cur, "left": left, "flip": flip, "sign": sign, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, act, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True)}
    phi, contacts, _, _ = _phi(feats, mem2)
    # ratchet: for each pool, find the best NOT-YET-USED worker that beats the pool's current minimum, resolving
    # collisions (two pools wanting the same worker) greedily so more pools get to insert this step.
    pool_min_val = jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(jnp.where(arch_valid, mem["slot_phi"], -jnp.inf))
    pool_min_val = jnp.where(jnp.isinf(pool_min_val), -jnp.inf, pool_min_val)            # empty pool: anything qualifies
    cand0 = (phi > pool_min_val[None, :]) & ~done[:, None]                                # (B, NH)
    # priority order: pools whose best candidate has the largest margin over the pool minimum go first
    margin = jnp.max(jnp.where(cand0, phi - pool_min_val[None, :], -jnp.inf), axis=0)     # (NH,)
    order = jnp.argsort(-jnp.nan_to_num(margin, neginf=-1e9))                              # (NH,) pool indices, best margin first
    def _step(used, c):
        p = order[c]
        avail = cand0[:, p] & ~used
        w = jnp.argmax(jnp.where(avail, phi[:, p], -jnp.inf))
        h = avail.any()
        used2 = used | (jnp.arange(B) == w) & h
        return used2, (p, w, h)
    used0 = jnp.zeros(B, bool)
    used_f, (ps, ws, hs) = jax.lax.scan(_step, used0, jnp.arange(NH))
    has = jnp.zeros(NH, bool).at[ps].set(hs)
    best_w = jnp.zeros(NH, jnp.int32).at[ps].set(ws)
    # slot: the pool's current minimum (or any empty slot of the pool)
    slot_score = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_slot = jnp.argmin(jnp.where(mem["slot_pool"][None, :] == jnp.arange(NH)[:, None], slot_score[None, :], jnp.inf), axis=1)   # (NH,)
    do = jnp.zeros(B, bool).at[best_w].max(has)
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(has, pool_slot, 0))
    slot_phi = mem["slot_phi"].at[jnp.where(has, pool_slot, 0)].set(jnp.where(has, phi[best_w, jnp.arange(NH)], mem["slot_phi"][0]))
    # AMS levels: EMA toward the pool's Q-quantile of member phi
    members = jnp.where(arch_valid[None, :] & (mem["slot_pool"][None, :] == jnp.arange(NH)[:, None]), slot_phi[None, :], jnp.nan)
    q = jnp.nan_to_num(jnp.nanquantile(members, Q, axis=1), nan=-jnp.inf)
    level = jnp.where(jnp.isfinite(q), (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], q) + LVL_EMA * q, mem["level"])
    # credit: a store made by a worker that was restored from pool p counts as yield for p
    src_pool = jnp.where(mem["src"] >= 0, mem["slot_pool"][jnp.maximum(mem["src"], 0)], 0)
    yld = mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    seen_c = mem["seen_c"].at[: contacts.shape[1]].set(mem["seen_c"][: contacts.shape[1]] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld, "seen_c": seen_c, "prev_pos": mem["prev_pos"].at[:, :N].set(pos), "have_prev": jnp.bool_(True)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
