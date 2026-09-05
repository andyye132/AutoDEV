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
FAM_P = (0.40, 0.25, 0.35)          # motor / thruster / oscillate
H = 128                             # max steps between splits
LVL_EMA = 0.2
P_RES0, P_RES1 = 0.05, 0.15         # restore prob = P_RES0 + P_RES1 * tau (time-discounted score => rare early)
WORLD = 5.0
RESERVE_FRAC = 0.125                # workers that never restore
MAX_SHAPES = 16
NW = 4                              # boundary walls are the first polygons
NG = 5                              # global coords: escape, max-KE, total-KE, displacement, contacts
NH = NG + 2 * MAX_SHAPES            # + per-shape impact pressure, + per-shape green reach
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
def _phi(feats, mem):
    """importance vector (B, NH): [escape, max-KE, total-KE, displacement, contacts, pressure_j (MAX_SHAPES), reach_j (MAX_SHAPES)].

    pressure_j = max over MOVING bodies i != j of (1 + v_i/2) / (1 + d_ij)^2 -- an impact intensity. Unlike a bare
    proximity it does NOT saturate on contact: it keeps rising as the approach gets faster, which is exactly the
    pass-through-a-solid condition, and it stays defined when shape j itself starts moving."""
    pos, spd, act, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act                                             # (B, N) moving & active
    dist_wall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)               # Chebyshev distance from the arena centre (2.5 = at a wall)
    escape = jnp.max(jnp.where(dyn, dist_wall, 0.0), axis=1) / (0.5 * WORLD)
    ke = jnp.where(dyn, spd, 0.0) ** 2
    maxke = jnp.max(ke, axis=1) / 64.0
    totke = jnp.sum(ke, axis=1) / 256.0                                    # runaway physics: many bodies hot at once
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), axis=1) / WORLD
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (B, N, N) source i -> target j
    noself = ~jnp.eye(N, dtype=bool)[None]
    tgt = act & (jnp.arange(N)[None, :] >= NW)                             # non-wall shapes are the per-shape hypotheses
    green = (role == 1) & act
    inv = 1.0 / (1.0 + d) ** 2
    press = jnp.max(jnp.where(dyn[:, :, None] & tgt[:, None, :] & noself, (1.0 + 0.5 * spd)[:, :, None] * inv, 0.0), axis=1) / 2.0
    rmask = green[:, :, None] & act[:, None, :] & ~green[:, None, :] & noself
    reach = jnp.max(jnp.where(rmask, (1.0 + 0.25 * spd)[:, :, None] * inv, 0.0), axis=1) / 2.0
    pad = lambda x: jnp.concatenate([x, jnp.zeros((B, MAX_SHAPES))], 1)[:, :MAX_SHAPES]
    contacts = feats[:, L["cfrom"]:] > 0.5
    nc = contacts.shape[1]
    newc = jnp.sum(contacts & ~mem["seen_c"][None, :nc], axis=1).astype(jnp.float32)
    con = (jnp.sum(contacts, axis=1).astype(jnp.float32) + 2.0 * newc) / 8.0   # contact-rich states, bonus for novel pairs
    return jnp.concatenate([escape[:, None], maxke[:, None], totke[:, None], disp[:, None], con[:, None], pad(press), pad(reach)], 1), contacts, tgt, green
def _live(tgt, green):
    """(NH,) which hypotheses are realisable on THIS level. Padded / absent shapes get no archive capacity."""
    ta = tgt.any(0); ga = green.any(0)
    padb = lambda x: jnp.concatenate([x, jnp.zeros(MAX_SHAPES, bool)])[:MAX_SHAPES]
    return jnp.concatenate([jnp.ones(NG, bool), padb(ta), padb(ta & ~ga) & jnp.any(ga)])
def _alloc(live, K):
    """Deal the K slots over the live pools, weight 3 for the global coords and 1 per shape coord."""
    w = jnp.where(live, jnp.where(jnp.arange(NH) < NG, 3.0, 1.0), 0.0)
    cum = jnp.cumsum(w); tot = jnp.maximum(cum[-1], 1e-6)
    p = (jnp.arange(K) + 0.5) * tot / K
    return jnp.clip(jnp.searchsorted(cum, p, side="left"), 0, NH - 1).astype(jnp.int32)
def _period(k, B):
    return jnp.floor(jnp.exp(jax.random.uniform(k, (B,), minval=0.69, maxval=3.87))).astype(jnp.int32)   # log-uniform in [2, 48]
def init(rng, F, n_act, K, B):
    k1, k2, k3 = jax.random.split(rng, 3)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "ph": jnp.zeros(B, jnp.int32),
            "per": _period(k3, B), "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 3, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, 64, 2)), "dyn": jnp.zeros(64, bool), "seen_c": jnp.zeros(256, bool), "have_start": jnp.bool_(False),
            "pool": (jnp.arange(B) % NH).astype(jnp.int32), "age": jnp.zeros(B, jnp.int32), "slot_pool": (jnp.arange(K) % NG).astype(jnp.int32),
            "slot_phi": jnp.full(K, -jnp.inf), "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 7)
    phi, contacts, tgt, green = _phi(feats, mem)
    sp = mem["slot_pool"]
    ok = jnp.zeros(NH, bool).at[sp].set(True)                                   # pools that actually own capacity
    order = jnp.argsort(jnp.where(ok, 0, 1)); n_ok = jnp.maximum(ok.sum(), 1)
    pool = jnp.where(ok[mem["pool"]], mem["pool"], order[jnp.arange(B) % n_ok])  # workers on dead pools fold onto live ones
    my = phi[jnp.arange(B), pool]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    pool_ok = (sp[None, :] == pool[:, None]) & arch_valid[None, :]               # (B, K)
    above = mem["slot_phi"] >= mem["level"][sp]                                  # AMS: split onto a member above the level
    score = jax.random.gumbel(k1, (B, K)) + above[None, :].astype(jnp.float32) - 0.4 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(pool_ok, score, -jnp.inf), axis=1)
    below = my < mem["level"][pool]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    do = ((below & (jax.random.uniform(k2, (B,)) < P_RES0 + P_RES1 * tau)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, pool, 0)].add(do.astype(jnp.float32))
    w = jnp.where(ok, (1.0 + mem["yield"]) / (1.0 + sel), 0.0); w = w / jnp.maximum(w.sum(), 1e-9)
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, pool)
    r = do & (jax.random.uniform(k5, (B,)) < 0.5)                                # a split also diversifies the POLICY, not just the noise
    fam = jnp.where(r, jax.random.choice(k6, 3, (B,), p=jnp.array(FAM_P)), mem["fam"])
    per = jnp.where(r, _period(k7, B), mem["per"])
    mu = jnp.where(r, jnp.array(MUS)[jax.random.randint(k6, (B,), 0, len(MUS))], mem["mu"])
    return do, slot, {**mem, "pool": pool, "age": jnp.where(do, 0, mem["age"]), "sel": sel, "src": jnp.where(do, slot, mem["src"]),
                      "left": jnp.where(do, 0, mem["left"]), "fam": fam, "per": per, "mu": mu}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3 = jax.random.split(rng, 3); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    motor = jax.random.randint(k1, (B,), 0, jnp.minimum(2 * NM, jnp.maximum(n_act, 1)))
    thr = jnp.where(n_act > 2 * NM + 1, (2 * NM + 1) + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - 2 * NM - 1, 1)), motor)
    fam = mem["fam"]; osc = fam == 2
    base = jnp.where(fam == 1, thr, motor)                    # family 2 (oscillate) drives a motor and flips its sign
    cur = jnp.where(new, base, mem["cur"])
    hold = _hold(mem, k3, B)
    hold = jnp.where(osc, jnp.clip(hold * 4, 16, N_MAX), hold)   # pumping needs many cycles on the SAME motor to resonate
    left = jnp.where(new, hold, mem["left"]) - 1
    ph = mem["ph"] + 1                                           # free-running phase: a fixed per-worker square wave, so the
    neg = ((ph // jnp.maximum(mem["per"], 1)) % 2) == 1          # batch sweeps drive frequencies instead of re-rolling each hold
    action = jnp.where(osc, (cur // 2) * 2 + neg.astype(jnp.int32), cur)
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0))
    return action, {**mem, "cur": cur, "left": left, "ph": ph, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, act, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn, "p0": p0}
    phi, contacts, tgt, green = _phi(feats, mem2)
    slot_pool = jnp.where(mem["have_start"], mem["slot_pool"], _alloc(_live(tgt, green), K))   # capacity dealt once, to live pools only
    prev = jnp.where(mem["have_start"], mem["slot_phi"], jnp.full(K, -jnp.inf))
    in_pool = slot_pool[None, :] == jnp.arange(NH)[:, None]                                   # (NH, K)
    has_slot = in_pool.any(1)
    sc = jnp.where(in_pool, jnp.where(arch_valid, prev, -jnp.inf)[None, :], jnp.inf)
    pool_min = jnp.min(sc, axis=1); pool_slot = jnp.argmin(sc, axis=1)
    thresh = jnp.where(has_slot, jnp.where(jnp.isinf(pool_min) & (pool_min > 0), -jnp.inf, pool_min), jnp.inf)
    cand = (phi > thresh[None, :]) & ~done[:, None]                                           # (B, NH) ratchet per hypothesis
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0); has = cand.any(0)
    # one store per worker per step: a worker chosen by several pools commits to exactly one, so slot_phi never claims a store that did not happen
    which = jnp.full(B, -1, jnp.int32).at[best_w].max(jnp.where(has, jnp.arange(NH), -1).astype(jnp.int32))
    do = which >= 0; wi = jnp.clip(which, 0, NH - 1); slot = pool_slot[wi]
    stored = jnp.zeros(NH, bool).at[wi].max(do)
    ts = jnp.where(stored, pool_slot, 0); val = jnp.where(stored, phi[best_w, jnp.arange(NH)], -jnp.inf)
    newv = jnp.full(K, -jnp.inf).at[ts].max(val); wrote = jnp.zeros(K, bool).at[ts].max(stored)
    slot_phi = jnp.where(wrote, newv, prev)
    # AMS levels: EMA toward the mean phi of each pool's members (now several per pool, so the level is meaningful)
    memb = in_pool & arch_valid[None, :] & jnp.isfinite(slot_phi)[None, :]
    cnt = memb.sum(1)
    q = jnp.sum(jnp.where(memb, slot_phi[None, :], 0.0), axis=1) / jnp.maximum(cnt, 1)
    level = jnp.where(cnt > 0, (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], q) + LVL_EMA * q, mem["level"])
    src_pool = jnp.where(mem["src"] >= 0, slot_pool[jnp.clip(mem["src"], 0, K - 1)], 0)
    yld = mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    seen_c = mem["seen_c"].at[:contacts.shape[1]].set(mem["seen_c"][:contacts.shape[1]] | contacts.any(0))
    return do, slot, {**mem2, "slot_pool": slot_pool, "slot_phi": slot_phi, "level": level, "yield": yld, "seen_c": seen_c, "have_start": jnp.bool_(True)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
