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
FAM_PAT = (0, 1, 2, 3, 0, 1, 2, 0)   # stratified families: 3/8 motor-hold, 2/8 thruster, 2/8 oscillate, 1/8 sweep
H = 160                              # max steps between splits
Q = 0.5                              # AMS level = this quantile of the pool
LVL_EMA = 0.2
P_RES0 = 0.03; P_RES1 = 0.12         # restore prob = P_RES0 + P_RES1 * tau (rare early: score is time-discounted)
WORLD = 5.0
RESERVE_FRAC = 0.2                   # workers that never restore (pure-exploration backbone)
KEEP_P = 0.85                        # prob of keeping an action that improved this worker's hypothesis
MAX_S = 128
NW = 4                               # boundary walls are the first polygons
NPROJ = 4                            # random-projection hypotheses (target-blind extremes)
NH = 11 + NPROJ
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pang=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np,
                     cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _P(K):
    return max(1, min(NH, K))        # one pool per hypothesis, but never more pools than archive slots
def _bodies(feats):
    """positions (B, N, 2), velocities (B, N, 2), speeds, active, role for all shapes; polygons first (walls are 0..NW-1)."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, vel, jnp.linalg.norm(vel, axis=-1), act, role, L
def _phi(feats, mem):
    """importance vector (B, NH): [+x, -x, +y, -y escape, energy, tunnel, green-goal, hazard, displacement, spin, contacts, 4 random projections]"""
    pos, vel, spd, act, role, L = _bodies(feats); B, N = spd.shape; nb = min(N, MAX_S)
    ever = mem["dyn"][:nb]
    if N > nb: ever = jnp.concatenate([ever, jnp.ones(N - nb, bool)])
    dyn = act & (ever[None, :] | (spd > 1e-3))                             # moving now or ever seen moving
    stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)                     # static non-wall shapes: tunnel/contact targets
    rel = pos - 0.5 * WORLD
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :] + 1e-9, axis=-1)   # (B, N, N)
    m3 = dyn[:, :, None]
    extp = jnp.max(jnp.where(m3, rel, -9.0), axis=1, initial=-9.0) / WORLD          # signed wall reach, +x/+y
    extn = jnp.max(jnp.where(m3, -rel, -9.0), axis=1, initial=-9.0) / WORLD         # signed wall reach, -x/-y
    vmax = jnp.max(jnp.where(dyn, spd, 0.0), axis=1, initial=0.0)
    energy = vmax * vmax / 64.0
    tun = jnp.max(jnp.where(dyn[:, :, None] & stat[:, None, :], spd[:, :, None] / 8.0 - d / WORLD, -9.0),
                  axis=(1, 2), initial=-9.0)                                        # fast AND close: the tunneling precursor
    r1 = act & (role > 0.5) & (role < 1.5); r2 = act & (role > 1.5) & (role < 2.5); r3 = act & (role > 2.5)
    near = lambda a, b: -jnp.min(jnp.where(a[:, :, None] & b[:, None, :], d, 9.0), axis=(1, 2), initial=9.0) / WORLD
    dsp = jnp.linalg.norm(pos[:, :nb] - mem["p0"][:, :nb], axis=-1)
    if N > nb: dsp = jnp.concatenate([dsp, jnp.zeros((B, N - nb))], 1)
    disp = jnp.max(jnp.where(dyn, dsp, 0.0), axis=1, initial=0.0) / WORLD
    Np = L["Np"]; om = feats[:, L["pang"]:L["pang"] + Np]
    spin = jnp.max(jnp.abs(om) * act[:, :Np], axis=1, initial=0.0) / 8.0             # runaway rotation
    contacts = feats[:, L["cfrom"]:] > 0.5; nc = contacts.shape[1]
    newc = jnp.sum(contacts & ~mem["seen_c"][None, :nc], axis=1).astype(jnp.float32)
    con = 3.0 * newc + jnp.sum(contacts, axis=1).astype(jnp.float32)
    x = jnp.where(act[:, :, None], jnp.concatenate([rel / WORLD, vel / 8.0], -1), 0.0)
    proj = jnp.einsum("bnf,pnf->bp", x, mem["W"][:, :N, :])                          # target-blind extremes in random state directions
    phi = jnp.stack([extp[:, 0], extn[:, 0], extp[:, 1], extn[:, 1], energy, tun,
                     near(r1, r2), near(r1 | r2, r3), disp, spin, con], axis=1)
    return jnp.concatenate([phi, proj], axis=1), contacts
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng); L = _layout(F); ncf = max(F - L["cfrom"], 1); P = _P(K); idx = jnp.arange(B)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32), "flip": jnp.zeros(B, jnp.int32),
            "mu": jnp.array(MUS)[(idx // len(FAM_PAT)) % len(MUS)], "fam": jnp.array(FAM_PAT)[idx % len(FAM_PAT)],
            "t": jnp.int32(0), "p0": jnp.zeros((B, MAX_S, 2)), "dyn": jnp.zeros(MAX_S, bool), "seen_c": jnp.zeros(ncf, bool), "have_start": jnp.bool_(False),
            "W": jax.random.normal(k1, (NPROJ, MAX_S, 4)) * (1.0 + jax.random.uniform(k2, (NPROJ, 1, 1))),
            "pool": idx % P, "age": jnp.zeros(B, jnp.int32), "slot_pool": jnp.arange(K) % P, "slot_phi": jnp.full(K, -jnp.inf),
            "level": jnp.full(P, -jnp.inf), "yield": jnp.zeros(P), "sel": jnp.zeros(P), "src": jnp.full(B, -1, jnp.int32),
            "my0": jnp.full(B, jnp.inf), "keep": jnp.zeros(B, bool)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; P = _P(K); k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    phi, _ = _phi(feats, mem)
    my = phi[jnp.arange(B), mem["pool"]]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    below = my < mem["level"][mem["pool"]]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == mem["pool"][:, None]) & arch_valid[None, :]           # (B, K)
    sp = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    rank = jnp.mean((sp[None, :] < sp[:, None]).astype(jnp.float32), axis=1)                      # scale-free, monotone within a pool
    vis = jnp.ravel(jnp.asarray(arch_visits)).astype(jnp.float32)[:K]
    key = 1.5 * rank[None, :] - 0.6 * jnp.log1p(vis)[None, :] + jax.random.gumbel(k1, (B, K))     # prefer high-phi, under-visited members
    slot = jnp.argmax(jnp.where(pool_ok, key, -jnp.inf), axis=1)
    do = ((below & (jax.random.uniform(k2, (B,)) < P_RES0 + P_RES1 * tau)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    # pool bandit: re-assign a restoring worker to a pool with probability ∝ (1 + yield) / (1 + sel)
    w = (1.0 + mem["yield"]) / (1.0 + sel); w = w / jnp.sum(w)
    newpool = jax.random.choice(k3, P, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, mem["pool"])
    # (1+1) hill-climb: at a hold boundary, keep the action that raised this worker's own hypothesis coordinate
    at_b = mem["left"] <= 0
    keep = at_b & (my > mem["my0"]) & (jax.random.uniform(k5, (B,)) < KEEP_P) & ~do
    my0 = jnp.where(do, jnp.inf, jnp.where(at_b, my, mem["my0"]))                                  # inf after a restore: first boundary resamples
    age = jnp.where(do, 0, mem["age"]); src = jnp.where(do, slot, mem["src"])
    return do, slot, {**mem, "pool": pool, "age": age, "sel": sel, "src": src, "keep": keep, "my0": my0, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3 = jax.random.split(rng, 3); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    motor = jax.random.randint(k1, (B,), 0, 2 * NM)
    thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + 1 + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - 2 * NM - 1, 1)), motor)
    fam = mem["fam"]
    base = jnp.where(fam == 1, thr, motor)                    # families 2/3 (oscillate, sweep) build on a motor action
    cur = jnp.where(new & ~mem["keep"], base, mem["cur"])      # keep == the last hold improved my hypothesis: redraw duration only
    hold = _hold(mem, k3, B); left = jnp.where(new, hold, mem["left"]) - 1
    flip = jnp.where(new, jnp.clip(hold // 4, 2, 40), mem["flip"])
    osc = (fam == 2); tick = (mem["left"] % jnp.maximum(flip, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    a_osc = (cur // 2) * 2 + (sign < 0).astype(jnp.int32)      # resonant pumping: motor sign flips on a heavy-tailed period
    a_swp = cur + mem["age"] // jnp.maximum(flip, 1)           # systematic sweep through the action space
    action = jnp.where(osc, a_osc, jnp.where(fam == 3, a_swp, cur))
    action = jnp.remainder(action, jnp.maximum(n_act, 1)).astype(jnp.int32)
    return action, {**mem, "cur": cur, "left": left, "flip": flip, "sign": sign, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; P = _P(K)
    pos, vel, spd, act, role, L = _bodies(feats); N = spd.shape[1]; nb = min(N, MAX_S)
    dyn = mem["dyn"].at[:nb].set(mem["dyn"][:nb] | (spd[:, :nb] > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :nb].set(pos[:, :nb]))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True)}
    phi_full, contacts = _phi(feats, mem2); phi = phi_full[:, :P]
    # ratchet: per pool, the best worker replaces the pool's weakest slot if it beats it
    valid_phi = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_min = jnp.full(P, jnp.inf).at[mem["slot_pool"]].min(valid_phi)
    pool_min = jnp.where(jnp.isposinf(pool_min), -jnp.inf, pool_min)                      # empty pool: anything qualifies
    cand = (phi > pool_min[None, :] + 1e-4) & ~done[:, None]                              # (B, P)
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0)                           # (P,) best worker per pool
    has = cand.any(0)
    pool_slot = jnp.argmin(jnp.where(mem["slot_pool"][None, :] == jnp.arange(P)[:, None], valid_phi[None, :], jnp.inf), axis=1)
    do = jnp.zeros(B, bool).at[best_w].max(has)
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(has, pool_slot, 0))
    won = has & (slot[best_w] == pool_slot)                                               # a worker claimed by 2 pools writes only one slot
    slot_phi = mem["slot_phi"].at[pool_slot].set(jnp.where(won, phi[best_w, jnp.arange(P)], mem["slot_phi"][pool_slot]))
    # AMS levels: EMA toward each pool's Q-quantile of member phi
    members = jnp.where(arch_valid[None, :] & (mem["slot_pool"][None, :] == jnp.arange(P)[:, None]), slot_phi[None, :], jnp.nan)
    q = jnp.nanquantile(members, Q, axis=1); ok = ~jnp.isnan(q); qs = jnp.where(ok, q, 0.0)
    level = jnp.where(ok, (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], qs) + LVL_EMA * qs, mem["level"])
    # credit: a store made by a worker restored from pool p counts as yield for p
    src_pool = jnp.where(mem["src"] >= 0, mem["slot_pool"][jnp.clip(mem["src"], 0, K - 1)], 0)
    yld = mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    nc = contacts.shape[1]
    seen_c = mem["seen_c"].at[:nc].set(mem["seen_c"][:nc] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld, "seen_c": seen_c}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
