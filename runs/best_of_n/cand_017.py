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
FAM_P = (0.34, 0.20, 0.26, 0.20); NFAM = 4   # motor hold / thruster hold / motor resonance / two-action alternation
S_STALL = 80                        # steps without improving your own pool coordinate -> kill & split
P_LO, P_HI = 0.06, 0.30             # below-level kill probability, ramped (score is time-discounted)
LVL_EMA = 0.15
WORLD = 5.0
RESERVE_FRAC = 0.125                # workers that never restore (keep fresh trajectories from reset)
NW = 4                              # boundary walls are the first polygons
NB = 128                            # padding for per-body memory
NH = 12                             # hypotheses = archive pools
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pw=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _bodies(feats):
    """positions (B,N,2), speeds (B,N), |angular| (B,N), active (B,N), role (B,N); polygons first (walls are 0..NW-1)."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    omg = jnp.concatenate([feats[:, L["pw"]:L["pw"] + Np], jnp.zeros((B, Nc))], 1)          # per-polygon scalar (angular velocity)
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), jnp.abs(omg), act, role, L
def _phi(feats, mem):
    """importance vector (B, NH): escape, energy, displacement, spread, tunnel-proximity, green-reach,
    new-contacts, spin, and the four signed position extremes (+x,-x,+y,-y)."""
    pos, spd, omg, act, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act                                             # (B,N) has-moved & active
    ctr = pos - 0.5 * WORLD
    esc = jnp.max(jnp.where(dyn, jnp.max(jnp.abs(ctr), -1), -9.0), 1) / (0.5 * WORLD)
    enr = jnp.max(jnp.where(dyn, spd, 0.0), 1) ** 2 / 64.0
    dsp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), 1) / WORLD
    spn = jnp.max(jnp.where(dyn, omg, 0.0), 1) / 8.0
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)   # (B,N,N)
    pair = act[:, :, None] & act[:, None, :]
    spr = jnp.sum(jnp.where(pair, d, 0.0), (1, 2)) / (jnp.sum(pair, (1, 2)) + 1e-6) / WORLD
    stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)                     # static non-wall shapes: tunnel/contact targets
    prox = -jnp.min(jnp.where(dyn[:, :, None] & stat[:, None, :], d, 9.0), (1, 2)) / WORLD
    green = (role == 1) & act
    rch = -jnp.min(jnp.where(green[:, :, None] & act[:, None, :] & ~green[:, None, :], d, 9.0), (1, 2)) / WORLD
    contacts = (feats[:, L["cfrom"]:] > 0.5)
    newc = jnp.sum(contacts & ~mem["seen_c"][None, : contacts.shape[1]], axis=1).astype(jnp.float32)
    ext = [jnp.max(jnp.where(dyn, s * ctr[:, :, a], -9.0), 1) / (0.5 * WORLD) for a in (0, 1) for s in (1.0, -1.0)]
    return jnp.stack([esc, enr, dsp, spr, prox, rch, newc, spn] + ext, 1), contacts, spd, act
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "alt": jnp.zeros(B, jnp.int32),
            "left": jnp.zeros(B, jnp.int32), "per": jnp.ones(B, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, NFAM, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, NB, 2)), "dyn": jnp.zeros(NB, bool), "seen_c": jnp.zeros(512, bool), "have_start": jnp.bool_(False),
            "pool": jnp.arange(B) % NH, "stall": jnp.zeros(B, jnp.int32), "best": jnp.full(B, -jnp.inf),
            "slot_pool": jnp.arange(K) % NH, "slot_phi": jnp.full(K, -jnp.inf),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    phi, contacts, spd, act = _phi(feats, mem)
    my = phi[jnp.arange(B), mem["pool"]]
    improved = my > mem["best"] + 1e-4
    best = jnp.maximum(mem["best"], my); stall = jnp.where(improved, 0, mem["stall"] + 1)
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 1024.0, 0.0, 1.0)
    below = my < mem["level"][mem["pool"]]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == mem["pool"][:, None]) & arch_valid[None, :]      # (B,K)
    surv = (mem["slot_phi"] >= mem["level"][mem["slot_pool"]]).astype(jnp.float32)            # AMS: split onto above-level members
    score = jax.random.gumbel(k1, (B, K)) + 0.7 * surv[None, :] - 0.4 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(pool_ok, score, -jnp.inf), axis=1)
    fire = (below & (jax.random.uniform(k2, (B,)) < P_LO + (P_HI - P_LO) * tau)) | (stall >= S_STALL)
    do = fire & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    # pool bandit over pools that actually own a live slot: weight ∝ (1 + downstream yield) / (1 + selections)
    pool_live = (jnp.zeros(NH, jnp.int32).at[mem["slot_pool"]].max(arch_valid.astype(jnp.int32)) > 0).astype(jnp.float32)
    w = (1.0 + mem["yield"]) / (1.0 + sel) * pool_live
    w = jnp.where(w.sum() > 0, w, jnp.ones(NH)); w = w / w.sum()
    swap = do & (jax.random.uniform(k4, (B,)) < 0.5)
    pool = jnp.where(swap, jax.random.choice(k3, NH, (B,), p=w), mem["pool"])
    # a split must diversify the continuation, not replay it: redraw the behaviour family and hold exponent
    fam = jnp.where(do, jax.random.choice(k5, NFAM, (B,), p=jnp.array(FAM_P)), mem["fam"])
    mu = jnp.where(do, jnp.array(MUS)[jax.random.randint(k6, (B,), 0, len(MUS))], mem["mu"])
    return do, slot, {**mem, "pool": pool, "fam": fam, "mu": mu, "sel": sel, "src": jnp.where(do, slot, mem["src"]),
                      "stall": jnp.where(do, 0, stall), "best": jnp.where(do, -jnp.inf, best), "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    motor = jax.random.randint(k1, (B,), 0, jnp.maximum(jnp.minimum(2 * NM, n_act), 1))
    thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + 1 + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - 2 * NM - 1, 1)), motor)
    anyv = jax.random.randint(k5, (B,), 0, jnp.maximum(n_act, 1))
    fam = mem["fam"]
    base = jnp.where(fam == 1, thr, jnp.where(fam == 3, anyv, motor))
    cur = jnp.where(new, base, mem["cur"])
    alt = jnp.where(new, jax.random.randint(k6, (B,), 0, jnp.maximum(n_act, 1)), mem["alt"])
    hold = _hold(mem, k3, B); left = jnp.where(new, hold, mem["left"]) - 1
    per = jnp.where(new, jnp.clip(jnp.round(2.0 ** (6.0 * jax.random.uniform(k4, (B,)))).astype(jnp.int32), 1, 64), mem["per"])
    ph = (mem["t"] // jnp.maximum(per, 1)) % 2                              # square wave, log-uniform period: sweeps resonances
    action = jnp.where(fam == 2, (cur // 2) * 2 + ph, jnp.where(fam == 3, jnp.where(ph > 0, alt, cur), cur))
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0))
    return action, {**mem, "cur": cur, "alt": alt, "left": left, "per": per, "t": mem["t"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, omg, act, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True)}
    phi, contacts, _, _ = _phi(feats, mem2)
    # ratchet: a pool's slots hold its top-|pool| states; a worker enters by beating the pool's current minimum
    valid_phi = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    has_slot = jnp.zeros(NH, jnp.int32).at[mem["slot_pool"]].max(jnp.ones(K, jnp.int32)) > 0
    pool_min = jnp.where(has_slot, jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(valid_phi), jnp.inf)
    cand = (phi > pool_min[None, :] + 1e-6) & ~done[:, None]
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0)                               # (NH,) best worker per pool
    has = cand.any(0)
    # one store per worker per step: a worker writes for the lowest-index pool that claimed it, others retry next step
    owner = jnp.full(B, NH, jnp.int32).at[best_w].min(jnp.where(has, jnp.arange(NH), NH))
    ok = has & (owner[best_w] == jnp.arange(NH))
    pool_slot = jnp.argmin(jnp.where(mem["slot_pool"][None, :] == jnp.arange(NH)[:, None], valid_phi[None, :], jnp.inf), axis=1)
    do = jnp.zeros(B, jnp.int32).at[best_w].max(ok.astype(jnp.int32)) > 0
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(ok, pool_slot, 0))
    # scatter-max is duplicate-index safe and equals a set here (the target is the pool's minimum)
    slot_phi = mem["slot_phi"].at[pool_slot].max(jnp.where(ok, phi[best_w, jnp.arange(NH)], -jnp.inf))
    # AMS levels: EMA toward the mean importance of each pool's members
    memb = (mem["slot_pool"][None, :] == jnp.arange(NH)[:, None]) & arch_valid[None, :]
    fin = jnp.where(jnp.isfinite(slot_phi), slot_phi, 0.0)
    cnt = jnp.sum(memb, 1)
    mean = jnp.sum(jnp.where(memb, fin[None, :], 0.0), 1) / jnp.maximum(cnt, 1)
    level = jnp.where(cnt > 0, (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], mean) + LVL_EMA * mean, mem["level"])
    # credit: a store made by a worker restored from pool p counts as yield for p
    src_pool = jnp.where(mem["src"] >= 0, mem["slot_pool"][jnp.clip(mem["src"], 0, K - 1)], 0)
    yld = mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    seen_c = mem["seen_c"].at[: contacts.shape[1]].set(mem["seen_c"][: contacts.shape[1]] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld, "seen_c": seen_c}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
