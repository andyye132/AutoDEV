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
MUS = (1.5, 2.0, 2.5, 3.0); N_MAX = 128
FAM_P = (0.32, 0.18, 0.26, 0.24)   # motor hold / thruster hold / oscillate / greedy action-bandit
H = 96                              # max steps between splits
QL = 0.5                            # AMS level = this quantile of the pool
LVL_EMA = 0.2
P_RES0, P_RES1 = 0.10, 0.35         # restore probability when below level: start -> late
WORLD = 5.0
RESERVE_FRAC = 0.12                 # workers that never restore (pure exploration)
NW = 4                              # boundary walls are the first polygons
NB = 4                              # target-shape buckets (shape index % NB)
NH = 8 + NB                         # hypotheses / pools
ELITE = 2.0                         # frontier bias when picking a member to split from
EPS = 0.25; QLR = 0.25              # action-bandit exploration / learning rate
NCMAX = 256
BIG = 9.0
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) >= 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, paux=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _bodies(feats):
    """positions (B,N,2), speeds (B,N), active, role, aux (angular scalar, polygons only) — polygons first (walls 0..NW-1)."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    aux = jnp.concatenate([feats[:, L["paux"]:L["paux"] + Np], jnp.zeros((B, Nc))], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), act, role, aux, L
def _phi(feats, mem):
    """importance vector (B, NH): 12 target-blind coordinates; every planted witness is a threshold on one of them."""
    pos, spd, act, role, aux, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act                                              # moving & active
    dwall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1, initial=0.0) / (0.5 * WORLD)   # 1.0 = at a wall
    escape = jnp.max(jnp.where(dyn, dwall, 0.0), axis=1, initial=0.0)
    e = spd ** 2 / 64.0
    emax = jnp.max(jnp.where(dyn, e, 0.0), axis=1, initial=0.0)             # runaway physics (single body)
    esum = jnp.sum(jnp.where(dyn, e, 0.0), axis=1)                          # collective energy
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), axis=1, initial=0.0) / WORLD
    spin = jnp.max(jnp.where(dyn, jnp.abs(aux), 0.0), axis=1, initial=0.0) / 8.0        # angular channel
    green = (role == 1) & act
    gesc = jnp.max(jnp.where(green, dwall, 0.0), axis=1, initial=0.0)       # the green body itself leaving
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :] + 1e-9, axis=-1)        # (B,N,N)
    gd = jnp.min(jnp.where(green[:, :, None] & act[:, None, :] & ~green[:, None, :], d, BIG), axis=1, initial=BIG)
    greach = -jnp.min(gd, axis=1, initial=BIG) / WORLD                      # green closing on anything (unintended goal)
    stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)                      # static non-wall shapes: tunnel/hazard targets
    nd = jnp.min(jnp.where(dyn[:, :, None] & stat[:, None, :], d, BIG), axis=1, initial=BIG)        # (B,N) per target
    bmask = (jnp.arange(N)[:, None] % NB) == jnp.arange(NB)[None, :]        # (N,NB)
    prox = -jnp.min(jnp.where(bmask[None, :, :], nd[:, :, None], BIG), axis=1, initial=BIG) / WORLD  # (B,NB)
    contacts = (feats[:, L["cfrom"]:] > 0.5)[:, :NCMAX]
    newc = jnp.sum(contacts & ~mem["seen_c"][None, : contacts.shape[1]], axis=1).astype(jnp.float32)
    phi = jnp.concatenate([escape[:, None], emax[:, None], esum[:, None], disp[:, None], spin[:, None],
                           gesc[:, None], greach[:, None], newc[:, None], prox], 1)
    return phi, contacts, spd, act
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32), "flip": jnp.zeros(B, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 4, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, 64, 2)), "dyn": jnp.zeros(64, bool), "seen_c": jnp.zeros(NCMAX, bool), "have_start": jnp.bool_(False),
            "pool": jnp.arange(B) % NH, "age": jnp.zeros(B, jnp.int32), "slot_pool": jnp.arange(K) % NH, "slot_phi": jnp.full(K, -jnp.inf),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32),
            "Q": jnp.zeros((B, int(n_act))), "prev_a": jnp.zeros(B, jnp.int32), "phi_prev": jnp.zeros(B), "fresh": jnp.ones(B, bool)}
def _hold(mu, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mu - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    phi, contacts, spd, act_ = _phi(feats, mem)
    my = phi[jnp.arange(B), mem["pool"]]
    # --- action bandit credit: the coordinate change since last step is credited to the action that was held ---
    ok = ~mem["fresh"]
    qcur = mem["Q"][jnp.arange(B), mem["prev_a"]]
    qn = jnp.where(ok, qcur + QLR * ((my - mem["phi_prev"]) - qcur), qcur)
    Qm = mem["Q"].at[jnp.arange(B), mem["prev_a"]].set(qn)
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 1024.0, 0.0, 1.0)
    below = my < mem["level"][mem["pool"]]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == mem["pool"][:, None]) & arch_valid[None, :]           # (B,K)
    # frontier-biased member selection: normalise phi inside each pool, add Gumbel noise, penalise revisits
    pmax = jnp.full(NH, -jnp.inf).at[mem["slot_pool"]].max(jnp.where(arch_valid, mem["slot_phi"], -jnp.inf))
    pmin = jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(jnp.where(arch_valid, mem["slot_phi"], jnp.inf))
    pmax = jnp.where(jnp.isfinite(pmax), pmax, 0.0); pmin = jnp.where(jnp.isfinite(pmin), pmin, 0.0)
    span = jnp.maximum(pmax - pmin, 1e-6)
    u = jnp.clip(jnp.nan_to_num((mem["slot_phi"] - pmin[mem["slot_pool"]]) / span[mem["slot_pool"]], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    score = ELITE * u[None, :] + jax.random.gumbel(k1, (B, K)) - 0.15 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(pool_ok, score, -jnp.inf), axis=1)
    p_res = P_RES0 + (P_RES1 - P_RES0) * tau
    do = ((below & (jax.random.uniform(k2, (B,)) < p_res)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, mem["pool"], 0)].add(do.astype(jnp.float32))
    # pool bandit: re-assign a restoring worker to a pool with probability proportional to (1 + yield) / (1 + selections)
    w = (1.0 + mem["yield"]) / (1.0 + sel); w = w / w.sum()
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, mem["pool"])
    # a split child resamples its behaviour family and exploration timescale, so branches actually diverge
    nf = jax.random.choice(k5, 4, (B,), p=jnp.array(FAM_P)); nm = jnp.array(MUS)[jax.random.randint(k5, (B,), 0, len(MUS))]
    fam = jnp.where(do, nf, mem["fam"]); mu = jnp.where(do, nm, mem["mu"])
    return do, slot, {**mem, "pool": pool, "fam": fam, "mu": mu, "Q": Qm, "phi_prev": my, "fresh": do,
                      "age": jnp.where(do, 0, mem["age"]), "sel": sel, "src": jnp.where(do, slot, mem["src"]), "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6); new = mem["left"] <= 0
    n_act = mem["n_act"]; NM = 4
    nmot = jnp.maximum(jnp.minimum(2 * NM, n_act), 1)
    motor = jax.random.randint(k1, (B,), 0, nmot)
    tlo = jnp.clip(2 * NM + 1, 0, jnp.maximum(n_act - 1, 0))
    thr = tlo + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - tlo, 1))
    greedy = jnp.argmax(mem["Q"] + 1e-3 * jax.random.uniform(k5, mem["Q"].shape), axis=1)
    ban = jnp.where(jax.random.uniform(k6, (B,)) < EPS, jax.random.randint(k4, (B,), 0, n_act), greedy)
    fam = mem["fam"]
    base = jnp.where(fam == 1, thr, jnp.where(fam == 3, ban, motor))        # family 2 (oscillate) uses a motor and flips its sign
    cur = jnp.where(new, base, mem["cur"])
    hold = _hold(mem["mu"], k3, B); hold = jnp.where(fam == 3, jnp.clip(hold, 1, 24), hold)   # the bandit re-decides often
    left = jnp.where(new, hold, mem["left"]) - 1
    flip = jnp.where(new, jnp.clip(hold // 4, 2, 40), mem["flip"])
    osc = (fam == 2); tick = (mem["left"] % jnp.maximum(flip, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    action = jnp.where(osc, (cur // 2) * 2 + (sign < 0).astype(jnp.int32), cur)
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0)).astype(jnp.int32)
    return action, {**mem, "cur": cur, "left": left, "flip": flip, "sign": sign, "prev_a": action, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, act, role, aux, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True)}
    phi, contacts, _, _ = _phi(feats, mem2)
    # ratchet: per pool, the best worker replaces that pool's weakest slot if it beats it
    pool_cnt = jnp.zeros(NH).at[mem["slot_pool"]].add(1.0)
    pool_min_val = jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(jnp.where(arch_valid, mem["slot_phi"], -jnp.inf))
    pool_min_val = jnp.where(jnp.isinf(pool_min_val), -jnp.inf, pool_min_val)            # empty pool: anything qualifies
    cand = (phi > pool_min_val[None, :]) & ~done[:, None]                                # (B,NH)
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0)                           # (NH,)
    has = cand.any(0) & (pool_cnt > 0)
    slot_score = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_slot = jnp.argmin(jnp.where(mem["slot_pool"][None, :] == jnp.arange(NH)[:, None], slot_score[None, :], jnp.inf), axis=1)   # (NH,)
    # one store per worker: if several pools picked the same worker, the highest-index pool owns it
    owner = jnp.full(B, -1, jnp.int32).at[best_w].max(jnp.where(has, jnp.arange(NH, dtype=jnp.int32), -1))
    has = has & (owner[best_w] == jnp.arange(NH))
    do = jnp.zeros(B, bool).at[best_w].max(has)
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(has, pool_slot, 0))
    slot_phi = mem["slot_phi"].at[jnp.where(has, pool_slot, K)].set(jnp.where(has, phi[best_w, jnp.arange(NH)], 0.0), mode="drop")
    # AMS levels: EMA toward each pool's Q-quantile of member phi
    members = jnp.where(arch_valid[None, :] & (mem["slot_pool"][None, :] == jnp.arange(NH)[:, None]), slot_phi[None, :], jnp.nan)
    q = jnp.nan_to_num(jnp.nanquantile(members, QL, axis=1), nan=-jnp.inf)
    level = jnp.where(jnp.isfinite(q), (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], q) + LVL_EMA * q, mem["level"])
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
