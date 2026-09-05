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
FAM_P = (0.40, 0.22, 0.23, 0.15)    # motor hold / thruster hold / resonant oscillation / systematic actuator sweep
H = 128                             # max steps between splits
Q = 0.5                             # AMS level = this quantile of the pool
LVL_EMA = 0.2
P_R0, P_R1 = 0.04, 0.20             # restore probability ramp (early exploration -> late exploitation)
WORLD = 5.0
RESERVE_FRAC = 0.15                 # workers that never restore
G = 3                               # shape groups folded into proximity / reach pools
NW = 4                              # boundary walls are the first polygons
MAXN = 128
VIS_W = 1.0                         # count-based (Go-Explore) penalty on archive slot re-selection
BEST_W = 0.6                        # bonus for a pool's current champion slot
NH = 9 + 2 * G                      # 0 escape, 1 energy, 2 spin, 3 disp, 4-7 dirs, prox(G), reach(G), contacts
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pang=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _bodies(feats):
    """positions (B,N,2), speeds (B,N), angular (B,N), active (B,N), role (B,N); polygons first (walls are 0..NW-1)."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    ang = jnp.concatenate([feats[:, L["pang"]:L["pang"] + Np], jnp.zeros((B, Nc))], 1)   # per-polygon rotational channel
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), ang, act, role, L
def _phi(feats, mem):
    """importance vector (B, NH) + which pools are live on this level."""
    pos, spd, ang, act, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act                                              # moving & active
    c = pos - 0.5 * WORLD                                                   # centred coordinates
    escape = jnp.max(jnp.where(dyn, jnp.max(jnp.abs(c), axis=-1), -9.0), 1) / (0.5 * WORLD)
    energy = jnp.max(jnp.where(dyn, spd, 0.0), 1) ** 2 / 64.0
    spin = jnp.max(jnp.where(dyn, jnp.abs(ang - mem["a0"][:, :N]), 0.0), 1) / 8.0        # runaway rotation
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), 1) / WORLD
    dirs = jnp.stack([jnp.max(jnp.where(dyn, c[:, :, 0], -9.0), 1), jnp.max(jnp.where(dyn, -c[:, :, 0], -9.0), 1),
                      jnp.max(jnp.where(dyn, c[:, :, 1], -9.0), 1), jnp.max(jnp.where(dyn, -c[:, :, 1], -9.0), 1)], 1) / (0.5 * WORLD)
    stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)                      # static non-wall shapes: tunnel/contact targets
    green = (role == 1) & act; other = act & ~green
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)   # (B, N, N)
    prox = -jnp.min(jnp.where(dyn[:, :, None] & stat[:, None, :], d, 9.0), axis=1)       # (B, N) nearest moving body per target
    reach = -jnp.min(jnp.where(green[:, :, None] & other[:, None, :], d, 9.0), axis=1)   # (B, N) green body to each other shape
    grp = (jnp.arange(N)[None, :] % G) == jnp.arange(G)[:, None]            # (G, N) shape -> group folding
    pg = jnp.max(jnp.where(grp[None], prox[:, None, :], -9.0), axis=2) / WORLD
    rg = jnp.max(jnp.where(grp[None], reach[:, None, :], -9.0), axis=2) / WORLD
    contacts = (feats[:, L["cfrom"]:] > 0.5)
    newc = jnp.sum(contacts & ~mem["seen_c"][None, : contacts.shape[1]], axis=1).astype(jnp.float32)
    phi = jnp.concatenate([escape[:, None], energy[:, None], spin[:, None], disp[:, None], dirs, pg, rg, newc[:, None]], 1)
    live = jnp.concatenate([jnp.ones(8, bool), jnp.any(grp & stat.any(0)[None, :], 1),
                            jnp.any(grp & other.any(0)[None, :], 1) & green.any(), jnp.ones(1, bool)])
    return phi, contacts, live
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32), "flip": jnp.zeros(B, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 4, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, MAXN, 2)), "a0": jnp.zeros((B, MAXN)), "dyn": jnp.zeros(MAXN, bool), "seen_c": jnp.zeros(256, bool), "have_start": jnp.bool_(False),
            "pool": jnp.arange(B) % NH, "age": jnp.zeros(B, jnp.int32), "slot_pool": jnp.arange(K) % NH, "slot_phi": jnp.full(K, -jnp.inf),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "live": jnp.ones(NH, bool), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mu, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mu - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    phi, _, _ = _phi(feats, mem)
    live = mem["live"]
    pool = jnp.where(live[mem["pool"]], mem["pool"], jax.random.randint(k5, (B,), 0, NH) % NH)   # never sit on a dead hypothesis
    my = phi[jnp.arange(B), pool]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    below = my < mem["level"][pool]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == pool[:, None]) & arch_valid[None, :]                  # (B, K)
    # Go-Explore slot choice inside the pool: rarely-visited states first, champion slot mildly preferred
    pool_max = jnp.full(NH, -jnp.inf).at[mem["slot_pool"]].max(jnp.where(arch_valid, mem["slot_phi"], -jnp.inf))
    is_best = arch_valid & (mem["slot_phi"] >= pool_max[mem["slot_pool"]] - 1e-9)
    score = jax.random.gumbel(k1, (B, K)) + BEST_W * is_best[None, :].astype(jnp.float32) - VIS_W * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(pool_ok, score, -jnp.inf), axis=1)
    p_res = P_R0 + (P_R1 - P_R0) * tau
    do = ((below & (jax.random.uniform(k2, (B,)) < p_res)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, pool, 0)].add(do.astype(jnp.float32))
    # pool bandit over LIVE pools: weight ∝ (1 + downstream yield) / (1 + selections)
    w = jnp.where(live, (1.0 + mem["yield"]) / (1.0 + sel), 0.0); w = w / jnp.maximum(w.sum(), 1e-9)
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, pool)
    # a split also re-rolls the action policy, so re-expanding a state explores it on a different timescale
    kf, km = jax.random.split(k4)
    fam = jnp.where(do, jax.random.choice(kf, 4, (B,), p=jnp.array(FAM_P)), mem["fam"])
    mu = jnp.where(do, jnp.array(MUS)[jax.random.randint(km, (B,), 0, len(MUS))], mem["mu"])
    age = jnp.where(do, 0, mem["age"]); src = jnp.where(do, slot, mem["src"])
    return do, slot, {**mem, "pool": pool, "age": age, "sel": sel, "src": src, "fam": fam, "mu": mu, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3 = jax.random.split(rng, 3); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    motor = jax.random.randint(k1, (B,), 0, 2 * NM)
    thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + 1 + jax.random.randint(k2, (B,), 0, jnp.maximum(n_act - 2 * NM - 1, 1)), motor)
    fam = mem["fam"]
    sweep = (mem["cur"] + 1) % jnp.maximum(n_act, 1)          # family 3: systematic actuator sweep
    base = jnp.where(fam == 3, sweep, jnp.where(fam == 1, thr, motor))
    cur = jnp.clip(jnp.where(new, base, mem["cur"]), 0, n_act - 1)
    hold = _hold(mem["mu"], k3, B); left = jnp.where(new, hold, mem["left"]) - 1
    flip = jnp.where(new, jnp.clip(hold // 4, 2, 40), mem["flip"])
    osc = (fam == 2); tick = (mem["left"] % jnp.maximum(flip, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    action = jnp.clip(jnp.where(osc, (cur // 2) * 2 + (sign < 0).astype(jnp.int32), cur), 0, n_act - 1)
    return action, {**mem, "cur": cur, "left": left, "flip": flip, "sign": sign, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, ang, act, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    started = mem["have_start"]
    p0 = jnp.where(started, mem["p0"], mem["p0"].at[:, :N].set(pos))
    a0 = jnp.where(started, mem["a0"], mem["a0"].at[:, :N].set(ang))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "a0": a0, "have_start": jnp.bool_(True)}
    phi, contacts, live_now = _phi(feats, mem2)
    live = jnp.where(started, mem["live"] | live_now, live_now)
    # ratchet: per live pool with slots, the best worker replaces that pool's weakest slot if it beats it
    pool_has = (jnp.zeros(NH).at[mem["slot_pool"]].add(1.0)) > 0
    pool_min = jnp.full(NH, jnp.inf).at[mem["slot_pool"]].min(jnp.where(arch_valid, mem["slot_phi"], -jnp.inf))
    cand = (phi > pool_min[None, :]) & pool_has[None, :] & live[None, :] & ~done[:, None]
    best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0)                            # (NH,)
    has = cand.any(0)
    # one store per worker: each winning worker commits to its lowest-index pool, so slot_phi stays truthful
    first = jnp.full(B, NH, jnp.int32).at[best_w].min(jnp.where(has, jnp.arange(NH), NH).astype(jnp.int32))
    eff = has & (first[best_w] == jnp.arange(NH))
    slot_score = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_slot = jnp.argmin(jnp.where(mem["slot_pool"][None, :] == jnp.arange(NH)[:, None], slot_score[None, :], jnp.inf), axis=1)
    do = jnp.zeros(B, bool).at[best_w].max(eff)
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(eff, pool_slot, 0))
    newv = phi[best_w, jnp.arange(NH)]
    upd = jnp.full(K, -jnp.inf).at[pool_slot].max(jnp.where(eff, newv, -jnp.inf))
    wrote = jnp.zeros(K, bool).at[pool_slot].max(eff)
    slot_phi = jnp.where(wrote, upd, mem["slot_phi"])
    # AMS levels: EMA toward each pool's Q-quantile of member phi
    mmask = arch_valid[None, :] & (mem["slot_pool"][None, :] == jnp.arange(NH)[:, None])
    hasmem = mmask.any(1)
    members = jnp.where(mmask, slot_phi[None, :], jnp.nan)
    q = jnp.where(hasmem, jnp.nan_to_num(jnp.nanquantile(members, Q, axis=1), nan=0.0), 0.0)
    lvl_new = (1 - LVL_EMA) * jnp.where(jnp.isfinite(mem["level"]), mem["level"], q) + LVL_EMA * q
    level = jnp.where(hasmem, lvl_new, mem["level"])
    # credit: a store by a worker restored from pool p counts as yield for p
    src_pool = jnp.where(mem["src"] >= 0, mem["slot_pool"][jnp.maximum(mem["src"], 0)], 0)
    yld = mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    seen_c = mem["seen_c"].at[: contacts.shape[1]].set(mem["seen_c"][: contacts.shape[1]] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld, "seen_c": seen_c, "live": live}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
