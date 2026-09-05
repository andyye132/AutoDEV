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
FAM_P = (0.40, 0.20, 0.20, 0.20)    # motor hold / thruster hold / resonant oscillation / 2-action chord
H = 160                              # max steps between splits
LVL_EMA = 0.2
P_RESTORE_MIN = 0.02                 # early phase: restores rare (the score is time-discounted)
P_RESTORE_MAX = 0.18
P_REFAM = 0.35                       # on restore: re-roll the action family (same state, new actuation)
WORLD = 5.0
RESERVE_FRAC = 0.15                  # workers that never restore
MAX_SHAPES = 12                      # non-wall shapes tracked as per-shape hypotheses
NW = 4                               # boundary walls are the first polygons
NB = 6                               # base coords: escape, energy, spin, displacement, cell-discovery, contact-discovery
NH = NB + 2 * MAX_SHAPES
GX = 8; NSP = 4; NCELL = GX * GX * NSP
DECAY = 0.98                         # half-life ~34 steps for the discovery-rate coords
LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pspin=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _bodies(feats):
    """positions (B,N,2), speeds (B,N), spin (B,N), active (B,N), role (B,N); polygons first (walls are 0..NW-1)."""
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    om = jnp.concatenate([feats[:, L["pspin"]:L["pspin"] + Np], jnp.zeros((B, Nc))], 1)     # polygons carry one extra scalar (angular rate)
    act = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), om, act, role, L
def _cell(pos, spd, dyn, green):
    """coarse (x, y, |v|) cell of the focus body (green if present, else any mover) — the coverage coordinate."""
    B, N = spd.shape
    f = jnp.argmax(2.0 * green.astype(jnp.float32) + dyn.astype(jnp.float32), axis=1)
    p = pos[jnp.arange(B), f]; s = spd[jnp.arange(B), f]
    qx = jnp.clip((p[:, 0] / WORLD * GX).astype(jnp.int32), 0, GX - 1)
    qy = jnp.clip((p[:, 1] / WORLD * GX).astype(jnp.int32), 0, GX - 1)
    qs = jnp.clip((jnp.log1p(jnp.abs(s)) * 1.5).astype(jnp.int32), 0, NSP - 1)
    return qx + GX * qy + GX * GX * qs
def _phi(feats, mem):
    """importance vector (B, NH): [escape, energy, spin, displacement, cell-rate, contact-rate, prox_j (MAX_SHAPES), reach_j (MAX_SHAPES)].
    Per-shape coords are gathered through mem['ord'], which compacts the level's real non-wall shapes to the front."""
    pos, spd, om, act, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act                                             # moving & active
    dist_wall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)               # Chebyshev distance from the arena centre
    escape = jnp.max(jnp.where(dyn, dist_wall, 0.0), axis=1) / (0.5 * WORLD)
    energy = jnp.max(jnp.where(dyn, spd, 0.0), axis=1) ** 2 / 64.0
    spin = jnp.max(jnp.where(dyn, jnp.abs(om), 0.0), axis=1) / 8.0
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), axis=1) / WORLD
    tgt = act & (jnp.arange(N)[None, :] >= NW)                             # every active non-wall shape is a target hypothesis
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (B, N, N)
    nself = ~jnp.eye(N, dtype=bool)[None, :, :]
    prox = -jnp.min(jnp.where(dyn[:, :, None] & tgt[:, None, :] & nself, d, 9.0), axis=1)   # nearest moving body to each target
    green = (role == 1) & act
    reach = -jnp.min(jnp.where(green[:, :, None] & act[:, None, :] & ~green[:, None, :], d, 9.0), axis=1)
    idx = jnp.clip(mem["ord"], 0, max(N - 1, 0)); gv = mem["ordv"][None, :]
    gprox = jnp.where(gv, prox[:, idx] / WORLD, -9.0)
    greach = jnp.where(gv, reach[:, idx] / WORLD, -9.0)
    cell = _cell(pos, spd, dyn, green)
    contacts = (feats[:, L["cfrom"]:] > 0.5)
    return jnp.concatenate([escape[:, None], energy[:, None], spin[:, None], disp[:, None],
                            mem["ncov"][:, None], mem["ncon"][:, None], gprox, greach], 1), contacts, cell
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng); v0 = min(NH, K)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "alt": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
            "sign": jnp.ones(B, jnp.int32), "flip": jnp.zeros(B, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 4, (B,), p=jnp.array(FAM_P)),
            "t": jnp.int32(0), "p0": jnp.zeros((B, 64, 2)), "dyn": jnp.zeros(64, bool), "seen_c": jnp.zeros(256, bool), "have_start": jnp.bool_(False),
            "vis": jnp.zeros(NCELL), "ncov": jnp.zeros(B), "ncon": jnp.zeros(B),
            "ord": jnp.arange(MAX_SHAPES, dtype=jnp.int32), "ordv": jnp.ones(MAX_SHAPES, bool),
            "pmap": jnp.arange(NH, dtype=jnp.int32), "veff": jnp.int32(v0), "pvalid": jnp.arange(NH) < v0,
            "slot_pool": (jnp.arange(K) % NH).astype(jnp.int32), "slot_phi": jnp.full(K, -jnp.inf),
            "pool": (jnp.arange(B) % NH).astype(jnp.int32), "age": jnp.zeros(B, jnp.int32),
            "level": jnp.full(NH, -jnp.inf), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH), "src": jnp.full(B, -1, jnp.int32)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    pool = jnp.where(mem["pvalid"][mem["pool"]], mem["pool"], mem["pmap"][jnp.arange(B) % mem["veff"]])   # only live hypotheses
    phi, _, _ = _phi(feats, mem)
    my = phi[jnp.arange(B), pool]
    lvl = mem["level"][pool]; lvl_f = jnp.where(jnp.isfinite(lvl), lvl, 0.0)
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    below = my < lvl
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    pool_ok = (mem["slot_pool"][None, :] == pool[:, None]) & arch_valid[None, :]                          # (B, K)
    bias = 0.8 * jnp.tanh(mem["slot_phi"][None, :] - lvl_f[:, None]) - 0.6 * jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    slot = jnp.argmax(jnp.where(pool_ok, jax.random.gumbel(k1, (B, K)) + bias, -jnp.inf), axis=1)
    p = P_RESTORE_MIN + (P_RESTORE_MAX - P_RESTORE_MIN) * tau
    do = ((below & (jax.random.uniform(k2, (B,)) < p)) | (mem["age"] >= H)) & pool_ok.any(1) & ~reserve
    sel = mem["sel"].at[jnp.where(do, pool, 0)].add(do.astype(jnp.float32))
    # pool bandit over LIVE hypotheses: weight ∝ (1 + downstream yield) / (1 + selections)
    w = jnp.where(mem["pvalid"], (1.0 + mem["yield"]) / (1.0 + sel), 0.0); w = w / jnp.maximum(w.sum(), 1e-9)
    newpool = jax.random.choice(k3, NH, (B,), p=w)
    pool = jnp.where(do & (jax.random.uniform(k4, (B,)) < 0.5), newpool, pool)
    fam = jnp.where(do & (jax.random.uniform(k5, (B,)) < P_REFAM), jax.random.choice(k4, 4, (B,), p=jnp.array(FAM_P)), mem["fam"])
    age = jnp.where(do, 0, mem["age"]); src = jnp.where(do, slot, mem["src"])
    return do, slot, {**mem, "pool": pool, "age": age, "sel": sel, "src": src, "fam": fam, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4); new = mem["left"] <= 0
    NM = 4; n_act = mem["n_act"]
    motor = jax.random.randint(k1, (B,), 0, jnp.maximum(jnp.minimum(2 * NM, n_act), 1))
    n_thr = jnp.maximum(n_act - 2 * NM - 1, 1)                                      # thrusters sit between the motors and the trailing no-op
    thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + jax.random.randint(k2, (B,), 0, n_thr), motor)
    fam = mem["fam"]
    base = jnp.where(fam == 1, thr, motor)                    # oscillate/chord start from a motor
    cur = jnp.where(new, base, mem["cur"])
    alt = jnp.where(new, jax.random.randint(k4, (B,), 0, jnp.maximum(n_act - 1, 1)), mem["alt"])
    hold = _hold(mem, k3, B); left = jnp.where(new, hold, mem["left"]) - 1
    flip = jnp.where(new, jnp.clip(hold // 4, 2, 40), mem["flip"])
    osc = (fam == 2); tick = (mem["left"] % jnp.maximum(flip, 1)) == 0
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    a_osc = (cur // 2) * 2 + (sign < 0).astype(jnp.int32)
    a_ch = jnp.where(((mem["t"] + jnp.arange(B)) % 2) == 0, cur, alt)               # chord: interleave two actuators every step
    action = jnp.where(fam == 3, a_ch, jnp.where(osc, a_osc, cur))
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0))
    return action, {**mem, "cur": cur, "alt": alt, "left": left, "flip": flip, "sign": sign, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, om, act, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    # --- compact the level's real non-wall shapes to the front; dead coords never get slots or workers
    live = act.any(0) & (jnp.arange(N) >= NW)
    o = jnp.argsort(jnp.where(live, 0, 1)).astype(jnp.int32)
    ordi = jnp.clip(jnp.concatenate([o, jnp.zeros(MAX_SHAPES, jnp.int32)])[:MAX_SHAPES], 0, max(N - 1, 0))
    ordv = jnp.concatenate([live[o], jnp.zeros(MAX_SHAPES, bool)])[:MAX_SHAPES]
    ordi = jnp.where(mem["have_start"], mem["ord"], ordi); ordv = jnp.where(mem["have_start"], mem["ordv"], ordv)
    vc = jnp.concatenate([jnp.ones(NB, bool), ordv, ordv])
    pmap = jnp.argsort(jnp.where(vc, 0, 1)).astype(jnp.int32)
    veff = jnp.minimum(jnp.sum(vc).astype(jnp.int32), jnp.int32(K))
    slot_pool = pmap[jnp.arange(K) % veff]
    pvalid = jnp.zeros(NH, jnp.int32).at[pmap].set(jnp.arange(NH, dtype=jnp.int32)) < veff
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True), "ord": ordi, "ordv": ordv,
            "pmap": pmap, "veff": veff, "pvalid": pvalid, "slot_pool": slot_pool}
    phi, contacts, cell = _phi(feats, mem2)
    # --- ratchet, worker-centric so no two pools fight over one worker (and no phantom slot_phi writes)
    slot_score = jnp.where(arch_valid, mem["slot_phi"], -jnp.inf)
    pool_min = jnp.full(NH, jnp.inf).at[slot_pool].min(slot_score)
    emptyp = ~jnp.isfinite(pool_min)                                                # unfilled (or unslotted) pool: take anything
    margin = jnp.where(emptyp[None, :], 10.0 + phi, phi - pool_min[None, :])
    ok = pvalid[None, :] & (margin > 0.0) & ~done[:, None]
    cb = jnp.argmax(jnp.where(ok, margin + 0.3 * jax.random.gumbel(rng, (B, NH)), -jnp.inf), axis=1)   # each worker claims one pool
    anyb = ok.any(1)
    msc = jnp.where(anyb, margin[jnp.arange(B), cb], -jnp.inf)
    selm = (cb[None, :] == jnp.arange(NH)[:, None]) & anyb[None, :]                  # (NH, B)
    has = selm.any(1)
    best_b = jnp.argmax(jnp.where(selm, msc[None, :], -jnp.inf), axis=1)             # (NH,) distinct workers
    pslot = jnp.argmin(jnp.where(slot_pool[None, :] == jnp.arange(NH)[:, None], slot_score[None, :], jnp.inf), axis=1)
    do = jnp.zeros(B, bool).at[best_b].max(has)
    slot = jnp.zeros(B, jnp.int32).at[best_b].max(jnp.where(has, pslot, 0))
    tgt = jnp.where(has, pslot, 0); newv = phi[best_b, jnp.arange(NH)]
    wrote = jnp.zeros(K, bool).at[tgt].max(has)
    wval = jnp.full(K, -jnp.inf).at[tgt].max(jnp.where(has, newv, -jnp.inf))
    slot_phi = jnp.where(wrote, wval, mem["slot_phi"])
    # --- AMS levels: EMA toward each pool's mean member importance
    mrow = slot_pool[None, :] == jnp.arange(NH)[:, None]
    va = (arch_valid | wrote) & jnp.isfinite(slot_phi)
    mem_mask = mrow & va[None, :]
    cnt = jnp.sum(mem_mask, axis=1)
    q = jnp.sum(jnp.where(mem_mask, slot_phi[None, :], 0.0), axis=1) / jnp.maximum(cnt, 1)
    prev = jnp.where(jnp.isfinite(mem["level"]), mem["level"], q)
    level = jnp.where(cnt > 0, (1 - LVL_EMA) * prev + LVL_EMA * q, mem["level"])
    # --- credit a store to the pool the storing worker was split from
    src_pool = jnp.where(mem["src"] >= 0, slot_pool[jnp.clip(mem["src"], 0, K - 1)], 0)
    yld = mem["yield"].at[src_pool].add((do & (mem["src"] >= 0)).astype(jnp.float32))
    # --- discovery rates (decayed): new coverage cells and new contact pairs, per worker
    firstv = (mem["vis"][cell] < 0.5).astype(jnp.float32)
    vis = mem["vis"].at[cell].add(1.0)
    nc = contacts.shape[1]
    newc = jnp.sum(contacts & ~mem["seen_c"][None, :nc], axis=1).astype(jnp.float32)
    ncov = DECAY * mem["ncov"] + firstv
    ncon = DECAY * mem["ncon"] + newc
    seen_c = mem["seen_c"].at[:nc].set(mem["seen_c"][:nc] | contacts.any(0))
    return do, slot, {**mem2, "slot_phi": slot_phi, "level": level, "yield": yld, "seen_c": seen_c,
                      "vis": vis, "ncov": ncov, "ncon": ncon}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
