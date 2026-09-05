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
FAM_P = (0.32, 0.20, 0.48)          # motor hold / thruster hold / reactive bang-bang
H = 160                             # max steps between restores
P0, P1 = 0.03, 0.12                 # restore prob = P0 + P1 * tau
WORLD = 5.0
RESERVE_FRAC = 0.15                 # workers that never restore
MAX_SHAPES = 16
NW = 4                              # boundary walls are the first polygons
NM = 4                              # motor bindings (actions 2j / 2j+1)
DEAD = -1.75                        # phi at/below this == the coordinate does not exist on this level
CD = 2                              # bang-bang refractory (steps)
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
NH = 3 + 2 * MAX_SHAPES + 1
def _phi(feats, mem):
    """importance vector (B, NH): [escape, energy, displacement, prox-to-shape j (MAX_SHAPES), green-reach-to-shape j (MAX_SHAPES), new contacts].
    Coordinates for shapes that do not exist / are not valid targets sit at the DEAD sentinel and are masked out downstream."""
    pos, spd, act, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & act                                             # (B, N) moving & active
    dist_wall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)               # Chebyshev distance from the arena centre (2.5 = at a wall)
    escape = jnp.max(jnp.where(dyn, dist_wall, 0.0), axis=1) / (0.5 * WORLD)
    energy = jnp.max(jnp.where(dyn, spd, 0.0), axis=1) ** 2 / 64.0
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), axis=1) / WORLD
    stat = act & ~dyn & (jnp.arange(N)[None, :] >= NW)                     # static non-wall shapes: tunnel/contact targets
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)   # (B, N, N)
    prox = -jnp.min(jnp.where(dyn[:, :, None] & stat[:, None, :], d, 9.0), axis=1)      # (B, N) per target shape: nearest moving body
    green = (role == 1) & act
    reach = -jnp.min(jnp.where(green[:, :, None] & act[:, None, :] & ~green[:, None, :], d, 9.0), axis=1)   # (B, N) green to each other shape
    pad = lambda x: jnp.concatenate([x, jnp.full((B, MAX_SHAPES), -9.0)], 1)[:, :MAX_SHAPES]
    contacts = (feats[:, L["cfrom"]:] > 0.5)
    newc = jnp.sum(contacts & ~mem["seen_c"][None, : contacts.shape[1]], axis=1).astype(jnp.float32)
    return jnp.concatenate([escape[:, None], energy[:, None], disp[:, None], pad(prox) / WORLD, pad(reach) / WORLD, newc[:, None]], 1), contacts
def init(rng, F, n_act, K, B):
    k1, k2, k3 = jax.random.split(rng, 3)
    return {"n_act": jnp.int32(n_act), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32),
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))], "fam": jax.random.choice(k2, 3, (B,), p=jnp.array(FAM_P)),
            "midx": jax.random.randint(k3, (B,), 0, NM), "stride": 1 + 2 * (jnp.arange(B) % 2), "bidx": jnp.zeros(B, jnp.int32),
            "prev": jnp.full(B, -jnp.inf), "cd": jnp.zeros(B, jnp.int32),
            "t": jnp.int32(0), "p0": jnp.zeros((B, 64, 2)), "dyn": jnp.zeros(64, bool), "seen_c": jnp.zeros(256, bool), "have_start": jnp.bool_(False),
            "pool": jnp.arange(B) % 3, "age": jnp.zeros(B, jnp.int32), "srcp": jnp.full(B, -1, jnp.int32),
            "own": jnp.full(NH, -1, jnp.int32), "rec": jnp.full(NH, -jnp.inf), "last": jnp.zeros(NH, jnp.int32),
            "owner": jnp.full(K, -1, jnp.int32), "yield": jnp.zeros(NH), "sel": jnp.zeros(NH)}
def _hold(mem, rng, B):
    u = jax.random.uniform(rng, (B,), minval=1e-6); return jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    """Each worker pursues one hypothesis; when it falls behind that hypothesis' record it is teleported back onto the
    record state and pushes again. Hypotheses are handed out by a yield bandit restricted to LIVE coordinates."""
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    phi, _ = _phi(feats, mem)
    own = mem["own"]; live = own >= 0
    w = jnp.where(live, (1.0 + mem["yield"]) / (1.0 + mem["sel"]), 0.0)
    w = jnp.where(w.sum() > 0, w, (jnp.arange(NH) < 3).astype(jnp.float32))    # before anything is live: escape / energy / displacement
    w = w / jnp.maximum(w.sum(), 1e-9)
    pool = jnp.where(jax.random.uniform(k1, (B,)) < 0.35, jax.random.choice(k3, NH, (B,), p=w), mem["pool"])
    slot = jnp.clip(own[pool], 0, K - 1)
    ok = live[pool] & arch_valid[slot]
    my = phi[jnp.arange(B), pool]
    tau = jnp.clip(mem["t"].astype(jnp.float32) / 2048.0, 0.0, 1.0)
    behind = my < mem["rec"][pool]
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    do = ((behind & (jax.random.uniform(k2, (B,)) < P0 + P1 * tau)) | (mem["age"] >= H)) & ok & ~reserve
    sel = mem["sel"].at[jnp.where(do, pool, 0)].add(do.astype(jnp.float32))
    return do, slot, {**mem, "pool": jnp.where(do, pool, mem["pool"]), "age": jnp.where(do, 0, mem["age"]), "sel": sel,
                      "srcp": jnp.where(do, pool, mem["srcp"]), "left": jnp.where(do, 0, mem["left"]),
                      "prev": jnp.where(do, -jnp.inf, mem["prev"]), "cd": jnp.where(do, CD, mem["cd"])}
def act(mem, feats, obs, rng):
    """Three families over zeta-distributed holds. Family 2 is a reactive bang-bang pump: the worker locks onto one moving
    body and reverses its motor as soon as that body's speed+escape signal stops improving, so the drive period locks to the
    level's own resonance instead of a random period."""
    B = feats.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4); n_act = mem["n_act"]
    pos, spd, active, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"][:N] & active
    new = mem["left"] <= 0
    hold = _hold(mem, k1, B); left = jnp.where(new, hold, mem["left"]) - 1
    midx = jnp.where(new, (mem["midx"] + mem["stride"]) % NM, mem["midx"])            # systematic motor sweep, per-worker stride
    pick = jnp.argmax(jnp.where(dyn, jax.random.gumbel(k2, (B, N)), -jnp.inf), axis=1)
    bidx = jnp.clip(jnp.where(new, pick, mem["bidx"]), 0, N - 1)                      # feedback channel: one moving body
    b = jnp.arange(B)
    wall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)
    sig = spd[b, bidx] / 8.0 + wall[b, bidx] / (0.5 * WORLD)
    prev = jnp.where(jnp.isfinite(mem["prev"]), mem["prev"], sig)
    cd = jnp.maximum(mem["cd"] - 1, 0)
    flip = (mem["fam"] == 2) & ((sig - prev) <= 1e-4) & (cd <= 0) & ~new              # anti-damping: reverse when the push stops paying
    sign = jnp.where(new, jnp.where(jax.random.uniform(k3, (B,)) < 0.5, 1, -1), jnp.where(flip, -mem["sign"], mem["sign"]))
    motor = 2 * midx + (sign < 0).astype(jnp.int32)
    thr_n = jnp.maximum(n_act - (2 * NM + 1), 1)
    thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + 1 + jax.random.randint(k4, (B,), 0, thr_n), motor)
    action = jnp.clip(jnp.where(mem["fam"] == 1, thr, motor), 0, jnp.maximum(n_act - 1, 0))
    return action, {**mem, "left": left, "midx": midx, "bidx": bidx, "sign": sign, "cd": jnp.where(flip, CD, cd),
                    "prev": sig, "t": mem["t"] + 1, "age": mem["age"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    """Per-coordinate record pointers: own[c] is the single slot holding hypothesis c's best state, ratcheted in place.
    Dead coordinates never claim a slot; own[] stays injective so distinct coordinates never collide on a slot."""
    B = feats.shape[0]; K = arch_valid.shape[0]
    pos, spd, active, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have_start"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "have_start": jnp.bool_(True)}
    phi, contacts = _phi(feats, mem2)
    rec = mem["rec"]; own = mem["own"]; owner = mem["owner"]; cid = jnp.arange(NH)
    alive = phi.max(0) > DEAD                                                          # coordinate actually exists on this level
    cand = (phi > rec[None, :]) & ~done[:, None] & alive[None, :]
    has = cand.any(0); best_w = jnp.argmax(jnp.where(cand, phi, -jnp.inf), axis=0)
    have = own >= 0
    free = jnp.argmax(~arch_valid); any_free = (~arch_valid).any()
    stale = jnp.where(owner >= 0, mem["last"][jnp.clip(owner, 0, NH - 1)], -1)
    alloc = jnp.where(any_free, free, jnp.argmin(stale))                               # free slot, else evict the stalest owner
    need = has & ~have; new_ok = need.any(); c_new = jnp.argmax(jnp.where(need, -cid, -jnp.inf))
    slot_of = jnp.where(have, jnp.clip(own, 0, K - 1), alloc)
    allowed = has & (have | ((cid == c_new) & new_ok))                                 # at most one allocation per step
    wc = jnp.full(B, -1, jnp.int32).at[best_w].max(jnp.where(allowed, cid, -1))        # one store per worker
    stored = allowed & (wc[best_w] == cid)
    do = jnp.zeros(B, jnp.int32).at[best_w].max(stored.astype(jnp.int32)) > 0
    slot = jnp.zeros(B, jnp.int32).at[best_w].max(jnp.where(stored, slot_of, 0))
    did = new_ok & stored[c_new] & ~have[c_new]
    old = jnp.where(did & ~any_free, owner[alloc], -1)
    own2 = jnp.where(cid == old, -1, jnp.where(stored, slot_of, own))
    owner2 = jnp.where((jnp.arange(K) == alloc) & did, c_new, owner)
    rec2 = jnp.where(stored, phi[best_w, cid], rec)
    last2 = jnp.where(stored, mem["t"], mem["last"])
    yld = mem["yield"].at[jnp.maximum(mem["srcp"], 0)].add(jnp.where(do & (mem["srcp"] >= 0), 1.0, 0.0))
    seen_c = mem["seen_c"].at[: contacts.shape[1]].set(mem["seen_c"][: contacts.shape[1]] | contacts.any(0))
    return do, slot, {**mem2, "rec": rec2, "own": own2, "owner": owner2, "last": last2, "yield": yld, "seen_c": seen_c}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
