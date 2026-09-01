"""H4 — "elite beam ratchet, quality-diversity pools"."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
WORLD = 5.0
NW = 4                       # boundary walls are the first polygons
NB = 64                      # padded body count
MAXS = 12                    # per-shape channel budget
G = 7                        # global channels: speed escape etot disp newc spread twist
M = G + 2 * MAXS             # total channels
NM = 4                       # Kinetix DISCRETE: 0..2NM-1 motors, 2NM no-op, then thrusters
W_GLOBAL = (4.0, 2.0, 1.6, 1.0, 1.5, 0.8, 1.0)
W_GAP, W_PROX = 2.0, 1.3
NCELL = 64                   # behaviour cells inside a pool: (agent cell 4x4) x (speed band 4)
AGE_EPS = 2.0e-4             # per-step erosion: a saturated cell recycles instead of wedging
PATS = (20, 36, 64, 120)     # per-worker patience portfolio
H = 224
RESERVE = 0.18
ELITE = 4.5
VIS = 0.35
P_CROSS = 0.08
P_SWAP = 0.35
WARM = 24
MUS = (1.5, 2.0, 2.5, 3.0); N_MAX = 256
PERIODS = (2, 3, 4, 6, 8, 11, 15, 21, 30, 42)
FAMS = (0, 0, 1, 2, 0, 3, 1, 2, 0, 1, 2, 3, 0, 1, 3, 2)
STALL_K = 6

LAYOUT = {}
def _layout(F):
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0); C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, prot=11 + 4 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np,
                     cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]

ALLOC = {}
def _alloc(w, n):
    """Largest-remainder allocation of n slots to the channels with weight > 0; returns (counts, bases)."""
    key = (w, n)
    if key in ALLOC: return ALLOC[key]
    live = [i for i, x in enumerate(w) if x > 0.0]
    tot = sum(w[i] for i in live)
    cnt = [0] * len(w)
    for i in live: cnt[i] = max(1, int(n * w[i] / tot))
    while sum(cnt) > n:
        i = max(live, key=lambda j: (cnt[j] / w[j], cnt[j]))
        if cnt[i] <= 1: break
        cnt[i] -= 1
    while sum(cnt) < n:
        cnt[max(live, key=lambda j: w[j] / (cnt[j] + 1))] += 1
    base, acc = [], 0
    for c in cnt: base.append(acc); acc += c
    ALLOC[key] = (cnt, base)
    return ALLOC[key]

def _bodies(feats):
    L = _layout(feats.shape[1]); Np, Nc = L["Np"], L["Nc"]; B = feats.shape[0]
    pp = feats[:, L["ppos"]:L["ppos"] + 2 * Np].reshape(B, Np, 2); pv = feats[:, L["pvel"]:L["pvel"] + 2 * Np].reshape(B, Np, 2)
    cp = feats[:, L["cpos"]:L["cpos"] + 2 * Nc].reshape(B, Nc, 2); cv = feats[:, L["cvel"]:L["cvel"] + 2 * Nc].reshape(B, Nc, 2)
    pos = jnp.concatenate([pp, cp], 1); vel = jnp.concatenate([pv, cv], 1)
    rot = jnp.concatenate([feats[:, L["prot"]:L["prot"] + Np], jnp.zeros((B, Nc))], 1)
    actv = jnp.concatenate([feats[:, L["pact"]:L["pact"] + Np], feats[:, L["cact"]:L["cact"] + Nc]], 1) > 0.5
    role = jnp.concatenate([feats[:, L["prole"]:L["prole"] + Np], feats[:, L["crole"]:L["crole"] + Nc]], 1)
    return pos, jnp.linalg.norm(vel, axis=-1), rot, actv, role, L

def _cell(p):
    qx = jnp.clip(jnp.floor(p[:, 0] / WORLD * 4.0), 0, 3).astype(jnp.int32)
    qy = jnp.clip(jnp.floor(p[:, 1] / WORLD * 4.0), 0, 3).astype(jnp.int32)
    return qx * 4 + qy

def _phi(feats, mem):
    pos, spd, rot, actv, role, L = _bodies(feats); B, N = spd.shape
    dyn = mem["dyn"][:N] & actv
    smax = feats[:, 4]
    dwall = jnp.max(jnp.abs(pos - 0.5 * WORLD), axis=-1)
    escape = jnp.max(jnp.where(dyn, dwall, 0.0), 1) / (0.5 * WORLD)
    etot = jnp.sqrt(jnp.sum(jnp.where(dyn, spd ** 2, 0.0), 1)) / 8.0
    disp = jnp.max(jnp.where(dyn, jnp.linalg.norm(pos - mem["p0"][:, :N], axis=-1), 0.0), 1) / WORLD
    twist = jnp.max(jnp.where(dyn, jnp.abs(rot - mem["r0"][:, :N]), 0.0), 1) / 6.2832
    d = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :] + 1e-9, axis=-1)
    spread = jnp.max(jnp.where(dyn[:, :, None] & dyn[:, None, :], d, 0.0), axis=(1, 2)) / WORLD
    contacts = feats[:, L["cfrom"]:] > 0.5
    newc = jnp.sum(contacts & ~mem["seen_c"][None, : contacts.shape[1]], 1).astype(jnp.float32) / 4.0
    notself = jnp.arange(N)[:, None] != jnp.arange(N)[None, :]
    agent = (role == 1) & actv
    gap = -jnp.min(jnp.where(agent[:, :, None] & notself[None], d, 9.0), 1)
    prox = -jnp.min(jnp.where(dyn[:, :, None] & notself[None], d, 9.0), 1)
    gap = jnp.where(actv, gap, -9.0) / WORLD; prox = jnp.where(actv, prox, -9.0) / WORLD
    take = lambda x: jnp.concatenate([x, jnp.full((B, NW + MAXS), -1.8)], 1)[:, NW:NW + MAXS]
    phi = jnp.concatenate([(smax / 8.0)[:, None], escape[:, None], etot[:, None], disp[:, None], newc[:, None],
                           spread[:, None], twist[:, None], take(gap), take(prox)], 1)
    gsel = jnp.where(agent.any(1)[:, None], agent, dyn)
    gi = jnp.argmax(gsel, 1)
    gp = jnp.take_along_axis(pos, jnp.broadcast_to(gi[:, None, None], (B, 1, 2)), 1)[:, 0, :]
    sb = (smax > 0.5).astype(jnp.int32) + (smax > 2.0).astype(jnp.int32) + (smax > 6.0).astype(jnp.int32)
    cell = _cell(gp) * 4 + sb
    rj = jnp.concatenate([role[0], jnp.zeros(NW + MAXS)])[NW:NW + MAXS]
    aj = jnp.concatenate([actv[0], jnp.zeros(NW + MAXS, bool)])[NW:NW + MAXS].astype(jnp.float32)
    mg = jnp.where(rj > 1.5, jnp.where(rj > 2.5, 0.5, 2.2), jnp.where(rj > 0.5, 0.4, 1.0)) * aj
    mp_ = jnp.where(rj > 2.5, 0.6, 1.0) * aj
    mult = jnp.concatenate([jnp.ones(G), mg, mp_])
    return jnp.nan_to_num(phi, nan=-9.0, posinf=9.0, neginf=-9.0), contacts, cell, mult

def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng)
    L = _layout(F); S = min(max(L["Np"] + L["Nc"] - NW, 0), MAXS)
    w = tuple(list(W_GLOBAL) + [W_GAP if j < S else 0.0 for j in range(MAXS)] + [W_PROX if j < S else 0.0 for j in range(MAXS)])
    cnt, base = _alloc(w, K)
    slot_ch = []
    for c, n in enumerate(cnt): slot_ch += [c] * n
    wcnt, _ = _alloc(w, B)
    wch = []
    for c, n in enumerate(wcnt): wch += [c] * n
    b = jnp.arange(B)
    return {"n_act": jnp.int32(n_act), "t": jnp.int32(0),
            "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "sign": jnp.ones(B, jnp.int32),
            "flip": jnp.full(B, 8, jnp.int32), "fam": jnp.array(FAMS, jnp.int32)[b % len(FAMS)],
            "mu": jnp.array(MUS)[jax.random.randint(k1, (B,), 0, len(MUS))],
            "chan": jnp.array(wch, jnp.int32)[(b * 37 + 11) % B],
            "pat": jnp.array(PATS, jnp.int32)[(b // 3) % len(PATS)],
            "nostep": jnp.zeros(B, jnp.int32), "age": jnp.zeros(B, jnp.int32), "stall": jnp.zeros(B, jnp.int32),
            "own_best": jnp.full(B, -jnp.inf),
            "p0": jnp.zeros((B, NB, 2)), "r0": jnp.zeros((B, NB)), "dyn": jnp.zeros(NB, bool),
            "seen_c": jnp.zeros(512, bool), "have0": jnp.bool_(False),
            "slot_ch": jnp.array(slot_ch, jnp.int32), "slot_val": jnp.full(K, -jnp.inf),
            "cbase": jnp.array(base, jnp.int32), "cnum": jnp.array(cnt, jnp.int32),
            "pri": jnp.array(w), "valid_ch": jnp.array([x > 0.0 for x in w]), "boost": jnp.ones(M),
            "yld": jnp.zeros(M), "sel": jnp.zeros(M),
            "cmin": jnp.full(M, jnp.inf), "cmax": jnp.full(M, -jnp.inf),
            "chmax": jnp.zeros(M), "chspan": jnp.ones(M)}

def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = mem["chan"].shape[0]; K = arch_valid.shape[0]
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    c = mem["chan"]
    mine = (mem["slot_ch"][None, :] == c[:, None]) & arch_valid[None, :]
    sv = mem["slot_val"]
    nov = jnp.log1p(arch_visits.astype(jnp.float32))[None, :]
    z = jnp.clip((sv[None, :] - mem["chmax"][c][:, None]) / mem["chspan"][c][:, None], -8.0, 0.0)
    lg = jnp.where(mine, ELITE * z - VIS * nov, -jnp.inf)
    slot_m = jnp.argmax(lg + jax.random.gumbel(k1, (B, K)), 1)
    lg2 = jnp.where(arch_valid[None, :], -VIS * nov, -jnp.inf)
    slot_x = jnp.argmax(lg2 + jax.random.gumbel(k2, (B, K)), 1)
    cross = (jax.random.uniform(k3, (B,)) < P_CROSS) | ~mine.any(1)
    slot = jnp.where(cross, slot_x, slot_m)
    reserve = jnp.arange(B) < jnp.int32(RESERVE * B)
    trig = (mem["nostep"] >= mem["pat"]) | (mem["age"] >= H)
    do = trig & ~reserve & (mem["t"] > WARM) & arch_valid.any() & arch_valid[jnp.clip(slot, 0, K - 1)]
    live = ((mem["cmax"] - mem["cmin"]) > 1e-6) | (mem["t"] <= 4 * WARM)
    w = mem["pri"] * mem["boost"] * (0.4 + 2.0 * mem["yld"] / (1.0 + mem["sel"])) * live.astype(jnp.float32)
    w = jnp.where(w.sum() > 0.0, w, mem["pri"]); w = w / w.sum()
    nc = jax.random.choice(k4, M, (B,), p=w)
    swap = do & (jax.random.uniform(k5, (B,)) < P_SWAP)
    chan = jnp.where(swap, nc, c)
    sel = mem["sel"] + jnp.zeros(M).at[chan].add(do.astype(jnp.float32))
    return do, slot, {**mem, "chan": chan, "sel": sel,
                      "age": jnp.where(do, 0, mem["age"]), "nostep": jnp.where(do, 0, mem["nostep"]),
                      "own_best": jnp.where(do, -jnp.inf, mem["own_best"]),
                      "left": jnp.where(do, 0, mem["left"]), "stall": jnp.where(do, 0, mem["stall"])}

def act(mem, feats, obs, rng):
    B = mem["cur"].shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    n_act = mem["n_act"]; fam = mem["fam"]
    climbing = mem["stall"] == 0
    new = (mem["left"] <= 0) | (mem["stall"] >= STALL_K)
    nmot = jnp.minimum(2 * NM, jnp.maximum(n_act - 1, 1))
    motor = jax.random.randint(k1, (B,), 0, nmot)
    hi = jnp.maximum(n_act - 2 * NM - 1, 1)
    thr = jnp.where(n_act > 2 * NM + 1, 2 * NM + 1 + jax.random.randint(k2, (B,), 0, hi), motor)
    base = jnp.where((fam == 1) | (fam == 3), thr, motor)
    cur = jnp.where(new, base, mem["cur"])
    u = jax.random.uniform(k3, (B,), minval=1e-6)
    hold = jnp.clip(jnp.floor(u ** (-1.0 / (mem["mu"] - 1.0))), 1, N_MAX).astype(jnp.int32)
    left = jnp.where(new, hold, mem["left"]) - (1 - climbing.astype(jnp.int32))
    per = jnp.array(PERIODS, jnp.int32)[(jnp.arange(B) * 7 + jax.random.randint(k4, (B,), 0, 3)) % len(PERIODS)]
    flip = jnp.where(new, per, mem["flip"])
    osc = (fam == 2) | (fam == 3)
    tick = ((mem["left"] % jnp.maximum(flip, 1)) == 0) & ~climbing
    sign = jnp.where(new, 1, jnp.where(osc & tick, -mem["sign"], mem["sign"]))
    noop = jnp.minimum(2 * NM, jnp.maximum(n_act - 1, 0))
    a_osc = (cur // 2) * 2 + (sign < 0).astype(jnp.int32)
    a_bb = jnp.where(sign < 0, noop, cur)
    action = jnp.where(fam == 2, a_osc, jnp.where(fam == 3, a_bb, cur))
    action = jnp.clip(action, 0, jnp.maximum(n_act - 1, 0)).astype(jnp.int32)
    return action, {**mem, "cur": cur, "left": left, "flip": flip, "sign": sign,
                    "t": mem["t"] + 1, "age": mem["age"] + 1, "stall": jnp.where(new, 0, mem["stall"])}

def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; bi = jnp.arange(B)
    pos, spd, rot, actv, role, L = _bodies(feats); N = spd.shape[1]
    dyn = mem["dyn"].at[:N].set(mem["dyn"][:N] | (spd > 1e-3).any(0))
    p0 = jnp.where(mem["have0"], mem["p0"], mem["p0"].at[:, :N].set(pos))
    r0 = jnp.where(mem["have0"], mem["r0"], mem["r0"].at[:, :N].set(rot))
    mem2 = {**mem, "dyn": dyn, "p0": p0, "r0": r0, "have0": jnp.bool_(True)}
    phi, contacts, cell, mult = _phi(feats, mem2)
    sv = jnp.where(arch_valid, mem["slot_val"] - AGE_EPS, -10.0)
    # --- MAP-Elites pools: a channel's slots are behaviour cells, so its beam keeps distinct launch points ---
    tgt = mem["cbase"][None, :] + jnp.mod(cell[:, None], jnp.maximum(mem["cnum"], 1)[None, :])      # (B,M)
    gain = phi - sv[tgt]
    cand = (gain > 1e-5) & (~done)[:, None] & mem["valid_ch"][None, :] & (mem["cnum"] > 0)[None, :]
    myc = jnp.argmax(jnp.where(cand, gain, -jnp.inf), 1)
    do0 = cand.any(1)
    slot = tgt[bi, myc]
    g = jnp.where(do0, gain[bi, myc], -jnp.inf)
    best = jnp.full(K, -jnp.inf).at[slot].max(g)
    win = do0 & (g >= best[slot])
    first = jnp.full(K, B, jnp.int32).at[slot].min(jnp.where(win, bi, B))
    do = win & (first[slot] == bi)                                                  # exactly one writer per cell
    slot_val = sv.at[jnp.where(do, slot, K)].set(phi[bi, myc], mode="drop")
    # --- pool statistics for the beam step ---
    mask = mem["slot_ch"][None, :] == jnp.arange(M)[:, None]
    filled = mask & arch_valid[None, :]
    ok = filled.any(1)
    chmax = jnp.where(ok, jnp.max(jnp.where(filled, sv[None, :], -jnp.inf), 1), 0.0)
    chmin = jnp.min(jnp.where(filled, sv[None, :], jnp.inf), 1)
    chspan = jnp.maximum(jnp.where(ok, chmax - chmin, 1.0), 1e-2)
    # --- per-worker ratchet bookkeeping on its own channel ---
    own = phi[bi, mem["chan"]]
    imp = own > mem["own_best"] + 1e-4
    nostep = jnp.where(imp | done, 0, mem["nostep"] + 1)
    stall = jnp.where(imp, 0, mem["stall"] + 1)
    own_best = jnp.where(done, -jnp.inf, jnp.maximum(mem["own_best"], own))
    yld = 0.998 * mem["yld"] + jnp.zeros(M).at[mem["chan"]].add(do.astype(jnp.float32))
    seen_c = mem["seen_c"].at[: contacts.shape[1]].set(mem["seen_c"][: contacts.shape[1]] | contacts.any(0))
    return do, slot, {**mem2, "slot_val": slot_val, "chmax": chmax, "chspan": chspan, "boost": mult,
                      "cmin": jnp.minimum(mem["cmin"], phi.min(0)), "cmax": jnp.maximum(mem["cmax"], phi.max(0)),
                      "nostep": nostep, "stall": stall, "own_best": own_best, "yld": yld, "seen_c": seen_c,
                      "age": jnp.where(done, 0, mem["age"])}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
