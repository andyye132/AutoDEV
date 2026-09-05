"""Tester program (evolvable). The four hooks below define how a batch of B workers explores a Kinetix level.
Kinetix states are save/restore-able: the harness keeps K archive slots; `restore` may jump a worker to a stored
state, `archive` may store the worker's current state. Everything must be pure JAX (jit/vmap-compatible, fixed shapes).

Feature vector `feats` (per worker, F floats, oracle-blind):
  [0:2] dyn_pos (centroid of moving shapes), [2:4] dyn_vel, [4] dyn_max_speed, [5:7] green_pos, [7] green_blue_dist,
  [8] time_frac, [9:11] blue_pos, then per-polygon pos/vel/rot/active/role blocks, per-circle pos/vel/active/role blocks,
  then contact flags. `obs` is Kinetix's symbolic observation. Actions: 0..2*NM-1 motor +/- (NM=4 motor bindings),
  2*NM = no-op, 2*NM+1.. thrusters (2 bindings).
Goal of the search: reach states that expose planted defects as fast as possible (steps-to-first-witness), and do so
in a way that generalizes to defect classes the search never saw."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 14              # hash bits for archive cells
TARGET_NEW = 0.03      # homeostat set-point: fraction of workers discovering a new cell per step
RES_MIN, RES_MAX = 0.02, 12.0
RES_ETA = 4.0
STAG_L = 40            # steps without a new cell before a worker jumps back to the frontier
P_RESTORE = 0.015      # small background jump rate on top of the stagnation trigger
P_ELITE = 0.60         # chance a resampled gene is derived from the elite gene
P_MUT = 0.35           # per-field mutation rate when deriving from the elite
MAX_REP = 20
EPS_MAX = 0.30
DGAIN = 5.0            # gain on the delta channel (z-score deltas are small)
W_VISIT, W_DIR, W_SCORE, W_REC = 1.0, 1.2, 0.7, 0.6
STATS_LR = 0.02
MIN_AGE = 24           # gene must live this long before it can become elite
A_DECAY = 0.999        # decay on the per-action novelty/usage counters
A_BETA = 1.0           # weight on measured per-action novelty yield
A_UCB = 1.5            # optimism bonus for under-tried actions (forces rare "verbs")
P_SWEEP = 0.30         # fraction of frontier jumps that run a systematic action sweep
SWEEP_REP = 2          # steps held per action during a sweep
GENE_KEYS = ("g_mode", "g_d1", "g_d2", "g_w1", "g_w2", "g_thr",
             "g_a", "g_b", "g_c", "g_rep", "g_str", "g_eps")
def _sample_genes(rng, B, F, n_act, alog=None):
    """A gene is a tiny closed-loop controller; modes 0/1 are reactive, 2/3 are committed."""
    ks = jax.random.split(rng, 12)
    u = jax.random.uniform(ks[0], (B,))
    mode = ((u > 0.35).astype(jnp.int32) + (u > 0.60).astype(jnp.int32) + (u > 0.80).astype(jnp.int32))
    eps = jax.random.uniform(ks[11], (B,)) * EPS_MAX * jnp.where(mode >= 2, 0.2, 1.0)
    # actions in a gene are drawn from the novelty/UCB bandit when it is available, uniform otherwise
    if alog is None:
        pick = lambda k: jax.random.randint(k, (B,), 0, n_act).astype(jnp.int32)
    else:
        pick = lambda k: jax.random.categorical(k, alog, shape=(B,)).astype(jnp.int32)
    return {"g_mode": mode,
            "g_d1": jax.random.randint(ks[1], (B,), 0, F),
            "g_d2": jax.random.randint(ks[2], (B,), 0, F),
            "g_w1": jax.random.normal(ks[3], (B,)),
            "g_w2": 0.5 * jax.random.normal(ks[4], (B,)),
            "g_thr": jnp.abs(jax.random.normal(ks[5], (B,))) * 0.5,
            "g_a": pick(ks[6]),
            "g_b": pick(ks[7]),
            "g_c": pick(ks[8]),
            "g_rep": jax.random.randint(ks[9], (B,), 1, MAX_REP + 1),
            "g_str": jax.random.randint(ks[10], (B,), 0, jnp.maximum(n_act, 2)),
            "g_eps": eps}
def _z(mem, x):
    return jnp.clip((x - mem["mu"]) * jax.lax.rsqrt(mem["var"] + 1e-4), -8.0, 8.0)
def _cell(mem, z):
    q = jnp.clip(jnp.round(z / mem["res"]), -1e3, 1e3)
    return (((q @ mem["proj"]) > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)
def _alog(mem):
    """UCB logits over the action alphabet: measured cell-novelty yield + optimism for untried actions."""
    v = mem["act_nov"] / (mem["act_cnt"] + 1.0)
    v = v - v.mean()
    v = v * jax.lax.rsqrt((v ** 2).mean() + 1e-8)
    return jnp.clip(A_BETA * v + A_UCB * jax.lax.rsqrt(mem["act_cnt"] + 1.0), -2.0, 2.0)
def init(rng, F, n_act, K, B):
    k0, k1 = jax.random.split(rng)
    g = _sample_genes(k1, B, F, n_act)
    mem = {"n_act": jnp.int32(n_act), "K": jnp.int32(K),
           "proj": jax.random.normal(k0, (F, BITS)),
           "mu": jnp.zeros(F, jnp.float32), "var": jnp.ones(F, jnp.float32),
           "prev": jnp.zeros((B, F), jnp.float32),
           "res": jnp.float32(1.0), "t": jnp.float32(0.0),
           "cell_slot": jnp.full(2 ** BITS + 1, -1, jnp.int32),      # last entry = scatter trash
           "slot_score": jnp.ones(K + 1, jnp.float32),               # parent-credit bandit value
           "slot_step": jnp.zeros(K + 1, jnp.float32),               # when the slot was written
           "slot_use": jnp.zeros(K + 1, jnp.float32),                # how often we jumped there
           "from_slot": jnp.full(B, K, jnp.int32),                   # parent slot of each worker
           "since": jnp.zeros(B, jnp.int32),                         # steps since this worker found a cell
           "age": jnp.zeros(B, jnp.int32), "nov": jnp.zeros(B, jnp.float32),
           "act_nov": jnp.zeros(int(n_act), jnp.float32),             # new cells credited to each action
           "act_cnt": jnp.zeros(int(n_act), jnp.float32),             # decayed usage count per action
           "last_a": jnp.zeros(B, jnp.int32), "sweep_ctr": jnp.int32(0),
           "elite_rate": jnp.float32(0.0), **g}
    for k_ in GENE_KEYS:
        mem["e_" + k_] = g[k_][0]
    return mem
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B, F = feats.shape
    K = arch_valid.shape[0]
    ks = jax.random.split(rng, 6 + len(GENE_KEYS))
    do = ((mem["since"] > STAG_L) | (jax.random.uniform(ks[0], (B,)) < P_RESTORE)) & arch_valid.any()
    # directional frontier: each worker pushes the reachable set outward along its own random direction
    za = _z(mem, arch_feats)
    r = jax.random.normal(ks[1], (B, F))
    r = r / (jnp.linalg.norm(r, axis=1, keepdims=True) + 1e-6)
    dirs = (za @ r.T).T / jnp.sqrt(jnp.float32(F))
    logits = (W_VISIT * (-jnp.log1p(arch_visits.astype(jnp.float32)))
              + W_SCORE * jnp.log1p(jnp.maximum(mem["slot_score"][:K], 0.0))
              + W_REC * (mem["slot_step"][:K] / (mem["t"] + 1.0)))[None, :] + W_DIR * dirs
    logits = jnp.where(arch_valid[None, :], logits, -jnp.inf)
    slot = jnp.argmax(logits + jax.random.gumbel(ks[2], (B, K)), axis=1).astype(jnp.int32)
    # resample the controller gene on a jump, or when a gene has proven barren
    rs = do | ((mem["age"] > 96) & (mem["nov"] <= 0.0))
    fresh = _sample_genes(ks[3], B, F, mem["n_act"], _alog(mem))
    use_e = jax.random.uniform(ks[4], (B,)) < P_ELITE
    out = dict(mem)
    for i, k_ in enumerate(GENE_KEYS):
        keep = use_e & (jax.random.uniform(ks[5 + i], (B,)) > P_MUT)
        cand = jnp.where(keep, mem["e_" + k_].astype(fresh[k_].dtype), fresh[k_])
        out[k_] = jnp.where(rs, cand, mem[k_])
    # systematic action-space sweep: some workers that reach the frontier cycle through EVERY action
    # from that state, phase-staggered across the batch, so rare state-changing "verbs" get fired there
    sweep = do & (jax.random.uniform(ks[5 + len(GENE_KEYS)], (B,)) < P_SWEEP)
    off = ((jnp.arange(B) + mem["sweep_ctr"]) % jnp.maximum(mem["n_act"], 1)).astype(jnp.int32)
    out["g_mode"] = jnp.where(sweep, 2, out["g_mode"]).astype(jnp.int32)
    out["g_a"] = jnp.where(sweep, off, out["g_a"]).astype(jnp.int32)
    out["g_rep"] = jnp.where(sweep, SWEEP_REP, out["g_rep"]).astype(jnp.int32)
    out["g_str"] = jnp.where(sweep, 1, out["g_str"]).astype(jnp.int32)
    out["g_eps"] = jnp.where(sweep, 0.0, out["g_eps"])
    out["sweep_ctr"] = mem["sweep_ctr"] + jnp.int32(1)
    out["age"] = jnp.where(rs, 0, mem["age"])
    out["nov"] = jnp.where(rs, 0.0, mem["nov"])
    out["since"] = jnp.where(do, 0, mem["since"])
    out["from_slot"] = jnp.where(do, slot, mem["from_slot"])
    return do, slot, out
def act(mem, feats, obs, rng):
    B = feats.shape[0]
    k1, k2 = jax.random.split(rng)
    z = _z(mem, feats)
    dz = (z - _z(mem, mem["prev"])) * DGAIN
    idx = jnp.arange(B)
    d1, d2 = mem["g_d1"], mem["g_d2"]
    s_pos = mem["g_w1"] * z[idx, d1] + mem["g_w2"] * z[idx, d2]
    s_del = mem["g_w1"] * dz[idx, d1] + mem["g_w2"] * dz[idx, d2]
    s = jnp.where(mem["g_mode"] == 1, s_del, s_pos)
    a = jnp.where(s > mem["g_thr"], mem["g_a"],
                  jnp.where(s < -mem["g_thr"], mem["g_b"], mem["g_c"]))
    cyc = (mem["g_a"] + (mem["age"] // mem["g_rep"]) * mem["g_str"]) % jnp.maximum(mem["n_act"], 1)
    a = jnp.where(mem["g_mode"] == 2, cyc, a)
    a = jnp.where(mem["g_mode"] == 3, mem["g_a"], a)
    # epsilon moves are drawn from the same UCB bandit, so exploration also favours under-tried actions
    rnd = jax.random.categorical(k1, _alog(mem), shape=(B,)).astype(jnp.int32)
    a = jnp.where(jax.random.uniform(k2, (B,)) < mem["g_eps"], rnd, a)
    a = jnp.clip(a.astype(jnp.int32), 0, mem["n_act"] - 1)
    mu = mem["mu"] + STATS_LR * (feats.mean(0) - mem["mu"])
    var = jnp.clip(mem["var"] + STATS_LR * (((feats - mem["mu"]) ** 2).mean(0) - mem["var"]), 0.0, 1e12)
    return a, {**mem, "prev": feats, "mu": mu, "var": var, "last_a": a}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]
    K = arch_valid.shape[0]
    cell = _cell(mem, _z(mem, feats))
    is_new = (mem["cell_slot"][cell] < 0) & (~done)
    cnt = is_new.astype(jnp.int32)
    order = jnp.clip(jnp.cumsum(cnt) - 1, 0, K - 1)
    prio = jnp.where(arch_valid, mem["slot_score"][:K] - 0.05 * mem["slot_use"][:K], -1e9)
    slot = jnp.argsort(prio).astype(jnp.int32)[order]           # fill empties first, then evict worst
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 2 ** BITS)].set(slot)
    si = jnp.where(is_new, slot, jnp.int32(K))
    slot_score = mem["slot_score"].at[si].set(2.0)
    slot_step = mem["slot_step"].at[si].set(mem["t"])
    slot_use = mem["slot_use"].at[si].set(0.0).at[mem["from_slot"]].add(1.0)
    slot_score = slot_score.at[jnp.where(is_new, mem["from_slot"], jnp.int32(K))].add(1.0) * 0.9995
    rate = cnt.sum() / jnp.float32(B)
    res = jnp.clip(mem["res"] * jnp.exp(RES_ETA * jnp.clip(rate - TARGET_NEW, -0.02, 0.05)), RES_MIN, RES_MAX)
    since = jnp.where(is_new, 0, mem["since"] + 1)
    nov = mem["nov"] + cnt.astype(jnp.float32)
    age = mem["age"] + 1
    # closed-loop credit for the action alphabet: which action just produced an unseen cell?
    la = jnp.clip(mem["last_a"], 0, mem["act_cnt"].shape[0] - 1)
    act_nov = (mem["act_nov"] * A_DECAY).at[la].add(cnt.astype(jnp.float32))
    act_cnt = (mem["act_cnt"] * A_DECAY).at[la].add(1.0)
    r = jnp.where(age >= MIN_AGE, nov / jnp.maximum(age, 1).astype(jnp.float32), -1.0)
    i = jnp.argmax(r)
    better = r[i] > mem["elite_rate"]
    out = {**mem, "cell_slot": cell_slot, "slot_score": slot_score, "slot_step": slot_step,
           "slot_use": slot_use, "res": res, "since": since, "nov": nov, "age": age,
           "act_nov": act_nov, "act_cnt": act_cnt,
           "t": mem["t"] + 1.0, "elite_rate": jnp.where(better, r[i], mem["elite_rate"] * 0.999)}
    for k_ in GENE_KEYS:
        out["e_" + k_] = jnp.where(better, mem[k_][i], mem["e_" + k_])
    return is_new, slot, out
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)