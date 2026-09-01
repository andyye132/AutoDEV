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
A = 13                 # size of the augmented feature vector z (domain-informed, origin-relative)
# z layout: 0,1 = dyn_pos - origin | 2,3 = dyn_vel | 4 = dyn_max_speed | 5,6 = green - origin
#           7 = green_blue_dist    | 8,9 = blue - origin | 10 = |dyn_pos-origin| | 11 = |blue-origin| | 12 = |vel|
Z_SCALE = (0.35, 0.35, 1.2, 1.2, 1.2, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 1.2)
ND_RAND = 14            # extra random goal/frontier directions on top of the +/- axis-aligned basis
NDIR = 2 * A + ND_RAND  # every direction owns a persistent per-direction bandit + frontier record
BITS = 12               # hash bits for the novelty ring buffer
POS_GOAL_DIMS = (0, 1, 10)   # z-dims meaning displacement from origin (a goal here => OOB-style excursion)
VEL_GOAL_DIMS = (4, 12)      # z-dims meaning speed (a goal here => momentum build-up, TUNNEL-style)
MAX_REP = 26            # max macro-action duration (log-uniform)
P_LOW, P_HI = 0.02, 0.30    # restore prob: productive worker -> stalled worker
STALL_T = 40.0          # stall steps to reach P_HI
W_DIR = 1.5             # weight on "most extreme along my goal direction" when picking a restore target
W_NOV = 0.6             # weight on visit-count novelty when picking a restore target
MARGIN = 0.02           # record improvement needed to re-archive a direction
FORGET = 0.003          # per-step relaxation of records (keeps the frontier refreshing)
LR = 0.25               # per-direction bandit learning rate
BETA = 1.5              # bandit softmax temperature (inverse)
EPS = 0.45              # uniform-random action mixing
P_OSC = 0.45            # probability a commitment is a two-action oscillation
DECAY = 0.999           # bandit value decay
DQ_DECAY = 0.995        # decay of the per-direction frontier-progress trace (~200-step horizon)
DQ_BETA = 1.2           # softmax temperature (inverse) for picking a goal direction
DIR_EPS = 0.3           # fraction of restores that pick a uniformly random goal direction
P_OSC_POS = 0.12        # displacement-seeking workers almost never oscillate: sustained drift beats wobble
EPS_POS = 0.22          # ...and exploit their bandit more (less random mixing) to keep pushing outward
DUR_MULT_POS = 2.5      # ...and commit to each macro-action much longer
P_OSC_VEL = 0.10        # speed-seeking workers oscillate even less: one-way thrust accumulates momentum
EPS_VEL = 0.20          # ...and exploit their bandit more to keep accelerating in the winning direction
DUR_MULT_VEL = 1.8      # ...long enough for thrust to build up speed
THRUSTER_BIAS = 1.0     # log-bias on the two thruster actions for speed-seeking workers
def init(rng, F, n_act, K, B):
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    eye = jnp.eye(A)
    rnd = jax.random.normal(k1, (max(NDIR - 2 * A, 1), A))
    rnd = rnd / (jnp.linalg.norm(rnd, axis=1, keepdims=True) + 1e-6)
    dirs = jnp.concatenate([eye, -eye, rnd], axis=0)[:NDIR]            # (NDIR, A)
    KA = max(1, min(int(K) // 2, NDIR))                                # frontier slots; the rest is a novelty ring
    dir_is_pos = jnp.abs(dirs[:, jnp.array(POS_GOAL_DIMS)]).sum(-1) > 0.5
    dir_is_vel = jnp.abs(dirs[:, jnp.array(VEL_GOAL_DIMS)]).sum(-1) > 0.5
    ab = jnp.zeros(n_act).at[n_act - 3].add(jnp.log(0.15))             # down-weight no-op
    ab = ab.at[n_act - 2].add(jnp.log(1.5)).at[n_act - 1].add(jnp.log(1.5))  # up-weight thrusters
    return {
        "n_act": jnp.int32(n_act), "K": jnp.int32(K), "KA": jnp.int32(KA),
        "dirs": dirs, "dir_is_pos": dir_is_pos, "dir_is_vel": dir_is_vel, "act_bias": ab,
        "proj": jax.random.normal(k2, (A, BITS)), "scale": jnp.array(Z_SCALE, jnp.float32),
        "rec": jnp.full((NDIR,), -5.0), "q": jnp.zeros((NDIR, n_act), jnp.float32), "dq": jnp.zeros(NDIR),
        "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "next_free": jnp.int32(0),
        "origin": jnp.zeros(6, jnp.float32), "oset": jnp.int32(0),
        "cur": jnp.zeros(B, jnp.int32), "alt": jnp.zeros(B, jnp.int32), "last": jnp.zeros(B, jnp.int32),
        "per": jnp.ones(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
        "di": jax.random.randint(k3, (B,), 0, NDIR), "prevz": jnp.zeros((B, A)),
        "fresh": jnp.ones(B), "stall": jnp.zeros(B),
    }
def _latch(mem, feats):
    """Latch the level origin from the initial (identical across workers) state."""
    o0 = jnp.concatenate([feats[0, 0:2], feats[0, 5:7], feats[0, 9:11]])
    o = jnp.where(mem["oset"] > 0, mem["origin"], o0)
    return {**mem, "origin": o, "oset": jnp.int32(1)}
def _aug(mem, f):
    o = mem["origin"]
    p = f[:, 0:2] - o[0:2]
    v = f[:, 2:4]
    b = f[:, 9:11] - o[4:6]
    r = jnp.sqrt((p * p).sum(-1, keepdims=True) + 1e-12)
    rb = jnp.sqrt((b * b).sum(-1, keepdims=True) + 1e-12)
    vm = jnp.sqrt((v * v).sum(-1, keepdims=True) + 1e-12)
    z = jnp.concatenate([p, v, f[:, 4:5], f[:, 5:7] - o[2:4], f[:, 7:8], b, r, rb, vm], axis=-1)
    return jnp.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
def _cell(mem, z):
    x = jnp.round(z / mem["scale"]) * mem["scale"]
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    mem = _latch(mem, feats)
    B = feats.shape[0]; K = arch_valid.shape[0]
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    p = P_LOW + (P_HI - P_LOW) * jnp.clip(mem["stall"] / STALL_T, 0.0, 1.0)   # stalled workers recycle fast
    do = (jax.random.uniform(k1, (B,)) < p) & arch_valid.any()
    # meta-bandit over goal directions: concentrate the batch on axes whose frontier record is still
    # advancing (an axis that never saturates = the state is running away), eps-uniform floor for coverage.
    dv = mem["dq"]
    dv = (dv - dv.mean()) / (dv.std() + 1e-6)
    dpick = jnp.argmax(DQ_BETA * dv[None, :] + jax.random.gumbel(k4, (B, NDIR)), axis=1)
    dnew = jnp.where(jax.random.uniform(k3, (B,)) < DIR_EPS,
                      jax.random.randint(k3, (B,), 0, NDIR), dpick)
    di = jnp.where(do, dnew, mem["di"])                                       # new state -> new goal direction
    az = _aug(mem, arch_feats) / mem["scale"]
    pz = az @ mem["dirs"].T                        # (K, NDIR) projection of every stored state
    dscore = pz.T[di]                              # (B, K) "how extreme is this state in MY goal direction"
    nov = -W_NOV * jnp.log1p(arch_visits.astype(jnp.float32))
    logits = W_DIR * dscore + nov[None, :] + jax.random.gumbel(k2, (B, K))
    slot = jnp.argmax(jnp.where(arch_valid[None, :], logits, -jnp.inf), axis=1)
    mem = {**mem, "di": di, "stall": jnp.where(do, 0.0, mem["stall"]),
           "fresh": jnp.where(do, 1.0, mem["fresh"]), "left": jnp.where(do, 0, mem["left"])}
    return do, jnp.clip(slot, 0, K - 1), mem
def act(mem, feats, obs, rng):
    mem = _latch(mem, feats)
    B = feats.shape[0]
    n_act = mem["n_act"]
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    zn = _aug(mem, feats) / mem["scale"]
    # contextual bandit credit: how far did the last executed action move us along this worker's direction?
    r = jnp.clip(((zn - mem["prevz"]) * mem["dirs"][mem["di"]]).sum(-1), -3.0, 3.0) * (1.0 - mem["fresh"])
    q = (mem["q"] * DECAY).at[mem["di"], mem["last"]].add(LR * r)
    new = mem["left"] <= 0
    dpos = mem["dir_is_pos"][mem["di"]]                # is this worker's goal a displacement axis?
    dvel = mem["dir_is_vel"][mem["di"]]                # is this worker's goal a speed axis?
    ai = jnp.arange(q.shape[1])
    thr = jnp.where(dvel[:, None] & (ai[None, :] >= n_act - 2), THRUSTER_BIAS, 0.0)
    logits = mem["act_bias"][None, :] + BETA * q[mem["di"]] + thr
    greedy = jnp.argmax(logits + jax.random.gumbel(k1, logits.shape), -1)
    u = jax.random.uniform(k2, (B, 4))
    eps = jnp.where(dpos, EPS_POS, jnp.where(dvel, EPS_VEL, EPS))
    pick = jnp.where(u[:, 0] < eps, jax.random.randint(k3, (B,), 0, n_act), greedy)
    # oscillation partner: usually the opposite motor direction (a ^ 1), else a fresh action
    flip = pick ^ 1
    flip = jnp.where(flip < n_act, flip, pick)
    partner = jnp.where(u[:, 1] < 0.6, flip, jax.random.randint(k4, (B,), 0, n_act))
    p_osc = jnp.where(dpos, P_OSC_POS, jnp.where(dvel, P_OSC_VEL, P_OSC))
    osc = jax.random.uniform(k5, (B,)) < p_osc
    cur = jnp.where(new, pick, mem["cur"])
    alt = jnp.where(new, jnp.where(osc, partner, pick), mem["alt"])
    per = jnp.where(new, 1 + (u[:, 2] * 4).astype(jnp.int32), mem["per"])     # half-period 1..4
    dur = 1 + jnp.exp(u[:, 3] * jnp.log(MAX_REP + 1.0)).astype(jnp.int32)     # log-uniform duration
    dur = jnp.where(dpos, jnp.minimum((dur.astype(jnp.float32) * DUR_MULT_POS).astype(jnp.int32),
                                       int(MAX_REP * DUR_MULT_POS)),
                     jnp.where(dvel, jnp.minimum((dur.astype(jnp.float32) * DUR_MULT_VEL).astype(jnp.int32),
                                                  int(MAX_REP * DUR_MULT_VEL)), dur))
    left = jnp.where(new, dur, mem["left"])
    a = jnp.where(((left // jnp.maximum(per, 1)) % 2) == 0, cur, alt)
    a = jnp.clip(a, 0, n_act - 1)
    return a, {**mem, "q": q, "cur": cur, "alt": alt, "per": per, "left": left - 1,
               "last": a, "prevz": zn, "fresh": jnp.zeros(B)}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    mem = _latch(mem, feats)
    B = feats.shape[0]
    z = _aug(mem, feats)
    scale = 0.99 * mem["scale"] + 0.01 * jnp.maximum(jnp.abs(z).mean(0), 1e-3)   # keep the space normalized
    pz = (z / mem["scale"]) @ mem["dirs"].T                                       # (B, NDIR)
    gain = pz - mem["rec"][None, :]
    jstar = jnp.argmax(gain, -1)
    alive = ~done
    gstar = jnp.max(gain, -1)
    rec_hit = (gstar > MARGIN) & alive                    # this state extends the reachable frontier
    # credit the extended direction: decaying trace of how fast each axis is still pushing its record
    dq = mem["dq"] * DQ_DECAY
    dq = dq.at[jstar].add(jnp.where(rec_hit, jnp.clip(gstar, 0.0, 5.0), 0.0))
    cell = _cell(mem, z)
    novel = (mem["cell_slot"][cell] < 0) & alive & ~rec_hit
    KA = mem["KA"]; ring = jnp.maximum(mem["K"] - KA, 1)
    fslot = jstar % KA                                    # direction j permanently owns slot j % KA
    nslot = KA + (mem["next_free"] + jnp.cumsum(novel) - 1) % ring
    slot = jnp.clip(jnp.where(rec_hit, fslot, nslot), 0, mem["K"] - 1)
    rec = mem["rec"].at[jstar].max(jnp.where(rec_hit, pz[jnp.arange(B), jstar], -jnp.inf)) - FORGET
    cell_slot = mem["cell_slot"].at[jnp.where(novel, cell, 0)].set(jnp.where(novel, slot, mem["cell_slot"][0]))
    return rec_hit | novel, slot, {**mem, "rec": rec, "scale": scale, "cell_slot": cell_slot, "dq": dq,
                                    "next_free": (mem["next_free"] + novel.sum()) % ring,
                                    "stall": jnp.where(rec_hit, 0.0, mem["stall"] + 1.0)}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
