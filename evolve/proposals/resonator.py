"""Energy-pumping resonance search ("resonator").

Idea: physics bugs (runaway integration, tunnelling through thin/soft geometry, shapes launched out of the arena) all
require *energy*, and energy is injected by driving actuators in phase with the level's own oscillation, not by
random button mashing.
(1) act() runs five structured drives instead of random actions: motor square waves at a per-worker period, thruster
    holds, thruster pulse trains, a velocity-feedback pump (push the motor in whichever direction the dynamic
    centroid is already moving -- the classic playground-swing resonance), and a thruster/motor interleave. Each
    worker re-rolls its drive after a restore, after an epoch, or when its energy plateaus.
(2) The archive is an energy *ladder*: slots are partitioned into NB bands by log speed and each band keeps a rotating
    set of states, so the archive can never collapse onto one state and always retains a staircase of increasingly
    energetic configurations.
(3) restore() is an explicit hill-climb: workers jump to the highest occupied band (or a random band for diversity)
    whenever their own energy record has not improved for PATIENCE steps, so every restore restarts pumping from the
    fastest state found so far. Repeated pump-and-checkpoint cycles multiply speed geometrically.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
NB = 8            # number of log-speed bands in the ladder archive
BPO = 1.6         # bands per octave of log2(1 + energy)
P_LADDER = 0.05   # baseline per-step restore probability
PATIENCE = 30     # steps without a personal energy record before forcing a restore
TOP_FRAC = 0.55   # probability of restoring from the highest occupied band (else a random band)
P_STORE = 0.10    # probability of storing a non-record state (keeps bands fresh)
EPOCH = 48        # steps before a worker re-rolls its drive pattern
PERIODS = (2, 3, 4, 6, 8, 12)
N_MODE = 5        # square wave / thruster hold / thruster pulse / velocity-feedback pump / interleave
VEL_W = 0.4       # weight of the centroid speed in the energy score (on top of dyn_max_speed)


def _energy(X):
    """Kinetic-energy proxy: peak movable-shape speed plus centroid speed. X: (N, F)."""
    return X[:, 4] + VEL_W * jnp.sqrt(X[:, 2] ** 2 + X[:, 3] ** 2)


def _roll(mem, key, B):
    """Draw a fresh drive pattern (mode, motor, thruster, period, feedback axis, direction bit) per worker."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    u = jax.random.normal(k5, (B, 2), jnp.float32)
    return (jax.random.randint(k1, (B,), 0, N_MODE),
            2 * jax.random.randint(k2, (B,), 0, mem["nm"]),
            mem["thr"][jax.random.randint(k3, (B,), 0, mem["thr"].shape[0])],
            mem["per"][jax.random.randint(k4, (B,), 0, mem["per"].shape[0])],
            u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-6),
            jax.random.randint(k6, (B,), 0, 2))


def init(rng, F, n_act, K, B):
    nm = max(1, (n_act - 3) // 2)               # motor bindings: n_act = 2*NM + 1 (no-op) + NT thrusters
    noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    S = max(1, K // NB)
    mem = {"n_act": jnp.int32(n_act), "nm": jnp.int32(nm), "noop": jnp.int32(noop),
           "thr": jnp.array(thr, jnp.int32), "per": jnp.array(PERIODS, jnp.int32),
           "S": jnp.int32(S), "K": jnp.int32(K),
           "ptr": jnp.zeros(NB, jnp.int32), "cnt": jnp.zeros(NB, jnp.int32),
           "bbest": jnp.full(NB, -1e6, jnp.float32),
           "ph": jnp.zeros(B, jnp.int32), "pb": jnp.full((B,), -1e6, jnp.float32),
           "stall": jnp.zeros(B, jnp.int32), "just": jnp.zeros(B, bool)}
    mode, mot, th, per, u, d = _roll(mem, rng, B)
    mem.update({"mode": mode, "mot": mot, "tha": th, "prd": per, "u": u, "dbit": d})
    return mem


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    hb = jnp.argmax(jnp.where(mem["cnt"] > 0, jnp.arange(NB), -1))          # highest occupied band
    band = jnp.where(jax.random.uniform(k1, (B,)) < TOP_FRAC, hb, jax.random.randint(k2, (B,), 0, NB))
    n = jnp.maximum(jnp.minimum(mem["cnt"][band], mem["S"]), 1)
    slot = jnp.clip(band * mem["S"] + jax.random.randint(k3, (B,), 0, 1 << 20) % n, 0, K - 1)
    do = ((jax.random.uniform(k4, (B,)) < P_LADDER) | (mem["stall"] > PATIENCE)) & arch_valid.any()
    return do, slot, {**mem, "just": do, "stall": jnp.where(do, 0, mem["stall"]),
                      "pb": jnp.where(do, jnp.float32(-1e6), mem["pb"])}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3 = jax.random.split(rng, 3)
    E = _energy(feats)
    rec = E > mem["pb"] + 1e-3
    stall = jnp.where(rec, 0, mem["stall"] + 1)
    reroll = mem["just"] | (mem["ph"] >= EPOCH) | (stall > PATIENCE)
    mode, mot, tha, prd, u, dbit = _roll(mem, k1, B)
    mode = jnp.where(reroll, mode, mem["mode"]); mot = jnp.where(reroll, mot, mem["mot"])
    tha = jnp.where(reroll, tha, mem["tha"]); prd = jnp.where(reroll, prd, mem["prd"])
    u = jnp.where(reroll[:, None], u, mem["u"]); dbit = jnp.where(reroll, dbit, mem["dbit"])
    ph = jnp.where(reroll, 0, mem["ph"])
    half = (ph // prd) % 2
    pump = (feats[:, 2] * u[:, 0] + feats[:, 3] * u[:, 1] < 0).astype(jnp.int32)
    a = jnp.where(mode == 0, mot + half,
        jnp.where(mode == 1, tha,
        jnp.where(mode == 2, jnp.where(half == 0, tha, mem["noop"]),
        jnp.where(mode == 3, mot + pump,
                  jnp.where(ph % 2 == 0, tha, mot + dbit)))))
    a = jnp.where(jax.random.uniform(k2, (B,)) < 0.05, jax.random.randint(k3, (B,), 0, mem["n_act"]), a)
    return a.astype(jnp.int32), {**mem, "mode": mode, "mot": mot, "tha": tha, "prd": prd, "u": u, "dbit": dbit,
                                 "ph": ph + 1, "pb": jnp.maximum(mem["pb"], E),
                                 "stall": jnp.where(reroll, 0, stall), "just": jnp.zeros(B, bool)}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    E = _energy(feats)
    band = jnp.clip((jnp.log2(1.0 + jnp.maximum(E, 0.0)) * BPO).astype(jnp.int32), 0, NB - 1)
    rank = ((band[:, None] == band[None, :]) & (jnp.arange(B)[:, None] > jnp.arange(B)[None, :])).sum(-1)
    slot = jnp.clip(band * mem["S"] + (mem["ptr"][band] + rank) % mem["S"], 0, K - 1)
    store = (~done) & ((mem["cnt"][band] < mem["S"]) | (E > mem["bbest"][band] - 0.02)
                       | (jax.random.uniform(rng, (B,)) < P_STORE))
    inc = store.astype(jnp.int32)
    ptr = mem["ptr"].at[band].add(inc)
    cnt = jnp.minimum(mem["cnt"].at[band].add(inc), mem["S"])
    bbest = mem["bbest"].at[band].max(jnp.where(store, E, jnp.float32(-1e6)))
    return store, slot, {**mem, "ptr": ptr, "cnt": cnt, "bbest": bbest}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
