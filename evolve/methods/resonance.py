"""M15 — resonant square-wave drive with a period sweep and an energy ratchet (miss diagnosis M4/M5): a worker drives one
actuator with a square wave (two opposed actions, half-period p), sweeps p over a log-spaced set across bouts, credits each
period by the growth of the moving bodies' energy E = v^2 + 2 g y during the bout, and re-uses the best period with
probability P_BEST; the archive keeps the K highest-energy states seen (energy ratchet) and restores go to the top-energy
slots, so pumping resumes from the most energetic checkpoints instead of from rest. Thruster workers hold the thruster."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
PERIODS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
BOUT = 128; P_BEST = 0.6; P_RESTORE = 0.04; G = 9.8
NM = 4
def init(rng, F, n_act, K, B):
    k1, k2 = jax.random.split(rng)
    return {"n_act": jnp.int32(n_act), "t": jnp.int32(0), "motor": jax.random.randint(k1, (B,), 0, NM), "pidx": jax.random.randint(k2, (B,), 0, len(PERIODS)),
            "credit": jnp.zeros((B, len(PERIODS))), "e0": jnp.zeros(B), "thr": jnp.zeros(B, bool), "slot_e": jnp.full(K, -jnp.inf)}
def _energy(feats):
    return feats[:, 4] ** 2 + 2 * G * feats[:, 1]
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    logits = jnp.where(arch_valid, mem["slot_e"] / 20.0 - 0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k1, (B, K)), axis=1)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    return do, slot, mem
def act(mem, feats, obs, rng):
    B = feats.shape[0]; t = mem["t"]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    new_bout = (t % BOUT) == 0
    e = _energy(feats)
    # credit the finished bout's period by the energy gained
    gain = e - mem["e0"]
    credit = mem["credit"].at[jnp.arange(B), mem["pidx"]].set(jnp.where(new_bout, 0.7 * mem["credit"][jnp.arange(B), mem["pidx"]] + 0.3 * gain, mem["credit"][jnp.arange(B), mem["pidx"]]))
    best = jnp.argmax(credit + 1e-3 * jax.random.gumbel(k1, credit.shape), axis=1)
    rnd = jax.random.randint(k2, (B,), 0, len(PERIODS))
    pidx = jnp.where(new_bout, jnp.where(jax.random.uniform(k3, (B,)) < P_BEST, best, rnd), mem["pidx"])
    motor = jnp.where(new_bout, jax.random.randint(k4, (B,), 0, NM), mem["motor"])
    thr = jnp.where(new_bout, jax.random.uniform(k1, (B,)) < 0.2, mem["thr"])
    p = jnp.array(PERIODS)[pidx]
    phase = ((t % BOUT) // jnp.maximum(p, 1)) % 2
    action = 2 * motor + phase                                             # motor +/- square wave
    thr_action = jnp.where(mem["n_act"] > 2 * NM + 1, 2 * NM + 1 + (motor % jnp.maximum(mem["n_act"] - 2 * NM - 1, 1)), action)
    action = jnp.where(thr, thr_action, action)
    e0 = jnp.where(new_bout, e, mem["e0"])
    return action, {**mem, "t": t + 1, "credit": credit, "pidx": pidx, "motor": motor, "thr": thr, "e0": e0}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    e = jnp.where(done, -jnp.inf, _energy(feats))
    worst = jnp.argmin(jnp.where(arch_valid, mem["slot_e"], -jnp.inf)); worst_e = jnp.where(arch_valid.any(), mem["slot_e"][worst], -jnp.inf)
    # the single best worker of the step replaces the worst slot if it beats it (one writer per step keeps the ratchet exact)
    b = jnp.argmax(e); do = jnp.zeros(B, bool).at[b].set(e[b] > worst_e)
    slot = jnp.where(do, worst, 0)
    slot_e = mem["slot_e"].at[worst].set(jnp.where(do.any(), e[b], mem["slot_e"][worst]))
    return do, slot, {**mem, "slot_e": slot_e}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
