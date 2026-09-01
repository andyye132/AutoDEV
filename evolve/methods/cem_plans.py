"""M5 — cross-entropy method over open-loop action-hold plans (falsification-style: Rubinstein 1999; S-TaLiRo/Breach use
CEM/SA over input signals). Time is cut into windows of H macro-steps of HOLD steps each. At the start of a window every worker
samples a plan (H actions) from a shared categorical distribution; during the window each worker tracks the running maximum of a
class-agnostic importance (travel, speed, boundary and goal proximity). At the window end the elite workers' plans update the
distribution (smoothed frequency update), so the population concentrates on macro-action sequences that produce extreme
states. No restores; the physics is explored open-loop like the strongest non-restoring baseline, but with adaptation."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
H = 8               # macro-steps per plan window
HOLD = 16           # steps per macro-step (window = H*HOLD = 128 steps)
ELITE_FRAC = 0.2
ALPHA = 0.5         # smoothing of the distribution update
EPS = 0.05          # floor probability per action (keeps exploration alive)
WORLD = 5.0
W_DIST, W_SPEED, W_BOUND, W_GOAL = 1.0, 0.6, 0.8, 0.4
def _importance(feats, start):
    d_travel = jnp.linalg.norm(feats[:, 0:2] - start[:, 0:2], axis=-1) / WORLD
    speed = jnp.minimum(feats[:, 4] / 8.0, 1.5)
    edge = jnp.minimum(jnp.minimum(feats[:, 0], WORLD - feats[:, 0]), jnp.minimum(feats[:, 1], WORLD - feats[:, 1]))
    near_bound = 1.0 - jnp.clip(edge / (0.5 * WORLD), 0.0, 1.0)
    goal = 1.0 - jnp.clip(feats[:, 7] / WORLD, 0.0, 1.0)
    return W_DIST * d_travel + W_SPEED * speed + W_BOUND * near_bound + W_GOAL * goal
def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "probs": jnp.full((H, n_act), 1.0 / n_act), "plan": jnp.zeros((B, H), jnp.int32),
            "t": jnp.int32(0), "obj": jnp.zeros(B), "start": jnp.zeros((B, F)), "have_start": jnp.bool_(False)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]
    start = jnp.where(mem["have_start"], mem["start"], feats)
    return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), {**mem, "start": start, "have_start": jnp.bool_(True)}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; t = mem["t"]; W = H * HOLD
    new_window = (t % W) == 0
    # sample a fresh plan for every worker at the start of a window
    keys = jax.random.split(rng, H)
    n_act = mem["probs"].shape[1]   # static (jax.random.choice needs a concrete population size)
    sampled = jnp.stack([jax.random.choice(keys[h], n_act, (B,), p=mem["probs"][h]) for h in range(H)], axis=1)
    plan = jnp.where(new_window, sampled, mem["plan"])
    obj = jnp.where(new_window, 0.0, mem["obj"])
    h = (t % W) // HOLD
    action = plan[jnp.arange(B), h]
    return action, {**mem, "plan": plan, "obj": obj}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; t = mem["t"]; W = H * HOLD
    obj = jnp.maximum(mem["obj"], _importance(feats, mem["start"]))
    end_window = (t % W) == (W - 1)
    n_elite = max(1, int(ELITE_FRAC * B))
    order = jnp.argsort(-obj); elite = jnp.zeros(B, bool).at[order[:n_elite]].set(True)
    counts = jnp.zeros((H, mem["probs"].shape[1])).at[jnp.arange(H)[None, :].repeat(B, 0), mem["plan"]].add(elite[:, None].astype(jnp.float32))
    freq = counts / jnp.maximum(counts.sum(-1, keepdims=True), 1.0)
    new_probs = (1 - ALPHA) * mem["probs"] + ALPHA * freq
    new_probs = jnp.maximum(new_probs, EPS); new_probs = new_probs / new_probs.sum(-1, keepdims=True)
    probs = jnp.where(end_window, new_probs, mem["probs"])
    return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), {**mem, "obj": obj, "probs": probs, "t": t + 1}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
