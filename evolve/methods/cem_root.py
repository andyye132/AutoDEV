"""M11 — synchronised cross-entropy method from a shifting root (falsification-style optimisation that only the restore
primitive makes possible): all workers are periodically restored to ONE root state, each executes a plan of H macro-actions
sampled from a shared categorical distribution, the plan fitness is the maximum class-agnostic importance reached, the elite
plans update the distribution, and the elite's terminal state becomes the next root. Restore makes the B fitnesses comparable
(same start), which open-loop CEM cannot achieve. A reserve of workers keeps free-running with zeta holds."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
H = 8; HOLD = 16                   # window = 128 steps
ELITE_FRAC = 0.15; ALPHA = 0.5; EPS = 0.04
RESERVE_FRAC = 0.25                # free-running workers (never restored) with zeta holds
MU = 2.0; N_MAX = 120
WORLD = 5.0
W_DIST, W_SPEED, W_BOUND, W_GOAL = 1.0, 0.6, 0.8, 0.4
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _importance(feats, start):
    d_travel = jnp.linalg.norm(feats[:, 0:2] - start[:, 0:2], axis=-1) / WORLD
    speed = jnp.minimum(feats[:, 4] / 8.0, 1.5)
    edge = jnp.minimum(jnp.minimum(feats[:, 0], WORLD - feats[:, 0]), jnp.minimum(feats[:, 1], WORLD - feats[:, 1]))
    near_bound = 1.0 - jnp.clip(edge / (0.5 * WORLD), 0.0, 1.0)
    goal = 1.0 - jnp.clip(feats[:, 7] / WORLD, 0.0, 1.0)
    return W_DIST * d_travel + W_SPEED * speed + W_BOUND * near_bound + W_GOAL * goal
def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "probs": jnp.full((H, n_act), 1.0 / n_act), "plan": jnp.zeros((B, H), jnp.int32), "t": jnp.int32(0),
            "obj": jnp.zeros(B), "start": jnp.zeros((B, F)), "have_start": jnp.bool_(False), "root": jnp.int32(0), "have_root": jnp.bool_(False),
            "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "elite_w": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; W = H * HOLD
    start = jnp.where(mem["have_start"], mem["start"], feats)
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    new_window = (mem["t"] % W) == 0
    do = new_window & mem["have_root"] & ~reserve & arch_valid[mem["root"]]
    slot = jnp.full(B, mem["root"], jnp.int32)
    return do, slot, {**mem, "start": start, "have_start": jnp.bool_(True)}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; t = mem["t"]; W = H * HOLD; k1, k2, k3 = jax.random.split(rng, 3)
    new_window = (t % W) == 0
    n_act = mem["probs"].shape[1]; keys = jax.random.split(k1, H)
    sampled = jnp.stack([jax.random.choice(keys[h], n_act, (B,), p=mem["probs"][h]) for h in range(H)], axis=1)
    plan = jnp.where(new_window, sampled, mem["plan"]); obj = jnp.where(new_window, 0.0, mem["obj"])
    h = (t % W) // HOLD; planned = plan[jnp.arange(B), h]
    # reserve workers: zeta holds
    new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k2, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k3, (B,))); left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    action = jnp.where(reserve, cur, planned)
    return action, {**mem, "plan": plan, "obj": obj, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; t = mem["t"]; W = H * HOLD
    obj = jnp.maximum(mem["obj"], _importance(feats, mem["start"]))
    reserve = jnp.arange(B) < jnp.int32(RESERVE_FRAC * B)
    end_window = (t % W) == (W - 1)
    n_elite = max(1, int(ELITE_FRAC * B))
    scored = jnp.where(reserve | done, -jnp.inf, obj)
    order = jnp.argsort(-scored); elite = jnp.zeros(B, bool).at[order[:n_elite]].set(True) & ~reserve & ~done
    counts = jnp.zeros((H, mem["probs"].shape[1])).at[jnp.arange(H)[None, :].repeat(B, 0), mem["plan"]].add(elite[:, None].astype(jnp.float32))
    freq = counts / jnp.maximum(counts.sum(-1, keepdims=True), 1.0)
    new_probs = jnp.maximum((1 - ALPHA) * mem["probs"] + ALPHA * freq, EPS); new_probs = new_probs / new_probs.sum(-1, keepdims=True)
    probs = jnp.where(end_window, new_probs, mem["probs"])
    # the elite's terminal state becomes the next root (store into slot (t // W + 1) % K); the very first root is the reset state at t=0
    best_w = order[0]; root_slot = ((t // W) + 1) % K
    do = jnp.zeros(B, bool).at[best_w].set(end_window & ~done[best_w])
    do0 = (t == 0) & (jnp.arange(B) == 0) & ~done                            # initial root: worker 0's state after step 0
    do = do | do0; slot = jnp.where(do0, 0, root_slot)
    root = jnp.where(end_window, root_slot, jnp.where(t == 0, 0, mem["root"]))
    return do, jnp.where(do, slot, 0), {**mem, "obj": obj, "probs": probs, "t": t + 1, "root": root, "have_root": mem["have_root"] | do.any()}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
