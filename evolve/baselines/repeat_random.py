"""Action-repeat random ('sticky actions') as a hooks program: identical to autodev.tester.RepeatRandomProgram (max_rep 12),
never restores and never archives. A fixed-harness fitness reference for the search runs (drop into the fitness queue)."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
MAX_REP = 12
def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    left = jnp.where(new, jax.random.randint(k2, (B,), 1, MAX_REP + 1), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)