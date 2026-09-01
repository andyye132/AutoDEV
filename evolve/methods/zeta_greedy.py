"""M1 — temporally-extended exploration with heavy-tailed hold durations (Dabney, Ostrovski & Barreto, "Temporally-Extended
epsilon-Greedy Exploration", ICLR 2021): each worker repeats a uniformly random action for n steps with n ~ zeta(mu), so most holds
are short but a heavy tail produces the long, coherent holds that build energy and cross rooms. No archive, no restores: the
minimal upgrade of action-repeat random (whose holds are uniform on 1..12)."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
MU = 2.0          # zeta exponent (Dabney et al. use mu = 2); smaller = heavier tail
N_MAX = 200       # truncation of the duration distribution
def _zeta_table():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _zeta_table()}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))            # n ~ zeta(MU), truncated at N_MAX
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
