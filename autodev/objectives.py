"""Exploration objectives with one interface:
   init(rng, feat_dim, n_actions) -> state
   reward(state, f_prev, f_next, actions) -> (r (B,), state)       # per environment step, batched over envs
   update(state, F_prev, F_next, A) -> (state, metrics)             # after each rollout, batched over T*B
Baselines: zero (task return only), count (SimHash), rnd, icm, curiosity_proximity (CCPT-style form).
`ProgramObjective` wraps an evolved program (the EVOLVE-BLOCK target)."""
from __future__ import annotations
import jax, jax.numpy as jnp, optax


def _mlp_init(rng, sizes):
    params = []
    for i in range(len(sizes) - 1):
        rng, k = jax.random.split(rng)
        w = jax.random.normal(k, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / sizes[i])
        params.append((w, jnp.zeros(sizes[i + 1])))
    return params


def _mlp(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def _norm_feats(f):
    # crude fixed scaling so positions (~0-5), velocities (~0-10) and flags live on similar scales
    return jnp.tanh(f / 5.0)


class ZeroObjective:
    name = "zero"
    def init(self, rng, feat_dim, n_actions): return {}
    def reward(self, state, f_prev, f_next, actions): return jnp.zeros(f_next.shape[0]), state
    def update(self, state, F_prev, F_next, A): return state, {}


class CountObjective:
    """SimHash count-based novelty (Tang et al. 2017): r = 1/sqrt(n(hash(features)))."""
    name = "count"
    def __init__(self, bits=16, keys=None):
        self.bits = bits; self.keys = keys  # keys: optional list of feature slices to hash (default: agent pos+vel)
    def init(self, rng, feat_dim, n_actions):
        proj = jax.random.normal(rng, (feat_dim, self.bits))
        return {"proj": proj, "counts": jnp.zeros(2 ** self.bits, jnp.float32)}
    def _idx(self, state, f):
        bits = (_norm_feats(f) @ state["proj"]) > 0
        return (bits * (2 ** jnp.arange(self.bits))).sum(-1).astype(jnp.int32)
    def reward(self, state, f_prev, f_next, actions):
        idx = self._idx(state, f_next)
        counts = state["counts"].at[idx].add(1.0)
        r = 1.0 / jnp.sqrt(counts[idx])
        return r, {**state, "counts": counts}
    def update(self, state, F_prev, F_next, A): return state, {}


class RNDObjective:
    """Random Network Distillation (Burda et al. 2018) over features."""
    name = "rnd"
    def __init__(self, hidden=64, out=32, lr=1e-3): self.h, self.o, self.lr = hidden, out, lr
    def init(self, rng, feat_dim, n_actions):
        k1, k2 = jax.random.split(rng)
        target = _mlp_init(k1, [feat_dim, self.h, self.o]); pred = _mlp_init(k2, [feat_dim, self.h, self.o])
        opt = optax.adam(self.lr)
        return {"target": target, "pred": pred, "opt": opt.init(pred), "rms": jnp.array([0.0, 1.0, 0.0])}  # mean, var, count
    def _err(self, state, f):
        x = _norm_feats(f)
        return ((_mlp(state["pred"], x) - jax.lax.stop_gradient(_mlp(state["target"], x))) ** 2).mean(-1)
    def reward(self, state, f_prev, f_next, actions):
        e = self._err(state, f_next)
        mean, var, n = state["rms"]
        r = e / jnp.sqrt(var + 1e-8)
        return r, state
    def update(self, state, F_prev, F_next, A):
        opt = optax.adam(self.lr)
        def loss(p): return self._err({**state, "pred": p}, F_next).mean()
        l, g = jax.value_and_grad(loss)(state["pred"])
        upd, opt_state = opt.update(g, state["opt"], state["pred"])
        pred = optax.apply_updates(state["pred"], upd)
        e = self._err(state, F_next)
        mean, var, n = state["rms"]; m = F_next.shape[0]
        new_mean = (mean * n + e.mean() * m) / (n + m); new_var = (var * n + e.var() * m) / (n + m)
        return {**state, "pred": pred, "opt": opt_state, "rms": jnp.array([new_mean, new_var, n + m])}, {"rnd_loss": l}


class ICMObjective:
    """Intrinsic Curiosity Module (Pathak et al. 2017): forward-model error in a learned feature space."""
    name = "icm"
    def __init__(self, emb=32, hidden=64, lr=1e-3, beta=0.2): self.e, self.h, self.lr, self.beta = emb, hidden, lr, beta
    def init(self, rng, feat_dim, n_actions):
        k1, k2, k3 = jax.random.split(rng, 3)
        params = {"enc": _mlp_init(k1, [feat_dim, self.h, self.e]), "fwd": _mlp_init(k2, [self.e + n_actions, self.h, self.e]),
                  "inv": _mlp_init(k3, [2 * self.e, self.h, n_actions])}
        opt = optax.adam(self.lr)
        self.n_actions = n_actions
        return {"params": params, "opt": opt.init(params), "rms": jnp.array([0.0, 1.0, 0.0])}
    def _fwd_err(self, params, f_prev, f_next, a, n_actions):
        z0, z1 = _mlp(params["enc"], _norm_feats(f_prev)), _mlp(params["enc"], _norm_feats(f_next))
        a1 = jax.nn.one_hot(a, n_actions)
        z1_hat = _mlp(params["fwd"], jnp.concatenate([z0, a1], -1))
        return ((z1_hat - jax.lax.stop_gradient(z1)) ** 2).mean(-1), z0, z1
    def reward(self, state, f_prev, f_next, actions):
        e, _, _ = self._fwd_err(state["params"], f_prev, f_next, actions, self.n_actions)
        return e / jnp.sqrt(state["rms"][1] + 1e-8), state
    def update(self, state, F_prev, F_next, A):
        opt = optax.adam(self.lr); nA = self.n_actions
        def loss(p):
            e, z0, z1 = self._fwd_err(p, F_prev, F_next, A, nA)
            logits = _mlp(p["inv"], jnp.concatenate([z0, z1], -1))
            inv = optax.softmax_cross_entropy_with_integer_labels(logits, A).mean()
            return (1 - self.beta) * inv + self.beta * e.mean()
        l, g = jax.value_and_grad(loss)(state["params"])
        upd, opt_state = opt.update(g, state["opt"], state["params"])
        params = optax.apply_updates(state["params"], upd)
        e, _, _ = self._fwd_err(params, F_prev, F_next, A, nA)
        mean, var, n = state["rms"]; m = e.shape[0]
        return {**state, "params": params, "opt": opt_state, "rms": jnp.array([(mean * n + e.mean() * m) / (n + m), (var * n + e.var() * m) / (n + m), n + m])}, {"icm_loss": l}


class CuriosityProximityObjective(CountObjective):
    """CCPT-style form (Sestini et al. 2022) re-implemented without demonstrations: count novelty plus a
    proximity term that rewards progress toward the intended goal, so exploration stays near task-relevant states."""
    name = "curiosity_proximity"
    def __init__(self, bits=16, beta=0.5): super().__init__(bits); self.beta = beta
    def reward(self, state, f_prev, f_next, actions):
        r, state = super().reward(state, f_prev, f_next, actions)
        progress = f_prev[:, 7] - f_next[:, 7]   # green_blue_dist is feature index 7 (see features spec)
        return r + self.beta * progress, state


class ProgramObjective:
    """An evolved objective: a Python source string defining
         def init_memory(feat_dim: int) -> memory (pytree of jnp arrays)
         def reward(f_prev, f_next, actions, memory) -> (r (B,), memory)
    with only jax/jnp available. This is the EVOLVE-BLOCK target."""
    name = "program"
    def __init__(self, source: str):
        self.source = source
        ns = {"jax": jax, "jnp": jnp}
        exec(source, ns)
        self._init, self._reward = ns["init_memory"], ns["reward"]
    def init(self, rng, feat_dim, n_actions): return {"mem": self._init(feat_dim)}
    def reward(self, state, f_prev, f_next, actions):
        r, mem = self._reward(f_prev, f_next, actions, state["mem"])
        return jnp.nan_to_num(r.astype(jnp.float32)), {**state, "mem": mem}
    def update(self, state, F_prev, F_next, A): return state, {}


BASELINES = {"zero": ZeroObjective, "count": CountObjective, "rnd": RNDObjective, "icm": ICMObjective, "curiosity_proximity": CuriosityProximityObjective}
