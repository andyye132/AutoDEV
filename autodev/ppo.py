"""PPO tester training with a pluggable exploration objective. `make_train` returns train(rng, level, meta) -> results,
which is pure JAX and can be vmapped over defect instances (levels and metas stacked along a leading axis)."""
from __future__ import annotations
import jax, jax.numpy as jnp, optax
from flax import struct
from .oracles import witness
from .objectives import _mlp_init, _mlp


@struct.dataclass
class PPOConfig:
    num_envs: int = 64
    rollout: int = 64
    num_updates: int = 60
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    hidden: int = 128
    ext_coef: float = 0.0     # weight on the game's own reward (task return)
    int_coef: float = 1.0     # weight on the exploration objective

    @property
    def budget(self):
        return self.num_updates * self.rollout * self.num_envs


def make_train(env, ep, sp, engine, objective, feat_fn, feat_dim, cfg: PPOConfig):
    n_act = env.action_space(ep).n
    obs_dim = env.observation_space(ep).shape[0]
    B, T = cfg.num_envs, cfg.rollout
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))

    def policy(params, obs):
        return _mlp(params["pi"], obs), _mlp(params["v"], obs)[..., 0]

    def train(rng, level, meta):
        rng, k_pi, k_v, k_obj, k_reset = jax.random.split(rng, 5)
        params = {"pi": _mlp_init(k_pi, [obs_dim, cfg.hidden, cfg.hidden, n_act]), "v": _mlp_init(k_v, [obs_dim, cfg.hidden, cfg.hidden, 1])}
        opt_state = opt.init(params)
        obj_state = objective.init(k_obj, feat_dim, n_act)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(jax.random.split(k_reset, B), ep, level)
        feats = jax.vmap(feat_fn)(env_state)
        wit = jax.vmap(lambda s, r, d: witness(s, r, d, meta, engine))
        reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))

        def env_step(carry, t_and_u):
            params, obj_state, env_state, obs, feats, rng, first_w, n_eps, n_solved = carry
            t, u = t_and_u
            rng, k_a, k_s, k_r = jax.random.split(rng, 4)
            logits, value = policy(params, obs)
            action = jax.random.categorical(k_a, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
            obs2, st2, r_ext, done, info = step_fn(jax.random.split(k_s, B), env_state, action, ep, level)
            feats2 = jax.vmap(feat_fn)(st2)
            w = wit(st2, r_ext, done)
            r_int, obj_state = objective.reward(obj_state, feats, feats2, action)
            reward = cfg.ext_coef * r_ext + cfg.int_coef * r_int
            obs_r, st_r = reset_fn(jax.random.split(k_r, B), ep, level)
            sel = lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
            st_n = jax.tree.map(sel, st_r, st2)
            obs_n = sel(obs_r, obs2)
            feats_n = jax.vmap(feat_fn)(st_n)
            gstep = ((u * T + t) * B).astype(jnp.float32)
            first_w = jnp.where(w.any() & (gstep < first_w), gstep, first_w)
            n_eps = n_eps + done.sum(); n_solved = n_solved + (done & (r_ext > 0.5)).sum()
            tr = dict(obs=obs, action=action, logp=logp, value=value, reward=reward, done=done, feats=feats, feats2=feats2, w=w, r_int=r_int, r_ext=r_ext)
            return (params, obj_state, st_n, obs_n, feats_n, rng, first_w, n_eps, n_solved), tr

        def gae(rewards, values, dones, last_val):
            def body(carry, x):
                adv, next_val = carry
                r, v, d = x
                delta = r + cfg.gamma * next_val * (1 - d) - v
                adv = delta + cfg.gamma * cfg.gae_lambda * (1 - d) * adv
                return (adv, v), adv
            _, advs = jax.lax.scan(body, (jnp.zeros_like(last_val), last_val), (rewards, values, dones.astype(jnp.float32)), reverse=True)
            return advs, advs + values

        def ppo_update(params, opt_state, batch, rng):
            obs, action, logp_old, value_old, adv, target = batch
            def loss_fn(p, obs, action, logp_old, value_old, adv, target):
                logits, value = policy(p, obs)
                logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
                ratio = jnp.exp(logp - logp_old)
                adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_n).mean()
                v_clip = value_old + jnp.clip(value - value_old, -cfg.clip_eps, cfg.clip_eps)
                vf = 0.5 * jnp.maximum((value - target) ** 2, (v_clip - target) ** 2).mean()
                ent = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
                return pg + cfg.vf_coef * vf - cfg.ent_coef * ent, (pg, vf, ent)
            n = T * B; mb = n // cfg.minibatches
            def epoch(carry, _):
                params, opt_state, rng = carry
                rng, k = jax.random.split(rng)
                perm = jax.random.permutation(k, n)
                shuffled = jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:])[perm], batch)
                mbs = jax.tree.map(lambda x: x.reshape((cfg.minibatches, mb) + x.shape[1:]), shuffled)
                def minibatch(carry, mbatch):
                    params, opt_state = carry
                    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, *mbatch)
                    upd, opt_state = opt.update(g, opt_state, params)
                    return (optax.apply_updates(params, upd), opt_state), (l, aux)
                (params, opt_state), stats = jax.lax.scan(minibatch, (params, opt_state), mbs)
                return (params, opt_state, rng), stats
            (params, opt_state, rng), stats = jax.lax.scan(epoch, (params, opt_state, rng), None, length=cfg.epochs)
            return params, opt_state, jax.tree.map(jnp.mean, stats)

        def update(runner, u):
            params, opt_state, obj_state, env_state, obs, feats, rng, first_w, n_eps, n_solved = runner
            carry, traj = jax.lax.scan(env_step, (params, obj_state, env_state, obs, feats, rng, first_w, n_eps, n_solved),
                                       (jnp.arange(T), jnp.full((T,), u)))
            params, obj_state, env_state, obs, feats, rng, first_w, n_eps, n_solved = carry
            _, last_val = policy(params, obs)
            adv, target = gae(traj["reward"], traj["value"], traj["done"], last_val)
            rng, k = jax.random.split(rng)
            params, opt_state, stats = ppo_update(params, opt_state, (traj["obs"], traj["action"], traj["logp"], traj["value"], adv, target), k)
            F = feat_dim
            obj_state, obj_metrics = objective.update(obj_state, traj["feats"].reshape(-1, F), traj["feats2"].reshape(-1, F), traj["action"].reshape(-1))
            metrics = dict(loss=stats[0], pg=stats[1][0], vf=stats[1][1], ent=stats[1][2], r_int=traj["r_int"].mean(), r_ext=traj["r_ext"].mean(),
                           witness_any=traj["w"].any(), done_rate=traj["done"].mean())
            return (params, opt_state, obj_state, env_state, obs, feats, rng, first_w, n_eps, n_solved), metrics

        runner = (params, opt_state, obj_state, env_state, obs, feats, rng, jnp.float32(jnp.inf), jnp.int32(0), jnp.int32(0))
        runner, metrics = jax.lax.scan(update, runner, jnp.arange(cfg.num_updates))
        first_w, n_eps, n_solved = runner[7], runner[8], runner[9]
        return dict(first_witness=jnp.where(jnp.isfinite(first_w), first_w, jnp.float32(cfg.budget)), witnessed=jnp.isfinite(first_w),
                    solved_frac=n_solved / jnp.maximum(n_eps, 1), n_episodes=n_eps, metrics=metrics)
    return train


def make_train_substrate(env, ep, oracle, objective, feat_fn, feat_dim, cfg: PPOConfig):
    """Substrate-generic PPO: the policy runs on the SAME feat_fn(state) vector the program testers see (so RL and the
    LLM tester get identical observations — a fairer comparison and no dependence on a per-engine observation_space), and
    it uses a generic 4-arg oracle(state, reward, done, meta) instead of Kinetix's 5-arg witness. Otherwise identical PPO."""
    n_act = env.action_space(ep).n
    B, T = cfg.num_envs, cfg.rollout
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))

    def policy(params, x):
        return _mlp(params["pi"], x), _mlp(params["v"], x)[..., 0]

    def train(rng, level, meta):
        rng, k_pi, k_v, k_obj, k_reset = jax.random.split(rng, 5)
        params = {"pi": _mlp_init(k_pi, [feat_dim, cfg.hidden, cfg.hidden, n_act]),
                  "v": _mlp_init(k_v, [feat_dim, cfg.hidden, cfg.hidden, 1])}
        opt_state = opt.init(params)
        obj_state = objective.init(k_obj, feat_dim, n_act)
        _obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(jax.random.split(k_reset, B), ep, level)
        feats = jax.vmap(feat_fn)(env_state)
        wit = jax.vmap(lambda s, r, d: oracle(s, r, d, meta))
        reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))

        def env_step(carry, t_and_u):
            params, obj_state, env_state, feats, rng, first_w, n_eps, n_solved = carry
            t, u = t_and_u
            rng, k_a, k_s, k_r = jax.random.split(rng, 4)
            logits, value = policy(params, feats)
            action = jax.random.categorical(k_a, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
            _o2, st2, r_ext, done, info = step_fn(jax.random.split(k_s, B), env_state, action, ep, level)
            feats2 = jax.vmap(feat_fn)(st2)
            w = wit(st2, r_ext, done)                       # (B, n_flags)
            r_int, obj_state = objective.reward(obj_state, feats, feats2, action)
            reward = cfg.ext_coef * r_ext + cfg.int_coef * r_int
            _or, st_r = reset_fn(jax.random.split(k_r, B), ep, level)
            sel = lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
            st_n = jax.tree.map(sel, st_r, st2)
            feats_n = jax.vmap(feat_fn)(st_n)
            gstep = ((u * T + t) * B).astype(jnp.float32)
            first_w = jnp.where(w.any() & (gstep < first_w), gstep, first_w)
            n_eps = n_eps + done.sum(); n_solved = n_solved + (done & (r_ext > 0.5)).sum()
            tr = dict(feats=feats, action=action, logp=logp, value=value, reward=reward, done=done,
                      feats_o=feats, feats2=feats2, w=w, r_int=r_int, r_ext=r_ext)
            return (params, obj_state, st_n, feats_n, rng, first_w, n_eps, n_solved), tr

        def gae(rewards, values, dones, last_val):
            def body(carry, x):
                adv, next_val = carry; r, v, d = x
                delta = r + cfg.gamma * next_val * (1 - d) - v
                adv = delta + cfg.gamma * cfg.gae_lambda * (1 - d) * adv
                return (adv, v), adv
            _, advs = jax.lax.scan(body, (jnp.zeros_like(last_val), last_val),
                                   (rewards, values, dones.astype(jnp.float32)), reverse=True)
            return advs, advs + values

        def ppo_update(params, opt_state, batch, rng):
            x, action, logp_old, value_old, adv, target = batch
            def loss_fn(p, x, action, logp_old, value_old, adv, target):
                logits, value = policy(p, x)
                logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
                ratio = jnp.exp(logp - logp_old)
                adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_n).mean()
                v_clip = value_old + jnp.clip(value - value_old, -cfg.clip_eps, cfg.clip_eps)
                vf = 0.5 * jnp.maximum((value - target) ** 2, (v_clip - target) ** 2).mean()
                ent = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
                return pg + cfg.vf_coef * vf - cfg.ent_coef * ent, (pg, vf, ent)
            n = T * B; mb = n // cfg.minibatches
            def epoch(carry, _):
                params, opt_state, rng = carry
                rng, k = jax.random.split(rng)
                perm = jax.random.permutation(k, n)
                shuffled = jax.tree.map(lambda z: z.reshape((n,) + z.shape[2:])[perm], batch)
                mbs = jax.tree.map(lambda z: z.reshape((cfg.minibatches, mb) + z.shape[1:]), shuffled)
                def minibatch(carry, mbatch):
                    params, opt_state = carry
                    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, *mbatch)
                    upd, opt_state = opt.update(g, opt_state, params)
                    return (optax.apply_updates(params, upd), opt_state), (l, aux)
                (params, opt_state), stats = jax.lax.scan(minibatch, (params, opt_state), mbs)
                return (params, opt_state, rng), stats
            (params, opt_state, rng), stats = jax.lax.scan(epoch, (params, opt_state, rng), None, length=cfg.epochs)
            return params, opt_state, jax.tree.map(jnp.mean, stats)

        def update(runner, u):
            params, opt_state, obj_state, env_state, feats, rng, first_w, n_eps, n_solved = runner
            carry, traj = jax.lax.scan(env_step, (params, obj_state, env_state, feats, rng, first_w, n_eps, n_solved),
                                       (jnp.arange(T), jnp.full((T,), u)))
            params, obj_state, env_state, feats, rng, first_w, n_eps, n_solved = carry
            _, last_val = policy(params, feats)
            adv, target = gae(traj["reward"], traj["value"], traj["done"], last_val)
            rng, k = jax.random.split(rng)
            params, opt_state, stats = ppo_update(params, opt_state, (traj["feats"], traj["action"], traj["logp"], traj["value"], adv, target), k)
            F = feat_dim
            obj_state, obj_metrics = objective.update(obj_state, traj["feats_o"].reshape(-1, F), traj["feats2"].reshape(-1, F), traj["action"].reshape(-1))
            metrics = dict(loss=stats[0], r_int=traj["r_int"].mean(), r_ext=traj["r_ext"].mean(), witness_any=traj["w"].any(), done_rate=traj["done"].mean())
            return (params, opt_state, obj_state, env_state, feats, rng, first_w, n_eps, n_solved), metrics

        runner = (params, opt_state, obj_state, env_state, feats, rng, jnp.float32(jnp.inf), jnp.int32(0), jnp.int32(0))
        runner, metrics = jax.lax.scan(update, runner, jnp.arange(cfg.num_updates))
        first_w, n_eps, n_solved = runner[6], runner[7], runner[8]
        return dict(first_witness=jnp.where(jnp.isfinite(first_w), first_w, jnp.float32(cfg.budget)),
                    witnessed=jnp.isfinite(first_w), solved_frac=n_solved / jnp.maximum(n_eps, 1), n_episodes=n_eps, metrics=metrics)
    return train
