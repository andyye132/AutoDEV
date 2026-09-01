"""Tester-program harness. B workers explore one level for T steps. States can be archived into K slots and restored
(Kinetix states are plain pytrees). A *program* supplies three hooks and its own memory:

    init(rng, F, n_actions, K, B)                                -> mem
    restore(mem, feats, arch_feats, arch_valid, arch_visits, rng) -> (do_restore (B,), slot (B,), mem)
    act(mem, feats, obs, rng)                                     -> (action (B,), mem)
    archive(mem, feats, done, arch_feats, arch_valid, rng)        -> (do_store (B,), slot (B,), mem)

The harness records, for the first oracle hit, the worker's action buffer and the archive chain back to the level
reset, so any reported bug can be re-simulated deterministically (see replay_chain)."""
from __future__ import annotations
import os
import jax, jax.numpy as jnp, numpy as np
from flax import struct


@struct.dataclass
class TesterConfig:
    num_workers: int = 64
    steps: int = 512
    slots: int = 256
    max_ep: int = 256
    chain_log: int = 0            # >0: append-only lineage log with this many entries (scans); 0: slot-pointer lineage (cheap, fitness)
    min_streak: tuple = ()   # per-flag consecutive-step persistence required to count a hit (default 1 for every flag)
    return_archive: bool = False     # return the final archive + lineage log (debug/audit)
    dedupe_writers: bool = True      # False reproduces the pre-Aug-29 race (several workers scattering into one slot) for audits ONLY
    invariants: bool = False         # count invariant violations each step (chimera slots, inherited witnesses, duplicate writers); ~15% slower

    @property
    def budget(self):
        return self.num_workers * self.steps


COVERAGE_BITS = 16


def coverage_cells(feats):
    """Coarse state cells for the coverage metric: dyn_pos (0.25 m), dyn_vel (1 m/s), green_pos (0.25 m) -> hash in [0, 2^16)."""
    q = jnp.floor(feats[:, jnp.array([0, 1, 2, 3, 5, 6])] / jnp.array([0.25, 0.25, 1.0, 1.0, 0.25, 0.25])).astype(jnp.int32)
    h = (q * jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], dtype=jnp.int32)).sum(-1)
    return h & ((1 << COVERAGE_BITS) - 1)


def make_tester(env, ep, sp, feat_fn, program, cfg: TesterConfig, oracle, n_flags: int, guide=None):
    """oracle(state, reward, done, oracle_params) -> (n_flags,) bool. Returns run(rng, level, oracle_params) -> dict."""
    B, T, K, L = cfg.num_workers, cfg.steps, cfg.slots, cfg.max_ep
    n_act = env.action_space(ep).n
    reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
    step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    get_obs = jax.vmap(env.get_obs)
    F = None

    def run(rng, level, oracle_params):
        if guide is not None:   # feasibility probe: append a guide scalar (progress toward the witness) as the last feature
            base_feat_fn = feat_fn
            def feat_fn_g(s): return jnp.concatenate([base_feat_fn(s), guide(s, oracle_params)[None].astype(jnp.float32)])
        else:
            feat_fn_g = feat_fn
        rng, k0, k1 = jax.random.split(rng, 3)
        obs, st = reset_fn(jax.random.split(k0, B), ep, level)
        feats = jax.vmap(feat_fn_g)(st)
        Fdim = feats.shape[-1]
        mem = program.init(k1, Fdim, n_act, K, B)
        # archive with one extra trash slot (index K)
        arch_state = jax.tree.map(lambda x: jnp.broadcast_to(x[:1], (K + 1,) + x.shape[1:]).copy(), st)
        arch_valid = jnp.zeros(K + 1, bool); arch_feats = jnp.zeros((K + 1, Fdim)); arch_visits = jnp.zeros(K + 1, jnp.int32)
        arch_parent = jnp.full(K + 1, -1, jnp.int32); arch_seg = jnp.zeros((K + 1, L), jnp.int32); arch_seglen = jnp.zeros(K + 1, jnp.int32)
        S = int(cfg.chain_log)
        log_parent = jnp.full(S + 1, -1, jnp.int32); log_seg = jnp.zeros((S + 1, L), jnp.int8); log_seglen = jnp.zeros(S + 1, jnp.int32)   # entry S = trash
        arch_logidx = jnp.full(K + 1, -1, jnp.int32); log_next = jnp.int32(0)
        w_origin = jnp.full(B, -1, jnp.int32); w_buf = jnp.zeros((B, L), jnp.int32); w_len = jnp.zeros(B, jnp.int32)
        first_hit = jnp.full(n_flags, jnp.inf)
        found = jnp.bool_(False); wfound = jnp.zeros((B, n_flags), bool)   # per-worker, per-flag 'raised at some step' (used by the exact verifier)
        min_streak = jnp.array(list(cfg.min_streak) + [1] * (n_flags - len(cfg.min_streak)), jnp.int32)[:n_flags]
        streak = jnp.zeros((B, n_flags), jnp.int32)
        hit = dict(flag=jnp.int32(-1), worker=jnp.int32(-1), t=jnp.int32(-1), origin=jnp.int32(-1), buf=jnp.zeros(L, jnp.int32), len=jnp.int32(0),
                   parent=arch_parent, seg=arch_seg, seglen=arch_seglen, state=jax.tree.map(lambda x: x[0], st),
                   log_parent=log_parent, log_seg=log_seg, log_seglen=log_seglen, log_size=jnp.int32(S))
        seen = jnp.zeros(1 << COVERAGE_BITS, bool)      # coverage: coarse cells visited by any worker (secondary metric)
        inv = jnp.zeros(3, jnp.int32)                   # invariant violations: [chimera slots, witness inherited on restore, duplicate writers]

        def body(carry, t):
            (rng, mem, st, obs, feats, arch_state, arch_valid, arch_feats, arch_visits, arch_parent, arch_seg, arch_seglen,
             w_origin, w_buf, w_len, first_hit, found, hit, streak, seen, arch_logidx, log_parent, log_seg, log_seglen, log_next, inv, wfound) = carry
            rng, kr, ka, ks, kc = jax.random.split(rng, 5)
            # 1. restore
            do_r, slot, mem = program.restore(mem, feats, arch_feats[:K], arch_valid[:K], arch_visits[:K], kr)
            slot = jnp.clip(slot, 0, K - 1); do_r = do_r & arch_valid[slot]
            sel = lambda a, b: jnp.where(do_r.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
            st = jax.tree.map(sel, jax.tree.map(lambda x: x[slot], arch_state), st)
            obs = jnp.where(do_r.reshape((-1,) + (1,) * (obs.ndim - 1)), get_obs(st), obs)     # obs may be multi-dimensional (grid worlds)
            feats = jnp.where(do_r[:, None], jax.vmap(feat_fn_g)(st), feats)
            w_origin = jnp.where(do_r, (arch_logidx[slot] if S > 0 else slot), w_origin); w_len = jnp.where(do_r, 0, w_len)
            streak = jnp.where(do_r[:, None], 0, streak)
            arch_visits = arch_visits.at[jnp.where(do_r, slot, K)].add(1)
            if cfg.invariants:
                w_restored = jax.vmap(lambda s_: oracle(s_, jnp.float32(0.0), jnp.bool_(False), oracle_params).any())(st)
                inv = inv.at[1].add(jnp.where(~found, (do_r & w_restored).sum(), 0))
            # 2. act
            action, mem = program.act(mem, feats, obs, ka)
            action = jnp.clip(action, 0, n_act - 1).astype(jnp.int32)
            # 3. step + oracle
            obs2, st2, r, done, info = step_fn(jax.random.split(ks, B), st, action, ep, level)
            feats2 = jax.vmap(feat_fn_g)(st2)
            seen = seen.at[coverage_cells(feats2)].set(True)
            raw_flags = jax.vmap(lambda s, rr, dd: oracle(s, rr, dd, oracle_params))(st2, r, done)   # (B, n_flags)
            streak = jnp.where(raw_flags, streak + 1, 0)
            flags = streak >= min_streak[None, :]
            # 4. record actions (with the lineage log, a full buffer is flushed into the log so long episodes stay replayable)
            if S > 0:
                full = w_len >= L; order_f = jnp.cumsum(full) - 1
                entry_f = jnp.where(full, (log_next + order_f) % S, S)
                log_parent = log_parent.at[entry_f].set(jnp.where(full, w_origin, log_parent[entry_f]))
                log_seg = log_seg.at[entry_f].set(jnp.where(full[:, None], w_buf.astype(jnp.int8), log_seg[entry_f]))
                log_seglen = log_seglen.at[entry_f].set(jnp.where(full, w_len, log_seglen[entry_f]))
                w_origin = jnp.where(full, entry_f, w_origin); w_len = jnp.where(full, 0, w_len); log_next = (log_next + full.sum()) % S
            w_buf = w_buf.at[jnp.arange(B), jnp.minimum(w_len, L - 1)].set(action); w_len = jnp.minimum(w_len + 1, L)
            # 5. hits
            any_f = flags.any(0)
            first_hit = jnp.where(any_f & (t * B < first_hit), (t * B).astype(jnp.float32), first_hit)
            new_hit = flags.any() & ~found; wfound = wfound | flags
            wk = jnp.argmax(flags.any(1)); fl = jnp.argmax(flags[wk])
            cand = dict(flag=fl.astype(jnp.int32), worker=wk.astype(jnp.int32), t=t.astype(jnp.int32), origin=w_origin[wk], buf=w_buf[wk], len=w_len[wk],
                        parent=arch_parent, seg=arch_seg, seglen=arch_seglen, state=jax.tree.map(lambda x: x[wk], st2),
                        log_parent=log_parent, log_seg=log_seg, log_seglen=log_seglen, log_size=jnp.int32(S))
            hit = jax.tree.map(lambda a, b: jnp.where(new_hit, a, b), cand, hit)
            found = found | new_hit
            # 6. archive
            do_a, slot_a, mem = program.archive(mem, feats2, done, arch_feats[:K], arch_valid[:K], kc)
            slot_a = jnp.where(do_a & ~done, jnp.clip(slot_a, 0, K - 1), K)
            # several workers may target the same slot in one step (e.g. 'replace the worst of a pool'); separate scatters would then
            # take state, features and lineage from different workers, so keep exactly one writer per slot (the lowest worker index)
            if cfg.dedupe_writers and os.environ.get("AUTODEV_LEGACY_RACE") != "1":   # env override: audits of the pre-fix behaviour only
                first_w = jnp.full(K + 1, B, jnp.int32).at[slot_a].min(jnp.arange(B))
                slot_a = jnp.where(first_w[slot_a] == jnp.arange(B), slot_a, K)
            arch_state = jax.tree.map(lambda a, b: a.at[slot_a].set(b), arch_state, st2)
            arch_feats = arch_feats.at[slot_a].set(feats2); arch_valid = arch_valid.at[slot_a].set(True)
            arch_parent = arch_parent.at[slot_a].set(w_origin); arch_seg = arch_seg.at[slot_a].set(w_buf); arch_seglen = arch_seglen.at[slot_a].set(w_len)
            if S > 0:
                storing = slot_a < K; order = jnp.cumsum(storing) - 1
                entry = jnp.where(storing, (log_next + order) % S, S)              # non-storing workers write into the trash entry S
                dummy_ok = jnp.where(storing, 1, 0)
                log_parent = log_parent.at[entry].set(jnp.where(storing, w_origin, log_parent[entry]))
                log_seg = log_seg.at[entry].set(jnp.where(storing[:, None], w_buf.astype(jnp.int8), log_seg[entry]))
                log_seglen = log_seglen.at[entry].set(jnp.where(storing, w_len, log_seglen[entry]))
                arch_logidx = arch_logidx.at[slot_a].set(jnp.where(storing, entry, arch_logidx[slot_a]))
                log_next = (log_next + storing.sum()) % S
            arch_valid = arch_valid.at[K].set(False)
            if cfg.invariants:
                counts = jnp.zeros(K + 1, jnp.int32).at[slot_a].add(1); inv = inv.at[2].add((counts[:K] > 1).sum())
                re = jax.vmap(feat_fn_g)(jax.tree.map(lambda x: x[slot_a], arch_state))
                bad = (slot_a < K) & (jnp.abs(re - arch_feats[slot_a]).max(-1) > 1e-3); inv = inv.at[0].add(bad.sum())
            # 7. reset finished workers (only when some worker is done: world generation is expensive on some engines)
            def _reset_done(_):
                obs_r, st_r = reset_fn(jax.random.split(kc, B), ep, level)
                seld = lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
                return jax.tree.map(seld, st_r, st2), seld(obs_r, obs2)
            st, obs = jax.lax.cond(done.any(), _reset_done, lambda _: (st2, obs2), None)
            feats = jax.vmap(feat_fn_g)(st)
            w_origin = jnp.where(done, -1, w_origin); w_len = jnp.where(done, 0, w_len)
            streak = jnp.where(done[:, None], 0, streak)
            return (rng, mem, st, obs, feats, arch_state, arch_valid, arch_feats, arch_visits, arch_parent, arch_seg, arch_seglen,
                    w_origin, w_buf, w_len, first_hit, found, hit, streak, seen, arch_logidx, log_parent, log_seg, log_seglen, log_next, inv, wfound), (any_f, done.mean())

        carry = (rng, mem, st, obs, feats, arch_state, arch_valid, arch_feats, arch_visits, arch_parent, arch_seg, arch_seglen,
                 w_origin, w_buf, w_len, first_hit, found, hit, streak, seen, arch_logidx, log_parent, log_seg, log_seglen, log_next, inv, wfound)
        carry, (hits_t, done_rate) = jax.lax.scan(body, carry, jnp.arange(T))
        first_hit, found, hit = carry[15], carry[16], carry[17]
        out = dict(first_hit=jnp.where(jnp.isfinite(first_hit), first_hit, jnp.float32(cfg.budget)), found=found, hit=hit,
                   n_valid_slots=carry[6][:K].sum(), done_rate=done_rate.mean(), hits_per_step=hits_t, coverage=carry[19].sum(), invariants=carry[25], worker_found=carry[26])
        if cfg.return_archive:   # debugging / lineage audits: the final archive and lineage log
            out["archive"] = dict(state=carry[5], valid=carry[6], feats=carry[7], parent=carry[9], seg=carry[10], seglen=carry[11],
                                  logidx=carry[20], log_parent=carry[21], log_seg=carry[22], log_seglen=carry[23], log_next=carry[24])
        return out
    return run


def chain_actions(hit) -> list[int]:
    """Host-side: reconstruct the action sequence from level reset to the flagged state. With a lineage log (scans), parents are
    log entries that are never overwritten; without it, parents are archive slots (valid only if no slot was overwritten)."""
    S = int(np.array(hit.get("log_size", 0)))
    if S > 0:
        parent = np.array(hit["log_parent"]); seg = np.array(hit["log_seg"]); seglen = np.array(hit["log_seglen"])
    else:
        parent = np.array(hit["parent"]); seg = np.array(hit["seg"]); seglen = np.array(hit["seglen"])
    segs = []
    s = int(hit["origin"])
    guard = 0
    while s >= 0 and guard < 100000:
        segs.append(list(seg[s][: seglen[s]])); s = int(parent[s]); guard += 1
    actions = [a for sg in reversed(segs) for a in sg] + list(np.array(hit["buf"])[: int(hit["len"])])
    return [int(a) for a in actions]


_REPLAY_CACHE = {}


def replay_chain(env, ep, level, actions, oracle, oracle_params, jit=True):
    """Deterministically re-simulate actions from the level reset; returns (flags_over_time (n, n_flags), final_state).
    Jitted as a lax.scan over the action chain padded to a power-of-two bucket (no-ops); flags are sliced to the true length.
    The returned state is the state after the padded no-ops (nothing downstream uses it)."""
    n = len(actions)
    if not jit or n == 0:
        rng = jax.random.PRNGKey(0)
        obs, st = env.reset(rng, ep, override_reset_state=level)
        flags = []
        for a in actions:
            obs, st, r, done, info = env.step(rng, st, jnp.int32(a), ep, level)
            flags.append(np.array(oracle(st, r, done, oracle_params)))
            if bool(done):
                obs, st = env.reset(rng, ep, override_reset_state=level)   # a chain never crosses a terminal step by construction
        return np.array(flags), st
    L = max(64, 1 << (n - 1).bit_length())
    noop = int(env.static_env_params.num_motor_bindings) * 2 if hasattr(env, "static_env_params") else int(actions[0])
    key = (id(env), id(oracle), L)
    fn = _REPLAY_CACHE.get(key)
    if fn is None:
        def _fn(level, oracle_params, acts):
            rng = jax.random.PRNGKey(0)
            obs, st = env.reset(rng, ep, override_reset_state=level)
            def body(st, a):
                obs, st2, r, done, info = env.step(rng, st, a, ep, level)
                return st2, oracle(st2, r, done, oracle_params)
            st_f, flags = jax.lax.scan(body, st, acts)
            return flags, st_f
        fn = _REPLAY_CACHE[key] = jax.jit(_fn)
    acts = jnp.array(list(actions) + [noop] * (L - n), dtype=jnp.int32)
    flags, st_f = fn(level, oracle_params, acts)
    return np.array(flags)[:n], st_f


_REPLAY_BATCH_CACHE = {}


def replay_chains(env, ep, levels, actions_list, oracle, params_list, max_batch=16):
    """Batched, jitted replay of several chains at once (vmap over chains; actions padded with no-ops to a power-of-two
    bucket, batch padded to `max_batch` by repetition so there is one compile per bucket). Returns a list of per-chain
    flags arrays sliced to each chain's true length. Equivalent to calling replay_chain on each chain."""
    H = len(actions_list)
    if H == 0: return []
    n_max = max(len(a) for a in actions_list); L = max(64, 1 << (max(n_max, 1) - 1).bit_length())
    noop = int(env.static_env_params.num_motor_bindings) * 2
    key = (id(env), id(oracle), L, max_batch)
    fn = _REPLAY_BATCH_CACHE.get(key)
    if fn is None:
        def _one(level, oracle_params, acts):
            rng = jax.random.PRNGKey(0)
            obs, st = env.reset(rng, ep, override_reset_state=level)
            def body(st, a):
                obs, st2, r, done, info = env.step(rng, st, a, ep, level)
                return st2, oracle(st2, r, done, oracle_params)
            _, flags = jax.lax.scan(body, st, acts)
            return flags
        fn = _REPLAY_BATCH_CACHE[key] = jax.jit(jax.vmap(_one, in_axes=(0, 0, 0)))
    out = []
    for c0 in range(0, H, max_batch):
        idx = list(range(c0, min(H, c0 + max_batch))); pad = idx + [idx[-1]] * (max_batch - len(idx))
        lv = jax.tree.map(lambda *x: jnp.stack(x), *[levels[i] for i in pad])
        pr = jax.tree.map(lambda *x: jnp.stack(x), *[params_list[i] for i in pad])
        acts = jnp.array([list(actions_list[i]) + [noop] * (L - len(actions_list[i])) for i in pad], dtype=jnp.int32)
        flags = np.array(fn(lv, pr, acts))
        out += [flags[j][:len(actions_list[i])] for j, i in enumerate(idx)]
    return out


_REPLAY_STATES_CACHE = {}


def replay_states(env, ep, level, actions, window=0):
    """Jitted replay returning the stacked states after each action (n,) and after each of `window` no-op continuation steps,
    plus the per-step done flags. Padded to a power-of-two bucket so few compiles are needed; slices are exact."""
    n = len(actions); m = n + window; L = max(64, 1 << (m - 1).bit_length())
    noop = int(env.static_env_params.num_motor_bindings) * 2
    key = (id(env), L)
    fn = _REPLAY_STATES_CACHE.get(key)
    if fn is None:
        def _fn(level, acts):
            rng = jax.random.PRNGKey(0)
            obs, st = env.reset(rng, ep, override_reset_state=level)
            def body(st, a):
                obs, st2, r, done, info = env.step(rng, st, a, ep, level)
                return st2, (st2, done)
            _, (states, dones) = jax.lax.scan(body, st, acts)
            return states, dones
        fn = _REPLAY_STATES_CACHE[key] = jax.jit(_fn)
    acts = jnp.array(list(actions) + [noop] * (L - n), dtype=jnp.int32)
    states, dones = fn(level, acts)
    states = jax.tree.map(lambda x: x[:m], states); dones = np.array(dones[:m])
    return [jax.tree.map(lambda x: x[t], states) for t in range(m)], dones     # jax arrays, as env.step would return


# ---------------- built-in programs ----------------
class RandomProgram:
    name = "random"
    def init(self, rng, F, n_act, K, B): self.n_act = n_act; return {}
    def restore(self, mem, feats, af, av, avis, rng):
        B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem
    def act(self, mem, feats, obs, rng): return jax.random.randint(rng, (feats.shape[0],), 0, self.n_act), mem
    def archive(self, mem, feats, done, af, av, rng):
        B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem


class RepeatRandomProgram(RandomProgram):
    """Random actions held for random durations (1..max_rep) — the standard 'sticky actions' tester."""
    name = "repeat_random"
    def __init__(self, max_rep=12): self.max_rep = max_rep
    def init(self, rng, F, n_act, K, B):
        self.n_act = n_act
        return {"cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32)}
    def act(self, mem, feats, obs, rng):
        B = feats.shape[0]
        cur, left = mem["cur"], mem["left"]
        k1, k2 = jax.random.split(rng)
        new = left <= 0
        cur = jnp.where(new, jax.random.randint(k1, (B,), 0, self.n_act), cur)
        left = jnp.where(new, jax.random.randint(k2, (B,), 1, self.max_rep + 1), left) - 1
        return cur, {**mem, "cur": cur, "left": left}


class GoExploreProgram(RepeatRandomProgram):
    """Go-Explore-style archive search: cells = SimHash of coarse (position, velocity) features; store new cells;
    with probability p_restore a worker jumps to a stored cell chosen with weight 1/sqrt(visits)."""
    name = "goexplore"
    def __init__(self, bits=12, p_restore=0.05, max_rep=12, feat_idx=(0, 1, 2, 3, 5, 6, 9, 10)):
        super().__init__(max_rep); self.bits, self.p, self.idx = bits, p_restore, jnp.array(feat_idx)
    def init(self, rng, F, n_act, K, B):
        m = super().init(rng, F, n_act, K, B)
        m.update({"proj": jax.random.normal(rng, (len(self.idx), self.bits)), "cell_slot": jnp.full(2 ** self.bits, -1, jnp.int32),
                  "next_free": jnp.int32(0), "K": jnp.int32(K)})
        return m
    def _cell(self, mem, feats):
        x = jnp.round(feats[:, self.idx] * 2.0) / 2.0   # coarse grid (0.5 units)
        return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(self.bits))).sum(-1).astype(jnp.int32)
    def restore(self, mem, feats, af, av, avis, rng):
        B = feats.shape[0]; k1, k2 = jax.random.split(rng)
        do = jax.random.uniform(k1, (B,)) < self.p
        logits = jnp.where(av, -0.5 * jnp.log1p(avis.astype(jnp.float32)), -jnp.inf)
        g = jax.random.gumbel(k2, (B, av.shape[0]))
        slot = jnp.argmax(logits[None, :] + g, axis=1)
        return do & av.any(), slot, mem
    def archive(self, mem, feats, done, af, av, rng):
        B = feats.shape[0]
        cell = self._cell(mem, feats)
        is_new = (mem["cell_slot"][cell] < 0) & ~done
        # allocate slots round-robin for new cells (duplicates within a step collapse to the last writer)
        order = jnp.cumsum(is_new) - 1
        slot = (mem["next_free"] + order) % mem["K"]
        cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
        return is_new, slot, {**mem, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new.sum()) % mem["K"]}


BUILTIN_PROGRAMS = {"random": RandomProgram, "repeat_random": RepeatRandomProgram, "goexplore": GoExploreProgram}


class SourceProgram:
    """A program defined by a Python source string with functions init/restore/act/archive (jax, jnp in scope)."""
    name = "source"
    def __init__(self, source: str):
        ns = {"jax": jax, "jnp": jnp}
        exec(source, ns)
        self._f = {k: ns[k] for k in ("init", "restore", "act", "archive")}
    def init(self, rng, F, n_act, K, B): return self._f["init"](rng, F, n_act, K, B)
    def restore(self, mem, feats, af, av, avis, rng): return self._f["restore"](mem, feats, af, av, avis, rng)
    def act(self, mem, feats, obs, rng): return self._f["act"](mem, feats, obs, rng)
    def archive(self, mem, feats, done, af, av, rng): return self._f["archive"](mem, feats, done, af, av, rng)


class ScriptedTable:
    """A program that replays a fixed action table on every worker and never restores or archives (exact verification)."""
    def __init__(self, table): self.table = table
    def init(self, rng, F, n_act, K, B): return {"t": jnp.int32(0)}
    def restore(self, mem, feats, af, av, avis, rng): B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem
    def act(self, mem, feats, obs, rng): B = feats.shape[0]; return jnp.full((B,), self.table[jnp.minimum(mem["t"], self.table.shape[0] - 1)], jnp.int32), {"t": mem["t"] + 1}
    def archive(self, mem, feats, done, af, av, rng): B = feats.shape[0]; return jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), mem


_HARNESS_REPLAY_CACHE = {}
def verify_chain_harness(env, ep, sp, feat_fn, level, actions, oracle, oracle_params, B, n_flags=1, min_streak=()):
    """Exact replay: run the harness itself (same scan body, same number of lanes B as the run that produced the chain) with a
    scripted program that plays `actions` from the level reset; returns True if any worker raised a witness at any step.
    XLA fuses the physics differently in the standalone step and at other lane counts, so only this reproduces bit-for-bit."""
    n = len(actions)
    if n == 0: return False
    L = 1 << (n - 1).bit_length()
    noop = int(env.static_env_params.num_motor_bindings) * 2 if hasattr(env, "static_env_params") else int(actions[0])
    key = (id(env), id(oracle), B, L, n_flags, tuple(min_streak))
    if key not in _HARNESS_REPLAY_CACHE:
        cfg = TesterConfig(num_workers=B, steps=L, slots=1, min_streak=tuple(min_streak), chain_log=0)
        def run(lvl, params, table):
            return make_tester(env, ep, sp, feat_fn, ScriptedTable(table), cfg, oracle, n_flags)(jax.random.PRNGKey(0), lvl, params)["worker_found"]
        _HARNESS_REPLAY_CACHE[key] = jax.jit(run)
    table = jnp.array(list(actions) + [noop] * (L - n), dtype=jnp.int32)
    per_flag = np.array(_HARNESS_REPLAY_CACHE[key](level, oracle_params, table)).any(0)     # (n_flags,) raised by any lane at any step
    return per_flag if n_flags > 1 else bool(per_flag.any())
