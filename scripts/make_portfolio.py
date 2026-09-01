"""Build a self-contained portfolio tester: worker groups run different sub-programs; the archive is partitioned so each
sub-program keeps its own bookkeeping. Usage: python scripts/make_portfolio.py out.py prog1.py prog2.py ..."""
import sys, re
args = [x for x in sys.argv[1:] if not x.startswith("--")]; ADAPTIVE = "--adaptive" in sys.argv
out, files = args[0], args[1:]
srcs = []
for f in files:
    s = open(f).read()
    s = s[: s.index("def run_experiment")] if "def run_experiment" in s else s
    srcs.append(s)
run_exp = open("evolve/initial.py").read(); run_exp = run_exp[run_exp.index("def run_experiment"):]
body = '''"""Portfolio tester: %d sub-programs (%s). Worker i starts on sub-program i %% n; archive slots are partitioned per sub-program; if ADAPTIVE, a bandit reallocates workers toward the sub-programs currently discovering new coarse feature cells."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
_SRC = %r
_P = []
for _s in _SRC:
    _ns = {"jax": jax, "jnp": jnp}
    exec(_s, _ns)
    _P.append(_ns)
N_SUB = len(_P)


def _sel(outs_b, group):
    """outs_b: list of (B,) arrays, one per sub-program -> per-worker selection by group."""
    st = jnp.stack(outs_b)
    return st[group, jnp.arange(group.shape[0])]


ADAPTIVE = %s
REALLOC = 64        # steps between worker reallocations (adaptive only)
MIN_SHARE = 0.10    # floor on each sub-program's share of workers
EMA = 0.02          # smoothing of the per-group novelty rate
TEMP = 0.7          # softmax temperature on the standardized novelty rate
CELL_RES = 0.3      # coarse cell size (m) for the shared novelty hash
NB = 14             # hash bits of the shared novelty table


def _cell(feats):
    """Program-agnostic novelty cell: coarse dyn_pos, dyn_vel, green_pos."""
    cols = jnp.array([0, 1, 2, 3, 5, 6]); scale = jnp.array([CELL_RES, CELL_RES, 1.0, 1.0, CELL_RES, CELL_RES])
    q = jnp.floor(feats[:, cols] / scale).astype(jnp.int32)
    h = (q * jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], dtype=jnp.int32)).sum(-1)
    return h & ((1 << NB) - 1)


def init(rng, F, n_act, K, B):
    keys = jax.random.split(rng, N_SUB)
    Ki = K // N_SUB
    return {"sub": [_P[i]["init"](keys[i], F, n_act, Ki, B) for i in range(N_SUB)], "group": jnp.arange(B) %% N_SUB, "Ki": jnp.int32(Ki),
            "credit": jnp.zeros(N_SUB), "t": jnp.int32(0), "seen": jnp.zeros(1 << NB, dtype=bool)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    keys = jax.random.split(rng, N_SUB); Ki = arch_valid.shape[0] // N_SUB
    dos, slots, subs = [], [], []
    for i in range(N_SUB):
        sl = slice(i * Ki, (i + 1) * Ki)
        d, s, m = _P[i]["restore"](mem["sub"][i], feats, arch_feats[sl], arch_valid[sl], arch_visits[sl], keys[i])
        dos.append(d); slots.append(jnp.clip(s, 0, Ki - 1) + i * Ki); subs.append(m)
    g = mem["group"]
    return _sel(dos, g), _sel(slots, g), {**mem, "sub": subs}


def act(mem, feats, obs, rng):
    keys = jax.random.split(rng, N_SUB); acts, subs = [], []
    for i in range(N_SUB):
        a, m = _P[i]["act"](mem["sub"][i], feats, obs, keys[i]); acts.append(a); subs.append(m)
    return _sel(acts, mem["group"]), {**mem, "sub": subs}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    keys = jax.random.split(rng, N_SUB + 1); Ki = arch_valid.shape[0] // N_SUB
    dos, slots, subs = [], [], []
    for i in range(N_SUB):
        sl = slice(i * Ki, (i + 1) * Ki)
        d, s, m = _P[i]["archive"](mem["sub"][i], feats, done, arch_feats[sl], arch_valid[sl], keys[i])
        # only this sub-program's own workers may store into its partition
        d = d & (mem["group"] == i)
        dos.append(d); slots.append(jnp.clip(s, 0, Ki - 1) + i * Ki); subs.append(m)
    g = mem["group"]
    out = {**mem, "sub": subs}
    if ADAPTIVE:
        # bandit: each group's novelty rate = fraction of its workers that entered a never-seen coarse cell this step
        cell = _cell(feats); new = ~mem["seen"][cell]; seen = mem["seen"].at[cell].set(True)
        cnt = jnp.stack([(g == i).sum() for i in range(N_SUB)]).astype(jnp.float32)
        nov = jnp.stack([(new & (g == i)).sum() for i in range(N_SUB)]).astype(jnp.float32) / jnp.maximum(cnt, 1.0)
        credit = (1 - EMA) * mem["credit"] + EMA * nov
        t = mem["t"] + 1
        z = (credit - credit.mean()) / (credit.std() + 1e-6)
        p = MIN_SHARE + (1 - N_SUB * MIN_SHARE) * jax.nn.softmax(z / TEMP)
        new_g = jax.random.categorical(keys[N_SUB], jnp.log(p)[None, :].repeat(g.shape[0], 0), axis=-1).astype(g.dtype)
        out.update(group=jnp.where(t %% REALLOC == 0, new_g, g), credit=credit, t=t, seen=seen)
    return _sel(dos, g), _sel(slots, g), out
# EVOLVE-BLOCK-END


''' % (len(files), ", ".join(files), srcs, ADAPTIVE) + run_exp
open(out, "w").write(body); print(f"wrote {out} with {len(files)} sub-programs")
