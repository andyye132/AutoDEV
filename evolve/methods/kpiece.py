"""M2 — KPIECE-style archive (Sucan & Kavraki, "Kinodynamic planning by interior-exterior cell exploration", WAFR 2008).
The state space is projected onto a coarse grid (position of the moving bodies, agent position, speed band). Each archived
state belongs to a cell; a cell's importance is  importance = (1 + neighbours_empty) / ((1 + coverage) * (1 + selections)),
i.e. exterior cells (few occupied neighbours) that are lightly covered and rarely selected are expanded first. Expansion =
restore a worker to a state in a cell chosen with probability proportional to importance, then apply a random control for a
random duration (heavy-tailed, as KPIECE does with random durations). Everything is vectorised: cells are hashed into a
table of counts; the archive keeps one representative state per cell (first-in), plus a rolling set of recent states."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12
CELL_DIMS = (0, 1, 2, 3, 5, 6)          # dyn_pos, dyn_vel, agent_pos
CELL_RES = (0.5, 0.5, 1.0, 1.0, 0.5, 0.5)
P_RESTORE = 0.05                        # per-step expansion probability per worker
MU = 2.0; N_MAX = 120                   # heavy-tailed hold durations
EXT_W = 1.0                             # weight of the exterior term (empty neighbours in position)
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def _pos_cells(mem, feats, dx, dy):
    f = feats.at[:, 0].add(dx * CELL_RES[0]).at[:, 1].add(dy * CELL_RES[1]); return _cell(mem, f)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], jnp.int32)[: len(CELL_DIMS)]
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult,
            "coverage": jnp.zeros(1 << BITS, jnp.int32), "selections": jnp.zeros(1 << BITS, jnp.int32),
            "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "slot_cell": jnp.zeros(K, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    cells = mem["slot_cell"]                                                # cell of every archived slot
    cov = mem["coverage"][cells].astype(jnp.float32); sel = mem["selections"][cells].astype(jnp.float32)
    # exterior term: number of empty neighbouring position cells of the archived state (4-neighbourhood)
    nb = sum(((mem["coverage"][_pos_cells(mem, arch_feats, dx, dy)] == 0).astype(jnp.float32)) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    imp = (1.0 + EXT_W * nb) / ((1.0 + cov) * (1.0 + sel))
    imp = jnp.where(arch_valid, imp, 0.0); p = imp / jnp.maximum(imp.sum(), 1e-9)
    slot = jax.random.choice(k1, K, (B,), p=p)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    sel_upd = mem["selections"].at[jnp.where(do, cells[slot], 0)].add(do.astype(jnp.int32))
    return do, slot, {**mem, "selections": sel_upd, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    cell = _cell(mem, feats); live = ~done
    coverage = mem["coverage"].at[jnp.where(live, cell, 0)].add(live.astype(jnp.int32))
    is_new = (mem["cell_slot"][cell] < 0) & live
    # one representative per new cell; dedupe new cells within the step by keeping the lowest worker index per cell
    first = jnp.full(1 << BITS, B, jnp.int32).at[jnp.where(is_new, cell, 0)].min(jnp.where(is_new, jnp.arange(B), B))
    is_new = is_new & (first[cell] == jnp.arange(B))
    order = jnp.cumsum(is_new) - 1; slot = (mem["next_free"] + order) % K
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    slot_cell = mem["slot_cell"].at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, cell, mem["slot_cell"][0]))
    return is_new, jnp.where(is_new, slot, 0), {**mem, "coverage": coverage, "cell_slot": cell_slot, "slot_cell": slot_cell, "next_free": (mem["next_free"] + is_new.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
