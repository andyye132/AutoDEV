"""M9 — contact-state branching (speedrunning "corner clipping" and QA seam-probing, Ariyurek/Inspector-style targeted
investigation): states where contacts have just begun (or two contacts are simultaneous) are archived with priority, and
restores concentrate on them; each restored worker then holds a DIFFERENT action (its worker index modulo the action count),
so a contact state is branched across all 11 controls at once — a vectorised one-step tree over the rare states where
solver mis-resolution, tunnelling and unintended goal touches happen. Elsewhere: heavy-tailed random holds and a coverage archive."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12; CELL_DIMS = (0, 1, 2, 3, 5, 6); CELL_RES = (0.5, 0.5, 1.0, 1.0, 0.5, 0.5)
P_RESTORE = 0.05
CONTACT_BONUS = 2.0        # restore-logit bonus for archived contact states
BRANCH_HOLD = 12           # steps the branched action is held after a contact restore
MU = 2.0; N_MAX = 120
LAYOUT = {}
def _layout(F):
    """Per-shape blocks from the feature width (features.py): 11 head scalars; poly pos/vel/rot/active/role (7 Np); circle
    pos/vel/active/role (6 Nc); contact flags (poly-poly pairs minus the 6 wall-wall pairs, circle-poly, circle-circle)."""
    if F in LAYOUT: return LAYOUT[F]
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41) if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    Np, Nc = sols[0] if len(sols) == 1 else (0, 0)
    C = 11 + 7 * Np
    LAYOUT[F] = dict(Np=Np, Nc=Nc, ppos=11, pvel=11 + 2 * Np, pact=11 + 5 * Np, prole=11 + 6 * Np, cpos=C, cvel=C + 2 * Nc, cact=C + 4 * Nc, crole=C + 5 * Nc, cfrom=C + 6 * Nc)
    return LAYOUT[F]
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def _ncontact(feats):
    return (feats[:, _layout(feats.shape[1])["cfrom"]:] > 0.5).sum(-1)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], jnp.int32)[: len(CELL_DIMS)]
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult,
            "nc_prev": jnp.zeros(B, jnp.int32), "is_contact": jnp.zeros(K, bool), "branch": jnp.zeros(B, bool),
            "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)) + CONTACT_BONUS * mem["is_contact"].astype(jnp.float32), -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k1, (B, K)), axis=1)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    branch = do & mem["is_contact"][slot]
    # branched workers hold action (worker index mod n_act) for BRANCH_HOLD steps
    cur = jnp.where(branch, jnp.arange(B) % mem["n_act"], mem["cur"]); left = jnp.where(branch, BRANCH_HOLD, jnp.where(do, 0, mem["left"]))
    return do, slot, {**mem, "cur": cur, "left": left, "branch": branch}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    nc = _ncontact(feats); onset = (nc > mem["nc_prev"]) | (nc >= 2)      # contact just began, or two simultaneous contacts
    cell = _cell(mem, feats); live = ~done
    is_new = ((mem["cell_slot"][cell] < 0) | (onset & (mem["nc_prev"] == 0))) & live
    first = jnp.full(1 << BITS, B, jnp.int32).at[jnp.where(is_new, cell, 0)].min(jnp.where(is_new, jnp.arange(B), B))
    is_new = is_new & (first[cell] == jnp.arange(B))
    order = jnp.cumsum(is_new) - 1; slot = (mem["next_free"] + order) % K
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    is_contact = mem["is_contact"].at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, onset, mem["is_contact"][0]))
    return is_new, jnp.where(is_new, slot, 0), {**mem, "cell_slot": cell_slot, "is_contact": is_contact, "nc_prev": nc, "next_free": (mem["next_free"] + is_new.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
