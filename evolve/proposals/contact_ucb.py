"""Contact-hunting search with a UCB bandit over macros ("contact_ucb").

Idea: three of the five defect classes are *touch* events -- a moving shape ending up inside a static obstacle, the
agent winning on the wrong (neutral) shape, the agent surviving contact with a former hazard. What they need is not
extreme physics but a rich set of shape-to-shape contacts, so this program makes contact itself the objective.
(1) Reward = (a) a rarity-weighted count of the currently active contact pairs, using an online per-pair counter so
    that a contact nobody has produced before is worth far more than resting on the floor, plus (b) dense shaping for
    closing the gap between any green (agent) shape and the nearest non-wall red/neutral/blue shape.
(2) act() picks short macros (motor holds, thruster holds, settle, alternations) with a UCB1 bandit whose statistics
    are shared across the whole batch, so behaviours that historically produced rare contacts are re-selected while
    untried macros keep their optimism bonus; per-worker Gumbel noise keeps the batch from collapsing to one arm.
(3) The archive is keyed by a SimHash of the *contact bitmask* plus a coarse agent/centroid cell, so every distinct
    contact configuration gets a checkpoint; restore() prefers slots with rare contacts and small agent-target gaps
    (and any worker that has been contact-free for DRY_T steps jumps back), which turns near-misses into repeated
    branching attempts at exactly the touch the oracles are waiting for.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12          # hash bits for the contact-pattern archive
UCB_C = 1.0        # UCB exploration constant
GTEMP = 0.35       # Gumbel noise on the per-worker arm choice
MACRO_L = 8        # macro length (one bandit pull)
SHAPE_W = 2.0      # weight of the gap-closing shaping reward
PROX_W = 1.5       # weight of stored-state proximity in the restore score
VIS_W = 0.4        # visit penalty in the restore score
P_R = 0.04         # baseline restore probability
DRY_T = 25         # contact-free steps before a worker is pulled back to a contact-rich checkpoint
EPS_RAND = 0.05    # random-action rate inside a macro
CELL_RES = 0.5     # coarse grid for the positional part of the archive key


def _idx(F):
    """Per-shape blocks and the contact-flag tail, recovered from the (size-unique) feature width."""
    sols = [(p, c) for p in range(4, 41) for c in range(0, 41)
            if 11 + 7 * p + 6 * c + (p * (p - 1) // 2 - 6) + c * p + c * (c - 1) // 2 == F]
    if len(sols) != 1:
        n = max(1, F // 4)
        return dict(use=0.0, px=[0], py=[1], ri=[8], ai=[8], wall=[1.0], cidx=list(range(F - n, F)))
    Np, Nc = sols[0]
    P = 11; PA = P + 5 * Np; PL = P + 6 * Np; C = P + 7 * Np; CA = C + 4 * Nc; CL = CA + Nc
    return dict(use=1.0,
                px=[P + 2 * i for i in range(Np)] + [C + 2 * i for i in range(Nc)],
                py=[P + 2 * i + 1 for i in range(Np)] + [C + 2 * i + 1 for i in range(Nc)],
                ai=[PA + i for i in range(Np)] + [CA + i for i in range(Nc)],
                ri=[PL + i for i in range(Np)] + [CL + i for i in range(Nc)],
                wall=[1.0 if i < 4 else 0.0 for i in range(Np)] + [0.0] * Nc,   # first 4 polys are the boundary walls
                cidx=list(range(CL + Nc, F)))


def _macros(n_act):
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    pats = [[2 * m + d] for m in range(nm) for d in (0, 1)] + [[t] for t in thr]
    pats += [[noop], [thr[0], noop], [0, 2], [1, 3]]
    return [[int(p[i % len(p)]) for i in range(MACRO_L)] for p in pats]


def _gap(mem, X):
    """Min distance from any green shape to the nearest attractive target (red/neutral/blue, walls excluded)."""
    px = X[:, mem["px"]]; py = X[:, mem["py"]]; role = X[:, mem["ri"]]; act = X[:, mem["ai"]] > 0.5
    pen = jnp.where(jnp.abs(role - 3.0) < 0.5, 0.0,
          jnp.where((jnp.abs(role) < 0.5) & (mem["wall"] < 0.5), 0.05,
          jnp.where(jnp.abs(role - 2.0) < 0.5, 0.6, 1e6)))
    pen = jnp.where(act, pen, 1e6)
    green = jnp.where((jnp.abs(role - 1.0) < 0.5) & act, 0.0, 1e6)
    d = jnp.sqrt((px[:, :, None] - px[:, None, :]) ** 2 + (py[:, :, None] - py[:, None, :]) ** 2 + 1e-9)
    g = jnp.min(d + pen[:, None, :] + green[:, :, None], axis=(1, 2))
    return mem["use"] * jnp.minimum(g, 5.0) + (1.0 - mem["use"]) * 5.0


def _rar(mem):
    return 1.0 / jnp.sqrt(1.0 + mem["dimcnt"])


def init(rng, F, n_act, K, B):
    ix = _idx(F); tab = _macros(n_act); M = len(tab); nc = len(ix["cidx"])
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "M": jnp.int32(M),
            "px": jnp.array(ix["px"], jnp.int32), "py": jnp.array(ix["py"], jnp.int32),
            "ri": jnp.array(ix["ri"], jnp.int32), "ai": jnp.array(ix["ai"], jnp.int32),
            "wall": jnp.array(ix["wall"], jnp.float32), "use": jnp.float32(ix["use"]),
            "cidx": jnp.array(ix["cidx"], jnp.int32), "dimcnt": jnp.zeros(nc, jnp.float32),
            "table": jnp.array(tab, jnp.int32),
            "uval": jnp.zeros(M, jnp.float32), "ucnt": jnp.full(M, 1e-3, jnp.float32), "tot": jnp.float32(0.0),
            "mid": jnp.arange(B, dtype=jnp.int32) % jnp.int32(M), "ph": jnp.zeros(B, jnp.int32),
            "acc": jnp.zeros(B, jnp.float32), "pgap": jnp.zeros(B, jnp.float32), "dry": jnp.zeros(B, jnp.int32),
            "proj": jax.random.normal(rng, (nc + 5, BITS), jnp.float32),
            "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "next_free": jnp.int32(0)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    s = ((arch_feats[:, mem["cidx"]] * _rar(mem)).sum(-1) + PROX_W * jnp.exp(-2.0 * _gap(mem, arch_feats))
         - VIS_W * jnp.log1p(arch_visits.astype(jnp.float32)))
    slot = jnp.argmax(jnp.where(arch_valid, s, -1e9)[None, :] + jax.random.gumbel(k2, (B, K), jnp.float32), axis=1)
    do = ((jax.random.uniform(k1, (B,)) < P_R) | (mem["dry"] > DRY_T)) & arch_valid.any()
    return do, slot, {**mem, "dry": jnp.where(do, 0, mem["dry"])}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3 = jax.random.split(rng, 3)
    c = feats[:, mem["cidx"]]
    gap = _gap(mem, feats)
    rew = (c * _rar(mem)).sum(-1) + SHAPE_W * jnp.clip(mem["pgap"] - gap, 0.0, 1.0)
    a = mem["table"][mem["mid"], jnp.clip(mem["ph"], 0, MACRO_L - 1)]
    a = jnp.where(jax.random.uniform(k1, (B,)) < EPS_RAND, jax.random.randint(k2, (B,), 0, mem["n_act"]), a)
    acc = mem["acc"] + rew
    end = (mem["ph"] + 1) >= MACRO_L
    uval = mem["uval"].at[mem["mid"]].add(jnp.where(end, acc, 0.0))
    ucnt = mem["ucnt"].at[mem["mid"]].add(jnp.where(end, 1.0, 0.0))
    tot = mem["tot"] + end.sum().astype(jnp.float32)
    ucb = uval / ucnt + UCB_C * jnp.sqrt(jnp.log(tot + 2.0) / ucnt)
    pick = jnp.argmax(ucb[None, :] + GTEMP * jax.random.gumbel(k3, (B, ucb.shape[0]), jnp.float32), axis=1)
    return a.astype(jnp.int32), {**mem, "uval": uval, "ucnt": ucnt, "tot": tot, "pgap": gap,
                                 "acc": jnp.where(end, 0.0, acc), "ph": jnp.where(end, 0, mem["ph"] + 1),
                                 "mid": jnp.where(end, pick, mem["mid"]),
                                 "dry": jnp.where(c.sum(-1) > 0, 0, mem["dry"] + 1)}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    c = feats[:, mem["cidx"]]
    code = jnp.concatenate([c, jnp.round(feats[:, 5:7] / CELL_RES), jnp.round(feats[:, 0:2] / CELL_RES),
                            jnp.round(_gap(mem, feats)[:, None] / 0.25)], axis=-1)
    cell = ((code @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)
    is_new = (mem["cell_slot"][cell] < 0) & ~done
    slot = (mem["next_free"] + jnp.cumsum(is_new) - 1) % mem["K"]
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    return is_new, slot, {**mem, "cell_slot": cell_slot, "dimcnt": mem["dimcnt"] + c.sum(0),
                          "next_free": (mem["next_free"] + is_new.sum()) % mem["K"]}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
