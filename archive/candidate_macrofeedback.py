"""Checkpointed hybrid sweep ("macrofeedback") -- breadth-first tree search that mixes the deterministic
open-loop macro library from the previous generation with a bank of closed-loop bang-bang controllers that
react to measured state every step.

Motivation for the change: paired-delta feedback showed the open-loop macro sweep is strong on TUNNEL/DISARMED/
ANOMALY but weak on SHORTCUT (winning via an unintended goal) -- and the one mutation known to transfer was
replacing open-loop action holds with closed-loop feedback control (bang-bang on state features). Open-loop
macros can't home in on a goal or ride an emerging oscillation; they just replay a fixed script. This version
keeps the macro library (it demonstrably works) but adds a second family of "controllers": each step they
read the true feature vector (dyn pos/vel, green pos, blue pos -- the actual goal marker) and pick the action
that pushes a chosen error signal in a chosen direction, so behaviour reacts immediately to what actually
happened last step instead of a pre-baked schedule.

(1) Open-loop half: same deterministic macro table as before (sustained holds, reversals, thruster
    holds/pulses, sweeps, settle) -- covers the classes where blind coverage already works well.
(2) Closed-loop half: bang-bang controllers keyed on an error signal e = feat[sig] - ref (ref is another
    feature, e.g. the goal marker position, or a constant threshold). Each step picks act_ge if e >= 0 else
    act_lt. Modes: (a) velocity-follow -- push the motor in the sign of the *current* velocity, pumping energy
    into a resonant swing (good for high-energy/anomaly/tunnel states); (b) goal-pursuit -- push the dynamic
    object's x/y toward the blue (goal) marker's x/y, and the mirror anti-pursuit that pushes away from it
    (goal-directed and goal-avoiding excursions are exactly what surfaces "win via unintended goal" states);
    (c) throttle-to-speed -- fire a thruster only while measured speed is below a target, coasting once past
    it, producing a self-sustaining pulse train instead of a fixed-period one.
(3) The frontier/archive machinery (novelty-gated storage, restore to the least-visited checkpoint) is
    unchanged from the previous generation -- it is what turns single macros/controllers into a breadth-first
    search over the (state x behaviour) product, and nothing here suggested it was the bottleneck.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
MACRO_L = 20        # length of a full macro / controller episode
SHORT_L = 6         # length of the short-burst variants (finer branching factor)
NOV_EPS = 0.30       # novelty radius (scaled feature units) required to store an endpoint
NOV_DIMS = (0, 1, 2, 3, 4, 5, 6, 9, 10)                    # dyn pos/vel/speed, green pos, blue pos
NOV_SCALE = (1.0, 1.0, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0)
EPS_RAND = 0.03      # tiny action noise, breaks symmetry between workers on identical checkpoints
SPD_HOT = 4.0        # mid-episode store threshold on dyn_max_speed, also the throttle-controller target
VIS_NOISE = 0.9       # tie-break noise on the frontier visit ranking

# feature indices (autodev.features.make_feature_fn): dyn_pos=0:2 dyn_vel=2:4 dyn_max_speed=4
# green_pos=5:7 green_blue_dist=7 time_frac=8 blue_pos=9:11
DYN_X, DYN_Y, VEL_X, VEL_Y, SPD = 0, 1, 2, 3, 4
GREEN_X, GREEN_Y, BLUE_X, BLUE_Y = 5, 6, 9, 10


def _macros(n_act):
    """Deterministic open-loop macro library: (M, MACRO_L) action table + per-macro length."""
    nm = max(1, (n_act - 3) // 2); noop = 2 * nm
    thr = [noop + 1 + t for t in range(max(0, n_act - 1 - 2 * nm))] or [noop]
    ms = []
    def add(pat, L): ms.append(([int(pat[i % len(pat)]) for i in range(MACRO_L)], L))
    for m in range(nm):
        add([2 * m], MACRO_L); add([2 * m + 1], MACRO_L)                    # sustained pushes
        add([2 * m, 2 * m, 2 * m + 1, 2 * m + 1], MACRO_L)                  # fast reversal
        add([2 * m] * 5 + [2 * m + 1] * 5, MACRO_L)                         # slow reversal
    for t in thr:
        add([t], MACRO_L); add([t, t, noop, noop], MACRO_L)                 # thruster hold / pulse train
    for m in range(nm):
        add([thr[m % len(thr)], 2 * m], MACRO_L)                            # thrust + motor interleave
    add([2 * m for m in range(nm)], MACRO_L); add([2 * m + 1 for m in range(nm)], MACRO_L)
    add([noop], MACRO_L)                                                    # settle
    for m in range(nm):
        add([2 * m], SHORT_L); add([2 * m + 1], SHORT_L)                    # short bursts
    add([thr[0], noop], SHORT_L)
    return [p for p, _ in ms], [L for _, L in ms], nm, noop, thr


def _reactive(n_act, nm, noop, thr):
    """Closed-loop bang-bang controller bank: err = feat[sig] - (feat[ref] if ref>=0 else const);
    action = act_ge if err >= 0 else act_lt. Returns parallel python lists, one entry per controller."""
    sig, ref, const, act_ge, act_lt, length = [], [], [], [], [], []
    def add(s, r, c, ge, le, L):
        sig.append(s); ref.append(r); const.append(c); act_ge.append(ge); act_lt.append(le); length.append(L)
    for m in range(nm):
        p, q = 2 * m, 2 * m + 1
        add(VEL_X, -1, 0.0, p, q, MACRO_L)                 # velocity-follow: pump energy into current swing
        add(VEL_Y, -1, 0.0, p, q, MACRO_L)
        add(DYN_X, BLUE_X, 0.0, q, p, MACRO_L)              # goal-pursuit: drive dyn x toward the real goal
        add(DYN_Y, BLUE_Y, 0.0, q, p, MACRO_L)
        add(DYN_X, BLUE_X, 0.0, p, q, MACRO_L)              # anti-pursuit: drive away from the goal
        add(DYN_X, GREEN_X, 0.0, q, p, SHORT_L)             # short-burst pursuit toward the agent marker
        add(VEL_X, -1, 0.0, p, q, SHORT_L)                  # short-burst resonance pump
    for t in thr:
        add(SPD, -1, SPD_HOT * 0.5, noop, t, MACRO_L)       # throttle-to-speed: self-timed pulse train
    return sig, ref, const, act_ge, act_lt, length


def init(rng, F, n_act, K, B):
    tab, ln_ol, nm, noop, thr = _macros(n_act)
    sig, ref, const, act_ge, act_lt, ln_re = _reactive(n_act, nm, noop, thr)
    M_ol, M_re = len(tab), len(sig)
    M = M_ol + M_re
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "M": jnp.int32(M), "M_ol": jnp.int32(M_ol),
            "table": jnp.array(tab, jnp.int32), "mlen_ol": jnp.array(ln_ol, jnp.int32),
            "r_sig": jnp.array(sig, jnp.int32), "r_ref": jnp.array(ref, jnp.int32),
            "r_const": jnp.array(const, jnp.float32), "r_ge": jnp.array(act_ge, jnp.int32),
            "r_lt": jnp.array(act_lt, jnp.int32), "r_len": jnp.array(ln_re, jnp.int32),
            "nov": jnp.array(NOV_DIMS, jnp.int32), "sc": jnp.array(NOV_SCALE, jnp.float32),
            "mid": jnp.arange(B, dtype=jnp.int32) % jnp.int32(M), "ph": jnp.zeros(B, jnp.int32),
            "at_end": jnp.zeros(B, bool), "next_free": jnp.int32(0)}


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    key = jnp.where(arch_valid, arch_visits.astype(jnp.float32) + VIS_NOISE * jax.random.uniform(rng, (K,)), 1e6)
    slot = jnp.argsort(key)[jnp.arange(B) % K]           # distinct least-expanded frontier nodes
    return mem["at_end"] & arch_valid.any(), slot, mem


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    M_ol = mem["M_ol"]; is_react = mem["mid"] >= M_ol
    F = feats.shape[1]

    ol_id = jnp.clip(mem["mid"], 0, mem["table"].shape[0] - 1)
    ph = jnp.clip(mem["ph"], 0, mem["table"].shape[1] - 1)
    a_ol = mem["table"][ol_id, ph]
    mlen_ol = mem["mlen_ol"][ol_id]

    n_re = mem["r_sig"].shape[0]
    r_id = jnp.clip(mem["mid"] - M_ol, 0, n_re - 1)
    sig = mem["r_sig"][r_id]; ref = mem["r_ref"][r_id]
    sig_v = jnp.take_along_axis(feats, jnp.clip(sig, 0, F - 1)[:, None], axis=1)[:, 0]
    ref_v = jnp.take_along_axis(feats, jnp.clip(ref, 0, F - 1)[:, None], axis=1)[:, 0]
    ref_use = jnp.where(ref >= 0, ref_v, mem["r_const"][r_id])
    err = sig_v - ref_use
    a_re = jnp.where(err >= 0, mem["r_ge"][r_id], mem["r_lt"][r_id])
    mlen_re = mem["r_len"][r_id]

    a = jnp.where(is_react, a_re, a_ol)
    mlen = jnp.where(is_react, mlen_re, mlen_ol)
    a = jnp.where(jax.random.uniform(k1, (B,)) < EPS_RAND, jax.random.randint(k2, (B,), 0, mem["n_act"]), a)

    nxt = mem["ph"] + 1
    end = nxt >= mlen
    return a.astype(jnp.int32), {**mem, "ph": jnp.where(end, 0, nxt),
                                 "mid": jnp.where(end, (mem["mid"] + 1) % mem["M"], mem["mid"]), "at_end": end}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    x = feats[:, mem["nov"]] * mem["sc"]
    y = arch_feats[:, mem["nov"]] * mem["sc"]
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    mind = jnp.min(jnp.where(arch_valid[None, :], d2, 1e12), axis=1)
    store = (~done) & (mem["at_end"] | (feats[:, SPD] > SPD_HOT)) & (mind > NOV_EPS ** 2)
    slot = (mem["next_free"] + jnp.cumsum(store) - 1) % mem["K"]
    return store, slot, {**mem, "next_free": (mem["next_free"] + store.sum()) % mem["K"]}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
