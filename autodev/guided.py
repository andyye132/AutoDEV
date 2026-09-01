"""Feasibility probe: an ORACLE-GUIDED tester that may see a scalar 'progress toward the witness' (computed from the planted
defect's meta). Not a fair tester — it establishes, per instance, whether the witness is reachable within budget, so that
detection rates can be normalized by the feasible ceiling."""
from __future__ import annotations
import jax, jax.numpy as jnp
from .kinetix_utils import (dynamic_masks, role_centroid, dyn_max_speed, shape_position, ROLE_AGENT)
from .defects import CLS


def guide_fn(state, meta):
    """Higher = closer to the witness. OOB: max outward excess of any moving shape beyond the box;
    TUNNEL: -min distance of moving centroids to target; SHORTCUT/DISARMED: -distance of green centroid to target; ANOMALY: max speed."""
    pm, cm = dynamic_masks(state)
    P = jnp.concatenate([state.polygon.position, state.circle.position]); D = jnp.concatenate([pm, cm])
    box = meta.box
    out = jnp.maximum(jnp.maximum(box[0] - P[:, 0], P[:, 0] - box[1]), jnp.maximum(box[2] - P[:, 1], P[:, 1] - box[3]))   # >0 outside
    g_oob = jnp.where(D, out, -jnp.inf).max()
    tgt = shape_position(state, meta.target_is_poly, meta.target_idx)
    d_dyn = jnp.where(D, jnp.linalg.norm(P - tgt[None], axis=-1), jnp.inf).min()
    g_tunnel = -d_dyn
    g_touch = -jnp.linalg.norm(role_centroid(state, ROLE_AGENT) - tgt)
    g_anom = dyn_max_speed(state)
    return jnp.select([meta.cls == CLS["OOB"], meta.cls == CLS["TUNNEL"], meta.cls == CLS["SHORTCUT"], meta.cls == CLS["ANOMALY"], meta.cls == CLS["DISARMED"]],
                      [g_oob, g_tunnel, g_touch, g_anom, g_touch], 0.0)


class GuidedProgram:
    """Go-Explore driven by the guide: archive the best-guide state in each of NB guide bands; restore mostly from the top bands;
    act with action-repeat random. Uses feats[:, -1] = guide (appended by the guided harness)."""
    name = "oracle_guided"
    def __init__(self, p_restore=0.1, max_rep=12, top_frac=0.25): self.p, self.max_rep, self.top_frac = p_restore, max_rep, top_frac
    def init(self, rng, F, n_act, K, B):
        self.n_act = n_act
        return {"cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "slot_guide": jnp.full(K, -jnp.inf), "next_free": jnp.int32(0), "K": jnp.int32(K)}
    def restore(self, mem, feats, af, av, avis, rng):
        B = feats.shape[0]; k1, k2 = jax.random.split(rng)
        do = jax.random.uniform(k1, (B,)) < self.p
        g = jnp.where(av, mem["slot_guide"][: av.shape[0]], -jnp.inf)
        # rank-based preference for high-guide slots + exploration noise
        order = jnp.argsort(-g); rank = jnp.zeros_like(order).at[order].set(jnp.arange(av.shape[0]))
        logits = jnp.where(av, -rank.astype(jnp.float32) / (self.top_frac * av.shape[0] + 1.0) - 0.3 * jnp.log1p(avis.astype(jnp.float32)), -jnp.inf)
        slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k2, (B, av.shape[0])), axis=1)
        return do & av.any(), slot, mem
    def act(self, mem, feats, obs, rng):
        B = feats.shape[0]; k1, k2 = jax.random.split(rng)
        new = mem["left"] <= 0
        cur = jnp.where(new, jax.random.randint(k1, (B,), 0, self.n_act), mem["cur"])
        left = jnp.where(new, jax.random.randint(k2, (B,), 1, self.max_rep + 1), mem["left"]) - 1
        return cur, {**mem, "cur": cur, "left": left}
    def archive(self, mem, feats, done, af, av, rng):
        B = feats.shape[0]; K = av.shape[0]; g = feats[:, -1]
        # store if the worker's guide beats the archive's worst (or archive not full); slot = worst slot
        worst = jnp.argmin(jnp.where(av, mem["slot_guide"][:K], -jnp.inf) + jnp.where(av, 0.0, -1e9))
        thresh = jnp.where(av.all(), mem["slot_guide"][:K][worst], -jnp.inf)
        do = (g > thresh) & ~done
        # only the best worker this step writes (avoid overwriting with worse)
        best_w = jnp.argmax(jnp.where(do, g, -jnp.inf))
        do = do & (jnp.arange(B) == best_w)
        slot = jnp.where(av.all(), worst, mem["next_free"] % K)
        sg = mem["slot_guide"].at[slot].set(jnp.where(do.any(), g[best_w], mem["slot_guide"][slot]))
        return do, jnp.full(B, slot, jnp.int32), {**mem, "slot_guide": sg, "next_free": mem["next_free"] + do.any().astype(jnp.int32)}
