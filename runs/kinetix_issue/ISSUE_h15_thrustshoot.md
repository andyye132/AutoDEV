# Draft upstream report (for Andy to review and send) — Kinetix / Jax2D

**Title:** Fast bodies tunnel into boundary walls and the floor at ~16 m/s in `m/h15_thrustshoot` (solver ejects them after a few steps)

**Environment:** kinetix-env 3.0.0 (Jax2D), static config from `static_env_params_from_size("l")`, level `worlds/m/h15_thrustshoot.json`, discrete action space, default env params.

**What happens.** The level's thruster can launch the neutral block (polygon 5) to ~16–17 m/s. At that speed the block moves ~0.55 m per simulator step, and the discrete-step collision solver misses the wall face:
1. In the attached 51-action sequence (deterministic from `env.reset(PRNGKey(0))`), the block's centroid goes from x = 4.60 m (step 50) to x = 5.15 m (step 51), i.e. 15 cm *inside* the right boundary polygon (x ∈ [5, 10]); the solver then ejects it (velocity flips from +16.5 to −2.9 m/s), so it re-enters the arena within one step.
2. In a second sequence (46 actions), the block enters the floor polygon near the bottom-right corner at 16 m/s and sits ~10 cm below the floor surface (y ≈ 0.30 vs floor top 0.40) for 13 consecutive steps before being pushed out.
Both were found automatically by three different exploration strategies in every seed (within 30k–150k environment steps) and re-verified by deterministic replay; the same signature shows up on generated levels with fast bodies near thin static geometry.

**Why it matters.** Visually the block sinks into a wall/floor for ~0.03–0.4 s; on levels where the goal or a hazard sits behind thin geometry this can produce unintended wins/losses, and if a wall were thinner than one step of travel the body would leave the arena entirely.

**Reproduce.**
```
python scripts/repro_h15.py                      # replays runs/kinetix_issue/h15_escape_actions.json and prints the crossing at step 51
```
(uses the helpers in `autodev/kinetix_utils.py` only for loading the level at size l and reading the arena box; the step loop is plain `env.step`.) Action list: `runs/kinetix_issue/h15_escape_actions.json` (goexplore tester, seed 0).

**Possible fix.** Continuous collision detection or substepping for bodies whose per-step displacement exceeds a fraction of the thinnest static feature; or clamping thruster-induced speed. Happy to share the other replay chains and the generated levels where it recurs.
