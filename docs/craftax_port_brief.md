# Craftax port brief (C-Ach suite) — from the reading agent, 29 Aug 2026

Verified against `craftax==1.6.1` (PyPI). Not installed in the repo venv; use a separate env (Hyak: `/mmfs1/gscratch/raivn/andy132/conda_envs/craftax`).

## Environment API
- `craftax.craftax_env.make_craftax_env_from_name(name, auto_reset)` / `make_craftax_env_from_params(classic, symbolic, auto_reset)`; 8 classes (Craftax/Classic × Symbolic/Pixels × AutoReset/NoAutoReset). Static params bind at construction (`default_static_params()`); `env.default_params` → `EnvParams()`.
- `reset(key, params)` / `step(key, state, action, params)`; **no level argument** — the world is generated from `key` inside `reset_env`. For a fixed-world task, pass the task's key to reset for every worker (same world), per-worker keys to step.
- Use the `*NoAutoReset` classes (`auto_reset=False`); the harness resets on done itself. `done` is tuple[3]; `is_game_over(state, params, static)` recomputes it from state.
- `env.action_space(params).n`: Craftax 43 actions, Classic 17. `Discrete.n` gymnax-compatible.

## State
`EnvState` (craftax/craftax/craftax_state.py): `map/item_map` (9,48,48) int32, `mob_map` bool, `light_map` f32, ladders (9,2), `player_position` (2,) int32, `player_level` () int32 (0–8), health f32, food/drink/energy/mana int32, `inventory` (14 scalars + armour (4,) + potions (6,)), mobs (`Mobs{position, health, mask, attack_cooldown, type_id}`), **`achievements` (67,) bool, sticky/monotonic** (read this for the oracle; `info` only reports achievements×done×100), `state_rng` PRNGKey, `timestep` int32. Pitfall: `fractal_noise_angles` is a tuple of `None`s (0 leaves; harmless). Classic constructs some scalars as Python ints — fine under jit.
State size ≈ 270–290 KB (4 × (9,48,48) grids) → archive K=256 × chunk 16 ≈ 1–1.5 GB; start with K≈64, B≈32, chunk 4–8.

## Achievements / actions
Craftax `Achievement` enum: 67 members (0–66); Classic 22. Pilot 10 (easy→hard): COLLECT_WOOD(0), PLACE_TABLE(1), MAKE_WOOD_PICKAXE(5), COLLECT_STONE(9), EAT_PLANT(11, near-unsolved), MAKE_IRON_PICKAXE(20), ENTER_DUNGEON(29), OPEN_CHEST(61), ENTER_GNOMISH_MINES(28, ≈0% for all baselines), DEFEAT_NECROMANCER(49, unreached).

## Feature adapter (~60 floats) — keep the first 11 slots semantically compatible with the Kinetix layout the programs index
[0:2] player_position/48; [2:4] position delta (0 if unavailable); [4] 0 (speed); [5:7] player_position/48 again ("green"); [7] distance proxy to the target (1.0 if unknown); [8] timestep/max; [9:11] 0.5,0.5 ("goal"); then: level/8, direction/4, health/food/drink/energy/mana /9, sleeping/resting, xp/dex/str/int, inventory (14, /10 clipped) + armour (4) + potions (6), achievements.sum()/67 (+ tier counts), day fraction + light, nearest-mob distances (3), hashed 9×9 map window → ~12 random projections.

## Throughput / pitfalls
Paper: 267k steps/s (Craftax) at 4096 workers on its own loop; our harness adds restore/archive/oracle overhead. `make_tester` calls `reset_fn` for all B workers every step (masked by done) — expensive for Craftax world-gen; replace with the "optimistic reset" pattern (regenerate M ≪ B worlds per step) or reset only when any done.

## Baselines (arXiv:2402.16801)
Craftax-1B: PPO-RNN ≈ 15–16% reward; PPO/ICM/E3B/RND ≈ 10–12% — intrinsic rewards do not beat plain PPO. Basic achievements 70–80%, Intermediate 15–30%, Advanced/Very-Advanced ≈ 0%. EAT_PLANT ≈ 0% except E3B (~5%); ENTER_GNOMISH_MINES ≈ 0%.
