# Miss diagnosis — Kinetix planted-defect benchmark

**Evidence.** First-hit arrays: `runs/hyak_eval_merged.json` (valid non-restoring rows only — `random`, `repeat_random`, `ppo_{count,rnd,icm,curiosity_proximity,task}`), `runs/hyak/eval_longbudget4x_cand1.json` (cand1 @4×), `runs/hyak/eval_goal.json` + `eval_ppo_goal.json` (K-Goal), `results/structural.json`, `runs/hyak/guided_ceiling.json`. New CPU measurement: for all 149 missed and 55 fast-hit instances, the **reachable envelope of every constant action** (11 held actions × one 256-step episode, frozen at the first `done`). Data: `results/miss_diagnosis.json`.

## 1. What is missed

| split | n | reached (union of all testers) | missed |
|---|---|---|---|
| search (planted) | 127 | 83 | **44** (4 unwitnessable → 40 genuine) |
| held-out levels | 59 | 37 (cand1 @4× adds 0) | **22** (1 unwitnessable) |
| held-out class, OOB | 148 | 60 (cand1 @4× adds 4) | **88** |
| K-Goal (level wins) | 52 | 41 (36 by baselines) | **11** by every program |

The seven valid testers are **nested**: exactly one instance in 127 (`m_h16_thrustcontrol_right__DISARMED__0`, ppo_rnd) is reached by only one of them. Quadrupling budget adds nothing on held-out levels. The oracle-guided feasibility prober reaches **1 of the 149** misses. This is one common blind spot, not a budget or a reward-signal problem.

*Unwitnessable* (no dynamic body may collide with the target): `m_arm_left|m_arm_right|m_arm_up|l_h15_swing_up __TUNNEL__0`; `m_arm_hard__TUNNEL__0` (held-out). *Free hits* (witness fires at step 6–8 under no-op): `l_car_launch__TUNNEL__0`, `l_car_swing_around__TUNNEL__1`, `l_h9_morph_direction__TUNNEL__0`, `l_lever_puzzle__TUNNEL__1`.

**Search, 40 genuine misses** (ANOMALY 17, SHORTCUT 17, DISARMED 5, TUNNEL 1):
`l_car_swing_around[SHOR1]; l_chain_thrust[ANOM0]; l_grasp_hard[TUNN0]; l_h0_angrybirds[ANOM0,SHOR0,SHOR1]; l_h11_obstacle_avoidance[ANOM0,DISA0,DISA1]; l_h15_swing_up[ANOM0,SHOR0,SHOR1]; l_h16_thruster_goal[ANOM0]; l_h17_unicycle[ANOM0,SHOR0]; l_h4_cartpole[ANOM0,SHOR0,SHOR1]; l_h7_maze_1[DISA0]; l_h8_maze_2[ANOM0,DISA0,DISA1]; l_hard_cartpole_thrust[ANOM0]; l_hard_cartpole_wheels[ANOM0]; l_mjc_half_cheetah[ANOM0]; l_mjc_half_cheetah_easy[ANOM0]; l_mjc_hopper[SHOR0]; l_mjc_hopper_easy[SHOR0]; l_mjc_walker[ANOM0,SHOR0,SHOR1]; l_mjc_walker_easy[SHOR0,SHOR1]; m_h0_unicycle[ANOM0]; m_h1_car_left[ANOM0]; m_h2_car_right[ANOM0]; s_h2_one_wheel_car[SHOR0]; s_h6_unicycle_right[SHOR0]; s_h7_unicycle_left[SHOR0]; s_h8_unicycle_balance[ANOM0]`

**Held-out levels, 21:** `l_h12_platformer_1[ANOM0,DISA0,DISA1,SHOR0,SHOR1,TUNN0,TUNN1]; l_h13_platformer_2[same 7]; l_h6_lorry[ANOM0]; l_pr[ANOM0,SHOR0,SHOR1]; m_arm_hard[ANOM0]; s_h0_weak_thrust[ANOM0]; s_h5_rotate_fall[ANOM0]`

**OOB, 88/148** — split by which wall was removed: ceiling 56/74 missed, right 32/74. Thrust decides it: ceiling with thrust/weight > 1 → 14/26 reached; ≤ 1 → **4/48**. Jointed agent + no thrust → 10/56 reached.

**K-Goal missed by all 11 programs:** `l_h11_obstacle_avoidance, l_h16_thruster_goal, l_h5_flappy_bird, l_h8_maze_2, l_hard_beam_balance, l_hard_cartpole_thrust, l_hard_cartpole_wheels, l_lever_puzzle, l_mjc_half_cheetah, l_mjc_hopper, l_mjc_walker` — all need the green body to end up *above* its start (mean Δy +1.6 m) versus −0.02 m for the goals baselines reach.

## 2. Why they are missed

Three physical barriers, all readable from the level state:

1. **No coherent long input.** Held actions travel a median 0.45 m and climb 0.00 m per episode in the missed set. Escapes and contacts need 0.4–4.7 m. For thrust/weight ≈ 1.02 the climb takes 57–172 steps of *uninterrupted* thrust; 37 of the 45 thrust-actuated misses need a hold longer than sticky-random's 12-step cap, 25 longer than macrosweep's 20-step macro.
2. **Termination truncates the horizon.** `done = reward ≠ 0`. In `h11/h12/h13/h7_maze_1/h8_maze_2` **all 11 held actions end by hazard contact in 13–19 steps**; the whole 262 k budget becomes ~15 k restarts of a half-second episode. Elsewhere (cars, arms, unicycles) a held drive *wins* the level at ~200 steps and resets — 36 misses sit in such levels.
3. **Energy has to be built, not sampled.** ANOMALY needs 12 m/s; free fall from the box top gives 9.5 m/s, and restitution 1.6 is combined as `min(a,b)`, so only **green-agent ↔ static** bounces amplify (×1.6 each). The best constant action reaches a median 7.0 m/s (max 10.7) — a multiplicative ladder that must be climbed, not stumbled into.

**Not reachable at all** (report, do not chase): 12 instances where the only actuator is a thruster with thrust/weight < 1 and the witness needs net climb — `l_h11_obstacle_avoidance[DISA0,DISA1,OOB0]`, `l_h12_platformer_1[DISA1,OOB0]`, `l_h13_platformer_2[DISA1,OOB0]`, `l_h16_thruster_goal[OOB0]`, `l_h8_maze_2[DISA0,DISA1,OOB0]`, `s_h0_weak_thrust[OOB0]`. A further 27 OOB ceiling escapes on motor-only jointed agents (walkers, cars, arms) show zero lift from any constant action against a 2.7–4.5 m requirement.

## 3. What a successful trajectory must do

*OOB-right:* hold one motor or side thruster in the same direction for 20–60 steps, avoiding the blue goal en route. *OOB-ceiling:* fire the up-thruster continuously for 60–180 steps (only where thrust/weight > 1). *ANOMALY:* pump the mechanism — resonant motor reversal at the pendulum period, or repeated slams of the green body into a wall — until a green↔static bounce chain crosses 12 m/s. *SHORTCUT on walkers/hoppers/cartpoles:* fold the linkage, driving **all** motors to their stops at once so the green body overlaps the recoloured link 0.38–0.87 m away. *DISARMED/TUNNEL in mazes:* steer to one static shape 0.6–3.0 m away without touching the other 3–6 lethal ones — only reachable by resuming from checkpoints.

## 4. Contrast — what action-repeat finds fast

The 55 search instances `repeat_random` hits inside 6 k steps are exactly the ones a *constant input* already solves: **a single held action witnesses 38 of those 55, and 0 of the 149 misses**. 40 % have thrust/weight > 1 (missed: 15 %), and their held-action envelope is far larger (median best speed 9.2 vs 4.8 m/s; median climb 0.44 vs 0.00 m). Easy = "the witness lies inside the reachable set of one constant input"; missed = "it does not".

## 5. Design brief — class-agnostic mechanisms

All are computable from the oracle-blind feature vector (positions, velocities, roles, contacts) plus the harness `done` flag.

| # | Mechanism | Unlocks (n) | Named targets |
|---|---|---|---|
| M1 | **Death-aware checkpoint chaining**: archive only non-terminal states keyed by (depth-since-reset, coarse novelty); restore to the deepest surviving slots; suppress the action that ended the last episode | 68 | all `h11/h12/h13/h7_maze_1/h8_maze_2` instances, cartpoles, unicycles |
| M2 | **Full-episode coherent holds**: sweep hold length over {8,32,96,256} instead of capping at 12/20 | 24 | `m_h12/h13_thrustfall[OOB0]`, `m_h14_thrustblock[OOB0,OOB1]`, `s_h1_thrust_over_ball[OOB0]`, `s_h4_thrust_aim[OOB0]`, `l_pr[OOB0]`, `l_chain_lander[OOB0]`, `l_h7_maze_1[DISA0]` |
| M3 | **Velocity-aligned drive**: every H steps keep the action whose effect projects positively on the current dyn velocity | 32 | every `__OOB__1` (right-wall) instance |
| M4 | **Energy ratchet + restore-to-max-energy**: archive keyed by `v²+2gy`; restore to top-E slots; keep an action only while E grows | 24 | all 24 ANOMALY misses |
| M5 | **Resonant square-wave drive**: two opposed actuator actions, half-period swept over {1…128} steps, keep the period whose E grows fastest | 43 | `l_h15_swing_up`, `l_h4_cartpole`, `l_h17_unicycle`, `m_h0_unicycle`, `m_arm_*`, `l_mjc_*` climbs |
| M6 | **Static-shape contact enumeration**: identify never-moving shapes; steer the nearest dynamic body at each in turn (bang-bang on position error) and hold to contact | 21 | all DISARMED + TUNNEL misses, `l_pr[SHOR0]`, `l_h0_angrybirds[SHOR1]` |
| M7 | **Actuator saturation / self-folding**: cycle *all* motor bindings in the same direction for the whole episode | 11 | `l_mjc_walker[SHOR0,SHOR1]`, `l_mjc_hopper[SHOR0]`, `l_h4_cartpole[SHOR0,SHOR1]`, `l_h17_unicycle[SHOR0]`, `l_car_swing_around[SHOR1]` |
| M8 | **Boundary seeking**: derive the world box from never-moving shapes and drive the nearest dynamic body at the nearest face | 32 | same set as M3 (pair M8 with M2/M3) |
| M9 | **Termination avoidance (goal shunning)**: after a first win, penalise decreasing green–blue distance so rollouts continue past it | 36 | `m_h1/h2_car_right[OOB*]`, `s_h5_rotate_fall`, `s_h6/h7_unicycle`, `m_arm_left/up[OOB*]`, `l_pr` |

**Priority.** M1+M9 (horizon), then M2/M3/M8 (coherence), M4/M5 (energy), M6/M7 (contact).

**Caveat.** Five cheap CPU probes at 1/16 budget — constant holds, 64-step sticky-random, a speed-ratchet, a resonance sweep, and restore-to-highest-energy — each detected 0/62 of a stratified missed subset. No single mechanism suffices; none of the five was death-aware or depth-keyed, so M1/M9 remain untested and are the first thing to build.
