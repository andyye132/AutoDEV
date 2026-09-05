# The generalization result: LLM-written testers vs trained RL across diverse RL tasks (Sep 2, 2026 — tuned)

_The decisive experiment for "does the equal-budget win generalize beyond Kinetix?" Answer: **at equal budget a
zero-training LLM-written tester matches or beats a properly-tuned PPO explorer on all 8 diverse RL tasks — clear win on 6,
tie on 2 (DeepSea-24, Craftax); it never loses.** Every number regenerates: `scripts/eval_ppo_substrate.py` (PPO on any
substrate) + `results/breadth.json` (LLM/evolved testers + baselines) → `scripts/make_generalization_table.py`. PPO column
is **best-of-{5 objectives} × best-of-{default, tuned lr=1e-4/ent=0.2}** — the fair, tuned comparison._

## The claim, made general

**At an equal environment-interaction budget, a zero-training LLM-written tester program matches or beats a tuned PPO
explorer at reaching target states — not just on Kinetix physics, but across eight structurally different RL tasks** (grid
navigation, grid+subgoal, arcade, puzzle/planning, continuous control, deep-exploration, physics, crafting/survival). This
is the generalized form of the project's strongest claim, and it clears the "too narrow / one substrate" objection.

## Result (search split, detection fraction; each row budget-matched: LLM vs tuned PPO at the SAME budget)

_The seven substrate tasks are at 131,072 env steps; Kinetix K-Goal at 262,144 (its native tester budget). Budget differs
between rows but every row compares LLM and PPO at one budget — no cross-budget comparison._

| Task | family | random | action-repeat | best PPO (tuned) | **best LLM tester** | verdict |
|---|---|---|---|---|---|---|
| MiniGrid DoorKey | grid nav + subgoal | 0.77 | 0.33 | 0.83 (count) | **1.00** (run7) | LLM > PPO |
| MiniGrid FourRooms | grid navigation | 0.27 | 0.40 | 0.50 (count) | **0.87** (run7) | LLM > PPO |
| MinAtar Breakout | arcade | 0.33 | 0.33 | 0.42 (task) | **1.00** (run7) | LLM ≫ PPO |
| Jumanji Sokoban | puzzle / planning | 0.66 | 0.56 | 0.41 (cur-prox) | **0.79** (run7) | LLM > PPO |
| Gymnax MountainCar | continuous control | 0.00 | 1.00 | 0.73 (count) | **1.00** (split-pump) | LLM > PPO¹ |
| Gymnax DeepSea-24 | deep exploration (RL's home turf) | 0.00 | 1.00 | **1.00** (cur-prox) | **1.00** (split-pump) | **tie**¹ |
| Craftax | crafting / survival | 0.25 | 0.16 | 0.34 (task) | 0.35 (saturate) | **tie** |
| Kinetix K-Goal | 2D physics | 0.36 | 0.58 | 0.56 (curiosity)² | **0.67** (macrosweep) | LLM > PPO |

**LLM-written tester ≥ tuned best-of-5-objectives PPO on all 8 tasks — clear win on 6, tie on DeepSea-24 (both 1.0) and
Craftax (0.35 vs 0.34). It never loses.** DeepSea-24 flips from a win to a tie once PPO is tuned (curiosity-proximity PPO
reaches 1.0), and Craftax ties because its dense achievement rewards let RL learn. So the honest scope: **the zero-training
LLM tester dominates on sparse-reward target-reaching; on reward-dense / trivially-reachable tasks, tuned RL matches it.**

¹ **action-repeat also scores 1.00 on MountainCar and DeepSea-24** — those two are solved by a dumb sustained-action
baseline, so the LLM's edge there is *not* LLM-specific skill. ² K-Goal PPO is the default config (K-Goal was not in the
tuned re-run; it is reward-dense like Craftax, where tuning did not help — a tuned K-Goal sweep is the one remaining gap).

## The genuinely LLM-hard wins (where the LLM beats EVERY baseline, including action-repeat)

Filtering to tasks where the LLM tester beats random **and** action-repeat **and** tuned PPO — i.e. where the win requires
real exploration strategy, not a cheap trick:

| task | random | action-repeat | tuned PPO | **LLM tester** |
|---|---|---|---|---|
| DoorKey | 0.77 | 0.33 | 0.83 | **1.00** |
| FourRooms | 0.27 | 0.40 | 0.50 | **0.87** |
| Breakout | 0.33 | 0.33 | 0.42 | **1.00** |
| Sokoban | 0.66 | 0.56 | 0.41 | **0.79** |

**On these 4 sparse-reward navigation/planning tasks the LLM tester is the single best method by a clear margin** — every
alternative (random, sustained-action, tuned intrinsic-motivation RL) is well below it. This is the load-bearing subset:
the LLM contributes exploration content no cheap baseline supplies. (MountainCar/DeepSea are tied by action-repeat;
Craftax ties PPO.)

## The win is the LLM's *strategy*, not just save/restore (the key skeptic defense)

The obvious objection: "the testers use a Go-Explore archive (save/restore) that PPO lacks — you've just shown Go-Explore
beats PPO." **False, and the data kills it.** The plain `goexplore` baseline has the *same* save/restore capability, and it
often *loses* to PPO and even random:

| task | plain Go-Explore | best LLM tester | Δ |
|---|---|---|---|
| DoorKey | 0.10 | **1.00** | +0.90 |
| Breakout | 0.33 | **1.00** | +0.67 |
| FourRooms | 0.40 | **0.87** | +0.47 |
| MountainCar | 0.67 | **1.00** | +0.33 |
| Sokoban | 0.51 | **0.79** | +0.28 |
| DeepSea-24 | 1.00 | 1.00 | +0.00 |

The LLM-evolved tester beats **plain Go-Explore on 5/6 tasks** (tie on DeepSea). On DoorKey, plain Go-Explore scores 0.10 —
*worse than PPO and far worse than random (0.77)* — while the LLM tester scores 1.00. **So the advantage is the specific
exploration strategy the LLM discovered, not the mere presence of an archive.**

## It is a genuine held-out-engine transfer result

The winning LLM tester on the four LLM-hard tasks is `run7_best` — one program evolved once on a *multi-engine* fitness
(Kinetix + Craftax + Gymnax) that **never saw MiniGrid, MinAtar, or Sokoban.** One LLM-evolved program beats task-specific
tuned PPO on engines it was never optimized for. That is the generalization claim in its strongest form: not "an LLM writes
a fresh tester per task," but "one LLM-evolved tester transfers across RL tasks and beats task-specific RL."

## Honest caveats (state these in the paper)

- **Tester type vs learner type.** The LLM testers use a save/restore archive (Go-Explore-style); PPO is a policy that
  cannot reset to saved states at test time. Both spend the same env-step budget, so this is a legitimate *strategy*
  difference — a smarter exploration procedure vs a learned policy. It must be stated: the win is "archive-based
  exploration program > learned policy at equal budget."
- **Two of the eight wins are cheap.** MountainCar and DeepSea-24 are solved by action-repeat too; the LLM-specific
  advantage is the 4 sparse-nav tasks above. Don't over-claim "8 tasks" as if all 8 needed the LLM.
- **PPO is tuned, not strawman.** Default PPO (entropy 0.01) was under-tuned; a LR×entropy sweep found entropy is the key
  lever, and we froze lr=1e-4/ent=0.2 and re-ran all 7 substrate tasks at 5 seeds. Tuning flipped DeepSea-24 win→tie and
  lifted DoorKey 0.28→0.83 — the numbers above are the tuned best. Even tuned, PPO loses the 4 sparse-nav tasks: high
  entropy pushes the policy toward random (already 0.77 on DoorKey), which the LLM's structured save/restore beats.
- **Budget crossover (tuned, measured).** The earlier 4×/16× curves used *default* (under-tuned) PPO and understated
  where RL lands. Re-run with the tuned config (`scripts/run_ppo_matrix_budget_tuned.sh`), the honest crossover on the 4
  LLM-hard tasks (LLM tester's budget fixed at 1×; only PPO's budget scales):

  | task | LLM (1×) | tuned PPO 1× | tuned PPO 4× | tuned PPO 16× |
  |---|---|---|---|---|
  | DoorKey | 1.00 | 0.83 | **1.00** | 1.00 |
  | FourRooms | 0.87 | 0.50 | 0.56 | 0.83 |
  | Breakout | 1.00 | 0.42 | 0.91 | **1.00** |
  | Sokoban | 0.79 | 0.41 | 0.72 | **0.81** |

  **The honest reading (this is the correction):** with tuning **PPO catches up to or passes the tester once it is given
  4–16× more interaction** — DoorKey at 4× (1.00), Breakout at 16× (1.00), Sokoban even slightly *overtakes* at 16×
  (0.81 vs 0.79), FourRooms nearly closes (0.83 vs 0.87). So the earlier "RL doesn't catch up even at 16×" claim was an
  artifact of under-tuned PPO and is **withdrawn.** The robust, honest claim is strictly **equal-budget**: at 1× the
  zero-training tester wins the 4 sparse-nav tasks (DoorKey 1.00 vs 0.83, FourRooms 0.87 vs 0.50, Breakout 1.00 vs 0.42,
  Sokoban 0.79 vs 0.41); tuned RL needs 4–16× more interaction to match it. That is still the load-bearing result — a
  program written once, at zero training, beats tuned task-specific RL at the budget a tester actually gets.
- **Some tasks are near-saturated for good testers** (MountainCar, Breakout at threshold 3), ceiling 1.00; the
  discrimination is real on DoorKey/FourRooms/Sokoban where scores spread widely.

## What this does for the paper

The equal-budget win is no longer a Kinetix curiosity; it is a property of **sparse-reward target-reaching RL testing**:
at a fixed interaction budget, one LLM-evolved exploration program beats tuned task-specific RL across four structurally
distinct engines (and ties or wins the rest). Part 1 leads with this; the Kinetix planted-defect benchmark is the
deep-dive substrate, not the whole story.

## Fresh per-task evolution: weaker than transfer (an honest secondary result)

We also tested the literal "LLM writes a fresh tester per task" framing: an 18-generation evolution seeded from the generic
Go-Explore program, scored only on DoorKey with a DoorKey-specific prompt. The champion, on the tester protocol (eval seeds
100–104), reaches **detected 0.33 on DoorKey** — it beats plain Go-Explore (0.10) but **loses to random (0.77), tuned PPO
(0.83), and the multi-engine transfer program `run7` (1.00).** **Lesson:** a short single-task search does not discover a
strong explorer; the generalization comes from **one program evolved on a multi-engine fitness that then transfers.** So
the paper's claim is "evolve a multi-engine tester once and transfer it," not "evolve a fresh tester per task."
