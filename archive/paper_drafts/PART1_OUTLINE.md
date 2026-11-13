# Part 1 outline (v3) — the generalization reframe (Sep 2)

_Part 1 is the agent-written ML research result (≤4 pp). This is the structure I propose given tonight's generalization
result; review it, then I write the full LaTeX against the AutoMLR template. The old `part1_v2.tex` was built on the
"steering-wheel/six-findings" framing and should be retired._

**The one claim (abstract + §1.1):** _At an equal environment-interaction budget, a zero-training LLM-written exploration
program reaches target states faster than a trained RL explorer — not on one benchmark but across eight structurally
different RL environments; on sparse-reward tasks RL does not catch up even with 16× more budget._ Why it matters:
game/sim testing needs exploration policies that reach defect/goal states; studios train RL for this (Wuji, CCPT, EA/SEED);
we show a cheaper, zero-training alternative that generalizes. Why it's new: TITAN/PromptPO claimed "LLM ≥ RL" but never
matched interaction budget on ground-truthed targets; ours is the budget-matched, multi-environment, replay-verified head-to-head.

## §1.1 — Result 1: the equal-budget win, generalized (≈1 pp, the LEAD; Fig. 1 = the 8-task bar chart)

- The table/figure: LLM tester vs best-of-5 PPO vs random, detection at fixed budget, 8 tasks (DoorKey, FourRooms,
  Breakout, Sokoban, MountainCar, DeepSea-24, Craftax, Kinetix K-Goal). **LLM ≥ tuned PPO on all 8: clear win on 6, ties DeepSea-24 & Craftax; the 4 genuinely-hard wins (LLM beats action-repeat too) are DoorKey/FourRooms/Breakout/Sokoban.**
- Three defenses, each one sentence + a number: (a) beats plain Go-Explore 5/6 (DoorKey 1.0 vs 0.10) → it's the discovered
  strategy, not the archive; (b) equal-budget only — tuned PPO catches up at ~4× budget on DoorKey, so we claim strictly equal-budget, not "RL never catches up" (Fig. 2, tuned curves); (c) held-out
  transfer — one multi-engine-evolved program beats task-specific PPO on engines it never saw.
- The honest boundary as a *strength*: Craftax ties because dense achievement rewards let RL learn → precise thesis is
  "zero-training exploration dominates sparse-reward target-reaching; RL competes only when reward is dense."

## §1.2 — Setup (≈0.75 pp)

- Tester = four pure-JAX hooks (init/restore/act/archive) over a save/restore state archive; the same interface runs on
  every substrate via a generic feature head. RL tester = PPO on the SAME features, 5 intrinsic objectives (count/RND/ICM/
  curiosity-proximity/task). Equal budget = identical env-step count per row (131k substrate / 262k Kinetix); PPO's
  training steps ARE its budget (detection logged during training, no separate eval rollout).
- The LLM testers: evolved by the loop (multi-engine fitness → the transfer program) + a survey of hand-designed procedures;
  zero training, written once.
- Metric: fraction of target states reached within budget (detection), 5-seed for testers / 3-seed PPO (→ 5 after the
  running re-run), paired where relevant.

## §1.3 — Result 2: the Kinetix planted-defect benchmark (the depth; ≈1 pp)

- This is where rigor lives — the reason to trust the numbers. 5 defect classes with programmatic witness predicates,
  **replay-from-reset verification** of every counted hit, sealed levels + a sealed defect class, reachability calibration,
  a frozen hashed protocol. ~790 planted defects, 74 shipped + 64 generated levels.
- The Kinetix accuracy result: best tester 0.62 search / 0.34 sealed class (~91% of the physically reachable set), beating
  every hand-designed, one-shot-LLM, and PPO tester at equal budget — the deep version of the general claim.
- 28 replay-verified real engine anomalies (validation, not headline): the h15 player-triggerable escape; scope honestly
  (Kinetix-physics-only by engine nature).

## §1.4 — Why it works + what doesn't (≈0.5 pp)

- Mechanism: the winning program reads only the generic feature head and discovers informative dimensions at runtime, so
  it transfers; single-engine and fresh-per-task programs don't (fresh DoorKey evolution = 0.33, beats PPO but loses to
  transfer) → "evolve a multi-engine tester once and transfer" is the recipe.

## §1.5 — Limitations (≈0.25 pp, state plainly)

- Equal-budget only: PPO overtakes at 4–16× on reward-dense tasks (K-Goal). Tester uses save/restore a policy can't (a
  legitimate strategy difference, same as the field's Go-Explore-vs-RL comparisons). Kinetix is a research substrate, not
  a shipped game (scope to physics-defect detection). PPO tuning: a LR/entropy sweep is running to close the fairness
  question; report its result.

---

### The abstract (draft, honest, one metric)

> Game and simulation testing needs exploration policies that drive a system into target or defect states; the standard
> approach trains reinforcement-learning explorers. We show that an autonomous loop can instead *write* such a tester as a
> short program, and that at an equal environment-interaction budget this zero-training program reaches target states
> faster than a trained PPO explorer across **eight structurally different RL environments** (grid navigation, subgoal,
> arcade, puzzle, continuous control, deep-exploration, 2D physics, and crafting/survival). Against a tuned PPO it never
> loses — a clear win on six, a tie on two — and on the four sparse-reward tasks it is the single best method. The advantage is not the save/restore archive (a plain Go-Explore baseline loses to both PPO and random on
> the hardest task) but the specific strategy the loop discovers, and it is a genuine held-out-engine transfer: one program
> evolved on a multi-engine fitness beats *task-specific* PPO on engines it never saw. On sparse-reward tasks RL does not
> catch up even with 16× more budget; on reward-dense tasks it does — a clean boundary. On a deep Kinetix substrate — a
> replay-verified benchmark of ~790 planted physics defects with sealed splits and reachability calibration — the same
> testers beat every hand-designed, one-shot-LLM, and RL tester at equal budget and surface 28 replay-verified real engine
> anomalies. The benchmark, the testers, and this Part were produced by the loop; the authors built the oracles, verified
> every number, and document every intervention (including a harness exploit the loop found and the replay check that
> caught it).
