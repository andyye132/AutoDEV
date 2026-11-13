# Part 1 — draft (Sep 2). Agent-written ML result for AutoMLR. ~4 pp of content; convert to the AutoMLR LaTeX template.

## Abstract

Game and simulation testing needs *exploration policies* that drive a system into target or defect states; the standard
approach trains a reinforcement-learning explorer per game. We show that an autonomous research loop can instead **write**
such a tester as a short program, and that at an **equal environment-interaction budget** this zero-training program
reaches target states faster than a trained PPO explorer across **eight structurally different RL environments** — grid
navigation, key-and-door subgoals, arcade, puzzle/planning, continuous control, deep-exploration, 2D physics, and
crafting/survival. Against a *tuned* PPO it never loses: a clear win on six and a tie on two (DeepSea-24 and Craftax); on
the four sparse-reward navigation/planning tasks it is the single best method, beating random, sustained-action, and tuned
RL alike. The advantage is not the save/restore archive it uses (a
plain Go-Explore baseline with the same archive loses to both PPO and random on the hardest task); it is the specific
strategy the loop discovers, and it is a genuine **held-out-engine transfer**: one program evolved on a multi-engine
fitness beats *task-specific* PPO on engines it never saw. The advantage is strictly *equal-budget*: with tuning and more budget RL approaches the tester on some tasks, but at the
budget a tester actually gets, the zero-training program wins the sparse-reward tasks outright. On a deep Kinetix substrate — a replay-verified benchmark
of ~790 planted physics defects with sealed splits and reachability calibration — the same testers beat every
hand-designed, one-shot-LLM, and RL tester at equal budget, and a difficulty-graded bug benchmark shows structured
testers (including ours) find *harder* bugs than trained RL as difficulty scales. The benchmark, the testers, and this
Part were produced by the loop; the authors built the oracles, verified every number, and document every intervention —
including a harness exploit the loop found and the replay check that caught it.

## 1.1 The claim, made general (the headline — Fig. 1)

A *tester* is a policy that explores one level to reach target states (a planted defect's witness, or the level's own
win condition). Game studios train RL explorers for this (Wuji [ASE'19], CCPT/Gordillo [CoG], EA-SEED). We ask what an
autonomous loop discovers instead, and find a sharp, general result:

**At an equal interaction budget, a zero-training LLM-written tester reaches target states faster than a trained PPO
explorer across 8 RL environments** (Fig. 1; detection fraction, search split, each row budget-matched):

| task | family | random | a-rep | best PPO (tuned) | **best LLM tester** |
|---|---|---|---|---|---|
| MiniGrid DoorKey | grid + subgoal | 0.77 | 0.33 | 0.83 | **1.00** |
| MiniGrid FourRooms | grid nav | 0.27 | 0.40 | 0.50 | **0.87** |
| MinAtar Breakout | arcade | 0.33 | 0.33 | 0.42 | **1.00** |
| Jumanji Sokoban | puzzle | 0.66 | 0.56 | 0.41 | **0.79** |
| Gymnax MountainCar | control | 0.00 | _1.00_ | 0.73 | **1.00** |
| Gymnax DeepSea-24 | deep-explore | 0.00 | _1.00_ | _1.00_ | 1.00 (tie) |
| Craftax | crafting/survival | 0.25 | 0.16 | 0.34 | 0.35 (tie) |
| Kinetix K-Goal | 2D physics | 0.36 | 0.58 | 0.56 | **0.67** |

_(a-rep = action-repeat, a cheap sustained-action baseline. Italic cells tie the LLM. Tuned PPO = best over 5 intrinsic
objectives × {default, lr=1e-4/ent=0.2}. K-Goal PPO is default — reward-dense like Craftax, where tuning did not help.)_

The sharp thesis: **at a low interaction budget, trained RL is the wrong tool for sparse-reward target-reaching; a
zero-training exploration program dominates it** — default PPO even trails *random* on DoorKey and Sokoban. The honest
reading of the table: the LLM tester **ties** tuned PPO on DeepSea-24 (both 1.0) and Craftax (0.35 vs 0.34), and its wins
on MountainCar and DeepSea are shared by a dumb action-repeat baseline; **the four wins that require genuine exploration
strategy — where the LLM beats random, action-repeat, *and* tuned PPO — are DoorKey, FourRooms, Breakout, and Sokoban.**
On those, the LLM tester is the single best method by a clear margin. Three facts make this subset robust:

- **It is not "Go-Explore beats PPO."** The plain `goexplore` baseline (same save/restore archive) scores 0.10 on DoorKey
  — below PPO and random — while the LLM tester scores 1.00; the LLM tester beats plain Go-Explore on 5 of 6 tasks. The
  win is the *discovered strategy*, not the archive.
- **Tuning does not close the equal-budget gap on the hard tasks.** A LR×entropy sweep found the default (entropy 0.01)
  was under-tuned; we froze lr=1e-4/ent=0.2 and re-ran all tasks (DoorKey 0.28→0.83, DeepSea-24 win→tie). Even tuned, PPO
  loses the four sparse-nav tasks: high entropy pushes the policy toward random (already 0.77 on DoorKey), which the LLM's
  structured save/restore beats. The budget-scaling curves (4×/16×) are being re-run with the tuned config — the earlier
  curves used default PPO and understate where RL lands (tuned 1× already exceeds default 16× on DoorKey), so we report
  only the strictly equal-budget result as the headline (Fig. 2, tuned).
- **It is held-out-engine transfer.** The winning program on the four sparse-nav tasks (`run7`) was evolved on a multi-engine fitness
  (Kinetix + Craftax + Gymnax) that **never saw MiniGrid, MinAtar, or Sokoban**; a fresh per-task search is weaker (0.33
  on DoorKey — beats PPO, loses to transfer), so the recipe is "evolve one multi-engine tester and transfer it."

Honest boundary: DeepSea-24 and Craftax tie tuned PPO (dense/trivially-reachable reward lets RL learn), and MountainCar/
DeepSea are also solved by action-repeat — so the precise thesis is *a zero-training exploration program dominates
sparse-reward navigation/planning target-reaching at equal budget; tuned RL matches it when reward is dense or the target
is trivially reachable.*

## 1.2 Setup

A tester is four pure-JAX hooks (`init/restore/act/archive`) over a save/restore state archive; the identical interface
runs on every substrate through a generic feature head. The RL tester is PPO on the **same** feature head (no observation
handicap), with 5 intrinsic-motivation objectives (count/RND/ICM/curiosity-proximity/task); we report best-of-objectives.
Equal budget = an identical env-step count per row (131k substrate / 262k Kinetix); PPO's training steps *are* its budget
(detection is logged during training). LLM testers are produced by the loop — evolved (the transfer program) and a survey
of hand-designed procedures — at zero training cost, written once. Metric: fraction of target states reached within
budget; testers at 5 seeds, PPO at 3, paired where relevant.

## 1.3 Depth: the Kinetix planted-defect benchmark + a graded bug benchmark

Rigor lives here — the reason to trust the numbers. **K-Defect**: 5 defect classes with programmatic witness predicates,
**replay-from-reset verification** of every counted hit, sealed levels *and* a sealed defect class, reachability
calibration, a frozen hashed protocol; ~790 planted defects over 74 shipped + 64 generated levels. The best tester scores
**0.62 search / 0.34 sealed class (~91% of the physically reachable set)** at equal budget, beating every hand-designed,
one-shot-LLM, and PPO tester — the deep version of §1.1. 28 replay-verified real engine anomalies fall out as validation
(a player-triggerable escape in a shipped level; scope: Kinetix-physics-only by engine nature).

**A difficulty-graded bug benchmark** (Fig. 3) plants a runaway-physics bug whose difficulty is a controlled parameter
(restitution: physics gives free energy → the tester must actively drive the state). As difficulty rises, **random
collapses (1.0→0.33) and trained RL degrades (→0.67), while structured testers — including our zero-training LLM ones —
hold (0.83)**: structured/LLM testers find harder bugs than trained RL at equal budget. (On this speed-axis bug the LLM
testers tie the strongest simple baselines; the LLM's edge over *all* methods is in the sparse-reward navigation regime
of §1.1.)

## 1.4 Why it transfers

The winning program reads only the generic feature head and discovers informative dimensions at runtime; single-engine
and fresh-per-task programs overfit and do not transfer. Composition of finished programs (portfolios, schedules,
routing) never beats re-aiming the fitness.

## 1.5 Limitations (stated plainly)

Equal-budget only — with tuning and more budget RL approaches or matches the tester on the reward-dense / trivially-reachable tasks (tuned budget curves in Fig. 2); the claim is the sparse-reward equal-budget regime. The tester uses a save/restore archive a policy cannot
(a legitimate strategy difference, the same comparison the Go-Explore/RL literature makes). Kinetix is a research
substrate, not a shipped game (we scope to physics-defect detection). PPO tuning: we swept LR/entropy on the hardest
tasks and report tuned numbers; a per-task sweep across all tasks is the one open lever, but the LLM's margin far exceeds
the tuning gains observed. Verification is single-author + replay; Part 2 documents it, including the harness exploit.
