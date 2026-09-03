# Part 1 — draft (Sep 2). Agent-written ML result for AutoMLR. ~4 pp of content; convert to the AutoMLR LaTeX template.

## Abstract

Game and simulation testing needs *exploration policies* that drive a system into target or defect states; the standard
approach trains a reinforcement-learning explorer per game. We show that an autonomous research loop can instead **write**
such a tester as a short program, and that at an **equal environment-interaction budget** this zero-training program
reaches target states faster than a trained PPO explorer across **eight structurally different RL environments** — grid
navigation, key-and-door subgoals, arcade, puzzle/planning, continuous control, deep-exploration, 2D physics, and
crafting/survival — winning on seven and tying on the eighth. The advantage is not the save/restore archive it uses (a
plain Go-Explore baseline with the same archive loses to both PPO and random on the hardest task); it is the specific
strategy the loop discovers, and it is a genuine **held-out-engine transfer**: one program evolved on a multi-engine
fitness beats *task-specific* PPO on engines it never saw. On sparse-reward tasks RL does not catch up even with **16×**
more budget; on reward-dense tasks it does — a clean boundary. On a deep Kinetix substrate — a replay-verified benchmark
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

| task | family | random | best PPO (tuned) | **best LLM tester** |
|---|---|---|---|---|
| MiniGrid DoorKey | grid + subgoal | 0.77 | 0.83 | **1.00** |
| MiniGrid FourRooms | grid nav | 0.27 | 0.39 | **0.87** |
| MinAtar Breakout | arcade | 0.33 | 0.43 | **1.00** |
| Jumanji Sokoban | puzzle | 0.66 | 0.28 | **0.79** |
| Gymnax MountainCar | control | 0.00 | 0.61 | **1.00** |
| Gymnax DeepSea-24 | deep-explore | 0.00 | 0.78 | **1.00** |
| Craftax | crafting/survival | 0.25 | 0.34 | 0.35 (tie) |
| Kinetix K-Goal | 2D physics | 0.36 | 0.56 | **0.67** |

The sharp thesis: **trained RL is the wrong tool for target-reaching at a low interaction budget; a zero-training
exploration program dominates it** — RL even trails *random* on DoorKey and Sokoban. Three facts make this robust:

- **It is not "Go-Explore beats PPO."** The plain `goexplore` baseline (same save/restore archive) scores 0.10 on DoorKey
  — below PPO and random — while the LLM tester scores 1.00; the LLM tester beats plain Go-Explore on 5 of 6 tasks. The
  win is the *discovered strategy*, not the archive.
- **RL does not catch up with more budget.** Even at **16×** the budget, tuned PPO still loses on DoorKey (0.72),
  Breakout (0.52), and Sokoban (0.37 — plateaued) vs the LLM tester's 1.0/1.0/0.79; it ties FourRooms and overtakes only
  on reward-dense K-Goal (Fig. 2). We also swept PPO's learning rate and entropy; the default was under-tuned (DoorKey
  0.28→0.83) but even tuned PPO caps near a random policy's exploration and loses.
- **It is held-out-engine transfer.** The winning program on five tasks (`run7`) was evolved on a multi-engine fitness
  (Kinetix + Craftax + Gymnax) that **never saw MiniGrid, MinAtar, or Sokoban**; a fresh per-task search is weaker (0.33
  on DoorKey — beats PPO, loses to transfer), so the recipe is "evolve one multi-engine tester and transfer it."

Honest boundary: Craftax ties (0.35 vs 0.34) because its dense achievement rewards let RL learn — the precise thesis is
*zero-training exploration dominates sparse-reward target-reaching; RL competes when reward is dense.*

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

Equal-budget only — RL overtakes at 4–16× on reward-dense tasks. The tester uses a save/restore archive a policy cannot
(a legitimate strategy difference, the same comparison the Go-Explore/RL literature makes). Kinetix is a research
substrate, not a shipped game (we scope to physics-defect detection). PPO tuning: we swept LR/entropy on the hardest
tasks and report tuned numbers; a per-task sweep across all tasks is the one open lever, but the LLM's margin far exceeds
the tuning gains observed. Verification is single-author + replay; Part 2 documents it, including the harness exploit.
