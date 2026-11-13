# Exploration Discovery Plan

Reframing AutoDev as *exploration-strategy discovery*: an autonomous loop that writes exploration procedures — programs over a save/restore archive — and a benchmark that says, honestly, how much of the reachable world they reach. Three chapters: how we will do it, the tests that decide whether it works at all, and the schedule for running them, with the ways the framing gets stronger as results come in.

Prepared 29 Aug 2026, Saturday evening · Deadline Fri 4 Sep, 5 PM PDT (OpenReview expiry Sep 5 00:00 UTC; abstract editable until then) · The ">90%" target is defined in §1.4 *before* any new run, so it cannot be met by relabeling.

## Chapter 1 — What we are going to do

### 1.1 The overarching goal

> **An autonomous research loop discovers exploration procedures that reach target states faster than learned intrinsic motivation, and the discovered procedures generalize: to games they never saw, and to a class of targets they were never scored on.**

Three words carry the claim. *Discovers* — the procedure is written by an LLM search, not designed by us; the loop, not the program, is the method. *Procedures* — code over a save/restore archive (`act` / `restore` / `archive` hooks), the family that includes Go-Explore [1] and that the field's own negative results point at (random exploration beats RND asymptotically in 3D games [2]; more search made MCTS *worse* at finding bugs [3]; coverage anti-correlates with what testers are for [4]). *Generalize* — the sealed held-out class is the decisive test; the precedent for searching over exploration algorithms, meta-learned curiosity [5], reported transfer across control tasks but had no game, no target-state benchmark, and no held-out class.

Game testing stays as the application that makes target states meaningful (a planted defect's witness state is a target with a reason to exist) and supplies replay-verified real anomalies as a sanity check that the oracles mean something. It is no longer the headline.

### 1.2 The benchmark: target-state reaching under a simulator budget

A *task* is (game level, target predicate); a *tester* gets B parallel workers and T steps each; the score is steps-to-first-witness normalised to [0,1], and *detection* is the fraction of tasks witnessed within budget. This is the injected-fault protocol of Ariyurek et al. [6] and GBQA [7] (recall at an interaction budget) with three additions those benchmarks lack: a random floor, a **reachability calibration**, and sealed held-out splits by level *and by target class*.

| Suite | Tasks | Target predicate | Status |
|---|---|---|---|
| **K-Defect** (Kinetix [8], 74 shipped levels) | 337 planted defects in 5 classes | witness oracle per class | built; splits sealed; 5-seed results for 20 testers |
| **K-Goal** (Kinetix, same levels) | 74 (+64 generated) | the level's own win condition (green touches blue) | *new*: zero-cost — the oracle is the environment reward; gives RL reviewers the comparison they expect (Kinetix reports PPO success rates per level tier) |
| **K-Defect-rot** (rotated held-out class) | 337 | ANOMALY held out instead of OOB | built; run-3 candidates being evaluated |
| **C-Ach** (Craftax [9]) | 65+ achievements × seeds | achievement unlocked | *stretch*: a second engine, symbolic rules instead of physics; a feature adapter is the risk (Test T3) |

Reachability is defined **now**, before new runs: a task is *reachable* if any tester, seed, or the oracle-guided probe witnesses it within **4× the standard budget** (the "long-budget union"). Detection is reported on all tasks *and* on reachable tasks, with the reachable fraction printed next to it. The current single-run numbers on K-Defect are 72% / 64% / 47% of all tasks (search / held-out levels / held-out class) and 88% / 96% / 76% of the reachable set at 1× budget; the long-budget union will lower the "reachable" numbers somewhat, which is the point — it makes 90% harder, not easier.

### 1.3 Methods under comparison

*Baselines that the exploration literature expects.* Uniform random; action-repeat (sticky) random; Go-Explore-style archive search (the seed); PPO with count-based novelty, RND [10], ICM [11], curiosity-proximity and task reward (already evaluated); a CCPT-style curiosity ⊕ imitation tester [4] (new; Kinetix's shipped solution trajectories stand in for expert demos); coverage reported as a secondary metric for all of them.

*Discovery methods on the same problem, equal proposal budget, equal GPU fitness (3 seeds, 128×1024 steps, size-l search split).* (i) Evolutionary program search — ShinkaEvolve [12] (runs 1–5, done or running); (ii) one-shot expert design — Claude Opus asked once, no fitness access (rounds 1–3, eleven programs); (iii) no-LLM search — random perturbation of the best program's constants, restart ensembles; (iv) **agentic loop** — a Claude Code session iterating edit → evaluate → reflect against the fitness server (the autoresearch pattern [13]); (v) **tree search** — AIDE-style [14] solution tree with debug/improve operators. (iv) and (v) are new; the comparison is the same one Meta's AIRA made for Kaggle tasks [15], on a problem with no known solution and a sealed transfer test.

*What is measured for each method:* best fitness vs evaluations spent; the best program's held-out numbers; what changed (structure vs constants, via diff + a rubric); cost in LLM tokens and GPU minutes.

### 1.4 The ">90%" target, pre-registered

We will call the result a success if a **single run** of one discovered procedure, at the standard budget, witnesses **≥ 90% of reachable tasks on the held-out splits** — held-out levels *and* the held-out class — where reachable is the long-budget union of §1.2, computed once and frozen. Today the numbers are 96% and 76%, so the held-out class is the whole fight. Three legitimate levers, in the order we will pull them: search on a cleaner signal (GPU fitness on the reachable-only search split; run 5 from a structurally different seed), budget-matched restart ensembles (the 5-seed union already reaches 80% of all tasks vs 72% single-run), and better procedures for the class that no tester improves (obstacle/target-directed designs, round 3). Illegitimate levers, named so nobody reaches for them: changing the reachable set after seeing results, dropping hard instances without the pre-registered rule, counting across-seed unions as single-run numbers, or raising the budget without raising it for every baseline. If 90% on the held-out class is not reached by Thursday, the paper reports the number we have and says why.

## Chapter 2 — Tests that decide whether this works at all

Each test is cheap, has a pass/fail criterion written before it runs, and kills or confirms one assumption of the framing. They run in parallel on the two fitness servers and the free A40s.

| # | Question | Test | Pass | Fail → consequence |
|---|---|---|---|---|
| **T1** | Is the discovered procedure an *exploration* result, or only a defect-finding result? | K-Goal: run the discovered procedure, the seed, random/sticky-random, and the four PPO objectives with the level win condition as the only target, same budget, 5 seeds | discovered ≥ every baseline on goal-reaching detection on held-out levels | the reframe is unsupported; keep the game-testing framing (current paper) |
| **T2** | Is 90% of the reachable held-out class *possible* at 1× budget? | compute the long-budget (4×) union for the held-out class; measure the best single run against it; measure the budget-matched 4-restart ensemble | best single run or budget-matched ensemble ≥ 85% of the long-budget reachable set | 90% is out of reach by Friday; the paper's target becomes the honest number |
| **T3** | Does the procedure family transfer *across engines*? | Craftax adapter: generic feature vector (agent position, inventory hash, map hash, achievement bits), the Go-Explore seed and the discovered program run unmodified on 10 achievements, 1 GPU-hour | both run; the discovered program ≥ seed on ≥ 6 of 10 achievements | drop C-Ach from this submission; keep it as the stated next step |
| **T4** | Can the fitness discriminate candidates at all? | test–retest: re-score 10 run-4 candidates with 3 fresh seeds; Spearman between the two scorings | ρ ≥ 0.6 | search results are noise-dominated; the paper's method section says so and the scaffold comparison becomes a study of that |
| **T5** | Do the two new scaffolds produce valid programs on this problem? | agentic-loop and tree-search plumbing: 6 evaluations each against the fitness server | ≥ 4 of 6 valid programs, ≥ 1 above the seed | fix plumbing or drop the scaffold from the comparison (report as "attempted") |
| **T6** | Is the CCPT-style baseline implementable from shipped solutions? | imitation term from Kinetix solution trajectories + RND, 1 seed, search split | runs; score within 0.05 of PPO+RND or better | report PPO+RND as the curiosity baseline and say the imitation term was not available |

T1 and T2 are the ones that matter: T1 decides whether the reframe is honest, T2 decides whether the 90% target is honest. Both are one GPU-hour each and can start tonight.

## Chapter 3 — Running the tests, and making the framing better as we go

### 3.1 Schedule (all times PDT; the human's tasks are marked [H])

| When | Agent | Cluster |
|---|---|---|
| **Sat night** | T1 + T2 scripts; agentic-loop scaffold; long-budget union job; K-Goal eval job | run 5 (movertarget seed) on both servers; scans + design evals in the chains |
| **Sun** | T4, T5, T6; Craftax adapter (T3); tree-search scaffold; new §1.1–1.3 text (exploration framing, RL related work) | run 6 on the reachable-only split; scaffold pilots; K-Goal 5-seed eval |
| **Mon** | read T1–T6 → go/no-go on each claim; full scaffold runs (equal budget); restart-ensemble and adaptive-portfolio results into tables | scaffold runs (2 servers); round-3 designs eval |
| **Tue** | held-out evaluation of every scaffold's best program; figures (steps-to-target curves; scaffold curves; per-class); abstract rewritten on OpenReview [H: approve] | evaluations |
| **Wed** | Part 1 complete draft (≤4 pp in the template); Part 2 [H] from the ledger; bug table and Craftax pilot as validation sections | buffer |
| **Thu** | numbers frozen; reproducibility pass (every number regenerated from logs); Part 3 [H] | buffer |
| **Fri** | final PDF by noon [H: read-through]; submit by 5 PM | — |

### 3.2 Ways the framing gets stronger

1. **Report steps-to-target curves, not just detection at one budget.** Exploration papers read curves; ours already show the gap opening within 10⁴ steps.
2. **Coverage as the secondary metric**, computed for every tester from the same runs (cells visited in the coarse feature hash). Expect the CCPT inversion — more coverage, fewer targets — and say so: it is the reason to optimise procedures against targets rather than coverage.
3. **Name the discovered mechanisms in exploration language.** Three archive pools (coverage / high-energy / goal-proximity), biased restores, long single-direction holds: this is a hand-readable exploration algorithm, and the paper should print it, as Alet et al. printed their discovered curiosity programs.
4. **Make the scaffold comparison a finding, not a table.** The interesting sentence is "every discovery method makes its gain in the first evaluations and then moves within evaluation noise" — if T4 confirms the noise floor, that is a result about autonomous research, stated on a problem with no known solution.
5. **Pool hygiene as a contribution.** The reachability calibration (long-budget union; geometric unreachability found by the round-3 agent) is what the injected-fault benchmarks of the field do not do; publish the suite with it.
6. **Keep the real anomalies, demoted.** One paragraph: the oracles were validated on unmodified levels (momentary wall-face tunnelling at 16.5 m/s; one sustained escape on a generated level; 15 unreproduced hits dropped), with the upstream report drafted.
7. **Second engine as the closing sentence, not a claim.** If T3 passes, the Craftax pilot is one figure and one honest sentence; if it fails, it is the stated next step.

### 3.3 What would make us stop

If T1 fails — the discovered procedure is not better than learned intrinsic motivation at plain goal-reaching — the exploration framing is not supported and the paper stays a game-testing paper (still submittable). If T2 fails and T4 shows ρ < 0.6, the honest paper is about *why* autonomous search stalls on this problem, and the 90% target is dropped in writing, not quietly.

## References

1. Ecoffet, Huizinga, Lehman et al.. *Go-Explore: a New Approach for Hard-Exploration Problems*. Nature 2021 / arXiv 2019; arXiv:1901.10995. <https://arxiv.org/abs/1901.10995>
2. Lu, Georgescu, Verwey. *Go-Explore Complex 3D Game Environments for Automated Reachability Testing*. IEEE ToG 2022 (Microsoft Research); arXiv:2209.00570. <https://arxiv.org/abs/2209.00570>
3. Ariyurek, Betin-Can, Surer. *Enhancing the Monte Carlo Tree Search Algorithm for Video Game Testing*. IEEE CoG 2020; arXiv:2003.07813. <https://arxiv.org/abs/2003.07813>
4. Sestini, Gisslén, Bergdahl, Tollmar, Bagdanov (EA SEED). *Automated Gameplay Testing and Validation with Curiosity-Conditioned Proximal Trajectories (CCPT)*. arXiv:2202.10057, 2022. <https://arxiv.org/abs/2202.10057>
5. Alet, Schneider, Lozano-Pérez, Kaelbling. *Meta-learning curiosity algorithms*. ICLR 2020; arXiv:2003.05325. <https://arxiv.org/abs/2003.05325>
6. Ariyurek, Betin-Can, Surer. *Automated Video Game Testing Using Synthetic and Humanlike Agents*. IEEE Transactions on Games, 2021. <https://doi.org/10.1109/TG.2019.2947597>
7. Jiang, Chen, Chen. *GBQA: A Game Benchmark for Evaluating LLMs as Quality Assurance Engineers*. ICLR 2026 workshop; arXiv:2604.02648. <https://arxiv.org/abs/2604.02648>
8. Matthews, Beukman, Lu, Foerster. *Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks*. ICLR 2025 (oral); arXiv:2410.23208; code github.com/FLAIROx/Kinetix. <https://arxiv.org/abs/2410.23208>
9. Matthews et al.. *Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning*. ICML 2024; arXiv:2402.16801. <https://arxiv.org/abs/2402.16801>
10. Burda, Edwards, Storkey et al.. *Exploration by Random Network Distillation*. ICLR 2019; arXiv:1810.12894. <https://arxiv.org/abs/1810.12894>
11. Pathak, Agrawal, Efros et al.. *Curiosity-driven Exploration by Self-supervised Prediction*. ICML 2017; arXiv:1705.05363. <https://arxiv.org/abs/1705.05363>
12. Lange, Imajuku, Cetin (Sakana AI). *ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution*. arXiv:2509.19349, 2025; code github.com/SakanaAI/ShinkaEvolve. <https://arxiv.org/abs/2509.19349>
13. Karpathy. *autoresearch (agent-driven overnight training-script optimization loop)*. GitHub, 2026. <https://github.com/karpathy/autoresearch>
14. Jiang, Schmidt, Srikanth et al.. *AIDE: AI-Driven Exploration in the Space of Code*. arXiv:2502.13138, 2025. <https://arxiv.org/abs/2502.13138>
15. Meta FAIR. *AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench (AIRA-dojo)*. arXiv:2507.02554, 2025. <https://arxiv.org/abs/2507.02554>
