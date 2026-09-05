# Improving the AutoDev tester algorithm — a research-grounded plan (Sep 3, 2026)

**Purpose.** Andy's push: our claim ("a zero-training LLM-written tester beats trained RL at equal budget") is *honest but
modest* — 6/8 clear wins (only 4 that beat every baseline), equal-budget-only, and an adjacent genre (PromptPO) already
exists. This doc proposes **concrete improvements to the core exploration algorithm** so we raise accuracy, raise
efficiency (steps-to-witness), win more of the 8 tasks, and stake out clear novelty over PromptPO — then validates them on
our own benchmark. It is grounded in a four-way recon: (A) what our algorithm actually does today, (B) PromptPO's method +
weaknesses, (C) the borrowable technique literature 2023–2026, (D) our eval + Hyak harness.

---

## 0. TL;DR — the bet, in five moves

Our tester is **already** an adaptive bandit-portfolio over a quality-diversity archive (`run7`), so the easy "make it
adaptive" wins are taken. The real levers, in priority order:

1. **Make the two load-bearing decisions *principled* instead of hand-tuned.** Replace `run7`'s heuristic restore/store
   scoring with **UCB-over-the-archive** (Monte Carlo Elites, reward = "did restoring here yield a new archive entry")
   and an **E3B episodic elliptical novelty** prior. Warm-up-free, pure-JAX, directly buys efficiency + per-task adaptivity.
2. **Add a reward/goal-following emitter to the portfolio.** The single missing primitive that most plausibly flips the
   **reward-dense ties** (Craftax, and RL's home-turf tasks) — neither `run7` nor PromptPO exploit it for *target-reaching*.
3. **Put efficiency in the fitness and scale evolution on now-free Hyak.** `run7` was still improving when a **$40 cap**
   stopped it; add a **steps-to-witness** term so evolution optimizes *speed*, and spend real compute (Hyak: 0 jobs, 6 GPUs).
4. **One genuinely new architectural axis for novelty over PromptPO:** an **IGE-style LLM cell-judge** invoked *only at
   cell-selection granularity* (never per step) for test-time re-planning + reflection — the adaptation axis PromptPO's
   static policy and `run7`'s cheap EWMA bandit both lack.
5. **Rigor that also strengthens the claim:** a **Go-Explore-PPO baseline** (RL *with* the save/restore archive) to kill
   the reset-asymmetry critique; a **feature-validity mask** to fix the leaky cross-engine feature head; and **fix
   `run7`'s 128-lane hang** so the transfer champion completes the real protocol.

Everything here is implementable in pure JAX under the existing 4-hook interface, evaluated on the existing 8-task +
graded benchmarks, with paired CIs. No new substrate required.

---

## 1. Where we are (honest baseline, read from code)

**The interface** (`autodev/tester.py`): a tester is 4 pure-JAX hooks — `init / restore / act / archive` — over a
save/restore state archive of `K` slots run across `B` parallel workers for `T` steps; budget = `B·T` env-steps; every
counted hit is **replayed from reset** and must re-trigger the oracle (`search_eval.py`). This is a **restore-based
Go-Explore** skeleton whose *rules are an LLM-evolved program*.

**Our best program `run7` is not naive.** It already implements:
- a **6-primitive action bank** (random / hold / sweep / bang-bang-reactive / linear-controller) selected by a **softmax
  bandit over per-mode EWMA reward-rate** with a 25% exploration floor;
- a **dual-pool quality-diversity archive**: per-dimension min/max *frontier* records + a coverage hash with exact bins
  for rarely-changing dims (inventory, achievements);
- ratchet + rare-dim detectors driving rarity-weighted restore.

`split_pump` is even principled rare-event simulation (**adaptive multilevel splitting**, Cérou & Guyader 2007) with a
φ-importance vector and a pool-allocation bandit. So "add a portfolio / a bandit / a QD archive" is **already done**.

**The genuine algorithmic weaknesses (the levers):**
- **W-A. The restore/store scoring is hand-tuned heuristic, not principled.** `run7` restores by
  `rarity − VISIT_W·log(visits) + ENERGY_W·energy` with Gumbel noise — a reasonable guess, but not a principled
  yield-maximizing selector. The literature has the right object (UCB with reward = archive growth).
- **W-B. No goal/reward-following primitive.** Every emitter is coverage- or physics-driven; none follows the task's own
  reward/goal gradient. This is exactly why we *tie* on reward-dense tasks (Craftax) and on tasks RL solves — we never
  exploit the signal that's actually there.
- **W-C. Evolution is compute-starved and one champion doesn't scale.** Runs stop on a **$-cap** (~35–40 gens, still
  improving); `run7` **hangs** on the full 128×2048 Kinetix protocol (an XLA blow-up at high lane×step count — it runs
  fine at 16×128, ~26 s, verified Sep 3), so its Kinetix headline is currently incomplete.
- **W-D. The 11-slot feature head is a leaky abstraction.** Dims 2–4 / 7–10 are placeholders on some engines (grid velocity
  = 0, Craftax "distance" = constant 5.0), so a program keying on their *semantics* silently misbehaves off-Kinetix.

**Current results (post-tuning, honest):** 6/8 clear wins vs tuned PPO, 2 ties (DeepSea-24, Craftax); only **4 tasks**
(DoorKey/FourRooms/Breakout/Sokoban) where the LLM tester beats *every* baseline including action-repeat; equal-budget only
(tuned PPO catches up at 4–16×). Efficiency (steps-to-witness) is measured but not optimized or headlined.

---

## 2. The bar to clear — PromptPO and the genre (what we must beat)

**PromptPO** (Hatgis-Kessell & Brunskill, *When are LLMs Sufficient Policy Optimizers for Sequential RL Tasks?*, arXiv
**2605.30719**, 2026): an LLM writes+refines executable Python **policies** (plus a `Feedback` class) from **rollout
returns**; matches/beats RL on **15/19** envs, far more sample-efficient. It is the closest prior and a reviewer will name
it. Its weaknesses are our openings (labels: P = paper-stated, I = inferred):

| # | PromptPO weakness | How we beat it |
|---|---|---|
| W1 (P) | **No interaction-budget matching**; LLM compute unpriced | We fix a single budget `N` for both sides and report at equal `N` |
| W2 (P) | **Return-maximization only** — no notion of reaching a rare target/defect | Our objective *is* target/defect reaching (first-hit, coverage) |
| W3 (P) | Needs an **interpretable, code-expressible reward** | We drive search with a **state-predicate verifier**, not a reward |
| W4 (P) | **Fails on fine-grained torque control** (loses the 4 MuJoCo tasks) | We scope to state-based/structured-action testing and cede torque honestly |
| **W5 (P)** | **Exploration ceiling = RL's** — "PromptPO, like RL, struggles with difficult exploration" (NoiseWorld 4/5 unsolved by *both*) | **This is our wedge:** a dedicated *exploration* mechanism for rare-state reaching beats both |
| W6 (I) | **No replay verification** of claimed successes | Every hit replays-from-reset; racy/lucky hits never count |
| W8 (I) | **No transfer/generalization claim** (fresh policy per env) | We show held-out-engine transfer |
| W11 (I) | **Static deliverable** — no test-time adaptation | Move 4: test-time LLM cell-judge / online portfolio |

**Genre baselines to cite as competitors, not peers:** Code-as-Policies (2209.07753, single-shot, no search/verify),
Voyager (2305.16291, novelty not target-reaching, LLM self-verify only), Eureka (2310.12931, LLM writes *reward*, still
trains RL — opposite regime), ADAS/Meta-Agent-Search (2408.08435, QA agents, no env budget). **Direct LLM-game-testing
prior:** TITAN (2509.22170), GBQA (2604.02648 — best model finds only **~48%** of 124 injected bugs; a clean "still hard"
anchor), KLPEG (2511.02534). **The gap none of them fill jointly:** budget-matched + target-reaching + replay-verified +
held-out transfer. That intersection is our paper; the algorithm work below is to make the *win inside it* decisive.

**Two positioning cautions (from the recon):** (1) don't over-claim general "LLMs explore better" — PromptPO *and* RL both
hit the exploration wall, so our win must come specifically from the exploration mechanism + testing objective + verification;
(2) PromptPO's efficiency claim is genuinely un-normalized, so **lead with budget-matching** — that beats a caveat the
authors themselves state.

---

## 3. The idea space (ranked, cited) — what the literature offers

Structurally we can improve three things: **(1) the cell representation** (what's a distinct archive entry), **(2) the
restore-selection + post-restore policy**, **(3) how those adapt per task at test time**. A load-bearing budget fact:
**no prediction-error intrinsic bonus (RND/ICM/BYOL/NovelD) converges anywhere near 10⁵ steps** — borrow their *formulas*
as cheap selection scores, never as trained policies. Six families (verified citations):

- **F1 — Go-Explore + learned cells.** Latent Go-Explore (2208.14928), Time-Myopic Go-Explore (2301.05635), **Intelligent
  Go-Explore / IGE** (2405.15143, ICLR'25 — foundation-model *judge* replaces every heuristic, targets low-sample regime).
- **F2 — Quality-Diversity with learned descriptors.** DSAGE surrogate (2206.04199, NeurIPS'22), QDHF (2310.12103), AURORA
  (1905.11874 / 2106.05648), CVT-MAP-Elites (1610.05729, O(1) archive), QDax JAX plumbing (2308.03665), QDAIF (2310.13032).
- **F3 — Low-budget intrinsic motivation (formulas only).** SimHash counts (1611.04717), pseudo-counts (1606.01868), **E3B
  elliptical episodic bonus** (2210.05805, NeurIPS'22 — Mahalanobis novelty from within-episode feature covariance, no
  warm-up), episodic-vs-global design rule (2306.03236).
- **F4 — LLM game-playing / agentic.** Voyager (2305.16291), Reflexion (2303.11366, verbal RL), LATS/RAP/ToT tree-search,
  TITAN/GBQA/KLPEG (competitors).
- **F5 — LLM-driven program evolution.** FunSearch (Nature'23, islands+verifier), ELM (2206.08896, LLM-mutation + MAP-Elites
  over code), **ReEvo (2402.01145) / EoH (2401.02051)** — reflection as verbal gradient at *lower* query budget; AlphaEvolve
  (2506.13131, Flash-breadth/Pro-depth ensemble); OMNI-EPIC (2405.15568).
- **F6 — Adaptive/test-time selection.** **Monte Carlo Elites** (2104.08781, GECCO'21 — *UCB over which archive elite to
  expand*, reward = offspring survived), **Multi-Emitter MAP-Elites** (2007.05352 — UCB over which emitter to run), **Agent57**
  (2003.13350 — sliding-window UCB over exploration strategies), Successive Halving / Hyperband (1502.07943 / 1603.06560).

**Top-5 shortlist (impact ÷ risk, all JAX-able, mapped to our levers):**
1. **UCB-over-archive restore** (Monte Carlo Elites) — smallest change to the decision we make most; efficiency + adaptivity.
2. **E3B / count-like cell-selection score** — the only budget-honest "intrinsic motivation"; a frontier prior for #1.
3. **DSAGE-style online surrogate** predicting a cell's expand-value — skips dead restores (pure efficiency). *Stretch.*
4. **Portfolio of emitters via sliding-window UCB**, evolved programs as arms + a **reward-following arm** — the literal
   "adapt per task," and the likely fix for reward-dense ties.
5. **Learned cell representation** (LGE/Time-Myopic + CVT) — highest accuracy ceiling for reward-dense/high-D, **but the
   only shortlist item with real budget risk** (encoder warm-up). Pilot against the cheap hand-cell baseline first.

---

## 4. The proposed algorithm — "PUCX": Portfolio-UCB eXplorer

A single **zero-training, pure-JAX** tester (one program, all 8 tasks) that keeps `run7`'s strengths but makes its two weak
decisions principled and adds the missing primitive. Components, in build order:

**C1 — UCB-over-archive restore (Monte Carlo Elites, 2104.08781).** Treat each archive cell as a bandit arm. Maintain per
cell an empirical **yield** `y_c` = fraction of restores-from-`c` that produced a *new* archive entry or advanced a frontier
record, and a pull count `n_c`. Restore ← `argmax_c [ y_c + β·sqrt(log Σn / n_c) + prior_c ]` with `prior_c` the novelty
score below. Warm-up-free, principled, self-adapts to which region of *this* instance is paying off. Replaces `run7`'s
hand-tuned rarity+energy heuristic.

**C2 — E3B episodic elliptical novelty (2210.05805) as store gate + restore prior.** Keep a running within-episode feature
covariance `Σ`; a state's novelty = Mahalanobis `xᵀΣ⁻¹x` over a cheap feature map (our normalized head). Store a state when
its elliptical bonus exceeds a threshold; use the bonus as `prior_c`. This is a continuous, warm-up-free generalization of
counts that separates near-goal states in **reward-dense** spaces where coarse hand-cells collide — directly aimed at the ties.

**C3 — Strategy portfolio under a sliding-window UCB meta-controller (Multi-Emitter 2007.05352 / Agent57 2003.13350).**
Arms = a small set of emitters: `run7`'s bang-bang-reactive (our known sparse-nav win), a frontier/coverage emitter,
`split_pump`'s energy-pumping oscillation, action-repeat/hold, **and a new goal/reward-following emitter** (C-B fix:
ascend the task's own reward or the `distance-to-target` feature via a short greedy/CEM inner loop). A per-task
sliding-window UCB allocates the budget among emitters, so the tester *self-selects* the right strategy per instance
instead of committing to one evolved bias. Non-stationarity within the episode ⇒ sliding-window, not vanilla UCB.

**C4 (stretch, efficiency) — DSAGE online surrogate (2206.04199).** A tiny online model predicting, per cell, `P(expanding
here reaches a new target)`; fold into C1's prior to skip low-value restores. Cold-start is the only risk (use the count
prior until it warms).

**C5 (Phase 2, the novelty axis) — IGE-style LLM cell-judge (2405.15143) + reflection.** At **cell-selection granularity
only** (every K restores, never per step): send a compact archive/coverage summary to the LLM, which (i) re-ranks restore
targets and (ii) reflects (Reflexion 2303.11366) on why exploration stalled, adjusting emitter weights for the next
segment. Bounded token cost (a handful of calls per instance); this is the **test-time adaptation** PromptPO's static
policy cannot do and `run7`'s EWMA bandit doesn't approach.

**How PUCX addresses Andy's four critiques:**
- *"Headline is 4 tasks, not 8"* → C3's reward-following emitter + C2's finer novelty target the reward-dense/tie tasks.
- *"Archive-reset asymmetry"* → answered by the **Go-Explore-PPO baseline** (§6), not by PUCX itself; PUCX makes the
  *strategy* contribution unambiguous once RL also has the archive.
- *"Equal-budget only"* → C1/C4 optimize **steps-to-witness**, so we add an **efficiency headline** (win at fewer steps),
  not just "at 1× budget."
- *"Genre isn't new (PromptPO)"* → C5 (test-time LLM adaptation) + testing objective + replay verification + transfer is a
  slice PromptPO explicitly does not occupy (its W2/W5/W6/W8/W11).

---

## 5. Fitness & evolution changes (so the *search* finds better testers)

- **Efficiency term in the fitness.** Current score = `1 − min(first_hit, budget)/budget` already rewards speed weakly;
  make it explicit — add a first-hit-time bonus and report median steps-to-witness in feedback, so evolution optimizes
  efficiency directly (Andy's ask).
- **Scale on Hyak (the biggest single lever).** `run7` stopped at a $40 cap still improving (0.405→0.589 @ gen 35/38).
  Re-launch multi-engine evolution **seeded from PUCX** with more generations/islands and an **AlphaEvolve-style ensemble**
  (Sonnet-breadth / Opus-depth) — compute is now free (0 jobs, 6 GPUs).
- **Reflection-as-verbal-gradient (ReEvo/EoH).** Feed execution traces of failed target-reaches back as verbal critique to
  the mutation prompt — the cheapest accuracy lever that costs no env steps.
- **Fix W-C so scaling is safe.** Root-cause `run7`'s 128-lane hang (likely archive-size/dual-pool XLA memory) *and* add a
  **per-candidate eval timeout guard** so a hanging candidate is penalized, not fatal to the whole run.

---

## 6. Rigor & fairness (these also make the claim stronger)

- **Go-Explore-PPO baseline** — give PPO the *same* save/restore archive (policy-based Go-Explore). Isolates the LLM
  *strategy* from the reset capability; kills the reviewer's asymmetry objection head-on.
- **Feature-validity mask (fix W-D)** — add a per-engine boolean mask marking real vs placeholder feature slots, passed to
  testers, so cross-engine transfer stops relying on placeholder semantics.
- **Efficiency as a co-headline** — report detection *and* median steps-to-witness with paired CIs (`scripts/paired_ci.py`).

---

## 7. Experiment plan (phased, Hyak-ready)

**Phase 0 — mechanism proof (laptop CPU, now).** Hand-write PUCX (C1+C2+C3) as `evolve/methods/pucx.py`. Eval on the 6 fast
substrates (DoorKey, FourRooms, Breakout, Sokoban, MountainCar, DeepSea-24) vs `run7` + baselines, 3 seeds, via
`scripts/eval_baselines.py --substrate … --program_files evolve/methods/pucx.py`. **Go/no-go gate:** PUCX ≥ `run7` on the 4
sparse-nav tasks *and* strictly beats `run7` on ≥1 tie task (MountainCar/DeepSea via the reward-following emitter), paired
CI non-overlapping.

**Phase 1 — full benchmark (Hyak GPU).** If Phase 0 passes: full 8-task + graded-bench eval of PUCX at protocol scale
(128×2048, seeds 100–104), plus the **Go-Explore-PPO** baseline and the **run7-hang fix**. Concrete jobs (env B =
`autodev_multi` for substrates, env A = `autodev` for Kinetix; ≤6 GPUs; cwd `/gscratch/raivn/andy132/AutoDev`):
```bash
# substrate tasks (per substrate)
sbatch --export=ALL,SUBSTRATE=xminigrid:MiniGrid-DoorKey-8x8,\
  SPLIT=runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json,PROGRAMS=,\
  PROGRAM_FILES=/gscratch/raivn/andy132/AutoDev/evolve/methods/pucx.py,\
  SPLITS=search:heldout_levels_instances,SEEDS=100:101:102:103:104,TAG=doorkey_pucx \
  scripts/slurm/eval_substrate.sbatch
# Kinetix K-Defect / K-Goal
sbatch --export=ALL,PROGRAMS=,PROGRAM_FILES=/gscratch/raivn/andy132/AutoDev/evolve/methods/pucx.py,\
  TAG=pucx,SEEDS=100:101:102:103:104,SPLIT=runs/hyak/split.json,\
  SPLITS=search:heldout_levels_instances:heldout_class_instances,STEPS=2048,WORKERS=128 \
  scripts/slurm/eval_programs.sbatch
# graded bench (restitution axis)
export AUTODEV_SPLIT=runs/graded/split.json && bash scripts/run_graded_eval.sh   # add pucx.py to its program list
```
Then `python scripts/breadth_table.py && python scripts/make_generalization_table.py && python scripts/fig_graded.py`.

**Phase 2 — scaled evolution + LLM judge (Hyak GPU).** Seed multi-engine evolution from PUCX with the efficiency fitness
and the reflection loop; pilot C5 (LLM cell-judge) on 2–3 tasks. Compare evolved-PUCX vs hand-PUCX vs `run7`.

**Success criteria (pre-registered):**
- **Accuracy:** win ≥6/8 vs tuned PPO with the LLM beating *every* baseline (incl. action-repeat) on ≥5 tasks (up from 4);
  ≥1 current tie flipped to a win. Graded bench: strict LLM > all non-LLM at the hardest difficulty.
- **Efficiency:** median steps-to-witness reduced ≥25% vs `run7` at equal detection, paired-CI significant.
- **Rigor:** PUCX still beats Go-Explore-PPO (archive-matched) → the win is the strategy, not the reset.

---

## 8. Risks & honest caveats (state these)

- **Learned descriptors (F1/F5-C5) need warm-up rollouts that compete with the 131k budget** — that's why C5/learned-cells
  are Phase 2, gated behind the cheap-cell baseline. If they don't beat hand-cells at budget, we drop them and say so.
- **LLM-in-loop adds token cost** — bounded to cell-selection granularity; we report the call count and keep the
  env-step budget matched (the axis we win on).
- **Some tasks saturate at 1.0** (MountainCar, Breakout@thr3) — discrimination lives on DoorKey/FourRooms/Sokoban/graded.
- **The reset asymmetry is real** — we address it with the RL+archive baseline, not by pretending it away.
- **Evolution may not beat the hand-written PUCX** (it barely beat the hand seed before) — that's itself a reportable
  finding (the lever is the mechanism + fitness, not the search operator), consistent with our prior results.

---

## 9. References (verified arXiv IDs; venue-cited where no arXiv)

PromptPO 2605.30719 · Code-as-Policies 2209.07753 · Voyager 2305.16291 · Eureka 2310.12931 · ADAS 2408.08435 · TITAN
2509.22170 · GBQA 2604.02648 · KLPEG 2511.02534 · Go-Explore 1901.10995 / Nature'21 2004.12919 · Latent Go-Explore
2208.14928 · Time-Myopic Go-Explore 2301.05635 · IGE 2405.15143 · DSAGE 2206.04199 · QDHF 2310.12103 · AURORA 1905.11874 /
2106.05648 · CVT-MAP-Elites 1610.05729 · QDax 2308.03665 · QDAIF 2310.13032 · SimHash counts 1611.04717 · pseudo-counts
1606.01868 · E3B 2210.05805 · episodic-vs-global 2306.03236 · Reflexion 2303.11366 · FunSearch (Nature 625, 2023) · ELM
2206.08896 · ReEvo 2402.01145 · EoH 2401.02051 · AlphaEvolve 2506.13131 · OMNI-EPIC 2405.15568 · Monte Carlo Elites
2104.08781 · Multi-Emitter MAP-Elites 2007.05352 · Agent57 2003.13350 · Successive Halving 1502.07943 · Hyperband 1603.06560.
_Flagged for re-check before camera-ready (recon-unverified): NovelD arXiv id (cite by NeurIPS'21 venue), QDax first author,
MAP-Elites original 1504.04909, and PromptPO fine-grained specifics (Gemini-3-Pro, N=10, seed count)._
