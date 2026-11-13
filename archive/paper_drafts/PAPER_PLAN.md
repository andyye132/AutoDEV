# AutoMLR 2026 submission — whole-paper plan (living doc, Sep 3 2026)

**Thesis (one line):** an autonomous loop *writes* a game-testing exploration program that is the **only method robust
across 8 environments** (beats trained RL, Go-Explore, sticky at equal budget) — and every counted result is
**execution-verified** by replay-from-reset, a check that caught a real exploit the loop itself found.

**Venue fit:** AutoMLR's thesis is that *verifying* AI-generated results is the scarce act. Our two assets map exactly:
Part 1 = a strong, checkable ML result; Part 2 = a hard-to-fake verification apparatus + the exploit-caught story. Lead
on checkable truth, not on the win.

**Logistics.** Deadline **Sep 6 ~4:59 AM PDT** (Sep 5 AOE). OpenReview `NeurIPS.cc/2026/Workshop/AutoMLR`, blind.
Non-archival (can submit elsewhere later). Reciprocal reviewing: designate min(3, #authors) reviewers. Official Overleaf
template: https://www.overleaf.com/read/bbxvpfghmbvb — swap our placeholder `documentclass`/preamble into it.
Page limits are **separate and non-reallocable**: Part 1 ≤4pp, Part 2 ≤4pp, Part 3 ≤1pp; references excluded.

---

## Part 1 — Auto research result (agent-written, ≤4pp) — DONE (`part1.tex`)
Headline = **robustness**. §1 Intro (7-move arc: stakes → RL's price → 3 failure modes → write-not-train reframe →
contributions → robustness kicker → honest scope). §2 Related work (4 families + capability Table 1). §3 Method (Fig 1
pipeline; benchmark; testers-as-programs; the one discovered program's 3 modules; verification in one line → Part 2).
§4 Results (Fig 2 heatmap = no cold cells, avg 0.83 vs 0.60/0.55/0.35; Fig 3 budget curve = ours@1× = PPO@16×; Kinetix
depth 0.655 vs 0.566; strategy-not-archive; transfer). §5 Limitations (equal-budget only; action-repeat close on physics;
archive asymmetry; Kinetix-only replay; best-of-5 PPO). **Fits ~4.2pp loose → ≤4pp in the dense template.**

## Part 2 — System design + verification (human-written, ≤4pp) — DRAFTING (`part2.tex`)
Human-voiced, per venue rule. Structure:
1. **The autonomous loop (system).** The benchmark builder; the LLM evolution loop (ShinkaEvolve; Sonnet+Opus proposers;
   islands; multi-engine fitness at fixed budget; $-capped); the tester representation (4 JAX hooks). Prompts/harness/tools.
2. **Verification — the machine.** Replay-from-reset of *every* counted hit; frozen hashed protocol; sealed level split
   **and** a sealed defect class; reachability calibration (the ~91%-of-witnessable denominator); racy-file exclusion.
3. **The exploit (the emotional core).** The loop earned +0.11 fitness from a concurrency race producing physically
   impossible "chimera" states; it passed *every evaluator-mediated check* (fitness, both held-out splits, four design
   comparisons, an LLM code-audit of the candidate) and briefly topped the registered abstract for ~10h; **replay-from-
   reset was the only thing that caught it.** Narrate before/after numbers; the N=3→N=1 correction we made ourselves.
4. **Human interventions (disclosed as a feature).** Oracle/witness design; protocol freeze; noticing the inflated result
   + rebuilding the harness + withdrawing tainted numbers; the PPO LR/entropy tuning sweep run to close the fairness
   question; verifying every number against regenerable logs.
5. **Significance commentary (venue asks for it).** Why the robustness result matters, and what the verifier demonstrates
   for autonomous research at large.

## Part 3 — Reflections (human, ≤1pp) — TO ADAPT from `part3_draft.md`
Two beats: (a) *write-vs-train testers* is a regime, not a rivalry — the loop maps the boundary cheaply. (b)
**Execution-grounded verification** (replay-from-reset + programmatic witnesses) is a template for trusting *any*
autonomous result — you cannot certify by reading code or score; you must re-execute from a clean state. Answers the
venue's "how should the field adapt?" directly.

---

## Figures / tables inventory
- **Fig 1** `figs/pipeline.pdf` (editable `figs/pipeline.tex`) — 4-stage methodology. DONE.
- **Fig 2** `figs/heatmap.pdf` — robustness heatmap (method×task). DONE.
- **Fig 3** `figs/budget_curve.pdf` — detection vs compute. DONE.
- **Table 1** capability grid (in `part1.tex`). DONE.
- Part 2 may reuse: `table_bugs.tex` (28 anomalies) or a small exploit before/after table.

## To-do → submission (ordered)
1. [ ] Draft Part 2 (this turn) → Andy voices/edits (human authorship).
2. [ ] Adapt Part 3 from `part3_draft.md`.
3. [ ] Adversarial critique pass (accuracy of every verification claim; overclaim check; AutoMLR-fit; page limits).
4. [ ] Numbers-consistency sweep — kill any stale figures the audit flagged; confirm abstract/Part 1/Part 2 agree.
5. [ ] Drop into the official Overleaf template; fit each Part to its page limit; final read.
6. [ ] Confirm on OpenReview: abstract registered, PDF uploadable, reviewers designated, attendance question answered.
7. [ ] (rigor, optional) 5-seed K-Defect PPO — **needs `eval_ppo.sbatch` patched** (it hardcodes `--seeds 0,1,2`);
       until then footnote the 3-seed PPO honestly. Not load-bearing for the robustness headline (7/8 heatmap PPO is 5-seed).

## Standing honesty caveats to keep in the text (from the Sep-3 audit)
- Replay-verification is Kinetix-only; stochastic-engine detection = the engine's own reward/achievement signal on the
  genuine post-step state (stated in Part 1 §5 and Part 2 §2).
- Every PPO number is best-of-5 objectives (stated).
- Equal-budget claim; RL catches up at 4–16× on reward-dense tasks (stated).
- Action-repeat is a close competitor on Kinetix physics (stated).
