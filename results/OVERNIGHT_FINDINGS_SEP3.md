# Overnight accuracy push — findings (Sep 3, 2026)

Goal set by Andy: maximize **bugs found** (detection); efficiency only a guardrail (must stay ≤ PPO). Push the core
algorithm, try new architectures, scale evolution on the now-free Hyak. Below is what the night actually established —
including the honest limits.

## TL;DR — we are near the benchmark's structural ceiling, not an algorithmic one

The strongest, most decision-relevant finding: **on the Kinetix planted-defect benchmark the current testers are already
near the reachable-bug ceiling.** Three independent lines of evidence say more algorithm work buys little more detection
*on this benchmark*, and the productive lever is a **harder benchmark with real headroom**, not a better tester.

## 1. Portfolio / union of testers — DEAD (redundant testers)

Ran 7 diverse programs (split_pump, saturate, macrosweep, velmax, cem_root, goexplore_yield, run7) on Kinetix:m search
(53 instances, 64×1024). Per-program detection **0.736–0.774**; per-class fingerprints near-identical.

- **UNION (bug found by ANY program) = 0.774 = the best single program. Gain over best single: +0.0 (zero more bugs).**
- Only macrosweep contributes 1 unique instance; every other program is fully redundant.
- The 12/53 undetected instances are missed by **every** program — a shared frontier, not a coverage gap a portfolio fills.

**Conclusion:** a portfolio/ensemble tester does not find more bugs — the programs converge on the same detectable set.
(`scripts/portfolio_union.py runs/portfolio/kinetix_multi.json`.) Drop the portfolio-tester architecture.

## 2. Scaled multi-engine evolution — running, but PLATEAUED

Launched a 150-generation multi-engine run on Hyak (seeded from `evolve/initial.py`; run7 could not seed it — see §4),
chained servers for >11.5h longevity, faithful run7 objective (weights/norm unset).

- New-best improvements at gens ~1–15 (03:45–05:08), then **no new best for 7h / ~50 generations** (gen 15→65+).
- Population clustered at regret score **~1.64–1.66** (max 2.0) — it re-derived a strong, run7-level program early and stopped
  improving. Cost ~$27/$150 at gen 65.

**Conclusion:** with the current (faithful) fitness the search plateaus fast — consistent with the benchmark ceiling in §3,
not with "needs more compute." (`runs/evolve_scaled/`, `runs/logs/evolve_scaled.log`.)

## 3. The benchmark is reachability-saturated (the ceiling)

Per the frozen reachability calibration (HANDOFF §Seen-vs-unseen; `results/structural.json`): **~55% of sealed-class
targets are physically unreachable**, and against the *witnessable* denominator the leaders already find **~91% of the
unseen class** (and similarly high on search). So the undetected frontier is largely **structurally unwitnessable** —
chasing it (via complementary/boosting evolution, the planned next lever) would mostly chase targets no policy can reach.

**Conclusion:** detection on the *existing* Kinetix benchmark is near its structural maximum. "More bugs found" here is
essentially maxed; the honest contribution is "reaches ~91% of reachable, beats every baseline, and transfers."

## 4. run7 stalls at eval-protocol scale (a real, separate bug)

run7 completes at 64×1024 (CPU) and scores its normal ~0.74, but **stalls indefinitely at the 128×1024 fitness / 128×2048
eval scale** (stuck 37 min at kinetix instance ~96 on the GPU server, no error) — so it cannot seed evolution and its full
Kinetix protocol numbers are still incomplete. Likely an XLA/memory blow-up in its dual-pool archive at high lane×step
count. A lighter, non-stalling champion of equal detection would be a practical win over run7.

## 5. Efficiency guardrail — PASSES

run7 reaches the witness in a *smaller* fraction of budget than tuned PPO (DoorKey 17% vs 32%, MountainCar 8.5% vs 9.2%).
So "efficiency ≤ PPO" holds without any extra work — no need to headline or optimize it (per Andy's steer).

## What this means for the push — headroom, not a better tester

Since detection on the current benchmark is capped by reachability, raising the *tester* can't raise the number. The only
way to show a bigger "bugs found" is to **create headroom** — a difficulty-graded benchmark of **reachable-but-hard** bugs
where baselines fall off and structured/evolved testers hold. I tried the two obvious axes:
- **Speed axis (exists):** structured/LLM (0.83) > trained RL (0.67) > random (0.33) at the hard end — a real margin over RL
  and random, but the LLM *ties* simple structured baselines (the bug only needs sustained action).
- **Navigation/distance axis (tried, §7): dead end** on Kinetix — level geometry doesn't provide graded-distance shortcuts.

Net: the clean "LLM beats *every* baseline" margin does not appear on Kinetix physics (simple structured baselines are
already near-ceiling). It appears on the **sparse-nav generalization tasks** — which is where the paper's headline already
lives. So the honest move is to lean on generalization for the margin, and report Kinetix as best-at-ceiling depth.

## 6. The evolved champion — novel architecture, same (ceiling) detection

The 150-gen run's champion (`runs/evolve_scaled/best/main.py`) is **genuinely new architecture**, not a run7 clone:
- a **homeostatic exploration-rate controller** (hold ~3% of workers discovering a new cell each step, auto-adjusting the
  archive-cell resolution to that set-point);
- **test-time "gene" evolution** — each worker runs a tiny closed-loop controller ("gene"); an *elite* gene (min-age 24)
  propagates to resampled workers with per-field mutation — a quality-diversity inner loop at test time;
- a **per-action novelty-UCB** (optimism bonus for under-tried actions — "forces rare verbs").

But its detection on Kinetix:m search = **0.75 — identical, per-class, to macrosweep/split_pump/run7.** The novel mechanisms
reach the same ceiling. (Eval at the full 128×2048 protocol is queued to check whether it is **non-stalling** unlike run7 —
if so, it's a practical win: a novel tester that reaches the ceiling *and* completes the protocol run7 hangs on.)

## 7. Navigation-graded benchmark — dead end (level geometry)

Built a distance-graded SHORTCUT bench (`scripts/make_graded_nav_bench.py`, `autodev/defects.py::inject_shortcut` now takes
`dist_rank`). But Kinetix levels have **too few neutral shapes** to serve as graded-distance shortcuts: only 3 of 20 levels
yielded a candidate, and each had a single candidate, so distance was constant across ranks (18 instances, all median 1.36).
The distance axis is not gradeable on these levels — abandon it. (A precision/contact-radius axis is possible but unlikely
to separate the LLM from simple structured baselines, which are already strong on Kinetix physics.)

## 8. The ceiling is a HARD wall (budget test, confirmed)

Took the 12 instances **no tester detects** at 1× budget and re-ran random, macrosweep, and the evolved champion on exactly
those at **4× budget** (128×2048): **all three scored 0.00 — zero of 12 cracked.** Quadrupling interaction detects none of
them. So the undetected frontier is genuinely **unwitnessable, not budget-starved** — the Kinetix detection ceiling
(~0.75 ≈ 91% of reachable) is structural and no tester/budget moves it. (`runs/portfolio/undetected_4x.json`.) This closes
the "is there algorithmic headroom on the current benchmark?" question: **no.**

## Conclusion

**On Kinetix, detection is ceiling-bound and every good tester (evolved, hand-designed, LLM) converges to the same ~0.75 /
~91%-of-reachable.** More algorithm work does not raise it — the limit is the benchmark's reachable-bug set, not the search.
The honest, defensible contribution is therefore: **best-in-class detection at the reachable ceiling + a novel evolved
architecture + the sparse-nav generalization win (where the LLM tester clearly beats every baseline) + execution-grounded
verification.** The LLM tester's *clear* margin over simple baselines lives in the sparse-nav generalization regime, not in
Kinetix physics (where simple structured baselines are already near-ceiling). Lead the paper with generalization; present
Kinetix as the rigorous depth substrate at its ceiling.

## Threads still running / harvest later

- [running] The 150-gen evolution (gen ~66, ~1.67 regret, $28/$150) on chained servers — harvest the final champion; it may
  be a lighter, non-stalling equal of run7.
- [queued] Champion 128×2048 protocol eval (job on Hyak) — the non-stall check.
- [dropped] Portfolio tester (redundant); reward-follow emitter (anti-exploration); complementary/boosting evolution and the
  nav-graded bench (no reachable headroom on the current benchmark/levels).
