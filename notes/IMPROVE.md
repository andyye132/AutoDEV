# Improvement Proposal — one method and one test per claim

Companion to `PROJECT.md`. For every claim: where it stands, the improvement methods ranked by expected gain, the test that decides whether each method worked (pass criterion fixed before the run), and the compute it needs. Everything below runs in parallel on the six GPUs; nothing waits for a day boundary. Numbers are single-run, 5 seeds, 128×2048 steps unless stated.

## C6 — ≥ 90% of reachable targets in a single run (necessary)

*Now:* 88% / 96% / 76% of the current reachable set (search / held-out levels / held-out class); the held-out class is the fight. The reachable set itself is being re-based by the pre-registered long-budget (4×) union (chain C, jobs 39311723/24).

| # | Method | Why it should help | Test (pass ⇒ keep) | Compute |
|---|---|---|---|---|
| 6.1 | **Search on the reachable-only split with GPU fitness** (run 6, seed = run-4 best) | the 24 unreachable search instances are constant zeros that shrink the fitness range and mask real gains; run 4 already gained +0.036 on the noisy split | run-6 best re-evaluated on held-out splits ≥ run-4 best by > 0.01 on the held-out class | H200 server, ~5 h |
| 6.2 | **Extremes archive ("elites") procedure** — a hand-designed MAP-Elites-style archive keyed by per-dimension extremes (max/min x, y, speed, per-body distance to walls) with restores biased to the frontier of each dimension; then used as a *seed* for a search (run 7) | escapes are extreme-position witnesses; the evolved program's "high-speed pool" is a special case of an extremes archive; a principled frontier should reach more of the OOB class | elites ≥ action-repeat random on the held-out class alone; run 7 from elites ≥ run 6 | 1 eval job (1.5 h) + a search on a server |
| 6.3 | **Budget-matched restart ensembles** (queued: chain B) | the 5-seed union reaches 80% vs 72% single-run; 4 restarts at ¼ budget each may recover most of that inside one budget | ensemble ≥ single run + 0.03 on the held-out class at equal total budget | queued |
| 6.4 | **Target-directed designs** (round 3, chain C) and the **adaptive portfolio** (chain A) | class-specific mechanisms for the classes no generic search improves | any of them ≥ evolved best on the held-out class | queued |
| 6.5 | **Per-level adaptivity inside one run**: a bandit over sub-strategies keyed by level features (the adaptive portfolio generalised) | unseen levels differ in which mechanism works; a fixed program cannot switch | adaptive ≥ fixed portfolio and ≥ evolved best on held-out levels | 1 eval job after 6.4 |

Decision rule: after 6.1–6.4 land, the single best *frozen* procedure is re-evaluated on the long-budget reachable set; if it is ≥ 90% on both held-out splits, C6 holds; otherwise the paper reports the number and which lever came closest.

## C5 — the procedures are useful (necessary)

| # | Method | Test | Compute |
|---|---|---|---|
| 5.1 | **Solve what RL cannot (K-Goal).** Same frozen procedures and the five PPO objectives on the level win condition; count levels solved by the procedure within 262k steps that no PPO objective solves; add PPO at 4× budget as the strong baseline | procedure solves ≥ 10 levels no PPO objective solves at 1×, and ≥ 5 at PPO's 4× | chain A (queued) + one extra PPO job at 4× |
| 5.2 | **Cost per target.** From every evaluation: median simulator steps to first witness on the instances all methods find, and wall-clock per target on one A40; a table per suite | the procedure's cost per target ≤ ½ of the best PPO objective's | free (from logs) |
| 5.3 | **Cross-engine reach (C-Ach).** The same frozen procedures on 4 Craftax worlds × 10 achievements (easy → DEFEAT_NECROMANCER) vs random / sticky / Go-Explore; PPO-RND numbers from the Craftax paper as the published reference | discovered ≥ Go-Explore seed on ≥ 6/10 achievements; any intermediate achievement (ENTER_DUNGEON, OPEN_CHEST, ENTER_GNOMISH_MINES) reached within 65k steps | 1 GPU job (~2 h) once the Craftax env is built |
| 5.4 | **Real anomalies as validation.** Evolved-tester scans (running) → replay-verified table with severities; the upstream report to the Kinetix authors (draft ready in `runs/kinetix_issue/`) | table complete; report sent by Andy | scan chain (running) |
| 5.5 | **Human anchor.** Ten planted instances (two per class) as playable Kinetix levels; 3–5 people play each for 5 minutes with the same witness oracles logging first-hit; compare their detection and steps to the procedures' | reported as a number, whatever it is | Andy recruits; I build the playable bundle + logger (2 h) |

## C1 — the loop's advantage is general

| # | Method | Test | Compute |
|---|---|---|---|
| 1.1 | **More suites with the same frozen procedure**: K-Goal, K-Fresh (defects on unseen generated levels), C-Ach | the procedure ranks first on ≥ 2 of the 3 new suites | queued / pending env |
| 1.2 | **Framework-agnostic feature interface**: the 11-slot semantic head (position, velocity, speed, agent, target distance, time, target position) + a hashed state window, implemented for Kinetix and Craftax (done) and a third JAX env (Jumanji or XLand-MiniGrid) in ≤ 100 lines | the Go-Explore seed and the discovered program run unmodified on the third env | 1 day if time allows |
| 1.3 | **Baselines the field expects**: CCPT-style curiosity ⊕ imitation (imitation from K-Goal solution chains), and PPO at 4× budget | procedure still first at equal budget; the 4× PPO gap reported honestly | 2 jobs |
| 1.4 | **Rival discovery methods at equal budget**: evolutionary (runs 4–6), agentic loop, tree search, one-shot expert designs (now scored on the same GPU fitness), constant tuning, restarts — every rival's best program gets the same held-out evaluation; figure: best fitness vs evaluations, held-out bars per rival, cost per gain | the figure exists and the winner is stated with CIs | pilots running; full runs at 40 evaluations each (~8 h of server time) |

## C2 / C3 — transfer to unseen levels and to an unseen class

| # | Method | Test | Compute |
|---|---|---|---|
| 2.1 | **Statistical power on unseen levels**: K-Fresh adds 64 levels' worth of instances to the held-out-level evidence | paired per-instance CI on held-out levels excludes zero when pooled with K-Fresh | chain B (queued) |
| 2.2 | **Search on a broader level distribution** (run 7): fitness over shipped search levels + generated levels (mixed pool), so the procedure is not tuned to 52 levels | run-7 best ≥ run-6 best on K-Fresh and on held-out levels | needs the K-Fresh pool (chain B) + a server |
| 2.3 | **Per-class transfer analysis**: report which classes carry the held-out-level gain and which are feasibility-capped (DISARMED) | in the paper | free |
| 3.1 | **Second held-out class** (run 3, ANOMALY held out; evaluation in chain C) | run-3 best beats the baselines on the unseen ANOMALY class | queued |
| 3.2 | **Class-rotation as a protocol**: one run per held-out class (5 runs) if the servers allow | transfer gain reported per class with CI | ~5 h per run on a server |

## Visualizations and statistics to build as results land

1. Best fitness vs evaluations for every discovery method (evolutionary runs 1–6, loop, tree; one-shot designs as points; tuning/restarts as horizontal bands), same fitness axis.
2. Held-out bars (search / held-out levels / held-out class) for each rival's best program, 5-seed CIs; paired per-instance CIs vs action-repeat random.
3. Steps-to-target curves per suite (K-Defect, K-Goal, K-Fresh, C-Ach); cost-per-target table.
4. Coverage vs detection scatter per tester (the CCPT inversion).
5. K-Goal "solved by procedure, unsolved by PPO" level list with steps.

## What runs where (now)

A40 ×5: scan/classify chain; chain A (K-Goal → run-4 → adaptive); chain B (K-Fresh → restarts → T4 → coverage); chain C (long-budget → round 3 → run 3); fitness server (run 5, scaffold pilots, four designs). H200 ×1: fitness server → successor for run 6. Craftax pilot joins chain C when its env is built.
