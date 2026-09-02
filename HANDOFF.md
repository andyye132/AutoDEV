# AutoDev — Handoff / Full Context (for a fresh Claude session)

_Last updated: 2026-09-01, ~19:30 PT. This is the single doc a new session should read first. Also read `CLAUDE.md` (rules), `PROJECT.md` (claims ledger), and `LOG.md` (timestamped decision ledger)._

---

## 0. TL;DR

- **Project:** an autonomous research loop that writes, evolves, and verifies **game-testing exploration programs** ("testers"). Target venue: **AutoMLR @ NeurIPS 2026** (a workshop about autonomous ML research — the human poses/verifies, the loop does the research). Author: Andy Ye (UW undergrad, solo). Claude = the research agent (writes code, runs experiments, writes Part 1). Andy verifies and writes Parts 2–3.
- **Deadlines:** abstract already registered on OpenReview (exploit-era numbers — MUST be swapped for the honest ones). OpenReview form said Aug 30; website says Sep 5 (extended). Andy is contacting organizers. Treat **Sep 5** as the wall.
- **Thesis (working):** _what an autonomous loop discovers, and how far it generalizes, is set by the **evaluator/fitness** it optimizes; the search is interchangeable, the fitness is the steering wheel._ (Andy is lukewarm on some of the six sub-claims — see §8.)
- **Paper:** `paper/main_v2.tex` (inputs `part1_v2.tex`, `part2_v2.tex`, `part3_v2.tex`). Compiles clean, **zero TBDs**, 16 pp. in a placeholder article class (swap for the AutoMLR Overleaf template before submit).
- **State:** benchmark + all experiments done and folded in; two "final push" experiments running (§6); repo continuously committed+pushed to `github.com/andyye132/AutoDEV` (trailer-free — see §5).

---

## 1. The one honest result table (fixed harness, 5 seeds 100–104, exclusions applied)

`search` = seen levels; `held-out levels` = 30% of levels sealed from search; `held-out class (OOB)` = an entire defect class (arena-escape) never scored during search. Cells = score / % detected. Score = `1 − min(steps,budget)/budget`, budget 128×2048.

| Tester | Search (seen) | Held-out levels | Held-out class (unseen) |
|---|---|---|---|
| Random | 0.456 / 49% | 0.539 / 56% | 0.105 / 12% |
| **Action-repeat random** (best hand-designed) | 0.569 / 60% | 0.579 / 59% | 0.259 / 29% |
| Best PPO (count / curiosity-prox) | 0.492 / 57% | 0.529 / 57% | 0.184 / 27% |
| Best Opus one-shot design (pair-ladder) | 0.600 / 64% | 0.582 / 60% | 0.300 / 33% |
| **Split-and-pump** (survey program) | 0.613 / 65% | 0.580 / 60% | 0.338 / 37% |
| **Run 11 best** (searched, hard-tier fitness) | 0.624 / 65% | 0.585 / 60% | 0.327 / 36% |
| **Run 13 best** (searched, multi-engine regret) | 0.619 / 64% | 0.588 / 61% | 0.326 / 36% |

- **Seen vs unseen headline:** on seen levels the best testers hit ~65% detected; on the sealed **class** ~36–38%. BUT ~55% of sealed-class targets are **physically unreachable** (confirmed by the pre-registered T2 run + a witness-told guided probe, both add 0 targets beyond a 66/148 witnessable set). **Against the witnessable denominator the leaders find ~91% of the unseen class.** Always report both numbers.
- **Significance:** everything above action-repeat is instance-significant on search + sealed class (paired bootstrap CIs exclude zero). Run 11 is the only program with a significant *search* gain over split-and-pump (+0.011±0.009). Run 13 ties run 11 on Kinetix but adds cross-engine transfer.

---

## 2. The benchmark (this is a real contribution — feature it; the abstract currently under-sells it)

- **Substrate:** Kinetix 3.0 (Jax2D 2D physics). 74 hand-authored "shipped" levels (s/m/l tiers) + 64 generated levels.
- **Planted defects → witness states.** 5 classes, each with a programmatic witness predicate on the post-step state:
  - OOB (boundary wall removed → a moving body leaves the arena),
  - TUNNEL (obstacle made non-solid → a collidable body's centroid inside it),
  - SHORTCUT (neutral shape given the goal role → win by touching it),
  - ANOMALY (restitution raised → speed over a per-level threshold),
  - DISARMED (hazard made harmless → green body touches it, episode continues).
- **Suites:** K-Defect (337 planted, shipped levels), K-Fresh (452 planted, generated levels), K-Goal (52/22, the level's own win condition — no injection).
- **Rigor that no prior game-testing benchmark combines:** (a) sealed levels **and** a sealed class; (b) a **reachability calibration** (T2: a pre-registered long-budget run defines the witnessable denominator); (c) **replay verification** — every counted hit re-simulates from the level reset along its recorded action chain and must re-trigger the oracle, else it scores as a miss; (d) an **exclusion policy** (unwitnessable-by-construction + free-hit instances dropped, counts stated in captions); (e) a **frozen protocol** hashed in `autodev/protocol.py`.
- **6 engine families for breadth:** Kinetix (physics), Gymnax (MountainCar/Acrobot/DeepSea), MiniGrid (DoorKey/FourRooms/Unlock), MinAtar Breakout, Jumanji Sokoban, Craftax. Adapters wrap each to a common contract (reset/step, an 11-slot feature head, witness oracle).

---

## 3. Architecture / repo map

- `autodev/tester.py` — the vectorized JAX harness. B=128 workers × K=256 archive slots per level; a program supplies `init/restore/act/archive` hooks. **One writer per slot per step** (the fix for the exploit). Append-only lineage log → any state's action chain is reconstructible. `verify_chain_harness()` = the exact replay verifier. `AUTODEV_LEGACY_RACE=1` reproduces the pre-fix race for audits.
- `autodev/search_eval.py` — `evaluate_hooks(...)`: scores a program on a split; verification default-on for Kinetix; `AUTODEV_REMOTE_EVAL=1` ships to a cluster fitness server instead (so no LLM creds leave the laptop).
- `autodev/protocol.py` — frozen splits/seeds/budgets/exclusions + hash.
- `autodev/exclusions.py`, `scripts/structural_feasibility.py` — exclusion policy + constant-input envelope probe (→ "constant-input tier" vs "exploration tier").
- `autodev/substrate.py` + `*_adapter.py` (gymnax/xminigrid/craftax/**jumanji**) — the 6 engine families. `make_substrate("jumanji:Sokoban-v0")` etc.
- `evolve/` — ShinkaEvolve loop. `run_evolution.py` (config + task prompt), `evaluate.py` (**aggregate()** = fitness → combined_score + Eureka-style text_feedback; now supports per-target **regret** vs `AUTODEV_TARGET_NORM` and **paired-Δ** vs `AUTODEV_REF_RESULT`), `methods/*.py` (24 hand-designed procedures), `baselines/`, `proposals*/`.
- `scripts/` — `eval_server.py` (GPU fitness server; **has a rung-1 racing gate**, `AUTODEV_RACE=1`), `eval_server_multi.sbatch` (multi-engine weighted fitness), `bench_methods.py`, `make_tables.py` / `methods_table.py` / `breadth_table.py` / `make_bug_table.py` / `paired_ci.py` / `cost_table.py` (all tables regenerate from logs), `fig_*.py` (figures), `scan_clean.py` + `classify_hits.py` + `update_bugs.sh` (real-bug scans → severity triage → bug table), `launch_run*.sh` (evolution launchers).
- `paper/` — `main_v2.tex` + `part{1,2,3}_v2.tex` + `table_*.tex` + `figures/`. `openreview_abstract.txt` = pasteable abstract. Superseded exploit-era files kept but not input.
- `runs/`, `results/` — evaluation JSONs, `bugs.json`, `breadth.json`, `best_known.json`, `miss_diagnosis.md/json`, `run7_code_analysis.txt`.

---

## 4. Hyak (SLURM cluster) — rules + how to use

- **Access:** `ssh hyak` (config now has `ControlPersist yes` + keepalives; still needs interactive Duo on a cold connect). Everything lives under `/gscratch/raivn/andy132/AutoDev` (a git-synced mirror of the laptop repo). **NEVER touch other users' files. At most 6 GPUs running at once** (`squeue -u andy132 -t R | wc -l`). Compute is NOT a bottleneck; wall-clock/serialization is.
- **Two conda envs (critical):** `conda_envs/autodev` = what `scripts/slurm/_common.sh` activates → fitness servers, `eval_programs.sbatch`, scans. `conda_envs/autodev_multi` = `eval_substrate.sbatch` / `eval_craftax.sbatch`. Install to the right one by **absolute** pip path and verify with its absolute python before submitting (a `pip install` broke `click` for fresh jobs once).
- **Fitness-server pattern:** submit `eval_server.sbatch` (single-engine) or `eval_server_multi.sbatch` (weighted multi-engine) → it polls a queue dir `runs/hyak/evalq*/{inbox,work,outbox,done}`, atomically claims programs, writes score JSONs. The laptop evolution loop ships candidates via `AUTODEV_REMOTE_EVAL=1 AUTODEV_REMOTE_QUEUE=...`. **Run multiple servers on one queue for N× throughput** (atomic claim handles races). Launchers cache a reference eval, then `python evolve/run_evolution.py ...`.
- **Sbatch pitfalls (each cost real GPU-hours once — all in `CLAUDE.md`):** `${PROGRAMS-default}` runs built-ins if unset → pass `PROGRAMS=`; only `eval_server*`/`eval_programs` convert colon SEEDS (`0:1:2`), scans take `--seeds` as an INT; TAG concatenates into `--out`; never touch a live queue's `work/`; result-writers must load-and-merge; **a candidate eval slower than Shinka's per-job timeout is logged "Incorrect" 0.0 → keep the fitness split small (42-target `split_run11.json` ≈ 8 min OK; full 127 ≈ 29 min NOT)**.
- **Monitoring:** the session used background `Monitor` tasks tailing job logs / polling `squeue`. A keepalive monitor pings ssh every 4 min. `run_in_background` Bash for one-shot waits.

---

## 5. Standing rules / preferences (from Andy)

- **Git:** for THIS repo only, **continuous commit + push authorized, NO Claude trailer** (memory: `autodev-push-policy`). Everywhere else the global rule holds: never commit/push. Commit at every meaningful checkpoint.
- **Model tiering:** Claude (Fable, top model) orchestrates and **writes the code/tests directly**. Fan out to **cheaper** subagents (`sonnet` for reading/inspection, `opus` for hard recon/judging) — never fan out Fable-priced agents. "use opus and sonnet for expensive tasks."
- **Real results only:** "don't mislabel data or screw with stuff to fake that result." Every table regenerates from logs; racy-era files (`*_racy`) never enter tables.
- **Ledger discipline:** LOG.md row times come from `date`, never estimates.
- **Autonomy:** Andy often leaves it running; keep pushing, report at natural checkpoints, don't block on questions that have a sensible default.

---

## 6. What is running RIGHT NOW (2026-09-01 evening) — the "final push"

Two experiments on the 6 free A40s (started after killing a misconfigured run 14):
- **Push #1 — Leave-one-engine-out matrix (hold KINETIX out):** `runs/evolve_lo_kinetix/`, servers `evalq_lo_kinetix` (fitness = Gymnax+MiniGrid+Sokoban only, Kinetix sealed). Q: can non-physics engines breed a Kinetix tester? Upgrades "generality is bought at the fitness" from run-7 anecdote to a rigorous rotate-the-held-out-engine result. Was at gen 3, score 1.549 (regret scale). **When done:** export best, eval on Kinetix protocol + all held-out engines; if it climbs a matrix table follows (hold out each engine in turn).
- **Push #2 — Exploit reproducibility (N=3):** `runs/evolve_exploit_repro_{A,B,C}/`, racy servers `evalqx{1,2,3}` (`AUTODEV_LEGACY_RACE=1`), 42-target split, seeds initial/macrosweep/split_pump, 15 gens. Q: does a search on the racy harness *reliably* find chimera exploits (climb above the ~0.62 honest ceiling)? If yes, finding (1) upgrades from "we got fooled once" to "untrusted evaluators reliably corrupt discovery." **EARLY RESULT (gen 0, all 3 seeds): the exploit reproduces before search even starts** — split_pump scores 0.783 on racy vs 0.613 fixed (+0.17), macrosweep 0.694, go-explore 0.591. The racy evaluator measurably inflates any restore-heavy program via chimeras. This revives finding (1) as a QUANTIFIED, N=3-reproducible result (answers Andy's 'just a bug' skepticism). Let the searches climb further, then compute the racy-vs-fixed gap per run.

Already-done pushes this session (folded into the paper): run-7 mechanistic analysis (why it transfers — reads only the generic head, discovers dims at runtime; `results/run7_code_analysis.txt`), bug-portfolio result (27/28 via 6-tester portfolio; 2 severity-A tears each found by one scanner; bug-rank ⊥ benchmark-rank), and the **bug-detector scoping result** (invariant oracles run clean on the 4 discrete engines → all anomalies localize to the continuous-physics solver; bug-finding is Kinetix-specific by engine nature, not tester limitation).

---

## 7. The 28 real bugs (Result 2)

Generic engine oracles (OOB / PENETRATE / JOINT_TEAR / BLOWUP), replay-verified, severity A (sustained escape / joint tear) or B (momentary crossing / deep penetration / runaway). **28 distinct (level,anomaly) pairs: 7 A, 21 B.** On **shipped** hand-authored levels: 3–4 pairs incl. the headline **`h15_thrustshoot` OOB (A)** — a thruster launches a neutral block clean through the boundary wall, out of the arena for ≥41 steps, **player-triggerable**. The rest are on generated levels. 6-tester portfolio → 27/28; two severity-A joint tears are each found by a single scanner. **Bug-finding is Kinetix-only** (physics oracles don't apply to grid engines — an honest scope, explained in §1.7).

---

## 8. OPEN QUESTIONS + Andy's live feedback on the abstract (ACT ON THESE)

Andy read the 6-way abstract and flagged:
- **(1) exploit-as-finding — skeptical:** _"this seems like just a bug that we had… kind of worthless as a finding."_ → Decide: keep it as a genuine methodology finding (evaluator trust) only if the **N=3 reproducibility** (push #2) lands; otherwise demote to a Part-2 cautionary paragraph, not an abstract headline.
- **(2) equal-budget win — his favorite:** _"the best claim here… build this one up."_ → Strengthen: sharpen the vs-PPO / vs-human framing, the cost-curve crossover, maybe add a human anchor.
- **(3) headroom placement — unclear:** _"what does this even mean?"_ → Rewrite in plain language or demote. (It means: the loop only improves when its fitness targets tasks where testers actually differ; runs 9–13 are the evidence.)
- **(4) evaluator breadth — ok**, and now stronger (Sokoban makes it **three** never-seen families; run 13 → five; leave-one-out matrix incoming).
- **(5) composition ties — unclear:** _"what is this?"_ → Clarify or cut. (It means: combining finished programs — portfolios/schedules/hybrids/LLM-routing — never beats just changing the fitness; supports the thesis but reads as jargon.)
- **(6) ok.**
- **Andy says the abstract is MISSING: (a) the benchmark as a named contribution, (b) the seen-vs-unseen accuracy numbers.** Add both. Use §1 and §2 numbers here.

**Suggested reframe direction (for the next session to consider with Andy):** lead with the **benchmark + the equal-budget "LLM-written testers beat RL at 5–9× less cost"** result (his favorite), keep breadth/generality as the second pillar, fold the exploit + headroom + composition into the methodology/rigor narrative rather than as co-equal headline "findings." The "steering wheel" thesis can stay as the connective tissue but Andy finds some sub-claims thin — don't over-commit to "six findings."

---

## 9. Immediate next steps (recommended order)

1. Harvest push #1 (leave-one-out) + #2 (exploit-repro) when they land; update §1.3/§1.6 and the abstract accordingly.
2. **Rewrite the abstract per §8** (benchmark featured, seen/unseen numbers in, findings (3)/(5) plain-language or demoted, (1) contingent on N=3).
3. **Swap the OpenReview abstract** to the honest numbers (`paper/openreview_abstract.txt`) — Andy's action, needs the reframed version.
4. Full verification pass is DONE once (all tables regenerate byte-identical from logs, Sep 1); re-run after any new result.
5. Human-playtester anchor (`runs/human/README.md`) — Andy's action; the one thing the loop can't produce.
6. **Freeze experiments** (~Sep 2) → verification + Andy writes Parts 2–3 → camera-ready figures → submit.
7. Optional stretch (Andy lukewarm): Craftax bug oracle; racing-speedup figure from the run-14 infra.

---

## 10. What NOT to redo / traps

- Runs 1–8 are exploit-era — quarantined, never cite their numbers.
- Two programs (`run7_best`, `resonator`) **hang** on the 128×2048 Kinetix protocol eval (stated as a limitation) — don't burn hours re-trying; their cross-engine/other numbers are fine.
- The sealed-class "37%" is **~91% of the witnessable set** — never quote 37% without the reachability context.
- Don't build cheap invariant bug-oracles for the discrete engines expecting bugs — they're correct by construction (0/4000 on Sokoban); the null is already stated honestly.
