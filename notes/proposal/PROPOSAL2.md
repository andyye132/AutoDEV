# AutoDev, Second Proposal

Where the system stands after the first night of autonomous runs, what the "72%" actually measures, a side plan for making the tester better, and three paths forward with a recommendation.

Prepared 29 Aug 2026 (Saturday morning) · Paper deadline Fri 4 Sep, 5 PM PDT · All numbers below are from Hyak runs (size-l configuration, 74 shipped Kinetix levels, 5 seeds, 262,144 simulator steps per instance) unless marked *local*.

## 1. What we built, in plain terms

**The game.** Kinetix [1]: 2D physics puzzles (vehicles, arms, thrusters, balls) inside a 5 m × 5 m arena; a level is a JSON file; the whole simulator state is a data structure we can save and restore at will, and thousands of copies run in parallel on one GPU.

**The bugs we plant.** For every shipped level we create broken variants in five classes, each with a *witness state* — a condition that, if a tester ever reaches it, proves the bug: a boundary wall removed (witness: a moving shape leaves the arena), a static obstacle made non-solid (a moving shape's center ends up inside it), a neutral shape given the goal role (the level is "won" by touching the wrong thing), physics made bouncy (speed exceeds a per-level threshold), a hazard made harmless (a green shape touches it and survives). 337 instances across the 74 levels; 3 that any tester finds within 512 steps are excluded as trivial.

**The bugs we look for without planting.** On unmodified levels: a moving shape leaves the arena; a moving shape's center sits inside a solid it should collide with (honoring the engine's own collision matrix); a joint's two anchors tear apart; speed or spin runs away or the state goes NaN. Each has a persistence requirement so one-frame solver hiccups don't count, and every flagged state is re-simulated from the level's reset along the recorded action chain before it is counted. A post-hoc classifier assigns severity (A: escapes/tears; B: deep or long penetration, long runaway; C: minor transient).

**The testers.** A tester is a *program* controlling B parallel workers through three hooks — *act* (choose actions), *restore* (jump a worker back to an archived state), *archive* (decide which states to keep) — over an archive of saved simulator states. Hand-designed baselines: uniform random, action-repeat random, a Go-Explore-style archive search. RL baselines: PPO with count, RND, ICM, curiosity-proximity, or task-return objectives. Seven independent designs written by Claude Opus on request (boundary-seeking, energy pumping, macro tree search, contact-hunting, pair-covering, distance-ladder frontier, goal-biased RRT), and two portfolios that run several designs side by side.

**The search.** ShinkaEvolve [2] evolves the tester program: an LLM (Claude Sonnet 5; Opus 5 in run 2) proposes patches to the hook code, each candidate is scored by how fast its workers reach witness states on the *search* split, and an island-model archive keeps the best. Two splits are sealed before the search starts: 22 of the 74 levels, and one entire defect class — leaving the arena — which the search never sees.

**The score.** For each instance, 1 − (steps to first witness / budget), or 0 if never; averaged over instances and seeds. "Detection" is the fraction of instances witnessed within budget. Every claim is a comparison at equal budget.

## 2. What "72%" means — and what the ceiling looks like

72% is the fraction of planted defects the evolved tester reaches within 262k simulator steps in a single run (search split). It is not an accuracy, and the reference points are: action-repeat random 59%, the Go-Explore seed 57%, the best RL tester 56%, uniform random 49%. The headline is the *held-out class*: 44–47% of arena-escape defects for the evolved program vs. 30% for the best hand-designed tester and 25% for the seed it evolved from — a class the search never scored.

Two measurements say how much room is left:

- **Union ceiling.** Taking every tester we have and every seed, the union reaches 81% on the search split, 66% on held-out levels, 61% on the held-out class. The evolved program *by itself*, across its five seeds, reaches 80% / 66% / 56%. In other words, one evolved program already finds nearly everything the entire population of testers can find at this budget; the per-run number (72%) is lower than the across-seed number (80%) because different seeds find different instances — so restarts or ensembles of seeds are the cheapest way to raise single-run detection.
- **Feasibility ceiling (measured this morning).** An *oracle-guided* probe — a tester that is told where the witness is and descends on it — reaches 63% / 58% / 24% of instances by itself, *less* than the evolved tester, and adds **no** instance the union of testers had not already reached. So the union (81% / 66% / 62%) is the best reachability bound we have, and the probe's failure to extend it says the remainder is hard or infeasible at this budget rather than overlooked. Normalized by the reachable set, one run of the evolved tester finds **88% / 96% / 76%** of what is findable, against 73% / 90% / 48% for action-repeat random. The honest headline is therefore not "72%" but "88% of the reachable search instances, 76% of the reachable held-out-class instances, in a single run".

What would be bad: if the feasible ceiling turns out to be ~75% on the search split, then 72% is near saturation and the remaining gains lie in budget and breadth, not in cleverer testers. What would be good: if the guided probe reaches 90%+, there is a real 15-point gap for better methods to close.

## 3. Side plan: how to make the tester better (ordered by expected gain per day)

1. **Metric hygiene (½ day).** Report per-run, across-seed and feasibility-normalized detection; per-class budgets; paired per-instance tests with CIs. Already scripted; waiting on the guided probe.
2. **Fix the search signal (done this morning, no token needed).** The search plateaued after generation 1 because a two-seed fitness on the small local pool cannot rank the 0.69-band candidates (noise ±0.02). Instead of moving the LLM to the cluster, a *fitness server* now runs on a Hyak GPU (`scripts/eval_server.py`): the evolution loop stays on the laptop, and every candidate program is shipped to the server and scored on the full size-l search split (127 instances, 3 seeds, 128×1024 steps, ~10 min). Run 4 — seeded from the run-1 best, Sonnet+Opus proposals, 40 generations — starts automatically when the server job begins. A `claude setup-token` would still let me run several searches in parallel on the cluster, but it is no longer a blocker.
3. **Restarts and adaptive portfolios (1 day).** Same total budget, split over k independent seeds, is a free lift (80% vs 72% here). An *adaptive* portfolio — a bandit that reallocates workers among sub-programs by their recent archive growth — is the natural next design; the fixed 4- and 7-program portfolios are being evaluated now.
4. **Harder search distributions (1 day).** Use the fresh-level generator to produce a *bug-rich* search pool (the generated levels already yield 10 severity-A/B real anomalies in 64 levels), and rotate the held-out class (run 3, started this morning, holds out runaway physics instead of escapes) for a second transfer data point.
5. **Pool hygiene (½ day, found this afternoon).** An Opus agent's per-instance diagnostic showed that 3 of the 7 TUNNEL instances on the local search split are *geometrically unreachable* — the jointed arm's workspace never comes within 0.47 m of the injected obstacle — so the class's ceiling there is 4/7, which action-repeat random already attains. `inject_tunnel` (and the other injectors) should run a reachability probe at pool-build time, and infeasible instances should be excluded from the *search fitness* (they are constant zeros that shrink its dynamic range) while staying in held-out evaluation. The one TUNNEL instance that separates random from obstacle-targeting testers is a thin slab that must be swept *through* at speed — a traversal objective, not closest approach.
6. **Broaden defect classes (1–2 days).** Planted joint tears, one-way collisions, trigger/timer bugs, reward loops; these also sharpen the real-bug oracles.
7. **Tester capability (2+ days, after the deadline).** Learned components inside the archive loop (a small policy that returns to and extends promising states), deeper restore trees with explicit tree search over macro sequences, and per-level-type macro libraries.
8. **Real-bug pipeline (ongoing).** Scans of shipped and fresh levels with the evolved and Opus testers are queued; severity triage and rendered clips are automatic; the h15 wall-face tunnelling (a block at 16 m/s momentarily inside the right wall / 10 cm into the floor — *not* a ceiling escape; corrected by step-by-step replay) and the generator's spawn-outside-the-arena defects should be reported to the Kinetix authors (draft in `runs/kinetix_issue/`).

## 4. Other games, and where bug finding has and hasn't been tried

The survey below was run this morning by an Opus reading agent (163 tool calls; every arXiv ID re-resolved through the arXiv API, repository facts through the GitHub API); I aggregated and checked the parts that touch our claims. Two things it found change what we say about ourselves; they are in 4.3.

### 4.1 Games we can test against

The harness needs three things from a game: a *state we can save and restore*, a *fast step*, and *something to write oracles against* (rules, physics, or a reference implementation). Sorted by how much of our current machinery transfers:

| Family | Substrate | Content | Save/restore | Oracle material | Prior bug-finding work |
|---|---|---|---|---|---|
| JAX (state is an immutable pytree: restore = copy a variable; `vmap` on GPU; **our harness runs as is**) | Craftax [3] | 67 achievements; crafting, hunger, mobs, dungeon floors; MIT | free | symbolic rules (inventory, stats, tiles) | none found |
| | PuzzleJAX [4] | ~900 human-authored PuzzleScript games (93 curated); JS/C++ reference engine; MIT, source install | free (purity unverified) | *differential*: JAX vs. reference; `broken_games.md` already lists real engine bugs | none (ScriptDoctor [5] checks generated *design* playability) |
| | Pgx [6] | 23 board games incl. chess, shogi, Go; Apache-2.0 | free | built-in `legal_action_mask`; published *perft* move counts for chess/shogi = external ground truth | none found |
| | Ludax [7], Jumanji [8], XLand-MiniGrid [9], Gymnax, JaxMARL, NAVIX [10] | 16 board games + DSL; 22 routing/logic envs; ~3M grid tasks; 26 classic; 11 multi-agent; MiniGrid | free | rules per env | none found |
| | Brax | 12 robotics envs | free | physics laws | **already fuzzed** by PhyFu [11] — not a novelty target |
| C/CUDA throughput | PufferLib Ocean [12] (54 games), envpool, CuLE | arcade/grid games | **none** (no snapshot API in the code base) | pixels + scalar state | none |
| Classic RL suites with real snapshot APIs | ALE [13] | 97 Atari games; `cloneState/restoreState` (RNG-excluded, "for planning"); 128-byte RAM | yes | RAM addresses | RL agents as *players*; no bug search found |
| | Procgen [14] | 16 games; `get_state/set_state` with a restore-determinism test; MIT | yes | internal state | none |
| | stable-retro [15] | ~1,000 console games; `get_state/set_state`; **`data.json` names RAM variables (lives, HP, x/y)** | yes | named RAM = free oracles | none |
| | NLE / MiniHack | roguelike; structured state (`glyphs`, `blstats`) | **no** snapshot; seed replay only | rich symbolic | none |
| Emulators | PyBoy [16] | Game Boy; `memory[addr]`, `save_state/load_state`, headless ~395× realtime, 7 game wrappers | yes | RAM + wiki glitch lists | none as bug search; [17] *generates and verifies* environments from it |
| | mGBA | GBA; `save_raw_state/load_raw_state`, memory API | yes | RAM | none |
| Open-source PC games with shipped determinism checks | Wesnoth, Endless Sky, OpenTTD, 0 A.D., SuperTuxKart | strategy/space/tycoon/RTS/kart | replay/rewind (Wesnoth validates replays by default; Endless Sky `--test --rngseed`; OpenTTD desync logs; 0 A.D. per-turn state hashes; STK `rewind_manager`) | **their own desync/state-hash checks** + public bug trackers | none |
| Disqualified | Godot | — | — | docs state physics is *not deterministic* | — |

Kinetix, for reference, ships **74** hand-designed levels (10 S / 24 M / 40 L, counted from the JSONs; the "90" in one secondary source is wrong).

### 4.2 Benchmarks for bug finding — what exists, what it scores, what you can run

| Benchmark | What it holds | What the agent sees | Metric | Runnable? |
|---|---|---|---|---|
| GBQA [18] (ICLR 2026 wksp) | 30 web games, **124 human-verified bugs**, 3 difficulty tiers | backend API state (not video) | **recall of ground-truth bugs** at a step budget; best 48.4% (Claude Opus 4.6, 500 steps) | **no** code or data released |
| GameXpert-Bench / GameFix [19] (Aug 2026) | 50 real HTML5 games, 19–27 **injected** defects each (reversible mutation patches), 100 repair tasks | source + headless Chromium | SWE-bench style Fail-to-Pass / Pass-to-Pass | **no** ("closed set remains internal") |
| VideoGameQA-Bench [20] (NeurIPS 2025 D&B) | 4,786 questions, 2,236 images + 1,200 videos, >800 games, 9 tasks (glitch detection, visual regression, bug-report generation…) | pixels | per-task accuracy; LLM judge for reports | yes, CC-BY-4.0 |
| VideoGlitchBench [21] | 5,238 videos / 120 games with **temporal spans** | pixels | detection + grounding | yes, Apache-2.0 |
| TempGlitch [22] | 750 glitch + 750 clean videos; includes a **"stuck in place"** class | pixels | accuracy/F1; best of 12 VLMs 54.1% ≈ chance | yes, MIT |
| PhysGame → GameBench [23, 24] | 880 physics-violation videos, 12 categories | pixels | accuracy | yes, Apache-2.0 |
| GlitchBench [25] (CVPR 2024) | 607 glitch images (public release) | pixels | LLM-judged open answers; GPT-4V 43.4% | yes, MIT |
| CLIP meets GamePhysics [26]; GameBugDescriptions [27] | 26,954 r/GamePhysics videos / 1,873 games; 167 videos + 334 QA | pixels | retrieval / QA accuracy | yes |
| BugCraft [28] (ASE 2025) | 86 Minecraft crash reports | game + report text | end-to-end **reproduction** rate 34.9% | code yes (AGPL), data gated |
| World of Bugs [29] (CoG 2022) | Unity "bug zoo" with injected bugs | pixels | detection | yes, GPL |
| NEATEST [30] (ASE 2022) | 25 Scratch games; neuroevolved networks act as tests *and* oracles | game attributes | **mutation score >65%** | yes |
| PhyFu [11] (ASE 2023); IcFuzz [31] | physics *engines* (Taichi, Nimble, Warp, Brax 0.1.0); Isaac Sim | engine state | physics-law consistency incl. gradients; >5,000 error-triggering inputs | yes |
| Industrial reality check [32] | 19,738 real keyframes from 41 h of gameplay | pixels | VLM precision 0.50 / accuracy 0.72 vs 82.8% on curated sets | data no |

Three consequences. (i) **Every published game-bug benchmark judges from pixels or source; none reads simulator state**, and the one field study says pixel-based QA does not transfer to real footage — that is our one-sentence differentiator, and the reason our results are not comparable to any VLM number. (ii) The two benchmarks closest to us in spirit (GBQA: recall of verified bugs at an interaction budget; GameXpert: injected defects) released nothing, so there is *no public state-based bug-finding benchmark* — our planted-defect suite with witness oracles can be released as one, and we should score it the way GBQA does (recall at budget), which is what "detection within 262k steps" already is. (iii) There is **no softlock benchmark or dataset**; the only softlock papers are verification (Super Metroid with CTL model checking [33]; softlock-free level generation [34]), which makes "stuck states" a natural extra oracle for a save/restore tester (a state from which no branch reaches the goal).

### 4.3 Where no automated bug-finding paper was found — and two corrections to our own claims

*No paper found* (not "never done"): Kinetix, Craftax, Jumanji, Pgx, XLand-MiniGrid, Gymnax, NAVIX; roguelikes (NetHack/NLE, Cataclysm-DDA); PuzzleScript games; open-source strategy games (OpenTTD, 0 A.D., FreeCiv, Wesnoth); emulated retro games driven by RL; board games. Every arXiv hit for "Kinetix" is the original paper, a robotics/VLA paper *using* it as a benchmark, or an unrelated chemical-kinetics tool. Two structural gaps: save/restore as a bug-search primitive is nearly unexploited in games (Nyx-Net [35] showed snapshot fuzzing gives 10–30× on Super Mario but used the game only as a speed benchmark), and LLM-evolved test programs have targeted compilers and unit tests, never games.

The corrections: **PhyFu already fuzzes Brax** with, verbatim, our penetration oracle as its motivating example (a robot palm below the ground plane that the engine never flags). It is a different problem — engine numerics on robotics micro-scenarios, no levels, rules, win conditions, exploration, or evolved testers — but our novelty claim must narrow to: *no published work searches for game-level rule and physics violations in GPU game environments using save/restore-driven tester programs discovered by LLM search.* (Done in the paper this morning.) And the NEATEST ID we had cited (2507.09414) is a branch-coverage paper; the tests-and-oracles paper is 2208.13632 [30].

### 4.4 Ground truth that could anchor a real-games paper

TASVideos [36] publishes 4,615 deterministic tool-assisted runs with an *applied* glitch taxonomy (`ace` "executes arbitrary code" ×62, `warpbug`, `rambugs`, `savebugs` ×39, `skipboss` ×99) — a labeled corpus of known exploitable states, and a measurable target ("does the tester rediscover the known warp?"). The Mario wikis hold 237 catalogued glitches across SMB1/SMB3/SMW/SM64, already classified against software-weakness taxonomies [37]. Open-source trackers give live defect sets: 0 A.D. 852 open defects (on its Gitea; the GitHub mirror is archived), OpenRCT2 774, Cataclysm-DDA 579, PCSX2 503, Wesnoth 500, Luanti 365, SuperTuxKart 147, Endless Sky ~100, OpenTTD 53. Minecraft's Mojira is no longer scrapable; BugCraft's 86 curated crash reports are the usable derivative.

## 5. Three paths

### Path 1 — Kinetix, done right (deadline path)

Finish the paper on the current substrate with the improvements that change conclusions: feasibility-normalized detection, GPU-side five-seed search, held-out-class rotation, the evolved testers' real-bug scans, portfolios, the tuning-vs-structure ablation, and the RL baselines — all of which are queued or running. Deliverable: the AutoMLR submission with a defensible transfer claim and a real-bug table. Risk: low. Ceiling: a good workshop paper whose result is specific to one engine.

### Path 2 — A second engine (breadth)

Port the harness to a second JAX game family and test whether a tester program discovered on Kinetix transfers *across engines*, not just across bug classes. The hooks are engine-agnostic (they see a feature vector, not Kinetix); what changes is the feature layout, the planted-defect generator, and the oracles. Best candidates from the survey: **Craftax** first (MIT; `flax.struct.dataclass` state, so save/restore and `vmap` transfer unchanged; 67 achievements plus crafting, hunger and mob rules give a *symbolic* rule surface — planted defects and oracles: item produced without consuming ingredients, achievement unlocked without its precondition, HP/food/energy outside range, entity inside a solid tile, floor transition landing off-map); **Pgx** second (23 board games; the built-in legal-action mask is an oracle we do not have to invent, and chess/shogi have published perft counts — an *external* ground truth, rare and persuasive; oracles: perft divergence, illegal move accepted, king left in check, piece non-conservation, terminal-state disagreement); **PuzzleJAX** third (~900 human-authored PuzzleScript games with a JS reference engine, so bugs are *differential* — JAX vs. reference on the same action sequence — and its `broken_games.md` is pre-labelled ground truth; risks: source install, unverified state purity, slow per-game JIT). Deliverable: a two-engine transfer result and a second set of never-before-scanned real bugs. Risk: medium-high in one week (a new planted-bug generator and oracles are 2–3 days of work); the pilot is worth 1–2 days if the Path 1 queue is idle.

### Path 3 — Games people actually play (the headline you asked for)

Point the tester at real games through emulators or open-source engines with RAM- or state-based oracles — "the agent found N previously unknown bugs in game X" is the result that lands outside ML. Candidates from the survey: **stable-retro** [15] (~1,000 console games; `get_state/set_state`; `data.json` names RAM variables such as lives, HP and position, so oracles come free; TASVideos and wiki glitch lists give ground truth), **PyBoy** [16] (Game Boy; memory access and save/load state, headless ~395× realtime, 7 game wrappers), **ALE** [13] (97 games, `cloneState/restoreState`, 128-byte RAM), **Procgen** [14] (16 games, `get_state/set_state`), and open-source PC games that already ship determinism oracles — Wesnoth (replay validation by default), Endless Sky (`--test`, fixed seed), OpenTTD (desync logs, NoAI API), SuperTuxKart (rewind manager) — with their public bug trackers as the reference set. Not Godot (non-deterministic physics by its own docs), not PufferLib Ocean (no snapshot API), not Brax (PhyFu's ground). Risk: high before Sep 4 (no vectorized simulation, per-game oracle engineering, slow emulators); but it is the right follow-up paper and thesis direction, and the methodology (save/restore search + programmatic oracles + severity triage + replay verification) carries over intact.

### Recommendation

**Path 1 for the September 4 submission, with the Path 2 pilot as a stretch goal if the queue frees up, and Path 3 as the follow-up.** The transfer claim on Kinetix is already the strongest evidence we will have by Friday, and its weak points (feasibility ceiling, search noise, real-bug counts) are all fixable within the deadline; a second engine started now would land half-finished and weaken both halves. Two survey-driven additions to Path 1 cost little and matter to reviewers: release the planted-defect suite as the first *state-based* bug-finding benchmark (scored GBQA-style: recall at an interaction budget), and add a "stuck state" oracle (no branch from the archive reaches the goal) since no softlock benchmark exists. The GPU-fitness search (run 4) is the single change most likely to improve the numbers before Friday; it is already running.

## References

1. Matthews, Beukman, Lu, Foerster. *Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks*. ICLR 2025 (oral); arXiv:2410.23208; code github.com/FLAIROx/Kinetix. <https://arxiv.org/abs/2410.23208>
2. Lange, Imajuku, Cetin (Sakana AI). *ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution*. arXiv:2509.19349, 2025; code github.com/SakanaAI/ShinkaEvolve. <https://arxiv.org/abs/2509.19349>
3. Matthews et al.. *Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning*. ICML 2024; arXiv:2402.16801. <https://arxiv.org/abs/2402.16801>
4. Earle, Todd, Li, Jiang, Khalifa, Nasir, Banburski-Fahey, Togelius. *PuzzleJAX: A Benchmark for Reasoning and Learning*. arXiv:2508.16821, 2025; code github.com/smearle/script-doctor. <https://arxiv.org/abs/2508.16821>
5. Earle et al.. *ScriptDoctor: Automatic Generation of PuzzleScript Games via Large Language Models and Tree Search*. arXiv:2506.06524, 2025. <https://arxiv.org/abs/2506.06524>
6. Koyamada, Okano, Nishimori et al.. *Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning*. NeurIPS 2023; arXiv:2303.17503. <https://arxiv.org/abs/2303.17503>
7. Todd, Padula, Soemers et al.. *Ludax: A GPU-Accelerated Domain Specific Language for Board Games*. arXiv:2506.22609, 2025 (25 pages, 6 figures). <https://arxiv.org/abs/2506.22609>
8. Bonnet et al. (InstaDeep). *Jumanji: a Diverse Suite of Scalable Reinforcement Learning Environments in JAX*. ICLR 2024; arXiv:2306.09884. <https://arxiv.org/abs/2306.09884>
9. Nikulin et al.. *XLand-MiniGrid: Scalable Meta-Reinforcement Learning Environments in JAX*. arXiv:2312.12044; code github.com/corl-team/xland-minigrid. <https://arxiv.org/abs/2312.12044>
10. Pignatelli et al.. *NAVIX: Scaling MiniGrid Environments with JAX*. arXiv:2407.19396, 2024. <https://arxiv.org/abs/2407.19396>
11. Xiao, Liu, Wang. *PHYFU: Fuzzing Modern Physics Simulation Engines*. arXiv:2307.10818, 2023. <https://arxiv.org/abs/2307.10818>
12. PufferAI. *PufferLib / Puffer Ocean environments*. GitHub, accessed Aug 2026. <https://github.com/PufferAI/PufferLib>
13. Bellemare, Naddaf, Veness et al.. *The Arcade Learning Environment: An Evaluation Platform for General Agents*. JAIR 47, 2013; arXiv:1207.4708. <https://arxiv.org/abs/1207.4708>
14. Cobbe, Hesse, Hilton et al.. *Leveraging Procedural Generation to Benchmark Reinforcement Learning*. ICML 2020; arXiv:1912.01588. <https://arxiv.org/abs/1912.01588>
15. Farama Foundation. *stable-retro: Gym Retro fork with ~1000 emulated console games, state save/load and RAM-variable definitions*. GitHub, MIT license (accessed 29 Aug 2026). <https://github.com/Farama-Foundation/stable-retro>
16. Baekalfen et al.. *PyBoy: Game Boy emulator written in Python with memory access and save/load state*. GitHub, LGPL-3.0, v2.7.0 (accessed 29 Aug 2026). <https://github.com/Baekalfen/PyBoy>
17. Karten, Appapogu, Jin. *Automatic Generation of High-Performance RL Environments*. arXiv 2026; arXiv:2603.12145. <https://arxiv.org/abs/2603.12145>
18. Jiang, Chen, Chen. *GBQA: A Game Benchmark for Evaluating LLMs as Quality Assurance Engineers*. ICLR 2026 workshop; arXiv:2604.02648. <https://arxiv.org/abs/2604.02648>
19. Chen, Hong, Gao, Lin et al.. *GameXpert-Bench: How Far Are Coding Agents from Expert Game Development?*. arXiv:2608.21833, 2026. <https://arxiv.org/abs/2608.21833>
20. Taesiri, Ghildyal, Zadtootaghaj et al.. *VideoGameQA-Bench: Evaluating Vision-Language Models for Video Game Quality Assurance*. NeurIPS 2025 Datasets & Benchmarks; arXiv:2505.15952. <https://arxiv.org/abs/2505.15952>
21. Zheng, Zhou, Wu et al.. *Open-Ended Video Game Glitch Detection with Agentic Reasoning and Temporal Grounding*. arXiv:2604.07818, 2026. <https://arxiv.org/abs/2604.07818>
22. Yu, Wiens, Barahona-Ríos et al.. *TempGlitch: Evaluating Vision-Language Models for Temporal Glitch Detection in Gameplay Videos*. arXiv:2605.21443, 2026. <https://arxiv.org/abs/2605.21443>
23. Cao, Tang, Zhao et al.. *PhysGame: Uncovering Physical Commonsense Violations in Gameplay Videos*. arXiv:2412.01800, 2024. <https://arxiv.org/abs/2412.01800>
24. Cao, Tang, Zhao et al.. *Order from Chaos: Physical World Understanding from Glitchy Gameplay Videos*. arXiv:2601.16471, 2026. <https://arxiv.org/abs/2601.16471>
25. Taesiri, Feng, Nguyen, Bezemer. *GlitchBench: Can large multimodal models detect video game glitches?*. CVPR 2024; arXiv:2312.05291. <https://arxiv.org/abs/2312.05291>
26. Taesiri, Macklon, Bezemer. *CLIP meets GamePhysics: Towards bug identification in gameplay videos using zero-shot transfer learning*. MSR 2022; arXiv:2203.11096. <https://arxiv.org/abs/2203.11096>
27. Taesiri, Macklon, Wang et al.. *Large Language Models are Pretty Good Zero-Shot Video Game Bug Detectors*. arXiv 2022; arXiv:2210.02506. <https://arxiv.org/abs/2210.02506>
28. Yapağcı, Öztürk, Tüzün. *Agents in the Sandbox: End-to-End Crash Bug Reproduction for Minecraft*. arXiv:2503.20036, 2025 (Accepted into ASE'25 (Automated Software Engineering). https://conf.re). <https://arxiv.org/abs/2503.20036>
29. Wilkins, Stathis. *World of Bugs: A Platform for Automated Bug Detection in 3D Video Games*. IEEE CoG 2022; arXiv:2206.11037. <https://arxiv.org/abs/2206.11037>
30. Feldmeier, Fraser. *Neuroevolution-Based Generation of Tests and Oracles for Games*. arXiv:2208.13632, 2022. <https://arxiv.org/abs/2208.13632>
31. Chen, Chen, Jia et al.. *IcFuzz: Fuzzing Isaac Sim with Semantic Stage Guidance and Multi-level Mutation*. arXiv 2026; arXiv:2608.06088. <https://arxiv.org/abs/2608.06088>
32. Lu, Senchenko, Sayle et al.. *How Far Can VLMs Go for Visual Bug Detection? Studying 19,738 Keyframes from 41 Hours of Gameplay Videos*. arXiv:2603.22706, 2026. <https://arxiv.org/abs/2603.22706>
33. Mawhorter, Smith. *Softlock Detection for Super Metroid with Computation Tree Logic*. FDG 2021, doi:10.1145/3472538.3472542. <https://doi.org/10.1145/3472538.3472542>
34. Cooper, Bazzaz. *Stuck in the Middle: Generating Levels without (or with) Softlocks*. FDG 2025, doi:10.1145/3723498.3723844. <https://doi.org/10.1145/3723498.3723844>
35. Schumilo, Aschermann, Jemmett et al.. *Nyx-Net: Network Fuzzing with Incremental Snapshots*. EuroSys 2022; arXiv:2111.03013. <https://arxiv.org/abs/2111.03013>
36. TASVideos. *TASVideos publications API (4,615 published tool-assisted movies with glitch tags)*. REST API v1 (accessed 29 Aug 2026). <https://tasvideos.org/api>
37. Forward, Limmer, Hallett et al.. *Super Mario in the Pernicious Kingdoms: Classifying glitches in old games*. arXiv:2404.14870, 2024. <https://arxiv.org/abs/2404.14870>
