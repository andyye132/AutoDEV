# Where AutoDev Stands

A positioning report: what we have built and measured so far, the lay of the land in automated game testing and bug finding, how we compare on the numbers people actually report, and how the problem and scope should be framed before we commit.

Prepared 29 Aug 2026 (Saturday afternoon), six days before the AutoMLR deadline (Fri 4 Sep, 5 PM PDT). Numbers are from Hyak runs at the size-l configuration (74 shipped Kinetix levels, 337 planted instances, 5 seeds, 262,144 simulator steps per instance) unless marked *local*. Every citation was resolved through the arXiv / Semantic Scholar / OpenAlex / Crossref APIs today or in the two earlier surveys.

## 1. What we have now

**The object of study.** Not a tester, but a *loop that discovers testers*: an LLM-driven program search (ShinkaEvolve [1]) writes the exploration procedure — the `act` / `restore` / `archive` hooks that drive B parallel workers over an archive of save/restore-able simulator states — and is scored by how fast the procedure reaches *witness states* of planted defects. Kinetix [2] is the substrate: 2D physics games with a JSON level format, a fully restorable state, and thousands of levels per GPU in parallel.

**What is measured.** Three things, kept separate on purpose:

| Measurement | What it is | Where we are (single run of the run-1 best, 5 seeds) |
|---|---|---|
| Planted-defect detection, *search split* | fraction of 127 planted instances (5 classes: missing wall, non-solid obstacle, unintended goal, runaway physics, disarmed hazard) whose witness state is reached within budget | **72%** (score 0.666) vs 59% action-repeat random, 57% Go-Explore seed, 56% best RL tester |
| Held-out *levels* (22 of 74 levels never seen by the search) | same, 59 instances | **64%** (0.617) vs 60% (0.575); paired per-instance gain +0.04 ± 0.05 (CI includes zero) |
| Held-out *defect class* (arena escape; the search never scored it) | 148 instances | **47%** (0.418) vs 30% (0.266) action-repeat random, 25% seed; paired gain +0.15 ± 0.05 |
| Reachability (union of every tester, seed, and an oracle-guided probe) | how much of the pool anything reaches at this budget | 81% / 66% / 62%; the guided probe adds nothing beyond the testers |
| Feasibility-normalised detection | detection over the reachable set | evolved **88% / 96% / 76%** vs action-repeat random 73% / 90% / 48% |
| Real anomalies on unmodified levels (replay-verified, severity A/B) | generic oracles: leaves arena, inside a solid, joint tear, runaway/NaN; persistence rule; re-simulated from reset | shipped levels: **2 B-level pairs** (one level, h15: a block at 16.5 m/s momentarily inside the right wall and 10 cm into the floor, ejected by the solver); generated levels: **10 pairs** (1 sustained escape, 4 wall-face crossings, 3 deep penetrations, 2 runaways) |
| Design flaws | no-input / random-play outcomes | h5_rotate_fall is won by random play (62%, median 26 steps); no-input death is the intended challenge on 9 levels |

**What the evidence supports today.** (1) An LLM search finds a tester program that reaches planted defects faster than every hand-designed, LLM-designed (four Opus designs, three round-2 designs pending, four round-3 designs pending) and RL tester we tried, at equal budget, and the four candidates the search produced agree within 0.02 on every split. (2) The advantage transfers to a defect class the search never scored on — a large, instance-significant gain — and to unseen levels only modestly. (3) On unmodified levels the engine is mostly sound; the anomalies we can verify are momentary tunnelling of fast bodies into thin static geometry, plus one sustained escape on a generated level.

**What it does not support (yet).** A claim of "found N serious bugs in shipped games" (the shipped-level finding is one tunnelling failure, severity B); a claim that the search keeps improving (every run makes its big gain in the first generations and then moves within the noise band — run 4 with a cleaner GPU fitness gains 0.610 → 0.625 over 24 candidates); a claim about any engine other than Kinetix.

**Corrections we made to our own claims today**, because they affect how honest the framing must be: PhyFu [3] already fuzzes Brax with a penetration oracle (novelty narrowed); the "ceiling escape" was a momentary wall-face crossing (severity downgraded, rule tightened); 3 of 7 TUNNEL instances on the local pool are geometrically unreachable (pool hygiene needed); 15 scan hits did not reproduce in the CPU replay and were dropped.

## 2. The lay of the land

Three reading agents (one Opus, two Sonnet; ~270 tool calls; every ID re-resolved through the arXiv, Semantic Scholar, OpenAlex or Crossref APIs today) mapped the field. WebSearch and Google Scholar were not reachable from this session, so citation counts are Semantic Scholar / Crossref (they undercount Google Scholar). "Not reported" means we could not confirm a number, not that it is zero.

### 2.1 The five formulations of "AI finds bugs in games"

| # | Formulation | Who | What is optimised | Oracle | Typical metric | Best reported numbers |
|---|---|---|---|---|---|---|
| 1 | **RL / search testers with a hand-designed exploration objective** | EA SEED (Bergdahl 2020 [4], Gordillo 2021 [5], CCPT 2022/24 [6], RAID 2026 [7]); NetEase Wuji (ASE 2019) [8]; Microsoft Inspector [9]; Ubisoft CA2 (2026) [10]; Ariyurek et al. (GVGAI, injected bugs) [11] | curiosity, coverage, novelty, win-vs-explore, code coverage | crash / stuck / injected bug / designer review | coverage; bugs found in one game | Ariyurek et al.: **100% of 45 injected bugs** (human-like MCTS, aggregated over 5 runs) vs 90% for humans; NEATEST 65.6% mutation score; CCPT 13/13 bugs vs 10/3 for curiosity-only; Go-Explore reachability 100% of nav-mesh goals; Wuji 3 / Inspector 2 / RAID 6 real bugs or exploits; Titan 82% vs Wuji 45.5% on 9 seeded bugs. Three published negative results: random beats RND asymptotically (Go-Explore reachability), more search made baseline MCTS *worse* at finding bugs (44% → 40%), and more coverage came with fewer highlighted bugs (CCPT). |
| 2 | **Bug / glitch detection from pixels or video** | GlitchBench [12], VideoGameQA-Bench [13], VideoGlitchBench [14], TempGlitch [15], PhysGame/GameBench [16, 17], EA's 19,738-keyframe field study [18], GLIB [19] | a classifier / VLM prompt | human-labelled glitch clips (real) or code-injected (TempGlitch) | accuracy, F1, recall | GPT-4V 43.4% (GlitchBench); GPT-4.1 81% on simple image glitches down to 19% on needle-in-haystack (VideoGameQA); 39% F1 with agentic grounding (VideoGlitchBench); **54% = chance** on temporal glitches (TempGlitch); precision 0.50 on real EA footage; the outlier is GLIB's learned oracle: 99.5% recall on 20 known-bug games, 48/53 confirmed in the field (NetEase, 2021) |
| 3 | **Design evaluation by simulated players** | procedural personas (Holmgård et al.) [20, 21], Monster Carlo (MCTS in Unity) [22], Cicero [23], A*-agent playability in PCG (Volz et al.) [24], difficulty/churn prediction at King (Gudmundsson [25], Roohi [26, 27], Kristensen [28]), AlphaZero rule-variant balance [29], softlock verification (Mawhorter & Smith [30], Cooper & Bazzaz [31]), PuzzleJAX/ScriptDoctor solvability [32, 33], LLM difficulty proxies [34] | play the level as a proxy player | solvability, difficulty, balance, softlock (logical spec) | correlation with human data; verified property | best-5%-of-agent-runs move counts predict real per-level completion at King; CTL model checking proves softlock freedom on Metroid abstractions; RAID found 6 goalie exploits in NHL 26 in one run |
| 4 | **LLM QA agents and benchmarks (closed loop, state-based)** | GBQA (ICLR'26 wksp) [35], GameXpert-Bench [36], BugCraft (ASE'25) [37], Titan (MMORPG QA) [38], SMART (LLM-guided RL, code coverage) [39] | an LLM agent's actions / test objective | injected + human-verified bugs; real crash reports; code coverage | **recall of verified bugs at a step budget**; reproduction rate; coverage | **48.4%** of 124 verified bugs (Claude 4.6 Opus, 500 steps); 34.9% end-to-end crash reproduction vs 83% for humans; 39/100 Strict repair; SMART >94% branch coverage of modified code (~2× plain RL); Titan claims 95% task completion and 4 new bugs (self-reported) |
| 5 | **Searching over the tester / objective itself** | meta-learned curiosity programs (Alet et al., ICLR 2020) [40]; LLM-evolved RL algorithms (REvolve) [41]; DeepMind's LLM-evolved multi-agent algorithms [42]; NEATEST co-evolves tests *and* oracles for Scratch [43] | the exploration algorithm / reward / test program | (none of these test games for bugs except NEATEST: mutation score) | held-out task transfer; mutation score | Alet: discovered curiosity on par with human designs and transfers across control tasks; DeepMind: **evolved code overfits its training games**, only a distilled core generalises; NEATEST: >65% mutation score on 25 Scratch games. **No published work searches over a game-testing exploration program with bug detection as fitness.** |

Two adjacent lines matter for framing. **Physics-engine fuzzing** (PhyFu [3], IcFuzz [44]) treats the *engine* as numerical software with physics-law oracles — PhyFu already fuzzes Brax and uses a penetration example as its motivation — but has no levels, rules, win conditions, or exploration policy. **Practitioner surveys** [45, 46, 47] say the same thing from the other side: studios test manually, adopt autonomous testers reluctantly, and name the *oracle* (what counts as a bug) and the *goal* (what the agent should try to do) as the blockers — not the agent.

### 2.2 The RL/search playtesting line in detail

The line closest to ours. Coverage is its default metric because bug oracles are expensive; only a handful of papers use a countable denominator (injected faults), and only one has a human upper bound.

| Paper (venue) | Substrate | Exploration objective | Oracle / bug definition | Headline number | Transfer measured? |
|---|---|---|---|---|---|
| Bergdahl et al., EA SEED (CoG 2020) [4] | proprietary FPS sandboxes, state | hand-authored reward per sandbox | stuck heat-maps; designer inspection | steps to 80% of max reward (88M–342M) | no |
| Gordillo et al., EA SEED (CoG 2021) [5] | proprietary 3D map, state | count-based novelty over 3D cells | coverage map, timeouts | ~24 h to 90% map coverage | no |
| CCPT, EA SEED (ToG 2024) [6] | AAA-scale 3D nav env + VizDoom, state | RND curiosity ⊕ imitation of expert demos | inserted + real oversights; trajectories auto-highlighted | **13/13 bugs found & highlighted vs curiosity-only 10 found / 3 highlighted** — with *less* coverage (1.91M vs 2.39M states) | no |
| RAID, EA (RLC 2026 wksp) [7] | NHL 26 dev build | goal reward re-shaped with a diversity penalty per round | goal rate vs goalie AI = exploit | 6 distinct exploits in one run, matching hours of human playtesting | no |
| Wuji, NetEase (ASE 2019; 193 cites) [8] | 2 commercial combat games, state | multi-objective EA over a policy population: win-rate × coverage | 4 oracles from 1,349 real bugs | 3 previously unknown bugs, developer-confirmed | no |
| Inspector, Microsoft (CoG 2022) [9] | UE4 sample games, **pixels** | RND on screenshots + object detector + imitation | human inspection | 2 real bugs; "super-human" coverage (figures only) | motivated, not measured |
| Preference-conditioned agent, Xbox (CoG 2023) [48] | proprietary open world, pixels | style (imitation) ⊕ novelty reward | objects-of-interest tested | 80% pathway coverage vs 16% (Inspector); 192/247 objects vs 19 | no |
| Go-Explore reachability, Microsoft (ToG 2022) [49] | proprietary 3D maps up to 1.5 km², state + nav-mesh | Go-Explore archive, random actions, nav-mesh resets, **no learned policy** | points >3 m off nav-mesh = bug | 100% of nav-mesh goals < 200k steps; 2 planted bugs found; **random exploration asymptotically better than RND** | 3 maps, one game |
| CA2, Ubisoft La Forge (2026) [10] | Crafter + a Godot game, call-stack signal | reach a target function (code coverage) | function reached | 56% / 47% success vs weaker baselines | no |
| Tufano et al. (ICSE 2022) [50] | CartPole, MsPacman, one 3D game | game reward **+50 for entering the injected FPS-drop region** | injected performance bugs (2 + 4) | 965/1000 episodes hit a bug vs 12 random — but the oracle is in the reward | no |
| Ariyurek et al. (ToG 2021; CoG 2020) [11, 51] | GVGAI/VGDL, 3 games × 12 levels, state | Sarsa(λ) / MCTS driven by test goals; human-like goals via IRL from 427 tester traces | **45 injected bugs** with a model-based oracle | agents reach **100% of the 45 bugs aggregated over 5 runs**, humans 90%; baseline MCTS ≤44% and **worse with more compute** | 3 games; goals re-derived per game |
| NEATEST (ASE 2022) [43] | 25 Scratch games | neuroevolved networks as executable tests + oracles | 243,835 mutants | mutation score 65.6%, false positives 19.7%; coverage 94.6% vs 89.1% random | robustness to program randomisation (+1.2% vs −8.2% static) |
| iv4XR line (A-TEST 2021–25, ASE 2022) [52, 53, 54, 55] | Lab Recruits, Space Engineers | tactical agents; curiosity; cooperative multi-agent | coverage criteria | 2 agents ≈35% faster than 1 | no |
| GLIB, NetEase (ESEC/FSE 2021) [19] | 34 real games, pixels | *no exploration* — a learned visual-glitch oracle | injected-glitch augmentation | 99.5% recall on 20 known-bug games; **48/53 field bugs confirmed** | across games |
| Holmgård et al. (ToG 2019) [20]; Monster Carlo (CIG 2018/19) [22] | MiniDungeons; Unity games | evolved MCTS heuristics; MCTS + behaviour cloning | designer inspection | personas enact play styles; MC2 transfers experience across playthroughs | across levels |
| Reveal-More (CoG 2019) [56] | two SNES games | automated exploration seeded from human traces | coverage | "multiplicatively amplifies" human coverage | no |
| Titan (2025) [38] | 2 commercial MMORPGs, abstracted state | LLM agent with memory/reflection | 9 seeded bugs + LLM oracle | **82% vs Wuji 45.5% vs ReAct 45.5%** on the seeded bugs; 4 new bugs (self-reported) | claimed, not controlled |
| Meta-learned curiosity (ICLR 2020) [40] | control tasks, not games | **program search over the curiosity algorithm** | RL return | on par with human-designed curiosity; transfers across tasks | yes — but no bug oracle |

Two structural facts. Coverage, the field's default, is **falsified as a proxy in-paper**: CCPT's curiosity-only baseline explores more states and surfaces fewer bugs. And the injected-fault formulation, the only one with a comparable denominator, is used with a stated N by five papers (Ariyurek 45, Tufano 6, NEATEST 243,835 mutants, GBQA 124, GameXpert 19–27 per level) — ours has 337.

### 2.3 How prior work handles the two things reviewers will ask about

**Oracles.** Eight oracle types recur (crash; coverage-as-proxy; human-labelled; injected/mutated; formal spec; physics invariant; learned visual anomaly; agent play-rate as difficulty). Injected/mutated bugs — our choice — are the fastest-growing category (GBQA, GameXpert, World of Bugs [57], NEATEST, Ariyurek), all but two post-2021. Two named categories came back **empty** on direct search: metamorphic testing of games and differential build-vs-build / replay-desync testing have no dedicated papers (industry does "golden replay" comparisons but has not published them) — replay verification, which we use for every counted bug, is therefore citable as a gap rather than as prior art.

**Are the injected bugs reachable?** Prior work validates injected bugs three ways: human verification at construction (GBQA, GameXpert), a human-tester baseline on the *same* bugs (Ariyurek et al. — synthetic and human-like agents matched human testers' detection rate on 45 injected bugs), and mutation-score plus false-positive reporting (NEATEST). **The check almost nobody runs is a random/brute-force floor on the same budget** — which is exactly the check that exposed our own pool problem today (3 of 7 local TUNNEL instances are geometrically unreachable). We run that floor (random, action-repeat random, the guided probe) and should say so.

**Transfer.** Explicit transfer numbers are rare and, where present, are the headline: Ariyurek (agents vs humans on the same bugs), GBQA/GameXpert (difficulty tiers; self-discovery vs given-bug-list collapses by up to −33 points for weaker models), Holmgård personas re-applied dungeon → Match-3, NEATEST across 25 games, GameBench (glitch-trained VLMs transfer +2.5% to real-world physics benchmarks). **No paper reports a tester's transfer to an unseen bug *class*.**

## 3. How we are doing against the field

The honest way to compare is by formulation, because the numbers are not on the same scale.

**Injected-fault detection (F2) — the formulation we share with Ariyurek, NEATEST, Tufano, GBQA, GameXpert.**

| | Ariyurek et al. 2021 | NEATEST 2022 | GBQA 2026 | **AutoDev (now)** |
|---|---|---|---|---|
| Substrate | GVGAI/VGDL, 3 games × 12 levels | 25 Scratch games | 30 LLM-built text games | 74 Kinetix physics levels + 64 generated |
| Faults | 45 hand-injected (taxonomy-diversified) | 243,835 mutants (8 operators) | 124 injected, human-verified | 337 planted, 5 classes, each with a witness state |
| Detectability check | human testers on the same bugs (90–100%) | mutation score + false-positive rate | human verification at construction | random / action-repeat / guided-probe floors; union reachability 81 / 66 / 62%; 3 of 7 local TUNNEL instances found unreachable |
| Best detection | **100%** (aggregated over 5 MCTS runs); single agents 40–44% | 65.6% mutation score | 48.4% (Claude 4.6 Opus, 500 steps) | 72% single run (88% of reachable); union of everything 81% |
| Baselines | humans; baseline game-playing agents | random tester (coverage) | 20 LLMs vs each other, no RL | random, action-repeat, Go-Explore, 4 PPO objectives, 4 + 3 + 4 LLM-designed testers |
| Who designed the tester | hand-designed + IRL from 427 human traces | neuroevolution per game | prompted LLM | **LLM search over the program**, then frozen |
| Transfer | goals re-derived per game; not measured for bugs | robustness to randomisation | zero-shot across its 30 games (no trained policy) | **held-out levels +0.04 ± 0.05; held-out class +0.15 ± 0.05** |

Read straight, our single-run 72% is *below* Ariyurek's aggregated 100% and *above* GBQA's 48% — but neither comparison means much: Ariyurek's bugs were validated by humans as findable (ours include unreachable ones), their 100% is a 5-run union (our 5-seed union is 80%), and GBQA's agent has no simulator budget concept at all. The apples-to-apples statement is: **under an injected-fault protocol with a countable denominator and a random floor, our discovered tester is the strongest tester we can construct or borrow, and it is the only one whose evaluation includes a sealed defect class.**

**Coverage-maximising exploration (F1) — where a CoG/FDG reviewer will place us by default.** Our hand-designed baselines are this line's objectives (count-based, RND, ICM, curiosity, Go-Explore), evaluated under our protocol: all of them lose to the discovered program, and the RL versions lose to *action-repeat random* — consistent with Microsoft's finding that random exploration beats RND asymptotically, and with CCPT's finding that coverage and bug-surfacing diverge. We do not report coverage as a metric. We should (as a secondary number), because it is their language and because our result *predicts* the CCPT inversion: the search improved bug-witnessing, not coverage.

**Real-bug discovery (F3) — where we are weakest.** Wuji found 3 developer-confirmed bugs in commercial games; GLIB confirmed 48 of 53 across 14 real games; RAID found 6 exploits in a shipping title; Inspector 2. We have one momentary tunnelling failure in a shipped research level and ten anomalies in generated levels. On this axis we are not competitive and should not pretend to be: Kinetix is a research substrate whose engine has been exercised by RL for a year, and our generic oracles are physics invariants, not a studio's bug taxonomy.

**The route question, answered directly.** Are we a worse version of something that exists? *As a tester*, yes: EA's, NetEase's and Microsoft's agents run on real games with real oracles and we would lose on every axis a studio cares about. *As a research loop that discovers testers and measures whether they transfer*, no — the survey found no paper that (i) searches over exploration-objective code, (ii) with detection of injected faults as fitness, (iii) reporting results on held-out games or held-out bug classes. Alet et al. own (i) without a game or an oracle; Ariyurek, NEATEST and GBQA own (ii) without (i) or (iii); nobody owns (iii). The three published negative results (random beats RND; more search hurts; coverage anti-correlates with bugs) are the reason (i)+(iii) is interesting: hand-designed exploration objectives are unreliable testers, and the field has no method for finding better ones.

**What the field will ask that we do not yet have.** (1) A CCPT-style baseline — curiosity blended with imitation of expert trajectories — evaluated under our protocol (we have no expert demonstrations; Kinetix's shipped solutions could stand in). (2) Coverage as a secondary metric. (3) A human anchor (Ariyurek's template): even a small human-tester sample on a few planted instances would be the strongest credibility signal. (4) A second game family, because every paper above is single-game and reviewers now read that as the norm rather than a flaw — which is exactly why our held-out-class result is worth more than it looks.

## 4. Framing: what problem are we actually solving?

There are four distinct problems that the phrase "AI finds bugs in games" hides, and prior work sits in different ones:

1. **Tester-agent performance** — given a game and a hand-designed exploration objective (curiosity, coverage, novelty), how much of the game does the agent cover and how many bugs (usually crashes, stuck states, or injected faults) does it trigger? This is the industrial RL line (EA, Ubisoft, NetEase) and the GVGAI/MCTS line. The metric is coverage or bugs-found on *one* game; generalization is rarely measured because the agent is retrained per game.
2. **Bug detection from observations** — given gameplay (pixels/video), decide whether a frame or clip shows a glitch. This is the VLM benchmark line; numbers are accuracy/recall on curated datasets, and the one field study says they do not transfer to real footage.
3. **Design evaluation** — is the level solvable, too easy, soft-lockable, exploitable? Procedural personas, MCTS playtesting, softlock model checking. The oracle is a design property, not an engine fault.
4. **Autonomous QA loops** — an agent that plays, detects, verifies and reports without a human in the loop, scored by recall of human-verified bugs at an interaction budget (GBQA) or by fix success (GameXpert). New (2026), state-based, artifacts unreleased.

We are not in (2) at all, only marginally in (3), and our paper's actual contribution sits *between* (1) and (4): **the exploration strategy is not hand-designed but discovered by a research loop, and the thing we measure is whether the discovered strategy transfers** — to levels it never saw and to a defect class it was never scored on. Nobody in line (1) searches over the tester itself (the closest is meta-learned curiosity on control tasks [40]), and nobody in lines (1) or (4) reports a tester's transfer to unseen bug types. That is the framing that is both true and not already occupied — and it is *made* interesting by the field's own negative results: hand-designed exploration objectives are unreliable testers (random beats RND [49]; more search hurts [51]; coverage anti-correlates with highlighted bugs [6]).

**Where we would be "a worse version of the same route".** If the paper were framed as *"an RL/search agent that finds bugs in a game"*, it would be compared with Bergdahl/Gordillo/CCPT/Wuji on their terms — coverage of commercial 3D games, crashes found in production — and lose on every axis that matters to a game studio (their games are real, ours are 2D physics toys; their bugs are production bugs, ours are planted; their agents run in Unity/proprietary engines, ours needs a save/restore-able JAX state). Likewise, framed as *"finding real bugs in Kinetix"*, the finding is one momentary tunnelling failure, which no reviewer will find impressive next to Wuji's production bugs or GBQA's 124 verified web-game bugs.

**Where we are not.** Framed as *"can an autonomous loop discover a testing strategy that generalizes?"*, the relevant comparison is the hand-designed objectives of line (1) evaluated under our protocol (equal budget, sealed held-out class), and there we win by a wide margin on the held-out class, modestly on held-out levels, and honestly report the classes where nothing helps (TUNNEL) and why (pool infeasibility). The planted-defect benchmark with witness oracles, state-based and replay-verified, is itself a contribution line (4) lacks (GBQA and GameXpert released nothing), and it should be released.

**Scope statement that survives review** (proposed): *We study whether an LLM-driven research loop can discover game-testing exploration programs that transfer, using Kinetix as a fully observable, restorable, GPU-parallel test bed with planted defects and witness oracles; we do not claim to test commercial games, and the real anomalies we report are a by-product used to validate the oracles, not the headline.* Under that scope the AutoMLR fit is exact: the agent designed the benchmark, the search, the baselines and the analysis; the human verified.

## 5. Recommendation

- **Keep Path 1 with the framing above.** Retitle toward "discovered testing strategies transfer" rather than "finds real bugs"; move the real-bug table to a validation role; add the benchmark release (planted-defect suite + witness oracles + replay verification) as a stated contribution, scored GBQA-style (recall at an interaction budget).
- **Add the comparisons the field expects:** a CCPT-style curiosity ⊕ imitation tester (using Kinetix's shipped solution trajectories as the "expert demos"), coverage as a secondary metric next to detection, and — if at all possible before Friday — a small human-tester anchor on a handful of planted instances, which is the Ariyurek template and the single most credible number a reviewer from CoG/FDG will look for. The MCTS baseline that community expects is covered by macrosweep (Opus's macro tree search).
- **Do not chase real-bug counts before Friday.** The shipped-level engine is sound; counting momentary penetrations as "bugs" would be the kind of overclaim that the field is tired of (see the VLM benchmarks' industrial reality check). Report what the oracles verified, with severities, and say the engine held up.
- **Second engine after the deadline**, chosen to move from physics to *rules* (Craftax), where the "unseen defect class" story becomes "unseen rule class" — that is the generality claim the workshop paper can only promise.

## References

1. Lange, Imajuku, Cetin (Sakana AI). *ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution*. arXiv:2509.19349, 2025; code github.com/SakanaAI/ShinkaEvolve. <https://arxiv.org/abs/2509.19349>
2. Matthews, Beukman, Lu, Foerster. *Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks*. ICLR 2025 (oral); arXiv:2410.23208; code github.com/FLAIROx/Kinetix. <https://arxiv.org/abs/2410.23208>
3. Xiao, Liu, Wang. *PHYFU: Fuzzing Modern Physics Simulation Engines*. arXiv:2307.10818, 2023. <https://arxiv.org/abs/2307.10818>
4. Bergdahl, Gordillo, Tollmar, Gisslén (EA SEED). *Augmenting Automated Game Testing with Deep Reinforcement Learning*. IEEE CoG 2020; arXiv:2103.15819. <https://arxiv.org/abs/2103.15819>
5. Gordillo, Bergdahl, Tollmar, Gisslén (EA SEED). *Improving Playtesting Coverage via Curiosity Driven Reinforcement Learning Agents*. IEEE CoG 2021; arXiv:2103.13798. <https://arxiv.org/abs/2103.13798>
6. Sestini, Gisslén, Bergdahl, Tollmar, Bagdanov (EA SEED). *Automated Gameplay Testing and Validation with Curiosity-Conditioned Proximal Trajectories (CCPT)*. arXiv:2202.10057, 2022. <https://arxiv.org/abs/2202.10057>
7. Fuchs, Gosselin-Grant, Skuin et al. (EA). *Reward-Adaptive Iterative Discovery: A Case Study on Automated Game Testing for NHL 26*. arXiv:2607.07498, 2026. <https://arxiv.org/abs/2607.07498>
8. Zheng, Xie, Su et al. (NetEase Fuxi). *Wuji: Automatic Online Combat Game Testing Using Evolutionary Deep Reinforcement Learning*. ASE 2019. <https://doi.org/10.1109/ASE.2019.00077>
9. Liu et al. (Microsoft). *Inspector: Pixel-Based Automated Game Testing via Exploration, Detection, and Investigation*. IEEE CoG 2022; arXiv:2207.08379. <https://arxiv.org/abs/2207.08379>
10. Adaikkappan, Martineau, Romoff et al.. *CA2: Code-Aware Agent for Automated Game Testing*. arXiv:2605.13918, 2026. <https://arxiv.org/abs/2605.13918>
11. Ariyurek, Betin-Can, Surer. *Automated Video Game Testing Using Synthetic and Humanlike Agents*. IEEE Transactions on Games, 2021. <https://doi.org/10.1109/TG.2019.2947597>
12. Taesiri, Feng, Nguyen, Bezemer. *GlitchBench: Can large multimodal models detect video game glitches?*. CVPR 2024; arXiv:2312.05291. <https://arxiv.org/abs/2312.05291>
13. Taesiri, Ghildyal, Zadtootaghaj et al.. *VideoGameQA-Bench: Evaluating Vision-Language Models for Video Game Quality Assurance*. NeurIPS 2025 Datasets & Benchmarks; arXiv:2505.15952. <https://arxiv.org/abs/2505.15952>
14. Zheng, Zhou, Wu et al.. *Open-Ended Video Game Glitch Detection with Agentic Reasoning and Temporal Grounding*. arXiv:2604.07818, 2026. <https://arxiv.org/abs/2604.07818>
15. Yu, Wiens, Barahona-Ríos et al.. *TempGlitch: Evaluating Vision-Language Models for Temporal Glitch Detection in Gameplay Videos*. arXiv:2605.21443, 2026. <https://arxiv.org/abs/2605.21443>
16. Cao, Tang, Zhao et al.. *PhysGame: Uncovering Physical Commonsense Violations in Gameplay Videos*. arXiv:2412.01800, 2024. <https://arxiv.org/abs/2412.01800>
17. Cao, Tang, Zhao et al.. *Order from Chaos: Physical World Understanding from Glitchy Gameplay Videos*. arXiv:2601.16471, 2026. <https://arxiv.org/abs/2601.16471>
18. Lu, Senchenko, Sayle et al.. *How Far Can VLMs Go for Visual Bug Detection? Studying 19,738 Keyframes from 41 Hours of Gameplay Videos*. arXiv:2603.22706, 2026. <https://arxiv.org/abs/2603.22706>
19. Chen, Li, Chen et al.. *GLIB: Towards Automated Test Oracle for Graphically-Rich Applications*. ESEC/FSE 2021; arXiv:2106.10507. <https://arxiv.org/abs/2106.10507>
20. Holmgård, Green, Liapis et al.. *Automated Playtesting with Procedural Personas through MCTS with Evolved Heuristics*. IEEE ToG 2019; arXiv:1802.06881. <https://arxiv.org/abs/1802.06881>
21. Mugrai, Silva, Holmgård et al.. *Automated Playtesting of Matching Tile Games*. IEEE CoG 2019; arXiv:1907.06570. <https://arxiv.org/abs/1907.06570>
22. Keehl, Smith. *Monster Carlo: An MCTS-based Framework for Machine Playtesting Unity Games*. IEEE CIG 2018, doi:10.1109/CIG.2018.8490363. <https://doi.org/10.1109/CIG.2018.8490363>
23. Machado, Gopstein, Nealen et al.. *AI-Assisted Game Debugging with Cicero*. IEEE CEC 2018, doi:10.1109/CEC.2018.8477829. <https://doi.org/10.1109/CEC.2018.8477829>
24. Volz, Schrum, Liu et al.. *Evolving Mario Levels in the Latent Space of a Deep Convolutional Generative Adversarial Network*. GECCO 2018; arXiv:1805.00728. <https://arxiv.org/abs/1805.00728>
25. Gudmundsson, Eisen, Poromaa et al. (King). *Human-Like Playtesting with Deep Learning*. IEEE CIG 2018. <https://doi.org/10.1109/CIG.2018.8490442>
26. Roohi, Relas, Takatalo, Heiskanen, Hämäläinen. *Predicting Game Difficulty and Churn Without Players*. CHI PLAY 2020; arXiv:2008.12937. <https://arxiv.org/abs/2008.12937>
27. Roohi, Guckelsberger, Relas, Heiskanen, Takatalo, Hämäläinen. *Predicting Game Engagement and Difficulty Using AI Players*. CHI PLAY 2021; arXiv:2107.12061. <https://arxiv.org/abs/2107.12061>
28. Kristensen, Valdivia, Burelli (King). *Estimating Player Completion Rate in Mobile Puzzle Games Using Reinforcement Learning*. IEEE CoG 2020; arXiv:2306.14626. <https://arxiv.org/abs/2306.14626>
29. Tomašev, Paquet, Hassabis, Kramnik (DeepMind). *Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess*. arXiv:2009.04374, 2020. <https://arxiv.org/abs/2009.04374>
30. Mawhorter, Smith. *Softlock Detection for Super Metroid with Computation Tree Logic*. FDG 2021, doi:10.1145/3472538.3472542. <https://doi.org/10.1145/3472538.3472542>
31. Cooper, Bazzaz. *Stuck in the Middle: Generating Levels without (or with) Softlocks*. FDG 2025, doi:10.1145/3723498.3723844. <https://doi.org/10.1145/3723498.3723844>
32. Earle, Todd, Li, Jiang, Khalifa, Nasir, Banburski-Fahey, Togelius. *PuzzleJAX: A Benchmark for Reasoning and Learning*. arXiv:2508.16821, 2025; code github.com/smearle/script-doctor. <https://arxiv.org/abs/2508.16821>
33. Earle et al.. *ScriptDoctor: Automatic Generation of PuzzleScript Games via Large Language Models and Tree Search*. arXiv:2506.06524, 2025. <https://arxiv.org/abs/2506.06524>
34. Xiao, Yang. *LLMs May Not Be Human-Level Players, But They Can Be Testers: Measuring Game Difficulty with LLM Agents*. Proc. ACM HCI (CHI PLAY) 2025. <https://doi.org/10.1145/3748634>
35. Jiang, Chen, Chen. *GBQA: A Game Benchmark for Evaluating LLMs as Quality Assurance Engineers*. ICLR 2026 workshop; arXiv:2604.02648. <https://arxiv.org/abs/2604.02648>
36. Chen, Hong, Gao, Lin et al.. *GameXpert-Bench: How Far Are Coding Agents from Expert Game Development?*. arXiv:2608.21833, 2026. <https://arxiv.org/abs/2608.21833>
37. Yapağcı, Öztürk, Tüzün. *Agents in the Sandbox: End-to-End Crash Bug Reproduction for Minecraft*. arXiv:2503.20036, 2025 (Accepted into ASE'25 (Automated Software Engineering). https://conf.re). <https://arxiv.org/abs/2503.20036>
38. Wang, Tang, Yuan et al.. *Leveraging LLM Agents for Automated Video Game Testing*. arXiv:2509.22170, 2025 (17 pages). <https://arxiv.org/abs/2509.22170>
39. Mu, Yoda, Zhang et al.. *Synergizing Code Coverage and Gameplay Intent: Coverage-Aware Game Playtesting with LLM-Guided Reinforcement Learning*. arXiv:2512.12706, 2025. <https://arxiv.org/abs/2512.12706>
40. Alet, Schneider, Lozano-Pérez, Kaelbling. *Meta-learning curiosity algorithms*. ICLR 2020; arXiv:2003.05325. <https://arxiv.org/abs/2003.05325>
41. Sygkounas, Loutfi, Persson. *Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models*. arXiv:2603.28416, 2026. <https://arxiv.org/abs/2603.28416>
42. Li, Schultz, Hennes, Lanctot (Google DeepMind). *Discovering Multiagent Learning Algorithms with Large Language Models*. arXiv:2602.16928, 2026. <https://arxiv.org/abs/2602.16928>
43. Feldmeier, Fraser. *Neuroevolution-Based Generation of Tests and Oracles for Games*. arXiv:2208.13632, 2022. <https://arxiv.org/abs/2208.13632>
44. Chen, Chen, Jia et al.. *IcFuzz: Fuzzing Isaac Sim with Semantic Stage Guidance and Multi-level Mutation*. arXiv 2026; arXiv:2608.06088. <https://arxiv.org/abs/2608.06088>
45. Politowski, Petrillo, Guéhéneuc. *A Survey of Video Game Testing*. AST 2021; arXiv:2103.06431. <https://arxiv.org/abs/2103.06431>
46. Politowski, Guéhéneuc, Petrillo. *Towards Automated Video Game Testing: Still a Long Way to Go*. ICSE GAS 2022; arXiv:2202.12777. <https://arxiv.org/abs/2202.12777>
47. Kato, Yoshida, Makihara et al.. *Software Testing Beyond Closed Worlds: Open-World Games as an Extreme Case*. FSE 2026 (IVR); arXiv; arXiv:2604.04047. <https://arxiv.org/abs/2604.04047>
48. Abdelfattah, Brown, Zhang. *Preference-conditioned Pixel-based AI Agent For Game Testing*. IEEE CoG 2023 (Xbox/MSR); arXiv:2308.09289. <https://arxiv.org/abs/2308.09289>
49. Lu, Georgescu, Verwey. *Go-Explore Complex 3D Game Environments for Automated Reachability Testing*. IEEE ToG 2022 (Microsoft Research); arXiv:2209.00570. <https://arxiv.org/abs/2209.00570>
50. Tufano, Scalabrino, Pascarella et al.. *Using Reinforcement Learning for Load Testing of Video Games*. ICSE 2022; arXiv:2201.06865. <https://arxiv.org/abs/2201.06865>
51. Ariyurek, Betin-Can, Surer. *Enhancing the Monte Carlo Tree Search Algorithm for Video Game Testing*. IEEE CoG 2020; arXiv:2003.07813. <https://arxiv.org/abs/2003.07813>
52. Prasetya. *Aplib: Tactical Programming of Intelligent Agents*. EMAS@AAMAS 2020; arXiv:1911.04710. <https://arxiv.org/abs/1911.04710>
53. Prasetya, Ricós, Kifetew et al.. *An Agent-based Approach to Automated Game Testing: an Experience Report*. A-TEST 2022; arXiv:2211.06386. <https://arxiv.org/abs/2211.06386>
54. Ferdous, Prasetya, Vos et al.. *Towards Agent-Based Testing of 3D Games using Reinforcement Learning*. ASE 2022, doi:10.1145/3551349.3560507. <https://doi.org/10.1145/3551349.3560507>
55. Shirzadeh-hajimahmood, Prasteya, Dastani et al.. *Cooperative Multi-agent Approach for Automated Computer Game Testing*. EMAS 2024; arXiv:2405.11347. <https://arxiv.org/abs/2405.11347>
56. Chang, Aytemiz, Smith. *Reveal-More: Amplifying Human Effort in Quality Assurance Testing Using Automated Exploration*. IEEE CoG 2019, doi:10.1109/CIG.2019.8848091. <https://doi.org/10.1109/CIG.2019.8848091>
57. Wilkins, Stathis. *World of Bugs: A Platform for Automated Bug Detection in 3D Video Games*. IEEE CoG 2022; arXiv:2206.11037. <https://arxiv.org/abs/2206.11037>
