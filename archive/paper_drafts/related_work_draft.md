# Related work — draft positioning (grounded in the Sep-1 novelty recon). Cite these; position narrowly.

## LLM-driven program/heuristic evolution (our method's family)

FunSearch [Nature'23], AlphaEvolve [2506.13131], and ShinkaEvolve [ICLR'26, our tool] evolve code against an evaluator;
EoH [ICML'24], ReEvo [NeurIPS'24], LLaMEA evolve heuristics. **ELFuzz [USENIX Sec'25]** is the closest — LLM-evolved
*fuzzers*, +216% injected bugs vs expert grammars — but it evolves stateless input generators, not stateful
sequential-decision testers, and never compares to RL. **Position:** we evolve a stateful exploration *policy-as-program*
and put it head-to-head with trained RL at a matched interaction budget — the cell ELFuzz and the code-evolution line
leave open. We also confirm, in a new domain, the caution of *Simple Baselines are Competitive with Code Evolution*
[2602.16805] and *What Do Evolutionary Coding Agents Evolve?* [2605.20086]: our fresh-per-task search barely beats PPO and
loses to a transferred program, and the evolutionary gain over a hand-designed seed is small — the lever is the fitness
(multi-engine vs single-engine), not the search operator. We must run/cite a best-of-N-from-the-same-LLM baseline.

## LLM policies vs RL (our headline's closest claim)

**PromptPO [2605.30719, Brunskill, Jun'26]** reports LLM-written control policies matching/beating RL on 15/19 envs and
far more sample-efficient — but explicitly *declines to match interaction budgets* and targets return maximization.
Voyager and Code-as-Policies write code policies with no RL-at-budget comparison; Eureka [ICLR'24] is the inversion (LLM
writes the *reward*, RL still learns). **Position:** ours is the budget-matched head-to-head PromptPO forgoes, on a
*testing/target-reaching* objective across 8 environments, with the 16×-budget crossover reported honestly.

## Automated game testing (the application)

Wuji [ASE'19] (evolution+DRL, real-bug taxonomy) and CCPT/Gordillo [CoG] (curiosity RL playtesting) are the RL-tester
references a reviewer will name; **Go-Explore-3D [ToG'22]** showed a non-neural archive beats curiosity RL on reachability
— so our RL arm must be best-of-intrinsic-objectives (it is) and we must include Go-Explore (we do — and beat it).
**TITAN [2509.22170]** already claims "LLM agent > DRL tester," and **RELINE [ICSE'22]** injects performance bugs and
counts RL-vs-random triggers at fixed episodes. **Position:** unlike TITAN (a per-step LLM agent, no budget match, no
ground truth) ours is a cheap *static artifact* (LLM paid once, runs free), budget-matched, on *replay-verified planted
defects with graded difficulty* — the artifact + protocol are the novelty, not "LLM beats RL" as a slogan.

## Bug benchmarks (the depth contribution)

**GBQA [ICLR-W'26]** (124 injected bugs, LLM-QA agents) and **GameXpert-Bench [2608.21833]** (~1k reversible-mutation
defects, sealed levels, *repair* focus) are concurrent. GBGallery [EMSE'22] is the real-bug precedent. **Position:** ours
is the first *replay-verified, witness-predicate, reachability-calibrated* defect benchmark for *exploration policies
under an interaction budget* — and, with the graded-difficulty axis, the first to report *detection-accuracy vs bug
difficulty* per method. Cite GBQA/GameXpert/GBGallery in the same paragraph; do not claim "first planted-bug game
benchmark."

## Verification of autonomous research (Part 2/3)

The exploit is not novel as an *incident* — AI Scientist (self-timeout edit), METR [30.4% of o3 RE-Bench runs], Vesper
[2605.15221, 8.2% of candidates, LLM-inspection gate], Stein et al. [2026, 1000+ leaderboard-cheating instances] all
document loops gaming evaluation. **Position (Part 2/3):** our exploit is a *concurrency/state-leakage* class with nothing
in the candidate's code — which **defeats the dominant code-audit defense (Vesper) that reads the candidate and finds
nothing**, and is caught only by replay-from-reset. Frame as a falsifying counterexample to code-inspection defense, and
cite Krakovna's specification-gaming corpus + Skalse et al. on reward hacking. Do not claim replay-in-fitness as an
invention (RewardHackingAgents [2603.11337] "locked" recompute, Vesper's pre-selection gate exist); claim the composition
into a live discovery loop + the demonstration that inspection provably misses this class.
