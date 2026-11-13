# AutoDev at AutoMLR

Feedback on the auto-game-dev idea, the prior work it has to clear, and three fully specified directions — each reframed so that an autonomous agent's contribution *is* the paper's result — with a recommendation and a nine-day plan.

Prepared 28 Aug 2026 for Andy (UW) · Venue: Workshop for Autonomous Machine Learning Research (AutoMLR), NeurIPS 2026, Sydney · Deadlines: abstract **Sat 29 Aug**, paper **Sat 5 Sep** (both already extended once; time zone to be posted) · Every citation below was checked against arXiv/Crossref/the live page on 27–28 Aug 2026.

## Bottom line

> **Recommendation.** Submit **Direction 2 — "Discovered testing objectives."** An autonomous code-evolution loop discovers an exploration objective for finding injected game defects; the paper's result is a *held-out transfer* claim (unseen games and unseen bug classes) against tuned hand-designed baselines. It has the crispest "decisive agent contribution" story (the discovered objective is agent-authored code), the harness already exists (ShinkaEvolve has native SLURM evaluation), it is the direction that converts unlimited GPU time most directly into evidence, and it keeps the auto-game-dev vision — an agent that plays your game and finds what is broken — at the center. Take **Direction 1** if you want the paper to be about the research agent itself rather than an artifact it discovered; take **Direction 3** only with a second person to own experimental design.

Three facts change the shape of the whole project:

1. **AutoMLR is not a games venue and not a systems venue.** It accepts "machine learning research in which an autonomous agent either conducted the research end-to-end or made a decisive contribution to the paper's primary result," and the FAQ answers "Can I submit a systems-only paper?" with "No" [1]. A paper *about* a game-testing tool is out. The fix is a reframing, not a new idea: **the game is the research problem, the "checker loop" is the autonomous researcher, and the paper's result is what that researcher discovered.**
2. **The deadline is nine days from when this was scoped**: abstract 29 August, paper 5 September [1]. Every design below is scoped to one full run on Hyak. Unity is out for this paper and stays the long-term target.
3. **The prior art is dense and very recent.** Each direction as first sketched has a 2025–26 near-neighbour — Agent² and LIMEN for D1, GBQA and GameXpert-Bench for D2's benchmark half and REvolve/Alet et al. for its discovery half, the Tetris-difficulty and Synthetic-Monitoring-Environments papers for D3. None is fatal, but each forces a narrower, more specific claim. Those claims are what the three directions are built on.

## What AutoMLR actually wants

Quoted or paraphrased from automlr.com as of 28 Aug 2026 [1]; the site says final disclosure instructions ship with the submission portal.

- **Eligibility.** "The paper must state the qualifying result in its abstract. Without the agent's contribution, the qualifying result could not have been established, or the paper's main conclusions would be materially different." Both a qualifying ML result *and* a detailed account of the system are required — "Neither component is sufficient on its own." Non-ML science is out of scope.
- **Three labelled parts; page limits cannot be reallocated.** Part 1 (≤4 pp) "Auto research result — the hypothesis, method, evidence, and primary result generated or developed by the agent." Part 2 (≤4 pp) "System design — the agent, harness, tools, research loop, etc.," plus "commentary on the significance of the main result." Part 3 (≤1 p) "Reflections" on how the field should adapt, feeding the closing town hall. Nine pages total excluding references; Overleaf template provided.
- **Who writes what.** "We expect much of Part 1 to be generated and written by the agent (including writing, figures, etc.)." "We expect Part 2 to be human-written; this will include details on the agent, prompts, harness, tools, research loop, human interventions, and verification." Authorship is exclusively human; authors "verify its claims, disclose agent involvement," and are responsible for "guarding against hallucinated results, figures, or citations." This tracks NeurIPS 2026's policy that agents cannot be authors and that non-standard LLM use must be documented [2].
- **Review.** Blind review under the NeurIPS conflict-of-interest policy; at least three reviews; every submission must supply min(3, #authors) reviewers; accepted papers "will be published with their reviews and reviewers"; a non-author **discussant** prepares "an account of the contribution, strongest evidence, unresolved limitations, and questions for the author" and leads the session.
- **Logistics.** Non-archival. "An individual who is listed as first author on one submission may not appear as an author on any other workshop submission." At least one author presents in person in Sydney on 11 or 12 December. Decisions 29 Sep; discussant notification 6 Oct; camera-ready 13 Nov.
- **Who is in the room.** Invited speakers: Sherry Yang (RL, world models, agents for ML engineering), Mengdi Wang (RL theory, agents for discovery), Nik Dawson (AI and labour markets). Three organizers — Prakash, Wang and Greenwald — are co-authors of a May 2026 paper on training a 3B model to turn game-rule descriptions into executable "game code world models" with a rule-verification framework [3]. RL-on-games research done by agents is squarely in this committee's taste.

What that implies for the design:

- **The primary result must be a falsifiable ML claim with a number**, stated in the abstract, that the agent produced. "We built a system that finds bugs" is not one. "An agent-discovered objective reduces steps-to-defect by X% on held-out games" is.
- **The agent must visibly decide things** — hypotheses, method, analysis — not merely execute a human-specified sweep. Every prior-art hunter in this study flagged the same failure mode independently: if humans build the substrate, the benchmark, the baselines and the analysis, the agent's contribution collapses to "ran the loop," which reads as human research with an LLM inside.
- **Provenance is a deliverable.** Part 2 needs prompts, the loop, every human intervention, and how the numbers were verified. Plan the audit trail on day 1; Claude Code's hook events give a timestamped tool-call ledger for free [4].
- **Overclaiming novelty is the cheapest way to lose.** Reviews are published and a discussant reads the paper adversarially. Related work must name the 2026 near-neighbours.

## Feedback on the auto-game-dev idea

### What is genuinely good about it

- **It decomposes exactly into an autonomous-research loop.** "Look at the game → build an agent that can play it → get it good → use its play to find problems" is perceive → hypothesize → implement → experiment → analyze. Every step is checkable by execution, which is the property this community has learned to care about most: benchmarks of research agents find that roughly 80% of agent-produced experimental results are fabricated or invalid [5], that agents complete a fully correct end-to-end experiment 0.5% of the time [6], and that even frontier scaffolds beat a paper's own baselines in 1 of 15 attempts [7].
- **Games are the cheapest honest testbed for agent-run research.** Programmatic state gives oracles — "the agent is outside the level bounds," "the goal is unreachable from here" — so the agent's claims can be verified without a human replaying anything. That directly answers the CFP's warning about hallucinated results.
- **Compute is your comparative advantage.** All three directions are bottlenecked by simulator throughput and seed counts, not by ideas. JAX substrates on Hyak turn "thousands of PPO runs" into hours.
- **Contamination control comes for free.** Games generated on 1 September are in no pretraining corpus. Almost no RL-automation or research-agent benchmark can say that: Agent² runs on MuJoCo/MetaDrive/MPE/SMAC [8]; MLGym and RE-Bench use public tasks [9, 10]. The one recent exception treats a privately synthesized environment explicitly as a contamination control [11], which shows reviewers value the property.

### What will not work as literally stated

- **"Build your game in Unity" is a 2027 sentence.** Unity ML-Agents' PyPI package has been frozen at 1.1.0 since October 2024, each new game needs Editor and C# work, and there is no headless vectorized path [12]; Godot RL Agents is lighter and explicitly lists automated gameplay testing as a goal, but still runs one OS process per environment [13]. Neither saturates a cluster in nine days. Keep engine integration as the product vision in Part 3.
- **"Games of all shapes and horizons" must become "one game family with a DSL."** The agent needs a uniform way to read the game (a text spec), a uniform action/observation contract, and short fixed horizons; otherwise interface construction eats the schedule. That is exactly why PuzzleScript-in-JAX, Kinetix and XLand-MiniGrid exist [14, 15, 16].
- **"The agent finds bugs" is an application, not an ML result.** Curiosity-driven RL testers are an EA SEED production line from 2020 to a July 2026 NHL 26 case study [17, 18, 19, 20]; NetEase and Microsoft shipped Wuji and Inspector [21, 22]; modl.ai sells integration-free VLM playtesters [23]; an LLM-agent bug-hunting benchmark with 124 injected bugs appeared this April [24]. To fit AutoMLR the *discovery* has to be the result — a discovered objective, a discovered recipe, a discovered causal explanation.
- **"An autonomous agent did science end-to-end" is no longer a headline.** The AI Scientist, Agent Laboratory, Kosmos and CodeScientist own that sentence [25, 26, 27, 28, 29]; a July 2026 paper already runs a semi-autonomous RL research loop with pre-registered, falsifiable hypotheses [30]; and an audit literature now documents agents silently shrinking datasets and training budgets [31], narrowing exploration toward their source material [32], and defending general claims their own data contradicts [33]. Novelty has to live in the *object of study*, and the process has to be logged well enough to survive that audit.

### How to constrain the game space — the scope rules

1. **Games are data, not engine code.** Choose a substrate where a game or level is a text/JSON spec the agent can read and a program can mutate. This is what makes "the agent reads the game" and "inject a bug" both one-line operations.
2. **Full programmatic state**, so bug oracles and learnability labels are computed, never eyeballed.
3. **Fixed horizon, discrete-ish actions, GPU-vectorized simulator** — thousands of parallel environments per GPU; PPO runs that finish in minutes.
4. **A frozen evaluation harness the agent cannot edit**, with a sealed held-out split (games and, for D2, bug classes) fixed *before* the agent starts. This is the defence against the two documented failure modes of agent-driven search: overfitting to the training distribution [34] and the agent editing its own evaluation (Sakana's system once removed its own timeout) [35].
5. **Contamination check by construction**: generate games after the model's training cutoff and hash-check them against public corpora (e.g., the 951 scraped PuzzleScript games shipped with PuzzleJAX).

### Substrate options (facts verified 27–28 Aug 2026)

| Substrate | A game is… | GPU-vectorized | Full state | Best for | Caveats |
|---|---|---|---|---|---|
| **PuzzleJAX** [14] — `smearle/script-doctor`, MIT | PuzzleScript text DSL; 951 scraped community games, 414 fully valid | Yes (JAX; 2–16× the JS engine at large batch) | Yes (multi-hot grids) + tree-search solver → exact reachability oracle | D1 (heterogeneous interfaces), D2 softlock oracles, D3 | Not on PyPI; Python 3.13; `rigid` keyword unimplemented; small deterministic games → tree search may find defects without learning |
| **Kinetix** [15] — `pip install kinetix-env`, MIT, pushed May 2026 | JSON levels (shapes, joints, thrusters) with a level editor | Yes (Jax2D physics) | Yes (pytree) | **D2** — one JSON field = one defect; continuous state defeats tabular novelty; D1 | Fixed physics and action space → "unseen games" are unseen *levels*; one genre |
| **XLand-MiniGrid** [16] — `pip install xminigrid`, Apache-2.0 | Rule-tree tasks as *data*; millions of pre-sampled tasks | Yes (millions of steps/s claimed) | Yes | **D3** — interpretable design features (rule depth, goals, distractors) | Gridworld only; well-known family |
| Craftax [36] / NAVIX [37] / Jumanji [38] / Gymnax | Fixed catalogs (one world / MiniGrid ports / 22 puzzles / classic control) | Yes | Yes | Baselines and sanity checks | New games need JAX code; heavily represented in pretraining |
| PufferLib Ocean [39] — MIT, pushed 20 Aug 2026 | ~55 C environments; a new one is ≈140 lines of C | CUDA (20M steps/s claimed) | Yes (C struct) | Long-term option for LLM-written games | Compiles from source; not JAX |
| GVGAI/VGDL [40, 41], Ludii/Ludax [42, 43], Griddly [44] | Text/YAML DSLs, 100+ games | No (Java/CPU), or new JAX ports | Yes | Vocabulary and corpora | Python paths stale since 2019–22; would burn the nine days |
| Unity ML-Agents [12], Godot RL Agents [13] | Engine scenes + scripts | No | Per-game instrumentation | The product vision (Part 3) | Not for this paper |

### Other ideas worth keeping on the shelf

- **Fresh-game contamination control as a reusable protocol** (D1's core) bolts onto D2 or D3 at low cost: generate the held-out games on the day, hash-check them, report the "never-seen" number separately.
- **Agent-versus-agent breaking.** One agent evolves levels a fixed tester fails on (unsupervised environment design), the other evolves the tester — a self-play QA curriculum. Strong follow-up; too much for nine days.
- **Human-feedback validation.** Correlate agent playtest reports with human game-jam feedback. A real study, but human data is out of reach before 5 September, and it is not an ML result.
- **Self-improving play as the product loop.** SIMA 2 already uses Gemini to generate tasks and rewards so the agent learns new skills in a new environment on its own [45]; the same shape, with a game DSL instead of 3D worlds, is the long-run auto-game-dev architecture.

## Previous work and where it falls short

### A. Autonomous ML-research systems and their benchmarks

**What exists.** The AI Scientist closed the loop from idea to LaTeX at roughly $15 per paper [25]; AI Scientist-v2 replaced fixed templates with agentic tree search built on AIDE [46] and produced the first fully AI-generated manuscript to clear workshop peer review — at ICLR 2025's ICBINB workshop, scores 6/7/6, with humans choosing only the broad topic and the best of three candidates; the paper was withdrawn before publication [26, 35]. Agent Laboratory automates literature review, experiments and report writing from a human-supplied idea [27]. Kosmos runs 12-hour cycles (about 42,000 lines of code and 1,500 papers per run) and reports 79.4% of its statements verified accurate [28]. CodeScientist ran hundreds of agent-generated experiments "broadly in the domain of agents and virtual environments," returning 19 discoveries of which 6 survived external review [29]. Agents4Science 2025 accepted 48 AI-first-authored papers — two tagged RL, none on games [47] — and among them BadScientist, which shows a fabrication-oriented agent can get past multi-model LLM reviewers [48]. Karpathy's autoresearch popularized the minimal pattern: an agent edits one training script under a five-minute budget, overnight [49]. A July 2026 paper runs a semi-autonomous RL research loop on quadruped navigation with immutable experiment cards and reports falsification rates [30].

Benchmarks grade the same capability: MLE-bench (Kaggle) [50]; RE-Bench, where agents out-score human experts 4× at a two-hour budget but humans reach 2× the agents' score at 32 hours [10]; MLGym, whose 13 tasks include RL and game theory and where frontier models "usually" improve baselines "by finding better hyperparameters, but do not generate novel hypotheses, algorithms, architectures" [9]; MLR-Bench (201 workshop-style tasks; roughly 80% of results fabricated or invalid) [5]; EXP-Bench (461 tasks; 0.5% fully correct experiments) [6]; PaperBench (replication of 20 ICML papers; best agent 21%) [51]; ResearchGym (methods withheld from five ICML/ICLR/ACL repositories; a GPT-5 agent beats the baseline in 1 of 15 evaluations, Claude Code and Codex show the same gap) [7]; AIRA-dojo (scheduler-integrated search over solutions; medal rate 39.6% → 47.7%) [52]; and Agent² RL-Bench (can agents engineer RL post-training pipelines end-to-end under a fixed budget? — sometimes, but "stable agent-driven RL post-training remains rare") [53]. Audits find "methodological hallucinations": silently reduced datasets and budgets, oracles substituted for failed components [31]; agents narrowing exploration toward their sources [32]; and correct answers reached by the wrong mechanism [33].

**Where it falls short for this project.** (i) Almost none of it does RL *from scratch on an unseen environment*: MLGym's RL tasks are standard environments; Agent² RL-Bench targets LLM post-training on ALFWorld and DeepSearchQA. (ii) Every task is public and in pretraining corpora; contamination is unaddressed. (iii) "The agent did science" is saturated; the discussant will ask what was discovered and how it was verified. (iv) Reliability is the known weakness — Sakana's own list of failure modes (removed its own timeout, misreported numbers, cannot compare magnitudes) and BadScientist define the bar.

### B. LLM-driven RL automation and objective/algorithm discovery

**What exists.** Eureka evolves reward code from training statistics and beats expert-written rewards on 83% of 29 Isaac Gym tasks [54]. Agent² takes a task description plus environment code and autonomously produces the MDP mapping, algorithm, architecture and hyperparameters, beating manually designed baselines by up to 55% on MuJoCo, MetaDrive, MPE and SMAC [8]. LIMEN evolves observation mappings *and* reward functions as executable programs from policy-training feedback, on novel gridworlds and continuous control, and finds that optimizing either alone "fails catastrophically on at least one domain" [55]. Environments themselves are generated by OMNI-EPIC (a foundation model writes environment and reward code filtered by learnability and interestingness) [56], Eurekaverse [57], EnvGen [58], and a 2026 closed loop that emits verified GPU implementations, using a privately synthesized environment as a contamination control [11]. Objective and algorithm discovery predates LLMs: Alet et al. searched a program DSL of curiosity mechanisms and found two that perform "on par or better than human-designed published curiosity algorithms" across gridworlds, Acrobot, Lunar Lander, Ant and Hopper [59]. The LLM-evolution wave then made "LLM evolves code, validated by full training runs" a standard recipe: DiscoPOP for preference losses [60]; DiscoRL's discovered update rules [61]; AlphaEvolve [62] and its open implementations ShinkaEvolve (native SLURM evaluation; bandit LLM ensemble; a circle-packing state of the art in 150 samples) [63] and OpenEvolve [64]; REvolve, which evolves complete RL update rules competitive with SAC/PPO/DQN/A2C on Gymnasium (GECCO 2026) [65]; and DeepMind's AlphaEvolve search over CFR/PSRO, which documents that the evolved code "optimizes for fitness on a specific training set," so only a distilled minimal core generalizes [34]. Surveys: AutoRL [66]; ADAS for agent architectures [67]; PCGRLLM for LLM-designed rewards in content generation [68].

**Where it falls short.** (i) Nobody searches for *testing or exploration objectives with a defect-detection fitness*; the target has always been task return or a learning rule. (ii) Nobody evaluates on environments that did not exist before the run. (iii) Held-out-operator or held-out-game generalization is rarely the headline, even though overfitting is the documented failure. (iv) None of these systems is framed as an autonomous *researcher* that states a falsifiable recipe and tests it.

### C. Learning-based game testing and injected-bug benchmarks

**What exists.** Wuji analyzed 1,349 real bugs to derive four test oracles and combined evolution with DRL to test NetEase combat games [21]. EA SEED's line runs from RL testers with tunable explore/exploit rewards [17], to curiosity-driven coverage in 3D levels [18], to CCPT — curiosity plus imitation near expert trajectories in a AAA-scale navigation environment [19] — to RAID, which adapts rewards across a population and found six distinct exploits against the NHL 26 goalie AI in a single experiment [20]. Inspector detects bugs from pixels via curiosity-driven exploration, object detection and imitation-based investigation, finding two potential bugs in two commercial games [22]; Ariyurek et al. compare synthetic and human-like testers [69]; GLIB learns a visual oracle for graphics glitches [70]; CA² conditions an RL tester on call-stack traces [71]. LLMs enter as oracles and testers: Taesiri et al. show zero-shot bug detection from event descriptions at up to 70.66% accuracy over 8 games [72] and build GlitchBench for multimodal glitch detection [73]; TITAN reaches 95% task completion testing commercial MMORPGs [74]; BugCraft reproduces Minecraft crash reports [75]; SMART reads AST diffs and constructs coverage-aware hybrid rewards for RL playtesters, reaching over 94% branch coverage of modified code at 98% task completion on Overcooked and Minecraft [76]; CureFuzz uses curiosity to fuzz trained policies [77]. Benchmarks with *injected* bugs: World of Bugs (Unity, perceptual bugs) [78]; GBQA — 30 games and 124 human-verified bugs built by a multi-agent pipeline, where the best frontier model finds 48.39% [24]; GameXpert-Bench (Aug 2026) — 100 repair tasks from 50 levels with 19–27 injected bugs each, and the finding that coding agents are "more reliable at producing playable foundations… than at discovering defects" [79]; and NEATEST's mutation analysis over 25 Scratch games [80]. Surveys and taxonomies: Politowski et al. [81], Lewis et al. [82]. Industry has converged on integration-free VLM testers [23].

**Where it falls short.** (i) Every testing objective is hand-designed — CCPT's curiosity form, RAID's diversity mechanism, SMART's per-diff template. (ii) The injected-bug benchmarks target prompted LLM agents or Unity; none is RL-trainable at the thousands-of-runs scale a discovery loop needs, and none ships RND/ICM/count-based baselines. (iii) Evaluation is coverage or qualitative agreement with human-found exploits, not steps-to-detection on ground-truth defects with held-out bug classes. (iv) CCPT's environment is proprietary; "beats CCPT" can only mean a re-implementation of its objective form.

### D. Foundation-model game agents and benchmarks

**What exists.** Cradle plays commercial games from screenshots [83]; SIMA 2 follows instructions across 3D games and self-improves from Gemini-generated tasks and rewards [45]; PORTAL synthesizes behavior-tree policies for thousands of FPS games in production, deliberately avoiding RL [84]. Benchmarks: BALROG [85]; VideoGameBench, where the best models complete 0.48% of ten 1990s games (three kept secret) [86]; lmgame-Bench [87]; Orak [88]; GVGAI-LLM, 118 VGDL games "very far from being solved" with "persistent limitations of LLMs in spatial reasoning and basic planning" [89]; and gg-bench, whose LLM-generated games with RL self-play opponents yield 7–9% win rates for GPT-4o-class models and 31–36% for reasoning models [90].

**Where it falls short.** They measure *playing*, not research. LLM agents are consistently weak at spatial, long-horizon play in exactly the 2D families we would use — so an "LLM policy" arm is a known negative result, worth one paragraph, not a week. None injects bugs or studies learnability.

### E. Game generation and DSL substrates

**What exists.** PuzzleJAX compiles any PuzzleScript game to a GPU engine and validates several hundred human-authored games, reporting that they are "simple and intuitive to understand, yet often deeply challenging to master" for search, RL and LLMs alike [14]; ScriptDoctor generates and play-tests PuzzleScript games with LLMs and tree search [91]; Autoverse [92] and Ludax [43] are the same group's evolvable and board-game DSLs. Kinetix procedurally generates tens of millions of 2D physics tasks and shows zero-shot transfer to unseen human-designed levels [15]; XLand-MiniGrid composes millions of rule-tree tasks [16]. LLM game generation spans VGDL rules and levels [41], Ludii games via evolution [93, 42], story-driven worlds [94], and the organizers' own distillation of game-code world models into a 3B model with rule verification [3]; surveys cover the LLMs-and-games landscape [95]. Genie 3 generates interactive worlds from text [96].

**Where it falls short.** These are substrates and generators. No autonomous agent runs research on top of them, and none reports contamination-controlled, fresh-game generalization of an RL pipeline.

### F. Difficulty, learnability and AI playtesting

**What exists.** King's group predicts per-level pass rates and churn from DRL (and DRL+MCTS) play, finding that an agent's best-case runs correlate with human data better than its average [97, 98]; Kristensen et al. find the moves taken by the best ~5% of RL runs are the strongest predictor of human completion rate [99]; Gudmundsson et al. train human-like playtesters for difficulty QA [100]. DeepMind retrained AlphaZero per chess-rule variant to assess balance [101]; a March 2026 paper reads Tetris-variant difficulty off AlphaZero training reward and convergence [102]; Synthetic Monitoring Environments run controlled ablations of how state/action size, reward sparsity and optimal-policy complexity affect PPO/TD3/SAC [103]. No Regrets defines *learnability* as the levels an agent "can sometimes solve, but not always," shows regret proxies fail to predict it, and trains directly on it (the curriculum Kinetix uses) [104]; TRACED refines the proxy [105]; PIC/POIC predict task difficulty before training [106]; the Pharos study shows tabular hardness metrics miss "representation hardness" in deep RL [107]. RuleSmith autonomously balances a civilization-style game with LLM self-play and Bayesian optimization [108]; Xiao & Yang show LLM testers' scores correlate with human-reported difficulty on Wordle and Slay the Spire [109]; MeepleLM simulates diverse subjective playtesters for board-game critique [110]; OMNI-EPIC uses foundation-model-judged learnability as a generation filter [56]; Automated Capability Discovery has one model propose thousands of tasks to map another's failure modes [111].

**Where it falls short.** Human-run studies on single game families with hand-picked knobs; correlational, not interventional; no predictor validated on sealed held-out designs; nothing attributes unlearnability to interpretable design features or tests repairs; and none of it is done by an autonomous agent.

## Direction 1 — Fresh-game AutoRL

*Does an RL-automation agent's recipe transfer to games that did not exist yesterday?*

**Question.** Given games generated after every model's training cutoff, can an autonomous research agent induce an explicit pipeline recipe — observation/action interface, reward shaping, algorithm and hyperparameters, conditioned on game properties — that transfers to a sealed second batch of fresh games, and does it beat both default PPO and the published automation recipes? The sharpest version, which the prior-art hunt surfaced: hand the recipe to a *search-free executor* with no iteration budget on the sealed batch. If recipe plus cheap executor matches the expensive agent-from-scratch, the agent produced transferable *knowledge*, not transferable compute — which is exactly the workshop's bar, and something none of the near-neighbours claims.

**Why it fits AutoMLR.** The agent does the research: it decides what to try per game, induces a written, falsifiable recipe, runs the transfer sweep, analyzes it and writes Part 1. The result — the recipe's held-out transfer number plus the taxonomy of game properties on which every recipe fails — cannot exist without the agent. It also targets precisely what MLGym found agents do *not* do: generate hypotheses rather than tune hyperparameters [9].

**Qualifying result (abstract template).** "On a sealed batch of *N* freshly generated PuzzleScript games unseen by any model, the recipe induced by an autonomous research agent on a disjoint batch — executed one-shot by a search-free executor with no iteration budget — raised the fixed-budget solve rate from *X*% (default PPO) and *Y*% (re-implemented Agent²/LIMEN-style automation) to *Z*% (95% CI over 10 seeds), within *W* points of the full research agent at a fraction of its cost, and identified {property classes} on which every recipe fails."

**Hypotheses the agent starts from (it may replace them — that is the point).** H1: observation design (object-channel encodings, action masking) explains more variance in solve rate than algorithm or hyperparameter choice. H2: LLM-written potential-based shaping derived from the win condition helps sparse-reward games but hurts games with irreversible moves. H3: a recipe induced on batch A transfers to batch B with at most ε loss in solve rate. H4: recipe plus a search-free executor matches the iterating agent on batch B — knowledge transfers without compute.

**Substrate and game space.** PuzzleJAX as the engine [14]. Fresh games come from (a) rule and level mutation of validated corpus games and (b) LLM-authored PuzzleScript in the ScriptDoctor pattern [91], compiled through PuzzleJAX. Keep a game only if it compiles, the tree-search solver certifies solvability within a depth bound, a random policy solves fewer than 5% of episodes, and a short PPO probe places it in the learnability band [104] — PuzzleJAX's own suite is far from solved, and floor effects would zero every arm. Hash every accepted game against the 951-game scraped corpus. Fallback: Kinetix levels [15] — faster and JSON-native, but the fixed interface removes the interface-construction task that distinguishes D1 from plain hyperparameter optimization.

**The autonomous system.** A coding agent (Claude Code driven by a `program.md`-style brief, in the autoresearch pattern [49]) with tools to read a game spec, write JAX wrappers, shaping functions and configs, submit SLURM arrays, read metrics, plot, and write LaTeX. A frozen evaluation script for batch B that the agent cannot modify. Every tool call is logged through hooks [4]; one git commit per experiment with the generating prompt in the message.

**Arms.** (a) default PPO (PuzzleJAX baseline); (b) time-boxed re-implementations of Agent²-style MDP mapping + HPO [8] and LIMEN-style joint observation/reward evolution [55]; (c) the research agent iterating from scratch on batch B (expensive reference); (d) the agent's written recipe applied one-shot by a search-free executor (the headline arm); (e) optional: learnability-sampled curriculum [104]. Equal environment-step budgets per game; 10 seeds; report the LLM/GPU cost of each arm.

**Metrics.** Solve rate at budget; normalized return; area under the learning curve; per-property breakdown; transfer gap (batch A → B); recipe length and specificity.

**Novelty positioning.**

| Closest work | What it does | What D1 keeps |
|---|---|---|
| Agent² [8] | LLM builds MDP mapping, picks algorithm/architecture/hyperparameters, iterates; up to 55% over manual baselines on public suites | Fresh-game contamination control; an explicit induced recipe as artifact; published recipes as baselines; a negative-result taxonomy |
| LIMEN [55] | Evolves observation and reward programs from training feedback; co-design finding | Algorithm/hyperparameter choice, cross-game recipe, transfer to sealed games |
| Agent² RL-Bench [53] | Benchmarks agents engineering RL post-training pipelines under a budget | From-scratch game-playing policies; never-seen environments |
| MLGym / ResearchGym [9, 7] | Research-agent benchmarks; withheld-method protocol | RL on generated games; contamination-free tasks |
| PuzzleJAX / GVGAI-LLM [14, 89] | Substrates reporting that the suites are far from solved | No autonomous agent configures RL there |
| PORTAL [84] | LLM writes behavior-tree policies for thousands of FPS games | Actual RL research; empirical findings about learning |
| AutoEnv [112] | Auto-generates heterogeneous environments (AutoEnv-36) to measure whether one learning method transfers; finds fixed methods plateau | Learners are in-context LLM agents there; D1 trains RL policies, engineers interfaces, uses a sealed second batch, and transfers a written recipe. Use its plateau finding as the motivating negative result |
| EnvHarness, GIFT, reward-observation-space evolution [113, 114, 115] | Automated wrapper synthesis; LLM-designed state features and shaping refined from PPO diagnostics | Cross-game recipe; fresh-game generalization; algorithm choice |
| DiscoGen / Camyla [116, 117] | Procedurally generated, contamination-free tasks for automated ML discovery / autonomous research | Playable games; interface and shaping engineering; the executor test. Cite both or the contamination framing reads as derivative |

**Risks and mitigations.** *Floor effects* — filter to the learnability band before the agent starts. *Prior-work collision* — treat Agent²/LIMEN as baselines, not as omitted related work. *Contamination leakage* — mutation of corpus games can leak; hash-check and prefer LLM-authored games. *Compile rate* of LLM-authored PuzzleScript is unreported; budget a day-1 spike. *Seed variance* — 10 seeds per game per arm; cheap on Hyak, but the logging must exist. *Human-pipeline baseline* — unaffordable in nine days; use published recipes instead and say so. *Decisiveness* — if the agent only tunes, reviewers will read "HPO"; require it to emit and test a recipe. *"Novel conjunction" reviews badly* — every ingredient (interface co-design [114, 55], wrapper synthesis [113], diagnostic-driven shaping [118, 119], fresh-environment measurement [112]) is individually claimed; the executor experiment is what makes D1 a result rather than a combination. No published work combines Kinetix or PuzzleJAX with an LLM-driven RL pipeline — that hole is real, but it is a hole, not a contribution.

**Nine-day feasibility: medium-hard.** Needs an agent that drives SLURM, a generator with four filters, two baseline re-implementations and the sealed split. One person can do it only with a mutation-only generator and simplified baselines.

**Paper sketch.** Part 1 (agent-written): the recipe, the transfer table, the failure taxonomy, learning curves. Part 2 (human): the brief, tools, filters, the frozen evaluator, every intervention, how numbers were recomputed from logs. Part 3: contamination-controlled evaluation as a norm for autonomous RL research.

## Direction 2 — Discovered testing objectives

*Does an agent-discovered exploration objective transfer across bug classes and games?*

**Question.** Can an autonomous code-evolution loop discover a single exploration objective, expressed as code, that finds injected defects faster than tuned hand-designed objectives on held-out games *and held-out bug classes* — and does searching against a defect-detection fitness produce something that task-return search does not?

**Why it fits AutoMLR.** The discovered objective is agent-authored code; the transfer study is agent-run; Part 1 is agent-written. Without the loop there is no result. Humans pose the question, build the substrate, the mutation catalogue, the oracles, the baselines and the sealed split — all disclosed in Part 2. To keep the agent's contribution decisive rather than "mutation operator," the agent also proposes the hypotheses, designs the transfer ablations, and writes the analysis, with its reasoning logged.

**Qualifying result (abstract template).** "An autonomous discovery loop found an exploration objective that reduced expected steps-to-first-defect by *X*% relative to the best tuned hand-designed baseline (count-based / RND / ICM / curiosity-proximity) on held-out games and held-out defect classes (95% CI over *K* mutants × 10 seeds), whereas objectives evolved against task return transferred to only *Y*% of held-out classes."

**Hypotheses.** H1: objectives evolved against defect-detection fitness transfer to unseen defect classes; objectives evolved against task return do not. H2: the transferable core of a discovered objective is small (distillation reduces complexity without losing transfer, as in [34]). H3: tabular count-based novelty is the strongest hand-designed baseline in discrete DSL games but not in continuous physics games.

**Substrate.** Kinetix primary [15]: JSON levels, so a defect is a field edit; continuous physics state gives headroom over tabular novelty; `pip install kinetix-env`; active repository. PuzzleJAX secondary [14]: its tree-search solver provides *exact* reachability oracles for softlocks, but small deterministic state spaces let search find defects without learning — a search baseline must be shown insufficient before learning is claimed.

**Defect catalogue — witness-state bugs with programmatic oracles.** Detection = first visit to a witness state.

| Class | Injection (Kinetix JSON) | Oracle |
|---|---|---|
| Collision hole | Remove or thin a wall segment | Agent body intersects previously solid geometry |
| Out-of-bounds | Delete a boundary polygon | Position outside level extents |
| Unintended shortcut | Open a path through a region designed impassable | Goal reached while trajectory crosses the flagged region |
| Reward exploit | Make a reward object re-triggerable | Episode return exceeds the designed maximum |
| Physics anomaly | Perturb mass/restitution/motor limits | Speed or penetration beyond engine tolerances |
| Softlock (PuzzleJAX only) | Rule edit that makes the goal unreachable from some visited states | Solver certifies goal unreachable from the current state |

Calibrate before searching: discard mutants a random policy triggers within budget and mutants nothing triggers; report the discarded fraction. Be explicit that this construction follows GBQA, World of Bugs and mutation analysis [24, 78, 80] rather than inventing the idea.

**Splits, frozen before the loop starts.** Games: search set / held-out set (fresh levels generated on the day, hash-checked). Defect classes: leave-one-class-out rotation; the headline number is on the held-out class.

**The loop.** ShinkaEvolve [63] (native SLURM job backends, multi-run noisy fitness, novelty rejection) or OpenEvolve [64]. The LLM proposes `intrinsic_reward(state, next_state, history)` in JAX inside the evolve block; fitness = normalized expected steps-to-first-witness aggregated over search games × search classes × seeds, with a dense surrogate (area under the detection curve) to tame binary noise; oracle predicates are *not* visible to the proposer — the interface exposes only raw state, so oracle-matching solutions cannot be written by construction. A control run uses the same loop with task-return fitness. Scale: hundreds of candidates, each scored by dozens of short vectorized PPO runs — hours on a few GPUs. Eureka's reward-reflection loop is the closest published analogue of the inner mechanism [54].

**Baselines — tuned, not strawmen.** Count-based novelty over hashed state; RND; ICM; a re-implementation of CCPT's curiosity-plus-proximity objective form (its environment is proprietary [19]); random and task-return PPO; tree search where applicable. Each gets a small hyperparameter sweep on the search set.

**Metrics.** Steps-to-first-trigger (mean, median); detection AUC; fraction detected within budget; the class × class transfer matrix; objective complexity before and after distillation.

**Novelty positioning.**

| Closest work | What it does | What D2 keeps |
|---|---|---|
| Alet et al. 2020 [59] | Program search over a hand-built curiosity DSL, task-return fitness, transfer across environments | LLM proposer over open code; defect-detection fitness; held-out defect-class transfer as the claim |
| GBQA / GameXpert-Bench / World of Bugs [24, 79, 78] | Injected-bug benchmarks for LLM agents or Unity | RL-trainable, GPU-vectorized suite with witness oracles and exploration baselines |
| CCPT / RAID / SMART [19, 20, 76] | Hand-designed or per-diff LLM-written testing rewards | A discovered, general objective evaluated on ground-truth defects |
| REvolve / DeepMind multiagent discovery [65, 34] | LLM evolves RL update rules / game-theoretic solvers | Different target (exploration objectives), defect fitness, explicit anti-overfitting protocol |
| Eureka [54] | Reward code evolved from training statistics for robot tasks | Testing objectives; games; transfer across bug classes |
| CureFuzz [77] | Curiosity to fuzz a trained policy | The environment, not the policy, is the system under test |

**Risks and mitigations.** *Baseline collapse* — count-based novelty is near-optimal in small discrete games; that is why Kinetix is primary. *Overfitting to operators* — the documented failure of evolved code [34]; the held-out class is the headline and the objective is distilled. *Degenerate oracle-matching objectives* — oracle-blind interface plus held-out classes. *Noisy binary fitness* — dense surrogate, many mutants and seeds. *"Agent = mutation operator"* — the agent proposes hypotheses, designs ablations, writes Part 1; the loop's reasoning is logged. *Time* — the mutation harness and oracles are the unglamorous 2–3 days; start them first.

**Nine-day feasibility: medium.** The search loop exists; the substrate installs with pip; the new code is the defect injector, five oracles, four baselines and the frozen evaluator.

**Paper sketch.** Part 1 (agent-written): the discovered objective (code), the transfer matrix, detection curves against baselines, the task-return control, the distillation ablation. Part 2 (human): substrate, catalogue, oracles, splits, loop configuration, prompts, interventions ledger, verification. Part 3: what counts as "decisive" when a human designs the search space — and why witness oracles make agent claims cheap to check.

## Direction 3 — Causes of RL unlearnability

*Can an agent discover, and interventionally validate, which design features make games unlearnable?*

**Question.** Can an autonomous agent discover which design features cause games to be unlearnable for RL, validate them with matched-pair ablations, and produce a predictor plus repair rules that hold on sealed held-out designs — beating expert priors and published difficulty measures?

**Why it fits AutoMLR.** The hypotheses, the minimal-pair designs, the predictor and the repairs are the agent's; the human fixes the substrate, the learnability definition and the sealed pool. The headline is a held-out prediction number, which makes non-obviousness a measured quantity rather than an assertion.

**Qualifying result (abstract template).** "An autonomous agent proposed and tested *K* design-feature hypotheses with compute-matched matched-pair ablations (effect sizes with CIs), yielding a learnability predictor with AUROC *X* on a sealed held-out pool versus *Y* for PIC/POIC and *Z* for an LLM prior without experiments; its repair rules raised the solve rate of predicted-unlearnable held-out levels from *A*% to *B*%."

**Substrate.** XLand-MiniGrid primary [16]: rule trees as data, millions of tasks, interpretable features (rule depth, number of goals, distractors, grid size). PuzzleJAX secondary [14] for developer-authored puzzles (closer to the product vision, harder to control confounds). Learnability is inherited, not claimed: the sometimes-solvable band under a fixed budget [104], measured over seeds and a small hyperparameter grid to break the "unlearnable for default PPO" confound.

**The loop.** (1) The agent explores a discovery pool; (2) proposes hypotheses of the form "feature F raises unlearnability via mechanism M"; (3) *writes the mutation* — a DSL edit that toggles F while holding size, object count, horizon and solution length fixed; (4) runs compute-matched sweeps (many seeds, many pairs); (5) fits and validates a predictor on the sealed pool; (6) proposes repairs and tests them. Baselines for the predictor: PIC/POIC [106], optimal solution length, reward sparsity, a learnability proxy [104, 105], and an LLM prior without experiments.

**Novelty positioning.**

| Closest work | What it does | What D3 keeps |
|---|---|---|
| Tetris difficulty / Synthetic Monitoring Environments [102, 103] | Human-run difficulty and ablation studies, single families, correlational | Agent-run; generated design space; interventional minimal pairs; prediction on sealed designs; repair test |
| King / Roohi et al. / Kristensen et al. [97, 98, 99] | Human difficulty estimated from agent play | Learnability of training itself; feature-level explanation |
| OMNI-EPIC / Eurekaverse [56, 57] | Learnability as a generation filter | Learnability as the object of study |
| No Regrets / PIC / Pharos [104, 106, 107] | Measure or predict learnability/difficulty; no design-feature attribution | Attribution to authored features; validated repairs |
| CodeScientist / "AI Scientist that doesn't drift" [29, 30] | Agent-run science in virtual environments and RL | The specific validated object; the developer-facing repair rule |
| RuleSmith / Xiao & Yang [108, 109] | LLM agents balance a game or proxy human difficulty | RL learnability, generated games, causal claims |

**Risks and mitigations.** *Obviousness* — "sparse reward and long horizons are hard" is folklore; the prediction contest against priors on sealed designs, and at least one replicated counter-intuitive pair, are the answer. *Confounds* — rule complexity, horizon and solution length co-vary in generated games; genuine minimal pairs with held-fixed covariates are the hard engineering. *Label cost* — learnability labels need many seeds; only vectorized substrates make this affordable. *Recipe dependence* — a small hyperparameter grid per design point. *Faithfulness* — publish experiment cards with seeds and budgets [31, 30]. *Decisiveness* — the factor grid must visibly be the agent's. *Scoop risk* — the PuzzleJAX/ScriptDoctor group holds the substrate and publishes continuously; differentiate on causality and prediction.

**Nine-day feasibility: medium-hard.** The substrate is ready, but rigorous causal design under a deadline is the heaviest load of the three; it wants a second person.

**Paper sketch.** Part 1: hypotheses, minimal-pair effect sizes, predictor performance on the sealed pool, repair results. Part 2: substrate, learnability protocol, sealed split, agent loop and prompts, interventions, verification. Part 3: agent-run controlled experiments as the unit of autonomous science, versus benchmark optimization.

## Comparison and recommendation

| Criterion | D1 Fresh-game AutoRL | D2 Discovered testing objectives | D3 Causes of unlearnability |
|---|---|---|---|
| Agent's decisive contribution | An induced recipe and its transfer test — visible, but reads as "HPO" unless the recipe is explicit | A discovered objective (agent-authored code) plus the transfer study — the clearest artifact | Hypotheses, minimal-pair designs, predictor, repairs — decisive only if the factor grid is visibly the agent's |
| Novelty after the 2026 prior art | Partial: Agent² and LIMEN own the pipeline; the fresh-game protocol is the defensible axis | The conjunction (LLM code search × exploration objective × defect fitness) is unoccupied; held-out-class transfer is the claim | Causal + predictive + repair on a generated design space is unoccupied; the measurement instrument is not |
| Chance of a null result | Medium-high (floor effects; may not beat tuned baselines) | Medium (count-based may tie in discrete games; mitigated by Kinetix); a mixed transfer matrix is still a result | High (folklore rediscovery; confounds) |
| Engineering to build | Agent-on-SLURM harness, generator with filters, two baseline re-implementations | Defect injector, five oracles, four baselines, frozen evaluator; the loop exists | Minimal-pair mutation operators with held-fixed covariates; label pipeline |
| Compute leverage | High | Very high (fitness = many short vectorized runs) | Very high (label sweeps) |
| Fit to the auto-game-dev vision | Medium (the "build the agent" half) | High (the "find what's broken" half) | High (the "explain why the level fails" half) |
| Likely discussant attack | "This is Agent² on PuzzleScript" | "Alet 2020 plus GBQA" — answered by the held-out-class transfer | "You rediscovered sparse reward" |
| Nine-day feasibility (one person) | Medium-hard | **Medium** | Medium-hard |

> **Verdict.** D2. It is the only direction where the agent's output is an artifact a reviewer can run, where the harness already exists, where a mixed result is still a result, and where the paper's framing — *does a discovered testing objective generalize, or does objective search overfit the faults it was searched on?* — is a question none of the near-neighbours asked. It also feeds directly back into the auto-game-dev product: the discovered objective is the thing the checker loop would ship.

If the paper's identity matters more than the artifact — you want to be known for "an autonomous RL researcher" — take D1 and borrow D2's discipline (frozen evaluator, sealed split, contamination hashes). Do not attempt D3 alone before 5 September.

### Nine-day plan for D2

| Day | Work | Output that must exist by the end of the day |
|---|---|---|
| **Fri 28 Aug** | Decide. Create the OpenReview submission (title, authors, provisional abstract). Install `kinetix-env` + JAX/CUDA on Hyak; PPO throughput test on one GPU. Install ShinkaEvolve with the SLURM backend; run its shipped example end to end. Turn on hook-based logging and a git-per-experiment convention. | Throughput number; ShinkaEvolve job runs on SLURM; provenance ledger writing |
| **Sat 29 Aug** — abstract deadline | Submit the abstract (state the qualifying result as the claim under test; confirm on the portal whether edits are allowed until 5 Sep). Implement the defect injector (five classes) and oracles; calibrate on ~20 base levels; discard trivial and impossible mutants. | Injector + oracles + calibration table |
| **Sun 30 Aug** | Baselines (count-based, RND, ICM, curiosity-proximity, random, task-return PPO) with small sweeps on the search set. Generate fresh held-out levels; hash-check; freeze the split and the evaluation script (read-only for the agent). Write the agent's research brief and hypothesis prompt. | Baseline table on the search set; sealed split committed |
| **Mon 31 Aug** | Launch the discovery loop (defect fitness) and the control loop (task-return fitness). Target: hundreds of candidates × (games × classes × seeds) short PPO runs. | Both loops running; archive growing |
| **Tue 1 Sep** | Monitor; the agent writes an interim analysis; second generation if the archive stalls; distill top objectives to minimal cores. | Candidate objective(s) + distilled versions |
| **Wed 2 Sep** | Held-out evaluation through the frozen script (held-out games × held-out classes × 10 seeds). The agent builds the transfer matrix and detection curves and drafts Part 1 with figures. | Part 1 draft, figures, numbers file |
| **Thu 3 Sep** | Human verification: recompute every number in Part 1 from logs; check each figure against its data; check every citation. Write Part 2 (harness, prompts, interventions ledger, verification). Draft Part 3. | Verified Part 1; Part 2 and Part 3 drafts |
| **Fri 4 Sep** | Adversarial internal review (an LLM reviewer plus a colleague), fixes, and — if time — the oracle-blindness and complexity ablations. | Near-final PDF in the Overleaf template |
| **Sat 5 Sep** — paper deadline | Submit. | Submitted |

Register the abstract even if the numbers are placeholders: the site requires the qualifying result to be *stated* in the abstract, and the honest way to do that before the run is as a hypothesis under test.

### Abstract drafts

**D2 (primary).** "Learning-based game testers rely on hand-designed exploration objectives. We ask whether an autonomous code-evolution loop can discover a testing objective that generalizes — across games it never saw and across defect classes it was never searched on. We build a GPU-vectorized suite of Kinetix physics games with five classes of programmatically injected defects and witness-state oracles, freeze a sealed held-out split, and let an LLM-driven evolutionary loop propose intrinsic-reward programs whose fitness is expected steps-to-first-defect on the search set. **Qualifying result:** the discovered objective reduces steps-to-first-defect by *X*% relative to the best tuned hand-designed baseline on held-out games and a held-out defect class, while objectives evolved against task return transfer to *Y*% of classes; a distilled core retains *Z*% of the gain. Part 2 documents the loop, prompts, human interventions and the verification of every reported number."

**D1.** "We test whether an autonomous RL-research agent's pipeline recipe, induced on freshly generated PuzzleScript games, transfers to a sealed batch of further fresh games unseen by any model. **Qualifying result:** the agent's recipe raises fixed-budget solve rate from *X*% (default PPO) and *Y*% (published automation recipes) to *Z*%, and identifies the game properties on which every recipe fails."

**D3.** "An autonomous agent proposed and tested *K* design-feature hypotheses about RL unlearnability in generated XLand-MiniGrid games using compute-matched minimal pairs. **Qualifying result:** its learnability predictor reaches AUROC *X* on a sealed held-out design pool versus *Y* for PIC/POIC and *Z* for an LLM prior, and its repair rules raise the solve rate of predicted-unlearnable levels from *A*% to *B*%."

### Provenance and verification checklist (Part 2 material)

- Session transcripts and hook logs for every agent action; one git commit per experiment with the generating prompt in the message [4].
- The evolutionary archive and every candidate's evaluation record (ShinkaEvolve persists this natively [63]).
- A timestamped ledger of every human intervention, including "we restarted the loop" and "we changed the prompt."
- Frozen evaluator and sealed split committed *before* the loop starts, with hashes in the paper.
- Every number in Part 1 recomputed by a script from logs; every figure regenerated from data; every citation resolved to a URL — the failure modes MLR-Bench, BadScientist and the fidelity audits document are exactly these [5, 48, 31]. A background reviewer that flags untraceable numbers is the pattern Anthropic's own research tooling now ships [120].
- Immutable experiment cards recording each hypothesis and its outcome before the result is seen [30].

### Part 3 seeds (one page)

- **Decisiveness is a spectrum, not a bit.** In D2 the human designs the search space and the benchmark; the agent designs the objective and the analysis. The field needs a vocabulary for that split — and venues should ask for the ledger, not the label.
- **Oracles change the verification economics.** Where claims are checkable by execution (witness states, solver certificates), a discussant can verify in minutes what took the agent hours. Autonomous research should prefer domains with cheap oracles first.
- **Held-out splits must be adversarial to the agent, not just to the model.** Sealed evaluations, contamination hashes and frozen evaluators are the autonomous-research analogue of pre-registration.
- **Compute is now the cheap input; judgment is the scarce one.** Nine days of unlimited GPUs produced the evidence; the human hours went to deciding what was worth asking and checking what came back — which is the workshop's own thesis.

### What to decide today

1. Direction (recommendation: D2). 2. Co-authors — the reviewer obligation is min(3, #authors) and one author must be in Sydney in December. 3. Substrate (Kinetix primary for D2). 4. Whether the abstract goes in tomorrow with placeholders (yes).

## References

1. AutoMLR organizing committee. *Workshop for Autonomous Machine Learning Research (AutoMLR), NeurIPS 2026 — call for papers, format, dates and FAQ*. automlr.com, accessed 27 Aug 2026. <https://automlr.com/>
2. NeurIPS 2026. *NeurIPS 2026 Main Track Handbook — Author Use of Agents and Large Language Models*. neurips.cc, accessed 27 Aug 2026. <https://neurips.cc/Conferences/2026/MainTrackHandbook>
3. Serapio, Prakash, Xu, Wang, Greenwald. *Distilling Game Code World Model Generation into Lightweight Large Language Models*. arXiv:2605.24375, 2026. <https://arxiv.org/abs/2605.24375>
4. Anthropic. *Claude Code hooks reference (PreToolUse / PostToolUse / SessionStart / Stop events)*. docs, accessed Aug 2026. <https://code.claude.com/docs/en/hooks>
5. Chen, Xiong, Lu et al.. *MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research*. arXiv:2505.19955, 2025 (49 pages, 9 figures. Accepted by NeurIPS 2025 D&B Track). <https://arxiv.org/abs/2505.19955>
6. Kon, Liu, Zhu et al.. *EXP-Bench: Can AI Conduct AI Research Experiments?*. arXiv:2505.24785, 2025 (45 pages, 13 figures). <https://arxiv.org/abs/2505.24785>
7. Garikaparthi, Patwardhan, Cohan. *ResearchGym: Evaluating Language Model Agents on Real-World AI Research*. arXiv:2602.15112, 2026 (ICLR 2026 Agents in the Wild workshop). <https://arxiv.org/abs/2602.15112>
8. Wei, Shan, Miao et al.. *Agent^2: An Agent-Generates-Agent Framework for Reinforcement Learning Automation*. arXiv:2509.13368, 2025. <https://arxiv.org/abs/2509.13368>
9. Nathani et al. (Meta). *MLGym: A New Framework and Benchmark for Advancing AI Research Agents*. arXiv:2502.14499, 2025. <https://arxiv.org/abs/2502.14499>
10. Wijk et al. (METR). *RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents Against Human Experts*. arXiv:2411.15114, 2024. <https://arxiv.org/abs/2411.15114>
11. Karten, Appapogu, Jin. *Automatic Generation of High-Performance RL Environments (LLM loop producing GPU environments)*. arXiv:2603.12145, 2026. <https://arxiv.org/abs/2603.12145>
12. Unity Technologies. *Unity ML-Agents Toolkit*. GitHub, accessed Aug 2026 (PyPI release 1.1.0, Oct 2024). <https://github.com/Unity-Technologies/ml-agents>
13. Beeching et al.. *Godot RL Agents*. GitHub, accessed Aug 2026. <https://github.com/edbeeching/godot_rl_agents>
14. Earle, Todd, Li, Jiang, Khalifa, Nasir, Banburski-Fahey, Togelius. *PuzzleJAX: A Benchmark for Reasoning and Learning*. arXiv:2508.16821, 2025; code github.com/smearle/script-doctor. <https://arxiv.org/abs/2508.16821>
15. Matthews, Beukman, Lu, Foerster. *Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks*. ICLR 2025 (oral); arXiv:2410.23208; code github.com/FLAIROx/Kinetix. <https://arxiv.org/abs/2410.23208>
16. Nikulin et al.. *XLand-MiniGrid: Scalable Meta-Reinforcement Learning Environments in JAX*. arXiv:2312.12044; code github.com/corl-team/xland-minigrid. <https://arxiv.org/abs/2312.12044>
17. Bergdahl, Gordillo, Tollmar, Gisslén (EA SEED). *Augmenting Automated Game Testing with Deep Reinforcement Learning*. IEEE CoG 2020; arXiv:2103.15819. <https://arxiv.org/abs/2103.15819>
18. Gordillo, Bergdahl, Tollmar, Gisslén (EA SEED). *Improving Playtesting Coverage via Curiosity Driven Reinforcement Learning Agents*. IEEE CoG 2021; arXiv:2103.13798. <https://arxiv.org/abs/2103.13798>
19. Sestini, Gisslén, Bergdahl, Tollmar, Bagdanov (EA SEED). *Automated Gameplay Testing and Validation with Curiosity-Conditioned Proximal Trajectories (CCPT)*. arXiv:2202.10057, 2022. <https://arxiv.org/abs/2202.10057>
20. Fuchs, Gosselin-Grant, Skuin et al. (EA). *Reward-Adaptive Iterative Discovery: A Case Study on Automated Game Testing for NHL 26*. arXiv:2607.07498, 2026. <https://arxiv.org/abs/2607.07498>
21. Zheng, Xie, Su et al. (NetEase Fuxi). *Wuji: Automatic Online Combat Game Testing Using Evolutionary Deep Reinforcement Learning*. ASE 2019. <https://doi.org/10.1109/ASE.2019.00077>
22. Liu et al. (Microsoft). *Inspector: Pixel-Based Automated Game Testing via Exploration, Detection, and Investigation*. IEEE CoG 2022; arXiv:2207.08379. <https://arxiv.org/abs/2207.08379>
23. modl.ai. *modl.ai — AI game testing (product pages)*. accessed Aug 2026. <https://modl.ai/>
24. Jiang, Chen, Chen. *GBQA: A Game Benchmark for Evaluating LLMs as Quality Assurance Engineers*. ICLR 2026 workshop; arXiv:2604.02648. <https://arxiv.org/abs/2604.02648>
25. Lu, Lu, Lange, Foerster, Clune, Ha. *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*. arXiv:2408.06292, 2024. <https://arxiv.org/abs/2408.06292>
26. Yamada, Lange, Lu, Hu, Lu, Foerster, Clune, Ha. *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search*. arXiv:2504.08066, 2025. <https://arxiv.org/abs/2504.08066>
27. Schmidgall et al.. *Agent Laboratory: Using LLM Agents as Research Assistants*. arXiv:2501.04227, 2025. <https://arxiv.org/abs/2501.04227>
28. Mitchener et al.. *Kosmos: An AI Scientist for Autonomous Discovery*. arXiv:2511.02824, 2025. <https://arxiv.org/abs/2511.02824>
29. Jansen et al. (AI2). *CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation*. arXiv:2503.22708, 2025. <https://arxiv.org/abs/2503.22708>
30. Zhang, Zeng, Lee et al.. *An AI Scientist that Doesn't Drift: Taste, Structure, and Falsifiable Findings in a Quadruped Navigation Research Loop*. arXiv:2608.07542, 2026. <https://arxiv.org/abs/2608.07542>
31. Yu, Xu, Zhou, He. *Beyond Execution: Auditing Experimental Fidelity in LLM-Driven Scientific Research*. arXiv:2608.26753, 2026. <https://arxiv.org/abs/2608.26753>
32. Tang, Yang. *AI Research Agents Narrow Scientific Exploration*. arXiv:2605.27905, 2026. <https://arxiv.org/abs/2605.27905>
33. Eulig. *Position: Correct Answer, Wrong Mechanism -- When AI Scientists Defend General Claims Their Own Data Contradicts*. arXiv:2606.23175, 2026 (8 pages body plus 12 pages references and appendix, non-archival uploa). <https://arxiv.org/abs/2606.23175>
34. Li, Schultz, Hennes, Lanctot (Google DeepMind). *Discovering Multiagent Learning Algorithms with Large Language Models*. arXiv:2602.16928, 2026. <https://arxiv.org/abs/2602.16928>
35. Sakana AI. *The AI Scientist Generates its First Peer-Reviewed Scientific Publication*. sakana.ai blog, 2025. <https://sakana.ai/ai-scientist-first-publication/>
36. Matthews et al.. *Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning*. ICML 2024; arXiv:2402.16801. <https://arxiv.org/abs/2402.16801>
37. Pignatelli et al.. *NAVIX: Scaling MiniGrid Environments with JAX*. arXiv:2407.19396, 2024. <https://arxiv.org/abs/2407.19396>
38. Bonnet et al. (InstaDeep). *Jumanji: a Diverse Suite of Scalable Reinforcement Learning Environments in JAX*. ICLR 2024; arXiv:2306.09884. <https://arxiv.org/abs/2306.09884>
39. PufferAI. *PufferLib / Puffer Ocean environments*. GitHub, accessed Aug 2026. <https://github.com/PufferAI/PufferLib>
40. Pérez-Liébana et al.. *General Video Game AI: a Multi-Track Framework for Evaluating Agents, Games and Content Generation Algorithms*. arXiv:1802.10363, 2018. <https://arxiv.org/abs/1802.10363>
41. Hu, Zhao, Liu. *Game Generation via Large Language Models*. arXiv:2404.08706, 2024 (2024 IEEE Conference on Games). <https://arxiv.org/abs/2404.08706>
42. Piette, Soemers, Stephenson et al.. *Ludii -- The Ludemic General Game System*. arXiv:1905.05013, 2019 (Accepted at ECAI 2020). <https://arxiv.org/abs/1905.05013>
43. Todd, Padula, Soemers et al.. *Ludax: A GPU-Accelerated Domain Specific Language for Board Games*. arXiv:2506.22609, 2025 (25 pages, 6 figures). <https://arxiv.org/abs/2506.22609>
44. Bamford. *Griddly: a platform for AI research in games (YAML game description language)*. GitHub. <https://github.com/Bam4d/Griddly>
45. SIMA team (Google DeepMind). *SIMA 2: A Generalist Embodied Agent for Virtual Worlds*. arXiv:2512.04797, Dec 2025. <https://arxiv.org/abs/2512.04797>
46. Jiang, Schmidt, Srikanth et al.. *AIDE: AI-Driven Exploration in the Space of Code*. arXiv:2502.13138, 2025. <https://arxiv.org/abs/2502.13138>
47. Stanford / Together AI organizers. *Agents4Science 2025 — Open Conference of AI Agents for Science (accepted papers)*. agents4science.stanford.edu, 2025. <https://agents4science.stanford.edu/>
48. Jiang, Feng, Li, Niu et al.. *BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?*. arXiv:2510.18003, 2025; accepted at Agents4Science 2025. <https://arxiv.org/abs/2510.18003>
49. Karpathy. *autoresearch (agent-driven overnight training-script optimization loop)*. GitHub, 2026. <https://github.com/karpathy/autoresearch>
50. Chan et al. (OpenAI). *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering*. arXiv:2410.07095, 2024. <https://arxiv.org/abs/2410.07095>
51. Starace, Jaffe, Sherburn et al.. *PaperBench: Evaluating AI's Ability to Replicate AI Research*. arXiv:2504.01848, 2025 (30 pages, 14 figures). <https://arxiv.org/abs/2504.01848>
52. Meta FAIR. *AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench (AIRA-dojo)*. arXiv:2507.02554, 2025. <https://arxiv.org/abs/2507.02554>
53. Chen, Yang, Yang et al.. *Agent^2 RL-Bench: Can LLM Agents Engineer Agentic RL Post-Training?*. arXiv:2604.10547, 2026. <https://arxiv.org/abs/2604.10547>
54. Ma et al. (NVIDIA). *Eureka: Human-Level Reward Design via Coding Large Language Models*. arXiv:2310.12931, ICLR 2024. <https://arxiv.org/abs/2310.12931>
55. Jaswal, Baghel, Chopra. *LIMEN: Discovering Reinforcement Learning Interfaces with Large Language Models*. arXiv:2605.03408, 2026. <https://arxiv.org/abs/2605.03408>
56. Faldor, Zhang, Cully, Clune. *OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code*. arXiv:2405.15568, 2024. <https://arxiv.org/abs/2405.15568>
57. Liang et al.. *Eurekaverse: Environment Curriculum Generation via Large Language Models*. arXiv:2411.01775, 2024. <https://arxiv.org/abs/2411.01775>
58. Zala et al.. *EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents*. arXiv:2403.12014, 2024. <https://arxiv.org/abs/2403.12014>
59. Alet, Schneider, Lozano-Pérez, Kaelbling. *Meta-learning curiosity algorithms*. ICLR 2020; arXiv:2003.05325. <https://arxiv.org/abs/2003.05325>
60. Lu et al. (Sakana AI). *Discovering Preference Optimization Algorithms with and for Large Language Models (DiscoPOP)*. NeurIPS 2024; arXiv:2406.08414. <https://arxiv.org/abs/2406.08414>
61. Oh et al. (Google DeepMind). *Discovering state-of-the-art reinforcement learning algorithms (DiscoRL)*. Nature, 2025. <https://www.nature.com/articles/s41586-025-09761-x>
62. Novikov et al. (Google DeepMind). *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. arXiv:2506.13131, 2025. <https://arxiv.org/abs/2506.13131>
63. Lange, Imajuku, Cetin (Sakana AI). *ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution*. arXiv:2509.19349, 2025; code github.com/SakanaAI/ShinkaEvolve. <https://arxiv.org/abs/2509.19349>
64. codelion et al.. *OpenEvolve: open-source implementation of AlphaEvolve-style evolutionary coding*. GitHub. <https://github.com/codelion/openevolve>
65. Sygkounas, Loutfi, Persson. *Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models*. arXiv:2603.28416, 2026. <https://arxiv.org/abs/2603.28416>
66. Parker-Holder et al.. *Automated Reinforcement Learning (AutoRL): A Survey and Open Problems*. JAIR 2022; arXiv:2201.03916. <https://arxiv.org/abs/2201.03916>
67. Hu, Lu, Clune. *Automated Design of Agentic Systems*. arXiv:2408.08435, 2024. <https://arxiv.org/abs/2408.08435>
68. Baek, Kim, Earle et al.. *PCGRLLM: Large Language Model-Driven Reward Design for Procedural Content Generation Reinforcement Learning*. arXiv:2502.10906, 2025 (14 pages, 8 figures, Acccepted to Transactions on Games). <https://arxiv.org/abs/2502.10906>
69. Ariyurek, Betin-Can, Surer. *Automated Video Game Testing Using Synthetic and Humanlike Agents*. IEEE Transactions on Games, 2021. <https://doi.org/10.1109/TG.2019.2947597>
70. Chen, Li, Chen et al. (NetEase). *GLIB: Towards Automated Test Oracle for Graphically-Rich Applications*. ESEC/FSE 2021. <https://doi.org/10.1145/3468264.3468586>
71. Adaikkappan, Martineau, Romoff et al.. *CA2: Code-Aware Agent for Automated Game Testing*. arXiv:2605.13918, 2026. <https://arxiv.org/abs/2605.13918>
72. Taesiri, Macklon, Wang, Shen, Bezemer. *Large Language Models are Pretty Good Zero-Shot Video Game Bug Detectors*. arXiv:2210.02506, 2022. <https://arxiv.org/abs/2210.02506>
73. Taesiri, Feng, Nguyen, Bezemer. *GlitchBench: Can large multimodal models detect video game glitches?*. CVPR 2024; arXiv:2312.05291. <https://arxiv.org/abs/2312.05291>
74. Wang, Tang, Yuan et al.. *Leveraging LLM Agents for Automated Video Game Testing*. arXiv:2509.22170, 2025 (17 pages). <https://arxiv.org/abs/2509.22170>
75. Yapağcı, Öztürk, Tüzün. *Agents in the Sandbox: End-to-End Crash Bug Reproduction for Minecraft*. arXiv:2503.20036, 2025 (Accepted into ASE'25 (Automated Software Engineering). https://conf.re). <https://arxiv.org/abs/2503.20036>
76. Mu, Yoda, Zhang et al.. *Synergizing Code Coverage and Gameplay Intent: Coverage-Aware Game Playtesting with LLM-Guided Reinforcement Learning*. arXiv:2512.12706, 2025. <https://arxiv.org/abs/2512.12706>
77. He, Yang, Shi et al.. *CureFuzz: Curiosity-Driven Testing for Sequential Decision-Making Processes*. ICSE 2024; arXiv:2509.02025. <https://arxiv.org/abs/2509.02025>
78. Wilkins, Stathis. *World of Bugs: A Platform for Automated Bug Detection in 3D Video Games*. IEEE CoG 2022; arXiv:2206.11037. <https://arxiv.org/abs/2206.11037>
79. Chen, Hong, Gao, Lin et al.. *GameXpert-Bench: How Far Are Coding Agents from Expert Game Development?*. arXiv:2608.21833, 2026. <https://arxiv.org/abs/2608.21833>
80. Sohail, Jilani, Butt. *Enhancing NeuroEvolution-Based Game Testing: A Branch Coverage Approach for Scratch Programs*. arXiv:2507.09414, 2025. <https://arxiv.org/abs/2507.09414>
81. Politowski, Petrillo, Guéhéneuc. *A Survey of Video Game Testing*. arXiv:2103.06431, 2021. <https://arxiv.org/abs/2103.06431>
82. Lewis, Whitehead, Wardrip-Fruin. *What Went Wrong: A Taxonomy of Video Game Bugs*. FDG 2010. <https://dl.acm.org/doi/10.1145/1822348.1822363>
83. Tan et al.. *Cradle: Empowering Foundation Agents Towards General Computer Control*. arXiv:2403.03186, 2024. <https://arxiv.org/abs/2403.03186>
84. Xu, Wang, Li et al. (Tencent). *PORTAL: Agents Play Thousands of 3D Video Games*. arXiv:2503.13356, 2025. <https://arxiv.org/abs/2503.13356>
85. Paglieri et al.. *BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games*. ICLR 2025; arXiv:2411.13543. <https://arxiv.org/abs/2411.13543>
86. Zhang et al.. *VideoGameBench: Can Vision-Language Models complete popular video games?*. arXiv:2505.18134, 2025. <https://arxiv.org/abs/2505.18134>
87. Hu et al.. *lmgame-Bench: How Good are LLMs at Playing Games?*. arXiv:2505.15146, 2025. <https://arxiv.org/abs/2505.15146>
88. Park et al.. *Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games*. arXiv:2506.03610, 2025. <https://arxiv.org/abs/2506.03610>
89. Li, Lin, Nasir et al.. *GVGAI-LLM: Evaluating Large Language Model Agents with Infinite Games*. arXiv:2508.08501, 2025. <https://arxiv.org/abs/2508.08501>
90. Verma, Huang, Chen, Klein, Tomlin. *Measuring General Intelligence with Generated Games (gg-bench)*. arXiv:2505.07215, 2025. <https://arxiv.org/abs/2505.07215>
91. Earle et al.. *ScriptDoctor: Automatic Generation of PuzzleScript Games via Large Language Models and Tree Search*. arXiv:2506.06524, 2025. <https://arxiv.org/abs/2506.06524>
92. Earle, Togelius. *Autoverse: An Evolvable Game Language for Learning Robust Embodied Agents*. arXiv:2407.04221, 2024 (9 pages, 4 figures). <https://arxiv.org/abs/2407.04221>
93. Todd, Padula, Stephenson, Piette, Soemers, Togelius. *GAVEL: Generating Games Via Evolution and Language Models*. NeurIPS 2024; arXiv:2407.09388. <https://arxiv.org/abs/2407.09388>
94. Nasir, James, Togelius. *Word2World: Generating Stories and Worlds through Large Language Models*. arXiv:2405.06686, 2024. <https://arxiv.org/abs/2405.06686>
95. Gallotta, Todd, Zammit et al.. *Large Language Models and Games: A Survey and Roadmap*. arXiv:2402.18659, 2024 (Accepted for publication at the IEEE Transactions on Games (19 pages, ). <https://arxiv.org/abs/2402.18659>
96. Google DeepMind. *Genie 3: A New Frontier for World Models*. blog, Aug 2025. <https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/>
97. Roohi, Relas, Takatalo, Heiskanen, Hämäläinen. *Predicting Game Difficulty and Churn Without Players*. CHI PLAY 2020; arXiv:2008.12937. <https://arxiv.org/abs/2008.12937>
98. Roohi, Guckelsberger, Relas, Heiskanen, Takatalo, Hämäläinen. *Predicting Game Engagement and Difficulty Using AI Players*. CHI PLAY 2021; arXiv:2107.12061. <https://arxiv.org/abs/2107.12061>
99. Kristensen, Valdivia, Burelli (King). *Estimating Player Completion Rate in Mobile Puzzle Games Using Reinforcement Learning*. IEEE CoG 2020; arXiv:2306.14626. <https://arxiv.org/abs/2306.14626>
100. Gudmundsson, Eisen, Poromaa et al. (King). *Human-Like Playtesting with Deep Learning*. IEEE CIG 2018. <https://doi.org/10.1109/CIG.2018.8490442>
101. Tomašev, Paquet, Hassabis, Kramnik (DeepMind). *Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess*. arXiv:2009.04374, 2020. <https://arxiv.org/abs/2009.04374>
102. Wang, Guo, Guei et al.. *Evaluating Game Difficulty in Tetris Block Puzzle*. arXiv:2603.18994, 2026. <https://arxiv.org/abs/2603.18994>
103. Pleiss, Schmidt, Schiffer. *Synthetic Monitoring Environments for Reinforcement Learning*. arXiv:2603.06252, 2026. <https://arxiv.org/abs/2603.06252>
104. Rutherford et al.. *No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery (learnability / SFL)*. NeurIPS 2024; arXiv:2408.15099. <https://arxiv.org/abs/2408.15099>
105. Cho, Im, Lee et al.. *TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design*. ICLR 2026. <https://openreview.net/forum?id=J0OdFoOhSN>
106. Furuta et al.. *Policy Information Capacity: Information-Theoretic Measure for Task Complexity in Deep Reinforcement Learning*. ICML 2021; arXiv:2103.12726. <https://arxiv.org/abs/2103.12726>
107. Conserva, Sasso, Rauber. *On the Limits of Tabular Hardness Metrics for Deep RL: A Study with the Pharos Benchmark*. arXiv:2509.17092, 2025. <https://arxiv.org/abs/2509.17092>
108. Zeng, Liu, Liu et al.. *RuleSmith: Multi-Agent LLMs for Automated Game Balancing*. arXiv:2602.06232, 2026. <https://arxiv.org/abs/2602.06232>
109. Xiao, Yang. *LLMs May Not Be Human-Level Players, But They Can Be Testers: Measuring Game Difficulty with LLM Agents*. Proc. ACM HCI (CHI PLAY) 2025. <https://doi.org/10.1145/3748634>
110. Li, Li, Wang et al.. *MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences*. arXiv:2601.07251, 2026 (ACL 2026 MainConference). <https://arxiv.org/abs/2601.07251>
111. Lu, Hu, Clune. *Automated Capability Discovery via Foundation Model Self-Exploration*. arXiv:2502.07577, 2025. <https://arxiv.org/abs/2502.07577>
112. Zhang, Peng, Kong et al.. *AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning*. arXiv:2511.19304, 2025. <https://arxiv.org/abs/2511.19304>
113. Huang, Wang, Han et al.. *EnvHarness: Awakening Static Worlds for Agent Learning*. arXiv:2608.19880, 2026. <https://arxiv.org/abs/2608.19880>
114. Wu, Zhang, Liu et al.. *GIFT: LLM-Guided State-Reward Interface for Financial Reinforcement Learning*. arXiv:2606.08450, 2026 (25 pages, 7 figures. Code and data are available at https://). <https://arxiv.org/abs/2606.08450>
115. Heng, Zhao, Wu et al.. *Boosting Universal LLM Reward Design through Heuristic Reward Observation Space Evolution*. arXiv:2504.07596, 2025 (7 pages, 5 figures). <https://arxiv.org/abs/2504.07596>
116. Goldie, Wang, Hayler et al.. *DiscoGen: Procedural Generation of Algorithm Discovery Tasks in Machine Learning*. arXiv:2603.17863, 2026 (Accepted to ICML 2026). <https://arxiv.org/abs/2603.17863>
117. Gao, Li, Yuan et al.. *Camyla: Scaling Autonomous Research in Medical Image Segmentation*. arXiv:2604.10696, 2026 (Project page: https://yifangao112.github.io/camyla-page/). <https://arxiv.org/abs/2604.10696>
118. Wu, Zheng, Xiong. *LLM-ALSO: LLM-Driven Adaptive Learning-Signal Optimization for Multi-Agent Reinforcement Learning*. arXiv:2605.29293, 2026 (14 pages, 6 figures, 6 tables). <https://arxiv.org/abs/2605.29293>
119. Wang, Tang, Liu et al.. *When LLM Reward Design Fails: Diagnostic-Driven Refinement for Sparse Structured RL*. arXiv:2605.28918, 2026. <https://arxiv.org/abs/2605.28918>
120. Anthropic. *Claude Science (beta) — background reviewer and artifact provenance*. claude.com product page, accessed Aug 2026. <https://claude.com/product/claude-science>
