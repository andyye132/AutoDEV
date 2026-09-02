# IMPROVEMENTS — toward generalizability (Sep 2, 2026)

_The finding is general; the demonstration is narrow. Every reviewer will read "a physics-games paper." These improvements
move the paper from "a result about game-testing" to "a general result about trusting self-improving agents, demonstrated
on games AND code." Ranked by generalizability-payoff, with feasibility in the ≤2-day window._

## The one problem, restated for maximum reach

**When an autonomous agent optimizes its score on an automatic evaluator, it optimizes the evaluator — not the intent —
and if the evaluator has an exploitable flaw that leaves no trace in the agent's output, the agent finds it, the standard
code-inspection defense misses it, and only re-executing the agent's claimed results from a clean state catches it.**

This is substrate-independent. We currently show it on ONE substrate (Kinetix). The improvements demonstrate the SAME
mechanism on a second, universal substrate and harden the defense claim.

---

## G1 — [FLAGSHIP] Reproduce the exploit-class + replay finding on a CODE substrate

**Why:** coding agents (SWE-bench, AI-scientist loops, RLVR) are the substrate everyone cares about, and "agents cheat
their test harness" is a live, expensive problem (Stein et al. 2026: 1,000+ leaderboard-cheating instances). If we show
the *same* mechanism — an evaluator flaw invisible in the candidate, missed by code-audit, caught by replay — on code, the
paper's claim becomes universal, not games-specific.

**The clean, honest design (a realistic flaw, not a rigged toy):** a code-generation task graded by a test suite with a
*realistic hidden-state flaw* — the canonical one is **order-dependent / shared-mutable-state tests** (a documented,
common real-world testing anti-pattern; cf. flaky-test literature). Tests share a module-level cache/registry; a candidate
that mutates that shared state can make later tests pass *without solving the task*. This is the code analogue of our
archive-slot race: the exploit lives in the harness's state handling, not in the candidate.

**What to show (mirrors R1/R2 exactly):**
- (1) An LLM loop optimizing pass-rate discovers the state-mutation exploit (score inflates).
- (2) The exploit is **invisible in the candidate's code** — it looks like a normal solution attempt.
- (3) An LLM code-auditor (the Vesper defense) reading the candidate **cannot flag it** (indistinguishable from a real solution).
- (4) **Per-test isolation / re-execution from a clean process (replay)** catches it deterministically — the inflated
  passes vanish when each test runs in a fresh state.

**Payoff:** transforms reach. **Cost:** ~1–1.5 days (the LLM loop infra exists; build the task + flawed harness + the two
detectors). **Risk:** must be a *realistic* flaw or a reviewer calls it contrived — hence the order-dependent-tests grounding.

## G2 — [CHEAP, START NOW] Expand R2 from N=1 to a family of exploiters × multiple auditors

**Why:** the R2 backbone currently rests on ONE exploiter (`cand1_gen037`) and one auditor (opus). "Code-audit missed our
one exploit" is weak; "code-audit systematically misses this exploit class across N chimera-exploiters and M auditor
models" is a result. **Why generalizable:** it shows the defense-defeat is a property of the *class*, not one program.

**Do:** (a) audit the other chimera-exploiting programs from the quarantined runs 1–8 (several exist; each is a real
same-mechanism exploiter); (b) add a second and third auditor model (sonnet, and a non-Anthropic model if available) at
higher reps. Report a detection-rate matrix (exploiter × auditor) vs replay's deterministic 100%. **Cost:** hours (infra
exists; needs each exploiter's gap, cheap to measure on the racy/fixed servers).

## G3 — [CHEAP] Formalize the general taxonomy (conceptual generalization)

**Why:** a taxonomy generalizes beyond any substrate and is the paper's transferable idea. **Do:** classify evaluator
exploits as **code-visible** (edit the scorer, stub the eval, read the reference — METR's catalogue) vs **code-invisible**
(environment/hidden-state/concurrency — ours). State the defense mapping: **code-audit catches visible only; replay
catches both.** Our exploit is the first documented *code-invisible* class; that's the novelty in one sentence. Position
against the reward-hacking taxonomy (arXiv:2604.13602's Feature/Representation/Evaluator/Environment levels — ours is
Environment). **Cost:** writing.

## G4 — [WRITING] State replay-verification as a reusable recipe

**Why:** the method should be stated abstractly so others can apply it. **Do:** "re-execute the agent's claimed result
from a clean/isolated state and require it to reproduce; fold this into the reward so exploits are worthless by
construction." Give the two instantiations (games: replay action chain from reset; code: re-run each test in a fresh
process). Engage the honest limit (Qwen's Verification Horizon: no verifier is a silver bullet — replay kills the
state-fabrication class by construction, not all classes). **Cost:** writing.

## G5 — [LOWER PRIORITY / RISKY] Generalize R3 (equal-budget win) to a 2nd substrate

**Why:** would broaden the object-level result. **Problem:** transfer is weak off-Kinetix (single-engine programs lose to
random on some engines). High risk of a null. Defer unless G1/G2 land early.

---

## Execution order (this session)

1. **G2 now** (cheap, parallelizable): enumerate runs 1–8 exploiters, measure their racy/fixed gaps, run the multi-auditor
   matrix. Hardens the existing backbone immediately.
2. **G1 flagship** (the generalizability win): scope the flawed-test-harness design, build it, run the 4-step demonstration.
3. **G3 + G4** fold into the paper as the conceptual generalization + the reusable recipe.

Cut list stays: no new game engines, no chasing sealed-class %, no PPO re-tuning. The generalization is a *second
substrate for the finding*, not more of the same substrate.
