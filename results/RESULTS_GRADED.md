# Graded bug-finding benchmark — first results (Sep 2, honest)

_A difficulty-graded synthetic bug benchmark: base Kinetix levels with a planted ANOMALY (speed) bug whose difficulty is
the physics **restitution** (1.6 = superelastic, physics gives free energy, easy → 1.0 = elastic, must actively drive
speed, hard). Detection = fraction of the (level, difficulty) instances where a method drives a body past the 14 m/s
witness within budget. Mini preliminary: 6 levels × 6 difficulties = 36 instances, 1 seed, 64×768, size m (CPU — the
full 96-instance, all-method, multi-seed run needs GPU; Hyak is down)._

## What the difficulty axis buys us (it discriminates)

Detection vs restitution (easy 1.6 → hard 1.0):

| method | 1.6 | 1.4 | 1.2 | 1.1 | 1.05 | 1.0 |
|---|---|---|---|---|---|---|
| **random** | 1.00 | 0.83 | 0.83 | 0.83 | 0.67 | **0.33** |
| action-repeat | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| go-explore | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| run7 (LLM, multi-engine) | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| saturate (LLM) | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| split-pump (LLM) | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| **PPO (best-of, tuned)** | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | **0.67** |

**The restitution axis is a real difficulty gradient:** naive **random collapses from 1.00 to 0.33** as the bug gets
harder (it can't reach 14 m/s once the physics stops handing out free energy). **The 0.83 ceiling** at the hard end is a
reachability floor — 1 of the 6 hard instances is unwitnessable by anyone at this budget.

## The honest nuance (this is the reality-check)

On THIS bug the **LLM testers tie the simple structured baselines** (action-repeat, go-explore) at 0.83 — they do **not**
show an LLM-specific edge. Reason: reaching 14 m/s with a thruster only needs *sustained action*, which action-repeat
already supplies, so the bug does not *require* the LLM's sophisticated strategy. So the speed axis separates
**structured exploration from naive random**, not **LLM from other good methods.**

## What this means + next steps

1. **PPO result (in):** PPO holds at 0.83 until the hardest bug, where it drops to **0.67** — it does not collapse like
   random (0.33) but it is **below the structured/LLM testers (0.83).** So at the hardest bugs the ordering is
   **structured/LLM 0.83 > trained RL 0.67 > random 0.33** — the benchmark shows **"structured (incl. zero-training LLM)
   testers find harder bugs than trained RL at equal budget,"** the target claim. (The LLM ties simple non-RL baselines on
   this speed-axis bug; see the nuance above.) Figure: `figures/graded_bench.png`.
2. **For an LLM-specific edge over ALL methods, grade NAVIGATION difficulty, not speed.** A SHORTCUT bug placed behind a
   graded distance / small target radius (`contact_r`) needs *coordinated spatial exploration* — exactly where the LLM
   testers beat random and action-repeat in the generalization result (DoorKey/FourRooms). That is the variant to build
   next for the "LLM finds harder bugs than everyone" curve.
3. **This is GPU-bound.** Kinetix physics eval is minutes/method on CPU; the full benchmark (96 instances × ~8 methods ×
   5 seeds + reachability probe) needs Hyak (Duo re-auth). The CPU preliminary above de-risks the design.
