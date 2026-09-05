# Morning execution plan (Sep 4) — you're ~1.5 days from the deadline

**Deadline:** Sep 5 AOE = **~5:00 AM PDT on Sep 6**. Comfortable, not rushed.

## Where the paper stands (done overnight)
The paper is **honest, confident, de-AI'd, and compiling** (`paper/main.pdf`, 8pp: Part 1 ~4pp, Part 2 ~2pp, Part 3 ~0.7pp,
refs ~1pp). Overnight I ran, in order:
1. **Reframe verification** — a hostile 6-lens adversarial pass caught the original "one program beats RL on 8" as a
   cherry-pick; a 3-way reframe panel + judge chose the honest spine; I executed it and **re-verified** with a fresh
   hostile pass that returned "minor-fixes-left, zero criticals," then cleared the minors.
2. **References** — all 10 web-verified against real sources (no fabrication); `refs.bib` carries only verified entries
   with source URLs in comments.
3. **Confidence pass** — put the ownership back (you were right that I'd over-corrected); every claim stays true.
4. **De-em-dash + human-voice pass** — **52 em-dashes → 0**, AI-tells removed, prose rewritten to read like a person.

The claim is now the *write-vs-train regime* boundary, which pre-concedes at the thesis level, so a reviewer's attacks
("RL catches up at 16x", "you picked 4 tasks") become confirmations rather than kills. It's a strong, defensible paper.

## What to do, in order

1. **[30 min] Read `main.pdf` end to end**, out loud if you can. Fix anything that doesn't sound like *you*. It's honest
   and confident now; make sure the voice is yours.

2. **[1–2 h] Voice Part 2 (`part2.tex`) — this is the one that must be human-written.** It's an accurate, complete
   scaffold, but the venue weights it most, and its power is *firsthand*. Rewrite §The-exploit and §Human-interventions in
   your own words: what it felt like to see the too-good number, how you diagnosed the race, the decision to throw it out.
   That narrative is your strongest card. Keep the facts (+0.11, 62/256 chimeras, passed every check, replay caught it,
   N=3→N=1).

3. **[30 min] Drop into the official AutoMLR Overleaf template** (https://www.overleaf.com/read/bbxvpfghmbvb). Upload
   `part1/2/3.tex`, `refs.bib`, and `figs/` (`fig1.pdf`, `pipeline.pdf`); swap the `\documentclass`/preamble for theirs;
   keep our tikz/color/pifont/natbib preamble lines. Recompile and check each Part fits its limit (Part 1 ≤4, Part 2 ≤4,
   Part 3 ≤1). Our loose `11pt/1in` article runs a touch longer than their denser template, so it should tighten.

4. **[20 min] OpenReview logistics.** (a) **Replace the registered abstract** — `openreview_abstract_v3.txt` is the OLD
   overclaim; paste the new one from `main.tex`. (b) Designate min(3, #authors) reviewers (reciprocal reviewing). (c)
   Answer the physical-attendance question. (d) Upload the PDF.

5. **[final] One consistency read**, then submit.

## Optional (not load-bearing — only if you want the extra rigor)
- **5-seed K-Defect PPO** (Hyak is back). The Kinetix *depth* comparison uses 3-seed PPO; the paper already discloses this
  and now headlines the *apparatus*, not the 0.655 margin, so it isn't load-bearing. If you want it: `eval_ppo.sbatch`
  **hardcodes `--seeds 0,1,2`** (line 15) — patch it to take `SEEDS=100:101:102:103:104` before submitting, or it runs the
  wrong seeds. I left this for you rather than fire a bad GPU job overnight.

## Key files
- `paper/main.tex` (title + abstract + wiring), `part1.tex`, `part2.tex`, `part3.tex`, `refs.bib`
- `paper/figs/`: `fig1.pdf` (headline, two-panel), `pipeline.pdf` (method; editable source `pipeline.tex`)
- `paper/PAPER_PLAN.md` (full plan), this file, and `results/OVERNIGHT_FINDINGS_SEP3.md` (the honest data record)
- Backups of the pre-reframe drafts: `part1_exploitera_backup.tex`, `part2_old_backup.tex`, `part3_old_backup.tex`

## The one-line status
It's a good paper that survived its own adversarial review and now reads with earned confidence. The remaining work is
yours to own: voice Part 2, template it, submit.
