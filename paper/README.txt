AutoMLR @ NeurIPS 2026 — submission package
============================================

WHAT THIS IS
  "Write the Tester, Don't Train It" — a zero-training exploration program the loop
  wrote out-reaches tuned PPO on sparse-reward game testing at an equal interaction
  budget; training catches up only at 4-16x budget or on reward-dense tasks.

HOW TO BUILD
  Single file:   latexmk -pdf submission.tex        (self-contained, 8 pp)
  Modular:       latexmk -pdf main.tex              (uses \input of part1/2/3)
  Pre-built PDF: PAPER_compiled.pdf

FILES
  submission.tex ......... flattened single file (main + Parts 1/2/3 inlined)
  main.tex ............... title + abstract + wiring (\input part1/2/3, \bibliography)
  part1.tex ............. Part 1, agent-written research result (<=4 pp)
  part2.tex ............. Part 2, human-written system account (SKELETON — voice this)
  part2_prose_reference.tex  fuller prose draft of the exploit/verification story
  part3.tex ............. Part 3, reflections (<=1 pp)
  refs.bib .............. 10 web-verified references (source URLs in comments)
  figs/fig1.pdf ......... headline: equal-budget bars (A) + budget crossover (B)
  figs/pipeline.pdf ..... method diagram (editable source: figs/pipeline.tex)
  figs/make_fig1.py ..... regenerates fig1 (run from repo ROOT, not paper/)

STILL TO DO BEFORE SUBMIT (yours)
  1. Voice Part 2 in your own words (the venue weights it most; the exploit story is
     firsthand — fill the [BRACKETS]: proposal model cost cap, generation count,
     check-in count).
  2. Drop into the official AutoMLR Overleaf template; confirm each Part fits its limit.
  3. OpenReview: replace the registered abstract (the new one is in main.tex), name
     min(3, #authors) reviewers, answer the attendance question, upload the PDF.

STATUS
  0 em-dashes in rendered prose; no internal run-number labels; every Part-1 number
  verified against regenerable logs. Claim is the write-vs-train regime boundary,
  which concedes at the thesis level, so "RL catches up at 16x" is a confirmation.
