# Leave-one-engine-out transfer matrix (push #1)

Five engine "families" available as substrates: Kinetix (physics), Gymnax (control), MiniGrid (gridworld), Breakout (arcade), Sokoban (puzzle), Craftax (survival).
Run 7 already = trained on {Kinetix, Craftax, Gymnax}, tested on held-out {MiniGrid, Breakout, Sokoban} -> 100/100/79%. That's ONE cell of the matrix.

Rigorous version: 3-4 additional multi-engine searches, each HOLDING OUT one family from the fitness, then evaluating on the held-out family.
Each search: reuse eval_server_multi with a SUBSTRATES string that omits the held-out engine; 40 gens; then eval the best on the held-out engine's suite.

Held-out configs (fitness engines | held-out test engine):
 M1: {Kinetix, Gymnax, Breakout, Sokoban}       | MiniGrid   (does breadth still buy MiniGrid transfer without Craftax?)
 M2: {Kinetix, Gymnax, MiniGrid, Sokoban}       | Breakout
 M3: {Kinetix, Gymnax, MiniGrid, Breakout}      | Sokoban
 M4: {Gymnax, MiniGrid, Breakout, Sokoban}      | Kinetix    (the hardest: can non-physics engines produce a Kinetix tester?)
Control per cell: a SINGLE-engine search on Kinetix only (=split_pump lineage) tested on the held-out engine (expected: collapse).

Result table: rows = held-out engine, cols = {multi-engine trained (this matrix), single-engine trained (collapse), uniform random floor}.
The claim it proves: "held-out-engine transfer scales with the number of engines in the fitness, and requires >1."

GPU budget: 4 searches x 3 servers each is too much at once (cap 6). Run them SEQUENTIALLY, 3 servers each with racing -> ~1h each -> 4h for the matrix. Or 2 concurrent searches x 3 servers.
Racing (rung-1 gate) applies only to single-engine Kinetix fitness; multi-engine searches run without racing (full multi-substrate eval each).
