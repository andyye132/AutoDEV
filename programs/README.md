# Programs referred to in the paper

Each file is an unedited copy; the original path is given so results can be traced back to it.

| File | What it is | Original path |
|---|---|---|
| `best_program.py` | The evolved program evaluated in Figure 2 (Section 1.3, Appendix A) | `runs/evolve_run7_multi/candidates/run7_best_gen035_0.5888.py` |
| `base_go_explore.py` | The Go-Explore base program every search started from | `evolve/initial.py` |
| `kinetix_only_search_best.py` | Best program of the Kinetix-only search (Table 2, "Evolved program") | `runs/evolve_run9_fixed/candidates/run9_best_gen028_0.5883.py` |
| `doorkey_only_search_best.py` | Best program of the shorter DoorKey-only search (Section 1.5) | `runs/evolve_fresh_doorkey/best/main.py` |
| `split_and_pump.py` | Program hand-written by the driver agent (Table 2) | `evolve/methods/split_pump.py` |

All 36 programs the main search wrote, with their scores, parents and the model that wrote each one, are in
`runs/evolve_run7_multi/programs.sqlite` (table `programs`).
