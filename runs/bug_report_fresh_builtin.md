# Bug report (automatic)

Scans: runs/hyak_scan_fresh.json

## Engine anomalies (replay-verified, persistence-filtered)

Severity: A = escapes the arena / joint torn; B = deep (≥5 cm) or long (≥10 steps) penetration, or ≥10-step runaway physics; C = minor transient overlap/spike.

| Level | Anomaly | Severity | Found by (program:seed → env steps) | Duration / depth | Verified |
|---|---|---|---|---|---|
| runs/hyak_fresh_levels/fresh_l_20260828_0016.json | OOB (spawn-time: level defect) | ? | random:2→122624, repeat_random:0→776192 |  | False |
| runs/hyak_fresh_levels/fresh_l_20260828_0051.json | JOINT_TEAR (spawn-time: level defect) | ? | repeat_random:0→34816 |  | False |
| runs/hyak_fresh_levels/fresh_l_20260828_0078.json | OOB (spawn-time: level defect) | ? | repeat_random:1→315904 |  | False |
| runs/hyak_fresh_levels/fresh_l_20260828_0001.json | OOB | A | random:1→72960, goexplore:0→86528, repeat_random:0→203264, goexplore:2→281088 | 1st; 1st; 2st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0010.json | OOB | A | repeat_random:2→9728, goexplore:2→9728, random:1→206336, repeat_random:0→469248, repeat_random:1→490240, random:2→1041664 | 1st; 1st; 1st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0033.json | OOB | A | repeat_random:1→64000, repeat_random:2→85248, random:0→163840, repeat_random:0→255232 | 0st; 0st; 1st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0039.json | OOB | A | random:0→193280, random:1→224512, repeat_random:2→291840, repeat_random:0→379904 | 1st; 1st | False |
| runs/hyak_fresh_levels/fresh_l_20260828_0076.json | OOB | A | repeat_random:2→14336, goexplore:0→102656, goexplore:1→577024 | 0st; 41st; 0st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0001.json | PENETRATE | B | random:2→139008, random:0→204032, goexplore:1→263680, repeat_random:1→401408 | 3st/0.054m; 3st/0.051m; 3st/0.049m | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0005.json | BLOWUP | B | repeat_random:1→9728, repeat_random:0→9984, repeat_random:2→9984, goexplore:1→17920, goexplore:0→66560, random:1→139008, random:2→140544, random:0→207104, goexplore:2→276736 | 44st; 0st; 0st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0039.json | PENETRATE | B | goexplore:0→45568 | 3st/0.135m | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0049.json | BLOWUP | B | goexplore:1→49152, goexplore:2→476672, repeat_random:2→649728 | 11st; 4st; 3st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0074.json | PENETRATE | B | repeat_random:0→11264, goexplore:1→36608, repeat_random:2→143360, repeat_random:1→341248, goexplore:0→891648, goexplore:2→927744 | 3st/0.040m; 3st/0.032m; 3st/0.078m | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0008.json | BLOWUP | C | repeat_random:2→4864, goexplore:2→7168, repeat_random:0→8448, goexplore:0→8448, repeat_random:1→24064, goexplore:1→32512 | 7st; 0st; 9st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0010.json | PENETRATE | C | goexplore:1→19712 | 0st/0.000m | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0033.json | BLOWUP | C | random:1→60928, random:2→498688 | 0st; 1st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0034.json | BLOWUP | C | goexplore:1→129536 | 3st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0051.json | PENETRATE | C | random:0→4608, random:2→4608, repeat_random:2→4864, goexplore:2→4864, goexplore:0→30208, goexplore:1→36608, random:1→70144, repeat_random:1→70144 | 3st/0.018m; 3st/0.024m; 3st/0.035m | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0076.json | BLOWUP | C | repeat_random:1→84224, repeat_random:0→145152, goexplore:2→148992 | 2st; 4st; 3st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0078.json | PENETRATE | C | repeat_random:0→118016 | 3st/0.045m | True |

Levels with at least one verified anomaly: 13; distinct (level, anomaly) pairs: 20; of which severity A/B: 10.

Distinct A/B (level, anomaly) pairs found per tester (excluding spawn-time level defects): goexplore: 5, random: 6, repeat_random: 8
