# Bug report (automatic)

Scans: runs/hyak_scan_shipped.json

## Engine anomalies (replay-verified, persistence-filtered)

Severity: A = escapes the arena / joint torn; B = deep (≥5 cm) or long (≥10 steps) penetration, or ≥10-step runaway physics; C = minor transient overlap/spike.

| Level | Anomaly | Severity | Found by (program:seed → env steps) | Duration / depth | Verified |
|---|---|---|---|---|---|
| m/h15_thrustshoot.json | OOB | A | repeat_random:0→30720, goexplore:1→40960, repeat_random:2→65280, goexplore:0→83968, repeat_random:1→147968 | 1st; 1st; 2st | True |
| m/h15_thrustshoot.json | PENETRATE | A | goexplore:2→26880, random:2→829696 | 11st/0.026m; 13st/0.099m | True |
| l/car_launch.json | PENETRATE | C | repeat_random:0→198912, goexplore:0→796160 | 0st/0.000m; 3st/0.047m | True |
| l/grasp_easy.json | PENETRATE | C | repeat_random:0→2304, repeat_random:1→2304, repeat_random:2→2304, goexplore:0→2304, goexplore:1→2304, goexplore:2→2304, random:1→4096, random:2→4352, random:0→4608 | 3st/0.024m; 3st/0.009m; 6st/0.025m | True |
| l/grasp_hard.json | PENETRATE | C | repeat_random:0→2048, repeat_random:1→2048, repeat_random:2→2048, goexplore:0→2048, goexplore:1→2048, goexplore:2→2048, random:2→5376, random:1→5632, random:0→7168 | 6st/0.023m; 5st/0.024m; 7st/0.037m | True |
| l/mjc_half_cheetah.json | PENETRATE | C | repeat_random:0→48896, goexplore:1→111616, goexplore:0→268544, goexplore:2→354816, repeat_random:2→463104 | 3st/0.009m | False |
| l/mjc_half_cheetah_easy.json | PENETRATE | C | repeat_random:2→12800, goexplore:2→58624, random:2→426496, goexplore:0→784128, repeat_random:0→800000, goexplore:1→805120 | 3st/0.008m; 3st/0.015m | False |
| l/rail.json | PENETRATE | C | goexplore:0→19200, repeat_random:2→19968, repeat_random:0→20992, random:1→21760, repeat_random:1→23040, random:0→24832, random:2→26880, goexplore:2→41216, goexplore:1→65280 | 4st/0.036m; 3st/0.022m; 4st/0.027m | True |
| m/h10_thrust_right_easy.json | BLOWUP | C | repeat_random:2→79616, repeat_random:0→233216, goexplore:2→285440, repeat_random:1→737024, goexplore:0→737280 | 3st; 4st; 3st | True |
| m/h18_thrust_right_very_easy.json | BLOWUP | C | goexplore:1→240128, repeat_random:2→390144, goexplore:0→873216, goexplore:2→1031168 | 3st; 3st; 3st | True |
| m/h3_car_thrust.json | BLOWUP | C | repeat_random:0→663040 | 4st | True |
| m/h6_thrust_over.json | BLOWUP | C | repeat_random:0→60160, repeat_random:2→83968, goexplore:1→155648, goexplore:0→159488, repeat_random:1→235264, goexplore:2→282880 | 5st; 3st; 4st | True |

Levels with at least one verified anomaly: 11; distinct (level, anomaly) pairs: 12; of which severity A/B: 2.

Distinct A/B (level, anomaly) pairs found per tester (excluding spawn-time level defects): goexplore: 2, random: 1, repeat_random: 1
