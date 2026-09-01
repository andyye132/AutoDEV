# Bug report (automatic)

Scans: runs/hyak_scan_shipped.json,runs/hyak_scan_fresh.json

## Engine anomalies (replay-verified, persistence-filtered)

Severity: A = escapes the arena / joint torn; B = deep (≥5 cm) or long (≥10 steps) penetration, or ≥10-step runaway physics; C = minor transient overlap/spike.

| Level | Anomaly | Severity | Found by (program:seed → env steps) | Duration / depth | Verified |
|---|---|---|---|---|---|
| runs/hyak_fresh_levels/fresh_l_20260828_0016.json | OOB | ? | random:2→122624, repeat_random:0→776192 |  | False |
| runs/hyak_fresh_levels/fresh_l_20260828_0051.json | JOINT_TEAR | ? | repeat_random:0→34816 |  | False |
| runs/hyak_fresh_levels/fresh_l_20260828_0078.json | OOB | ? | repeat_random:1→315904 |  | False |
| m/h15_thrustshoot.json | OOB | A | repeat_random:0→30720, goexplore:1→40960, repeat_random:2→65280, goexplore:0→83968, repeat_random:1→147968 | 1st; 1st; 2st | True |
| m/h15_thrustshoot.json | PENETRATE | A | goexplore:2→26880, random:2→829696 | 11st/0.026m; 13st/0.099m | True |
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
| runs/hyak_fresh_levels/fresh_l_20260828_0008.json | BLOWUP | C | repeat_random:2→4864, goexplore:2→7168, repeat_random:0→8448, goexplore:0→8448, repeat_random:1→24064, goexplore:1→32512 | 7st; 0st; 9st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0010.json | PENETRATE | C | goexplore:1→19712 | 0st/0.000m | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0033.json | BLOWUP | C | random:1→60928, random:2→498688 | 0st; 1st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0034.json | BLOWUP | C | goexplore:1→129536 | 3st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0051.json | PENETRATE | C | random:0→4608, random:2→4608, repeat_random:2→4864, goexplore:2→4864, goexplore:0→30208, goexplore:1→36608, random:1→70144, repeat_random:1→70144 | 3st/0.018m; 3st/0.024m; 3st/0.035m | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0076.json | BLOWUP | C | repeat_random:1→84224, repeat_random:0→145152, goexplore:2→148992 | 2st; 4st; 3st | True |
| runs/hyak_fresh_levels/fresh_l_20260828_0078.json | PENETRATE | C | repeat_random:0→118016 | 3st/0.045m | True |

Levels with at least one verified anomaly: 24; distinct (level, anomaly) pairs: 32; of which severity A/B: 12.

Distinct A/B (level, anomaly) pairs found per tester (excluding spawn-time level defects): goexplore: 7, random: 7, repeat_random: 9

## Design checks (automatic)

| Level | No-input outcome | Random-play win rate | Median win step | Flags |
|---|---|---|---|---|
| s/h0_weak_thrust.json | dies | 0.00 | None | NOOP_DEATH |
| s/h1_thrust_over_ball.json | survives | 0.00 | None |  |
| s/h2_one_wheel_car.json | survives | 0.00 | None |  |
| s/h3_point_the_thruster.json | survives | 0.00 | None |  |
| s/h4_thrust_aim.json | dies | 0.10 | 25.0 | NOOP_DEATH |
| s/h5_rotate_fall.json | dies | 0.62 | 26.0 | NOOP_DEATH, TRIVIAL_WIN |
| s/h6_unicycle_right.json | survives | 0.00 | None |  |
| s/h7_unicycle_left.json | survives | 0.00 | None |  |
| s/h8_unicycle_balance.json | dies | 0.13 | 25.0 | NOOP_DEATH |
| s/h9_explode_then_thrust_over.json | survives | 0.00 | None |  |
| m/arm_hard.json | survives | 0.01 | 120.0 |  |
| m/arm_left.json | survives | 0.00 | None |  |
| m/arm_right.json | survives | 0.00 | None |  |
| m/arm_up.json | dies | 0.00 | None | NOOP_DEATH |
| m/h0_unicycle.json | dies | 0.00 | None | NOOP_DEATH |
| m/h10_thrust_right_easy.json | survives | 0.00 | None |  |
| m/h11_thrust_left_easy.json | survives | 0.00 | None |  |
| m/h12_thrustfall_left.json | survives | 0.00 | None |  |
| m/h13_thrustfall_right.json | survives | 0.00 | None |  |
| m/h14_thrustblock.json | dies | 0.00 | None | NOOP_DEATH |
| m/h15_thrustshoot.json | survives | 0.29 | 172.0 |  |
| m/h16_thrustcontrol_right.json | survives | 0.00 | None |  |
| m/h17_thrustcontrol_left.json | survives | 0.00 | None |  |
| m/h18_thrust_right_very_easy.json | survives | 0.00 | None |  |
| m/h19_thrust_left_very_easy.json | survives | 0.00 | None |  |
| m/h1_car_left.json | survives | 0.31 | 128.0 |  |
| m/h2_car_right.json | survives | 0.39 | 127.5 |  |
| m/h3_car_thrust.json | survives | 0.00 | None |  |
| m/h4_thrust_the_needle.json | dies | 0.00 | None | NOOP_DEATH |
| m/h5_angry_birds.json | survives | 0.08 | 51.5 |  |
| m/h6_thrust_over.json | survives | 0.00 | None |  |
| m/h7_car_flip.json | survives | 0.00 | None |  |
| m/h8_weird_vehicle.json | survives | 0.00 | None |  |
| m/h9_spin_the_right_way.json | dies | 0.09 | 44.0 | NOOP_DEATH |

NOOP_DEATH = the level is lost with no input (usually the intended challenge, reported as a statistic); TRIVIAL_WIN = random play wins quickly (design flaw candidate).
