# Bug report (automatic)

Scans: runs/scan_clean_v2.json

## Engine anomalies (replay-verified, persistence-filtered)

| Level | Anomaly | Severity | Found by (program:seed → steps) | Verified |
|---|---|---|---|---|
| m/h15_thrustshoot.json | OOB | A (escapes arena) | repeat_random:0→4864, repeat_random:1→9280, goexplore:1→13248, goexplore:0→26176 | True |
| m/h18_thrust_right_very_easy.json | BLOWUP | B (persistent runaway physics) | goexplore:0→11584 | True |
| m/h6_thrust_over.json | BLOWUP | B (persistent runaway physics) | repeat_random:0→4928, goexplore:0→56384, repeat_random:1→60928, goexplore:1→75904 | True |

Levels with at least one verified anomaly: 3; distinct (level, anomaly) pairs: 3.

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
