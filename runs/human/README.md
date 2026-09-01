# Human anchor session — player `PLAYER`

Nine planted-defect levels, 5 minutes each (or until the defect is witnessed). The same witness oracle the automated testers use runs every step and logs when you trigger it.
Controls: arrows/WASD = motors 1–2 (±), Q/E = motors 3–4, Z/C = motors 5–6, 1/2 = thrusters, R = reset the level, ESC = give up. No hints beyond the class description.

```
cd /Users/andy132/projects/research/ACT/AutoDev && . .venv/bin/activate
```

## ANOMALY: `l_h11_obstacle_avoidance__ANOMALY__0`
Some shapes are super-bouncy. Make something move unreasonably fast.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_h11_obstacle_avoidance__ANOMALY__0.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## ANOMALY: `l_hard_cartpole_thrust__ANOMALY__0`
Some shapes are super-bouncy. Make something move unreasonably fast.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_hard_cartpole_thrust__ANOMALY__0.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## DISARMED: `l_h7_maze_1__DISARMED__1`
One red hazard is harmless. Touch a red shape with a green one and survive.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_h7_maze_1__DISARMED__1.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## DISARMED: `m_h16_thrustcontrol_right__DISARMED__0`
One red hazard is harmless. Touch a red shape with a green one and survive.

```
python scripts/play_instance.py --instance runs/hyak/instances/m_h16_thrustcontrol_right__DISARMED__0.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## SHORTCUT: `l_car_swing_around__SHORTCUT__1`
A neutral (grey) shape now counts as the goal. Win the level by touching the wrong shape, away from the real goal.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_car_swing_around__SHORTCUT__1.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## SHORTCUT: `l_mjc_walker_easy__SHORTCUT__0`
A neutral (grey) shape now counts as the goal. Win the level by touching the wrong shape, away from the real goal.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_mjc_walker_easy__SHORTCUT__0.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## TUNNEL: `l_grasp_hard__TUNNEL__0`
One static obstacle is no longer solid. Find it: get a moving shape's centre inside a solid-looking obstacle.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_grasp_hard__TUNNEL__0.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## OOB: `l_mjc_hopper_easy__OOB__0`
One boundary wall of the arena has been removed. Get any moving shape out of the 5 m arena.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_mjc_hopper_easy__OOB__0.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

## OOB: `l_mjc_hopper_easy__OOB__1`
One boundary wall of the arena has been removed. Get any moving shape out of the 5 m arena.

```
python scripts/play_instance.py --instance runs/hyak/instances/l_mjc_hopper_easy__OOB__1.json --meta runs/hyak/pool.json --size l --budget 2048 --minutes 5.0 --player PLAYER --log runs/human/PLAYER.jsonl
```

When done, send `runs/human/<player>.jsonl`. Each line: instance, class, steps, resets, wall_s, first_witness_step (null if not found), reason.