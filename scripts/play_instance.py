"""Human-anchor harness (C5.5, Ariyurek's template): a person plays a planted-defect instance with the keyboard while the SAME
witness oracle the testers use runs every step; the log records simulator steps and wall-clock to the first witness (or none
within the step budget). Ten instances (two per class), 5 minutes each, is the intended session.

    python scripts/play_instance.py --instance runs/hyak/instances/<name>.json --meta runs/hyak/pool.json --budget 2048 --log runs/human/<player>.jsonl
    python scripts/play_instance.py --instance ... --meta ... --scripted 200        # dry run without a window (random actions)

Keys: arrows/WASD = motor bindings 1-2 (+/-), Q/E = motors 3-4, SPACE = no-op, 1/2 = thrusters, R = reset, ESC = quit.
The instance JSON is the mutated level; `--meta` is the pool file that holds its DefectMeta (looked up by file name).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from autodev.kinetix_utils import make_env
from autodev.evaluate import load_instance_level, meta_from_json
from autodev.oracles import witness

ap = argparse.ArgumentParser()
ap.add_argument("--instance", required=True); ap.add_argument("--meta", default="runs/hyak/pool.json"); ap.add_argument("--size", default="l")
ap.add_argument("--budget", type=int, default=2048, help="simulator steps allowed (one human episode budget)"); ap.add_argument("--minutes", type=float, default=5.0)
ap.add_argument("--log", default="runs/human/session.jsonl"); ap.add_argument("--player", default="anon"); ap.add_argument("--scripted", type=int, default=0, help="dry run: N random actions, no window")
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
name = os.path.basename(a.instance).replace(".json", "")
pool = json.load(open(a.meta)); entry = next((p for p in pool["instances"] if p["name"] == name), None)
if entry is None: sys.exit(f"instance {name} not found in {a.meta}")
st, sp, ep = load_instance_level(a.instance, a.size); env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
meta = meta_from_json(entry["meta"]); cls = entry["cls"]
step = jax.jit(lambda s, act: env.step(jax.random.PRNGKey(0), s, jnp.int32(act), ep, st))
oracle = jax.jit(lambda s, r, d: witness(s, r, d, meta, engine))
n_act = env.action_space(ep).n; noop = int(sp.num_motor_bindings) * 2
obs, s = env.reset(jax.random.PRNGKey(0), ep, st)
os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
t0 = time.time(); steps = 0; first = None; resets = 0


def record(reason):
    row = dict(player=a.player, instance=name, cls=cls, steps=steps, resets=resets, wall_s=time.time() - t0, first_witness_step=first, reason=reason, budget=a.budget)
    with open(a.log, "a") as f: f.write(json.dumps(row) + "\n")
    print("LOGGED:", row)


if a.scripted:
    rng = np.random.default_rng(0)
    for _ in range(a.scripted):
        obs, s, r, d, info = step(s, int(rng.integers(0, n_act))); steps += 1
        if first is None and bool(oracle(s, r, d)): first = steps
        if bool(d): obs, s = env.reset(jax.random.PRNGKey(0), ep, st); resets += 1
    record("scripted"); sys.exit(0)

import pygame
from kinetix.render.renderer_pixels import make_render_pixels
render = jax.jit(make_render_pixels(ep, sp))
pygame.init(); H, W = 600, 600; screen = pygame.display.set_mode((W, H)); pygame.display.set_caption(f"AutoDev human anchor — {cls} — {name}")
clock = pygame.time.Clock(); font = pygame.font.SysFont(None, 22)
KEYS = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3, pygame.K_w: 0, pygame.K_s: 1, pygame.K_a: 2, pygame.K_d: 3,
        pygame.K_q: 4, pygame.K_e: 5, pygame.K_z: 6, pygame.K_c: 7, pygame.K_1: noop + 1, pygame.K_2: noop + 2}
running = True; last = "no-op"
while running and steps < a.budget and (time.time() - t0) < a.minutes * 60 and first is None:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE): running = False
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r: obs, s = env.reset(jax.random.PRNGKey(0), ep, st); resets += 1
    pressed = pygame.key.get_pressed(); act = noop
    for k, v in KEYS.items():
        if pressed[k] and v < n_act: act = v; last = pygame.key.name(k); break
    obs, s, r, d, info = step(s, act); steps += 1
    if bool(oracle(s, r, d)): first = steps
    if bool(d): obs, s = env.reset(jax.random.PRNGKey(0), ep, st); resets += 1
    img = np.array(render(s)); img = np.clip(img, 0, 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(np.transpose(img[::-1], (1, 0, 2))) if img.ndim == 3 else pygame.Surface((W, H))
    screen.blit(pygame.transform.scale(surf, (W, H)), (0, 0))
    screen.blit(font.render(f"{cls}  step {steps}/{a.budget}  {int(time.time()-t0)}s  key {last}", True, (255, 255, 255)), (8, 8))
    pygame.display.flip(); clock.tick(a.fps)
record("witness" if first is not None else ("quit" if not running else "budget/time"))
pygame.quit()
