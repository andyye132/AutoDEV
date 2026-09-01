"""Does task-reward PPO learn? solved_frac should rise on an easy clean level."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import load_level
from autodev.defects import DefectMeta
from autodev.evaluate import Evaluator
from autodev.objectives import ZeroObjective
from autodev.ppo import PPOConfig
lv = sys.argv[1] if len(sys.argv) > 1 else "m/h18_thrust_right_very_easy.json"
ev = Evaluator("m", PPOConfig(num_envs=32, rollout=64, num_updates=int(sys.argv[2]) if len(sys.argv) > 2 else 40, ext_coef=1.0, int_coef=0.0))
st, sp, ep = load_level(lv, "m")
train = ev.train_fn(ZeroObjective())
t = time.time(); out = train(jax.random.split(jax.random.PRNGKey(0), 2), jax.tree.map(lambda *x: jnp.stack(x), st, st), jax.tree.map(lambda *x: jnp.stack(x), DefectMeta.clean(sp), DefectMeta.clean(sp))); jax.block_until_ready(out)
m = out["metrics"]
print(f"{lv}: {time.time()-t:.1f}s for 2x{ev.cfg.budget} steps; solved_frac {np.array(out['solved_frac']).round(3).tolist()}; episodes {np.array(out['n_episodes']).tolist()}")
print("r_ext per update (seed 0):", np.array(m["r_ext"])[0].round(3).tolist())
print("done_rate per update (seed 0):", np.array(m["done_rate"])[0].round(3).tolist())
