from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
import numpy as np

task = generate_task(task_generator_id='general')
env = CausalWorld(task=task, enable_visualization=True)
for _ in range(10):
  env.reset()
  success_signal, obs = env.do_intervention(
          {'stage_color': np.random.uniform(0, 1, [
              3,
          ])})
  print("Intervention success signal", success_signal)
  for _ in range(100):
      obs, reward, done, info = env.step(env.action_space.sample())
env.close()