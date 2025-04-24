from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
# from causal_world.wrappers.action_wrappers import DeltaActionEnvWrapper
from causal_world.wrappers.action_wrappers import MovingAverageActionEnvWrapper

task = generate_task(task_generator_id='pushing')
env = CausalWorld(
  task=task,
  action_mode='joint_torques',
  enable_visualization=True)
# env = DeltaActionEnvWrapper(env)
env = MovingAverageActionEnvWrapper(env)
print(env.action_space)
print(env.observation_space)

for _ in range(1):
  env.reset()
  for _ in range(20000):
      
      obs, reward, done, info = env.step((env.action_space.sample()*0))
env.close()