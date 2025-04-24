from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators import generate_task

from causal_world.loggers.data_recorder import DataRecorder
from causal_world.loggers.data_loader import DataLoader

data_recorder = DataRecorder(output_directory='pushing_episodes',
                             rec_dumb_frequency=11)
task = generate_task(task_generator_id='pushing')
env = CausalWorld(task=task,
                  enable_visualization=True,
                  data_recorder=data_recorder)

for _ in range(23):
    env.reset()
    for _ in range(50):
        env.step(env.action_space.sample())
env.close()

data = DataLoader(episode_directory='pushing_episodes')
episode = data.get_episode(14)