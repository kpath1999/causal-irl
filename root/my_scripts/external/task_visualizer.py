import time
import torch
import numpy as np
from absl import app, flags
from causal_world.envs import CausalWorld
from causal_world.task_generators.task import generate_task
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import VecEnv

FLAGS = flags.FLAGS
flags.DEFINE_string("task_name", "pushing", "Name of the task to visualize.")
flags.DEFINE_string("model_path", None, "Path to the PyTorch model representing the policy network.")
flags.DEFINE_float("speed", 1.0, "Playback speed of the visualization.")
flags.DEFINE_integer("max_steps", 1000, "Maximum number of simulation steps.")

def visualize_task(task_name, model_path=None, speed=1.0, max_steps=1000):


    # task = generate_task(task_generator_id=task_name)
    # env = CausalWorld(task=task, skip_frame=10, enable_visualization=True)
    # obs = env.reset()
    # print("Action Space High:", env.action_space.high)
    # print("Action Space Low:", env.action_space.low)
    # print("Observation Space Low:", env.observation_space.low)
    # print("Observation Space High:", env.observation_space.high)

    # data_recorder = DataRecorder(output_directory=f'{model_path}_pushing_episodes',
                                # rec_dumb_frequency=11)
    task = generate_task(
        task_generator_id='pushing',
        variables_space='space_a',
        fractional_reward_weight=1,
        dense_reward_weights=np.array([750, 250, 0]),
        )
    env = CausalWorld(
        task=task,
        skip_frame=3,
        enable_visualization=True,
        normalize_observations=True,
        action_mode='joint_positions',
        seed=3
        )
    # env = MovingAverageActionEnvWrapper(env)

    # policy_kwargs = dict(net_arch=[256, 256])
    model = SAC(MlpPolicy,
                env,
                verbose=1,
                )
    # model.load(model_path)
    obs = env.reset()

    print(f"Visualizing task '{task_name}' at {speed}x speed...")
    for step in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, info = model.predict(obs_tensor, deterministic=True)
        obs, reward, done, info = env.step(action[0])
        time.sleep(env.dt / speed)

        if (step + 1) % 1000 == 0 or done:
            env.reset()
            if done:
                print(f"Task completed in {step + 1} steps.")

    env.close()
    print("Visualization finished.")

def main(argv):
    visualize_task(
        task_name=FLAGS.task_name,
        model_path=FLAGS.model_path,
        speed=FLAGS.speed,
        max_steps=FLAGS.max_steps,
    )

if __name__ == "__main__":
    app.run(main)