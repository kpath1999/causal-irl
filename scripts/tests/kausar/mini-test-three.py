from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from stable_baselines import PPO2
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv


def _make_env(rank):
    def _init():
        task = generate_task(task_generator_id='pushing')
        env = CausalWorld(task=task)
        return env
    return _init

policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
env = SubprocVecEnv([_make_env(rank=i) for i in range(10)])
model = PPO2(MlpPolicy,
             env,
             policy_kwargs=policy_kwargs,
             verbose=1)
model.learn(total_timesteps=1000000)