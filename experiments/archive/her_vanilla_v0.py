import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import app, flags
import os
import datetime
import json
import argparse
import time  # Added for timing measurements
import tensorflow as tf
import gym
print("Gym version:", gym.__version__)

from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PUSHING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis

from stable_baselines import HER, SAC
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import logger
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.common import set_global_seeds
from causal_world.wrappers.action_wrappers import MovingAverageActionEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', True, 'Train the model.')
flags.DEFINE_bool('eval', False, 'Evaluate the model.')
flags.DEFINE_float('timesteps', 1e6, 'Total timesteps.')    # 1M timesteps
flags.DEFINE_integer('num_envs', 1, 'Number of envs.')
flags.DEFINE_string('task', 'pushing', 'Task selected.')

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"rl_model_{self.num_timesteps}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True

class SimpleEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_path=None, verbose=1):
        super(SimpleEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            obs = self.eval_env.reset()
            for _ in range(10):  # run 10 episodes
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    total_reward += reward
                episode_rewards.append(total_reward)
                obs = self.eval_env.reset()

            mean_reward = np.mean(episode_rewards)
            if self.verbose > 0:
                print(f"[Eval] Step: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = os.path.join(self.log_path or ".", "best_model")
                os.makedirs(best_path, exist_ok=True)
                self.model.save(os.path.join(best_path, "best_model"))
                if self.verbose > 0:
                    print(f"New best model saved at {best_path}")
        return True
    
def make_env(rank, task_name, seed_num, skip_frame, max_episode_length):
    def _init():
        task = generate_task(task_generator_id=task_name)
        env = CausalWorld(
            task=task,
            skip_frame=skip_frame,
            enable_visualization=False,
            seed=seed_num + rank,
            max_episode_length=max_episode_length
        )
        return HERGoalEnvWrapper(env)
    set_global_seeds(seed_num)
    return _init


def save_config_file(config, env, file_path):
    task_config = env.get_task().get_task_params()
    env_config = env.get_world_params()
    # Serialize
    for k, v in task_config.items():
        if not isinstance(v, str):
            task_config[k] = str(v)
    for k, v in env_config.items():
        if not isinstance(v, str):
            env_config[k] = str(v)
    env.close()
    with open(file_path, 'w') as fout:
        json.dump([task_config, env_config, config], fout, indent=4)


def train_policy(args):
    os.makedirs(args['log_relative_path'], exist_ok=True)
    
    # Create VecEnv with HER-wrapped CausalWorld envs
    env = SubprocVecEnv([
        make_env(rank=i,
                 task_name=args['task_name'],
                 seed_num=args['seed_num'],
                 skip_frame=args['skip_frame'],
                 max_episode_length=args['max_episode_length'])
        for i in range(args['num_of_envs'])
    ])

    # Define HER + SAC
    model = HER(
        'MlpPolicy',
        env,
        SAC,
        verbose=1,
        policy_kwargs=dict(layers=[256, 256, 256]),
        **args['sac_config']
    )

    # Save config
    save_config_file(args['sac_config'], make_env(0, args['task_name'], args['seed_num'], args['skip_frame'], args['max_episode_length'])(), os.path.join(args['log_relative_path'], 'config.json'))

    # Training loop
    for i in range(int(args['total_timesteps'] / args['validate_every_timesteps'])):
        model.learn(total_timesteps=args['validate_every_timesteps'], tb_log_name="her_sac", reset_num_timesteps=False)

    model.save(os.path.join(args['log_relative_path'], 'saved_model'))
    env.close()