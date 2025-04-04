import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import app, flags
import os
import datetime

from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PUSHING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis

from algo.sac.sac import SAC
# from stable_baselines3.sac.policies import MlpPolicy
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, 'Train the model.')
flags.DEFINE_bool('eval', False, 'Evaluate the model.')
flags.DEFINE_integer('num_envs', 10, 'Number of envs.')
flags.DEFINE_string('task', 'pushing', 'Task selected.')
flags.DEFINE_string('load_dir', None, 'Load dir.')

def _make_env(task_id, task_settings, world_settings, seed=0, log_dir='test', vis=False):
    def _init():
        task = generate_task(
            task_generator_id=task_id,
            **task_settings
            )
        env = CausalWorld(
            task=task,
            **world_settings,
            seed=seed
            )
        return env
    return _init

def init(task, algorithm_settings, policy_kwargs, task_settings,
          world_settings, seed=0, num_envs=10, log_dir='test', vis=False):
    env = _make_env(task, task_settings, world_settings, log_dir=log_dir, vis=vis)()
    model = SAC(env,
                _init_setup_model=True,
                policy_kwargs=policy_kwargs,
                gradient_steps=num_envs,
                **algorithm_settings['train_configs'],
                verbose=1,
                )
    return model

def train(algorithm_config, task_config, world_config, network_settings, task='pushing', n_envs=10, seed=0):
    validate_every_timesteps = algorithm_config['validate_every_timesteps']
    ckpt_frequency = max(validate_every_timesteps // n_envs, 1)
    total_time_steps = algorithm_config['total_time_steps']
    
    save_dir = f'sac_{task}_{n_envs}_{total_time_steps}_mva_real_{datetime.datetime.now().ctime()}'
    # new_logger = configure(save_dir, ['stdout', "csv"])

    # if not os.path.exists(os.path.join(save_dir, 'logs')):
    #     os.makedirs(os.path.join(save_dir, 'logs'))

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=ckpt_frequency,
    #     save_path=os.path.join(save_dir, './logs/'),
    #     name_prefix='model')
    
    # callback = CallbackList([checkpoint_callback, 
    #                         #  eval_callback
    #                          ])

    model = init(task, algorithm_config, network_settings, task_config, world_config, seed=seed, num_envs=n_envs, log_dir=save_dir)

    # model.set_logger(new_logger)
    model.train(total_time_steps)
    model.save(os.path.join(save_dir, 'model'))

def eval(load_dir):
    df = pd.read_csv(os.path.join(load_dir,'progress.csv'))
    print(df.head())
    print(df.keys())
    plt.plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'])
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward Mean')
    plt.title('Training Progress')
    plt.show()

def main(argv):
    del argv

    ppo = {'num_of_envs': 20,
           'algorithm': 'PPO',
           'validate_every_timesteps': int(2000000),
           'total_time_steps': int(100000000),
           'train_configs': {
               "gamma": 0.99,
               "n_steps": int(120000 / 20),
               "ent_coef": 0.01,
               "learning_rate": 0.00025,
               "vf_coef": 0.5,
               "max_grad_norm": 0.5,
               "nminibatches": 40,
               "noptepochs": 4
           }}

    sac = {'num_of_envs': 1,
           'algorithm': 'SAC',
           'validate_every_timesteps': int(500000),
           'total_time_steps': int(10_000_000),
           'train_configs': {
               "gamma": 0.95,
               "tau": 1e-3,
               "ent_coef": 1e-3,
               "target_entropy": 'auto',
               "learning_rate":  1e-4,
               "buffer_size": 1000000,
               "learning_starts": 1000,
               "batch_size": 256
           }}

    task_configs = {
            'variables_space': 'space_a',
            'fractional_reward_weight': 1,
            'dense_reward_weights': [750, 250, 0]
        }

    world_params = {
            'skip_frame': 3,
            'enable_visualization': False,
            'observation_mode': 'structured',
            'normalize_observations': True,
            'action_mode': 'joint_positions'
        }

    net_layers = dict(net_arch=[256, 256])
    world_seed = 0
   
    if FLAGS.train:
        train(
            algorithm_config=sac,
            task_config=task_configs,
            world_config=world_params,
            network_settings=net_layers,
            n_envs=FLAGS.num_envs,
            seed=world_seed
        )
    if FLAGS.eval:
        eval(load_dir=FLAGS.load_dir)
        # gen_eval(FLAGS.load_dir)
        
if __name__ == '__main__':
    app.run(main)