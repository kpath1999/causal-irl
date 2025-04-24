"""
This tutorial shows you how to train a policy using stable baselines with PPO
"""
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import json
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import BaseCallback
import argparse
import numpy as np
import csv

class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        self.log_path = os.path.join(log_dir, 'training_metrics.csv')
        # AttributeError: 'CausalWorld' object has no attribute 'fractional_success'
        self.fieldnames = [
            "timesteps",
            "reward_mean",
            "fractional_success",
            "value_loss",
            "policy_loss",
            "explained_variance"
        ]
        self.fieldnames.extend([
            'goal_position_x',
            'goal_position_y',
            'robot_joint_positions'
        ])
    
    def _on_training_start(self):
        with open(self.log_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def _on_step(self):
        # log every 1000 steps (adjust as needed)
        if self.n_calls % 1000 == 0:
            # get environment metrics
            frac_success = np.mean(self.training_env.get_attr('fractional_success'))
            
            # get training metrics
            stats = {
                'timesteps': self.num_timesteps,
                'reward_mean': np.mean(self.model.ep_info_buf),
                'fractional_success': frac_success,
                'value_loss': self.locals['values']['value_loss'],
                'policy_loss': self.locals['values']['policy_loss'],
                'explained_variance': self.locals['values']['explained_variance']
            }
            
            stats.update({
                'goal_position_x': envs[0].task.goal_position[0],
                'goal_position_y': envs[0].task.goal_position[1],
                'robot_joint_positions': np.mean(envs[0].robot.get_joint_positions())
            })
            
            # write to CSV
            with open(self.log_path, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(stats)
        return True

def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, ppo_config, total_time_steps,
                 validate_every_timesteps, task_name):

    def _make_env(rank):

        def _init():
            task = generate_task(task_generator_id=task_name)
            env = CausalWorld(task=task,
                              skip_frame=skip_frame,
                              enable_visualization=False,
                              seed=seed_num + rank,
                              max_episode_length=maximum_episode_length)
            return env

        set_global_seeds(seed_num)
        return _init

    # create a csv logger callback
    csv_logger = CSVLoggerCallback(log_relative_path)
    
    os.makedirs(log_relative_path, exist_ok=True)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    model = PPO2(MlpPolicy,
                 env,
                 _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1,
                 **ppo_config)
    save_config_file(ppo_config,
                     _make_env(0)(),
                     os.path.join(log_relative_path, 'config.json'))
    for i in range(int(total_time_steps / validate_every_timesteps)):
        model.learn(total_timesteps=validate_every_timesteps,
                    tb_log_name="ppo2",
                    reset_num_timesteps=False,
                    callback=csv_logger)
        model.save(os.path.join(log_relative_path, 'saved_model'))
    return


def save_config_file(ppo_config, env, file_path):
    task_config = env._task.get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = env.get_world_params()
    env.close()
    configs_to_save = [task_config, env_config, ppo_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    #TODO: pass reward weights here!!
    ap.add_argument("--seed_num", required=False, default=0, help="seed number")
    ap.add_argument("--skip_frame",
                    required=False,
                    default=5,
                    help="skip frame")
    ap.add_argument("--max_episode_length",
                    required=False,
                    default=500,
                    help="maximum episode length")
    ap.add_argument("--total_time_steps_per_update",
                    required=False,
                    default=100000,
                    help="total time steps per update")
    ap.add_argument("--num_of_envs",
                    required=False,
                    default=2,
                    help="number of parallel environments")
    ap.add_argument("--task_name",
                    required=False,
                    default="reaching",
                    help="the task nam for training")
    ap.add_argument("--fixed_position",
                    required=False,
                    default=True,
                    help="define the reset intervention wrapper")
    default_log_dir = os.path.join(os.getcwd(), "logs")
    ap.add_argument(
        "--log_relative_path",
        required=False,
        default=default_log_dir,
        help="log folder (default: ./logs in current working directory)"
    )
    args = vars(ap.parse_args())
    total_time_steps_per_update = int(args['total_time_steps_per_update'])
    num_of_envs = int(args['num_of_envs'])
    maximum_episode_length = int(args['max_episode_length'])
    skip_frame = int(args['skip_frame'])
    seed_num = int(args['seed_num'])
    task_name = str(args['task_name'])
    fixed_position = bool(args['fixed_position'])
    log_relative_path = str(args['log_relative_path'])
    assert (((float(total_time_steps_per_update) / num_of_envs) /
             5).is_integer())
    ppo_config = {
        "gamma": 0.99,
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "nminibatches": 64,
        "noptepochs": 3,
        "tensorboard_log": log_relative_path
    }
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 ppo_config=ppo_config,
                 total_time_steps=1000000,
                 validate_every_timesteps=100000,
                 task_name=task_name)