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
import argparse


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

    os.makedirs(log_relative_path)
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
                    reset_num_timesteps=False)
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
                    default=10,
                    help="skip frame")
    ap.add_argument("--max_episode_length",
                    required=False,
                    default=2500,
                    help="maximum episode length")
    ap.add_argument("--total_time_steps_per_update",
                    required=False,
                    default=100000,
                    help="total time steps per update")
    ap.add_argument("--num_of_envs",
                    required=False,
                    default=5,
                    help="number of parallel environments")
    ap.add_argument("--task_name",
                    required=False,
                    default="reaching",
                    help="the task nam for training")
    ap.add_argument("--fixed_position",
                    required=False,
                    default=True,
                    help="define the reset intervention wrapper")
    ap.add_argument("--log_relative_path", required=True, help="log folder")
    args = vars(ap.parse_args())
    total_time_steps_per_update = int(args['total_time_steps_per_update'])
    num_of_envs = int(args['num_of_envs'])
    log_relative_path = str(args['log_relative_path'])
    maximum_episode_length = int(args['max_episode_length'])
    skip_frame = int(args['skip_frame'])
    seed_num = int(args['seed_num'])
    task_name = str(args['task_name'])
    fixed_position = bool(args['fixed_position'])
    assert (((float(total_time_steps_per_update) / num_of_envs) /
             5).is_integer())
    ppo_config = {
        "gamma": 0.9995,
        "n_steps": 5000,
        "ent_coef": 0,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 10,
        "nminibatches": 1000,
        "noptepochs": 4,
        "tensorboard_log": log_relative_path
    }
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 ppo_config=ppo_config,
                 total_time_steps=60000000,
                 validate_every_timesteps=1000000,
                 task_name=task_name)
