import numpy as np
import torch
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

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from causal_world.wrappers.action_wrappers import MovingAverageActionEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, 'Train the model.')
flags.DEFINE_bool('eval', False, 'Evaluate the model.')
flags.DEFINE_float('timesteps', 1e4, 'Total timesteps.')
flags.DEFINE_integer('num_envs', 5, 'Number of envs.')
flags.DEFINE_string('task', 'pushing', 'Task selected.')
flags.DEFINE_integer('param_intervention', 1, 'Number of parameters to intervent on.')

def generate_grid_points(range_values, num_points, exclude_edges=False):
    if range_values.shape[0] != 2:
        raise ValueError("Input array must have two rows: the first for minimum values and the second for maximum values.")
    min_values = range_values[0]
    max_values = range_values[1]
    if len(min_values) != len(max_values):
        raise ValueError("Minimum and maximum value vectors must have the same length.")
    ranges = []
    for min_val, max_val in zip(min_values, max_values):
        if min_val == max_val:
            ranges.append(np.array([min_val] * num_points))
        else:
            if exclude_edges:
                ranges.append(np.linspace(min_val, max_val, num_points + 2)[1:-1])
            else:
                ranges.append(np.linspace(min_val, max_val, num_points))
    grid_points = np.stack(ranges, axis=-1).reshape(-1, len(min_values))
    return grid_points

def generate_interventions(intervention_parameters, env, num_envs, train=False):
    intervention_space = env.get_intervention_space_a_b()

    lb, up = [], []
    for intervention_name in intervention_parameters:
        base, argument = intervention_name.split('.')
        parameter_space = intervention_space[base][argument]
        lb.extend(parameter_space[0].tolist())
        up.extend(parameter_space[1].tolist())
    grid_points = generate_grid_points(np.array([lb, up]), num_envs, not train)
    interventions = []
    print(grid_points)

    for point in grid_points:
        intervention = {}

        for i, intervention_name  in enumerate(intervention_parameters):
            base, argument = intervention_name.split('.')
            if not intervention.get(base, False):
                intervention[base] = {argument: point[3*i:3*(i+1)]}
            else:
                intervention[base][argument] = point[3*i:3*(i+1)]

        interventions.append(intervention)
    parameter_space = intervention_space[base][argument]

    return interventions

def _make_env_with_intervention(task_id, task_settings, world_settings, intervention, seed=0, log_dir='test', vis=False, train=True):
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
        print(intervention)
        success_signal, obs = env.do_intervention(intervention)
        print("Intervention success signal:", success_signal)
        if train:
            env = Monitor(env, log_dir)
        return env
    return _init

def init(task_id, task_settings, world_settings, intervention_parameters, seed=0, num_envs=10, log_dir='test', vis=False, train=True):
    temp_env = CausalWorld(
        task = generate_task(
            task_generator_id=task_id,
            **task_settings
            ),
        **world_settings,
        seed=seed,
        )
    interventions = generate_interventions(intervention_parameters, temp_env, num_envs, train=train)
    print(interventions)
    print(f'Envs setups:')
    for i, setup in enumerate(interventions):
        print(f'Env {i}: {setup}')
    temp_env.close()
    del temp_env
    env = SubprocVecEnv([
        _make_env_with_intervention(
             task_id, task_settings, world_settings, log_dir=log_dir, vis=vis, intervention=interventions[i], train=train)
        for i in range(num_envs)
    ])
    return env


def train(algorithm_config, task_config, world_config, network_settings,
        intervention_parameters, task='pushing', n_envs=10, seed=0):
    validate_every_timesteps = algorithm_config['validate_every_timesteps']
    ckpt_frequency = max(validate_every_timesteps // n_envs, 1)
    total_time_steps = algorithm_config['total_time_steps']

    save_dir = f'sac_{task}_{n_envs}_{total_time_steps}_mva_real_{datetime.datetime.now().ctime()}'
    new_logger = configure(save_dir, ['stdout', "csv"])

    if not os.path.exists(os.path.join(save_dir, 'logs')):
        os.makedirs(os.path.join(save_dir, 'logs'))

    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_frequency,
        save_path=os.path.join(save_dir, './logs/'),
        name_prefix='model')
    
    callback = CallbackList([checkpoint_callback, 
                            #  eval_callback
                             ])

    env = init(task, task_config, world_config, intervention_parameters, num_envs=n_envs, log_dir=save_dir, train=True)

    model = SAC(MlpPolicy,
                env,
                _init_setup_model=True,
                policy_kwargs=network_settings,
                gradient_steps=n_envs,
                **algorithm_config['train_configs'],
                verbose=1,
                )
    
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_time_steps, callback=callback)
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

def gen_eval(load_dir):
    task_params = dict()
    task_params['task_generator_id'] = 'pushing'
    task_params['dense_reward_weights'] = np.array([750, 250, 0])
    task_params['fractional_reward_weight'] = 1
    world_params = dict()
    world_params['skip_frame'] = 3
    world_params['action_mode'] = 'joint_torques'
    
    evaluation_protocols = PUSHING_BENCHMARK['evaluation_protocols']
    evaluator = EvaluationPipeline(
        evaluation_protocols=evaluation_protocols,
        task_params=task_params,
        world_params=world_params,
        visualize_evaluation=False
        )
    stable_baselines_policy_path = os.path.join(load_dir, 'logs', 'best_model', 'best_model.zip')
    print(stable_baselines_policy_path)
    model = SAC.load(f'{stable_baselines_policy_path}')

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]
    scores_model = evaluator.evaluate_policy(policy_fn, fraction=0.005)
    experiments = dict()
    experiments['SAC'] = scores_model
    vis.generate_visual_analysis('./', experiments=experiments)

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
            'variables_space': 'space_a_b',
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
    
    intervention_parameters = ['goal_block.cylindrical_position', 'tool_block.cylindrical_position', 'tool_block.size']
    intervention_parameters = intervention_parameters[:FLAGS.param_intervention]

    net_layers = dict(net_arch=[256, 256])
    world_seed = 0
   
    if FLAGS.train:
        train(
            algorithm_config=sac,
            task_config=task_configs,
            world_config=world_params,
            network_settings=net_layers,
            intervention_parameters=intervention_parameters,
            n_envs=FLAGS.num_envs,
            seed=world_seed
        )
    if FLAGS.eval:
        eval(load_dir=FLAGS.load_dir)
        # gen_eval(save_dir)
        
if __name__ == '__main__':
    app.run(main)