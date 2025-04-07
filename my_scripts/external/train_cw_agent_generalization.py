import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from absl import app, flags
import os

from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PUSHING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, 'Train the model.')
flags.DEFINE_bool('eval', False, 'Evaluate the model.')
flags.DEFINE_float('timesteps', 1e4, 'Total timesteps.')
flags.DEFINE_integer('num_envs', 10, 'Number of envs.')
flags.DEFINE_string('task', 'pushing', 'Task selected.')

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

def generate_interventions(env, num_envs, train=False):
    intervention_space = env.get_intervention_space_a_b()
    goal_block_position_space = intervention_space['goal_block']['cylindrical_position']

    grid_points = generate_grid_points(goal_block_position_space, num_envs, not train)
    interventions = []
    for point in grid_points:
        intervention = {'goal_block': {'cylindrical_position': point}}
        interventions.append(intervention)

    return interventions

def _make_env_with_intervention(
        rank, name, intervention, seed=0, log_dir='test', vis=False, train=True):
    def _init():
        task = generate_task(task_generator_id=name)
        env = CausalWorld(
            task=task, enable_visualization=vis, seed=seed+rank)
        print(intervention)
        success_signal, obs = env.do_intervention(intervention)
        print("Intervention success signal:", success_signal)
        if train:
            env = Monitor(env, log_dir)
        return env
    return _init

def init(task='pushing', num_envs=10, log_dir='test', vis=False, train=True):
    temp_env = CausalWorld(
        task=generate_task(task_generator_id=task), enable_visualization=False)
    interventions = generate_interventions(temp_env, num_envs, train=train)
    print(interventions)
    print(f'Envs setups:')
    for i, setup in enumerate(interventions):
        print(f'Env {i}: {setup}')
    temp_env.close()
    
    env = SubprocVecEnv([
        _make_env_with_intervention(
            rank=i, name=task, log_dir=log_dir, vis=vis, intervention=interventions[i], train=train)
        for i in range(num_envs)
    ])
    return env

def train(task, save_dir, total_timesteps=1e4, save_freq=1e3, n_envs=10):
    new_logger = configure(save_dir, ["stdout", "csv"])

    if not os.path.exists(os.path.join(save_dir, 'logs')):
        os.makedirs(os.path.join(save_dir, 'logs'))

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(save_dir, 'logs')
        )
    eval_env = init(
        task=task, num_envs=5, log_dir=None, train=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, 'logs', 'best_model'),
        log_path=os.path.join(save_dir, 'logs', 'results'),
        eval_freq=save_freq,
        deterministic=True
        )
    
    callback = CallbackList([checkpoint_callback, eval_callback])

    env = init(task=task, num_envs=n_envs, log_dir=save_dir, train=True)

    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[256, 128])
    model = PPO(MlpPolicy,
                env,
                policy_kwargs=policy_kwargs,
                verbose=1)
    
    if os.path.exists(os.path.join(save_dir)):
        checkpoints = [f for f in os.listdir(os.path.join(save_dir, 'logs')) if f.startswith('rl_model_')]
        checkpoints.sort()
        done_timesteps = 0
        if len(checkpoints) > 0:
            print(f'Loaded checkpoint from {checkpoints[-1]}')
            done_timesteps = int(checkpoints[-1].split('.')[0].split('_')[-2])
            model.load(os.path.join(save_dir, 'logs', checkpoints[-1]))

    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps-done_timesteps, callback=callback)
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
    world_params = dict()
    world_params['skip_frame'] = 10
    evaluation_protocols = PUSHING_BENCHMARK['evaluation_protocols']
    evaluator = EvaluationPipeline(
        evaluation_protocols=evaluation_protocols,
        task_params=task_params,
        world_params=world_params,
        visualize_evaluation=False
        )
    stable_baselines_policy_path = os.path.join(load_dir, 'logs', 'best_model', 'best_model.zip')
    print(stable_baselines_policy_path)
    model = PPO.load(f'{stable_baselines_policy_path}')

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]
    scores_model = evaluator.evaluate_policy(policy_fn, fraction=0.005)
    experiments = dict()
    experiments['PPO'] = scores_model
    vis.generate_visual_analysis('./', experiments=experiments)

def main(argv):
    del argv
    save_dir = f'ppo_generalization_{FLAGS.task}_{FLAGS.num_envs}'
    if FLAGS.train:
        train(FLAGS.task, save_dir, int(FLAGS.timesteps), save_freq=1e5, n_envs=FLAGS.num_envs)
    if FLAGS.eval:
        eval(save_dir)
        gen_eval(save_dir)
        
if __name__ == '__main__':
    app.run(main)