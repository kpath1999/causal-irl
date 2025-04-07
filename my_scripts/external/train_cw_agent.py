import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import app, flags
import os
import datetime
import time  # Added for timing measurements
import gymnasium as gym
print("Gym version:", gym.__version__)
from shimmy import GymV21CompatibilityV0

import sys
sys.modules['gym'] = __import__('gymnasium', fromlist=[''])

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
from stable_baselines3.common.env_checker import check_env
from causal_world.wrappers.action_wrappers import MovingAverageActionEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', True, 'Train the model.')
flags.DEFINE_bool('eval', False, 'Evaluate the model.')
flags.DEFINE_float('timesteps', 1e4, 'Total timesteps.')
flags.DEFINE_integer('num_envs', 10, 'Number of envs.')
flags.DEFINE_string('task', 'pushing', 'Task selected.')

def _make_env(rank, name, seed=0, log_dir='test', vis=False):
    def _init():
        print(f"\n=== Initializing environment {rank} ===")
        print(f"• Task generator: {name}")
        print(f"• Seed: {seed + rank}")
        print(f"• Visualization enabled: {vis}")
        
        task = generate_task(
            task_generator_id=name,
            dense_reward_weights=np.array([1, 0, 0]),
            variables_space='space_a',
            fractional_reward_weight=1
        )
        env = CausalWorld(
            task=task,
            skip_frame=3,
            action_mode='joint_torques',
            enable_visualization=vis,
            seed=seed+rank
        )
        env = GymV21CompatibilityV0(env)
        env = MovingAverageActionEnvWrapper(env)
        env = Monitor(env, log_dir)
        
        print(f"• Action space: {env.action_space}")
        print(f"• Observation space: {env.observation_space}\n")
        
        check_env(env, warn=True)
        print(f"Environment {rank} initialized successfully.")
        
        return env
    return _init

def init(task='pushing', num_envs=10, log_dir='test', vis=False):
    print("\n====== Initializing Model ======")
    print(f"• Policy network: MLP [256, 256] with Tanh activation")
    print(f"• Number of parallel environments: {num_envs}")
    print(f"• Gamma: 0.99")
    print(f"• Batch size: 64")
    print(f"• Learning rate: 2.5e-4")
    
    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[256, 256])
    env = SubprocVecEnv([_make_env(rank=i, name=task, log_dir=log_dir, vis=vis) for i in range(num_envs)])
    
    model = PPO(MlpPolicy,
                env,
                policy_kwargs=policy_kwargs,
                gamma=.99,
                verbose=1,
                batch_size=64,
                learning_rate=2.5e-4,
                n_epochs=10,
                n_steps=50_000//num_envs)
    
    print("\nModel architecture summary:")
    print(model.policy)
    return model

def train(task, save_dir, total_timesteps=1e4, save_freq=1e3, n_envs=10):
    print("\n====== Training Setup ======")
    print(f"• Save directory: {save_dir}")
    print(f"• Total timesteps: {total_timesteps:,}")
    print(f"• Checkpoint frequency: {save_freq:,} steps")
    print(f"• Number of environments: {n_envs}")

    new_logger = configure(save_dir, ["stdout", "csv"])

    if not os.path.exists(os.path.join(save_dir, 'logs')):
        os.makedirs(os.path.join(save_dir, 'logs'))
        print(f"Created directory: {os.path.join(save_dir, 'logs')}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(save_dir, './logs/'))
    
    task = generate_task(task_generator_id='pushing')
    eval_env = CausalWorld(task=task)
    eval_env = GymV21CompatibilityV0(eval_env)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(save_dir, './logs/best_model'),
        log_path=os.path.join(save_dir, './logs/results'), 
        eval_freq=save_freq,
        verbose=2  # Increased verbosity
    )
    
    callback = CallbackList([checkpoint_callback, eval_callback])

    model = init(num_envs=n_envs, log_dir=save_dir)
    
    # Checkpoint loading with detailed logging
    checkpoints = [f for f in os.listdir(os.path.join(save_dir, 'logs')) 
                  if f.startswith('rl_model_')] if os.path.exists(save_dir) else []
    if checkpoints:
        checkpoints.sort()
        latest_checkpoint = checkpoints[-1]
        print(f"\nFound existing checkpoint: {latest_checkpoint}")
        model.load(os.path.join(save_dir, 'logs', latest_checkpoint))
        done_timesteps = int(latest_checkpoint.split('_')[-2])
        print(f"Resuming training from {done_timesteps:,} timesteps")
    else:
        done_timesteps = 0
        print("\nNo existing checkpoints found. Starting fresh training.")

    model.set_logger(new_logger)
    
    print("\n====== Starting Training ======")
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=total_timesteps-done_timesteps, 
                   callback=callback,
                   reset_num_timesteps=False if done_timesteps > 0 else True)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model state...")
        model.save(os.path.join(save_dir, 'interrupted_model'))
        print("Model saved successfully.")
        exit()

    print(f"\nTraining completed in {(time.time() - start_time)/3600:.2f} hours")
    model.save(os.path.join(save_dir, 'final_model'))
    print("Final model saved successfully.")

def eval(load_dir):
    print("\n====== Evaluation Mode ======")
    print(f"Loading results from: {load_dir}")
    
    progress_file = os.path.join(load_dir, 'progress.csv')
    if not os.path.exists(progress_file):
        raise FileNotFoundError(f"No progress data found at {progress_file}")

    df = pd.read_csv(progress_file)
    
    print("\nTraining Statistics Summary:")
    print(f"• Total training episodes: {df['time/episodes'].iloc[-1]:,}")
    print(f"• Final Mean Reward: {df['rollout/ep_rew_mean'].iloc[-1]:.2f}")
    print(f"• Max Reward Achieved: {df['rollout/ep_rew_max'].max():.2f}")
    print(f"• Min Reward Achieved: {df['rollout/ep_rew_min'].min():.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], label='Mean Reward')
    plt.plot(df['time/total_timesteps'], df['rollout/ep_rew_max'], alpha=0.3, label='Max Reward')
    plt.plot(df['time/total_timesteps'], df['rollout/ep_rew_min'], alpha=0.3, label='Min Reward')
    plt.fill_between(df['time/total_timesteps'], 
                    df['rollout/ep_rew_mean'] - df['rollout/ep_rew_std'],
                    df['rollout/ep_rew_mean'] + df['rollout/ep_rew_std'],
                    alpha=0.2)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(f'Training Progress - {os.path.basename(load_dir)}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(load_dir, 'training_progress.png'))
    print("\nSaved training progress plot to 'training_progress.png'")
    plt.show()

def gen_eval(load_dir):
    print("\n====== Benchmark Evaluation ======")
    model_path = os.path.join(load_dir, 'logs', 'best_model', 'best_model.zip')
    print(f"Loading best model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    task_params = {'task_generator_id': 'pushing', 'skip_frame': 3}
    world_params = {'skip_frame': 3, 'action_mode': 'joint_torques'}

    evaluator = EvaluationPipeline(
        evaluation_protocols=PUSHING_BENCHMARK['evaluation_protocols'],
        task_params=task_params,
        world_params=world_params,
        visualize_evaluation=False
    )

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    print("\nRunning evaluation protocol...")
    scores_model = evaluator.evaluate_policy(policy_fn, fraction=0.005)
    
    print("\nEvaluation Results:")
    for metric, value in scores_model.items():
        print(f"• {metric.replace('_', ' ').title()}: {value:.2f}")

    experiments = {'PPO': scores_model}
    vis.generate_visual_analysis('./', experiments=experiments)
    print("\nSaved visual analysis to current directory.")

def main(argv):
    del argv
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f'ppo_{FLAGS.task}_{FLAGS.num_envs}_mva_block_catching_1e-3_50k_{timestamp}'
    
    print("\n====== Experiment Configuration ======")
    print(f"• Task: {FLAGS.task}")
    print(f"• Number of environments: {FLAGS.num_envs}")
    print(f"• Total timesteps: {FLAGS.timesteps:,}")
    print(f"• Save directory: {save_dir}")

    if FLAGS.train:
        print("\n=== Training Mode Activated ===")
        train(FLAGS.task, save_dir, int(FLAGS.timesteps), save_freq=1e5, n_envs=FLAGS.num_envs)
    
    if FLAGS.eval:
        print("\n=== Evaluation Mode Activated ===")
        eval(save_dir)
        gen_eval(save_dir)

if __name__ == '__main__':
    app.run(main)