import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import app, flags
import os
import datetime
import time  # Added for timing measurements
import tensorflow as tf
import gym
print("Gym version:", gym.__version__)

from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PUSHING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import logger
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.callbacks import CallbackList
# from stable_baselines.common.env_checker import check_env
from causal_world.wrappers.action_wrappers import MovingAverageActionEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, 'Train the model.')
flags.DEFINE_bool('eval', True, 'Evaluate the model.')
flags.DEFINE_float('timesteps', 1e6, 'Total timesteps.')    # 1M timesteps
flags.DEFINE_integer('num_envs', 5, 'Number of envs.')
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
        env = MovingAverageActionEnvWrapper(env)
        env = Monitor(env, log_dir)
        
        print(f"• Action space: {env.action_space}")
        print(f"• Observation space: {env.observation_space}\n")
        
        # check_env(env, warn=True)
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
    
    # policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[256, 256])
    
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256])
    env = SubprocVecEnv([_make_env(rank=i, name=task, log_dir=log_dir, vis=vis) for i in range(num_envs)])
    
    model = PPO2(MlpPolicy,
             env,
             policy_kwargs=policy_kwargs,
             gamma=0.99,
             verbose=1,
             learning_rate=2.5e-4,
             n_steps=50_000 // num_envs,
             nminibatches=4,  # you can adjust this
             noptepochs=4,    # number of training epochs per update
             ent_coef=0.0,    # entropy coefficient (tweak if needed)
             lam=0.95,        # GAE lambda
             cliprange=0.2)   # PPO clipping range
    
    print("\nModel architecture summary:")
    print(model.policy)
    return model

def train(task, save_dir, total_timesteps=1e10, save_freq=500, n_envs=10):
    print("\n====== Training Setup ======")
    print(f"• Save directory: {save_dir}")
    print(f"• Total timesteps: {total_timesteps:,}")
    print(f"• Checkpoint frequency: {save_freq:,} steps")
    print(f"• Number of environments: {n_envs}")

    logger.configure(save_dir, ["stdout", "csv"])

    if not os.path.exists(os.path.join(save_dir, 'logs')):
        os.makedirs(os.path.join(save_dir, 'logs'))
        print(f"Created directory: {os.path.join(save_dir, 'logs')}")

    checkpoint_callback = CustomCheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(save_dir, './logs/'))
    
    task = generate_task(task_generator_id='pushing')
    eval_env = CausalWorld(task=task)
    # eval_env = GymV21CompatibilityV0(eval_env)

    eval_callback = SimpleEvalCallback(
        eval_env=eval_env,
        eval_freq=save_freq,
        log_path=os.path.join(save_dir, './logs'),
        verbose=2
    )
    
    def combined_callback(_locals, _globals):
        return checkpoint_callback.on_step() and eval_callback.on_step()
        
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

    # model.set_logger(new_logger)
    
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
    
    root_dir = os.path.dirname(os.path.abspath(__file__))  # script's directory
    load_dir = os.path.join(root_dir, load_dir)
    
    progress_file = os.path.join(load_dir, 'progress.csv')
    # progress_file = r"C:\Users\kausa\causal-experiments\ppo_pushing_5_mva_block_catching_1e-3_50k_20250421-164954\progress.csv"
    
    if not os.path.exists(progress_file):
        raise FileNotFoundError(f"No progress data found at {progress_file}")

    df = pd.read_csv(progress_file)
    
    
    
    # Summary stats
    print("\nTraining Statistics Summary:")
    print(f"• Total training episodes: {len(df):,}")
    print(f"• Final Mean Reward: {df['ep_reward_mean'].iloc[-1]:.2f}")
    print(f"• Max Reward Achieved: {df['ep_reward_mean'].max():.2f}")
    print(f"• Min Reward Achieved: {df['ep_reward_mean'].min():.2f}")
    
    # Plotting reward metrics
    plt.figure(figsize=(12, 6))
    plt.plot(df['total_timesteps'], df['ep_reward_mean'], label='Mean Reward')
    
    # If ep_reward_std available, fill; otherwise skip
    if 'ep_reward_std' in df.columns:
        plt.fill_between(df['total_timesteps'], 
                         df['ep_reward_mean'] - df['ep_reward_std'],
                         df['ep_reward_mean'] + df['ep_reward_std'],
                         alpha=0.2, label='±1 Std Dev')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(f'Training Progress - PPO Vanilla')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(load_dir, 'training_progress.png'))
    # viz_dir = r"C:\Users\kausa\causal-experiments\ppo_pushing_5_mva_block_catching_1e-3_50k_20250421-164954"
    # plt.savefig(os.path.join(viz_dir, 'training_progress.png'))
    print("\nSaved training progress plot to 'training_progress.png'")
    plt.show()

def gen_eval(load_dir):
    print("\n====== Benchmark Evaluation ======")
    model_dir = os.path.join(load_dir, 'logs')
    # model_dir = r"C:\Users\kausa\causal-experiments\ppo_pushing_5_mva_block_catching_1e-3_50k_20250421-164954"
    model_path = os.path.join(model_dir, 'best_model', 'best_model.zip')
    # model_path = os.path.join(model_dir, 'logs', 'rl_model_400000.zip')
    print(f"Loading best model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    
    try:
        model = PPO2.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    task_params = {'task_generator_id': 'pushing'}
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
        if isinstance(value, (int, float)):
            print(f"• {metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"• {metric.replace('_', ' ').title()}: {value}")

    experiments = {'PPO': scores_model}
    vis.generate_visual_analysis('./', experiments=experiments)
    print("\nSaved visual analysis to current directory.")

def main(argv):
    del argv
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # save_dir = f'ppo_{FLAGS.task}_{FLAGS.num_envs}_mva_block_catching_1e-3_50k_{timestamp}'
    save_dir = f'ppo_vanilla_second_run'
    
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