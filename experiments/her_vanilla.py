from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper
from stable_baselines import HER, SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback, CallbackList
from stable_baselines import logger
import tensorflow as tf
import numpy as np
import os
import argparse
import json

# --- Custom Callbacks ---

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
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
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            obs = self.eval_env.reset()
            for _ in range(10):  # Run 10 episodes
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

# --- Environment Factory ---

def make_env(task_name, log_dir=None, seed=0, skip_frame=3, max_episode_length=2500, vis=False):
    task = generate_task(task_generator_id=task_name)
    env = CausalWorld(
        task=task,
        skip_frame=skip_frame,
        enable_visualization=vis,
        seed=seed,
        max_episode_length=max_episode_length
    )
    if log_dir is not None:
        env = Monitor(env, log_dir)
    env = HERGoalEnvWrapper(env)
    return env

# --- Training Function ---

def train_policy(
    log_relative_path, maximum_episode_length, skip_frame, seed_num,
    sac_config, total_time_steps, validate_every_timesteps, task_name
):
    os.makedirs(log_relative_path, exist_ok=True)
    log_dir = os.path.join(log_relative_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Configure logger for stdout and CSV
    logger.configure(log_relative_path, ["stdout", "csv"])

    # --- Create training and evaluation environments ONCE and persistently ---
    env = make_env(task_name, log_dir, seed=seed_num, skip_frame=skip_frame, max_episode_length=maximum_episode_length)
    eval_env = make_env(task_name, None, seed=seed_num+100, skip_frame=skip_frame, max_episode_length=maximum_episode_length)

    # --- Model ---
    model = HER(
        MlpPolicy,
        env,
        SAC,
        verbose=1,
        policy_kwargs=dict(layers=[256, 256, 256]),
        **sac_config
    )

    # --- Save config ---
    save_config_file(sac_config, env, os.path.join(log_relative_path, 'config.json'))

    # --- Callbacks ---
    checkpoint_callback = CustomCheckpointCallback(
        save_freq=validate_every_timesteps,
        save_path=log_dir
    )
    eval_callback = SimpleEvalCallback(
        eval_env=eval_env,  # persistent, not recreated
        eval_freq=validate_every_timesteps,
        log_path=log_dir,
        verbose=2
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    # --- Training loop ---
    for i in range(int(total_time_steps / validate_every_timesteps)):
        print(f"\n=== Training Iteration {i+1} ===")
        model.learn(
            total_timesteps=validate_every_timesteps,
            tb_log_name="sac",
            reset_num_timesteps=False,
            callback=callback
        )
        model.save(os.path.join(log_relative_path, f'saved_model_{(i+1)*validate_every_timesteps}'))

    # --- Final save ---
    model.save(os.path.join(log_relative_path, 'final_model'))
    print("Training complete. Final model saved.")

    # --- Explicitly close environments at the end ---
    env.close()
    eval_env.close()

# --- Save Config Utility ---

def save_config_file(sac_config, env, file_path):
    task_config = env.get_task().get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = env.get_world_params()
    # Only close here if this is a temporary env (not the main one)
    configs_to_save = [task_config, env_config, sac_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)

# --- Main ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_num", default=0, type=int)
    parser.add_argument("--skip_frame", default=10, type=int)
    parser.add_argument("--max_episode_length", default=2500, type=int)
    parser.add_argument("--total_time_steps_per_update", default=150000, type=int)
    parser.add_argument("--task_name", default="reaching")
    parser.add_argument("--log_relative_path", required=True)
    args = parser.parse_args()

    total_time_steps_per_update = args.total_time_steps_per_update
    log_relative_path = args.log_relative_path
    maximum_episode_length = args.max_episode_length
    skip_frame = args.skip_frame
    seed_num = args.seed_num
    task_name = args.task_name

    sac_config = {
        "n_sampled_goal": 4,
        "goal_selection_strategy": 'future',
        "learning_rate": 0.001,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 1000,
        "tensorboard_log": log_relative_path,
        "buffer_size": int(1e6),
        "gamma": 0.95,
        "batch_size": 256,
        "ent_coef": 'auto',
        "random_exploration": 0.1
    }

    train_policy(
        log_relative_path=log_relative_path,
        maximum_episode_length=maximum_episode_length,
        skip_frame=skip_frame,
        seed_num=seed_num,
        sac_config=sac_config,
        total_time_steps=60000000,
        validate_every_timesteps=1000000,
        task_name=task_name
    )