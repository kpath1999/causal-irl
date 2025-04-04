import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

show_only_reward = True

def eval_and_compare(load_dirs, labels):
    """
    Compare training progress from multiple data collections.

    Parameters:
    load_dirs (list): List of directories containing 'progress.csv'.
    labels (list): List of labels corresponding to each data collection.
    """
    if len(load_dirs) != len(labels):
        raise ValueError("Number of directories and labels must match.")

    plt.style.use('dark_background')  # Apply dark background style

    # Define metrics to plot and their corresponding y-axis labels
    metrics = {
        'rollout/ep_rew_mean': 'Mean Episode Reward (Train)',
        'train/value_loss': 'Value Loss',
        'train/policy_gradient_loss': 'Policy Gradient Loss'
    }

    # Determine the maximum timesteps across all datasets
    max_timesteps = np.inf
    for load_dir in load_dirs:
        csv_path = os.path.join(load_dir, 'progress.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'time/total_timesteps' in df:
                max_timesteps = min(max_timesteps, df['time/total_timesteps'].max())

    for metric, ylabel in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('black')  # Set figure background to black

        for load_dir, label in zip(load_dirs, labels):
            print(load_dir)
            if metric == 'train/value_loss' and load_dir.split('_')[0] == 'sac': metric = 'train/critic_loss'
            if metric == 'train/policy_gradient_loss' and load_dir.split('_')[0] == 'sac': metric = 'train/actor_loss'
            csv_path = os.path.join(load_dir, 'progress.csv')
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            if 'time/total_timesteps' not in df or metric not in df:
                print(f"Required columns not found in {csv_path}")
                continue

            ax.plot(df['time/total_timesteps'], df[metric], label=label)

        ax.set_xlim([0, max_timesteps])  # Lock the x-axis to the maximum timesteps
        ax.set_xlabel('Timesteps', color='white')  # Set x-axis label color to white
        ax.set_ylabel(ylabel, color='white')  # Set y-axis label color to white
        ax.set_title(f"{ylabel} Over Time", color='white')  # Set title color to white
        ax.legend()
        ax.grid(True, color='gray')  # Make grid lines visible in dark theme
        ax.set_facecolor('black')  # Set subplot background to black

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    eval_and_compare([
        # f'ppo_generalization_pushing_10', 
        # f'ppo_pushing_20',
        # 'ppo_pushing_10_delta_1e7',
        # 'sac_pushing_10_mva_real_2025-01-10 14:26:03.423526',
        # 'sac_pushing_10_mva_real_2',
        # 'sac_generalization_pushing_5_1.0e+07',
        # 'sac_generalization_pushing_5_1.0e+07',
        'sac_no_pushing_5_10000000_mva_real_Tue Jan 14 165414 2025',
        'sac_1_pushing_5_10000000_mva_real_Tue Jan 14 170126 2025',
        'models_my/sac_pushing_5_10000000_mva_real_Fri-2',
        'models_my/sac_pushing_5_10000000_mva_real_Fri-3',
        'models_my/sac_pushing_1_10000000_mva_real_Wed',
        # 'models_my/cw/22',
        # 'models_my/cw/25'
    ], [
        # 'PPO_1',
        # 'PPO_0', 
        # 'PPO_0_relative_action',
        # 'sac_1e-3',
        # 'sac_1e-4+',
        # 'sac_generalization',
        'sac_0',
        'sac_1',
        'sac_2',
        'sac_3',
        'sac_baseline',
        # '22',
        # '25',
        # 'models'
    ])