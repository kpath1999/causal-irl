import numpy as np
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators import generate_task
from causal_world.loggers.data_recorder import DataRecorder
# from causal_world.loggers.data_loader import DataLoader  # For loading after evaluation if needed

from stable_baselines3 import PPO


def generate_grid_points(range_values, num_points, exclude_edges=False):
    """
    Generates a grid of points within given ranges.
    If exclude_edges is True, the generated points exclude the absolute min/max.
    """
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


def create_evaluation_interventions(env, num_envs):
    """
    Creates an evaluation set of interventions that excludes edges.
    """
    intervention_space_a_b = env.get_intervention_space_a_b()
    range_values = intervention_space_a_b['goal_block']['cylindrical_position']

    # Generate evaluation points (exclude edges)
    generated_grid_points_eval = generate_grid_points(range_values, num_envs, exclude_edges=True)

    # Create dictionary representations for evaluation interventions
    eval_interventions = [{'goal_block': {'cylindrical_position': generated_grid_points_eval[i],}}
                          for i in range(len(generated_grid_points_eval))]

    return eval_interventions


def run_evaluation(env, model, interventions, pre_intervention_steps=2000, num_steps=5000, verbose=True):
    """
    Run evaluation episodes with a two-phase process:
    1. Perform some steps without intervention.
    2. Apply the intervention.
    3. Perform the remaining steps after the intervention.

    Parameters:
    - env: The CausalWorld environment
    - model: A trained model/policy with a .predict(obs) method.
    - interventions: A list of intervention dictionaries to apply before running an episode.
    - pre_intervention_steps: Number of steps to run before applying the intervention.
    - num_steps: Number of steps to run after applying the intervention.
    - verbose: Whether to print out results.
    """
    results = []
    for i, intervention in enumerate(interventions):
        if verbose:
            print(f"Evaluation Intervention {i+1}/{len(interventions)}: {intervention}")

        # Reset the environment to start a new episode
        obs = env.reset()

        # Now apply the intervention
        success_signal, obs = env.do_intervention(intervention)
        if verbose:
            print("Intervention success signal:", success_signal)

        # Perform the main evaluation steps after the intervention
        episode_rewards = 0.0
        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break

        results.append({
            'intervention': intervention,
            'success_signal': success_signal,
            'total_reward': episode_rewards
        })

        if verbose:
            print("Total reward for this episode:", episode_rewards)

    return results


if __name__ == "__main__":
    # Set up the data recorder for the environment
    data_recorder = DataRecorder(output_directory='pushing_episodes',
                                 rec_dumb_frequency=11)

    # Create the environment with recording enabled
    task = generate_task(task_generator_id='pushing')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      data_recorder=data_recorder)

    # Load your trained PPO model
    # Replace "path_to_ppo_model.zip" with the actual path to your model file
    model = PPO.load("./ppo_generalization_pushing_10/logs/best_model/best_model.zip")

    # Number of different interventions for evaluation
    num_envs = 5

    # Create evaluation interventions (exclude edges)
    eval_interventions = create_evaluation_interventions(env, num_envs)

    # Evaluate on evaluation interventions and record episodes
    print("Evaluating on evaluation interventions...")
    eval_results = run_evaluation(env, model, eval_interventions, num_steps=5000, verbose=True)

    # Close environment and finalize recording
    env.close()

    # Optionally load and inspect recorded data
    # data = DataLoader(episode_directory='pushing_episodes')
    # episode_data = data.get_episode(0)  # for example, load the first recorded episode
    # print("Loaded episode data:", episode_data)