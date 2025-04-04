import numpy as np
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task

def chose_value(range_values):
    if range_values.shape[0] != 2:
        raise ValueError("Input array must have two rows: the first for minimum values and the second for maximum values.")
    min_values = range_values[0]
    max_values = range_values[1]
    if len(min_values) != len(max_values):
        raise ValueError("Minimum and maximum value vectors must have the same length.")
    random_values = np.random.uniform(min_values, max_values)
    return random_values

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

env = CausalWorld(task=generate_task('pushing'), enable_visualization=True)
env.reset()
intervention_space_a_b = env.get_intervention_space_a_b()
num_envs = 20
generated_grid_points_train = generate_grid_points(intervention_space_a_b['goal_block']['cylindrical_position'], num_envs, exclude_edges=False)
generated_grid_points_eval = generate_grid_points(intervention_space_a_b['goal_block']['cylindrical_position'], num_envs, exclude_edges=True)
train_ranges = [{'goal_block': {'cylindrical_position': generated_grid_points_train[i],},} for i in range(len(generated_grid_points_train))]
eval_ranges = [{'goal_block': {'cylindrical_position': generated_grid_points_train[i],},} for i in range(len(generated_grid_points_eval))]



# for i in range(num_envs):
#     test_intervention = train_ranges[i]
#     print(f"Test Intervention {i+1}: {test_intervention}")
#     success_signal, obs = env.do_intervention(test_intervention)
#     print("Intervention success signal:", success_signal)
#     for _ in range(2000):
#         obs, reward, done, info = env.step(env.action_space.sample())

for i in range(num_envs):
    test_intervention = eval_ranges[i]
    print(f"Evaluation Intervention {i+1}: {test_intervention}")
    success_signal, obs = env.do_intervention(test_intervention)
    print("Intervention success signal:", success_signal)
    for _ in range(2000):
        obs, reward, done, info = env.step()

env.close()