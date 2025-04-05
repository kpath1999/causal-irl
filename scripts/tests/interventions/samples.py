"""
My goal is to combine RL training with causal interventions.
This would build causally aware policies that are robust to distriution shifts.

Intervention strategies include:
- Systematic sweeping (vary one variable at a time while holding others constant)
- Random sampling (randomly sample values from the intervention space)
- Adaptive interventions (use active learning to prioritize interventions to reduce uncertainty in causal graphs)

After collecting interventional data, I would go into analysis and causal discovery. This would involve:
- Fitting structural causal models (SCMs) to infer directed edges between variables
- Computing causal effects using do-calculus or other potential outcomes frameworks
- Validating via counterfactual simulations in CausalWorld

And then it would be time to modify the RL training loop to include interventions.
Such RL policies that have causal structure baked in them could exhibit positive transfer to new CausalWorld tasks.
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
from rlpyt.envs.gym import GymEnvWrapper
import numpy as np

def get_full_state():
    task = generate_task(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task, enable_visualization=True)
    for _ in range(1):
        env.reset()
        for _ in range(10):
            obs, reward, done, info = env.step(env.action_space.sample())
    print(env.get_current_state_variables())
    env.close()

def _make_env(rank):
    task = generate_task(task_generator_id='reaching')
    env = CausalWorld(task=task,
                      skip_frame=10,
                      enable_visualization=False,
                      seed=0 + rank,
                      max_episode_length=600)
    env = GymEnvWrapper(env)
    return env


def random_intervention():
    task = generate_task(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(50):
        random_intervention_dict, success_signal, obs = \
            env.do_single_random_intervention()
        print("The random intervention performed is ",
              random_intervention_dict)
        for i in range(100):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()

def without_intervention_split():
    task = generate_task(task_generator_id='pushing')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(2):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal, obs = env.do_intervention(
            {'stage_color': np.random.uniform(0, 1, [
                3,
            ])})
        print("Intervention success signal", success_signal)
    env.close()


def with_intervention_split_1():
    task = generate_task(task_generator_id='pushing',
                          variables_space='space_a')
    env = CausalWorld(task=task, enable_visualization=False)
    env.reset()
    for _ in range(2):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal, obs = env.do_intervention(
            {'stage_color': np.random.uniform(0, 1, [
                3,
            ])})
        print("Intervention success signal", success_signal)
    env.close()


def with_intervention_split_2():
    task = generate_task(task_generator_id='pushing',
                          variables_space='space_b')
    env = CausalWorld(task=task, enable_visualization=False)
    interventions_space = task.get_intervention_space_a()
    env.reset()
    for _ in range(2):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal, obs = env.do_intervention({
            'stage_color':
                np.random.uniform(interventions_space['stage_color'][0],
                                  interventions_space['stage_color'][1])
        })
        print("Intervention success signal", success_signal)
    env.close()

if __name__ == '__main__':
    # random_intervention()
    without_intervention_split()
    with_intervention_split_1()
    with_intervention_split_2()