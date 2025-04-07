# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allows multiple OpenMP runtimes (unsafe but quick)
# TODO: install torch, scipy and rlpyt
from causal_world.envs import CausalWorld

import numpy as np
import torch
from torch.nn import Linear, Sequential, ReLU, Module
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.algos.pg.ppo import PPO
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper

# import sys
# sys.path.append(r"C:\Users\kausa\CausalWorld")


from causal_world.task_generators.task import generate_task


class CausalWorldRlpytEnv:
    """Wrapper for CausalWorld to work with rlpyt."""
    
    def __init__(self, task_id='stacked_blocks', enable_interventions=True, 
                 intervention_type='random', max_episode_length=600):
        self.task = generate_task(task_generator_id=task_id)
        self.env = CausalWorld(task=self.task,
                               skip_frame=10,
                               enable_visualization=False,
                               max_episode_length=max_episode_length)
        self.enable_interventions = enable_interventions
        self.intervention_type = intervention_type
        self.intervention_frequency = 100  # steps between interventions
        self.step_count = 0
        
        # Store the observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self):
        obs = self.env.reset()
        self.step_count = 0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        
        # Apply intervention based on strategy
        if self.enable_interventions and self.step_count % self.intervention_frequency == 0:
            if self.intervention_type == 'random':
                intervention_dict, success, obs = self.env.do_single_random_intervention()
                info['intervention'] = intervention_dict
                info['intervention_success'] = success
            elif self.intervention_type == 'systematic':
                # Example of systematic intervention on a specific variable
                intervention_dict = {'stage_color': np.random.uniform(0, 1, [3])}
                success, obs = self.env.do_intervention(intervention_dict)
                info['intervention'] = intervention_dict
                info['intervention_success'] = success
            elif self.intervention_type == 'adaptive':
                # This would require a more complex strategy based on uncertainty
                # For now, just use random as placeholder
                intervention_dict, success, obs = self.env.do_single_random_intervention()
                info['intervention'] = intervention_dict
                info['intervention_success'] = success
                
        return obs, reward, done, info
    
    def get_intervention_space(self):
        return self.task.get_intervention_space()
    
    def close(self):
        self.env.close()
        

class CausalMlpModel(Module):
    """MLP model for the policy and value function."""
    
    def __init__(self, observation_shape, action_size):
        super().__init__()
        
        # Determine input size from observation shape
        input_size = int(np.prod(observation_shape))
        
        # Policy network
        self.pi = Sequential(
            Linear(input_size, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, action_size)
        )
        
        # Value network
        self.value = Sequential(
            Linear(input_size, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 1)
        )
        
    def forward(self, observation, prev_action, prev_reward):
        # Flatten observation if needed
        if len(observation.shape) > 2:
            observation = observation.reshape(observation.shape[0], -1)
        else:
            observation = observation.view(observation.shape[0], -1)
            
        pi = self.pi(observation)
        value = self.value(observation).squeeze(-1)
        
        return pi, value


def build_and_train(task_id='stacked_blocks', 
                    intervention_type='random',
                    run_ID=0, 
                    cuda_idx=None,
                    n_parallel=4,
                    log_dir='./logs',
                    n_steps=1e6):
    """Build and train a PPO agent in CausalWorld with interventions."""
    
    # Create environment
    def make_env_fn():
        return GymEnvWrapper(CausalWorldRlpytEnv(
            task_id=task_id,
            enable_interventions=True,
            intervention_type=intervention_type
        ))
    
    # Create sampler
    if n_parallel > 1:
        sampler = CpuSampler(
            EnvCls=make_env_fn,
            env_kwargs={},
            batch_T=128,  # Timesteps per batch
            batch_B=n_parallel,  # Number of parallel environments
            max_decorrelation_steps=0
        )
    else:
        sampler = SerialSampler(
            EnvCls=make_env_fn,
            env_kwargs={},
            batch_T=128,
            batch_B=1,
        )
    
    # Create agent
    env = make_env_fn()
    observation_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]  # Continuous action space
    
    model = CausalMlpModel(
        observation_shape=observation_shape,
        action_size=action_size
    )
    
    agent = CategoricalPgAgent(
        ModelCls=CausalMlpModel,
        model_kwargs=dict(observation_shape=observation_shape, action_size=action_size)
    )
    
    # Create algorithm
    algo = PPO(
        learning_rate=3e-4,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        minibatches=4,
        epochs=4,
        ratio_clip=0.2,
        linear_lr_schedule=True,
        normalize_advantage=True,
    )
    
    # Create runner
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=10000,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    )
    
    # Configure logging
    config = dict(task_id=task_id, intervention_type=intervention_type)
    name = f"{task_id}_{intervention_type}"
    log_dir = f"{log_dir}/{name}"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()
    
    return runner


def analyze_causal_effects(agent, env, n_samples=1000):
    """Analyze causal effects using the trained agent."""
    # This is a placeholder for more sophisticated causal analysis
    
    # Get the intervention space
    intervention_space = env.get_intervention_space()
    
    # Collect data with interventions
    intervention_results = {}
    
    for var_name in list(intervention_space.keys())[:3]:  # Limit to a few variables for demonstration
        intervention_results[var_name] = []
        
        # Systematic intervention on this variable
        for _ in range(10):  # 10 different values
            if isinstance(intervention_space[var_name][0], (list, np.ndarray)):
                # For vector variables
                intervention_value = np.random.uniform(
                    intervention_space[var_name][0],
                    intervention_space[var_name][1]
                )
            else:
                # For scalar variables
                intervention_value = np.random.uniform(
                    intervention_space[var_name][0],
                    intervention_space[var_name][1]
                )
                
            intervention = {var_name: intervention_value}
            
            # Apply intervention and collect results
            obs = env.reset()
            success, obs = env.do_intervention(intervention)
            
            if not success:
                continue
                
            # Run agent for a few steps
            total_reward = 0
            for _ in range(100):
                action = agent.get_action(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
                    
            intervention_results[var_name].append((intervention_value, total_reward))
    
    # Simple analysis: print correlation between intervention values and rewards
    print("Causal Effect Analysis:")
    for var_name, results in intervention_results.items():
        if len(results) == 0:
            continue
            
        values = np.array([r[0] if not isinstance(r[0], (list, np.ndarray)) 
                           else np.mean(r[0]) for r in results])
        rewards = np.array([r[1] for r in results])
        
        correlation = np.corrcoef(values, rewards)[0, 1] if len(values) > 1 else 0
        print(f"Variable: {var_name}, Correlation with reward: {correlation:.4f}")
    
    return intervention_results


if __name__ == "__main__":
    # Train agents with different intervention strategies
    strategies = ['random', 'systematic', 'adaptive']
    tasks = ['stacked_blocks', 'pushing', 'reaching']
    
    # Choose one task and strategy for demonstration
    task_id = tasks[0]
    intervention_type = strategies[0]
    
    # Build and train the agent
    runner = build_and_train(
        task_id=task_id,
        intervention_type=intervention_type,
        n_parallel=4,  # Adjust based on your CPU
        n_steps=500000  # Reduced for demonstration
    )
    
    # Create environment for analysis
    env = CausalWorldRlpytEnv(
        task_id=task_id,
        enable_interventions=True,
        intervention_type=intervention_type
    )
    
    # Analyze causal effects
    causal_effects = analyze_causal_effects(runner.agent, env)
    
    # Close environment
    env.close()
    
    print(f"Completed training and analysis for {task_id} with {intervention_type} interventions")