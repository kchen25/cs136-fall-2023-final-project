import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count
from typing import cast, Union

import numpy as np
import numpy.typing as npt
import torch
from experiment.strategies.dqn_agent import DQNAgent

from replay_memory import Transition
from plotting import plot_durations

ObsType = npt.NDArray[Union[np.float64, np.int64]]
ActionType = np.int64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env: gym.Env[npt.NDArray[np.float64], np.int64] = gym.make("CartPole-v1")

plt.ion()


BATCH_SIZE = 128
"""BATCH_SIZE is the number of transitions sampled from the replay buffer"""
GAMMA = 0.99
"""GAMMA is the discount factor as mentioned in the previous section"""
EPS_START = 0.9  # ! Original was 0.9
"""EPS_START is the starting value of epsilon"""
EPS_END = 0.05
"""EPS_END is the final value of epsilon"""
EPS_DECAY = 1000
"""EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay"""
TAU = 0.005  # ! Original was 0.005
"""TAU is the update rate of the target network"""
LR = 2.5e-4
"""LR is the learning rate of the ``AdamW`` optimizer"""

NUM_EPISODES: int = 500
"""NUM_EPISODES is the number of episodes for which to run the environment"""

# Get the number of state observations
state, info = env.reset()

env_observation_space = env.observation_space
env_action_space = cast(gym.spaces.Discrete, env.action_space)

agent = DQNAgent(env)


episode_durations: list[int] = []


if torch.cuda.is_available():  # TODO Change based on CUDA availability
    num_episodes = NUM_EPISODES
else:
    num_episodes = NUM_EPISODES

for i_episode in range(num_episodes):
    print(f"Episode {i_episode + 1}/{num_episodes}")
    # Initialize the environment and get it's state
    state, info = env.reset()
    tensor_state: torch.Tensor = torch.tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)
    for t in count():
        action = agent.select_action(tensor_state)
        action_id = cast(np.int64, action.item())
        observation, reward, terminated, truncated, _ = env.step(action_id)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Store the transition in memory
        agent.push_memory(Transition(tensor_state, action, next_state, reward))

        # Move to the next state
        tensor_state = cast(
            torch.Tensor, next_state
        )  # The cast is necessary because `next_state` can be `None`, but this does not matter because when that happens the loop will break

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        agent.soft_update_target_weights()

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

print("Complete")
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()
