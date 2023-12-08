import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count
from typing import cast, Union, Optional

import numpy as np
import numpy.typing as npt
import torch
from dqn_agent import DQNAgent

from replay_memory import Transition
from experiment_args import ExperimentArgs
from plotting import plot_durations

ObsType = npt.NDArray[Union[np.float64, np.int64]]
ActionType = np.int64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env: gym.Env[npt.NDArray[np.float64], np.int64] = gym.make("CartPole-v1")

plt.ion()

EXPERIMENT_ARGS = ExperimentArgs()

# Get the number of state observations
state, info = env.reset()

env_observation_space = env.observation_space
env_action_space = cast(gym.spaces.Discrete, env.action_space)

agent = DQNAgent(env)


episode_durations: list[int] = []


num_episodes: int = (
    EXPERIMENT_ARGS.num_episodes_with_cuda
    if torch.cuda.is_available()
    else EXPERIMENT_ARGS.num_episodes_without_cuda
)

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

        next_state: Optional[torch.Tensor] = None
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

        # Perform necessary updates to learn from replay memory and update networks
        agent.learn()

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

print("Complete")
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()
