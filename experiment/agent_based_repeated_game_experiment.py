import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count
from typing import cast, Union, Optional

import numpy as np
import numpy.typing as npt
import torch

from enviornments.multi_agent_repeated_game import (
    MultiAgentRepeatedGameEnvironment,
    ObsType,
    ActionType,
    PHI_EMPTY_ACTION,
)

from strategies.base_strategy import AbstractAgent
from games import ipd_game

from strategies.dqn_agent import DQNAgent
from strategies.tft_agent import TFTAgent
from pytorch_dqn.replay_memory import Transition
from experiment_args import ExperimentArgs
from utils.plotting import plot_scores

NUM_AGENTS = 2  # ! DO NOT EDIT, NASHPY ONLY SUPPORTS 2 AGENTS

# ! ========== SET EXPERIMENT ARGS HERE ==========

AGENT_MEMORY_LENGTH = 5  # ! Edit for experiments
GAME = ipd_game  # ! Edit to change games

EXPERIMENT_ARGS = ExperimentArgs()
EXPERIMENT_ARGS.gamma = 0.99  # ! Set up discount factor; edit for experiments

# ! ========== NO MORE EXPERIMENT ARGS BELOW THIS LINE ==========


ObsType = npt.NDArray[Union[np.float64, np.int64]]
ActionType = np.int64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

repeated_game = GAME

parallel_env = MultiAgentRepeatedGameEnvironment(
    repeated_game, agent_memory_length=AGENT_MEMORY_LENGTH, max_rounds=500
)

plt.ion()


# Get the number of state observations
states, infos = parallel_env.reset()

# ! ========== MAKE AGENTS HERE ==========

agent_0 = DQNAgent(
    parallel_env.observation_space(0),
    cast(gym.spaces.Discrete, parallel_env.action_space(0)),
    EXPERIMENT_ARGS,
)
agent_1 = TFTAgent(
    parallel_env.observation_space(1),
    cast(gym.spaces.Discrete, parallel_env.action_space(1)),
    EXPERIMENT_ARGS,
)

agents: dict[int, AbstractAgent[ObsType]] = {
    0: agent_0,
    1: agent_1,
}

# ! ========== DO NOT EDIT BELOW THIS LINE ==========

# TODO Replace this for something we would like to measure about the games
episode_scores: list[dict[int, float]] = []

num_episodes: int = (
    EXPERIMENT_ARGS.num_episodes_with_cuda
    if torch.cuda.is_available()
    else EXPERIMENT_ARGS.num_episodes_without_cuda
)

for i_episode in range(num_episodes):
    print(f"Episode {i_episode + 1}/{num_episodes}")
    # Initialize the environment and get it's state
    states, infos = parallel_env.reset()
    tensor_states: dict[int, torch.Tensor] = {
        agent_id: (
            torch.tensor(
                states[agent_id], dtype=torch.float32, device=device
            ).unsqueeze(0)
        )
        for agent_id in states.keys()
    }
    for t in count():
        actions = {
            agent_id: agents[agent_id].select_action(tensor_states[agent_id])
            for agent_id in tensor_states.keys()
        }
        action_ids = {
            agent_id: cast(np.int64, action.item())
            for agent_id, action in actions.items()
        }

        observations, rewards, terminations, truncations, infos = parallel_env.step(
            action_ids
        )

        tensor_rewards = {
            agent_id: torch.tensor([reward], device=device)
            for agent_id, reward in rewards.items()
        }

        terminated = any(term for term in terminations.values())
        truncated = any(trunc for trunc in truncations.values())
        done = terminated or truncated

        next_states: Optional[dict[int, torch.Tensor]] = None
        if terminated:
            next_states = None
        else:
            next_states = {
                agent_id: torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                for agent_id, observation in observations.items()
            }

        # Store the transition in memory
        if next_states is not None:
            for agent_id, agent in agents.items():
                tensor_state = tensor_states[agent_id]
                action = actions[agent_id]
                next_state = next_states[agent_id]
                tensor_reward = tensor_rewards[agent_id]
                agent.push_memory(
                    Transition(tensor_state, action, next_state, tensor_reward)
                )

        # Move to the next state
        tensor_states = cast(
            dict[int, torch.Tensor], next_states
        )  # The cast is necessary because `next_states` can be `None`, but this does not matter because when that happens the loop will break

        # Perform necessary updates to learn from replay memory and update networks
        for agent in agents.values():
            agent.learn()

        if done:
            new_episode_score = {
                agent_id: infos[agent_id]["total_reward"] for agent_id in rewards.keys()
            }
            episode_scores.append(
                new_episode_score
            )  # TODO Think of a more useful metric
            plot_scores(episode_scores)
            break

print("Complete")
plot_scores(episode_scores, show_result=True)
plt.ioff()
plt.show()
