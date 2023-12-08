from typing import Optional

import torch.nn as nn
import torch
import numpy as np
import numpy.typing as npt
from experiment.dqn_args import DQNArgs

from experiment.strategies.base_strategy import AbstractAgent
from experiment.environments.multi_agent_repeated_game import (
    MultiAgentRepeatedGameEnvironment,
    PHI_EMPTY_ACTION,
    ActionType,
    AgentID,
    ObsType,
)


class QNetwork(nn.Module):
    def __init__(self, env: MultiAgentRepeatedGameEnvironment, agent_id: int):
        super().__init__()

        agent_obseravtion_space_shape = env.observation_space(agent_id).shape
        flattened_input_size = np.array(agent_obseravtion_space_shape).prod()

        agent_num_actions = int(env.action_space(agent_id).n)

        self.network = nn.Sequential(
            nn.Linear(flattened_input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, agent_num_actions),
        )

    def forward(self, x: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNAgent(AbstractAgent):
    q_network: QNetwork
    optimizer: torch.optim.Adam
    target_netowkr: QNetwork

    args: DQNArgs

    def __init__(
        self,
        env: MultiAgentRepeatedGameEnvironment,
        agent_id: int,
        args: DQNArgs,
    ):
        super().__init__(env, agent_id)

        self.q_network = QNetwork(env, agent_id)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.target_network = QNetwork(env, agent_id)

        self.args = args

    def pick_action(self, global_step: int) -> npt.NDArray[ActionType]:
        epsilon = linear_schedule(
            self.args.start_e,
            self.args.end_e,
            self.args.exploration_fraction * self.args.total_timesteps,
            global_step,
        )
        actions: Optional[npt.NDArray[ActionType]] = None
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return cast(npt.NDArray[ActionType], actions)

    def add_to_replay(
        self,
        observation: dict[AgentID, ObsType],
        action: dict[AgentID, ActionType],
        reward: float,
        post_action_observation: dict[AgentID, ObsType],
    ) -> None:
        # Random agents don't have a replay buffer
        pass

    def train_from_replay(self) -> None:
        # Random agents don't train
        pass
