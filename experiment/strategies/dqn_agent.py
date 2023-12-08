import torch.nn as nn
import torch
import numpy as np
import numpy.typing as npt

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
    return max((end_e - start_e) / duration * t + start_e, end_e)


class DQNAgent(AbstractAgent):
    q_network: QNetwork
    optimizer: torch.optim.Adam
    target_netowkr: QNetwork

    def __init__(self, env: MultiAgentRepeatedGameEnvironment, agent_id: int, args: DQNArgs):
        super().__init__(env, agent_id)

        self.q_network = QNetwork(env, agent_id)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.target_network = QNetwork(env, agent_id)

        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

    def pick_action(self) -> ActionType:
        return self.env.action_space(self.agent_id).sample()

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
