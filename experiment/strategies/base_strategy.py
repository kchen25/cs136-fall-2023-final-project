from abc import ABC, abstractmethod

from experiment.environments.multi_agent_repeated_game import (
    MultiAgentRepeatedGameEnvironment,
    ActionType,
    ObsType,
    AgentID,
)


class AbstractAgent(ABC):
    env: MultiAgentRepeatedGameEnvironment
    agent_id: int

    def __init__(self, env: MultiAgentRepeatedGameEnvironment, agent_id: int):
        self.env = env
        self.agent_id = agent_id

    @abstractmethod
    def pick_action(self) -> ActionType:
        pass

    @abstractmethod
    def add_to_replay(
        self,
        observation: dict[AgentID, ObsType],
        action: dict[AgentID, ActionType],
        reward: float,
        post_action_observation: dict[AgentID, ObsType],
    ) -> None:
        pass

    @abstractmethod
    def train_from_replay(self) -> None:
        pass
