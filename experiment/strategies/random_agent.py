from experiment.strategies.base_strategy import AbstractAgent
from experiment.environments.multi_agent_repeated_game import (
    MultiAgentRepeatedGameEnvironment,
    PHI_EMPTY_ACTION,
    ActionType,
    AgentID,
    ObsType,
)


class RandomAgent(AbstractAgent):
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
