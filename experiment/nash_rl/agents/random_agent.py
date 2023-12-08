import torch
import gymnasium as gym

from nash_rl.utils.experiment_args import ExperimentArgs
from nash_rl.dqn_implementation.replay_memory import Transition
from nash_rl.agents.abstract_agent import AbstractAgent, ObsType


class RandomAgent(AbstractAgent[ObsType]):
    def __init__(
        self,
        env_observation_space: gym.spaces.Space[ObsType],
        env_action_space: gym.spaces.Discrete,
        experiment_args: ExperimentArgs,
    ):
        super().__init__(
            env_observation_space, env_action_space, experiment_args
        )  # TODO Fix type error

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Return random answer
        return torch.tensor(
            [[self.env_action_space.sample()]], device=device, dtype=torch.long
        )

    def push_memory(self, transition: Transition):
        # Random agents don't have a replay buffer
        pass

    def learn(self):
        # Random agents do not learn
        pass
