import torch
import gymnasium as gym

from nash_rl.utils.experiment_args import ExperimentArgs
from nash_rl.dqn_implementation.replay_memory import Transition
from nash_rl.agents.abstract_agent import AbstractAgent, ObsType


class PavlovAgent(AbstractAgent[ObsType]):
    def __init__(
        self,
        env_observation_space: gym.spaces.Space[ObsType],
        env_action_space: gym.spaces.Discrete,
        experiment_args: ExperimentArgs,
    ):
        super().__init__(
            env_observation_space, env_action_space, experiment_args
        )  # TODO Fix type error

    # Cooperates with probability 1 if (C,C), 0 if (C, D) or (D,C), and approx. 1 if (D,D)
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        # This assumes that it is only playing against a single opponent, if more, it grabs the last action of the last opponent
        if state[:, -2:-1] == state[:, -1:] and state[:, -1:] == 2: # (D,D)
            return torch.bernoulli(torch.tensor([0.95])).long() # cooperate with probability 0.95
        elif state[:, -2:-1] == state[:, -1:]:
            return torch.tensor([[0]]).long() # cooperate with probability 1
        else: 
            return torch.tensor([[1]]).long() #defect

    def push_memory(self, transition: Transition):
        # TfT agents don't have a replay buffer
        pass

    def learn(self):
        # TfT agents do not learn
        pass
