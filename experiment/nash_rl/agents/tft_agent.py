import torch
import gymnasium as gym

from nash_rl.utils.experiment_args import ExperimentArgs
from nash_rl.dqn_implementation.replay_memory import Transition
from nash_rl.agents.abstract_agent import AbstractAgent, ObsType


class TFTAgent(AbstractAgent[ObsType]):
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
        previous_actions = state[:, -1:] # This assumes that it is only playing against a single opponent, if more, it grabs the last action of the last opponent
        previous_actions[previous_actions == 0] = 1
        previous_actions = previous_actions - 1
        previous_actions = previous_actions.long()
        return previous_actions

    def push_memory(self, transition: Transition):
        # TfT agents don't have a replay buffer
        pass

    def learn(self):
        # TfT agents do not learn
        pass
