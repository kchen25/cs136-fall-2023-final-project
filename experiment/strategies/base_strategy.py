from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from experiment.experiment_args import ExperimentArgs
from experiment.pytorch_dqn.replay_memory import Transition

ObsType = TypeVar("ObsType", npt.NDArray[np.int64], npt.NDArray[np.float64])
ActionType = np.int64


class AbstractAgent(ABC, Generic[ObsType]):
    env_observation_space: gym.spaces.Space[ObsType]
    env_action_space: gym.spaces.Discrete
    experiment_args: ExperimentArgs

    def __init__(
        self,
        env_observation_space: gym.spaces.Space[ObsType],
        env_action_space: gym.spaces.Discrete,
        experiment_args: ExperimentArgs,
    ):
        self.env_observation_space = env_observation_space
        self.env_action_space = env_action_space
        self.experiment_args = experiment_args

    @abstractmethod
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def push_memory(self, transition: Transition):
        pass

    @abstractmethod
    def learn(self):
        pass
