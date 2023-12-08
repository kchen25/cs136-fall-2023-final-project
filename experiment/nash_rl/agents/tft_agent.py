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
        super().__init__(env_observation_space, env_action_space, experiment_args) # TODO Fix type error

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        # TODO Write code to access the oponents previous action
        # Do that action unless that action is 0, in which case cooperate
        previous_action = state[0][0].item()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Return random answer
        return torch.tensor(
            [[self.env_action_space.sample()]], device=device, dtype=torch.long
        )

    def push_memory(self, transition: Transition):
        # TfT agents don't have a replay buffer
        pass

    def learn(self):
        # TfT agents do not learn
        pass