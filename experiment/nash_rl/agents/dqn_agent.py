import gymnasium as gym
import math
import random
from typing import Union, Generic

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from nash_rl.utils.experiment_args import ExperimentArgs
from nash_rl.agents.abstract_agent import AbstractAgent, ObsType, ActionType
from nash_rl.dqn_implementation.replay_memory import ReplayMemory, Transition


# ========== Neural Network Implementation ==========


class DQN(nn.Module, Generic[ObsType]):
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: ObsType) -> Union[ActionType, npt.NDArray[ActionType]]:
        return self.network(x)


# ========== Neural Network Utilities ==========


def make_agent_network_for_env(
    observation_space: gym.spaces.Space[ObsType], action_space: gym.spaces.Discrete
) -> DQN[ObsType]:
    n_observations = np.array(observation_space.shape).prod()
    n_actions = int(action_space.n)  # type: ignore

    return DQN(
        n_observations=n_observations,
        n_actions=n_actions,
    )


# ========== Agent Implementation ==========


class DQNAgent(AbstractAgent[ObsType]):
    device: torch.device  # Device

    env_observation_space: gym.spaces.Space[ObsType]  # Observation Space
    env_action_space: gym.spaces.Discrete  # Action Space

    experiment_args: ExperimentArgs

    # Networks
    policy_net: DQN[ObsType]
    target_net: DQN[ObsType]

    optimizer: optim.AdamW  # Optimizer
    memory: ReplayMemory  # Replay Memory

    # Training tracking
    steps_done: int

    def __init__(
        self,
        env_observation_space: gym.spaces.Space[ObsType],
        env_action_space: gym.spaces.Discrete,
        experiment_args: ExperimentArgs,
    ):
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_observation_space = env_observation_space
        self.env_action_space = env_action_space

        self.experiment_args = experiment_args

        self.policy_net = make_agent_network_for_env(
            self.env_observation_space, self.env_action_space
        ).to(self.device)
        self.target_net = make_agent_network_for_env(
            self.env_observation_space, self.env_action_space
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        LR = self.experiment_args.learning_rate
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        EPS_END = self.experiment_args.epsilon_end
        EPS_START = self.experiment_args.epsilon_start
        EPS_DECAY = self.experiment_args.epsilon_decay

        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1) + self.env_action_space.start # ! Added `self.env_action_space.start` to match the action space
        else:
            return torch.tensor(
                [[self.env_action_space.sample()]], device=self.device, dtype=torch.long
            )

    def push_memory(self, transition: Transition):
        self.memory.push(transition)

    def optimize_model(self):
        BATCH_SIZE = self.experiment_args.batch_size
        GAMMA = self.experiment_args.gamma

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update_target_weights(self):
        TAU = self.experiment_args.tau

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def learn(self):
        # Perform one step of the optimization (on the policy network)
        self.optimize_model()
        # Soft update of the target network's weights
        self.soft_update_target_weights()
