import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from itertools import count
from typing import cast, Any, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from replay_memory import ReplayMemory, Transition
from experiment.experiment_args import ExperimentArgs
from plotting import plot_durations

ObsType = npt.NDArray[Union[np.float64, np.int64]]
ActionType = np.int64

env: gym.Env[npt.NDArray[np.float64], np.int64] = gym.make("CartPole-v1")

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
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


def make_agent_network_for_env(
    observation_space: gym.spaces.Space[ObsType], action_space: gym.spaces.Discrete
) -> DQN:
    n_observations = np.array(observation_space.shape).prod()
    n_actions = int(action_space.n)  # type: ignore

    return DQN(
        n_observations=n_observations,
        n_actions=n_actions,
    )

experiment_args = ExperimentArgs

# Get the number of state observations
state, info = env.reset()

env_observation_space = env.observation_space
env_action_space = cast(gym.spaces.Discrete, env.action_space)

policy_net = make_agent_network_for_env(env_observation_space, env_action_space).to(
    device
)
target_net = make_agent_network_for_env(env_observation_space, env_action_space).to(
    device
)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state: torch.Tensor) -> torch.Tensor:
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


episode_durations: list[int] = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():  # TODO Change based on CUDA availability
    num_episodes = NUM_EPISODES
else:
    num_episodes = NUM_EPISODES

for i_episode in range(num_episodes):
    print(f"Episode {i_episode + 1}/{num_episodes}")
    # Initialize the environment and get it's state
    state, info = env.reset()
    tensor_state: torch.Tensor = torch.tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)
    for t in count():
        action = select_action(tensor_state)
        action_id = cast(np.int64, action.item())
        observation, reward, terminated, truncated, _ = env.step(action_id)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Store the transition in memory
        memory.push(Transition(tensor_state, action, next_state, reward))

        # Move to the next state
        tensor_state = cast(torch.Tensor, next_state) # The cast is necessary because `next_state` can be `None`, but this does not matter because when that happens the loop will break

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

print("Complete")
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()
