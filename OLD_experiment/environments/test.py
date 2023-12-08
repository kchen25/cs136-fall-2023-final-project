import json

import numpy as np
import numpy.typing as npt
import nashpy as nash

from experiment.environments.multi_agent_repeated_game import RepeatedGameEnvironment, get_game_rewards

# Utilities

def print_observations(obs: dict[int, npt.NDArray[np.int64]]):
    for agent_id in obs:
        print(f"- Agent {agent_id}:")
        print(obs[agent_id])

# Main Functionality

U_1 = np.array([[4, 1], [7, 2]])
U_2 = np.array([[4, 6], [1, 2]])
game = nash.Game(U_1, U_2)
print(game)

rge = RepeatedGameEnvironment(game, 3)

print(rge)

# print(get_game_rewards(game, tuple((0, 1))))

rge.reset()

for i, plays in enumerate([(1,1), (1, 2), (2, 1)]):
    (observations, rewards, terminations, truncations, agent_infos) = rge.step(plays)
    print(f"========== Step {i} ==========")
    print("Observations:")
    print_observations(observations)
    print("Rewards:")
    print(rewards)
    print("Terminations:")
    print(terminations)
    print("Truncations:")
    print(truncations)
    print("Agent Infos:")
    print(agent_infos)
