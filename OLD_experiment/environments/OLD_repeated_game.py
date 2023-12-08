from typing import Any, Literal, TypedDict, Union

import numpy as np
import numpy.typing as npt
import pygame
import gymnasium as gym
from gymnasium import spaces


class RepeatedGameEnvObservation(TypedDict):
    """Dictionary containing the locations of the agent and the target."""

    agent: tuple[int, int]
    """The agent's location."""
    target: tuple[int, int]
    """The target's location."""


RepeatedGameEnvRenderMode = str | None

RewardType = tuple[int]


class RepeatedGameEnvInfo(TypedDict):
    """Dictionary containing auxiliary information about the environment's internal state."""

    distance: float
    """The L1 distance between the agent's and the target's location."""


class RepeatedGameEnv(gym.Env[RepeatedGameEnvObservation, RepeatedGameEnvInfo]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    size: int
    """The size of the square grid."""
    window_size: int = 512
    """The size of the PyGame window."""
    render_mode: RepeatedGameEnvRenderMode = None
    """The render mode to use. Can be `None` or `"human"`."""
    _action_to_direction: dict[int, npt.NDArray[np.int64]] = {
        0: np.array([1, 0]),  # right
        1: np.array([0, 1]),  # up
        2: np.array([-1, 0]),  # left
        3: np.array([0, -1]),  # down
    }
    """The following dictionary maps abstract actions from `self.action_space` to the direction we will walk in if that action is taken (e.g., 0 corresponds to "right", 1 to "up")"""

    def __init__(self, render_mode: Literal["human"] | None = None, size=5):
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size` - 1}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to the direction we will walk in if that action is taken (e.g., 0 corresponds to "right", 1 to "up")
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-renderin is used, `self.window` will be a reference to the window that we draw to.
        `self.clock` wil be a clock that is used to ensure that the environment is rendered at the
        correct framerate in human-mode. They will remain `None` until human-mode is used for the first time.
        """
        self.window = None
        self.clock = None

        self._agent_location = None  # TODO Look at best practices for initialization
        self._target_location = None  # TODO Idem

    def _get_obs(self) -> Observation:
        """
        Compute the current observation from the environment's internal state.
        """
        return {
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def _get_info(self):
        """
        Compute current auxiliary information from the environment's internal state.
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location,
                # Compute the L1 norm (Manhattan distance) between the agent's and the target's location
                ord=1,
            )
        }

    def reset(self, seed=None, options=None):
        """
        awd
        """
        super().reset(seed=seed)  # Reset the random number generator in the base class

        # Choose the agent's location unifornmly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's locations randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return (observation, info)

    def step(self, action) -> tuple[Observation, Reward, bool, bool, Info]:
        # Map the action(element of {0, 1, 2, 3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        # TODO Add a discount factor here
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return (observation, reward, terminated, False, info)
