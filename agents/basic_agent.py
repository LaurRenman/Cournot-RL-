"""
Basic agent definitions for Cournot reinforcement learning.

This file defines:
1. A BaseAgent interface that all agents must follow
2. A simple RandomAgent used as a baseline
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Agents:
    - observe a state
    - choose an action
    - update internal parameters based on rewards
    """

    def __init__(self, n_actions: int, q_max: float):
        """
        Parameters
        ----------
        n_actions : int
            Dimension of the action space (number of firms or outputs)
        q_max : float
            Maximum allowable quantity
        """
        self.n_actions = n_actions
        self.q_max = q_max

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Choose an action given the current state.

        Parameters
        ----------
        state : np.ndarray
            Current observation from the environment

        Returns
        -------
        np.ndarray
            Action vector (quantities)
        """
        pass

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Update internal parameters after observing a transition.
        """
        pass

    def reset(self):
        """
        Reset agent state at the beginning of a new episode.
        Optional for simple agents.
        """
        pass


class RandomAgent(BaseAgent):
    """
    Baseline agent that chooses random quantities uniformly.
    """

    def __init__(self, n_actions: int, q_max: float, seed: int | None = None):
        super().__init__(n_actions, q_max)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select random quantities independently for each action dimension.
        """
        return self.rng.uniform(0.0, self.q_max, size=self.n_actions)

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Random agent does not learn.
        """
        pass
