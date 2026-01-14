import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



from model.demand import inverse_demand
from model.payoff import profit

class CournotEnv:
    """
    Cournot competition environment for N firms.

    Firms choose quantities simultaneously.
    The environment computes market price and firm profits.

    State (default):
        [q_1(t-1), ..., q_N(t-1), p(t-1)]

    Action:
        q_i(t) in [0, q_max]

    Reward:
        profit_i(t) = (price - cost_i) * q_i
    """

    def __init__(
        self,
        n_firms: int,
        a: float,
        b: float,
        costs: list,
        q_max: float,
        horizon: int,
        seed: int | None = None
    ):
        if len(costs) != n_firms:
            raise ValueError("costs must match number of firms")

        self.n_firms = n_firms
        self.a = a                  # Demand intercept
        self.b = b                  # Demand slope
        self.costs = np.array(costs)
        self.q_max = q_max
        self.horizon = horizon

        self.rng = np.random.default_rng(seed)

        self.reset()

    # --------------------------------------------------
    # Core API
    # --------------------------------------------------

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns
        -------
        state : np.ndarray
            Initial observation
        """
        self.t = 0
        self.last_quantities = np.zeros(self.n_firms)
        self.last_price = self.a

        return self._get_state()

    def step(self, actions):
        """
        Executes one period of Cournot competition.

        Parameters
        ----------
        actions : array-like
            Quantities chosen by firms [q_1, ..., q_N]

        Returns
        -------
        next_state : np.ndarray
        rewards : np.ndarray
            Profits for each firm
        done : bool
            Whether episode has ended
        info : dict
            Additional diagnostics
        """
        actions = self._validate_actions(actions)

        # Aggregate quantity
        total_quantity = np.sum(actions)

        # Market price (non-negative)
        price = inverse_demand(
            total_quantity=total_quantity,
            a=self.a,
            b=self.b
        )

        # Firm profits
        rewards = np.array([
            profit(price, actions[i], self.costs[i])
            for i in range(self.n_firms)
        ])

        # Update internal state
        self.last_quantities = actions
        self.last_price = price
        self.t += 1

        done = self.t >= self.horizon

        info = {
            "price": price,
            "total_quantity": total_quantity,
            "time":self.t
        }

        return self._get_state(), rewards, done, info

    # --------------------------------------------------
    # Helper methods
    # --------------------------------------------------

    def _validate_actions(self, actions):
        """
        Ensures actions are valid quantities.
        """
        actions = np.asarray(actions, dtype=float)

        if actions.shape != (self.n_firms,):
            raise ValueError(f"Expected action shape ({self.n_firms},),got {actions.shape}")

        return np.clip(actions, 0.0, self.q_max)

    def _get_state(self):
        """
        Constructs the current state.
        """
        return np.concatenate([
            self.last_quantities,
            np.array([self.last_price])
        ])

    # --------------------------------------------------
    # Dimensions
    # --------------------------------------------------

    def state_dim(self):
        return self.n_firms + 1

    def action_dim(self):
        return self.n_firms

