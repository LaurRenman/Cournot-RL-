"""
Cournot Competition Environment Module

This module defines the market environment for Cournot competition,
including demand functions, cost structures, and Nash equilibrium calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class CournotEnvironment:
    """
    Environment for Cournot competition with stochastic demand and costs.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Cournot environment.
        
        Args:
            config: Dictionary containing environment parameters
        """
        self.config = config or self.get_default_config()
        self.rng = np.random.RandomState(self.config.get("seed", 42))
        
    @staticmethod
    def get_default_config() -> Dict:
        """Return default environment configuration."""
        return {
            "a_min": 80.0,      # minimum demand intercept
            "a_max": 120.0,     # maximum demand intercept
            "b_min": 0.8,       # minimum demand slope
            "b_max": 1.2,       # maximum demand slope
            "cost_min": 5.0,    # minimum marginal cost
            "cost_max": 100.0,  # maximum marginal cost
            "q_max": 100.0,     # maximum quantity constraint
            "seed": 42
        }
    
    def sample_scenario(self, n_players: int) -> Tuple[float, float, np.ndarray]:
        """
        Sample a random market scenario.
        
        Args:
            n_players: Number of firms in the market
            
        Returns:
            Tuple of (demand_intercept, demand_slope, marginal_costs)
        """
        a = self.rng.uniform(self.config["a_min"], self.config["a_max"])
        b = self.rng.uniform(self.config["b_min"], self.config["b_max"])
        costs = self.rng.uniform(
            self.config["cost_min"], 
            self.config["cost_max"], 
            size=n_players
        )
        return a, b, costs
    
    def price(self, quantities: np.ndarray, a: float, b: float) -> float:
        """
        Calculate market price given total quantity.
        
        Args:
            quantities: Array of quantities produced by each firm
            a: Demand intercept
            b: Demand slope
            
        Returns:
            Market price
        """
        Q_total = np.sum(quantities)
        return max(a - b * Q_total, 0.0)
    
    def profit(self, quantities: np.ndarray, costs: np.ndarray, 
               a: float, b: float) -> np.ndarray:
        """
        Calculate profits for all firms.
        
        Args:
            quantities: Array of quantities produced by each firm
            costs: Array of marginal costs for each firm
            a: Demand intercept
            b: Demand slope
            
        Returns:
            Array of profits for each firm
        """
        p = self.price(quantities, a, b)
        return (p - costs) * quantities
    
    def best_response(self, player_idx: int, other_quantities: np.ndarray,
                     cost: float, a: float, b: float) -> float:
        """
        Calculate best response quantity for a player.
        
        Args:
            player_idx: Index of the player
            other_quantities: Quantities of other players
            cost: Marginal cost of the player
            a: Demand intercept
            b: Demand slope
            
        Returns:
            Best response quantity
        """
        Q_minus_i = np.sum(other_quantities)
        q_br = max((a - b * Q_minus_i - cost) / (2 * b), 0.0)
        return min(q_br, self.config["q_max"])
    
    def nash_equilibrium(self, a: float, b: float, costs: np.ndarray) -> np.ndarray:
        """
        Calculate symmetric Nash equilibrium quantities.
        
        Args:
            a: Demand intercept
            b: Demand slope
            costs: Array of marginal costs for each firm
            
        Returns:
            Array of Nash equilibrium quantities
        """
        costs = np.array(costs)
        n = len(costs)
        total_cost = np.sum(costs)
        q_star = (a - (n + 1) * costs + total_cost) / (b * (n + 1))
        q_star = np.maximum(q_star, 0.0)
        q_star = np.minimum(q_star, self.config["q_max"])
        return q_star
    
    def create_state(self, a: float, b: float, cost: float) -> np.ndarray:
        """
        Create state representation for a single agent.
        
        Args:
            a: Demand intercept
            b: Demand slope
            cost: Agent's marginal cost
            
        Returns:
            State vector [a, b, cost]
        """
        return np.array([a, b, cost])
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.config["seed"] = seed
        self.rng = np.random.RandomState(seed)
