"""
SARSA Agent for Cournot competition.

This agent uses the SARSA (State-Action-Reward-State-Action) algorithm
with discretized action space and epsilon-greedy exploration.
"""

import numpy as np
from agents.basic_agent import BaseAgent


class SARSAAgent(BaseAgent):
    """
    SARSA agent with Q-learning for Cournot competition.
    
    Uses:
    - Discretized action space (quantity bins)
    - Epsilon-greedy exploration
    - Q-table for state-action values
    """

    def __init__(
        self,
        n_actions: int,
        q_max: float,
        n_quantity_bins: int = 20,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int | None = None
    ):
        """
        Parameters
        ----------
        n_actions : int
            Number of action dimensions (should be 1 for single firm)
        q_max : float
            Maximum quantity
        n_quantity_bins : int
            Number of discrete quantity levels to choose from
        alpha : float
            Learning rate
        gamma : float
            Discount factor
        epsilon : float
            Initial exploration rate
        epsilon_decay : float
            Decay rate for epsilon after each episode
        epsilon_min : float
            Minimum epsilon value
        seed : int, optional
            Random seed
        """
        super().__init__(n_actions, q_max)
        
        self.n_quantity_bins = n_quantity_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.rng = np.random.default_rng(seed)
        
        # Discretize action space
        self.action_space = np.linspace(0, q_max, n_quantity_bins)
        
        # Q-table: discretized state -> action index -> Q-value
        # State discretization bins
        self.n_state_bins = 10
        self.state_bins = self._create_state_bins()
        
        # Initialize Q-table
        # Dimensions: [quantity_bins]^n_firms x [price_bins] x [action_bins]
        self.q_table = {}
        
        # Track last action for SARSA updates
        self.last_action_idx = None
        
    def _create_state_bins(self):
        """Create bins for discretizing continuous state."""
        return {
            'quantity': np.linspace(0, self.q_max, self.n_state_bins),
            'price': np.linspace(0, 100, self.n_state_bins)  # Assume max price ~100
        }
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Convert continuous state to discrete state tuple.
        
        Parameters
        ----------
        state : np.ndarray
            [q_1(t-1), ..., q_N(t-1), p(t-1)]
            
        Returns
        -------
        tuple
            Discretized state representation
        """
        discrete_state = []
        
        # Discretize quantities (all but last element)
        for q in state[:-1]:
            q_idx = np.digitize(q, self.state_bins['quantity'])
            discrete_state.append(q_idx)
        
        # Discretize price (last element)
        price = state[-1]
        price_idx = np.digitize(price, self.state_bins['price'])
        discrete_state.append(price_idx)
        
        return tuple(discrete_state)
    
    def _get_q_value(self, state_tuple: tuple, action_idx: int) -> float:
        """Get Q-value for state-action pair."""
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.n_quantity_bins)
        return self.q_table[state_tuple][action_idx]
    
    def _set_q_value(self, state_tuple: tuple, action_idx: int, value: float):
        """Set Q-value for state-action pair."""
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.n_quantity_bins)
        self.q_table[state_tuple][action_idx] = value
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state observation
            
        Returns
        -------
        np.ndarray
            Selected quantity (continuous value)
        """
        state_tuple = self._discretize_state(state)
        
        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon:
            # Explore: random action
            action_idx = self.rng.integers(0, self.n_quantity_bins)
        else:
            # Exploit: best action
            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = np.zeros(self.n_quantity_bins)
            action_idx = np.argmax(self.q_table[state_tuple])
        
        # Store for SARSA update
        self.last_action_idx = action_idx
        
        # Return continuous quantity value
        quantity = self.action_space[action_idx]
        return np.array([quantity])
    
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        SARSA update rule.
        
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        
        where a' is the action actually taken in state s'
        """
        if self.last_action_idx is None:
            return
        
        state_tuple = self._discretize_state(state)
        current_q = self._get_q_value(state_tuple, self.last_action_idx)
        
        if done:
            # Terminal state: no future reward
            target = reward
        else:
            # Need next action for SARSA
            # Select next action (already stored in last_action_idx from next select_action call)
            # For SARSA, we use the action we'll actually take in next_state
            next_state_tuple = self._discretize_state(next_state)
            
            # Epsilon-greedy for next action
            if self.rng.random() < self.epsilon:
                next_action_idx = self.rng.integers(0, self.n_quantity_bins)
            else:
                if next_state_tuple not in self.q_table:
                    self.q_table[next_state_tuple] = np.zeros(self.n_quantity_bins)
                next_action_idx = np.argmax(self.q_table[next_state_tuple])
            
            next_q = self._get_q_value(next_state_tuple, next_action_idx)
            target = reward + self.gamma * next_q
        
        # SARSA update
        new_q = current_q + self.alpha * (target - current_q)
        self._set_q_value(state_tuple, self.last_action_idx, new_q)
    
    def reset(self):
        """Reset episode-specific variables."""
        self.last_action_idx = None
        
        # Decay epsilon after each episode
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)