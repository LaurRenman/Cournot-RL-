"""
Trainer Module with Regret-Based Learning

This module implements regret minimization for multi-agent Cournot competition.
Now supports agents with varying information levels:
- 4D (Nash): [a, b, own_cost, opponent_cost] - Full information
- 3D (Asymmetric): [a, b, own_cost] - Knows demand, not opponent cost
- 1D (Minimal): [own_cost] - Only knows own cost, no demand info

KEY INSIGHT: Why normalization is necessary
--------------------------------------------
Raw regret values are often small (e.g., 0.1-5.0 profit units) because:
1. With exploration noise, agents often play near-optimal actions
2. The profit function is smooth, so small deviations = small regret

Problem: advantage = -0.5 creates tiny gradients → slow/no learning

Solution: Normalize regret to create consistent gradient magnitude while
preserving the regret minimization objective. This is NOT a bias toward Nash,
but a numerical scaling to make optimization tractable.

At equilibrium: regret → 0, so normalized regret → 0, gradient → 0 (fixed point)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from environment import CournotEnvironment
from policy import PolicyNetwork_asymmetrical, PolicyNetwork_Nash


class CournotTrainer:
    """
    Universal trainer for multi-agent Cournot competition with regret minimization.
    Automatically detects and handles 1D, 3D, and 4D policy inputs.
    """
    
    def __init__(self, n_players: int, env: CournotEnvironment, 
                 policies: List[Union[PolicyNetwork_asymmetrical, PolicyNetwork_Nash]], 
                 config: Optional[Dict] = None):
        """
        Initialize trainer.
        
        Args:
            n_players: Number of players/agents
            env: Cournot environment instance
            policies: List of policy networks (can be 1D, 3D, or 4D)
            config: Training configuration
        """
        self.n_players = n_players
        self.env = env
        self.policies = policies
        self.config = config or self.get_default_config()
        
        # Self-play configuration
        self.self_play_mode = self.config.get('opponent_update_interval', None) is not None
        self.opponent_update_interval = self.config.get('opponent_update_interval', 1000)
        self.learning_agent_id = self.config.get('learning_agent_id', 0)
        
        # Regret tracking (EMA for smoothing)
        self.regret_ema = np.zeros(n_players)
        self.regret_alpha = self.config.get('regret_alpha', 0.01)
        
        # Regret normalization for stable learning (RECOMMENDED)
        self.normalize_regret = self.config.get('normalize_regret', True)
        self.regret_mean = np.zeros(n_players)
        self.regret_std = np.ones(n_players)
        self.stats_alpha = self.config.get('stats_alpha', 0.01)
        
        # Exploration schedule
        self.sigma = np.full(n_players, self.config['sigma_init'])
        
        # History tracking
        self.history = {
            'actions': [],
            'rewards': [],
            'policy_means': [],
            'costs': [],
            'demand_a': [],
            'demand_b': [],
            'sigma': [],
            'learning_rate': [],
            'opponent_updates': [],
            'regret': [],  # Instantaneous regret
            'regret_ema': [],  # Exponential moving average
            'cumulative_regret': []  # Running sum
        }
        
        # Cumulative regret tracking
        self.cumulative_regret = np.zeros(n_players)
    
    @staticmethod
    def get_default_config() -> Dict:
        """Return default training configuration."""
        return {
            'episodes': 150000,
            'lr_init': 0.003,
            'lr_final': 0.0001,
            'warmup_steps': 10000,
            'sigma_init': 12.0,
            'sigma_final': 3.0,
            'sigma_decay': 0.99996,
            'regret_alpha': 0.01,  # EMA smoothing for regret
            'normalize_regret': True,  # STRONGLY RECOMMENDED for stable learning
            'stats_alpha': 0.01,
            'print_interval': 15000
        }
    
    def copy_policy_parameters(self, source_idx: int, target_idx: int):
        """
        Copy parameters from source policy to target policy.
        
        Args:
            source_idx: Index of source policy
            target_idx: Index of target policy
        """
        source = self.policies[source_idx]
        target = self.policies[target_idx]
        
        target.w_linear = source.w_linear.copy()
        target.b_linear = source.b_linear
        target.W1 = source.W1.copy()
        target.b1 = source.b1.copy()
        target.W2 = source.W2.copy()
        target.b2 = source.b2.copy()
        target.input_mean = source.input_mean.copy()
        target.input_std = source.input_std.copy()
        target._init_adam_params()
    
    def get_learning_rate(self, episode: int) -> float:
        """Calculate learning rate with warmup schedule."""
        warmup = self.config['warmup_steps']
        total = self.config['episodes']
        lr_init = self.config['lr_init']
        lr_final = self.config['lr_final']
        
        if episode < warmup:
            return lr_init * (episode / warmup)
        else:
            progress = (episode - warmup) / (total - warmup)
            return lr_final + (lr_init - lr_final) * (1 - progress) ** 0.9
    
    def decay_exploration(self):
        """Decay exploration noise."""
        decay = self.config['sigma_decay']
        sigma_final = self.config['sigma_final']
        self.sigma = self.sigma * decay
        self.sigma = np.maximum(self.sigma, sigma_final)
    
    def compute_regret(self, actions: np.ndarray, a: float, b: float, 
                       costs: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous regret for each agent.
        
        Regret_i = max_{q'_i} π_i(q'_i, q_{-i}) - π_i(q_i, q_{-i})
        
        Given opponents' realized actions q_{-i}, find the best response and
        compute how much additional profit could have been earned.
        
        Args:
            actions: Realized actions of all agents
            a: Demand intercept
            b: Demand slope
            costs: Agent costs
            rewards: Realized profits (already computed)
            
        Returns:
            Array of regret values (always >= 0)
        """
        regret = np.zeros(self.n_players)
        
        for i in range(self.n_players):
            # Sum of opponents' actual quantities (fixed/observed)
            Q_minus_i = actions.sum() - actions[i]
            
            # Best response: maximize π_i(q_i, Q_minus_i) over q_i
            # Solution: q*_i = max((a - b*Q_minus_i - c_i) / (2b), 0)
            q_optimal = (a - b * Q_minus_i - costs[i]) / (2 * b)
            q_optimal = np.clip(q_optimal, 0.0, self.env.config["q_max"])
            
            # Price with optimal quantity
            Q_total_optimal = Q_minus_i + q_optimal
            price_optimal = max(a - b * Q_total_optimal, 0.0)
            
            # Optimal profit (counterfactual)
            profit_optimal = (price_optimal - costs[i]) * q_optimal
            
            # Regret = opportunity cost
            regret[i] = max(profit_optimal - rewards[i], 0.0)
        
        return regret
    
    def update_regret_stats(self, regret: np.ndarray):
        """Update running statistics for regret tracking and normalization."""
        alpha = self.regret_alpha
        self.regret_ema = (1 - alpha) * self.regret_ema + alpha * regret
        self.cumulative_regret += regret
        
        # Always update normalization stats
        alpha_stats = self.stats_alpha
        self.regret_mean = (1 - alpha_stats) * self.regret_mean + alpha_stats * regret
        self.regret_std = (1 - alpha_stats) * self.regret_std + alpha_stats * np.abs(regret - self.regret_mean)
    
    def create_state_for_agent(self, a: float, b: float, costs: np.ndarray, 
                               agent_id: int) -> np.ndarray:
        """
        Create state for a specific agent based on their information level.
        
        Args:
            a: Demand intercept
            b: Demand slope
            costs: Array of all agents' costs
            agent_id: Index of the agent
            
        Returns:
            State array (1D, 3D, or 4D depending on policy type)
        """
        input_dim = self.policies[agent_id].input_dim
        
        if input_dim == 4:
            # 4D state (Nash): [a, b, own_cost, opponent_cost]
            # Full information - knows everything
            own_cost = costs[agent_id]
            opponent_id = 1 - agent_id  # For 2-player game
            opponent_cost = costs[opponent_id]
            return np.array([a, b, own_cost, opponent_cost])
        
        elif input_dim == 3:
            # 3D state (Asymmetric): [a, b, own_cost]
            # Knows demand parameters but not opponent cost
            return np.array([a, b, costs[agent_id]])
        
        elif input_dim == 2:
            # 2D state (Partial): [b, own_cost]
            # Knows demand slope but not intercept
            return np.array([b, costs[agent_id]])
        
        elif input_dim == 1:
            # 1D state (Minimal): [own_cost]
            # Only knows own cost - no demand information!
            return np.array([costs[agent_id]])
        
        else:
            raise ValueError(f"Unsupported input dimension: {input_dim}. "
                           f"Supported: 1 (minimal), 3 (asymmetric), 4 (Nash)")
    
    def get_info_level_name(self, input_dim: int) -> str:
        """Get human-readable name for information level."""
        info_names = {
            1: "Minimal (cost only)",
            2: "Partial (slope + cost)",
            3: "Asymmetric (demand + cost)",
            4: "Nash (full info)"
        }
        return info_names.get(input_dim, f"{input_dim}D")
    
    def train_step(self, episode: int) -> Tuple[np.ndarray, np.ndarray]:
        """Execute one training step with regret minimization."""
        # Self-play: Update opponent every N episodes
        if self.self_play_mode and episode > 0 and episode % self.opponent_update_interval == 0:
            opponent_id = 1 - self.learning_agent_id
            self.copy_policy_parameters(self.learning_agent_id, opponent_id)
            self.history['opponent_updates'].append(episode)
        
        # Get learning rate and decay exploration
        lr = self.get_learning_rate(episode)
        self.decay_exploration()
        
        # Sample scenario
        a, b, costs = self.env.sample_scenario(self.n_players)
        
        # Compute policy means
        policy_means = np.zeros(self.n_players)
        for i in range(self.n_players):
            state = self.create_state_for_agent(a, b, costs, i)
            policy_means[i] = self.policies[i].forward(state)
            policy_means[i] = min(policy_means[i], self.env.config["q_max"])
        
        # Sample actions with exploration
        actions = np.random.normal(policy_means, self.sigma)
        actions = np.clip(actions, 0.0, self.env.config["q_max"])
        
        # Compute rewards
        rewards = self.env.profit(actions, costs, a, b)
        
        # Compute instantaneous regret
        regret = self.compute_regret(actions, a, b, costs, rewards)
        
        # Update regret statistics
        self.update_regret_stats(regret)
        
        # Learning signal: negative regret (we want to minimize regret)
        # Use normalization to create stable gradient magnitude
        advantage = -regret
        
        if self.normalize_regret:
            # Normalize to create consistent gradient strength
            advantage = (advantage - (-self.regret_mean)) / (self.regret_std + 1e-8)
        else:
            # Scale by a constant to ensure reasonable gradient magnitude
            # Without this, raw regret values may be too small for effective learning
            advantage = advantage * 0.1  # Tune this scaling factor as needed
        
        # Policy gradient update
        if self.self_play_mode:
            agents_to_update = [self.learning_agent_id]
        else:
            agents_to_update = range(self.n_players)
        
        for i in agents_to_update:
            state = self.create_state_for_agent(a, b, costs, i)
            score_grad = (actions[i] - policy_means[i]) / (self.sigma[i] ** 2)
            grad_mu = advantage[i] * score_grad
            grads = self.policies[i].backward(state, grad_mu)
            self.policies[i].adam_update(grads, lr)
        
        # Store history
        self.history['actions'].append(actions.copy())
        self.history['rewards'].append(rewards.copy())
        self.history['policy_means'].append(policy_means.copy())
        self.history['costs'].append(costs.copy())
        self.history['demand_a'].append(a)
        self.history['demand_b'].append(b)
        self.history['sigma'].append(self.sigma[0])
        self.history['learning_rate'].append(lr)
        self.history['regret'].append(regret.copy())
        self.history['regret_ema'].append(self.regret_ema.copy())
        self.history['cumulative_regret'].append(self.cumulative_regret.copy())
        
        return actions, rewards
    
    def train(self, verbose: bool = True) -> Dict:
        """Run full training loop."""
        episodes = self.config['episodes']
        print_interval = self.config['print_interval']
        
        if verbose:
            mode_str = "self-play" if self.self_play_mode else "simultaneous"
            norm_str = "normalized" if self.normalize_regret else "unnormalized"
            
            print(f"Training {self.n_players} agents for {episodes} episodes ({mode_str} mode)")
            print(f"  Learning signal: regret minimization ({norm_str})")
            
            # Print information level for each agent
            for i, policy in enumerate(self.policies):
                info_level = self.get_info_level_name(policy.input_dim)
                policy_type = type(policy).__name__
                print(f"  Agent {i}: {policy_type} - {info_level}")
            
            if self.self_play_mode:
                print(f"  Learning agent: {self.learning_agent_id}")
                print(f"  Opponent update interval: {self.opponent_update_interval}")
        
        for episode in range(episodes):
            if verbose and episode % print_interval == 0:
                avg_regret = self.regret_ema.mean()
                print(f"Episode {episode}/{episodes} | Avg Regret EMA: {avg_regret:.4f}")
            
            self.train_step(episode)
        
        if verbose:
            print("Training complete!")
            final_avg_regret = self.regret_ema.mean()
            final_cumulative = self.cumulative_regret.mean()
            final_average_regret = final_cumulative / episodes
            print(f"  Final average regret (EMA): {final_avg_regret:.4f}")
            print(f"  Final average regret (cumulative/T): {final_average_regret:.4f}")
            if self.self_play_mode:
                print(f"  Total opponent updates: {len(self.history['opponent_updates'])}")
        
        # Convert history lists to arrays
        for key in ['actions', 'rewards', 'policy_means', 'costs', 'regret', 'regret_ema', 'cumulative_regret']:
            self.history[key] = np.array(self.history[key])
        for key in ['demand_a', 'demand_b', 'sigma', 'learning_rate']:
            self.history[key] = np.array(self.history[key])
        
        return self.history
    
    def evaluate(self, a: float, b: float, costs: List[float], 
                 n_episodes: int = 5000) -> Dict:
        """Evaluate learned policies on a specific scenario."""
        costs = np.array(costs)
        actions = np.zeros((self.n_players, n_episodes))
        rewards = np.zeros((self.n_players, n_episodes))
        regrets = np.zeros((self.n_players, n_episodes))
        
        # Compute policy means
        policy_means = np.zeros(self.n_players)
        for i in range(self.n_players):
            state = self.create_state_for_agent(a, b, costs, i)
            policy_means[i] = min(self.policies[i].forward(state), 
                                 self.env.config["q_max"])
        
        # Run episodes with final exploration noise
        sigma_final = self.config['sigma_final']
        for t in range(n_episodes):
            actions[:, t] = np.random.normal(policy_means, sigma_final)
            actions[:, t] = np.clip(actions[:, t], 0.0, self.env.config["q_max"])
            rewards[:, t] = self.env.profit(actions[:, t], costs, a, b)
            regrets[:, t] = self.compute_regret(actions[:, t], a, b, costs, rewards[:, t])
        
        # Compute Nash equilibrium
        nash_q = self.env.nash_equilibrium(a, b, costs)
        nash_rewards = self.env.profit(nash_q, costs, a, b)
        
        return {
            'policy_means': policy_means,
            'nash_q': nash_q,
            'mean_actions': actions.mean(axis=1),
            'std_actions': actions.std(axis=1),
            'mean_rewards': rewards.mean(axis=1),
            'std_rewards': rewards.std(axis=1),
            'nash_rewards': nash_rewards,
            'error': np.abs(policy_means - nash_q),
            'profit_ratio': rewards.mean(axis=1) / nash_rewards,
            'mean_regret': regrets.mean(axis=1),
            'std_regret': regrets.std(axis=1)
        }