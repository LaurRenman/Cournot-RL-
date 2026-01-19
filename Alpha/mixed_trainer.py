"""
Mixed Trainer for Symmetric Multi-Agent Learning with Regret Minimization

This trainer supports agents with heterogeneous information structures
(e.g. 3D asymmetric and 4D Nash policies) via MixedPolicyAdapter.

Key properties:
- True simultaneous learning (no self-play)
- No parameter copying
- State reconstruction handled by adapters
- Regret-based learning signal

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
from typing import List, Dict, Optional, Tuple
from environment import CournotEnvironment
from mixed_policy import MixedPolicyAdapter


class MixedCournotTrainer:
    """
    Symmetric multi-agent trainer for Cournot competition
    with heterogeneous policies using regret minimization.
    """

    def __init__(
        self,
        n_players: int,
        env: CournotEnvironment,
        policies: List[MixedPolicyAdapter],
        config: Optional[Dict] = None
    ):
        self.n_players = n_players
        self.env = env
        self.policies = policies
        self.config = config or self.get_default_config()

        # Regret tracking (EMA for smoothing)
        self.regret_ema = np.zeros(n_players)
        self.regret_alpha = self.config.get('regret_alpha', 0.01)
        
        # Regret normalization for stable learning (RECOMMENDED)
        self.normalize_regret = self.config.get('normalize_regret', True)
        self.regret_mean = np.zeros(n_players)
        self.regret_std = np.ones(n_players)
        self.stats_alpha = self.config.get('stats_alpha', 0.01)

        # Exploration
        self.sigma = np.full(n_players, self.config["sigma_init"])

        # History
        self.history = {
            "actions": [],
            "rewards": [],
            "policy_means": [],
            "costs": [],
            "demand_a": [],
            "demand_b": [],
            "sigma": [],
            "learning_rate": [],
            'regret': [],  # Instantaneous regret
            'regret_ema': [],  # Exponential moving average
            'cumulative_regret': []  # Running sum
        }
        
        # Cumulative regret tracking
        self.cumulative_regret = np.zeros(n_players)

    @staticmethod
    def get_default_config() -> Dict:
        return {
            "episodes": 150000,
            "lr_init": 0.003,
            "lr_final": 0.0001,
            "warmup_steps": 10000,
            "sigma_init": 12.0,
            "sigma_final": 3.0,
            "sigma_decay": 0.99996,
            'regret_alpha': 0.01,  # EMA smoothing for regret
            'normalize_regret': True,  # STRONGLY RECOMMENDED for stable learning
            'stats_alpha': 0.01,
            "print_interval": 15000,
        }

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def get_learning_rate(self, episode: int) -> float:
        warmup = self.config["warmup_steps"]
        total = self.config["episodes"]
        lr_init = self.config["lr_init"]
        lr_final = self.config["lr_final"]

        if episode < warmup:
            return lr_init * (episode / warmup)
        progress = (episode - warmup) / (total - warmup)
        return lr_final + (lr_init - lr_final) * (1 - progress) ** 0.9

    def decay_exploration(self):
        self.sigma *= self.config["sigma_decay"]
        self.sigma = np.maximum(self.sigma, self.config["sigma_final"])

    # ------------------------------------------------------------------
    # Regret computation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, episode: int) -> Tuple[np.ndarray, np.ndarray]:
        lr = self.get_learning_rate(episode)
        self.decay_exploration()

        # Sample environment
        a, b, costs = self.env.sample_scenario(self.n_players)

        # Policy means
        policy_means = np.zeros(self.n_players)
        for i in range(self.n_players):
            # Trainer always creates the FULL state
            full_state = np.array([a, b, costs[i], costs[1 - i]])
            policy_means[i] = self.policies[i].forward(full_state)
            policy_means[i] = min(policy_means[i], self.env.config["q_max"])

        # Sample actions
        actions = np.random.normal(policy_means, self.sigma)
        actions = np.clip(actions, 0.0, self.env.config["q_max"])

        # Rewards
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

        # Gradient updates (ALL agents)
        for i in range(self.n_players):
            full_state = np.array([a, b, costs[i], costs[1 - i]])
            score_grad = (actions[i] - policy_means[i]) / (self.sigma[i] ** 2)
            grad_mu = advantage[i] * score_grad
            grads = self.policies[i].backward(full_state, grad_mu)
            self.policies[i].adam_update(grads, lr)

        # Logging
        self.history["actions"].append(actions.copy())
        self.history["rewards"].append(rewards.copy())
        self.history["policy_means"].append(policy_means.copy())
        self.history["costs"].append(costs.copy())
        self.history["demand_a"].append(a)
        self.history["demand_b"].append(b)
        self.history["sigma"].append(self.sigma[0])
        self.history["learning_rate"].append(lr)
        self.history['regret'].append(regret.copy())
        self.history['regret_ema'].append(self.regret_ema.copy())
        self.history['cumulative_regret'].append(self.cumulative_regret.copy())

        return actions, rewards

    def train(self, verbose: bool = True) -> Dict:
        episodes = self.config["episodes"]
        print_interval = self.config["print_interval"]

        if verbose:
            norm_str = "normalized" if self.normalize_regret else "unnormalized"
            print(
                f"Training {self.n_players} agents (symmetric mixed-information learning)"
            )
            print(f"  Learning signal: regret minimization ({norm_str})")

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

        # Convert history to arrays
        for key in ["actions", "rewards", "policy_means", "costs", 'regret', 'regret_ema', 'cumulative_regret']:
            self.history[key] = np.array(self.history[key])
        for key in ["demand_a", "demand_b", "sigma", "learning_rate"]:
            self.history[key] = np.array(self.history[key])

        return self.history
    
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    
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
            full_state = np.array([a, b, costs[i], costs[1 - i]])
            policy_means[i] = min(self.policies[i].forward(full_state), 
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