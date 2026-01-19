"""
Utility Functions Module

This module provides utility functions for visualization, analysis,
and experiment management. Works with both PolicyNetwork types.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
from environment import CournotEnvironment
from policy import PolicyNetwork_asymmetrical, PolicyNetwork_Nash


def smooth_curve(data: np.ndarray, window: int = 1000) -> np.ndarray:
    """
    Smooth a curve using moving average.
    
    Args:
        data: Input data array
        window: Window size for smoothing
        
    Returns:
        Smoothed data array
    """
    if len(data) < window:
        window = len(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_convergence(history: Dict, window: int = 1000, 
                              figsize: Tuple[int, int] = (12, 5)):
    """
    Plot training convergence for all agents.
    
    Args:
        history: Training history dictionary
        window: Smoothing window size
        figsize: Figure size
    """
    rewards = history['rewards']
    n_players = rewards.shape[1]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for i in range(n_players):
        smoothed = smooth_curve(rewards[:, i], window)
        ax.plot(smoothed, label=f'Agent {i}', linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Training Episodes", fontsize=12)
    ax.set_ylabel("Profit (smoothed)", fontsize=12)
    ax.set_title("Training Convergence", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_policy_response(policies: List[Union[PolicyNetwork_asymmetrical, PolicyNetwork_Nash]], 
                         env: CournotEnvironment,
                         a_fixed: float = 100.0, b_fixed: float = 1.0,
                         figsize: Tuple[int, int] = (14, 5)):
    """
    Plot learned policy response to varying costs.
    Works with 1D, 2D, 3D, and 4D policy networks.
    
    Args:
        policies: List of policy networks (any type)
        env: Environment instance
        a_fixed: Fixed demand intercept
        b_fixed: Fixed demand slope
        figsize: Figure size
    """
    cost_range = np.linspace(env.config["cost_min"], env.config["cost_max"], 100)
    n_players = len(policies)
    
    learned_q = np.zeros((n_players, len(cost_range)))
    
    for i in range(n_players):
        input_dim = policies[i].input_dim
        
        for idx, c in enumerate(cost_range):
            # Create state based on policy input dimension
            if input_dim == 1:
                # 1D: [own_cost]
                state = np.array([c])
            elif input_dim == 2:
                # 2D: [b, own_cost]
                state = np.array([b_fixed, c])
            elif input_dim == 3:
                # 3D: [a, b, own_cost]
                state = np.array([a_fixed, b_fixed, c])
            elif input_dim == 4:
                # 4D: [a, b, own_cost, opponent_cost]
                # Use symmetric cost assumption
                state = np.array([a_fixed, b_fixed, c, c])
            else:
                raise ValueError(f"Unsupported input_dim: {input_dim}")
            
            learned_q[i, idx] = min(
                policies[i].forward(state),
                env.config["q_max"]
            )
    
    # Nash equilibrium
    nash_q = (a_fixed - cost_range) / (b_fixed * (n_players + 1))
    nash_q = np.clip(nash_q, 0, env.config["q_max"])
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Policy responses
    for i in range(n_players):
        info_level = {1: "1D", 2: "2D", 3: "3D", 4: "4D"}.get(policies[i].input_dim, "?D")
        axes[0].plot(cost_range, learned_q[i], linewidth=3, 
                    label=f'Agent {i} ({info_level})', alpha=0.8)
    
    axes[0].plot(cost_range, nash_q, '--', linewidth=3, 
                label='Nash Equilibrium', color='red', alpha=0.8)
    
    axes[0].set_xlabel("Marginal Cost", fontsize=12)
    axes[0].set_ylabel("Quantity Produced", fontsize=12)
    axes[0].set_title(f"Policy Response to Cost (a={a_fixed}, b={b_fixed})", 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Error from Nash
    for i in range(n_players):
        info_level = {1: "1D", 2: "2D", 3: "3D", 4: "4D"}.get(policies[i].input_dim, "?D")
        error = np.abs(learned_q[i] - nash_q)
        axes[1].plot(cost_range, error, linewidth=3,
                    label=f'Agent {i} ({info_level})', alpha=0.8)
    
    axes[1].set_xlabel("Marginal Cost", fontsize=12)
    axes[1].set_ylabel("Absolute Error from Nash", fontsize=12)
    axes[1].set_title("Deviation from Nash Equilibrium", 
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_exploration_schedule(history: Dict, figsize: Tuple[int, int] = (12, 4)):
    """
    Plot exploration noise and learning rate schedules.
    
    Args:
        history: Training history dictionary
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Exploration noise
    axes[0].plot(history['sigma'], linewidth=2, color='orange')
    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Exploration Noise (σ)", fontsize=11)
    axes[0].set_title("Exploration Schedule", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(history['learning_rate'], linewidth=2, color='green')
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].set_ylabel("Learning Rate", fontsize=11)
    axes[1].set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_action_distribution(history: Dict, window: int = 10000, 
                             figsize: Tuple[int, int] = (12, 5)):
    """
    Plot distribution of actions over time.
    
    Args:
        history: Training history dictionary
        window: Window for analysis
        figsize: Figure size
    """
    actions = history['actions']
    n_players = actions.shape[1]
    n_episodes = actions.shape[0]
    
    # Split into early, middle, and late training
    periods = {
        'Early': actions[:window],
        'Middle': actions[n_episodes//2 - window//2:n_episodes//2 + window//2],
        'Late': actions[-window:]
    }
    
    fig, axes = plt.subplots(1, n_players, figsize=figsize)
    if n_players == 1:
        axes = [axes]
    
    for i in range(n_players):
        ax = axes[i]
        for label, data in periods.items():
            ax.hist(data[:, i], bins=30, alpha=0.5, label=label, density=True)
        
        ax.set_xlabel("Quantity", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Agent {i} Actions", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_profit_analysis(history: Dict, window: int = 1000,
                        figsize: Tuple[int, int] = (14, 10)):
    """
    Comprehensive profit analysis dashboard.
    
    Args:
        history: Training history dictionary
        window: Smoothing window
        figsize: Figure size
    """
    rewards = history['rewards']
    n_players = rewards.shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Smoothed profits over time
    ax = axes[0, 0]
    for i in range(n_players):
        smoothed = smooth_curve(rewards[:, i], window)
        ax.plot(smoothed, label=f'Agent {i}', linewidth=2, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Profit", fontsize=11)
    ax.set_title("Profit Over Time", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Profit distribution
    ax = axes[0, 1]
    for i in range(n_players):
        ax.hist(rewards[-10000:, i], bins=40, alpha=0.6, label=f'Agent {i}', density=True)
    ax.set_xlabel("Profit", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Profit Distribution (Last 10k Episodes)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative profits
    ax = axes[1, 0]
    for i in range(n_players):
        cumulative = np.cumsum(rewards[:, i])
        ax.plot(cumulative, label=f'Agent {i}', linewidth=2, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Profit", fontsize=11)
    ax.set_title("Cumulative Profits", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Rolling mean and std
    ax = axes[1, 1]
    window_rolling = 5000
    for i in range(n_players):
        rolling_mean = smooth_curve(rewards[:, i], window_rolling)
        episodes = np.arange(len(rolling_mean))
        ax.plot(episodes, rolling_mean, label=f'Agent {i}', linewidth=2)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Profit", fontsize=11)
    ax.set_title(f"Rolling Mean (window={window_rolling})", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_agents(policies: List[Union[PolicyNetwork_asymmetrical, PolicyNetwork_Nash]], 
                  env: CournotEnvironment,
                  scenarios: List[Dict], figsize: Tuple[int, int] = (14, 8)):
    """
    Compare agent performance across multiple scenarios.
    Works with both PolicyNetwork_asymmetrical and PolicyNetwork_Nash.
    
    Args:
        policies: List of policy networks (any type)
        env: Environment instance
        scenarios: List of scenario dictionaries with 'a', 'b', 'costs' keys
        figsize: Figure size
    """
    n_scenarios = len(scenarios)
    n_players = len(policies)
    
    results = []
    for scenario in scenarios:
        a = scenario['a']
        b = scenario['b']
        costs = np.array(scenario['costs'])
        
        # Get policy outputs
        policy_q = np.zeros(n_players)
        for i in range(n_players):
            state = env.create_state(a, b, costs[i])
            policy_q[i] = min(policies[i].forward(state), env.config["q_max"])
        
        # Nash equilibrium
        nash_q = env.nash_equilibrium(a, b, costs)
        
        # Profits
        policy_profit = env.profit(policy_q, costs, a, b)
        nash_profit = env.profit(nash_q, costs, a, b)
        
        results.append({
            'policy_q': policy_q,
            'nash_q': nash_q,
            'policy_profit': policy_profit,
            'nash_profit': nash_profit,
            'error': np.abs(policy_q - nash_q)
        })
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Quantity comparison
    ax = axes[0, 0]
    x = np.arange(n_scenarios)
    width = 0.25
    for i in range(n_players):
        policy_qs = [r['policy_q'][i] for r in results]
        nash_qs = [r['nash_q'][i] for r in results]
        ax.bar(x + i*width, policy_qs, width, label=f'Agent {i} (Learned)', alpha=0.8)
        ax.plot(x + i*width, nash_qs, 'rx', markersize=10, label=f'Agent {i} (Nash)' if i == 0 else '')
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Quantity", fontsize=11)
    ax.set_title("Quantity Comparison", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"S{i+1}" for i in range(n_scenarios)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Error
    ax = axes[0, 1]
    for i in range(n_players):
        errors = [r['error'][i] for r in results]
        ax.bar(x + i*width, errors, width, label=f'Agent {i}', alpha=0.8)
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Absolute Error", fontsize=11)
    ax.set_title("Quantity Error vs Nash", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"S{i+1}" for i in range(n_scenarios)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Profit comparison
    ax = axes[1, 0]
    for i in range(n_players):
        policy_profits = [r['policy_profit'][i] for r in results]
        nash_profits = [r['nash_profit'][i] for r in results]
        ax.bar(x + i*width, policy_profits, width, label=f'Agent {i} (Learned)', alpha=0.8)
        ax.plot(x + i*width, nash_profits, 'rx', markersize=10, label=f'Agent {i} (Nash)' if i == 0 else '')
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Profit", fontsize=11)
    ax.set_title("Profit Comparison", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"S{i+1}" for i in range(n_scenarios)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Profit ratio
    ax = axes[1, 1]
    for i in range(n_players):
        ratios = [r['policy_profit'][i] / r['nash_profit'][i] for r in results]
        ax.bar(x + i*width, ratios, width, label=f'Agent {i}', alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Nash Level')
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Profit Ratio (Learned/Nash)", fontsize=11)
    ax.set_title("Profit Efficiency", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"S{i+1}" for i in range(n_scenarios)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, results


def print_evaluation_summary(eval_results: Dict):
    """
    Print formatted evaluation summary.
    
    Args:
        eval_results: Dictionary from trainer.evaluate()
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Policy Means:     {eval_results['policy_means']}")
    print(f"Nash Equilibrium: {eval_results['nash_q']}")
    print(f"Absolute Error:   {eval_results['error']}")
    print(f"Mean Profit:      {eval_results['mean_rewards']}")
    print(f"Nash Profit:      {eval_results['nash_rewards']}")
    print(f"Profit Ratio:     {eval_results['profit_ratio']}")
    print("="*60)