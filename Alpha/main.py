"""
Main script for quick training and testing of Cournot MARL agents.

This script provides a simple command-line interface to train and evaluate
multi-agent reinforcement learning policies in Cournot competition.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import CournotEnvironment
from policy import PolicyNetwork
from trainer import CournotTrainer
from utils import (
    plot_training_convergence,
    plot_policy_response,
    plot_exploration_schedule,
    compare_agents,
    print_evaluation_summary
)


def main():
    """Main function to run training and evaluation."""
    
    # Set random seed
    np.random.seed(42)
    
    print("="*80)
    print("MULTI-AGENT REINFORCEMENT LEARNING: COURNOT COMPETITION")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    n_agents = 2
    
    env_config = {
        "a_min": 80.0,
        "a_max": 120.0,
        "b_min": 0.8,
        "b_max": 1.2,
        "cost_min": 5.0,
        "cost_max": 100.0,
        "q_max": 100.0,
        "seed": 42
    }
    
    train_config = {
        'episodes': 150000,
        'lr_init': 0.003,
        'lr_final': 0.0001,
        'warmup_steps': 10000,
        'sigma_init': 12.0,
        'sigma_final': 3.0,
        'sigma_decay': 0.99996,
        'print_interval': 15000
    }
    
    # -------------------------------------------------------------------------
    # Create environment and policies
    # -------------------------------------------------------------------------
    print(f"\nInitializing {n_agents} agents...")
    env = CournotEnvironment(env_config)
    policies = [PolicyNetwork(input_dim=3, hidden_dim=64) for _ in range(n_agents)]
    
    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    trainer = CournotTrainer(n_agents, env, policies, train_config)
    history = trainer.train(verbose=True)
    
    # -------------------------------------------------------------------------
    # Visualize training
    # -------------------------------------------------------------------------
    print("\nGenerating training visualizations...")
    
    # Plot 1: Training convergence
    fig1 = plot_training_convergence(history, window=1000)
    plt.savefig('training_convergence.png', dpi=150, bbox_inches='tight')
    print("  Saved: training_convergence.png")
    
    # Plot 2: Policy response
    fig2 = plot_policy_response(policies, env, a_fixed=100.0, b_fixed=1.0)
    plt.savefig('policy_response.png', dpi=150, bbox_inches='tight')
    print("  Saved: policy_response.png")
    
    # Plot 3: Exploration schedule
    fig3 = plot_exploration_schedule(history)
    plt.savefig('exploration_schedule.png', dpi=150, bbox_inches='tight')
    print("  Saved: exploration_schedule.png")
    
    # -------------------------------------------------------------------------
    # Evaluate on test scenarios
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EVALUATION ON TEST SCENARIOS")
    print("="*80)
    
    test_scenarios = [
        {"name": "Symmetric Low Cost", "a": 100.0, "b": 1.0, "costs": [10.0, 10.0]},
        {"name": "Asymmetric Cost", "a": 100.0, "b": 1.0, "costs": [10.0, 30.0]},
        {"name": "High Demand", "a": 110.0, "b": 1.0, "costs": [10.0, 15.0]},
        {"name": "Low Slope", "a": 100.0, "b": 0.9, "costs": [10.0, 15.0]},
        {"name": "High Slope", "a": 100.0, "b": 1.1, "costs": [10.0, 15.0]},
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  a={scenario['a']}, b={scenario['b']}, costs={scenario['costs']}")
        
        eval_results = trainer.evaluate(
            scenario['a'], 
            scenario['b'], 
            scenario['costs'], 
            n_episodes=5000
        )
        
        print(f"  Policy μ:       {eval_results['policy_means']}")
        print(f"  Nash q*:        {eval_results['nash_q']}")
        print(f"  Error:          {eval_results['error']}")
        print(f"  Profit Ratio:   {eval_results['profit_ratio']}")
    
    # -------------------------------------------------------------------------
    # Comparative analysis
    # -------------------------------------------------------------------------
    print("\nGenerating comparative analysis...")
    
    comparison_scenarios = [
        {"a": 100.0, "b": 1.0, "costs": [10.0, 15.0]},
        {"a": 90.0, "b": 1.0, "costs": [15.0, 20.0]},
        {"a": 110.0, "b": 1.0, "costs": [10.0, 20.0]},
        {"a": 100.0, "b": 0.9, "costs": [10.0, 15.0]},
        {"a": 100.0, "b": 1.1, "costs": [10.0, 15.0]},
    ]
    
    fig4, results = compare_agents(policies, env, comparison_scenarios)
    plt.savefig('comparative_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: comparative_analysis.png")
    
    # -------------------------------------------------------------------------
    # Save policies
    # -------------------------------------------------------------------------
    print("\nSaving trained policies...")
    for i, policy in enumerate(policies):
        policy.save(f'policy_agent_{i}.npz')
        print(f"  Saved: policy_agent_{i}.npz")
    
    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    window = 10000
    final_rewards = history['rewards'][-window:]
    
    print(f"\nFinal Performance (last {window} episodes):")
    for i in range(n_agents):
        print(f"  Agent {i}:")
        print(f"    Mean Profit: {final_rewards[:, i].mean():.2f}")
        print(f"    Std Profit:  {final_rewards[:, i].std():.2f}")
    
    correlation = np.corrcoef(final_rewards[:, 0], final_rewards[:, 1])[0, 1]
    print(f"\nProfit Correlation: {correlation:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
