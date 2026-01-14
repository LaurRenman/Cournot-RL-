"""
Runs baseline (non-learning) agents for Cournot competition.
"""

import numpy as np

from environment.cournot_env import CournotEnv
from agents.basic_agent import RandomAgent
from experiments.config import ENV_CONFIG


def run_baselines():
    env = CournotEnv(**ENV_CONFIG)

    agents = [
        RandomAgent(n_actions=1, q_max=ENV_CONFIG["q_max"], seed=i)
        for i in range(env.n_firms)
    ]

    state = env.reset()
    done = False
    total_rewards = np.zeros(env.n_firms)

    while not done:
        actions = np.array([
            agent.select_action(state)[0] for agent in agents
        ])

        state, rewards, done, info = env.step(actions)
        total_rewards += rewards

    print("Baseline total profits:")
    for i, r in enumerate(total_rewards):
        print(f"Firm {i}: {r:.2f}")

    print(f"Final market price: {info['price']:.2f}")
    print(f"Final total quantity: {info['total_quantity']:.2f}")

    return total_rewards


if __name__ == "__main__":
    run_baselines()
