"""
Evaluation script for trained Cournot agents.
"""

import numpy as np

from environnement.cournot_env import CournotEnv
from experiments.config import ENV_CONFIG, EVAL_CONFIG


def evaluate(agents):
    env = CournotEnv(**ENV_CONFIG)

    for agent in agents:
        agent.epsilon = 0.0

    num_episodes = EVAL_CONFIG["num_episodes"]
    total_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = np.zeros(env.n_firms)

        while not done:
            actions = np.array([
                agent.select_action(state)[0] for agent in agents
            ])

            state, rewards, done, _ = env.step(actions)
            episode_rewards += rewards

        total_rewards.append(episode_rewards)

    avg_rewards = np.mean(total_rewards, axis=0)

    print("Average evaluation profits per firm:")
    for i, r in enumerate(avg_rewards):
        print(f"Firm {i}: {r:.2f}")

    return avg_rewards
