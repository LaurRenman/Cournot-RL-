"""
Training loop for Cournot competition experiments.
"""

import numpy as np

from environment.cournot_env import CournotEnv
from agents.basic_agent import RandomAgent
from experiments.config import ENV_CONFIG, TRAINING_CONFIG


def train():
    env = CournotEnv(**ENV_CONFIG)

    # One agent per firm (decentralized learning)
    agents = [
        RandomAgent(n_actions=1, q_max=ENV_CONFIG["q_max"], seed=i)
        for i in range(env.n_firms)
    ]

    num_episodes = TRAINING_CONFIG["num_episodes"]
    log_every = TRAINING_CONFIG["log_every"]

    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(env.n_firms)

        for agent in agents:
            agent.reset()

        while not done:
            actions = np.array([
                agent.select_action(state)[0] for agent in agents
            ])

            next_state, rewards, done, info = env.step(actions)

            for i, agent in enumerate(agents):
                agent.update(
                    state=state,
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_state,
                    done=done
                )
                total_rewards[i] += rewards[i]

            state = next_state

        episode_rewards.append(total_rewards)

        if episode % log_every == 0:
            avg_reward = np.mean(total_rewards)
            print(
                f"Episode {episode:4d} | "
                f"Avg profit per firm: {avg_reward:8.2f}"
            )

    return np.array(episode_rewards)


if __name__ == "__main__":
    train()
