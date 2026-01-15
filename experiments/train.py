"""
Training loop for Cournot competition experiments.
"""

import numpy as np

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environnement.cournot_env import CournotEnv
from experiments.config import ENV_CONFIG, TRAINING_CONFIG


def train(agent_class, agent_kwargs):
    env = CournotEnv(**ENV_CONFIG)

    # One agent per firm (decentralized learning)
    agents = [
        agent_class(**agent_kwargs, seed=i)
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
            print(
                f"Episode {episode:4d} | "
                f"Avg profit per firm: {total_rewards.mean():8.2f}"
            )
       
    return np.array(episode_rewards), agents

