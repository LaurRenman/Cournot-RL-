"""
Training loop for Cournot competition experiments (Independent Q-Learning).
Mirrors train.py structure (RandomAgent), but uses IndependentQLearningAgent.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to Python path (so imports work when running the file directly)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NOTE: your folder is spelled "environnement" in this file.
# If your actual folder is "environement" (typo), change the import accordingly.
from environnement.cournot_env import CournotEnv

from experiments.config import ENV_CONFIG, TRAINING_CONFIG

# Q-learning agent (should live in agents/qlearning_agent.py)
from agents.qlearning_agent import IndependentQLearningAgent, QLearningConfig


def train():
    env = CournotEnv(**ENV_CONFIG)

    # ----- Q-learning hyperparams -----
    # Try to read from config if you added them there; otherwise fall back to sensible defaults.
    # You can optionally add something like TRAINING_CONFIG["qlearning"] = {...}
    qcfg_dict = TRAINING_CONFIG.get("qlearning", {})

    alpha = qcfg_dict.get("alpha", 0.15)
    gamma = qcfg_dict.get("gamma", 0.95)
    epsilon = qcfg_dict.get("epsilon", 0.30)
    epsilon_min = qcfg_dict.get("epsilon_min", 0.05)
    epsilon_decay = qcfg_dict.get("epsilon_decay", 0.9995)

    n_action_bins = qcfg_dict.get("n_action_bins", 31)
    n_state_bins_q = qcfg_dict.get("n_state_bins_q", 21)
    n_state_bins_p = qcfg_dict.get("n_state_bins_p", 21)

    # price_max helps state discretization for price dimension
    price_max = ENV_CONFIG.get("a", None)

    # ----- Agents: one per firm (independent learning) -----
    # IMPORTANT: each agent must have its own config instance (epsilon decays per agent).
    agents = []
    for i in range(env.n_firms):
        cfg_i = QLearningConfig(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
        agents.append(
            IndependentQLearningAgent(
                q_max=ENV_CONFIG["q_max"],
                n_action_bins=n_action_bins,
                n_state_bins_q=n_state_bins_q,
                n_state_bins_p=n_state_bins_p,
                price_max=price_max,
                config=cfg_i,
                seed=i,
            )
        )

    num_episodes = TRAINING_CONFIG["num_episodes"]
    log_every = TRAINING_CONFIG["log_every"]

    episode_rewards = []
    episode_avg_price = []
    episode_avg_q = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        total_rewards = np.zeros(env.n_firms, dtype=float)
        sum_price = 0.0
        sum_q = np.zeros(env.n_firms, dtype=float)
        steps = 0

        for agent in agents:
            agent.reset()

        while not done:
            # each agent returns shape (1,), environment expects shape (n_firms,)
            actions = np.array([agent.select_action(state)[0] for agent in agents], dtype=float)

            next_state, rewards, done, info = env.step(actions)

            # update each independent agent with its own reward
            for i, agent in enumerate(agents):
                agent.update(
                    state=state,
                    action=np.array([actions[i]], dtype=float),  # shape (1,)
                    reward=float(rewards[i]),
                    next_state=next_state,
                    done=done,
                )
                total_rewards[i] += float(rewards[i])

            # tracking
            # Prefer info["price"] if present; else price is last component of state by your design.
            price_t = float(info.get("price", next_state[-1])) if isinstance(info, dict) else float(next_state[-1])
            sum_price += price_t
            sum_q += actions
            steps += 1

            state = next_state

        episode_rewards.append(total_rewards)
        episode_avg_price.append(sum_price / max(steps, 1))
        episode_avg_q.append(sum_q / max(steps, 1))

        if episode % log_every == 0:
            avg_profit = float(np.mean(total_rewards))
            print(
                f"Episode {episode:4d} | "
                f"Avg profit/firm: {avg_profit:10.2f} | "
                f"Avg price: {episode_avg_price[-1]:8.2f} | "
                f"Avg q: {np.round(episode_avg_q[-1], 2)}"
            )

    return (
        np.array(episode_rewards),        # shape (episodes, n_firms)
        np.array(episode_avg_price),      # shape (episodes,)
        np.array(episode_avg_q),          # shape (episodes, n_firms)
    )


if __name__ == "__main__":
    train()
