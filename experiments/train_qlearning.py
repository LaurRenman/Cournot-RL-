"""
Training loop for Cournot competition experiments (Q-learning).
Also records and plots training metrics.
"""

import numpy as np
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environnement.cournot_env import CournotEnv
from experiments.config import ENV_CONFIG, TRAINING_CONFIG
from agents.qlearning_agent import IndependentQLearningAgent as QLearningAgent, QLearningConfig


def _ensure_results_dir() -> Path:
    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _plot_training_curves(
    avg_profit_per_firm: np.ndarray,
    avg_price: np.ndarray,
    avg_q_per_firm: np.ndarray,
    title_prefix: str,
    out_path: Path,
    smooth_window: int = 25,
):
    episodes = np.arange(len(avg_profit_per_firm))

    fig = plt.figure(figsize=(10, 10))

    # Profit
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(episodes, avg_profit_per_firm, label="Avg profit/firm")
    if len(avg_profit_per_firm) >= smooth_window:
        y = _moving_average(avg_profit_per_firm, smooth_window)
        ax1.plot(np.arange(len(y)), y, label=f"Smoothed (window={smooth_window})")
    ax1.set_title(f"{title_prefix} - Avg profit per firm")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Profit")
    ax1.legend()

    # Price
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(episodes, avg_price, label="Avg price")
    if len(avg_price) >= smooth_window:
        y = _moving_average(avg_price, smooth_window)
        ax2.plot(np.arange(len(y)), y, label=f"Smoothed (window={smooth_window})")
    ax2.set_title(f"{title_prefix} - Avg price")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Price")
    ax2.legend()

    # Quantities
    ax3 = plt.subplot(3, 1, 3)
    for i in range(avg_q_per_firm.shape[1]):
        ax3.plot(episodes, avg_q_per_firm[:, i], label=f"Firm {i} avg q")
    ax3.set_title(f"{title_prefix} - Avg quantities")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Quantity")
    ax3.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(env: CournotEnv, agents, n_eval_episodes: int = 20):
    """
    Evaluate learned policies with epsilon=0 (greedy).
    Returns dict with avg_q, avg_price, avg_profit.
    """
    for ag in agents:
        ag.cfg.epsilon = 0.0

    total_q = np.zeros(env.n_firms, dtype=float)
    total_profit = np.zeros(env.n_firms, dtype=float)
    total_price = 0.0
    steps = 0

    for _ in range(n_eval_episodes):
        state = env.reset()
        done = False

        while not done:
            actions = np.array([agent.select_action(state)[0] for agent in agents], dtype=float)
            next_state, rewards, done, info = env.step(actions)

            price_t = float(info.get("price", next_state[-1])) if isinstance(info, dict) else float(next_state[-1])

            total_q += actions
            total_profit += rewards
            total_price += price_t
            steps += 1

            state = next_state

    return {
        "avg_q": total_q / max(steps, 1),
        "avg_price": total_price / max(steps, 1),
        "avg_profit": total_profit / max(steps, 1),
        "steps": steps,
    }


def train():
    env = CournotEnv(**ENV_CONFIG)

    num_episodes = TRAINING_CONFIG["num_episodes"]
    log_every = TRAINING_CONFIG["log_every"]
    smooth_window = TRAINING_CONFIG.get("smooth_window", 25)

    # Optional: Q-learning specific config under TRAINING_CONFIG["qlearning"]
    qcfg = TRAINING_CONFIG.get("qlearning", {})

    # One agent per firm (decentralized learning)
    agents = []
    for i in range(env.n_firms):
        cfg_i = QLearningConfig(
            alpha=qcfg.get("alpha", 0.15),
            gamma=qcfg.get("gamma", 0.95),
            epsilon=qcfg.get("epsilon", 0.30),
            epsilon_min=qcfg.get("epsilon_min", 0.05),
            epsilon_decay=qcfg.get("epsilon_decay", 0.9995),
        )

        agents.append(
            QLearningAgent(
                q_max=ENV_CONFIG["q_max"],
                seed=i,
                config=cfg_i,
                n_action_bins=qcfg.get("n_action_bins", 31),
                n_state_bins_q=qcfg.get("n_state_bins_q", 21),
                n_state_bins_p=qcfg.get("n_state_bins_p", 21),
                price_max=ENV_CONFIG.get("a", None),
            )
        )

    # tracking
    episode_rewards = []
    avg_profit_per_firm = np.zeros(num_episodes, dtype=float)
    avg_price = np.zeros(num_episodes, dtype=float)
    avg_q_per_firm = np.zeros((num_episodes, env.n_firms), dtype=float)
    epsilon_trace = np.zeros(num_episodes, dtype=float)

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
            actions = np.array([agent.select_action(state)[0] for agent in agents], dtype=float)
            next_state, rewards, done, info = env.step(actions)

            for i, agent in enumerate(agents):
                agent.update(
                    state=state,
                    action=np.array([actions[i]], dtype=float),  # BaseAgent expects np.ndarray
                    reward=float(rewards[i]),
                    next_state=next_state,
                    done=done
                )
                total_rewards[i] += float(rewards[i])

            price_t = float(info.get("price", next_state[-1])) if isinstance(info, dict) else float(next_state[-1])
            sum_price += price_t
            sum_q += actions
            steps += 1

            state = next_state

        episode_rewards.append(total_rewards)

        avg_profit_per_firm[episode] = float(np.mean(total_rewards))
        avg_price[episode] = sum_price / max(steps, 1)
        avg_q_per_firm[episode, :] = sum_q / max(steps, 1)
        epsilon_trace[episode] = agents[0].cfg.epsilon

        if episode % log_every == 0:
            print(
                f"Episode {episode:4d} | "
                f"Avg profit/firm: {avg_profit_per_firm[episode]:10.2f} | "
                f"Avg price: {avg_price[episode]:8.2f} | "
                f"Avg q: {np.round(avg_q_per_firm[episode], 2)} | "
                f"Epsilon: {epsilon_trace[episode]:.3f}"
            )

    # evaluation
    eval_stats = evaluate(env, agents, n_eval_episodes=20)
    print("\n=== Evaluation (epsilon=0) ===")
    print("Avg quantities:", np.round(eval_stats["avg_q"], 3))
    print("Avg price:", round(float(eval_stats["avg_price"]), 3))
    print("Avg profits:", np.round(eval_stats["avg_profit"], 3))

    # save + plot
    out_dir = _ensure_results_dir()

    np.savez(
        out_dir / "qlearning_training.npz",
        episode_rewards=np.array(episode_rewards),
        avg_profit_per_firm=avg_profit_per_firm,
        avg_price=avg_price,
        avg_q_per_firm=avg_q_per_firm,
        epsilon_trace=epsilon_trace,
        eval_avg_q=eval_stats["avg_q"],
        eval_avg_price=eval_stats["avg_price"],
        eval_avg_profit=eval_stats["avg_profit"],
        env_config=ENV_CONFIG,
        training_config=TRAINING_CONFIG,
    )

    _plot_training_curves(
        avg_profit_per_firm=avg_profit_per_firm,
        avg_price=avg_price,
        avg_q_per_firm=avg_q_per_firm,
        title_prefix="Q-learning",
        out_path=out_dir / "qlearning_training.png",
        smooth_window=smooth_window,
    )

    # epsilon plot
    fig = plt.figure(figsize=(10, 4))
    plt.plot(np.arange(num_episodes), epsilon_trace, label="epsilon")
    plt.title("Q-learning - Epsilon over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "qlearning_epsilon.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved results to: {out_dir}")
    return np.array(episode_rewards)


if __name__ == "__main__":
    train()
