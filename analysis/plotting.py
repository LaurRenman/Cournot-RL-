import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(episode_rewards: np.ndarray):
    """
    Plot average profit per episode.
    """
    avg_rewards = np.mean(episode_rewards, axis=1)

    plt.figure()
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average profit per firm")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.show()


def plot_cumulative_profit(cumulative_profit: np.ndarray):
    """
    Plot cumulative profit over episodes.
    """
    plt.figure()
    plt.plot(cumulative_profit)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative average profit")
    plt.title("Cumulative Profit")
    plt.grid(True)
    plt.show()


def plot_baseline_comparison(learned_profit: float, baseline_profit: float):
    """
    Compare learned vs baseline profits.
    """
    plt.figure()
    plt.bar(
        ["Baseline", "Learned"],
        [baseline_profit, learned_profit]
    )
    plt.ylabel("Average profit per firm")
    plt.title("Baseline vs Learned Performance")
    plt.show()

def plot_quantities_vs_nash(quantities: np.ndarray, q_nash: np.ndarray, title: str = "Quantities vs Nash"):
    """
    Plot each firm's quantity over time against its Nash quantity.
    quantities shape: (T, n_firms)
    """
    T, n_firms = quantities.shape
    plt.figure()
    for i in range(n_firms):
        plt.plot(quantities[:, i], label=f"Firm {i} quantity")
        plt.hlines(q_nash[i], 0, T - 1, linestyles="dashed", label=f"Firm {i} Nash")
    plt.xlabel("Time step")
    plt.ylabel("Quantity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_price_vs_nash(prices: np.ndarray, p_nash: float, title: str = "Price vs Nash"):
    T = len(prices)
    plt.figure()
    plt.plot(prices, label="Price")
    plt.hlines(p_nash, 0, T - 1, linestyles="dashed", label="Nash price")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_distance_to_nash(dist: np.ndarray, title: str = "Distance to Nash over time"):
    plt.figure()
    plt.plot(dist)
    plt.xlabel("Time step")
    plt.ylabel("Distance")
    plt.title(title)
    plt.grid(True)
    plt.show()


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Simple rolling mean (valid mode).
    """
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_distance_to_nash_rolling(dist: np.ndarray, window: int = 20, title: str = "Distance to Nash (rolling mean)"):
    rm = rolling_mean(dist, window)
    plt.figure()
    plt.plot(rm)
    plt.xlabel("Time step")
    plt.ylabel("Distance (rolling mean)")
    plt.title(f"{title} | window={window}")
    plt.grid(True)
    plt.show()
