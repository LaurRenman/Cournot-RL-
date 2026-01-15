import numpy as np
import numpy as np
from model.equilibrium import cournot_nash_profile, cournot_nash_price


def cumulative_profit(episode_rewards: np.ndarray) -> np.ndarray:
    """
    Cumulative profit over episodes.

    episode_rewards shape: (num_episodes, n_firms)
    """
    return np.cumsum(np.mean(episode_rewards, axis=1))


def average_profit_per_firm(episode_rewards: np.ndarray) -> np.ndarray:
    """
    Average profit per firm over all episodes.
    """
    return np.mean(episode_rewards, axis=0)


def profit_variance(episode_rewards: np.ndarray) -> float:
    """
    Variance of profits across episodes (learning stability).
    """
    return np.var(np.mean(episode_rewards, axis=1))


def final_episode_profit(episode_rewards: np.ndarray) -> float:
    """
    Mean profit in the final episode.
    """
    return np.mean(episode_rewards[-1])

def extract_last_quantities_from_states(states: np.ndarray, n_firms: int) -> np.ndarray:
    """
    States are [q_1(t-1), ..., q_N(t-1), p(t-1)].
    Return quantities part: shape (T, n_firms).
    """
    return states[:, :n_firms]


def extract_last_price_from_states(states: np.ndarray) -> np.ndarray:
    """
    Return the price component from states: shape (T,).
    """
    return states[:, -1]


def nash_quantities(env_config: dict) -> np.ndarray:
    """
    Symmetric Cournot-Nash quantities vector for given ENV_CONFIG.
    Requires symmetric costs.
    """
    a = env_config["a"]
    b = env_config["b"]
    costs = env_config["costs"]
    n_firms = env_config["n_firms"]

    if not all(abs(costs[i] - costs[0]) < 1e-9 for i in range(len(costs))):
        raise ValueError("nash_quantities requires symmetric costs in this implementation.")

    c = float(costs[0])
    return cournot_nash_profile(a, b, c, n_firms)


def nash_price(env_config: dict) -> float:
    """
    Symmetric Cournot-Nash price for given ENV_CONFIG.
    """
    a = env_config["a"]
    b = env_config["b"]
    costs = env_config["costs"]
    n_firms = env_config["n_firms"]

    if not all(abs(costs[i] - costs[0]) < 1e-9 for i in range(len(costs))):
        raise ValueError("nash_price requires symmetric costs in this implementation.")

    c = float(costs[0])
    return cournot_nash_price(a, b, c, n_firms)


def distance_to_nash(quantities: np.ndarray, q_nash: np.ndarray, metric: str = "l2") -> np.ndarray:
    """
    Distance from observed quantities to Nash quantities per time step.

    quantities shape: (T, n_firms)
    q_nash shape: (n_firms,)
    Returns shape: (T,)
    """
    diff = quantities - q_nash[None, :]

    if metric == "l2":
        return np.sqrt(np.sum(diff**2, axis=1))
    if metric == "l1":
        return np.sum(np.abs(diff), axis=1)
    raise ValueError("metric must be 'l1' or 'l2'")


def mean_quantity_error(quantities: np.ndarray, q_nash: np.ndarray) -> float:
    """
    Mean absolute deviation from Nash across all firms and time steps.
    """
    return float(np.mean(np.abs(quantities - q_nash[None, :])))


def mean_price_error(prices: np.ndarray, p_nash: float) -> float:
    """
    Mean absolute deviation from Nash price.
    """
    return float(np.mean(np.abs(prices - p_nash)))
