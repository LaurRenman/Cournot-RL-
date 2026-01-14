from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from basic_agent import BaseAgent  # your interface :contentReference[oaicite:2]{index=2}


def _make_edges(low: float, high: float, n_bins: int) -> np.ndarray:
    """
    Create (n_bins-1) internal bin edges for np.digitize.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    # internal edges only (exclude endpoints)
    return np.linspace(low, high, n_bins + 1)[1:-1]


def _discretize(x: np.ndarray, edges_list: list[np.ndarray]) -> tuple[int, ...]:
    """
    Map continuous vector x to a discrete tuple using per-dimension edges.
    """
    idxs = []
    for v, edges in zip(x, edges_list):
        idxs.append(int(np.digitize(v, edges, right=False)))
    return tuple(idxs)


@dataclass
class QLearningConfig:
    alpha: float = 0.1          # learning rate
    gamma: float = 0.95         # discount
    epsilon: float = 0.2        # exploration
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999


class IndependentQLearningAgent(BaseAgent):
    """
    Tabular Q-learning agent controlling ONE firm's quantity.
    - Use one instance per firm.
    - n_actions must be 1 (one scalar quantity output).
    """

    def __init__(
        self,
        q_max: float,
        *,
        n_action_bins: int = 21,
        n_state_bins_q: int = 21,
        n_state_bins_p: int = 21,
        price_max: float | None = None,
        config: QLearningConfig = QLearningConfig(),
        seed: int | None = None,
    ):
        super().__init__(n_actions=1, q_max=q_max)  # BaseAgent interface :contentReference[oaicite:3]{index=3}
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        self.n_action_bins = n_action_bins
        self.action_grid = np.linspace(0.0, q_max, n_action_bins)

        # State bins:
        # state = [q_1,...,q_N, p]  :contentReference[oaicite:4]{index=4}
        # We don't know N at init time; bins are built lazily on first call when we see state length.
        self.n_state_bins_q = n_state_bins_q
        self.n_state_bins_p = n_state_bins_p
        self.price_max = float(price_max) if price_max is not None else None

        self._edges_list: list[np.ndarray] | None = None  # per-dimension edges
        self.Q: dict[tuple[int, ...], np.ndarray] = {}    # state_key -> Q-values over action indices

    def _ensure_state_bins(self, state: np.ndarray):
        if self._edges_list is not None:
            return

        state = np.asarray(state, dtype=float)
        if state.ndim != 1 or state.size < 2:
            raise ValueError("State must be 1D and include at least one quantity and price.")

        n_dims = state.size
        n_q_dims = n_dims - 1
        if self.price_max is None:
            # A safe default if you didn't pass demand intercept 'a':
            # observed prices are non-negative; use max(q_max, 1.0) as a crude scale.
            self.price_max = max(self.q_max, 1.0)

        edges_list: list[np.ndarray] = []
        for _ in range(n_q_dims):
            edges_list.append(_make_edges(0.0, self.q_max, self.n_state_bins_q))
        edges_list.append(_make_edges(0.0, self.price_max, self.n_state_bins_p))  # price dim
        self._edges_list = edges_list

    def _state_key(self, state: np.ndarray) -> tuple[int, ...]:
        self._ensure_state_bins(state)
        assert self._edges_list is not None
        return _discretize(np.asarray(state, dtype=float), self._edges_list)

    def _get_Q_row(self, key: tuple[int, ...]) -> np.ndarray:
        row = self.Q.get(key)
        if row is None:
            row = np.zeros(self.n_action_bins, dtype=float)
            self.Q[key] = row
        return row

    def _action_to_index(self, action: np.ndarray) -> int:
        # action is shape (1,) per BaseAgent :contentReference[oaicite:5]{index=5}
        q = float(np.asarray(action, dtype=float).reshape(-1)[0])
        return int(np.argmin(np.abs(self.action_grid - q)))

    def select_action(self, state: np.ndarray) -> np.ndarray:
        key = self._state_key(state)
        q_row = self._get_Q_row(key)

        if self.rng.random() < self.cfg.epsilon:
            a_idx = int(self.rng.integers(0, self.n_action_bins))
        else:
            a_idx = int(np.argmax(q_row))

        return np.array([self.action_grid[a_idx]], dtype=float)

    def update(self, state, action, reward, next_state, done):
        s = self._state_key(state)
        ns = self._state_key(next_state)
        a_idx = self._action_to_index(action)

        q_row = self._get_Q_row(s)
        next_row = self._get_Q_row(ns)

        target = float(reward)
        if not done:
            target += self.cfg.gamma * float(np.max(next_row))

        q_row[a_idx] += self.cfg.alpha * (target - q_row[a_idx])

        # epsilon decay (simple schedule)
        if self.cfg.epsilon > self.cfg.epsilon_min:
            self.cfg.epsilon *= self.cfg.epsilon_decay

    def reset(self):
        # Optional: nothing required for tabular Q-learning
        pass
