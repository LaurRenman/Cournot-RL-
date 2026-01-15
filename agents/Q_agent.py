import numpy as np

print("Q_AGENT LOADED FROM:", __file__)


class Qlearning_agent:
    def __init__(
        self,
        q_max,
        n_quantities=21,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.q_max = float(q_max)
        self.n_quantities = int(n_quantities)

        self.quantities = np.linspace(0.0, self.q_max, self.n_quantities)

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self.Q = {}

    def reset(self):
        pass

    def _state_key(self, state):
        state = np.asarray(state)
        return tuple(np.round(state, 2))

    def _get_Q(self, state):
        key = self._state_key(state)
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_quantities)
        return self.Q[key]

    def select_action(self, state):
        Q_s = self._get_Q(state)

        if self.rng.random() < self.epsilon:
            idx = self.rng.integers(self.n_quantities)
        else:
            idx = np.argmax(Q_s)

        return np.array([self.quantities[idx]])
    
    def act_greedy(self, state):
        """
        Deterministic action (no exploration).
        """
        Q_s = self._get_Q(state)
        idx = np.argmax(Q_s)
        return np.array([self.quantities[idx]])


    def update(self, state, action, reward, next_state, done):
        Q_s = self._get_Q(state)
        Q_next = self._get_Q(next_state)

        action = float(action)
        idx = np.argmin(np.abs(self.quantities - action))

        target = reward if done else reward + self.gamma * np.max(Q_next)
        Q_s[idx] += self.alpha * (target - Q_s[idx])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
