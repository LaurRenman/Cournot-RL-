import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Environment settings
# --------------------
ENV_CONFIG = {
    "a_min": 80.0,
    "a_max": 120.0,
    "b_min": 0.8,
    "b_max": 1.2,
    "cost_min": 5.0,
    "cost_max": 100.0,
    "q_max": 100.0,
    "horizon": 150_000,
    "seed": 42,
}

N_player = 2
np.random.seed(ENV_CONFIG["seed"])

# --------------------
# Price function
# --------------------
def price(acts, a, b):
    Q = np.sum(acts)
    return max(a - b * Q, 0.0)

# --------------------
# Nash equilibrium (asymmetric costs)
# --------------------
def cournot_nash_asymmetric(a, b, costs):
    costs = np.array(costs, dtype=float)
    N = len(costs)
    total_cost = np.sum(costs)
    q_star = (a - (N + 1) * costs + total_cost) / (b * (N + 1))
    return np.maximum(q_star, 0.0)

# --------------------
# Replay Buffer
# --------------------
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((self.capacity, self.state_dim), dtype=float)
        self.a = np.zeros((self.capacity,), dtype=int)
        self.r = np.zeros((self.capacity,), dtype=float)
        self.sp = np.zeros((self.capacity, self.state_dim), dtype=float)
        self.d = np.zeros((self.capacity,), dtype=float)

    def add(self, s, a, r, sp, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.sp[self.ptr] = sp
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.d[idx]

# --------------------
# Q-Network (Residual)
# Q(s,a_idx) = baseline_profit(s, q[a_idx]) + NN_correction(s)[a_idx]
# --------------------
class QNetwork:
    def __init__(self, state_dim=3, hidden_dim=64, n_actions=101, q_max=100.0):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.q_max = q_max

        # Discrete action grid
        self.action_grid = np.linspace(0.0, q_max, n_actions)

        # NN correction params
        limit1 = np.sqrt(6.0 / (state_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        limit2 = np.sqrt(6.0 / (hidden_dim + n_actions))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, n_actions))
        self.b2 = np.zeros(n_actions)

        # Normalization (match your script)
        self.input_mean = np.array([100.0, 1.0, 52.5], dtype=float)
        self.input_std = np.array([15.0, 0.15, 35.0], dtype=float)

        # Adam
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def elu_derivative(self, x, alpha=1.0):
        return np.where(x > 0, 1.0, alpha * np.exp(x))

    def baseline_profit(self, states):
        """
        Baseline "solo firm" profit curve per action:
        pi(q) ≈ (a - c) q - b q^2
        states: (B,3) = [a,b,c]
        returns: (B, n_actions)
        """
        a = states[:, 0:1]
        b = states[:, 1:2]
        c = states[:, 2:3]
        q = self.action_grid[None, :]  # (1, n_actions)
        return (a - c) * q - b * (q ** 2)

    def forward(self, states):
        """
        states: (B,3)
        returns Q-values: (B, n_actions)
        """
        x = (states - self.input_mean) / (self.input_std + 1e-8)  # (B,3)
        z1 = x @ self.W1 + self.b1
        h1 = self.elu(z1)
        corr = h1 @ self.W2 + self.b2  # (B, n_actions)

        base = self.baseline_profit(states)
        qvals = base + corr

        cache = (x, z1, h1)
        return qvals, cache

    def adam_update(self, grads, lr):
        self.t += 1

        def upd(param, m, v, g):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param = param - lr * m_hat / (np.sqrt(v_hat) + self.eps)
            return param, m, v

        self.W1, self.m_W1, self.v_W1 = upd(self.W1, self.m_W1, self.v_W1, grads["W1"])
        self.b1, self.m_b1, self.v_b1 = upd(self.b1, self.m_b1, self.v_b1, grads["b1"])
        self.W2, self.m_W2, self.v_W2 = upd(self.W2, self.m_W2, self.v_W2, grads["W2"])
        self.b2, self.m_b2, self.v_b2 = upd(self.b2, self.m_b2, self.v_b2, grads["b2"])

    def train_step(self, states, actions_idx, targets, lr):
        """
        MSE on chosen actions:
        loss = 0.5 * (Q(s,a) - target)^2
        """
        B = states.shape[0]
        qvals, cache = self.forward(states)
        x, z1, h1 = cache

        q_sa = qvals[np.arange(B), actions_idx]
        td = q_sa - targets
        loss = 0.5 * np.mean(td ** 2)

        # Gradient wrt qvals (only at chosen actions)
        grad_q = np.zeros_like(qvals)
        grad_q[np.arange(B), actions_idx] = td / B  # mean

        # Only NN correction has params (baseline has no gradients)
        # corr = h1 @ W2 + b2
        grad_W2 = h1.T @ grad_q
        grad_b2 = np.sum(grad_q, axis=0)

        grad_h1 = grad_q @ self.W2.T
        grad_z1 = grad_h1 * self.elu_derivative(z1)

        grad_W1 = x.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        grads = {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2}
        self.adam_update(grads, lr)
        return loss

# --------------------
# DQN Agent (independent learners)
# --------------------
class DQNAgent:
    def __init__(
        self,
        q_max,
        n_action_bins=101,
        hidden_dim=64,
        gamma=0.0,                 # contextual bandit -> 0.0 is the right default
        epsilon=0.3,
        epsilon_min=0.05,
        epsilon_decay=0.99996,
        buffer_capacity=200_000,
        batch_size=256,
        lr_init=0.003,
        lr_final=0.0001,
        warmup_steps=10_000,
        target_update_every=1000,
        seed=0,
    ):
        np.random.seed(seed)
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.warmup_steps = warmup_steps

        self.net = QNetwork(state_dim=3, hidden_dim=hidden_dim, n_actions=n_action_bins, q_max=q_max)
        self.target = QNetwork(state_dim=3, hidden_dim=hidden_dim, n_actions=n_action_bins, q_max=q_max)

        # copy params
        self._hard_update_target()

        self.buffer = ReplayBuffer(buffer_capacity, state_dim=3)
        self.target_update_every = target_update_every

        self.step = 0

    def _hard_update_target(self):
        self.target.W1 = self.net.W1.copy()
        self.target.b1 = self.net.b1.copy()
        self.target.W2 = self.net.W2.copy()
        self.target.b2 = self.net.b2.copy()

    def select_action_idx(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.net.n_actions)

        qvals, _ = self.net.forward(state[None, :])
        return int(np.argmax(qvals[0]))

    def lr_schedule(self, t, T):
        if t < self.warmup_steps:
            return self.lr_init * (t / max(self.warmup_steps, 1))
        progress = (t - self.warmup_steps) / max((T - self.warmup_steps), 1)
        progress = np.clip(progress, 0.0, 1.0)
        return self.lr_final + (self.lr_init - self.lr_final) * (1 - progress) ** 0.9

    def update(self, T):
        self.step += 1

        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.buffer.size < self.batch_size:
            return None

        lr_t = self.lr_schedule(self.step, T)

        s, a_idx, r, sp, d = self.buffer.sample(self.batch_size)

        # target = r + gamma * max_a' Q_target(s',a') * (1-done)
        # In this setup done is always 0; also gamma=0 is recommended.
        q_next, _ = self.target.forward(sp)
        max_next = np.max(q_next, axis=1)
        targets = r + self.gamma * max_next * (1.0 - d)

        loss = self.net.train_step(s, a_idx, targets, lr=lr_t)

        if self.step % self.target_update_every == 0:
            self._hard_update_target()

        return loss

# --------------------
# Training loop (like your script)
# --------------------
ITE = ENV_CONFIG["horizon"]

agents = [
    DQNAgent(
        q_max=ENV_CONFIG["q_max"],
        n_action_bins=101,       # quantities 0..100 step 1
        hidden_dim=64,
        gamma=0.0,               # IMPORTANT here
        epsilon=0.30,
        epsilon_min=0.05,
        epsilon_decay=0.99996,
        buffer_capacity=200_000,
        batch_size=256,
        lr_init=0.003,
        lr_final=0.0001,
        warmup_steps=10_000,
        target_update_every=1000,
        seed=ENV_CONFIG["seed"] + i,
    )
    for i in range(N_player)
]

# Storage (same idea as yours)
act = np.zeros((N_player, ITE))
u = np.zeros((N_player, ITE))
a_history = np.zeros(ITE)
b_history = np.zeros(ITE)
costs_history = np.zeros((N_player, ITE))
eps_history = np.zeros(ITE)
loss_history = np.zeros(ITE)

print("Training DQN-style Q-learning (residual Q network)...")
print_interval = 15_000

for t in range(ITE):
    if t % print_interval == 0:
        print(f"Step {t}/{ITE}")

    # sample random parameters
    a = np.random.uniform(ENV_CONFIG["a_min"], ENV_CONFIG["a_max"])
    b = np.random.uniform(ENV_CONFIG["b_min"], ENV_CONFIG["b_max"])
    costs = np.random.uniform(ENV_CONFIG["cost_min"], ENV_CONFIG["cost_max"], size=N_player)

    a_history[t] = a
    b_history[t] = b
    costs_history[:, t] = costs

    # select actions (epsilon-greedy)
    states = [np.array([a, b, costs[i]], dtype=float) for i in range(N_player)]
    a_idx = np.array([agents[i].select_action_idx(states[i]) for i in range(N_player)], dtype=int)

    q = np.array([agents[i].net.action_grid[a_idx[i]] for i in range(N_player)], dtype=float)
    act[:, t] = q

    # reward
    p = price(q, a, b)
    u[:, t] = (p - costs) * q

    # next state (fresh draw; contextual bandit)
    a2 = np.random.uniform(ENV_CONFIG["a_min"], ENV_CONFIG["a_max"])
    b2 = np.random.uniform(ENV_CONFIG["b_min"], ENV_CONFIG["b_max"])
    costs2 = np.random.uniform(ENV_CONFIG["cost_min"], ENV_CONFIG["cost_max"], size=N_player)
    next_states = [np.array([a2, b2, costs2[i]], dtype=float) for i in range(N_player)]

    # store + update
    for i in range(N_player):
        agents[i].buffer.add(states[i], a_idx[i], float(u[i, t]), next_states[i], 0.0)

    losses = []
    for i in range(N_player):
        loss = agents[i].update(T=ITE)
        if loss is not None:
            losses.append(loss)

    loss_history[t] = float(np.mean(losses)) if losses else np.nan
    eps_history[t] = agents[0].epsilon

print("Training complete!")

# --------------------
# Plotting (similar spirit to yours)
# --------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: profits (smoothed)
ax1 = axes[0]
window = 2000
u0 = u[0, :]
u1 = u[1, :]
u_smooth_0 = np.convolve(u0, np.ones(window)/window, mode='valid')
u_smooth_1 = np.convolve(u1, np.ones(window)/window, mode='valid')

ax1.plot(u_smooth_0, label='Agent 0', linewidth=2, alpha=0.8)
ax1.plot(u_smooth_1, label='Agent 1', linewidth=2, alpha=0.8)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Profit (smoothed)")
ax1.set_title("Training Convergence (Profit)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: learned policy response to cost (greedy)
ax2 = axes[1]
cost_range = np.linspace(ENV_CONFIG["cost_min"], ENV_CONFIG["cost_max"], 100)
a_fixed = 100.0
b_fixed = 1.0

learned_q0 = np.zeros(len(cost_range))
learned_q1 = np.zeros(len(cost_range))

# Force greedy for plotting
old_eps0 = agents[0].epsilon
old_eps1 = agents[1].epsilon
agents[0].epsilon = 0.0
agents[1].epsilon = 0.0

for idx, c in enumerate(cost_range):
    s0 = np.array([a_fixed, b_fixed, c], dtype=float)
    s1 = np.array([a_fixed, b_fixed, c], dtype=float)
    i0 = agents[0].select_action_idx(s0)
    i1 = agents[1].select_action_idx(s1)
    learned_q0[idx] = agents[0].net.action_grid[i0]
    learned_q1[idx] = agents[1].net.action_grid[i1]

# restore eps
agents[0].epsilon = old_eps0
agents[1].epsilon = old_eps1

# Nash (symmetric costs shown here)
nash_q = (a_fixed - cost_range) / (b_fixed * (N_player + 1))
nash_q = np.maximum(nash_q, 0.0)

ax2.plot(cost_range, learned_q0, linewidth=3, label='Agent 0 (Q policy)')
ax2.plot(cost_range, learned_q1, linewidth=3, label='Agent 1 (Q policy)')
ax2.plot(cost_range, nash_q, '--', linewidth=3, label='Nash (symmetric)', color='red', alpha=0.8)
ax2.set_xlabel("Marginal Cost")
ax2.set_ylabel("Quantity Produced")
ax2.set_title(f"Policy Response to Cost (a={a_fixed}, b={b_fixed})")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------
# Testing (same spirit as yours)
# --------------------
print("\n" + "="*60)
print("TESTING ON MULTIPLE DEMAND & COST SCENARIOS (GREEDY)")
print("="*60)

test_scenarios = [
    {"a": 100.0, "b": 1.0, "costs": [10.0, 15.0]},
    {"a": 90.0, "b": 1.0, "costs": [15.0, 20.0]},
    {"a": 110.0, "b": 1.0, "costs": [10.0, 20.0]},
    {"a": 100.0, "b": 0.9, "costs": [10.0, 15.0]},
    {"a": 100.0, "b": 1.1, "costs": [10.0, 15.0]},
]

# set greedy for test
old_eps = [ag.epsilon for ag in agents]
for ag in agents:
    ag.epsilon = 0.0

for scenario in test_scenarios:
    a_test = scenario["a"]
    b_test = scenario["b"]
    costs_test = np.array(scenario["costs"], dtype=float)

    mu = np.zeros(N_player)
    for i in range(N_player):
        s = np.array([a_test, b_test, costs_test[i]], dtype=float)
        idx = agents[i].select_action_idx(s)
        mu[i] = agents[i].net.action_grid[idx]

    q_nash = cournot_nash_asymmetric(a_test, b_test, costs_test)
    p_nash = max(a_test - b_test * q_nash.sum(), 0.0)
    profit_nash = (p_nash - costs_test) * q_nash

    p_mu = price(mu, a_test, b_test)
    profit_mu = (p_mu - costs_test) * mu

    print(f"\nScenario: a={a_test}, b={b_test}, costs={scenario['costs']}")
    print(f"  Learned q: {np.round(mu, 3)}")
    print(f"  Nash q*:   {np.round(q_nash, 3)}")
    print(f"  Abs error: {np.round(np.abs(mu - q_nash), 3)}")
    print(f"  Profit learned: {np.round(profit_mu, 3)}")
    print(f"  Profit Nash:    {np.round(profit_nash, 3)}")

# restore eps
for ag, e in zip(agents, old_eps):
    ag.epsilon = e

print("\n" + "="*60)
