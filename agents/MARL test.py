import numpy as np
import matplotlib.pyplot as plt

# Number of firms
n = 2

# --------------------
# Environment settings
# --------------------
ENV_CONFIG = {
    "a_min": 80.0,      # minimum demand intercept
    "a_max": 120.0,     # maximum demand intercept
    "b_min": 0.8,       # minimum demand slope
    "b_max": 1.2,       # maximum demand slope
    "cost_min": 5.0,    # minimum cost
    "cost_max": 100.0,  # maximum cost
    "q_max": 100.0,
    "horizon": 150_000,
    "seed": 42
}

N_player = 2

# --------------------
# Set random seed
# --------------------
np.random.seed(ENV_CONFIG["seed"])

# --------------------
# Neural Network Policy with Residual Connection
# --------------------
class PolicyNetwork:
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        """
        Neural network with residual/skip connection to prevent mode collapse
        Architecture: Linear baseline + NN correction
        output = linear(input) + NN(input)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear baseline (like our original linear policy)
        # This ensures the network always has a reasonable starting point
        self.w_linear = np.array([0.33, -33.0, -0.33])  # [w_a, w_b, w_c]
        self.b_linear = 0.0
        
        # Neural network correction (learns residuals from linear baseline)
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        
        limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)  # Start with zero correction
        
        # Normalization statistics
        self.input_mean = np.array([100.0, 1.0, 52.5])
        self.input_std = np.array([15.0, 0.15, 35.0])
        
        # Adam optimizer parameters
        self.m_w_linear = np.zeros_like(self.w_linear)
        self.v_w_linear = np.zeros_like(self.w_linear)
        self.m_b_linear = 0.0
        self.v_b_linear = 0.0
        
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.adam_t = 0
    
    def elu(self, x, alpha=1.0):
        """ELU activation: smooth, non-zero gradients everywhere"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def elu_derivative(self, x, alpha=1.0):
        return np.where(x > 0, 1.0, alpha * np.exp(x))
    
    def forward(self, state):
        """
        Forward pass: linear baseline + neural network correction
        """
        # Linear baseline
        self.linear_output = np.dot(self.w_linear, state) + self.b_linear
        
        # Normalize input for NN
        x = (state - self.input_mean) / (self.input_std + 1e-8)
        self.x_norm = x
        
        # Neural network correction
        self.z1 = x @ self.W1 + self.b1
        self.h1 = self.elu(self.z1)
        
        self.z2 = self.h1 @ self.W2 + self.b2
        nn_correction = self.z2[0]
        
        # Combined output (linear + correction)
        output = self.linear_output + nn_correction
        
        # Ensure positive with smooth clipping
        output = np.maximum(0.1, output)  # small minimum to avoid exactly 0
        
        return output
    
    def backward(self, state, grad_output):
        """
        Backward pass for both linear and NN components
        """
        # Gradient through max(0.1, output)
        if self.linear_output + self.z2[0] > 0.1:
            grad_combined = grad_output
        else:
            grad_combined = 0.0  # gradient stops if we hit the floor
        
        # Gradients for NN correction
        grad_z2 = grad_combined
        grad_W2 = np.outer(self.h1, grad_z2)
        grad_b2 = grad_z2
        
        grad_h1 = grad_z2 * self.W2.flatten()
        grad_z1 = grad_h1 * self.elu_derivative(self.z1)
        grad_W1 = np.outer(self.x_norm, grad_z1)
        grad_b1 = grad_z1
        
        # Gradients for linear baseline
        grad_w_linear = grad_combined * state
        grad_b_linear = grad_combined
        
        return {
            'w_linear': grad_w_linear,
            'b_linear': grad_b_linear,
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2
        }
    
    def adam_update(self, grads, lr):
        """Update all parameters using Adam"""
        self.adam_t += 1
        
        # Update linear parameters
        self.m_w_linear = self.adam_beta1 * self.m_w_linear + (1 - self.adam_beta1) * grads['w_linear']
        self.v_w_linear = self.adam_beta2 * self.v_w_linear + (1 - self.adam_beta2) * (grads['w_linear'] ** 2)
        m_hat = self.m_w_linear / (1 - self.adam_beta1 ** self.adam_t)
        v_hat = self.v_w_linear / (1 - self.adam_beta2 ** self.adam_t)
        self.w_linear += lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
        
        self.m_b_linear = self.adam_beta1 * self.m_b_linear + (1 - self.adam_beta1) * grads['b_linear']
        self.v_b_linear = self.adam_beta2 * self.v_b_linear + (1 - self.adam_beta2) * (grads['b_linear'] ** 2)
        m_hat = self.m_b_linear / (1 - self.adam_beta1 ** self.adam_t)
        v_hat = self.v_b_linear / (1 - self.adam_beta2 ** self.adam_t)
        self.b_linear += lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
        
        # Update NN parameters (same as before)
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            m = getattr(self, f'm_{param_name}')
            v = getattr(self, f'v_{param_name}')
            grad = grads[param_name]
            
            m = self.adam_beta1 * m + (1 - self.adam_beta1) * grad
            v = self.adam_beta2 * v + (1 - self.adam_beta2) * (grad ** 2)
            
            m_hat = m / (1 - self.adam_beta1 ** self.adam_t)
            v_hat = v / (1 - self.adam_beta2 ** self.adam_t)
            
            param = getattr(self, param_name)
            param += lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            
            setattr(self, f'm_{param_name}', m)
            setattr(self, f'v_{param_name}', v)
            setattr(self, param_name, param)

# --------------------
# Parameters
# --------------------
ITE = ENV_CONFIG["horizon"]

# Learning rate schedule
lr_init = 0.003
lr_final = 0.0001
warmup_steps = 10000

# Create policy networks for each agent
policies = [PolicyNetwork() for _ in range(N_player)]

# Exploration schedule
sigma_init = 12.0
sigma_final = 3.0
sigma_decay = 0.99996
sigma = np.full(N_player, sigma_init)

# --------------------
# Price function
# --------------------
def price(acts, a, b):
    Q = np.sum(acts)
    return max(a - b * Q, 0.0)

# --------------------
# Nash equilibrium
# --------------------
def cournot_nash_asymmetric(a, b, costs):
    costs = np.array(costs)
    N = len(costs)
    total_cost = np.sum(costs)
    q_star = (a - (N + 1) * costs + total_cost) / (b * (N + 1))
    return np.maximum(q_star, 0.0)

# --------------------
# Storage
# --------------------
act = np.zeros((N_player, ITE))
u = np.zeros((N_player, ITE))
mu_t = np.zeros((N_player, ITE))
costs_history = np.zeros((N_player, ITE))
a_history = np.zeros(ITE)
b_history_demand = np.zeros(ITE)
sigma_history = np.zeros(ITE)

# Running statistics for advantage normalization
advantage_mean = np.zeros(N_player)
advantage_std = np.ones(N_player)
stats_alpha = 0.01

# --------------------
# Learning loop
# --------------------
print("Training neural network policies with residual architecture...")
print_interval = 15000

for t in range(ITE):
    if t % print_interval == 0:
        print(f"Episode {t}/{ITE}")
    
    # Learning rate schedule with warmup
    if t < warmup_steps:
        lr_t = lr_init * (t / warmup_steps)
    else:
        progress = (t - warmup_steps) / (ITE - warmup_steps)
        lr_t = lr_final + (lr_init - lr_final) * (1 - progress) ** 0.9
    
    # Decay exploration
    sigma = sigma * sigma_decay
    sigma = np.maximum(sigma, sigma_final)
    sigma_history[t] = sigma[0]

    # Sample random environment parameters
    a = np.random.uniform(ENV_CONFIG["a_min"], ENV_CONFIG["a_max"])
    b_demand = np.random.uniform(ENV_CONFIG["b_min"], ENV_CONFIG["b_max"])
    ind_cost = np.random.uniform(ENV_CONFIG["cost_min"], ENV_CONFIG["cost_max"], size=N_player)
    
    a_history[t] = a
    b_history_demand[t] = b_demand
    costs_history[:, t] = ind_cost

    # Compute policy mean using neural network
    for i in range(N_player):
        state = np.array([a, b_demand, ind_cost[i]])
        mu_t[i, t] = policies[i].forward(state)
        mu_t[i, t] = min(mu_t[i, t], ENV_CONFIG["q_max"])

    # Sample actions
    act[:, t] = np.random.normal(mu_t[:, t], sigma)
    act[:, t] = np.clip(act[:, t], 0.0, ENV_CONFIG["q_max"])

    # Market price and rewards
    p = price(act[:, t], a, b_demand)
    u[:, t] = (p - ind_cost) * act[:, t]
    
    # Best-response baseline
    baseline = np.zeros(N_player)
    for i in range(N_player):
        Q_minus_i = act[:, t].sum() - act[i, t]
        q_br = max((a - b_demand * Q_minus_i - ind_cost[i]) / (2 * b_demand), 0)
        p_br = max(a - b_demand * (Q_minus_i + q_br), 0)
        baseline[i] = (p_br - ind_cost[i]) * q_br
    
    # Advantage calculation
    advantage_signal = u[:, t] - baseline
    advantage_mean = (1 - stats_alpha) * advantage_mean + stats_alpha * advantage_signal
    advantage_std = (1 - stats_alpha) * advantage_std + stats_alpha * np.abs(advantage_signal - advantage_mean)
    advantage_normalized = (advantage_signal - advantage_mean) / (advantage_std + 1e-8)

    # Policy gradient update
    for i in range(N_player):
        state = np.array([a, b_demand, ind_cost[i]])
        
        score_grad = (act[i, t] - mu_t[i, t]) / (sigma[i] ** 2)
        grad_mu = advantage_normalized[i] * score_grad
        
        grads = policies[i].backward(state, grad_mu)
        policies[i].adam_update(grads, lr_t)

print("Training complete!")

# --------------------
# Plotting
# --------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training convergence
ax1 = axes[0]
window = 1000
u_smooth_0 = np.convolve(u[0, :], np.ones(window)/window, mode='valid')
u_smooth_1 = np.convolve(u[1, :], np.ones(window)/window, mode='valid')

ax1.plot(u_smooth_0, label='Agent 0', linewidth=2, alpha=0.8)
ax1.plot(u_smooth_1, label='Agent 1', linewidth=2, alpha=0.8)
ax1.set_xlabel("Training Episodes", fontsize=12)
ax1.set_ylabel("Profit (smoothed)", fontsize=12)
ax1.set_title("Training Convergence", fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Policy response to cost
ax2 = axes[1]

cost_range = np.linspace(ENV_CONFIG["cost_min"], ENV_CONFIG["cost_max"], 100)
a_fixed = 100.0
b_fixed = 1.0

learned_q0 = np.zeros(len(cost_range))
learned_q1 = np.zeros(len(cost_range))

for idx, c in enumerate(cost_range):
    state0 = np.array([a_fixed, b_fixed, c])
    state1 = np.array([a_fixed, b_fixed, c])
    
    learned_q0[idx] = min(policies[0].forward(state0), ENV_CONFIG["q_max"])
    learned_q1[idx] = min(policies[1].forward(state1), ENV_CONFIG["q_max"])

nash_q = (a_fixed - cost_range) / (b_fixed * (N_player + 1))
nash_q = np.maximum(nash_q, 0)

ax2.plot(cost_range, learned_q0, linewidth=3, label='Agent 0 (NN policy)', color='C0')
ax2.plot(cost_range, learned_q1, linewidth=3, label='Agent 1 (NN policy)', color='C1')
ax2.plot(cost_range, nash_q, '--', linewidth=3, label='Nash Equilibrium', 
         color='red', alpha=0.8)

ax2.set_xlabel("Marginal Cost", fontsize=12)
ax2.set_ylabel("Quantity Produced", fontsize=12)
ax2.set_title(f"Policy Response to Cost (a={a_fixed}, b={b_fixed})", 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------
# Testing
# --------------------
print("\n" + "="*60)
print("TESTING ON MULTIPLE DEMAND & COST SCENARIOS")
print("="*60)

test_scenarios = [
    {"a": 100.0, "b": 1.0, "costs": [10.0, 15.0]},
    {"a": 90.0, "b": 1.0, "costs": [15.0, 20.0]},
    {"a": 110.0, "b": 1.0, "costs": [10.0, 20.0]},
    {"a": 100.0, "b": 0.9, "costs": [10.0, 15.0]},
    {"a": 100.0, "b": 1.1, "costs": [10.0, 15.0]},
]

for scenario in test_scenarios:
    a_test = scenario["a"]
    b_test = scenario["b"]
    costs_test = np.array(scenario["costs"])
    
    N_test = 5_000
    act_test = np.zeros((N_player, N_test))
    profit_test = np.zeros((N_player, N_test))
    
    mu_test = np.zeros(N_player)
    for i in range(N_player):
        state = np.array([a_test, b_test, costs_test[i]])
        mu_test[i] = min(policies[i].forward(state), ENV_CONFIG["q_max"])
    
    for t in range(N_test):
        act_test[:, t] = np.random.normal(mu_test, sigma_final)
        act_test[:, t] = np.clip(act_test[:, t], 0.0, ENV_CONFIG["q_max"])
        
        p = price(act_test[:, t], a_test, b_test)
        profit_test[:, t] = (p - costs_test) * act_test[:, t]
    
    q_nash_test = cournot_nash_asymmetric(a_test, b_test, costs_test)
    p_nash = max(a_test - b_test * q_nash_test.sum(), 0.0)
    profit_nash = (p_nash - costs_test) * q_nash_test
    
    print(f"\nScenario: a={a_test}, b={b_test}, costs={scenario['costs']}")
    print(f"  Learned μ: {mu_test}")
    print(f"  Nash q*:   {q_nash_test}")
    print(f"  Error:     {np.abs(mu_test - q_nash_test)}")
    print(f"  Profit ratio: {profit_test.mean(axis=1) / profit_nash}")

print("\n" + "="*60)