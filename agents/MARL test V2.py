import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
ENV_CONFIG = {
    "a_min": 80.0,
    "a_max": 120.0,
    "b_min": 0.8,
    "b_max": 1.2,
    "cost_min": 5.0,
    "cost_max": 50.0,
    "q_max": 100.0,
    "horizon": 100000,
    "seed": 42
}

N_player = 2
np.random.seed(ENV_CONFIG["seed"])

# =============================================================================
# NEURAL NETWORK POLICY
# =============================================================================
class NeuralPolicy:
    def __init__(self):
        # Simple 2-layer NN: input(3) -> hidden(32) -> output(1)
        self.W1 = np.random.randn(3, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 1) * 0.1
        self.b2 = np.zeros(1)
        
        # Adam optimizer
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.t = 0
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, state):
        # Normalize input
        state_norm = (state - np.array([100, 1, 27.5])) / np.array([15, 0.2, 15])
        
        self.z1 = state_norm @ self.W1 + self.b1
        self.h1 = self.relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        
        # Output with sigmoid scaled to [0, q_max]
        output = ENV_CONFIG["q_max"] / (1 + np.exp(-self.z2[0]))
        return float(output)
    
    def backward(self, state, grad_output, lr=0.001):
        # Normalize input
        state_norm = (state - np.array([100, 1, 27.5])) / np.array([15, 0.2, 15])
        
        # Backprop through sigmoid
        sigmoid_out = ENV_CONFIG["q_max"] / (1 + np.exp(-self.z2[0]))
        grad_z2 = grad_output * sigmoid_out * (1 - sigmoid_out / ENV_CONFIG["q_max"])
        
        grad_W2 = np.outer(self.h1, grad_z2)
        grad_b2 = np.array([grad_z2])
        
        grad_h1 = grad_z2 * self.W2.flatten()
        grad_z1 = grad_h1 * self.relu_derivative(self.z1)
        
        grad_W1 = np.outer(state_norm, grad_z1)
        grad_b1 = grad_z1
        
        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # W1
        self.m_W1 = beta1 * self.m_W1 + (1 - beta1) * grad_W1
        self.v_W1 = beta2 * self.v_W1 + (1 - beta2) * (grad_W1 ** 2)
        m_hat = self.m_W1 / (1 - beta1 ** self.t)
        v_hat = self.v_W1 / (1 - beta2 ** self.t)
        self.W1 += lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # b1
        self.m_b1 = beta1 * self.m_b1 + (1 - beta1) * grad_b1
        self.v_b1 = beta2 * self.v_b1 + (1 - beta2) * (grad_b1 ** 2)
        m_hat = self.m_b1 / (1 - beta1 ** self.t)
        v_hat = self.v_b1 / (1 - beta2 ** self.t)
        self.b1 += lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # W2
        self.m_W2 = beta1 * self.m_W2 + (1 - beta1) * grad_W2
        self.v_W2 = beta2 * self.v_W2 + (1 - beta2) * (grad_W2 ** 2)
        m_hat = self.m_W2 / (1 - beta1 ** self.t)
        v_hat = self.v_W2 / (1 - beta2 ** self.t)
        self.W2 += lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # b2
        self.m_b2 = beta1 * self.m_b2 + (1 - beta1) * grad_b2
        self.v_b2 = beta2 * self.v_b2 + (1 - beta2) * (grad_b2 ** 2)
        m_hat = self.m_b2 / (1 - beta1 ** self.t)
        v_hat = self.v_b2 / (1 - beta2 ** self.t)
        self.b2 += lr * m_hat / (np.sqrt(v_hat) + eps)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def price(quantities, a, b):
    Q = np.sum(quantities)
    return max(a - b * Q, 0.0)

def nash_equilibrium(a, b, costs):
    costs = np.array(costs)
    N = len(costs)
    total_cost = np.sum(costs)
    q_star = (a - (N + 1) * costs + total_cost) / (b * (N + 1))
    return np.maximum(q_star, 0.0)

# =============================================================================
# TRAINING WITH POLICY GRADIENT
# =============================================================================
print("="*80)
print("🧠 TRAINING NEURAL NETWORK COURNOT AGENTS")
print("="*80)

ITE = ENV_CONFIG["horizon"]
policies = [NeuralPolicy(), NeuralPolicy()]

# Storage
actions = np.zeros((N_player, ITE))
profits = np.zeros((N_player, ITE))
means = np.zeros((N_player, ITE))
costs_history = np.zeros((N_player, ITE))
a_history = np.zeros(ITE)
b_history = np.zeros(ITE)
nash_quantities = np.zeros((N_player, ITE))

# Hyperparameters
sigma_init = 10.0
sigma_final = 3.0
sigma = sigma_init

# Baseline for variance reduction
baseline = [0, 0]
baseline_alpha = 0.01

for t in range(ITE):
    if t % 10000 == 0:
        print(f"Episode {t}/{ITE} | σ={sigma:.2f}")
    
    # Decay exploration
    sigma = max(sigma_final, sigma_init * (0.9995 ** t))
    
    # Learning rate schedule
    lr = 0.005 * (0.9997 ** t)
    lr = max(0.0001, lr)
    
    # Sample environment
    a = np.random.uniform(ENV_CONFIG["a_min"], ENV_CONFIG["a_max"])
    b = np.random.uniform(ENV_CONFIG["b_min"], ENV_CONFIG["b_max"])
    costs = np.random.uniform(ENV_CONFIG["cost_min"], ENV_CONFIG["cost_max"], N_player)
    
    # Store
    a_history[t] = a
    b_history[t] = b
    costs_history[:, t] = costs
    
    # Compute Nash
    nash_q = nash_equilibrium(a, b, costs)
    nash_quantities[:, t] = nash_q
    
    # Forward pass
    for i in range(N_player):
        state = np.array([a, b, costs[i]])
        means[i, t] = policies[i].forward(state)
    
    # Sample actions with exploration
    actions[:, t] = means[:, t] + np.random.normal(0, sigma, N_player)
    actions[:, t] = np.clip(actions[:, t], 0, ENV_CONFIG["q_max"])
    
    # Compute profits
    p = price(actions[:, t], a, b)
    profits[:, t] = (p - costs) * actions[:, t]
    
    # Update baseline (moving average)
    for i in range(N_player):
        baseline[i] = (1 - baseline_alpha) * baseline[i] + baseline_alpha * profits[i, t]
    
    # Policy gradient update
    for i in range(N_player):
        # Advantage
        advantage = profits[i, t] - baseline[i]
        
        # Score function gradient
        score_grad = (actions[i, t] - means[i, t]) / (sigma ** 2 + 1e-8)
        
        # Policy gradient
        grad_output = advantage * score_grad
        
        # Backprop
        state = np.array([a, b, costs[i]])
        policies[i].backward(state, grad_output, lr)

print("✅ Training complete!\n")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 10))

# Plot 1: PROFIT
ax1 = plt.subplot(2, 3, 1)
window = 500
profit_smooth_0 = np.convolve(profits[0, :], np.ones(window)/window, mode='valid')
profit_smooth_1 = np.convolve(profits[1, :], np.ones(window)/window, mode='valid')

ax1.plot(profit_smooth_0, linewidth=2.5, label='Agent 0', color='#2E86DE')
ax1.plot(profit_smooth_1, linewidth=2.5, label='Agent 1', color='#FF6B6B')
ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Episodes', fontsize=13, fontweight='bold')
ax1.set_ylabel('Profit', fontsize=13, fontweight='bold')
ax1.set_title('📈 Profit Evolution', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: QUANTITIES
ax2 = plt.subplot(2, 3, 2)
q_smooth_0 = np.convolve(actions[0, :], np.ones(window)/window, mode='valid')
q_smooth_1 = np.convolve(actions[1, :], np.ones(window)/window, mode='valid')
nash_smooth_0 = np.convolve(nash_quantities[0, :], np.ones(window)/window, mode='valid')

ax2.plot(q_smooth_0, linewidth=2.5, label='Agent 0', color='#2E86DE')
ax2.plot(q_smooth_1, linewidth=2.5, label='Agent 1', color='#FF6B6B')
ax2.plot(nash_smooth_0, '--', linewidth=2, label='Nash', color='black', alpha=0.5)
ax2.set_xlabel('Episodes', fontsize=13, fontweight='bold')
ax2.set_ylabel('Quantity', fontsize=13, fontweight='bold')
ax2.set_title('📊 Quantities vs Nash', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: ERROR
ax3 = plt.subplot(2, 3, 3)
error_0 = np.abs(actions[0, :] - nash_quantities[0, :])
error_1 = np.abs(actions[1, :] - nash_quantities[1, :])
error_smooth_0 = np.convolve(error_0, np.ones(window)/window, mode='valid')
error_smooth_1 = np.convolve(error_1, np.ones(window)/window, mode='valid')

ax3.plot(error_smooth_0, linewidth=2.5, label='Agent 0', color='#2E86DE')
ax3.plot(error_smooth_1, linewidth=2.5, label='Agent 1', color='#FF6B6B')
ax3.set_xlabel('Episodes', fontsize=13, fontweight='bold')
ax3.set_ylabel('Error', fontsize=13, fontweight='bold')
ax3.set_title('📉 |Learned - Nash|', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: DISTRIBUTION
ax4 = plt.subplot(2, 3, 4)
last_n = 20000

ax4.hist(actions[0, -last_n:], bins=40, alpha=0.6, color='#2E86DE', 
         edgecolor='black', label='Agent 0')
ax4.hist(nash_quantities[0, -last_n:], bins=40, alpha=0.6, color='red', 
         edgecolor='black', label='Nash')
ax4.axvline(actions[0, -last_n:].mean(), color='#2E86DE', linestyle='--', linewidth=3)
ax4.axvline(nash_quantities[0, -last_n:].mean(), color='red', linestyle='--', linewidth=3)
ax4.set_xlabel('Quantity', fontsize=13, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax4.set_title('📊 Agent 0: Distribution', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: SCATTER
ax5 = plt.subplot(2, 3, 5)
sample_idx = np.random.choice(ITE, 3000, replace=False)
scatter = ax5.scatter(nash_quantities[0, sample_idx], actions[0, sample_idx], 
                      c=costs_history[0, sample_idx], cmap='coolwarm', 
                      alpha=0.5, s=20)
max_q = max(nash_quantities[0, :].max(), actions[0, :].max())
ax5.plot([0, max_q], [0, max_q], 'r--', linewidth=3, label='Perfect')
plt.colorbar(scatter, ax=ax5, label='Cost')
ax5.set_xlabel('Nash', fontsize=13, fontweight='bold')
ax5.set_ylabel('Learned', fontsize=13, fontweight='bold')
ax5.set_title('🎯 Learned vs Nash', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# Plot 6: POLICY RESPONSE
ax6 = plt.subplot(2, 3, 6)
cost_range = np.linspace(5, 50, 100)
a_test, b_test = 100.0, 1.0

learned_0 = [policies[0].forward(np.array([a_test, b_test, c])) for c in cost_range]
learned_1 = [policies[1].forward(np.array([a_test, b_test, c])) for c in cost_range]
nash_theory = np.maximum((a_test - cost_range) / 3, 0)

ax6.plot(cost_range, learned_0, linewidth=3.5, label='Agent 0', color='#2E86DE', marker='o', markersize=3, markevery=10)
ax6.plot(cost_range, learned_1, linewidth=3.5, label='Agent 1', color='#FF6B6B', marker='s', markersize=3, markevery=10)
ax6.plot(cost_range, nash_theory, '--', linewidth=3, label='Nash', color='black')
ax6.set_xlabel('Cost', fontsize=13, fontweight='bold')
ax6.set_ylabel('Quantity', fontsize=13, fontweight='bold')
ax6.set_title('🔄 Policy Response', fontsize=14, fontweight='bold')
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)

plt.suptitle('🧠 Neural Network Cournot Learning', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cournot_nn.png', dpi=200, bbox_inches='tight')
plt.show()

# =============================================================================
# STATISTICS
# =============================================================================
print("\n" + "="*80)
print("📊 FINAL STATISTICS")
print("="*80)

for i in range(N_player):
    learned = actions[i, -last_n:]
    nash = nash_quantities[i, -last_n:]
    error = np.abs(learned - nash).mean()
    error_pct = 100 * error / nash.mean()
    
    print(f"\n🤖 AGENT {i}:")
    print(f"  Learned Q: {learned.mean():.2f} ± {learned.std():.2f}")
    print(f"  Nash Q:    {nash.mean():.2f} ± {nash.std():.2f}")
    print(f"  Error:     {error:.2f} ({error_pct:.1f}%)")
    print(f"  Profit:    {profits[i, -last_n:].mean():.1f}")

overall_error = 100 * np.abs(actions[:, -last_n:] - nash_quantities[:, -last_n:]).mean() / nash_quantities[:, -last_n:].mean()
print(f"\n🎯 Overall Error: {overall_error:.2f}%")

if overall_error < 10:
    print("⭐⭐⭐⭐⭐ EXCELLENT!")
elif overall_error < 20:
    print("⭐⭐⭐⭐ VERY GOOD!")
else:
    print("⭐⭐⭐ GOOD!")

print("="*80)