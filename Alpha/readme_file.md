# Multi-Agent Reinforcement Learning: Cournot Competition

This project implements neural network policies for multi-agent reinforcement learning in a Cournot oligopoly market. Agents learn to produce optimal quantities through policy gradient methods.

## Project Structure

```
.
├── environment.py              # Market environment and dynamics
├── policy.py                   # Neural network policy architecture
├── trainer.py                  # Training loop and MARL algorithms
├── utils.py                    # Visualization and analysis tools
├── main.py                     # Command-line training script
├── cournot_marl_experiments.ipynb  # Jupyter notebook for experiments
└── README.md                   # This file
```

## File Descriptions

### `environment.py`
Defines the Cournot competition environment:
- **CournotEnvironment**: Main environment class
  - `sample_scenario()`: Generate random market parameters (demand, costs)
  - `price()`: Calculate market price from total quantity
  - `profit()`: Calculate firm profits
  - `best_response()`: Compute best-response quantity
  - `nash_equilibrium()`: Calculate Nash equilibrium quantities

**Easy modifications:**
- Change demand function shape (linear to non-linear)
- Add fixed costs or capacity constraints
- Modify cost structures (increasing, decreasing)
- Add market shocks or time-varying parameters

### `policy.py`
Neural network policy with residual architecture:
- **PolicyNetwork**: Deep learning policy
  - Linear baseline + neural network correction
  - ELU activation for smooth gradients
  - Adam optimizer
  - Forward/backward passes for training

**Easy modifications:**
- Change network architecture (depth, width)
- Experiment with different activations (ReLU, tanh, etc.)
- Modify initialization strategies
- Add dropout or batch normalization

### `trainer.py`
Multi-agent training loop:
- **CournotTrainer**: Main training class
  - Policy gradient updates
  - Advantage estimation with best-response baseline
  - Exploration scheduling
  - Learning rate scheduling
  - History tracking

**Easy modifications:**
- Change learning algorithm (PPO, TRPO, etc.)
- Modify advantage estimation methods
- Adjust exploration strategies
- Add curriculum learning

### `utils.py`
Visualization and analysis utilities:
- `plot_training_convergence()`: Training progress
- `plot_policy_response()`: Policy behavior vs cost
- `plot_exploration_schedule()`: Exploration and learning rates
- `plot_action_distribution()`: Action histograms over time
- `plot_profit_analysis()`: Comprehensive profit dashboard
- `compare_agents()`: Multi-scenario comparison
- `print_evaluation_summary()`: Text results

**Easy modifications:**
- Add new visualization types
- Create custom metrics
- Export data to CSV/JSON
- Add statistical tests

## Quick Start

### Command Line
```bash
python main.py
```

This will:
1. Train 2 agents for 150,000 episodes
2. Generate 4 visualization plots
3. Evaluate on 5 test scenarios
4. Save trained policies

### Jupyter Notebook
```bash
jupyter notebook cournot_marl_experiments.ipynb
```

The notebook includes:
- Basic 2-agent training
- Training convergence analysis
- Policy response visualization
- Action distribution evolution
- Comprehensive profit analysis
- Multi-scenario evaluation
- 3-agent experiments
- Custom environment experiments
- Market dynamics analysis

## Common Modifications

### 1. Change Number of Agents
```python
n_agents = 3  # Or any number
policies = [PolicyNetwork() for _ in range(n_agents)]
trainer = CournotTrainer(n_agents, env, policies, config)
```

### 2. Modify Market Parameters
```python
env_config = {
    "a_min": 100.0,      # Higher demand
    "a_max": 150.0,
    "b_min": 1.5,        # Steeper demand curve
    "b_max": 2.0,
    "cost_min": 10.0,    # Different cost range
    "cost_max": 50.0,
    "q_max": 100.0,
    "seed": 42
}
env = CournotEnvironment(env_config)
```

### 3. Adjust Training Parameters
```python
train_config = {
    'episodes': 200000,     # More training
    'lr_init': 0.005,       # Higher initial learning rate
    'lr_final': 0.00005,    # Lower final learning rate
    'sigma_init': 15.0,     # More exploration
    'sigma_final': 2.0,     # Less final noise
}
```

### 4. Evaluate Specific Scenario
```python
results = trainer.evaluate(
    a=100.0,           # Demand intercept
    b=1.0,             # Demand slope
    costs=[10, 20],    # Agent costs
    n_episodes=5000
)
print_evaluation_summary(results)
```

### 5. Compare Different Configurations
```python
scenarios = [
    {"a": 100, "b": 1.0, "costs": [10, 15]},
    {"a": 120, "b": 0.8, "costs": [20, 25]},
    # Add more scenarios
]
fig, results = compare_agents(policies, env, scenarios)
```

## Dependencies

```bash
pip install numpy matplotlib
```

For Jupyter notebook:
```bash
pip install jupyter
```

## Key Concepts

### Cournot Competition
- Firms compete by choosing quantities
- Market price determined by total quantity
- Each firm maximizes profit given others' quantities

### Nash Equilibrium
- No firm can improve profit by unilaterally changing quantity
- Computed analytically for comparison with learned policies

### Policy Gradient Learning
- Agents learn through experience sampling
- Advantage function guides policy updates
- Exploration vs exploitation balance

### Residual Architecture
- Linear baseline ensures stable learning
- Neural network learns corrections/refinements
- Prevents mode collapse

## Metrics and Analysis

The code tracks and visualizes:

1. **Training Metrics:**
   - Profit over time (smoothed)
   - Cumulative profits
   - Action distributions
   - Exploration noise decay
   - Learning rate schedule

2. **Performance Metrics:**
   - Absolute error vs Nash equilibrium
   - Profit ratio (learned/Nash)
   - Mean and variance of profits
   - Correlation between agents

3. **Market Metrics:**
   - Market prices
   - Total quantities
   - Market concentration (HHI)
   - Strategic interactions

## Extending the Code

### Add New Policy Architecture
```python
class MyCustomPolicy(PolicyNetwork):
    def __init__(self):
        super().__init__()
        # Your custom architecture
    
    def forward(self, state):
        # Your forward pass
        pass
```

### Add New Environment Dynamics
```python
class CustomEnvironment(CournotEnvironment):
    def price(self, quantities, a, b):
        # Your custom price function
        # E.g., non-linear demand, taxes, etc.
        pass
```

### Add New Training Algorithm
```python
class CustomTrainer(CournotTrainer):
    def train_step(self, episode):
        # Your custom training logic
        # E.g., PPO, A3C, MADDPG, etc.
        pass
```

## Tips for Experimentation

1. **Start small**: Train with fewer episodes first to test changes
2. **Use the notebook**: Interactive experimentation is easier
3. **Save policies**: Save successful policies before trying new things
4. **Compare baselines**: Always compare with Nash equilibrium
5. **Visualize early**: Plot results frequently to catch issues

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cournot_marl_2024,
  title={Multi-Agent Reinforcement Learning for Cournot Competition},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/cournot-marl}}
}
```

## License

MIT License - feel free to use and modify as needed.

## Contact

For questions or suggestions, please open an issue on GitHub.
