# Regret-Minimizing Reinforcement Learning in Cournot Competition

This repository implements a **regret-minimization–based reinforcement learning framework** for Cournot competition with agents that have **heterogeneous information structures**.

Agents learn quantity-setting policies in a stochastic Cournot market using **policy gradient updates driven by counterfactual regret**, rather than direct reward maximization. The framework supports **minimal, partial, asymmetric, and full-information (Nash) agents**, enabling controlled comparisons of informational advantage.

---

## Core Idea

Each firm learns to minimize **instantaneous regret**:

Regret_i = max_{q_i'} π_i(q_i', q_{−i}) − π_i(q_i, q_{−i})

Instead of learning to maximize profit directly, agents learn policies whose **regret converges to zero**, which characterizes Nash equilibrium behavior in Cournot games.

Key properties:

* Regret → 0 ⇒ no profitable unilateral deviation
* Learning signal is stable even under stochastic exploration
* Naturally supports asymmetric information

---

## Features

* **Multiple information regimes**

  * `1D` — Minimal: own cost only
  * `2D` — Partial: demand slope + own cost
  * `3D` — Asymmetric: full demand + own cost
  * `4D` — Nash: full information (demand + opponent cost)

* **Residual policy architecture**

  * Linear Cournot-inspired baseline
  * Neural network learns only residual corrections
  * Prevents mode collapse and stabilizes learning

* **Regret-normalized policy gradients**

  * Automatically rescales regret to ensure stable gradients
  * Preserves equilibrium fixed points

* **Self-play support**

  * Learning agent periodically clones itself into opponent
  * Enables Nash convergence without hard-coding equilibrium

* **Full evaluation suite**

  * Policy response plots
  * Training convergence
  * Cross-information agent comparisons
  * Profit and quantity distributions

---

## Project Structure

```
.
├── environment.py      # Cournot market dynamics and Nash solver
├── policy.py           # Policy networks (asymmetric + Nash)
├── trainer.py          # Regret-based multi-agent trainer
├── utils.py            # Plotting and evaluation utilities
├── experiments.ipynb   # Training + evaluation notebook
├── README.md
```

---

## Environment: Cournot Game

Market price is linear:

[
P(Q) = \max(a - bQ, 0)
]

Firm ( i )’s profit:

[
\pi_i = (P(Q) - c_i) q_i
]

Where:

* ( a, b ) are sampled each episode
* Costs ( c_i ) are private information
* Quantity is bounded: ( q_i \in [0, q_{\max}] )

The environment provides:

* Exact Nash equilibrium (closed-form)
* Best-response computation
* Regret evaluation

---

## Policy Architecture

All policies share the same structure:

[
q(s) = \underbrace{w^\top s + b}*{\text{Cournot baseline}}
;;+;;
\underbrace{\text{NN}(s)}*{\text{learned correction}}
]

### Why residual learning?

* The linear term encodes approximate Cournot logic
* The NN only learns deviations
* Faster convergence, better stability, interpretable behavior

Activation: **ELU**
Optimizer: **Adam (manual NumPy implementation)**

---

## Training Algorithm

Each episode:

1. Sample market parameters ((a, b)) and costs
2. Agents observe states according to their information level
3. Actions sampled with Gaussian exploration
4. Profits computed
5. **Counterfactual best responses** computed
6. Regret calculated
7. Policy updated via regret-minimizing policy gradient

### Learning Signal

[
\nabla J \propto -\text{Regret} \times \nabla \log \pi(q | s)
]

Regret is optionally **normalized online** to stabilize learning.

---

## Example: Training Agents

```python
env = CournotEnvironment(env_config)

learning_agent = PolicyNetwork_Nash(input_dim=4)
opponent_agent = PolicyNetwork_Nash(input_dim=4)

trainer = CournotTrainer(
    n_players=2,
    env=env,
    policies=[learning_agent, opponent_agent],
    config=config
)

history = trainer.train()
```

---

## Information Regimes Compared

The project explicitly trains and compares:

| Agent Type   | Input | Knowledge               |
| ------------ | ----- | ----------------------- |
| Perfect Info | 4D    | Demand + both costs     |
| Asymmetric   | 3D    | Demand + own cost       |
| Partial      | 2D    | Demand slope + own cost |
| Minimal      | 1D    | Own cost only           |

Agents are evaluated **head-to-head** under identical market conditions.

---

## Evaluation & Results

The evaluation framework computes:

* Mean quantities and profits
* Variance under exploration
* Distance from Nash equilibrium
* Profit advantage by information level

Example tests:

* Perfect vs Perfect
* Perfect vs Partial
* Partial vs Minimal
* Asymmetric vs Nash

Theoretical Nash outcomes are printed alongside learned behavior.

---

## Why Regret Normalization Matters

Raw regret values are often small (e.g. 0.1–5.0).
Without normalization:

* Gradients vanish
* Learning stagnates

Normalization **rescales regret without biasing equilibrium**:

* Regret = 0 remains a fixed point
* Only learning speed is affected

---

## Dependencies

* Python ≥ 3.9
* NumPy
* Matplotlib

No deep learning frameworks required — everything is implemented in NumPy for transparency.

---

## Intended Use

This codebase is suitable for:

* Research on learning in games
* Information asymmetry experiments
* Economic RL benchmarks
* Teaching regret minimization and Nash learning
* Extensions to auctions, pricing, or oligopoly models

---

## Future Extensions

* More than 2 players
* Endogenous information acquisition
* Correlated equilibrium learning
* Demand uncertainty with belief learning
* Comparison to Q-learning / PPO baselines

---

## Citation

If you use or adapt this code for academic work, please cite appropriately.
