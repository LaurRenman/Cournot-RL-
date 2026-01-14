"""
Configuration file for Cournot RL experiments.
"""

# --------------------
# Environment settings
# --------------------
ENV_CONFIG = {
    "n_firms": 2,
    "a": 100.0,
    "b": 1.0,
    "costs": [10.0, 10.0],
    "q_max": 100.0,
    "horizon": 50,
    "seed": 42
}

# --------------------
# Training settings
# --------------------
TRAINING_CONFIG = {
    "num_episodes": 1000,
    "log_every": 50
}

# --------------------
# Evaluation settings
# --------------------
EVAL_CONFIG = {
    "num_episodes": 100
}
