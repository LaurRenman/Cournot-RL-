import numpy as np

class MixedPolicyAdapter:
    """
    Adapter that allows mixing 3D and 4D policies in the same trainer
    without modifying trainer.py or policy.py.
    """

    def __init__(self, policy, agent_id: int):
        self.policy = policy
        self.agent_id = agent_id
        self.input_dim = policy.input_dim  # for trainer

    def _rebuild_state(self, state_from_trainer: np.ndarray) -> np.ndarray:
        """
        Rebuild the correct state for this policy from the trainer-provided state.
        """
        a = state_from_trainer[0]
        b = state_from_trainer[1]

        if self.policy.input_dim == 4:
            # Trainer already gives [a, b, c_i, c_j]
            return state_from_trainer
        else:
            # Trainer gives [a, b, c_i, c_j] → asym needs [a, b, c_i]
            own_cost = state_from_trainer[2]
            return np.array([a, b, own_cost])

    # ---- methods used by CournotTrainer ----

    def forward(self, state):
        state = self._rebuild_state(state)
        return self.policy.forward(state)

    def backward(self, state, grad_output):
        state = self._rebuild_state(state)
        return self.policy.backward(state, grad_output)

    def adam_update(self, grads, lr):
        self.policy.adam_update(grads, lr)

    def _init_adam_params(self):
        self.policy._init_adam_params()

    # Safe copying (used by safe_copy)
    def copy_from(self, other_adapter):
        src = other_adapter.policy
        tgt = self.policy

        # Only copy if shapes match
        if tgt.w_linear.shape == src.w_linear.shape:
            tgt.w_linear = src.w_linear.copy()
            tgt.b_linear = src.b_linear
            tgt.W1 = src.W1.copy()
            tgt.b1 = src.b1.copy()
            tgt.W2 = src.W2.copy()
            tgt.b2 = src.b2.copy()
            tgt.input_mean = src.input_mean.copy()
            tgt.input_std = src.input_std.copy()

        tgt._init_adam_params()
