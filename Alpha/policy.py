"""
Policy Network Module

This module defines neural network policies for agents in the Cournot game,
including residual architectures and optimization methods.
Supports flexible input dimensions: 1D (minimal), 2D (partial demand), 3D (asymmetric), 4D (Nash).
"""

import numpy as np
from typing import Dict, Optional, Union


class PolicyNetwork_asymmetrical:
    """
    Neural network policy with residual connection to prevent mode collapse.
    Architecture: output = linear(input) + NN(input)
    Supports flexible input dimensions for different information levels.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, 
                 output_dim: int = 1, config: Optional[Dict] = None):
        """
        Initialize policy network.
        
        Args:
            input_dim: Dimension of input state (1, 2, 3, or 4)
            hidden_dim: Number of hidden units
            output_dim: Dimension of output
            config: Optional configuration dictionary
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config or {}
        
        # Initialize linear baseline based on input dimension
        if input_dim == 1:
            # 1D: [own_cost] only
            # Better baseline: assume average demand (a≈100, b≈1)
            # Nash formula: q ≈ (100 - 1*q_opp - c) / 2
            # Assume opponent plays ~30 → q ≈ (70 - c) / 2 = 35 - 0.5*c
            self.w_linear = np.array([-0.5])  # Cost coefficient
            self.b_linear = 35.0  # Intercept based on average market
            self.input_mean = np.array([27.5])
            self.input_std = np.array([15.0])
            
        elif input_dim == 2:
            # 2D: [b, own_cost] - knows slope but not intercept
            # Nash formula: q = (a - b*q_opp - c) / (2b)
            # Assume a≈100, q_opp≈30 → q ≈ (70 - c) / (2b) = 35/b - c/(2b)
            self.w_linear = np.array([35.0, -0.5])  # [w_b, w_c]
            self.b_linear = 0.0
            self.input_mean = np.array([1.0, 27.5])
            self.input_std = np.array([0.15, 15.0])
            
        elif input_dim == 3:
            # 3D: [a, b, own_cost]
            self.w_linear = np.array([0.33, -33.0, -0.33])
            self.b_linear = 0.0
            self.input_mean = np.array([100.0, 1.0, 27.5])
            self.input_std = np.array([15.0, 0.15, 15.0])
            
        elif input_dim == 4:
            # 4D: [a, b, own_cost, opponent_cost]
            self.w_linear = np.array([0.33, -33.0, -0.33, 0.16])
            self.b_linear = 0.0
            self.input_mean = np.array([100.0, 1.0, 27.5, 27.5])
            self.input_std = np.array([15.0, 0.15, 15.0, 15.0])
            
        else:
            raise ValueError(f"Unsupported input_dim: {input_dim}. Use 1, 2, 3, or 4.")
        
        # Neural network correction (learns residuals from linear baseline)
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        
        limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)
        
        # Adam optimizer parameters
        self._init_adam_params()
        
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.adam_t = 0
    
    def _init_adam_params(self):
        """Initialize Adam optimizer parameters."""
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
    
    def elu(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation: smooth, non-zero gradients everywhere."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def elu_derivative(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Derivative of ELU activation."""
        return np.where(x > 0, 1.0, alpha * np.exp(x))
    
    def forward(self, state: np.ndarray) -> float:
        """
        Forward pass: linear baseline + neural network correction.
        
        Args:
            state: Input state (1D, 2D, 3D, or 4D depending on initialization)
            
        Returns:
            Output quantity
        """
        # Ensure state matches expected dimension
        if len(state) != self.input_dim:
            raise ValueError(f"Expected state of dimension {self.input_dim}, got {len(state)}")
        
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
        output = np.maximum(0.1, output)
        
        return output
    
    def backward(self, state: np.ndarray, grad_output: float) -> Dict[str, np.ndarray]:
        """
        Backward pass for both linear and NN components.
        
        Args:
            state: Input state
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Dictionary of gradients for all parameters
        """
        # Gradient through max(0.1, output)
        if self.linear_output + self.z2[0] > 0.1:
            grad_combined = grad_output
        else:
            grad_combined = 0.0
        
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
    
    def adam_update(self, grads: Dict[str, np.ndarray], lr: float):
        """
        Update all parameters using Adam optimizer.
        
        Args:
            grads: Dictionary of gradients
            lr: Learning rate
        """
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
        
        # Update NN parameters
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
    
    def get_action(self, state: np.ndarray, sigma: float = 0.0, 
                   q_max: float = 100.0) -> float:
        """
        Get action with optional exploration noise.
        
        Args:
            state: Input state
            sigma: Standard deviation of exploration noise
            q_max: Maximum quantity constraint
            
        Returns:
            Action (quantity)
        """
        mu = self.forward(state)
        mu = min(mu, q_max)
        
        if sigma > 0:
            action = np.random.normal(mu, sigma)
            action = np.clip(action, 0.0, q_max)
        else:
            action = mu
            
        return action
    
    def save(self, filepath: str):
        """Save policy parameters to file."""
        params = {
            'input_dim': self.input_dim,
            'w_linear': self.w_linear,
            'b_linear': self.b_linear,
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        np.savez(filepath, **params)
    
    def load(self, filepath: str):
        """Load policy parameters from file."""
        params = np.load(filepath)
        
        # Verify input dimension matches
        if params['input_dim'] != self.input_dim:
            raise ValueError(f"Saved policy has input_dim={params['input_dim']}, "
                           f"but current policy has input_dim={self.input_dim}")
        
        self.w_linear = params['w_linear']
        self.b_linear = params['b_linear']
        self.W1 = params['W1']
        self.b1 = params['b1']
        self.W2 = params['W2']
        self.b2 = params['b2']
        self.input_mean = params['input_mean']
        self.input_std = params['input_std']
        self._init_adam_params()


class PolicyNetwork_Nash:
    """
    Neural network policy with residual connection to prevent mode collapse.
    Architecture: output = linear(input) + NN(input)
    For Nash equilibrium games with 4-dimensional input.
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 1):
        """
        Initialize policy network.
        
        Args:
            input_dim: Dimension of input state
            hidden_dim: Number of hidden units
            output_dim: Dimension of output
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear baseline (ensures reasonable starting point)
        self.w_linear = np.array([0.33, -33.0, -0.33, +0.16])  # [w_a, w_b, w_c, w_d]
        self.b_linear = 0.0
        
        # Neural network correction (learns residuals from linear baseline)
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        
        limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)
        
        # Normalization statistics
        self.input_mean = np.array([100.0, 1.0, 27.5, 27.5])
        self.input_std = np.array([15.0, 0.15, 15.0, 15.0])
        
        # Adam optimizer parameters
        self._init_adam_params()
        
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.adam_t = 0
    
    def _init_adam_params(self):
        """Initialize Adam optimizer parameters."""
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
    
    def elu(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation: smooth, non-zero gradients everywhere."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def elu_derivative(self, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Derivative of ELU activation."""
        return np.where(x > 0, 1.0, alpha * np.exp(x))
    
    def forward(self, state: np.ndarray) -> float:
        """
        Forward pass: linear baseline + neural network correction.
        
        Args:
            state: Input state
            
        Returns:
            Output quantity
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
        output = np.maximum(0.1, output)
        
        return output
    
    def backward(self, state: np.ndarray, grad_output: float) -> Dict[str, np.ndarray]:
        """
        Backward pass for both linear and NN components.
        
        Args:
            state: Input state
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Dictionary of gradients for all parameters
        """
        # Gradient through max(0.1, output)
        if self.linear_output + self.z2[0] > 0.1:
            grad_combined = grad_output
        else:
            grad_combined = 0.0
        
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
    
    def adam_update(self, grads: Dict[str, np.ndarray], lr: float):
        """
        Update all parameters using Adam optimizer.
        
        Args:
            grads: Dictionary of gradients
            lr: Learning rate
        """
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
        
        # Update NN parameters
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
    
    def get_action(self, state: np.ndarray, sigma: float = 0.0, 
                   q_max: float = 100.0) -> float:
        """
        Get action with optional exploration noise.
        
        Args:
            state: Input state
            sigma: Standard deviation of exploration noise
            q_max: Maximum quantity constraint
            
        Returns:
            Action (quantity)
        """
        mu = self.forward(state)
        mu = min(mu, q_max)
        
        if sigma > 0:
            action = np.random.normal(mu, sigma)
            action = np.clip(action, 0.0, q_max)
        else:
            action = mu
            
        return action
    
    def save(self, filepath: str):
        """Save policy parameters to file."""
        params = {
            'w_linear': self.w_linear,
            'b_linear': self.b_linear,
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        np.savez(filepath, **params)
    
    def load(self, filepath: str):
        """Load policy parameters from file."""
        params = np.load(filepath)
        self.w_linear = params['w_linear']
        self.b_linear = params['b_linear']
        self.W1 = params['W1']
        self.b1 = params['b1']
        self.W2 = params['W2']
        self.b2 = params['b2']
        self.input_mean = params['input_mean']
        self.input_std = params['input_std']
        self._init_adam_params()