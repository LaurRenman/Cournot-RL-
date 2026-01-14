"""
Analytical equilibrium results for the Cournot model.
"""

import numpy as np


def cournot_nash_quantity(a: float, b: float, c: float, n_firms: int) -> float:
    """
    Symmetric Cournot–Nash equilibrium quantity per firm.

    q* = (a - c) / [b (N + 1)]

    Parameters
    ----------
    a : float
        Demand intercept
    b : float
        Demand slope
    c : float
        Marginal cost (assumed identical)
    n_firms : int
        Number of firms

    Returns
    -------
    float
        Equilibrium quantity per firm
    """
    q = (a - c) / (b * (n_firms + 1))
    return max(q, 0.0)


def cournot_nash_profile(a: float, b: float, c: float, n_firms: int) -> np.ndarray:
    """
    Equilibrium quantity vector.

    Returns
    -------
    np.ndarray
        Quantity for each firm
    """
    q_star = cournot_nash_quantity(a, b, c, n_firms)
    return np.full(n_firms, q_star)


def cournot_nash_price(a: float, b: float, c: float, n_firms: int) -> float:
    """
    Equilibrium market price.
    """
    q = cournot_nash_quantity(a, b, c, n_firms)
    total_quantity = n_firms * q
    return max(a - b * total_quantity, 0.0)
