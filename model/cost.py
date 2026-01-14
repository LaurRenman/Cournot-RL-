"""
Cost functions for firms.
"""

def total_cost(quantity: float, marginal_cost: float) -> float:
    """
    Total production cost with constant marginal cost.

    C(q) = c * q

    Parameters
    ----------
    quantity : float
        Firm output
    marginal_cost : float
        Constant marginal cost

    Returns
    -------
    float
        Total cost
    """
    return marginal_cost * quantity
