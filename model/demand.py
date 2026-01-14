"""
Demand functions for the Cournot model.
"""

def inverse_demand(total_quantity: float, a: float, b: float) -> float:
    """
    Linear inverse demand function.

    P(Q) = a - bQ, with non-negative prices.

    Parameters
    ----------
    total_quantity : float
        Aggregate output Q
    a : float
        Demand intercept
    b : float
        Demand slope

    Returns
    -------
    float
        Market price
    """
    return max(a - b * total_quantity, 0.0)
