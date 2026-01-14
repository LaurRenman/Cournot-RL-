"""
Payoff (profit) functions.
"""

def profit(price: float, quantity: float, marginal_cost: float) -> float:
    """
    Firm profit.

    π = (P - c) * q

    Parameters
    ----------
    price : float
        Market price
    quantity : float
        Firm output
    marginal_cost : float
        Firm marginal cost

    Returns
    -------
    float
        Profit
    """
    return (price - marginal_cost) * quantity
