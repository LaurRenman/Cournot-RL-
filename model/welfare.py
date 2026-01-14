"""
Welfare and performance measures.
"""

def consumer_surplus(price: float, total_quantity: float, a: float) -> float:
    """
    Consumer surplus for linear demand.

    CS = 0.5 * Q * (a - P)

    Parameters
    ----------
    price : float
        Market price
    total_quantity : float
        Aggregate output
    a : float
        Demand intercept

    Returns
    -------
    float
        Consumer surplus
    """
    return 0.5 * total_quantity * (a - price)


def producer_surplus(profits: list | tuple) -> float:
    """
    Producer surplus (sum of firm profits).
    """
    return sum(profits)


def total_surplus(consumer_surplus: float, producer_surplus: float) -> float:
    """
    Total welfare.
    """
    return consumer_surplus + producer_surplus


def lerner_index(price: float, marginal_cost: float) -> float:
    """
    Lerner index: (P - c) / P.
    """
    if price <= 0:
        return 0.0
    return (price - marginal_cost) / price
