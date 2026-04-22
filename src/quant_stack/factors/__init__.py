"""Factor computation layer.

All functions take a wide-format close price DataFrame and return a
DataFrame of the same shape (DatetimeIndex × symbols).

Usage::

    from quant_stack.factors import momentum_21, sma_200, volatility_20

    close = repo.load_close(["SPY", "QQQ", "IEF"])
    mom   = momentum_21(close)
    trend = sma_200(close)
    vol   = volatility_20(close)
"""

from quant_stack.factors.momentum import momentum, momentum_21, momentum_63, momentum_126
from quant_stack.factors.trend import sma, sma_50, sma_200
from quant_stack.factors.volatility import realized_volatility, volatility_20

__all__ = [
    # Momentum
    "momentum",
    "momentum_21",
    "momentum_63",
    "momentum_126",
    # Trend
    "sma",
    "sma_50",
    "sma_200",
    # Volatility
    "realized_volatility",
    "volatility_20",
]
