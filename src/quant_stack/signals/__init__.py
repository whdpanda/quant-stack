"""Signal generation layer.

Signal generators consume factor values and produce SignalFrames.
They are decoupled from vectorbt, LEAN, and the execution layer.

Usage::

    from quant_stack.signals import (
        absolute_momentum_signal,
        relative_momentum_ranking_signal,
        trend_filter_signal,
        as_eligibility_mask,
        SignalFrame,
    )
"""

from quant_stack.signals.base import SignalFrame
from quant_stack.signals.momentum import (
    absolute_momentum_signal,
    relative_momentum_ranking_signal,
)
from quant_stack.signals.trend import as_eligibility_mask, trend_filter_signal

__all__ = [
    "SignalFrame",
    "absolute_momentum_signal",
    "relative_momentum_ranking_signal",
    "trend_filter_signal",
    "as_eligibility_mask",
]
