"""Execution layer public API.

Primary path (new):
    from quant_stack.execution.domain import (
        PositionSnapshot, TargetWeights, OrderPlan, ExecutionResult,
        target_weights_from_portfolio_weights,
    )
    from quant_stack.execution.service import RebalanceService
    from quant_stack.execution.adapters import (
        DryRunExecutionAdapter, PaperExecutionAdapter, LeanExecutionAdapter,
    )

Legacy stubs (kept for backward compatibility):
    from quant_stack.execution import Executor, LeanBridge
"""

from quant_stack.execution.adapters import (
    DryRunExecutionAdapter,
    LeanExecutionAdapter,
    PaperExecutionAdapter,
)
from quant_stack.execution.base import Executor
from quant_stack.execution.domain import (
    ExecutionResult,
    OrderIntent,
    OrderPlan,
    PositionDiff,
    PositionSnapshot,
    RebalanceDecision,
    RiskCheckResult,
    TargetWeights,
    target_weights_from_portfolio_weights,
)
from quant_stack.execution.lean_bridge import LeanBridge
from quant_stack.execution.service import RebalanceService, check_order_plan

__all__ = [
    # new execution layer
    "DryRunExecutionAdapter",
    "ExecutionResult",
    "LeanExecutionAdapter",
    "OrderIntent",
    "OrderPlan",
    "PaperExecutionAdapter",
    "PositionDiff",
    "PositionSnapshot",
    "RebalanceDecision",
    "RebalanceService",
    "RiskCheckResult",
    "TargetWeights",
    "check_order_plan",
    "target_weights_from_portfolio_weights",
    # legacy stubs
    "Executor",
    "LeanBridge",
]
