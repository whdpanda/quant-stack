"""Custom exceptions for quant-stack."""


class QuantStackError(Exception):
    """Base exception for all quant-stack errors."""


class DataProviderError(QuantStackError):
    """Raised when a data provider fails to fetch or validate data."""


class BacktestError(QuantStackError):
    """Raised when a backtest cannot be run."""


class PortfolioOptimizationError(QuantStackError):
    """Raised when portfolio optimization fails or produces infeasible results."""


class ExecutionError(QuantStackError):
    """Raised when order execution or broker communication fails."""


class AgentError(QuantStackError):
    """Raised when the AI agent encounters an unrecoverable error."""


class ConfigError(QuantStackError):
    """Raised when configuration is missing or invalid."""
