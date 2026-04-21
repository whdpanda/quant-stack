"""PyPortfolioOpt wrapper for mean-variance portfolio optimization."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from quant_stack.core.exceptions import PortfolioOptimizationError
from quant_stack.core.schemas import PortfolioConfig, PortfolioMethod, PortfolioWeights


def optimize_portfolio(
    returns: pd.DataFrame,
    config: PortfolioConfig,
) -> PortfolioWeights:
    """Compute optimal portfolio weights.

    Args:
        returns: Daily simple returns, columns = symbol names.
        config: PortfolioConfig specifying method and constraints.

    Returns:
        PortfolioWeights with allocation per symbol and expected metrics.
    """
    try:
        from pypfopt import expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier
    except ImportError as e:
        raise PortfolioOptimizationError(
            "PyPortfolioOpt is not installed: pip install 'quant-stack[portfolio]'"
        ) from e

    mu = expected_returns.mean_historical_return(returns, returns_data=True)
    S = risk_models.sample_cov(returns, returns_data=True)

    ef = EfficientFrontier(mu, S, weight_bounds=config.weight_bounds)

    logger.info(f"Optimising portfolio: method={config.method}")

    match config.method:
        case PortfolioMethod.MAX_SHARPE:
            ef.max_sharpe(risk_free_rate=config.risk_free_rate)
        case PortfolioMethod.MIN_VOLATILITY:
            ef.min_volatility()
        case PortfolioMethod.EFFICIENT_RISK:
            target = config.target_volatility
            if target is None:
                raise PortfolioOptimizationError(
                    "target_volatility must be set when method=efficient_risk"
                )
            ef.efficient_risk(target_volatility=target)
        case _:
            raise PortfolioOptimizationError(f"Unknown method: {config.method}")

    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=config.risk_free_rate)
    exp_return, exp_vol, sharpe = perf

    logger.info(
        f"Portfolio optimised — E[r]={exp_return:.2%}, σ={exp_vol:.2%}, Sharpe={sharpe:.3f}"
    )

    from datetime import date
    return PortfolioWeights(
        weights={k: v for k, v in cleaned.items() if v > 1e-6},
        method=config.method,
        rebalance_date=date.today(),
        expected_return=exp_return,
        expected_volatility=exp_vol,
        sharpe_ratio=sharpe,
        optimization_metadata={
            "risk_free_rate": config.risk_free_rate,
            "weight_bounds": list(config.weight_bounds),
        },
    )
