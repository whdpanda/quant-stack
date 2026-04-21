"""Experiment orchestrator — runs a full research pipeline end-to-end."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from quant_stack.core.schemas import BacktestConfig, PortfolioConfig
from quant_stack.data.base import DataProvider
from quant_stack.data.transforms import simple_returns
from quant_stack.research.backtest import run_backtest
from quant_stack.research.base import Strategy


@dataclass
class ExperimentResult:
    backtest_config: BacktestConfig
    backtest_result: "quant_stack.core.schemas.BacktestResult | None" = None  # type: ignore[name-defined]
    portfolio_weights: "quant_stack.core.schemas.PortfolioWeights | None" = None  # type: ignore[name-defined]
    agent_analysis: str = ""
    report_path: Path | None = None
    errors: list[str] = field(default_factory=list)


class Orchestrator:
    """Coordinates data → backtest → portfolio → agent → report pipeline.

    Design intent: each step is optional and independently skippable.
    The orchestrator never raises — it logs errors and stores them in
    ExperimentResult.errors so the caller decides how to proceed.
    """

    def __init__(
        self,
        provider: DataProvider,
        strategy: Strategy,
        run_portfolio: bool = True,
        run_agent: bool = True,
        reports_dir: str = "./reports",
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.run_portfolio = run_portfolio
        self.run_agent = run_agent
        self.reports_dir = reports_dir

    def run(
        self,
        backtest_config: BacktestConfig,
        portfolio_config: PortfolioConfig | None = None,
    ) -> ExperimentResult:
        result = ExperimentResult(backtest_config=backtest_config)

        # 1. Fetch data
        try:
            raw = self.provider.fetch(backtest_config.data)
            close = raw["close"] if isinstance(raw.columns, type(raw.columns)) and "close" in raw.columns.get_level_values(0) else raw
        except Exception as exc:
            result.errors.append(f"data fetch failed: {exc}")
            logger.error(f"Orchestrator: data fetch failed — {exc}")
            return result

        # 2. Backtest
        try:
            result.backtest_result = run_backtest(self.strategy, close, backtest_config)
        except Exception as exc:
            result.errors.append(f"backtest failed: {exc}")
            logger.error(f"Orchestrator: backtest failed — {exc}")

        # 3. Portfolio optimisation
        if self.run_portfolio and portfolio_config and result.backtest_result:
            try:
                from quant_stack.portfolio.optimizer import optimize_portfolio
                returns = simple_returns(close)
                result.portfolio_weights = optimize_portfolio(returns, portfolio_config)
            except Exception as exc:
                result.errors.append(f"portfolio optimisation failed: {exc}")
                logger.warning(f"Orchestrator: portfolio step failed — {exc}")

        # 4. Agent analysis
        if self.run_agent and result.backtest_result:
            try:
                from quant_stack.agent.researcher import Researcher
                researcher = Researcher()
                result.agent_analysis = researcher.analyse_backtest(result.backtest_result)
            except Exception as exc:
                result.errors.append(f"agent analysis failed: {exc}")
                logger.warning(f"Orchestrator: agent step failed — {exc}")

        # 5. Report
        if result.backtest_result:
            try:
                from quant_stack.agent.reporter import Reporter
                reporter = Reporter(self.reports_dir)
                result.report_path = reporter.generate(
                    backtest=result.backtest_result,
                    weights=result.portfolio_weights,
                    analysis=result.agent_analysis,
                )
            except Exception as exc:
                result.errors.append(f"report generation failed: {exc}")
                logger.warning(f"Orchestrator: report step failed — {exc}")

        return result
