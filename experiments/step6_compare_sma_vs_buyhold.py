from datetime import date

import pandas as pd

from quant_stack.core.logging import setup_logging
from quant_stack.core.schemas import BacktestConfig, DataConfig
from quant_stack.data.providers.yahoo import YahooProvider
from quant_stack.research.base import Strategy
from quant_stack.research.backtest import run_backtest
from quant_stack.research.strategies.sma_cross import SmaCrossStrategy

setup_logging()


class BuyAndHoldStrategy(Strategy):
    name = "buy_and_hold"

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        # 从第一天开始一直持有
        return pd.DataFrame(1.0, index=close.index, columns=close.columns)


def run_one(symbol: str) -> list[dict]:
    data_cfg = DataConfig(
        symbols=[symbol],
        start=date(2018, 1, 1),
        end=date(2023, 12, 31),
    )

    provider = YahooProvider()
    close = provider.fetch_close(data_cfg)

    bt_cfg = BacktestConfig(
        data=data_cfg,
        strategy_name="comparison",
        initial_cash=100_000,
        commission=0.001,
        slippage=0.001,
        freq="1D",
    )

    sma_strategy = SmaCrossStrategy(fast_window=20, slow_window=50)
    buy_hold_strategy = BuyAndHoldStrategy()

    sma_result = run_backtest(sma_strategy, close, bt_cfg)
    buy_hold_result = run_backtest(buy_hold_strategy, close, bt_cfg)

    return [
        {
            "symbol": symbol,
            "strategy": "SMA 20/50",
            "total_return": sma_result.total_return,
            "cagr": sma_result.cagr,
            "sharpe": sma_result.sharpe_ratio,
            "max_drawdown": sma_result.max_drawdown,
            "trades": sma_result.n_trades,
        },
        {
            "symbol": symbol,
            "strategy": "Buy & Hold",
            "total_return": buy_hold_result.total_return,
            "cagr": buy_hold_result.cagr,
            "sharpe": buy_hold_result.sharpe_ratio,
            "max_drawdown": buy_hold_result.max_drawdown,
            "trades": buy_hold_result.n_trades,
        },
    ]


def main() -> None:
    rows = []
    for symbol in ["SPY", "QQQ", "IEF"]:
        rows.extend(run_one(symbol))

    df = pd.DataFrame(rows)

    print("\n=== SMA 20/50 vs Buy & Hold ===")
    print(df.to_string(index=False))

    print("\n=== Pretty View ===")
    for _, row in df.iterrows():
        print(
            f"{row['symbol']:>4} | {row['strategy']:<10} | "
            f"TR={row['total_return']:.2%} | "
            f"CAGR={row['cagr']:.2%} | "
            f"Sharpe={row['sharpe']:.3f} | "
            f"MDD={row['max_drawdown']:.2%} | "
            f"Trades={row['trades']}"
        )


if __name__ == "__main__":
    main()