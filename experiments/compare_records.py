from __future__ import annotations

import pandas as pd

from quant_stack.agent.experiment_tracker import load_records


def main() -> None:
    records = load_records("experiments/records")
    if not records:
        raise RuntimeError("No experiment records found in experiments/records")

    rows: list[dict] = []

    for record in records:
        backtest = record.backtest_result
        if backtest is None:
            continue

        rows.append(
            {
                "experiment_id": record.experiment_id,
                "created_at": record.created_at,
                "symbol": ",".join(record.symbols) if record.symbols else "",
                "strategy_name": backtest.strategy_name,
                "description": record.description,
                "tags": ",".join(record.tags) if record.tags else "",
                "total_return": backtest.total_return,
                "cagr": backtest.cagr,
                "sharpe": backtest.sharpe_ratio,
                "max_drawdown": backtest.max_drawdown,
                "trades": backtest.n_trades,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No valid backtest records found")

    # 1. 只保留 baseline 实验
    df = df[df["tags"].str.contains("baseline", na=False)].copy()

    # 2. 去掉 smoke_test
    df = df[~df["tags"].str.contains("smoke_test", na=False)].copy()

    if df.empty:
        raise RuntimeError("No baseline experiment records found")

    # 3. 同一个 symbol + strategy_name 只保留最新的一条
    df = df.sort_values(by="created_at")
    df = df.groupby(["symbol", "strategy_name"], as_index=False).tail(1)

    # 4. 排序
    df = df.sort_values(by=["symbol", "strategy_name"]).reset_index(drop=True)

    # 5. 主比较表只保留最关键列
    summary_df = df[
        [
            "symbol",
            "strategy_name",
            "total_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "trades",
        ]
    ].copy()

    print("\n=== Experiment Comparison ===")
    print(summary_df.to_string(index=False))

    output_path = "reports/experiment_comparison.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Comparison\n\n")
        f.write("## Latest Baseline Results\n\n")
        f.write("```text\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n```\n")

    print(f"\nComparison report written to: {output_path}")


if __name__ == "__main__":
    main()