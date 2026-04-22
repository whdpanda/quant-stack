from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_stack.agent.experiment_tracker import load_records


def pick_latest_baseline(records) -> pd.DataFrame:
    rows: list[dict] = []

    for record in records:
        backtest = record.backtest_result
        if backtest is None:
            continue

        tags = ",".join(record.tags) if record.tags else ""
        if "baseline" not in tags:
            continue
        if "smoke_test" in tags:
            continue

        rows.append(
            {
                "experiment_id": record.experiment_id,
                "created_at": record.created_at,
                "symbol": ",".join(record.symbols) if record.symbols else "",
                "strategy_name": backtest.strategy_name,
                "description": record.description,
                "total_return": backtest.total_return,
                "cagr": backtest.cagr,
                "sharpe": backtest.sharpe_ratio,
                "max_drawdown": backtest.max_drawdown,
                "trades": backtest.n_trades,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No baseline experiment records found")

    df = df.sort_values(by="created_at")
    df = df.groupby(["symbol", "strategy_name"], as_index=False).tail(1)
    df = df.sort_values(by=["symbol", "strategy_name"]).reset_index(drop=True)
    return df


def format_pct(x: float) -> str:
    return f"{x:.2%}"


def build_conclusion(symbol: str, sma_row: pd.Series, bh_row: pd.Series) -> str:
    parts: list[str] = []

    if sma_row["total_return"] > bh_row["total_return"]:
        parts.append("SMA 20/50 在总收益上优于 Buy & Hold")
    else:
        parts.append("Buy & Hold 在总收益上优于 SMA 20/50")

    if sma_row["sharpe"] > bh_row["sharpe"]:
        parts.append("SMA 20/50 的风险调整后收益更好")
    else:
        parts.append("Buy & Hold 的风险调整后收益更好")

    if sma_row["max_drawdown"] < bh_row["max_drawdown"]:
        parts.append("SMA 20/50 的最大回撤更低")
    else:
        parts.append("Buy & Hold 的最大回撤更低")

    if sma_row["trades"] > bh_row["trades"]:
        parts.append("SMA 20/50 的交易次数明显更多")
    else:
        parts.append("Buy & Hold 的交易次数更少")

    return f"- **{symbol}**： " + "；".join(parts) + "。"


def main() -> None:
    records = load_records("experiments/records")
    df = pick_latest_baseline(records)

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

    # 生成逐资产结论
    conclusions: list[str] = []
    for symbol in sorted(summary_df["symbol"].unique()):
        sub = summary_df[summary_df["symbol"] == symbol].copy()
        sma = sub[sub["strategy_name"] == "sma_cross"]
        bh = sub[sub["strategy_name"] == "buy_and_hold"]

        if sma.empty or bh.empty:
            continue

        conclusions.append(build_conclusion(symbol, sma.iloc[0], bh.iloc[0]))

    # 写 markdown
    output_path = Path("reports/baseline_summary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Baseline Research Summary",
        "",
        "## Scope",
        "",
        "- Assets: SPY / QQQ / IEF",
        "- Period: 2018-01-01 to 2023-12-31",
        "- Strategies compared:",
        "  - SMA 20/50 (`sma_cross`)",
        "  - Buy & Hold (`buy_and_hold`)",
        "- Purpose: establish a baseline benchmark for future strategy research",
        "",
        "## Latest Baseline Results",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
        "## Per-Asset Conclusions",
        "",
        *conclusions,
        "",
        "## Overall Conclusion",
        "",
        "- 当前 baseline 表明，趋势策略 **并不天然优于** Buy & Hold。",
        "- 在 `SPY` 和 `QQQ` 上，Buy & Hold 在本样本区间内总体更强。",
        "- 在 `IEF` 上，SMA 20/50 明显优于 Buy & Hold，说明趋势过滤在该类资产上更有价值。",
        "- 因此，后续新策略研究不应只追求收益提升，也应关注不同资产上的适配性与回撤控制能力。",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Baseline summary written to: {output_path}")


if __name__ == "__main__":
    main()