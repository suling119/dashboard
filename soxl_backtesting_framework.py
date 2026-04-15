#!/usr/bin/env python3
"""
SOXL Daily-Close Backtesting Framework

Implements four long-only strategies using DAILY CLOSE values only for signal logic:
1) Price Single MA
2) Price Dual MA
3) RS Single MA (RS = SOXL Close / SPY Close)
4) RS Dual MA

Execution rule:
- Signal is generated after day t close.
- Trade executes at day t+1 close.

No intraday/high/low/open values are used for signal detection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Configuration dataclasses
# -----------------------------
@dataclass
class StrategyConfig:
    name: str
    signal_left_col: str
    signal_right_col: str
    indicator_cols: List[str]
    chart_title: str
    chart_ylabel: str


@dataclass
class BacktestConfig:
    initial_capital: float
    commission_rate: float


# -----------------------------
# Data download
# -----------------------------
def _download_single_close_with_retry(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 5,
) -> pd.Series:
    """
    Download one ticker's daily close with retries.
    Retries help with transient yfinance/database lock/network issues.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if raw.empty:
                raise ValueError(f"{ticker}: empty data from yfinance.")
            if "Close" not in raw.columns:
                raise ValueError(f"{ticker}: missing Close column in yfinance response.")

            close = raw["Close"].copy()
            if isinstance(close, pd.DataFrame):
                # Some yfinance versions can still return DataFrame for one ticker.
                close = close.iloc[:, 0]
            close = close.rename(f"{ticker}_Close").dropna().sort_index()

            if close.empty:
                raise ValueError(f"{ticker}: close series is empty after cleaning.")

            return close

        except Exception as exc:  # broad catch to handle transient OperationalError/HTTP issues
            last_error = exc
            if attempt < max_retries:
                sleep_sec = 2 ** attempt
                print(
                    f"[WARN] Download attempt {attempt}/{max_retries} for {ticker} failed: {exc}. "
                    f"Retrying in {sleep_sec}s..."
                )
                time.sleep(sleep_sec)
            else:
                break

    raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts: {last_error}")


def download_daily_close_data(start_date: str, end_date: str, max_retries: int = 5) -> pd.DataFrame:
    """
    Download daily SOXL and SPY data, then keep CLOSE prices only.
    """
    soxl_close = _download_single_close_with_retry(
        ticker="SOXL",
        start_date=start_date,
        end_date=end_date,
        max_retries=max_retries,
    )
    spy_close = _download_single_close_with_retry(
        ticker="SPY",
        start_date=start_date,
        end_date=end_date,
        max_retries=max_retries,
    )

    df = pd.concat([soxl_close, spy_close], axis=1, join="inner").dropna().sort_index()

    if len(df) < 50:
        raise ValueError("Not enough daily bars after cleaning. Expand date range.")

    return df


# -----------------------------
# Indicator calculation
# -----------------------------
def add_indicators(
    df: pd.DataFrame,
    price_single_ma: int,
    price_dual_fast: int,
    price_dual_slow: int,
    rs_single_ma: int,
    rs_dual_fast: int,
    rs_dual_slow: int,
) -> pd.DataFrame:
    """
    Compute all indicators from daily close values only.
    """
    out = df.copy()
    out["RS"] = out["SOXL_Close"] / out["SPY_Close"]  # RS uses close-only rule

    # Price MAs (close-based)
    out["Price_MA_Single"] = out["SOXL_Close"].rolling(price_single_ma, min_periods=price_single_ma).mean()
    out["Price_MA_Fast"] = out["SOXL_Close"].rolling(price_dual_fast, min_periods=price_dual_fast).mean()
    out["Price_MA_Slow"] = out["SOXL_Close"].rolling(price_dual_slow, min_periods=price_dual_slow).mean()

    # RS MAs (RS line is close-based, then MAs on RS)
    out["RS_MA_Single"] = out["RS"].rolling(rs_single_ma, min_periods=rs_single_ma).mean()
    out["RS_MA_Fast"] = out["RS"].rolling(rs_dual_fast, min_periods=rs_dual_fast).mean()
    out["RS_MA_Slow"] = out["RS"].rolling(rs_dual_slow, min_periods=rs_dual_slow).mean()

    return out


# -----------------------------
# Signal generation
# -----------------------------
def crossover(series_left: pd.Series, series_right: pd.Series) -> pd.Series:
    """
    True when left crosses ABOVE right using prior and current CLOSE-derived values:
      left[t-1] <= right[t-1]  and  left[t] > right[t]
    """
    return (series_left.shift(1) <= series_right.shift(1)) & (series_left > series_right)


def crossunder(series_left: pd.Series, series_right: pd.Series) -> pd.Series:
    """
    True when left crosses BELOW right using prior and current CLOSE-derived values:
      left[t-1] >= right[t-1]  and  left[t] < right[t]
    """
    return (series_left.shift(1) >= series_right.shift(1)) & (series_left < series_right)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all strategy signals from close-derived series only.
    """
    out = df.copy()

    # 1) Price Single MA
    out["PriceSingle_Buy"] = crossover(out["SOXL_Close"], out["Price_MA_Single"])
    out["PriceSingle_Sell"] = crossunder(out["SOXL_Close"], out["Price_MA_Single"])
    out["PriceSingle_Event"] = np.select(
        [out["PriceSingle_Buy"], out["PriceSingle_Sell"]],
        [1, -1],
        default=0,
    )

    # 2) Price Dual MA
    out["PriceDual_Buy"] = crossover(out["Price_MA_Fast"], out["Price_MA_Slow"])
    out["PriceDual_Sell"] = crossunder(out["Price_MA_Fast"], out["Price_MA_Slow"])
    out["PriceDual_Event"] = np.select(
        [out["PriceDual_Buy"], out["PriceDual_Sell"]],
        [1, -1],
        default=0,
    )

    # 3) RS Single MA
    out["RSSingle_Buy"] = crossover(out["RS"], out["RS_MA_Single"])
    out["RSSingle_Sell"] = crossunder(out["RS"], out["RS_MA_Single"])
    out["RSSingle_Event"] = np.select(
        [out["RSSingle_Buy"], out["RSSingle_Sell"]],
        [1, -1],
        default=0,
    )

    # 4) RS Dual MA
    out["RSDual_Buy"] = crossover(out["RS_MA_Fast"], out["RS_MA_Slow"])
    out["RSDual_Sell"] = crossunder(out["RS_MA_Fast"], out["RS_MA_Slow"])
    out["RSDual_Event"] = np.select(
        [out["RSDual_Buy"], out["RSDual_Sell"]],
        [1, -1],
        default=0,
    )

    return out


# -----------------------------
# Backtest engine (next-day execution)
# -----------------------------
def backtest_long_only_next_close(
    df: pd.DataFrame,
    event_col: str,
    strategy_name: str,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Backtest rule:
    - Signals are produced from daily close values at date t.
    - Orders execute at date t+1 close.
    - Long-only, full position on buy, flat on sell.
    - Commission charged on each execution.
    """
    work = df.copy()
    work["Equity"] = np.nan
    work["Cash"] = np.nan
    work["Shares"] = np.nan
    work["Position"] = 0

    cash = float(cfg.initial_capital)
    shares = 0.0
    pending_order: Optional[dict] = None
    open_trade: Optional[dict] = None
    trade_log: List[dict] = []

    idx = work.index
    close = work["SOXL_Close"]

    for i, current_date in enumerate(idx):
        current_close = float(close.iloc[i])

        # 1) Execute pending order at today's close (next-day execution after signal)
        if pending_order is not None and pending_order["execute_on"] == current_date:
            side = pending_order["side"]
            signal_date = pending_order["signal_date"]

            if side == "BUY" and shares == 0:
                commission = cash * cfg.commission_rate
                investable = cash - commission
                if investable > 0:
                    shares = investable / current_close
                    cash = 0.0
                    open_trade = {
                        "Strategy": strategy_name,
                        "EntrySignalDate": signal_date,
                        "EntryDate": current_date,
                        "EntryPrice": current_close,
                        "EntryCommission": commission,
                        "EntryShares": shares,
                    }
            elif side == "SELL" and shares > 0:
                gross_proceeds = shares * current_close
                commission = gross_proceeds * cfg.commission_rate
                net_proceeds = gross_proceeds - commission
                cash = net_proceeds

                if open_trade is None:
                    raise RuntimeError("Sell executed without an open trade record.")

                gross_pnl = (current_close - open_trade["EntryPrice"]) * open_trade["EntryShares"]
                net_pnl = gross_pnl - open_trade["EntryCommission"] - commission
                return_pct = net_pnl / (open_trade["EntryPrice"] * open_trade["EntryShares"])
                holding_days = (current_date - open_trade["EntryDate"]).days

                trade_log.append(
                    {
                        **open_trade,
                        "ExitSignalDate": signal_date,
                        "ExitDate": current_date,
                        "ExitPrice": current_close,
                        "ExitCommission": commission,
                        "GrossPnL": gross_pnl,
                        "NetPnL": net_pnl,
                        "ReturnPct": return_pct,
                        "HoldingDays": holding_days,
                    }
                )

                shares = 0.0
                open_trade = None

            pending_order = None

        # 2) Mark-to-market equity after possible execution
        equity = cash + shares * current_close
        work.at[current_date, "Cash"] = cash
        work.at[current_date, "Shares"] = shares
        work.at[current_date, "Equity"] = equity
        work.at[current_date, "Position"] = 1 if shares > 0 else 0

        # 3) Evaluate today's signal and schedule execution for next day close
        event = int(work.at[current_date, event_col])
        has_next_bar = i < (len(idx) - 1)
        if pending_order is None and has_next_bar:
            if event == 1 and shares == 0:
                pending_order = {
                    "side": "BUY",
                    "signal_date": current_date,
                    "execute_on": idx[i + 1],
                }
            elif event == -1 and shares > 0:
                pending_order = {
                    "side": "SELL",
                    "signal_date": current_date,
                    "execute_on": idx[i + 1],
                }

    trade_df = pd.DataFrame(trade_log)
    metrics = compute_metrics(work["Equity"], trade_df, cfg.initial_capital)
    return work, trade_df, metrics


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    equity = equity_curve.dropna().astype(float)
    if equity.empty:
        raise ValueError("Equity curve is empty.")

    total_return = equity.iloc[-1] / initial_capital - 1
    annualized_return = (equity.iloc[-1] / initial_capital) ** (252 / len(equity)) - 1

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()

    daily_returns = equity.pct_change().fillna(0.0)
    if daily_returns.std(ddof=0) > 0:
        sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std(ddof=0))
    else:
        sharpe = np.nan

    num_trades = int(len(trades))
    win_rate = float((trades["NetPnL"] > 0).mean()) if num_trades > 0 else np.nan

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Number of Trades": float(num_trades),
        "Win Rate": win_rate,
    }


# -----------------------------
# Validation / debug
# -----------------------------
def validate_crossover_logic(
    df: pd.DataFrame,
    strategy_name: str,
    left_col: str,
    right_col: str,
    buy_col: str,
    sell_col: str,
) -> None:
    """
    Confirms crossover/crossunder columns are exactly close-to-close definitions.
    """
    expected_buy = crossover(df[left_col], df[right_col]).fillna(False)
    expected_sell = crossunder(df[left_col], df[right_col]).fillna(False)
    actual_buy = df[buy_col].fillna(False)
    actual_sell = df[sell_col].fillna(False)

    if not actual_buy.equals(expected_buy):
        mismatches = int((actual_buy != expected_buy).sum())
        raise AssertionError(f"{strategy_name}: Buy crossover mismatch count={mismatches}")

    if not actual_sell.equals(expected_sell):
        mismatches = int((actual_sell != expected_sell).sum())
        raise AssertionError(f"{strategy_name}: Sell crossunder mismatch count={mismatches}")

    overlap = int((actual_buy & actual_sell).sum())
    if overlap > 0:
        raise AssertionError(f"{strategy_name}: Found {overlap} bars with simultaneous buy and sell.")

    print(f"[VALIDATION] {strategy_name}: crossover/crossunder logic OK (close-to-close).")


def validate_next_day_execution(trades: pd.DataFrame, full_index: pd.DatetimeIndex, strategy_name: str) -> None:
    """
    Confirms entry/exit executions happen on the next trading day after signal date.
    """
    if trades.empty:
        print(f"[VALIDATION] {strategy_name}: no completed trades to verify next-day execution.")
        return

    index_map = {d: i for i, d in enumerate(full_index)}
    for _, row in trades.iterrows():
        entry_sig = row["EntrySignalDate"]
        entry_exec = row["EntryDate"]
        exit_sig = row["ExitSignalDate"]
        exit_exec = row["ExitDate"]

        if pd.notna(entry_sig):
            if index_map[entry_exec] != index_map[entry_sig] + 1:
                raise AssertionError(
                    f"{strategy_name}: Entry execution is not next trading day ({entry_sig} -> {entry_exec})"
                )
        if pd.notna(exit_sig):
            if index_map[exit_exec] != index_map[exit_sig] + 1:
                raise AssertionError(
                    f"{strategy_name}: Exit execution is not next trading day ({exit_sig} -> {exit_exec})"
                )

    print(f"[VALIDATION] {strategy_name}: next-day execution timing OK.")


def print_signal_samples(
    df: pd.DataFrame,
    strategy_name: str,
    left_col: str,
    right_col: str,
    buy_col: str,
    sell_col: str,
    max_samples_each_side: int = 3,
) -> None:
    """
    Print sample rows around buy/sell dates to visually verify close-to-close signal logic.
    """
    print(f"\n[DEBUG] Sample rows around {strategy_name} buy/sell signals")
    print("Rule: buy if left[t-1] <= right[t-1] and left[t] > right[t];")
    print("      sell if left[t-1] >= right[t-1] and left[t] < right[t].")

    temp = df[[left_col, right_col, buy_col, sell_col]].copy()
    temp["Left_prev"] = temp[left_col].shift(1)
    temp["Right_prev"] = temp[right_col].shift(1)

    def _print_samples(signal_col: str, label: str) -> None:
        signal_dates = temp.index[temp[signal_col].fillna(False)]
        if len(signal_dates) == 0:
            print(f"  - No {label} signals.")
            return
        sample_dates = signal_dates[:max_samples_each_side]
        for d in sample_dates:
            loc = temp.index.get_loc(d)
            start = max(0, loc - 2)
            end = min(len(temp), loc + 2)
            snippet = temp.iloc[start:end][
                [left_col, right_col, "Left_prev", "Right_prev", buy_col, sell_col]
            ]
            print(f"\n  {label} sample around {d.date()}:")
            print(snippet.to_string())

    _print_samples(buy_col, "BUY")
    _print_samples(sell_col, "SELL")


def validate_trade_log_plot_alignment(
    trades: pd.DataFrame,
    bt_df: pd.DataFrame,
    strategy_name: str,
) -> None:
    """
    Confirms plotted buy/sell markers are sourced from trade log entries/exits.
    """
    if trades.empty:
        print(f"[VALIDATION] {strategy_name}: no completed trades, no buy/sell plot markers expected.")
        return

    required_cols = {"EntryDate", "ExitDate"}
    if not required_cols.issubset(set(trades.columns)):
        raise AssertionError(f"{strategy_name}: trade log missing required date columns for marker plotting.")

    entry_dates = pd.DatetimeIndex(trades["EntryDate"].dropna().unique()).sort_values()
    exit_dates = pd.DatetimeIndex(trades["ExitDate"].dropna().unique()).sort_values()

    # Marker Y-values are fetched from bt_df during plotting, so all marker dates must exist in bt_df index.
    missing_entries = entry_dates.difference(bt_df.index)
    missing_exits = exit_dates.difference(bt_df.index)
    if len(missing_entries) > 0 or len(missing_exits) > 0:
        raise AssertionError(
            f"{strategy_name}: trade log contains marker dates not found in backtest index. "
            f"Missing entries={list(missing_entries)}, missing exits={list(missing_exits)}"
        )

    if trades["EntryDate"].isna().any() or trades["ExitDate"].isna().any():
        raise AssertionError(f"{strategy_name}: trade log has null EntryDate/ExitDate values.")

    print(
        f"[VALIDATION] {strategy_name}: trade log and plotted markers aligned "
        f"(entries={len(trades)}, exits={len(trades)})."
    )


# -----------------------------
# Visualization
# -----------------------------
def plot_strategy(
    df: pd.DataFrame,
    bt_df: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top panel: strategy line(s) + executed trade markers
    if strategy_cfg.name in {"Price Single MA", "Price Dual MA"}:
        ax_top.plot(df.index, df["SOXL_Close"], label="SOXL Close", linewidth=1.3)
    else:
        ax_top.plot(df.index, df["RS"], label="RS (SOXL_Close / SPY_Close)", linewidth=1.3)

    for col in strategy_cfg.indicator_cols:
        ax_top.plot(df.index, df[col], label=col, linewidth=1.2)

    if not trades.empty:
        ax_top.scatter(
            trades["EntryDate"],
            bt_df.loc[trades["EntryDate"], strategy_cfg.signal_left_col],
            marker="^",
            s=90,
            label="Buy (executed)",
        )
        ax_top.scatter(
            trades["ExitDate"],
            bt_df.loc[trades["ExitDate"], strategy_cfg.signal_left_col],
            marker="v",
            s=90,
            label="Sell (executed)",
        )

    ax_top.set_title(strategy_cfg.chart_title)
    ax_top.set_ylabel(strategy_cfg.chart_ylabel)
    ax_top.legend(loc="best")
    ax_top.grid(alpha=0.25)

    # Bottom panel: equity curve
    ax_bottom.plot(bt_df.index, bt_df["Equity"], label="Equity Curve", linewidth=1.4)
    ax_bottom.set_ylabel("Equity")
    ax_bottom.set_xlabel("Date")
    ax_bottom.legend(loc="best")
    ax_bottom.grid(alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_equity_comparison(
    equity_map: Dict[str, pd.Series],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(14, 6))
    for name, series in equity_map.items():
        plt.plot(series.index, series.values, label=name, linewidth=1.4)
    plt.title("Equity Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------
# Utility / reporting
# -----------------------------
def ensure_dual_windows_valid(fast_window: int, slow_window: int, label: str) -> None:
    if fast_window >= slow_window:
        raise ValueError(f"{label}: fast window must be < slow window (got {fast_window} >= {slow_window}).")


def print_metrics(strategy_name: str, metrics: Dict[str, float]) -> None:
    print(f"\n{'=' * 78}")
    print(f"{strategy_name} Metrics")
    print("=" * 78)
    print(f"Total Return      : {metrics['Total Return']:.2%}")
    print(f"Annualized Return : {metrics['Annualized Return']:.2%}")
    print(f"Max Drawdown      : {metrics['Max Drawdown']:.2%}")
    sharpe_val = metrics["Sharpe Ratio"]
    print(f"Sharpe Ratio      : {sharpe_val:.4f}" if pd.notna(sharpe_val) else "Sharpe Ratio      : NaN")
    print(f"Number of Trades  : {int(metrics['Number of Trades'])}")
    win_rate_val = metrics["Win Rate"]
    print(f"Win Rate          : {win_rate_val:.2%}" if pd.notna(win_rate_val) else "Win Rate          : NaN")


def save_trade_log(trades: pd.DataFrame, path: str) -> None:
    trades_out = trades.copy()
    trades_out.to_csv(path, index=False)
    print(f"Trade log saved: {path}")
    if trades_out.empty:
        print("Trade log is empty (no completed trades).")
    else:
        print("Trade log preview:")
        print(trades_out.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL daily-close backtesting framework.")
    parser.add_argument("--start-date", type=str, default="2015-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end-date",
        type=str,
        default=str(date.today()),
        help="Backtest end date (YYYY-MM-DD). Note yfinance end date is exclusive.",
    )
    parser.add_argument("--initial-capital", type=float, default=100_000.0, help="Initial capital")
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.001,
        help="Commission rate per trade side (e.g. 0.001 = 0.10%)",
    )

    parser.add_argument("--price-single-ma", type=int, default=50, help="Window for price single MA strategy")
    parser.add_argument("--price-dual-fast", type=int, default=20, help="Fast MA for price dual strategy")
    parser.add_argument("--price-dual-slow", type=int, default=100, help="Slow MA for price dual strategy")

    parser.add_argument("--rs-single-ma", type=int, default=50, help="Window for RS single MA strategy")
    parser.add_argument("--rs-dual-fast", type=int, default=20, help="Fast MA for RS dual strategy")
    parser.add_argument("--rs-dual-slow", type=int, default=100, help="Slow MA for RS dual strategy")

    parser.add_argument(
        "--debug-signal-samples",
        type=int,
        default=3,
        help="Number of buy and sell sample windows to print for signal logic validation",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="soxl_backtest",
        help="Prefix for output CSV/PNG files",
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Disable interactive plot display (still saves images).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dual_windows_valid(args.price_dual_fast, args.price_dual_slow, "Price dual MA")
    ensure_dual_windows_valid(args.rs_dual_fast, args.rs_dual_slow, "RS dual MA")

    print("\nDownloading daily data (SOXL, SPY)...")
    base_df = download_daily_close_data(args.start_date, args.end_date)
    print(f"Data rows: {len(base_df)} | Range: {base_df.index.min().date()} -> {base_df.index.max().date()}")

    df = add_indicators(
        base_df,
        price_single_ma=args.price_single_ma,
        price_dual_fast=args.price_dual_fast,
        price_dual_slow=args.price_dual_slow,
        rs_single_ma=args.rs_single_ma,
        rs_dual_fast=args.rs_dual_fast,
        rs_dual_slow=args.rs_dual_slow,
    )
    df = generate_signals(df)

    backtest_cfg = BacktestConfig(
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
    )

    strategy_map = [
        {
            "cfg": StrategyConfig(
                name="Price Single MA",
                signal_left_col="SOXL_Close",
                signal_right_col="Price_MA_Single",
                indicator_cols=["Price_MA_Single"],
                chart_title="Price Single MA Strategy (Signals from Daily Close, Execute Next-Day Close)",
                chart_ylabel="Price",
            ),
            "buy_col": "PriceSingle_Buy",
            "sell_col": "PriceSingle_Sell",
            "event_col": "PriceSingle_Event",
        },
        {
            "cfg": StrategyConfig(
                name="Price Dual MA",
                signal_left_col="Price_MA_Fast",
                signal_right_col="Price_MA_Slow",
                indicator_cols=["Price_MA_Fast", "Price_MA_Slow"],
                chart_title="Price Dual MA Strategy (Signals from Daily Close, Execute Next-Day Close)",
                chart_ylabel="Price",
            ),
            "buy_col": "PriceDual_Buy",
            "sell_col": "PriceDual_Sell",
            "event_col": "PriceDual_Event",
        },
        {
            "cfg": StrategyConfig(
                name="RS Single MA",
                signal_left_col="RS",
                signal_right_col="RS_MA_Single",
                indicator_cols=["RS_MA_Single"],
                chart_title="RS Single MA Strategy (RS from Close, Execute Next-Day Close)",
                chart_ylabel="RS",
            ),
            "buy_col": "RSSingle_Buy",
            "sell_col": "RSSingle_Sell",
            "event_col": "RSSingle_Event",
        },
        {
            "cfg": StrategyConfig(
                name="RS Dual MA",
                signal_left_col="RS_MA_Fast",
                signal_right_col="RS_MA_Slow",
                indicator_cols=["RS_MA_Fast", "RS_MA_Slow"],
                chart_title="RS Dual MA Strategy (RS from Close, Execute Next-Day Close)",
                chart_ylabel="RS",
            ),
            "buy_col": "RSDual_Buy",
            "sell_col": "RSDual_Sell",
            "event_col": "RSDual_Event",
        },
    ]

    equity_curves: Dict[str, pd.Series] = {}
    all_metrics: Dict[str, Dict[str, float]] = {}

    print("\nExecution rule:")
    print("Signals are generated after daily close at t; orders execute on next trading day close (t+1).")
    print("Signal detection uses only close-to-close comparisons (no open/high/low/intraday).")

    for s in strategy_map:
        cfg = s["cfg"]
        buy_col = s["buy_col"]
        sell_col = s["sell_col"]
        event_col = s["event_col"]

        validate_crossover_logic(
            df=df,
            strategy_name=cfg.name,
            left_col=cfg.signal_left_col,
            right_col=cfg.signal_right_col,
            buy_col=buy_col,
            sell_col=sell_col,
        )

        print_signal_samples(
            df=df,
            strategy_name=cfg.name,
            left_col=cfg.signal_left_col,
            right_col=cfg.signal_right_col,
            buy_col=buy_col,
            sell_col=sell_col,
            max_samples_each_side=args.debug_signal_samples,
        )

        bt_df, trade_df, metrics = backtest_long_only_next_close(
            df=df,
            event_col=event_col,
            strategy_name=cfg.name,
            cfg=backtest_cfg,
        )
        validate_next_day_execution(trade_df, df.index, cfg.name)
        validate_trade_log_plot_alignment(trade_df, bt_df, cfg.name)
        print_metrics(cfg.name, metrics)

        trade_path = f"{args.output_prefix}_{cfg.name.replace(' ', '_').lower()}_trades.csv"
        save_trade_log(trade_df, trade_path)

        chart_path = f"{args.output_prefix}_{cfg.name.replace(' ', '_').lower()}.png"
        plot_strategy(
            df=df,
            bt_df=bt_df,
            trades=trade_df,
            strategy_cfg=cfg,
            save_path=chart_path,
            show=not args.no_show_plots,
        )
        print(f"Chart saved: {chart_path}")

        equity_curves[cfg.name] = bt_df["Equity"].copy()
        all_metrics[cfg.name] = metrics

    comparison_chart_path = f"{args.output_prefix}_equity_comparison.png"
    plot_equity_comparison(
        equity_map=equity_curves,
        save_path=comparison_chart_path,
        show=not args.no_show_plots,
    )
    print(f"\nEquity comparison chart saved: {comparison_chart_path}")

    summary_df = pd.DataFrame(all_metrics).T
    summary_df.to_csv(f"{args.output_prefix}_metrics_summary.csv")
    print("\nMetrics summary:")
    print(summary_df.to_string())
    print(f"\nMetrics summary saved: {args.output_prefix}_metrics_summary.csv")


if __name__ == "__main__":
    main()
