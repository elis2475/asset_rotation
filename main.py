from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


# =======================
# Data source: STOOQ
# =======================
def stooq_symbol(ticker: str) -> str:
    return f"{ticker.lower()}.us"


def fetch_stooq_daily_close(ticker: str) -> pd.Series:
    sym = stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if "Close" not in df.columns or df["Close"].dropna().empty:
        raise ValueError(f"No Close data for {ticker}")

    return df["Close"].astype(float)


# =======================
# Indicators
# =======================
def rsi14(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def sma(close: pd.Series, window: int) -> float:
    return float(close.rolling(window).mean().iloc[-1])


def pct_return(close: pd.Series, trading_days_ago: int) -> float:
    if len(close) <= trading_days_ago:
        raise ValueError("Not enough history")
    return (float(close.iloc[-1]) / float(close.iloc[-1 - trading_days_ago])) - 1.0


# =======================
# Data container
# =======================
@dataclass
class Metrics:
    ticker: str
    perf_1m: float      # decimal, e.g. 0.043
    perf_3m: float      # decimal, e.g. 0.123
    rsi: float
    sma200: float
    below200: bool
    price: float


def compute_metrics_for_ticker(ticker: str) -> Metrics:
    close = fetch_stooq_daily_close(ticker)

    price = float(close.iloc[-1])
    sma200_v = sma(close, 200)
    below200 = price < sma200_v

    # Approx trading days: 1M ~ 21, 3M ~ 63
    perf_1m = pct_return(close, 21)
    perf_3m = pct_return(close, 63)

    rsi_v = rsi14(close, 14)

    return Metrics(
        ticker=ticker,
        perf_1m=perf_1m,
        perf_3m=perf_3m,
        rsi=rsi_v,
        sma200=sma200_v,
        below200=below200,
        price=price,
    )


# =======================
# Helpers
# =======================
def load_tickers_keep_duplicates(path: Path) -> List[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def sequential_rank_after_sort(df: pd.DataFrame, sort_col: str, ascending: bool, rank_col: str) -> pd.DataFrame:
    # mergesort is stable => duplicates keep relative order
    df = df.sort_values(by=sort_col, ascending=ascending, kind="mergesort").reset_index(drop=True)
    df[rank_col] = np.arange(1, len(df) + 1)
    return df


# =======================
# Main pipeline
# =======================
def main() -> None:
    root = Path(__file__).resolve().parent
    tickers_path = root / "tickers.txt"
    results_dir = root / "results"
    docs_dir = root / "docs"

    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    # 1) Read tickers (KEEP duplicates as separate rows)
    tickers = load_tickers_keep_duplicates(tickers_path)
    if not tickers:
        raise ValueError("tickers.txt is empty")

    # 2) Fetch+compute once per UNIQUE ticker
    unique_tickers = sorted(set(tickers))
    metrics_map: Dict[str, Metrics] = {}
    errors: List[Tuple[str, str]] = []

    for t in unique_tickers:
        try:
            metrics_map[t] = compute_metrics_for_ticker(t)
        except Exception as e:
            errors.append((t, str(e)))

    # 3) Build row-per-input-line (duplicates preserved)
    rows = []
    for t in tickers:
        m = metrics_map.get(t)
        if m is None:
            rows.append({
                "Ticker": t,
                "Perf_3M": np.nan,
                "Perf_1M": np.nan,
                "RSI14": np.nan,
                "Below_200SMA": "",
                "Price": np.nan,
                "SMA200": np.nan,
            })
        else:
            rows.append({
                "Ticker": t,
                "Perf_3M": m.perf_3m,
                "Perf_1M": m.perf_1m,
                "RSI14": m.rsi,
                "Below_200SMA": "Yes" if m.below200 else "No",
                "Price": m.price,
                "SMA200": m.sma200,
            })

    df = pd.DataFrame(rows)

    # 4) Rank like your Excel process (sequential after sorting)
    df = sequential_rank_after_sort(df, "Perf_3M", ascending=False, rank_col="Rank_3M")
    df = sequential_rank_after_sort(df, "Perf_1M", ascending=False, rank_col="Rank_1M")
    df = sequential_rank_after_sort(df, "RSI14", ascending=True, rank_col="Rank_RSI")

    # 5) Aggregate Score EXACTLY like Excel:
    # (E + H + K) * IF(L="Yes";100000;1)
    penalty = np.where(
        df["Below_200SMA"].astype(str).str.lower().isin(["yes", "si", "sí"]),
        100000,
        1,
    )
    df["Aggregate_Score"] = (df["Rank_3M"] + df["Rank_1M"] + df["Rank_RSI"]) * penalty
    df["Aggregate_Score"] = df["Aggregate_Score"].round(0).astype(int)

    # Final sort by Aggregate Score ASC + sequential Aggregate Ranking
    df = sequential_rank_after_sort(df, "Aggregate_Score", ascending=True, rank_col="Aggregate_Ranking")

    # 6) Format output for CSV (percent with 2 decimals)
    out = df.copy()
    out["Perf_3M"] = (out["Perf_3M"] * 100).map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    out["Perf_1M"] = (out["Perf_1M"] * 100).map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    out["RSI14"] = out["RSI14"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["Price"] = out["Price"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["SMA200"] = out["SMA200"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    # Ensure Aggregate_Score / rankings are integers in the CSV
    out["Rank_3M"] = out["Rank_3M"].astype(int)
    out["Rank_1M"] = out["Rank_1M"].astype(int)
    out["Rank_RSI"] = out["Rank_RSI"].astype(int)
    out["Aggregate_Score"] = out["Aggregate_Score"].astype(int)
    out["Aggregate_Ranking"] = out["Aggregate_Ranking"].astype(int)

    # 7) Write files
    today = date.today()
    snap_name = f"{today:%Y-%m}-01.csv"

    out.to_csv(results_dir / snap_name, index=False)
    out.to_csv(docs_dir / "latest.csv", index=False)

    if errors:
        pd.DataFrame(errors, columns=["Ticker", "Error"]).to_csv(
            results_dir / f"errors_{today:%Y-%m}-01.csv", index=False
        )

    print("Pipeline completed successfully.")
    print(f"Wrote results/{snap_name} and docs/latest.csv")


if __name__ == "__main__":
    main()
