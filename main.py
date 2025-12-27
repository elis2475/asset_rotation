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
# Data source STOOQ
# =======================
def stooq_symbol(ticker: str) -> str:
    return f"{ticker.lower()}.us"



def fetch_stooq_daily_close(ticker: str) -> pd.Series:
    sym = stooq_symbol(ticker)
    url = fhttpsstooq.comqdls={sym}&i=d
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df[Date] = pd.to_datetime(df[Date])
    df = df.sort_values(Date).set_index(Date)

    if Close not in df.columns or df[Close].dropna().empty
        raise ValueError(fNo Close data for {ticker})

    return df[Close].astype(float)


# =======================
# Indicators
# =======================
def rsi14(close pd.Series, period int = 14) - float
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = gains.ewm(alpha=1  period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1  period, adjust=False).mean()

    rs = avg_gain  avg_loss.replace(0, np.nan)
    rsi = 100 - (100  (1 + rs))
    return float(rsi.iloc[-1])


def sma(close pd.Series, window int) - float
    return float(close.rolling(window).mean().iloc[-1])


def pct_return(close pd.Series, trading_days_ago int) - float
    if len(close) = trading_days_ago
        raise ValueError(Not enough history)
    return (float(close.iloc[-1])  float(close.iloc[-1 - trading_days_ago])) - 1.0


@dataclass
class Metrics
    ticker str
    perf_1m float
    perf_3m float
    rsi float
    sma200 float
    below200 bool
    price float


def compute_metrics_for_ticker(ticker str) - Metrics
    close = fetch_stooq_daily_close(ticker)

    price = float(close.iloc[-1])
    sma200_v = sma(close, 200)
    below200 = price  sma200_v

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
# Ranking helpers
# =======================
def sequential_rank_after_sort(
    df pd.DataFrame, sort_col str, ascending bool, rank_col str
) - pd.DataFrame
    df = df.sort_values(
        by=sort_col, ascending=ascending, kind=mergesort
    ).reset_index(drop=True)
    df[rank_col] = np.arange(1, len(df) + 1)
    return df


def load_tickers_keep_duplicates(path Path) - List[str]
    lines = [ln.strip() for ln in path.read_text(encoding=utf-8).splitlines()]
    return [ln for ln in lines if ln]


# =======================
# Main pipeline
# =======================
def main()
    root = Path(__file__).resolve().parent
    tickers_path = root  tickers.txt
    results_dir = root  results
    docs_dir = root  d_



