import sys
import os

# Add project root to Python path so "src" is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data_loader import download_data
from src.features import add_features, add_label
from src.model import walk_forward_train, FEATURES
from src.backtest import generate_signals, run_backtest

st.title("ML Trading Strategy Backtester")

ticker = st.selectbox("Choose ticker", ["AAPL", "MSFT", "SPY", "NVDA", "JPM", "TSLA"])

# --- Load and process data for the selected ticker ---
@st.cache_data
def load_and_run(ticker):
    data = download_data([ticker])
    df = data[ticker].copy()
    df = add_features(df)
    df = add_label(df)
    df = df.dropna().reset_index(drop=True)

    preds, fold_results = walk_forward_train(df)
    df_signals = generate_signals(df, preds)
    pf = run_backtest(df_signals)

    equity_curve = pf.value()
    benchmark_curve = df["Close"] / df["Close"].iloc[0] * 10000

    buy_hold_returns = df["Close"].pct_change().dropna()
    benchmark_sharpe = np.sqrt(252) * buy_hold_returns.mean() / buy_hold_returns.std()

    stats = pf.stats()

    return {
        "sharpe": stats["Sharpe Ratio"],
        "max_dd": stats["Max Drawdown [%]"] / 100,
        "win_rate": stats["Win Rate [%]"] / 100,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "benchmark_sharpe": benchmark_sharpe,
    }

results = load_and_run(ticker)

# --- Display metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Strategy Sharpe", f"{results['sharpe']:.2f}")
col2.metric("Max Drawdown", f"{results['max_dd']:.1%}")
col3.metric("Win Rate", f"{results['win_rate']:.1%}")

st.metric("Buy & Hold Sharpe (for comparison)", f"{results['benchmark_sharpe']:.2f}")

# --- Plot equity curves ---
fig, ax = plt.subplots()
ax.plot(results["equity_curve"], label="Strategy")
ax.plot(results["benchmark_curve"].values, label="Buy & Hold")
ax.set_title(f"{ticker}: Strategy vs Buy & Hold")
ax.legend()
st.pyplot(fig)