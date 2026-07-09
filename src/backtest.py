import vectorbt as vbt
import pandas as pd

def generate_signals(df, predictions):
    df = df.copy()
    df["signal"] = predictions
    df["position"] = df["signal"].shift(1)  # trade on next day, avoid lookahead
    return df.dropna(subset=["position"])

def run_backtest(df, initial_cash=10000, fees=0.001):
    entries = df["position"] == 1
    exits = df["position"] == 0

    pf = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        freq="1D"
    )
    return pf