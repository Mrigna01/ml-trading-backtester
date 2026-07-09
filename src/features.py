import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Trend
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]

    # Momentum: RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volatility
    df["volatility_10d"] = df["return_1d"].rolling(10).std()

    # Volume
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Bollinger Band position
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()
    df["bb_position"] = (df["Close"] - rolling_mean) / (2 * rolling_std)

    return df

def add_label(df, horizon=5):
    df = df.copy()
    df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["label"] = (df["future_return"] > 0).astype(int)
    return df