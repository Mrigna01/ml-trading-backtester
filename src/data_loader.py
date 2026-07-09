import yfinance as yf
import pandas as pd
import os

def download_data(tickers, start="2015-01-01", end="2026-01-01", cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    all_data = {}
    price_fields = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}

    for ticker in tickers:
        cache_path = f"{cache_dir}/{ticker}.csv"
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)

            if isinstance(df.columns, pd.MultiIndex):
                level0_vals = set(df.columns.get_level_values(0))
                # pick whichever level actually contains Open/High/Low/Close/Volume
                if level0_vals & price_fields:
                    df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = df.columns.get_level_values(1)

            df.index.name = "Date"
            df.to_csv(cache_path)
        all_data[ticker] = df
    return all_data