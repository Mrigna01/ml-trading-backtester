import yfinance as yf
import pandas as pd
import os

def download_data(tickers, start="2015-01-01", end="2026-01-01", cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    all_data = {}
    for ticker in tickers:
        cache_path = f"{cache_dir}/{ticker}.csv"
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)
            df.to_csv(cache_path)
        all_data[ticker] = df
    return all_data

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "JPM", "NVDA", "TSLA"]
    data = download_data(tickers)
    print({t: df.shape for t, df in data.items()})  