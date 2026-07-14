import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # always attempt this — harmless no-op if .env doesn't exist (e.g. on Streamlit Cloud)


def get_secret(key):
    """
    Reads a credential from Streamlit secrets (works on Streamlit Cloud, or
    locally with a .streamlit/secrets.toml file), falling back to a
    .env-loaded environment variable otherwise.
    """
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError, st.errors.StreamlitAPIException):
        return os.getenv(key)


def get_trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key=get_secret("ALPACA_API_KEY"),
        secret_key=get_secret("ALPACA_SECRET_KEY"),
        paper=True
    )


if __name__ == "__main__":
    client = get_trading_client()
    account = client.get_account()
    print(f"Account status: {account.status}")
    print(f"Buying power: ${account.buying_power}")
    print(f"Portfolio value: ${account.portfolio_value}")