import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # always attempt this — harmless no-op if .env doesn't exist (e.g. on Streamlit Cloud)


def get_secret(key):
    try:
        value = st.secrets[key]
    except (FileNotFoundError, KeyError, st.errors.StreamlitAPIException):
        value = os.getenv(key)
    return value.strip() if value else value


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