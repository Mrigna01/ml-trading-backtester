import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()
def get_secret(key):
    """
    Reads a credential from Streamlit secrets (works both on Streamlit Cloud
    and locally if you have a .streamlit/secrets.toml file), falling back to
    a .env-loaded environment variable for non-Streamlit scripts like
    live_trading.py run directly from the terminal.
    """
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.getenv(key)


def get_trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key=get_secret("ALPACA_API_KEY"),
        secret_key=get_secret("ALPACA_SECRET_KEY"),
        paper=True
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # only needed when running this file directly via terminal

    client = get_trading_client()
    account = client.get_account()
    print(f"Account status: {account.status}")
    print(f"Buying power: ${account.buying_power}")
    print(f"Portfolio value: ${account.portfolio_value}")