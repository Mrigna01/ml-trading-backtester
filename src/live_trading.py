import os
import logging
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.data_loader import download_data
from src.features import add_features, add_label
from src.model import walk_forward_train, FEATURES

load_dotenv()

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_log.txt"),
        logging.StreamHandler()  # also prints to console, so you still see it live
    ]
)
logger = logging.getLogger(__name__)


def get_trading_client():
    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True
    )


def get_current_position(client, ticker):
    """Returns number of shares currently held, or 0 if none."""
    try:
        position = client.get_open_position(ticker)
        return float(position.qty)
    except Exception:
        return 0.0


def generate_todays_signal(ticker):
    """Re-runs the pipeline fresh and returns today's prediction (1 = buy/hold, 0 = flat)."""
    try:
        data = download_data([ticker])
        df = data[ticker].copy()
        df = add_features(df)
        df = add_label(df)
        df = df.dropna().reset_index(drop=True)

        # Train on all available history, predict on the most recent row
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(df[FEATURES].iloc[:-1], df["label"].iloc[:-1])  # exclude last row (no future label yet)
        todays_features = df[FEATURES].iloc[[-1]]
        prediction = model.predict(todays_features)[0]

        logger.info(f"[{ticker}] Signal generated: {int(prediction)}")
        return int(prediction)

    except Exception as e:
        logger.error(f"[{ticker}] Failed to generate signal: {e}")
        raise


def execute_trade(client, ticker, signal, position_size_usd=1000):
    current_qty = get_current_position(client, ticker)

    try:
        if signal == 1 and current_qty == 0:
            order = MarketOrderRequest(
                symbol=ticker,
                notional=position_size_usd,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            logger.info(f"[{ticker}] BUY order submitted for ${position_size_usd} (no prior position)")

        elif signal == 0 and current_qty > 0:
            order = MarketOrderRequest(
                symbol=ticker,
                qty=current_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            logger.info(f"[{ticker}] SELL order submitted for {current_qty} shares")

        else:
            logger.info(f"[{ticker}] No action needed (signal={signal}, current_qty={current_qty})")

    except Exception as e:
        logger.error(f"[{ticker}] Order execution failed: {e}")


if __name__ == "__main__":
    logger.info("=== Starting daily trading run ===")

    client = get_trading_client()
    account = client.get_account()
    logger.info(f"Paper account buying power: ${account.buying_power}")

    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "JPM", "NVDA", "TSLA"]

    for ticker in TICKERS:
        try:
            signal = generate_todays_signal(ticker)
            execute_trade(client, ticker, signal, position_size_usd=1000)
        except Exception:
            logger.error(f"[{ticker}] Skipped due to earlier error, moving to next ticker")
            continue

    logger.info("=== Daily trading run complete ===")