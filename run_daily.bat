cd C:\Users\mrign\ml-trading-backtester
call venv\Scripts\activate
python -m src.live_trading >> logs\trading_log.txt 2>&1
deactivate