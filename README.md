# ML-Driven Algorithmic Trading Backtester

A machine learning system that predicts short-term stock price direction using technical
indicators, then rigorously backtests the resulting trading strategy against a buy-and-hold
benchmark across 8 tickers — with walk-forward validation to avoid look-ahead bias, honest
reporting of where the strategy does and doesn't add value, and a documented trail of
modeling experiments that were tested and rejected when they didn't hold up.

**[Live Demo](https://YOUR-APP-URL.streamlit.app)** — try it yourself, pick any ticker, see live results.
**Live paper trading**: this strategy runs daily against real market data via Alpaca's paper
trading API, fully automated via GitHub Actions — see [Live Trading](#live-paper-trading) below.

![Dashboard Screenshot](docs/screenshot.png)
*(Replace this with an actual screenshot of your Streamlit app)*

---

## Summary

Given a stock's recent price/volume history, can a machine learning model predict whether the
price will be higher or lower in 5 trading days — well enough to build a trading strategy that
beats simply buying and holding?

**Short answer: on a risk-adjusted basis, sometimes — and figuring out exactly when required
testing and rejecting two reasonable-sounding improvement ideas along the way.** The final
model (a Random Forest classifier using 8 pure price/volume technical indicators) shows a
modest but real Sharpe ratio edge over buy-and-hold on 3 of 8 tickers tested, concentrated in
more volatile, less relentlessly trending names.

---

## Methodology

**Data**: Daily OHLCV price data for 8 liquid tickers (AAPL, MSFT, GOOGL, AMZN, SPY, JPM, NVDA,
TSLA) from 2015–2026, sourced via the `yfinance` API.

**Features**: 8 technical indicators computed from price/volume alone — 1/5/10-day returns,
10 vs 50-day moving average ratio, 14-day RSI, 10-day rolling volatility, volume ratio, and
Bollinger Band position. All features use only information available as of the current day
(no look-ahead).

**Label**: Binary classification — did the stock's price rise over the following 5 trading days?

**Model**: Random Forest classifier, trained and evaluated with **walk-forward (time-series)
cross-validation** — never a random train/test split, since that would leak future information
into training and produce artificially inflated accuracy.

**Backtest**: Simulated using `vectorbt`, converting daily predictions into next-day positions
(shifted by one day to avoid trading on same-day information), with a 0.1% transaction cost
assumption per trade.

**Live validation**: The final model runs daily against live market data via Alpaca's paper
trading API (simulated capital, real market data and order mechanics), automated through
GitHub Actions — see [Live Paper Trading](#live-paper-trading).

---

## Results (final model)

| Ticker | Strategy Return | Benchmark Return | Strategy Sharpe | Benchmark Sharpe | Win Rate | Max Drawdown | Beats Benchmark (Sharpe)? |
|--------|-----------------:|------------------:|-----------------:|-------------------:|----------:|---------------:|:---:|
| AAPL   | 249.1%  | 879.4%   | 0.79 | 0.88 | 54.5% | 36.5% | No |
| MSFT   | 275.9%  | 1264.6%  | 0.87 | 1.04 | 62.1% | 33.6% | No |
| GOOGL  | 553.9%  | 1027.9%  | 1.10 | 0.92 | 65.3% | 34.1% | **Yes** |
| AMZN   | 334.5%  | 1143.6%  | 0.81 | 0.88 | 61.8% | 63.4% | No |
| SPY    | 174.0%  | 297.2%   | 0.86 | 0.81 | 53.5% | 33.7% | **Yes** |
| JPM    | 109.7%  | 604.8%   | 0.54 | 0.80 | 52.5% | 49.7% | No |
| NVDA   | 3740.2% | 34098.0% | 1.30 | 1.35 | 64.5% | 57.8% | No |
| TSLA   | 1542.6% | 3621.7%  | 1.02 | 0.87 | 49.6% | 72.7% | **Yes** |

*Walk-forward validated, 2015–2026, 0.1% transaction cost per trade.*

---

## Experiments tried, and why two were rejected

A documented record of the modeling approaches tested beyond the baseline, since knowing what
*doesn't* work — and why — is as much a part of this project as the final result.

### 1. Regression instead of classification (rejected)

**Hypothesis**: predicting the actual future return (a continuous number) instead of just
up/down direction would carry more information and let trades be filtered by predicted
magnitude, not just direction.

**Result**: the regression model's mean absolute error (~0.031 average across folds) was
essentially identical to a naive baseline that always predicts 0% return (0.029 MAE) — meaning
the model added no meaningful predictive value over simply assuming "no big move happens."
**Rejected** in favor of the classification approach.

### 2. Market-context / cross-asset features (rejected)

**Hypothesis**: individual stocks are partly driven by overall market movement, not just their
own technicals. Added SPY-based features (market return, relative strength vs. market, market
volatility regime).

**Result**: these features earned real, non-trivial importance in the trained model — not
ignored — but full walk-forward backtesting showed **worse** risk-adjusted performance across
almost every ticker, and the number of tickers beating benchmark on Sharpe dropped from 3 to 1.
This is a textbook overfitting signal: with only ~2,700 rows per ticker, adding 5 more features
increased model complexity without enough data to support it, letting the model fit noise in
the additional features that didn't generalize out-of-sample. **Rejected**, reverted to the
original 8-feature set.

### Why this matters

Both experiments were reasonable, well-motivated ideas — the kind an interviewer might expect
to see. Testing each one honestly against the existing baseline (not just accuracy, but full
walk-forward backtest performance) and being willing to discard both when they underperformed,
rather than keeping whichever result looked most impressive, is the actual point.

---

## Key learnings

**Total return is the wrong metric to lead with.** 2015–2026 included one of the strongest
sustained bull markets in history; almost no strategy that periodically exits the market beats
"just hold" in that regime. Sharpe ratio (return per unit of risk) is the more honest lens.

**The Sharpe edge is concentrated in more volatile, less relentlessly trending names.** GOOGL,
TSLA, and SPY all have more real pullbacks along the way, where correctly timing exits has more
value, versus AAPL/MSFT/AMZN/NVDA's smoother sustained climbs where any time out of the market
is costly regardless of timing quality.

**The model is a genuinely low-confidence, low-conviction predictor**, confirmed both by its
close-to-50% accuracy and by its probability outputs, which almost never leave the 0.4–0.7
range (verified via calibration curve analysis) — it never expresses strong conviction either
way. This is consistent across both the classification and regression framings.

**Adding more features doesn't necessarily help, even features with a sound theoretical
motivation** — the market-context experiment is a concrete, quantified example of overfitting
from added complexity outweighing the added signal, given limited training data.

**Modest Sharpe improvements (1.10 vs 0.92, not 3.0 vs 0.9) are the believable outcome.**
A dramatically larger edge would be a stronger signal of a bug or leakage than of genuine
skill — daily stock direction prediction from basic technicals is a hard, close-to-efficient
problem.

---

## Live Paper Trading

The final (8-feature, classification) model runs daily via a scheduled **GitHub Actions**
workflow, generating fresh signals from live market data and executing simulated trades through
**Alpaca's paper trading API** (no real capital involved). This validates the full pipeline —
data ingestion, feature engineering, prediction, and order execution — against live conditions,
not just historical backtests.

- Runs weekdays, automated end-to-end, no manual intervention required
- Full logs of every signal and trade decision retained as run artifacts
- Live position/P&L status visible in the [dashboard](#) under "Live Paper Trading Status"

---

## Limitations & what I'd try next

- **No formal significance testing yet** on the Sharpe differences — a bootstrap confidence
  interval on returns would clarify whether the 3-ticker edge is likely real or within noise.
- **Single prediction horizon (5 days)** and **single confidence threshold** — a sensitivity
  analysis across horizons/thresholds would test robustness to these specific choices.
- **Pooled multi-ticker training** (one model trained across all 8 tickers' data, ~20,000+ rows
  instead of ~2,700 per ticker) is a promising untested direction — the market-context failure
  suggests the real bottleneck may be data volume per model, not feature availability.
- **Single-ticker position sizing** in live trading — no portfolio-level risk management yet
  (correlation across positions, max total exposure, etc.).

---

## Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `vectorbt` · `yfinance` · `Streamlit` ·
`Alpaca API` · `GitHub Actions` · `matplotlib`

---

## Project Structure

```
ml-trading-backtester/
├── src/
│   ├── data_loader.py    # yfinance data pull + local caching
│   ├── features.py       # technical indicator engineering + label creation
│   ├── model.py           # walk-forward validated Random Forest training
│   ├── backtest.py       # vectorbt-based strategy simulation
│   ├── metrics.py         # Sharpe, max drawdown, win rate calculations
│   ├── broker.py           # Alpaca client + credential handling
│   └── live_trading.py    # daily signal generation + paper order execution
├── app/
│   └── streamlit_app.py  # interactive dashboard, deployed live
├── .github/workflows/
│   └── daily_trading.yml # scheduled automation for live paper trading
├── notebooks/             # exploration and experiment notebooks
├── data/                  # cached price data (not committed)
└── requirements.txt
```

---

## Running it locally

```bash
git clone https://github.com/Mrigna01/ml-trading-backtester.git
cd ml-trading-backtester
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

To run live paper trading locally, add a `.env` file with `ALPACA_API_KEY` and
`ALPACA_SECRET_KEY` (see `src/broker.py`), then:
```bash
python -m src.live_trading
```