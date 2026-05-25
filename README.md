# Adaptive Options Trading Bot

This repository contains a sophisticated day trading options bot built with the [Lumibot](https://lumibot.lumiwealth.com/) framework. The bot uses an ensemble of five different pricing models with adaptive weighting to identify mispriced options.

## Features

- **Ensemble Pricing**: Combines 5 models:
  - 20% Black-Scholes
  - 15% Monte Carlo
  - 15% Binomial
  - 15% Heston
  - 35% Machine Learning (Random Forest)
- **Adaptive Weighting**: A meta-model dynamically adjusts these weights based on recent market accuracy and persists its state.
- **Multi-Source Data**: Primarily uses Yahoo Finance, with Alpha Vantage as a secondary source (includes query rate limiting and intraday support).
- **Options Database**: Local SQLite storage for historical prices and option chains.
- **Reporting**: Detailed HTML backtest reports via Quantstats (saved in `reports/`) and Lumibot logs.
- **Margin Control**: Global `ALLOW_MARGIN` toggle to prevent over-leveraged trades.
- **Persistence**: Machine Learning models and Adaptive Weights are saved to `.joblib` files for future use.

## Setup Instructions

### 1. Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install lumibot quantstats yfinance alpha_vantage pandas numpy scikit-learn scipy matplotlib python-dotenv joblib
```

### 2. Configuration

Create a `.env` file in the root directory and add your API credentials:

```env
# Tradier Credentials
TRADIER_ACCOUNT_NUMBER=your_account_number
TRADIER_ACCESS_TOKEN=your_access_token
TRADIER_IS_PAPER=True

# Alpha Vantage (Free subscription)
AV_API_KEY=your_alpha_vantage_key
```

### 3. Execution

The bot is orchestrated through `main.py`.

#### Backtesting
Runs a historical simulation. 
*Note: Since free data sources like Yahoo Finance do not provide historical option chains, the backtest uses synthetic option pricing (Black-Scholes proxy) to simulate and verify the bot's ensemble logic.*

```bash
python main.py --mode backtest --ticker SPY --start 2023-01-01 --end 2023-12-31
```
Reports will be generated in `logs/` and `reports/`.

#### Paper Trading
Runs the bot in a live simulated environment using Tradier Paper.
```bash
python main.py --mode paper --ticker SPY
```

#### Live Trading
Runs the bot with real capital (ensure `TRADIER_IS_PAPER=False` in your `.env`).
```bash
python main.py --mode live --ticker SPY
```

## Disclaimer

Trading options involves significant risk. This bot is for educational and research purposes only. Always thoroughly backtest any strategy before using real capital.
