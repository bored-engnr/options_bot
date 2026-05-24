# Adaptive Options Trading Bot

This repository contains a sophisticated day trading options bot built with the [Lumibot](https://lumibot.lumiwealth.com/) framework. The bot uses an ensemble of five different pricing models with adaptive weighting to identify mispriced options.

## Features

- **Ensemble Pricing**: Combines 5 models:
  - 20% Black-Scholes
  - 15% Monte Carlo
  - 15% Binomial
  - 15% Heston
  - 35% Machine Learning (Random Forest)
- **Adaptive Weighting**: A meta-model dynamically adjusts these weights based on recent market accuracy.
- **Multi-Source Data**: Primarily uses Yahoo Finance, with Alpha Vantage as a secondary source (includes query rate limiting).
- **Options Database**: Local SQLite storage for historical prices and option chains.
- **Reporting**: Detailed HTML backtest reports via Quantstats.
- **Execution**: Supports Backtesting, Paper Trading, and Live Trading via the Tradier broker.

## Setup Instructions

### 1. Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install lumibot quantstats yfinance alpha_vantage pandas numpy scikit-learn scipy matplotlib python-dotenv
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

The bot is orchestrated through `main.py`. You can run it in three modes:

#### Backtesting
Runs a historical simulation from 2023-01-01 to 2023-12-31 (configurable in `main.py`).
```bash
python main.py backtest
```
Reports will be generated in the `logs/` directory as HTML files.

#### Paper Trading
Runs the bot in a live simulated environment using Tradier Paper.
```bash
python main.py paper
```

#### Live Trading
Runs the bot with real capital (ensure `TRADIER_IS_PAPER=False` in your `.env`).
```bash
python main.py live
```

## Project Structure

- `data/`: Database (`OptionsDB`) and Data Fetching (`DataFetcher`) logic.
- `models/pricing/`: Implementations of BS, Monte Carlo, Binomial, and Heston models.
- `models/ml/`: Machine Learning predictor and the Adaptive Weighting mechanism.
- `strategy/`: The core `AdaptiveOptionsStrategy` inheriting from Lumibot.
- `utils/`: Configuration and Reporting helpers.
- `main.py`: Entry point for all execution modes.

## Disclaimer

Trading options involves significant risk. This bot is for educational and research purposes only. Always thoroughly backtest any strategy before using real capital.
