import argparse
import logging
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Tradier
from lumibot.traders import Trader
from strategy.options_strategy import AdaptiveOptionsStrategy
from utils.config import Config
from utils.reporting import ReportGenerator
import os
import pandas as pd

def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

def run_backtest(ticker, start_date, end_date):
    results = AdaptiveOptionsStrategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={
            "symbol": ticker,
            "av_api_key": Config.AV_API_KEY,
            "risk_free_rate": Config.RISK_FREE_RATE,
            "model_path": os.path.join("models", "ml", f"{ticker}_model.joblib")
        }
    )
    
    # Generate QuantStats Report
    if results is not None:
        reporter = ReportGenerator()
        # results in newer lumibot might be a dict containing 'strategy' or a dataframe
        # Usually it's a dataframe of performance
        if isinstance(results, pd.DataFrame):
            reporter.generate_quantstats_report(results, ticker=ticker)
        elif isinstance(results, dict) and 'strategy' in results:
             # handle dict if applicable
             pass

    print(f"Backtest completed for {ticker}. Check logs/ and reports/ directories.")

def run_live(ticker, mode):
    is_paper = (mode == "paper")
    broker = Tradier(Config.TRADIER_ACCOUNT_NUMBER, Config.TRADIER_ACCESS_TOKEN, paper=is_paper)
    strategy = AdaptiveOptionsStrategy(broker=broker, parameters={
            "symbol": ticker,
            "av_api_key": Config.AV_API_KEY,
            "risk_free_rate": Config.RISK_FREE_RATE,
            "model_path": os.path.join("models", "ml", f"{ticker}_model.joblib")
        })
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Options Trading Bot")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], default="backtest")
    parser.add_argument("--ticker", default=Config.DEFAULT_SYMBOL)
    parser.add_argument("--start", type=parse_date, default="2023-01-01")
    parser.add_argument("--end", type=parse_date, default="2023-12-31")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    if args.mode == "backtest":
        run_backtest(args.ticker, args.start, args.end)
    else:
        run_live(args.ticker, args.mode)
