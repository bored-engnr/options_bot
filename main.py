from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Tradier
from lumibot.traders import Trader
from strategy.options_strategy import AdaptiveOptionsStrategy
from utils.config import Config
from utils.reporting import ReportGenerator
from datetime import datetime
import pandas as pd
import os
import logging

def run_backtest():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Run backtest using Yahoo Finance data
    results = AdaptiveOptionsStrategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={
            "symbol": "SPY",
            "av_api_key": Config.AV_API_KEY,
            "risk_free_rate": Config.RISK_FREE_RATE
        }
    )
    
    # Manually generate quantstats report if needed, though lumibot generates some
    # We can use the results from lumibot backtest
    # results is a DataFrame with strategy performance
    if not os.path.exists("reports"):
        os.makedirs("reports")
    
    reporter = ReportGenerator(output_dir="reports")
    # Lumibot backtest returns a dict or dataframe depending on version
    # In recent versions it returns a lot of info.
    # For now, we rely on Lumibot's auto-generated reports in logs/ as seen in verification
    print("Backtest completed. Check logs/ directory for auto-generated HTML reports.")

def run_live(is_paper=True):
    broker = Tradier(
        Config.TRADIER_ACCOUNT_NUMBER,
        Config.TRADIER_ACCESS_TOKEN,
        paper=is_paper
    )
    
    strategy = AdaptiveOptionsStrategy(
        broker=broker,
        parameters={
            "symbol": "SPY",
            "av_api_key": Config.AV_API_KEY,
            "risk_free_rate": Config.RISK_FREE_RATE
        }
    )
    
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()

if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Default to backtest if no args
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    
    if mode == "backtest":
        run_backtest()
    elif mode == "paper":
        run_live(is_paper=True)
    elif mode == "live":
        run_live(is_paper=False)
