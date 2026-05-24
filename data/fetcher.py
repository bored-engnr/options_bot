import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import logging

class DataFetcher:
    def __init__(self, av_api_key=None):
        self.av_api_key = av_api_key
        self.last_av_call = 0
        self.av_rate_limit = 12  # seconds between calls for free tier (5 calls per minute)

    def fetch_historical_yfinance(self, symbol, start_date=None, end_date=None, interval="1d"):
        """Fetch historical prices from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        return df

    def fetch_options_chain_yfinance(self, symbol, expiration):
        """Fetch options chain from Yahoo Finance for a specific expiration."""
        ticker = yf.Ticker(symbol)
        opt = ticker.option_chain(expiration)

        calls = opt.calls.copy()
        calls['option_type'] = 'call'

        puts = opt.puts.copy()
        puts['option_type'] = 'put'

        df = pd.concat([calls, puts])

        # Standardize columns
        rename_map = {
            'strike': 'strike',
            'lastPrice': 'last_price',
            'bid': 'bid',
            'ask': 'ask',
            'volume': 'volume',
            'openInterest': 'open_interest',
            'impliedVolatility': 'implied_volatility'
        }
        df = df.rename(columns=rename_map)
        return df[['strike', 'option_type', 'last_price', 'bid', 'ask', 'volume', 'open_interest', 'implied_volatility']]

    def fetch_expirations_yfinance(self, symbol):
        ticker = yf.Ticker(symbol)
        return ticker.options

    def fetch_historical_alpha_vantage(self, symbol, interval="daily"):
        """Fetch historical prices from Alpha Vantage with rate limiting."""
        if not self.av_api_key:
            logging.warning("Alpha Vantage API key not provided.")
            return pd.DataFrame()

        # Simple rate limiting
        elapsed = time.time() - self.last_av_call
        if elapsed < self.av_rate_limit:
            time.sleep(self.av_rate_limit - elapsed)

        ts = TimeSeries(key=self.av_api_key, output_format='pandas')
        try:
            if interval == "daily":
                data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
            else:
                # Add more intervals as needed
                data, meta_data = ts.get_intraday(symbol=symbol, interval='5min', outputsize='full')

            self.last_av_call = time.time()

            # Standardize AV columns to match YFinance
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            return data
        except Exception as e:
            logging.error(f"Error fetching from Alpha Vantage: {e}")
            return pd.DataFrame()

    def get_data(self, symbol, start_date=None, end_date=None):
        """Primary method to get historical data, using yfinance primarily."""
        df = self.fetch_historical_yfinance(symbol, start_date, end_date)
        if df.empty and self.av_api_key:
            logging.info(f"YFinance failed for {symbol}, trying Alpha Vantage.")
            df = self.fetch_historical_alpha_vantage(symbol)
        return df
