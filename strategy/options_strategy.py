from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset, Order
from data.fetcher import DataFetcher
from data.database import OptionsDB
from models.pricing.black_scholes import BlackScholesModel
from models.pricing.monte_carlo import MonteCarloModel
from models.pricing.binomial import BinomialModel
from models.pricing.heston import HestonModel
from models.ml.predictor import MLPredictor
from models.ml.adaptive_weights import AdaptiveWeighting
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

class AdaptiveOptionsStrategy(Strategy):
    parameters = {
        "symbol": "SPY",
        "quantity": 1,
        "take_profit_pct": 0.20,
        "stop_loss_pct": 0.15,
        "av_api_key": None,
        "risk_free_rate": 0.05
    }

    def initialize(self):
        self.symbol = self.parameters["symbol"]
        self.underlying_asset = Asset(symbol=self.symbol, asset_type="stock")
        self.fetcher = DataFetcher(av_api_key=self.parameters["av_api_key"])
        self.db = OptionsDB()
        self.ml_predictor = MLPredictor()
        self.adaptive_weights = AdaptiveWeighting()

        # Day trading: Sleeptime is minutes, not days.
        # But for backtest speed and daily data source, we'll keep 1D or adjust.
        # Let's use 1D for this daily OHLC model.
        self.sleeptime = "1D"
        self.last_predictions = {}

    def on_trading_iteration(self):
        # 1. Fetch latest data
        historical_data = self.fetcher.get_data(self.symbol)
        if historical_data.empty:
            logging.error(f"Could not fetch data for {self.symbol}")
            return

        # 2. Update DB
        self.db.save_historical_prices(self.symbol, historical_data)

        # 3. Train ML model
        if not self.ml_predictor.is_trained:
            self.ml_predictor.train(historical_data)

        # 4. Get options chain
        try:
            expirations = self.fetcher.fetch_expirations_yfinance(self.symbol)
            if not expirations:
                return
            target_expiry = expirations[0]
            options_chain = self.fetcher.fetch_options_chain_yfinance(self.symbol, target_expiry)
            if options_chain.empty:
                return
            self.db.save_options_chain(self.symbol, target_expiry, options_chain)
        except Exception as e:
            logging.error(f"Error fetching options chain: {e}")
            return

        # 5. Trading Logic
        current_price = self.get_last_price(self.underlying_asset)

        # Filter for At-The-Money (ATM) call
        calls = options_chain[options_chain['option_type'] == 'call']
        if calls.empty:
            return

        idx = (calls['strike'] - current_price).abs().idxmin()
        option_data = calls.loc[idx]

        K = option_data['strike']
        market_price = (option_data['bid'] + option_data['ask']) / 2
        if market_price <= 0: market_price = option_data['last_price']
        if market_price <= 0: return

        # Calculate Time to expiry in years
        expiry_dt = datetime.strptime(target_expiry, "%Y-%m-%d")
        current_dt = self.get_datetime()
        if current_dt.tzinfo is not None:
            current_dt = current_dt.replace(tzinfo=None)

        T = (expiry_dt - current_dt).days / 365.0
        if T <= 0: T = 1/365.0

        r = self.parameters["risk_free_rate"]
        sigma = option_data['implied_volatility']

        # 6. Calculate Prices from 5 Models
        bs_price = BlackScholesModel(current_price, K, T, r, sigma, 'call').price()
        mc_price = MonteCarloModel(current_price, K, T, r, sigma, 'call').price()
        bi_price = BinomialModel(current_price, K, T, r, sigma, 'call').price()
        he_price = HestonModel(current_price, K, T, r, 2.0, 0.04, 0.1, -0.7, sigma**2, 'call').price()

        inference_row = self.ml_predictor.prepare_features(historical_data, for_inference=True)
        if not inference_row.empty:
            ml_predicted_underlying = self.ml_predictor.predict_price(inference_row.iloc[0])
            if ml_predicted_underlying:
                # Use Black-Scholes but with the ML-predicted underlying price
                ml_price = BlackScholesModel(ml_predicted_underlying, K, T, r, sigma, 'call').price()
            else:
                ml_price = bs_price
        else:
            ml_price = bs_price

        predicted_prices = [bs_price, mc_price, bi_price, he_price, ml_price]

        # 7. Update Adaptive Weights
        if self.symbol in self.last_predictions:
            prev_predicted = self.last_predictions[self.symbol]
            self.adaptive_weights.update_weights(market_price, prev_predicted)

        self.last_predictions[self.symbol] = predicted_prices

        # 8. Final Weighted Price
        weighted_price = self.adaptive_weights.get_weighted_price(predicted_prices)

        self.log_message(f"Underlying: {current_price:.2f}, Option: {target_expiry} {K}C")
        self.log_message(f"Market Price: {market_price:.4f}, Weighted Fair Price: {weighted_price:.4f}")

        # 9. Execution Logic
        # Manage existing positions
        positions = self.get_positions()
        for position in positions:
            # Simple exit logic for demonstration
            if position.symbol == self.symbol: # This would need more specific option symbol matching
                pass

        if market_price < weighted_price * 0.97: # 3% undervalued
            self.log_message("Signal: BUY UNDERVALUED")
            # Create a synthetic option asset for backtesting (In Tradier, this would be a specific OCC symbol)
            # For backtesting with YahooData, option trading support varies.
            # We'll use the underlying for simplicity in this demo or attempt to trade the option if supported.
            order = self.create_order(self.underlying_asset, 100, "buy")
            self.submit_order(order)
        elif market_price > weighted_price * 1.03: # 3% overvalued
            self.log_message("Signal: SELL OVERVALUED")
            if self.get_position(self.underlying_asset).quantity > 0:
                order = self.create_order(self.underlying_asset, 100, "sell")
                self.submit_order(order)
