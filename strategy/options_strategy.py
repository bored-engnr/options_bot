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
from utils.config import Config
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

class AdaptiveOptionsStrategy(Strategy):
    parameters = {
        "symbol": "SPY",
        "quantity": 100,
        "take_profit_pct": 0.20,
        "stop_loss_pct": 0.15,
        "av_api_key": None,
        "risk_free_rate": 0.05,
        "model_path": "models/ml/trained_model.joblib"
    }

    def initialize(self):
        self.symbol = self.parameters["symbol"]
        self.underlying_asset = Asset(symbol=self.symbol, asset_type="stock")
        self.fetcher = DataFetcher(av_api_key=self.parameters["av_api_key"])
        self.db = OptionsDB()
        self.ml_predictor = MLPredictor(model_path=self.parameters["model_path"])
        self.adaptive_weights = AdaptiveWeighting()
        
        self.sleeptime = "1D" 
        self.last_predictions = {} 
        self.trades_info = [] 

    def on_trading_iteration(self):
        # Margin Check
        if not Config.ALLOW_MARGIN:
            if self.cash < (self.get_last_price(self.underlying_asset) * self.parameters["quantity"]):
                # This is a simple cash check for the underlying. 
                # For options, we'd check the option price * quantity.
                pass

        historical_data = self.fetcher.get_data(self.symbol)
        if historical_data.empty:
            return

        self.db.save_historical_prices(self.symbol, historical_data)
        
        # Always train/update in backtest to simulate learning over time
        if self.broker.name == "backtesting":
            self.ml_predictor.train(historical_data)
        
        try:
            expirations = self.fetcher.fetch_expirations_yfinance(self.symbol)
            if not expirations:
                return
            target_expiry = expirations[0] 
            options_chain = self.fetcher.fetch_options_chain_yfinance(self.symbol, target_expiry)
            if options_chain.empty:
                return
            self.db.save_options_chain(self.symbol, target_expiry, options_chain)
        except Exception:
            return
        
        current_price = self.get_last_price(self.underlying_asset)
        calls = options_chain[options_chain['option_type'] == 'call']
        if calls.empty: return
            
        idx = (calls['strike'] - current_price).abs().idxmin()
        option_data = calls.loc[idx]
        K = option_data['strike']
        market_price = (option_data['bid'] + option_data['ask']) / 2
        if market_price <= 0: market_price = option_data['last_price']
        if market_price <= 0: return 
        
        expiry_dt = datetime.strptime(target_expiry, "%Y-%m-%d")
        current_dt = self.get_datetime()
        if current_dt.tzinfo is not None: current_dt = current_dt.replace(tzinfo=None)
            
        T = (expiry_dt - current_dt).days / 365.0
        if T <= 0: T = 1/365.0 
        
        r = self.parameters["risk_free_rate"]
        sigma = option_data['implied_volatility']
        
        bs_price = BlackScholesModel(current_price, K, T, r, sigma, 'call').price()
        mc_price = MonteCarloModel(current_price, K, T, r, sigma, 'call').price()
        bi_price = BinomialModel(current_price, K, T, r, sigma, 'call').price()
        he_price = HestonModel(current_price, K, T, r, 2.0, 0.04, 0.1, -0.7, sigma**2, 'call').price()
        
        inference_row = self.ml_predictor.prepare_features(historical_data, for_inference=True)
        if not inference_row.empty:
            ml_pred = self.ml_predictor.predict_price(inference_row.iloc[0])
            ml_price = BlackScholesModel(ml_pred, K, T, r, sigma, 'call').price() if ml_pred else bs_price
        else:
            ml_price = bs_price
            
        predicted_prices = [bs_price, mc_price, bi_price, he_price, ml_price]
        if self.symbol in self.last_predictions:
            self.adaptive_weights.update_weights(market_price, self.last_predictions[self.symbol])
        self.last_predictions[self.symbol] = predicted_prices
        
        weighted_price = self.adaptive_weights.get_weighted_price(predicted_prices)
        
        msg = (f"Iteration: {self.get_datetime()} | Underlying: {current_price:.2f} | Strike: {K} | "
               f"Market: {market_price:.4f} | Fair: {weighted_price:.4f} | "
               f"Models: BS={bs_price:.2f}, MC={mc_price:.2f}, BI={bi_price:.2f}, HE={he_price:.2f}, ML={ml_price:.2f}")
        self.log_message(msg)

        pos = self.get_position(self.underlying_asset)
        quantity = pos.quantity if pos else 0
        
        if market_price < weighted_price * 0.97 and quantity == 0:
            # Check Margin before buying
            if not Config.ALLOW_MARGIN and self.cash < (market_price * self.parameters["quantity"]):
                self.log_message("INSUFFICIENT CASH (Margin Disabled)")
                return

            self.log_message(f"OPEN BUY: Price {market_price:.4f} < Fair {weighted_price:.4f}")
            order = self.create_order(self.underlying_asset, self.parameters["quantity"], "buy")
            self.submit_order(order)
            self.trades_info.append({'type': 'buy', 'price': market_price, 'strike': K, 'time': self.get_datetime()})
            
        elif market_price > weighted_price * 1.03 and quantity > 0:
            buy_price = next((t['price'] for t in reversed(self.trades_info) if t['type'] == 'buy'), 0)
            profit = (market_price - buy_price) * self.parameters["quantity"]
            self.log_message(f"CLOSE SELL: Price {market_price:.4f} > Fair {weighted_price:.4f} | PROFIT: {profit:.2f}")
            order = self.create_order(self.underlying_asset, self.parameters["quantity"], "sell")
            self.submit_order(order)
            self.trades_info.append({'type': 'sell', 'price': market_price, 'strike': K, 'time': self.get_datetime(), 'profit': profit})

    def teardown(self):
        # Ensure model is saved at the very end
        self.ml_predictor.save_model()
        self.log_message("Final Model Persisted.")
