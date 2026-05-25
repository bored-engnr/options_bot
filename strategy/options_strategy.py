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
import os
from datetime import datetime, timedelta

class AdaptiveOptionsStrategy(Strategy):
    parameters = {
        "symbol": "SPY",
        "quantity": 1,
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

        model_dir = os.path.dirname(self.parameters["model_path"])
        weights_path = os.path.join(model_dir, f"{self.symbol}_weights.joblib")

        self.ml_predictor = MLPredictor(model_path=self.parameters["model_path"])
        self.adaptive_weights = AdaptiveWeighting(persistence_path=weights_path)

        self.sleeptime = "1D"
        self.iteration_count = 0
        self.last_predictions = {}
        self.trades_info = []

    def on_trading_iteration(self):
        historical_data = self.fetcher.get_data(self.symbol)
        if historical_data.empty: return

        # Use Alpha Vantage past trades (intraday) if available to refine volatility
        if self.parameters["av_api_key"]:
            intraday = self.fetcher.fetch_past_trades_alpha_vantage(self.symbol)
            if not intraday.empty:
                # Better simulate operations: entry check based on intraday trend
                last_5 = intraday.head(5)
                if last_5['Close'].iloc[0] < last_5['Close'].iloc[-1]:
                    # Price trending down intraday, maybe wait
                    pass

        if self.broker.name == "backtesting" and self.iteration_count % 5 == 0:
            self.ml_predictor.train(historical_data)
        self.iteration_count += 1

        current_price = self.get_last_price(self.underlying_asset)
        current_dt = self.get_datetime().replace(tzinfo=None) if self.get_datetime().tzinfo else self.get_datetime()

        days_to_friday = (4 - current_dt.weekday()) % 7
        target_expiry_dt = current_dt + timedelta(days=days_to_friday)
        target_expiry = target_expiry_dt.strftime("%Y-%m-%d")

        K = round(current_price)
        T = max(1/365.0, (target_expiry_dt - current_dt).days / 365.0)
        r = self.parameters["risk_free_rate"]

        # Calculate historical volatility from recent returns
        returns = historical_data['Close'].pct_change().dropna()
        sigma = returns.tail(30).std() * np.sqrt(252) if len(returns) >= 30 else 0.20

        market_price = None
        if self.broker.name != "backtesting" or not Config.USE_SYNTHETIC_PROXY:
            try:
                chain = self.fetcher.fetch_options_chain_yfinance(self.symbol, target_expiry)
                if not chain.empty:
                    opt_row = chain[chain['strike'] == K].iloc[0]
                    market_price = (opt_row['bid'] + opt_row['ask']) / 2
                    if market_price <= 0: market_price = opt_row['last_price']
            except: pass

        if market_price is None:
            if Config.USE_SYNTHETIC_PROXY:
                market_price = BlackScholesModel(current_price, K, T, r, sigma, 'call').price()
            else:
                return

        # Models
        bs_price = BlackScholesModel(current_price, K, T, r, sigma, 'call').price()
        mc_price = MonteCarloModel(current_price, K, T, r, sigma, 'call').price()
        bi_price = BinomialModel(current_price, K, T, r, sigma, 'call').price()
        he_price = HestonModel(current_price, K, T, r, 2.0, 0.04, 0.1, -0.7, sigma**2, 'call').price()

        inference_row = self.ml_predictor.prepare_features(historical_data, for_inference=True)
        ml_pred = self.ml_predictor.predict_price(inference_row.iloc[0]) if not inference_row.empty else None
        ml_price = BlackScholesModel(ml_pred, K, T, r, sigma, 'call').price() if ml_pred else bs_price

        predicted_prices = [bs_price, mc_price, bi_price, he_price, ml_price]
        if self.symbol in self.last_predictions:
            self.adaptive_weights.update_weights(market_price, self.last_predictions[self.symbol])
        self.last_predictions[self.symbol] = predicted_prices
        weighted_price = self.adaptive_weights.get_weighted_price(predicted_prices)

        option_asset = Asset(symbol=self.symbol, asset_type="option", expiration=target_expiry_dt, strike=K, right="call")

        print(f"CALL {self.symbol}@{K:.2f} for {market_price:.2f} at {current_dt.strftime('%Y-%m-%d %H:%M')}")

        pos = self.get_position(option_asset)
        quantity = pos.quantity if pos else 0

        if market_price < weighted_price * 0.95 and quantity == 0:
            if not Config.ALLOW_MARGIN and self.cash < (market_price * 100 * self.parameters["quantity"]):
                return
            order = self.create_order(option_asset, self.parameters["quantity"], "buy")
            self.submit_order(order)
            self.trades_info.append({'asset': option_asset, 'type': 'buy', 'price': market_price, 'time': current_dt})

        elif (market_price > weighted_price * 1.05 or (target_expiry_dt - current_dt).days < 1) and quantity > 0:
            buy_price = next((t['price'] for t in reversed(self.trades_info) if t['asset'] == option_asset and t['type'] == 'buy'), 0)
            profit = (market_price - buy_price) * 100 * self.parameters["quantity"]
            order = self.create_order(option_asset, self.parameters["quantity"], "sell")
            self.submit_order(order)
            self.trades_info.append({'asset': option_asset, 'type': 'sell', 'price': market_price, 'time': current_dt, 'profit': profit})

    def teardown(self):
        self.ml_predictor.save_model()
        self.adaptive_weights.save_state()
