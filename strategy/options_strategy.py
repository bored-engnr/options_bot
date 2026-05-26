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
import logging
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
        self.last_predictions_map = {} # {option_key: [preds]}
        self.trades_info = []

    def on_trading_iteration(self):
        historical_data = self.fetcher.get_data(self.symbol)
        if historical_data.empty: return

        current_dt = self.get_datetime().replace(tzinfo=None) if self.get_datetime().tzinfo else self.get_datetime()
        current_price = self.get_last_price(self.underlying_asset)
        r = self.parameters["risk_free_rate"]

        if self.broker.name == "backtesting" and self.iteration_count % 5 == 0:
            self.ml_predictor.train(historical_data)
        self.iteration_count += 1

        # 1. Fetch Option Chain from DB (Cached)
        chain = self.db.get_chain_for_date(self.symbol, current_dt)

        if chain.empty and Config.USE_SYNTHETIC_PROXY:
            strikes = [round(current_price * m, 2) for m in np.linspace(0.95, 1.05, 5)]
            rows = []
            for s in strikes:
                rows.append({'strike': s, 'option_type': 'call', 'expiration': current_dt + timedelta(days=7), 'bid': 0, 'ask': 0, 'last_price': 0, 'implied_volatility': 0.20})
            chain = pd.DataFrame(rows)

        if chain.empty: return

        # 2. Brute Force Analysis & Weight Update
        best_undervalued_option = None
        max_edge = -999

        inference_row = self.ml_predictor.prepare_features(historical_data, for_inference=True)
        ml_underlying_pred = self.ml_predictor.predict_price(inference_row.iloc[0]) if not inference_row.empty else current_price

        for _, opt in chain.iterrows():
            if opt['option_type'] != 'call': continue

            K = opt['strike']
            opt_expiry = opt['expiration'].replace(tzinfo=None) if hasattr(opt['expiration'], 'replace') else opt['expiration']
            T = max(1/365.0, (opt_expiry - current_dt).days / 365.0)
            sigma = opt['implied_volatility'] if opt['implied_volatility'] > 0 else 0.20

            mkt_price = (opt['bid'] + opt['ask']) / 2
            if mkt_price <= 0:
                mkt_price = BlackScholesModel(current_price, K, T, r, sigma, 'call').price() if Config.USE_SYNTHETIC_PROXY else opt['last_price']
            if mkt_price <= 0: continue

            # Update weights if we have previous prediction for THIS strike/expiry
            opt_key = f"{K}_{opt_expiry}"
            if opt_key in self.last_predictions_map:
                self.adaptive_weights.update_weights(mkt_price, self.last_predictions_map[opt_key])

            # Ensemble
            bs = BlackScholesModel(current_price, K, T, r, sigma, 'call').price()
            mc = MonteCarloModel(current_price, K, T, r, sigma, 'call').price()
            bi = BinomialModel(current_price, K, T, r, sigma, 'call').price()
            he = HestonModel(current_price, K, T, r, 2.0, 0.04, 0.1, -0.7, sigma**2, 'call').price()
            ml = BlackScholesModel(ml_underlying_pred, K, T, r, sigma, 'call').price()

            preds = [bs, mc, bi, he, ml]
            self.last_predictions_map[opt_key] = preds # Store for next iteration

            fair_val = self.adaptive_weights.get_weighted_price(preds)
            edge = (fair_val - mkt_price) / mkt_price if mkt_price > 0 else 0

            if edge > max_edge:
                max_edge = edge
                best_undervalued_option = (opt, fair_val, mkt_price, preds)

        # 3. Decision
        if best_undervalued_option and max_edge > 0.05:
            opt_data, fair_val, mkt_price, preds = best_undervalued_option
            asset = Asset(symbol=self.symbol, asset_type="option", expiration=opt_data['expiration'], strike=opt_data['strike'], right="call")

            print(f"CALL {self.symbol}@{opt_data['strike']:.2f} for {mkt_price:.2f} at {current_dt.strftime('%Y-%m-%d %H:%M')} (Edge: {max_edge:.2%})")

            pos = self.get_position(asset)
            if (pos is None or pos.quantity == 0) and (Config.ALLOW_MARGIN or self.cash > mkt_price * 100):
                order = self.create_order(asset, self.parameters["quantity"], "buy")
                self.submit_order(order)
                self.trades_info.append({'asset': asset, 'type': 'buy', 'price': mkt_price, 'time': current_dt})

        # 4. Sell Logic & Profit Reporting
        positions = self.get_positions()
        for p in positions:
            if p.asset.asset_type == "option":
                p_expiry = p.asset.expiration.replace(tzinfo=None) if p.asset.expiration.tzinfo else p.asset.expiration
                if (p_expiry - current_dt).days <= 0:
                    # Estimate sell price at expiration (intrinsic value)
                    sell_price = max(0, current_price - p.asset.strike)
                    buy_info = next((t for t in reversed(self.trades_info) if t['asset'] == p.asset and t['type'] == 'buy'), None)
                    if buy_info:
                        profit = (sell_price - buy_info['price']) * 100 * p.quantity
                        self.log_message(f"PROFIT REPORT: {p.asset} | BUY: {buy_info['price']:.2f} | SELL: {sell_price:.2f} | NET: {profit:.2f}")
                        self.trades_info.append({'asset': p.asset, 'type': 'sell', 'price': sell_price, 'time': current_dt, 'profit': profit})

                    order = self.create_order(p.asset, p.quantity, "sell")
                    self.submit_order(order)

    def teardown(self):
        self.ml_predictor.save_model()
        self.adaptive_weights.save_state()
        total_profit = sum(t.get('profit', 0) for t in self.trades_info)
        self.log_message(f"TOTAL BACKTEST PROFIT: {total_profit:.2f}")
