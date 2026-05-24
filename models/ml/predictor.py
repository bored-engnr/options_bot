import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import logging

class MLPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.is_trained = False

    def prepare_features(self, df, for_inference=False):
        """Prepare features for the ML model from historical price data."""
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['volatility'] = df['returns'].rolling(window=20).std()

        if not for_inference:
            # Target is the next day's close price
            df['target'] = df['Close'].shift(-1)
            return df.dropna()
        else:
            return df.tail(1)

    def train(self, historical_df):
        df = self.prepare_features(historical_df)
        if len(df) < 60: # Reduced for backtest compatibility
            logging.warning(f"Not enough data to train ML model. Need 60, got {len(df)}")
            return False

        X = df[['Close', 'sma_10', 'sma_50', 'volatility', 'Volume']]
        y = df['target']

        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict_price(self, current_features):
        if not self.is_trained:
            return None

        # Ensure we have all needed features
        required = ['Close', 'sma_10', 'sma_50', 'volatility', 'Volume']
        if any(col not in current_features or pd.isna(current_features[col]) for col in required):
            return None

        X = np.array([current_features[c] for c in required]).reshape(1, -1)
        return self.model.predict(X)[0]
