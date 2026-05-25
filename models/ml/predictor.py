import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import logging
import joblib
import os

class MLPredictor:
    def __init__(self, model_path="models/ml/trained_model.joblib"):
        self.model_path = os.path.abspath(model_path)
        self.model = RandomForestRegressor(n_estimators=100)
        self.is_trained = False
        self.feature_names = ['Close', 'sma_10', 'sma_50', 'volatility', 'Volume']
        self.load_model()

    def prepare_features(self, df, for_inference=False):
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['volatility'] = df['returns'].rolling(window=20).std()

        if not for_inference:
            df['target'] = df['Close'].shift(-1)
            return df.dropna()
        else:
            return df.tail(1)

    def train(self, historical_df):
        df = self.prepare_features(historical_df)
        if len(df) < 60:
            return False

        X = df[self.feature_names]
        y = df['target']

        self.model.fit(X, y)
        self.is_trained = True
        self.save_model()
        return True

    def predict_price(self, current_features):
        if not self.is_trained:
            return None

        # Fix UserWarning by using a DataFrame with feature names
        X = pd.DataFrame([current_features[self.feature_names]])
        return self.model.predict(X)[0]

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logging.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logging.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                self.is_trained = False
