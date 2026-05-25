import numpy as np
import joblib
import os

class AdaptiveWeighting:
    def __init__(self, initial_weights=None, persistence_path=None):
        self.persistence_path = persistence_path
        if persistence_path and os.path.exists(persistence_path):
            self.load_state()
        else:
            if initial_weights is None:
                self.weights = np.array([0.20, 0.15, 0.15, 0.15, 0.35])
            else:
                self.weights = np.array(initial_weights)
        
        self.model_names = ['Black-Scholes', 'Monte Carlo', 'Binomial', 'Heston', 'Machine Learning']
        self.learning_rate = 0.05

    def update_weights(self, actual_price, predicted_prices):
        predicted_prices = np.array(predicted_prices)
        current_errors = np.abs(predicted_prices - actual_price) / (actual_price + 1e-6)
        scores = 1.0 / (current_errors + 1e-6)
        new_weights = scores / np.sum(scores)
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * new_weights
        self.weights = self.weights / np.sum(self.weights)
        
        if self.persistence_path:
            self.save_state()
            
        return self.weights

    def get_weighted_price(self, predicted_prices):
        return np.dot(self.weights, predicted_prices)

    def get_weights_dict(self):
        return dict(zip(self.model_names, self.weights))

    def save_state(self):
        if self.persistence_path:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            joblib.dump(self.weights, self.persistence_path)

    def load_state(self):
        if self.persistence_path and os.path.exists(self.persistence_path):
            self.weights = joblib.load(self.persistence_path)
