import numpy as np

class AdaptiveWeighting:
    def __init__(self, initial_weights=None):
        if initial_weights is None:
            self.weights = np.array([0.20, 0.15, 0.15, 0.15, 0.35])
        else:
            self.weights = np.array(initial_weights)

        self.model_names = ['Black-Scholes', 'Monte Carlo', 'Binomial', 'Heston', 'Machine Learning']
        self.errors = [[] for _ in range(len(self.weights))]
        self.learning_rate = 0.05

    def update_weights(self, actual_price, predicted_prices):
        """
        actual_price: The actual market price observed.
        predicted_prices: List of prices predicted by the 5 models.
        """
        predicted_prices = np.array(predicted_prices)

        # Calculate absolute percentage errors
        current_errors = np.abs(predicted_prices - actual_price) / actual_price

        # Update weights based on inverse error (simple heuristic)
        # We use an Exponential Moving Average approach for weights

        # 1. Calculate performance scores (inverse of error)
        # Adding a small epsilon to avoid division by zero
        scores = 1.0 / (current_errors + 1e-6)

        # 2. Normalize scores to sum to 1
        new_weights = scores / np.sum(scores)

        # 3. Smooth the weight transition
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * new_weights

        # 4. Final normalization just in case
        self.weights = self.weights / np.sum(self.weights)

        return self.weights

    def get_weighted_price(self, predicted_prices):
        return np.dot(self.weights, predicted_prices)

    def get_weights_dict(self):
        return dict(zip(self.model_names, self.weights))
