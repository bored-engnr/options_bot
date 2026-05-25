import numpy as np

class MonteCarloModel:
    def __init__(self, S, K, T, r, sigma, option_type='call', simulations=10000):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.simulations = simulations

    def price(self):
        if self.T <= 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)
        
        # Standard Geometric Brownian Motion paths
        z = np.random.standard_normal(self.simulations)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * z)
        
        if self.option_type == 'call':
            payoff = np.maximum(ST - self.K, 0)
        else:
            payoff = np.maximum(self.K - ST, 0)
            
        return np.exp(-self.r * self.T) * np.mean(payoff)
