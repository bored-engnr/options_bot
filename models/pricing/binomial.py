import numpy as np

class BinomialModel:
    def __init__(self, S, K, T, r, sigma, option_type='call', steps=100):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.steps = steps

    def price(self):
        if self.T <= 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)

        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.r * dt) - d) / (u - d)
        
        # Price tree at maturity
        S_T = self.S * (u ** np.arange(self.steps, -1, -1)) * (d ** np.arange(0, self.steps + 1))
        
        if self.option_type == 'call':
            values = np.maximum(S_T - self.K, 0)
        else:
            values = np.maximum(self.K - S_T, 0)
            
        # Step back through the tree
        for i in range(self.steps - 1, -1, -1):
            values = np.exp(-self.r * dt) * (q * values[:-1] + (1 - q) * values[1:])
            
        return values[0]
