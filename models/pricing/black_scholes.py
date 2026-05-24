import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S          # Underlying price
        self.K = K          # Strike price
        self.T = T          # Time to expiration (years)
        self.r = r          # Risk-free rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type

    def _d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        if self.T <= 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)

        d1 = self._d1()
        d2 = self._d2()
        if self.option_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def delta(self):
        if self.T <= 0: return 0
        d1 = self._d1()
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self):
        if self.T <= 0: return 0
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        if self.T <= 0: return 0
        d1 = self._d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self):
        if self.T <= 0: return 0
        d1 = self._d1()
        d2 = self._d2()
        if self.option_type == 'call':
            res = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            res = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return res

    def rho(self):
        if self.T <= 0: return 0
        d2 = self._d2()
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
