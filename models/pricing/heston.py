import numpy as np
from scipy.integrate import quad

class HestonModel:
    def __init__(self, S, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa   # Mean reversion speed
        self.theta = theta   # Long-term variance
        self.sigma = sigma   # Volatility of volatility
        self.rho = rho       # Correlation between stock and volatility
        self.v0 = v0         # Initial variance
        self.option_type = option_type

    def _char_func(self, phi, j):
        # Characteristic function for Heston model
        # Based on Heston (1993)
        a = self.kappa * self.theta
        if j == 1:
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:
            u = -0.5
            b = self.kappa

        d = np.sqrt((self.rho * self.sigma * phi * 1j - b)**2 - self.sigma**2 * (2 * u * phi * 1j - phi**2))
        g = (b - self.rho * self.sigma * phi * 1j + d) / (b - self.rho * self.sigma * phi * 1j - d)
        
        C = self.r * phi * 1j * self.T + a / self.sigma**2 * (
            (b - self.rho * self.sigma * phi * 1j + d) * self.T - 2 * np.log((1 - g * np.exp(d * self.T)) / (1 - g))
        )
        D = (b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2 * (
            (1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T))
        )
        
        return np.exp(C + D * self.v0 + phi * 1j * np.log(self.S))

    def _probability(self, phi, j):
        # Integral part of the probability P_j
        num = np.exp(-phi * 1j * np.log(self.K)) * self._char_func(phi, j)
        return (num / (phi * 1j)).real

    def price(self):
        if self.T <= 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)

        # Integration to find P1 and P2
        P1 = 0.5 + 1/np.pi * quad(self._probability, 0, 100, args=(1,))[0]
        P2 = 0.5 + 1/np.pi * quad(self._probability, 0, 100, args=(2,))[0]
        
        call_price = self.S * P1 - self.K * np.exp(-self.r * self.T) * P2
        
        if self.option_type == 'call':
            return call_price
        else:
            # Put-Call Parity: P = C - S + K*exp(-rT)
            return call_price - self.S + self.K * np.exp(-self.r * self.T)
