import unittest
import numpy as np
from models.pricing.black_scholes import BlackScholesModel
from models.pricing.monte_carlo import MonteCarloModel
from models.pricing.binomial import BinomialModel
from models.pricing.heston import HestonModel
from models.ml.adaptive_weights import AdaptiveWeighting

class TestModels(unittest.TestCase):
    def setUp(self):
        self.S = 100
        self.K = 100
        self.T = 1/12 # 1 month
        self.r = 0.05
        self.sigma = 0.2
        
    def test_black_scholes(self):
        bs = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma, 'call')
        price = bs.price()
        self.assertGreater(price, 0)
        self.assertLess(price, self.S)
        print(f"Black-Scholes Price: {price}")

    def test_monte_carlo(self):
        mc = MonteCarloModel(self.S, self.K, self.T, self.r, self.sigma, 'call', simulations=1000)
        price = mc.price()
        self.assertGreater(price, 0)
        print(f"Monte Carlo Price: {price}")

    def test_binomial(self):
        bm = BinomialModel(self.S, self.K, self.T, self.r, self.sigma, 'call', steps=50)
        price = bm.price()
        self.assertGreater(price, 0)
        print(f"Binomial Price: {price}")

    def test_heston(self):
        # kappa, theta, sigma, rho, v0
        hm = HestonModel(self.S, self.K, self.T, self.r, 2.0, 0.04, 0.1, -0.7, 0.04, 'call')
        price = hm.price()
        self.assertGreater(price, 0)
        print(f"Heston Price: {price}")

    def test_adaptive_weighting(self):
        aw = AdaptiveWeighting()
        predicted = [2.5, 2.6, 2.4, 2.55, 2.7]
        weighted_before = aw.get_weighted_price(predicted)
        
        actual = 2.65
        new_weights = aw.update_weights(actual, predicted)
        weighted_after = aw.get_weighted_price(predicted)
        
        self.assertEqual(len(new_weights), 5)
        self.assertAlmostEqual(sum(new_weights), 1.0)
        print(f"Weighted Price Before: {weighted_before}, After: {weighted_after}")

if __name__ == '__main__':
    unittest.main()
