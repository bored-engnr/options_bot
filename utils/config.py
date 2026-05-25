import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Tradier Credentials
    TRADIER_ACCOUNT_NUMBER = os.getenv("TRADIER_ACCOUNT_NUMBER")
    TRADIER_ACCESS_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")
    TRADIER_IS_PAPER = os.getenv("TRADIER_IS_PAPER", "True").lower() == "true"

    # Alpha Vantage (Free subscription)
    AV_API_KEY = os.getenv("AV_API_KEY")

    # Strategy Parameters
    DEFAULT_SYMBOL = "SPY"
    QUANTITY = 100
    RISK_FREE_RATE = 0.05

    # Global Margin Control
    ALLOW_MARGIN = False
