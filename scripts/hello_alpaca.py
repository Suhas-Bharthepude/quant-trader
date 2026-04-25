# Smoke test for Alpaca paper trading connection.
# Loads credentials from .env and prints key account fields.

import os
import sys

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    print("Error: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env")
    sys.exit(1)

try:
    client = TradingClient(API_KEY, API_SECRET, paper=True)
    account = client.get_account()

    print("Alpaca Paper Account")
    print("-" * 30)
    print(f"Status:          {account.status}")
    print(f"Buying power:    ${float(account.buying_power):,.2f}")
    print(f"Cash:            ${float(account.cash):,.2f}")
    print(f"Portfolio value: ${float(account.portfolio_value):,.2f}")

except Exception as e:
    print(f"Error connecting to Alpaca: {e}")
    sys.exit(1)
