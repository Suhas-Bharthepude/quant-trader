# Smoke test for Alpaca paper trading connection.
# Loads credentials from .env via AlpacaBroker.from_env() and prints account fields.

import sys

# AlpacaBroker is the concrete broker that wraps alpaca-py.
# from_env() reads .env, constructs the client, and enforces paper=True.
from src.brokers.alpaca_broker import AlpacaBroker

try:
    # from_env() calls load_dotenv() internally and raises EnvironmentError
    # with a clear message if either API key is missing from .env.
    broker = AlpacaBroker.from_env()
    # verify_paper_account() raises RuntimeError if is_paper is False.
    broker.verify_paper_account()
    # get_account() returns our AccountSnapshot dataclass, not an alpaca object.
    account = broker.get_account()

    print("Alpaca Paper Account")
    print("-" * 30)
    print(f"Account number:  {account.account_number}")
    print(f"Buying power:    ${account.buying_power:,.2f}")
    print(f"Cash:            ${account.cash:,.2f}")
    print(f"Portfolio value: ${account.portfolio_value:,.2f}")

except Exception as e:
    # Catches EnvironmentError (missing keys) and RuntimeError (live account).
    print(f"Error: {e}")
    sys.exit(1)