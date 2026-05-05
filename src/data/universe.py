"""Universe loader — single source of truth for which tickers are in scope.

Anything that needs to know "what stocks are in our trading universe" loads
from here.  The tickers listed in config/universe.yaml are authoritative;
this module never touches the network.  The S&P 500 list is refreshed
periodically by an external script (Day 11 Step 5) which writes back to the
same YAML, so callers always see the current snapshot without needing to know
how or when it was updated.
"""

import yaml  # PyYAML — parses the YAML config file

def load_universe(name: str, config_path: str = "config/universe.yaml") -> list[str]:
    """Load a named universe of tickers from the YAML config.

    Raises KeyError if the universe name isn't found.
    Raises FileNotFoundError if the config file is missing.
    Raises ValueError if the universe has no tickers.
    Returns list[str] of ticker symbols, e.g. ["SPY", "QQQ", ...].
    """
    # Open the config file — let FileNotFoundError propagate naturally so the
    # caller gets a clear OS-level message with the path that was missing.
    with open(config_path, "r") as fh:
        # safe_load parses YAML without executing arbitrary Python objects.
        config = yaml.safe_load(fh)

    # Top-level key in the YAML is "universes"; all named universes live under it.
    universes = config["universes"]

    # Check that the requested name exists before trying to index it so we can
    # give a helpful error that lists what IS available.
    if name not in universes:
        available = ", ".join(sorted(universes.keys()))  # sorted for stable output
        raise KeyError(
            f"Universe '{name}' not found in {config_path}. "
            f"Available universes: {available}"
        )

    # Pull the tickers list for the requested universe.
    tickers: list[str] = universes[name]["tickers"]

    # An empty universe would silently produce no work downstream — that is almost
    # certainly a misconfiguration (e.g. sp500 before the refresh script has run),
    # so we raise rather than return an empty list.
    if not tickers:
        raise ValueError(
            f"Universe '{name}' in {config_path} has no tickers. "
            "Run the appropriate refresh script before loading this universe."
        )

    # Cast each element to str in case the YAML parser produced ints for
    # all-numeric symbols (edge case, but safer to guarantee the return type).
    return [str(ticker) for ticker in tickers]


def list_universes(config_path: str = "config/universe.yaml") -> dict[str, str]:
    """Return {name: description} for every universe defined in the config."""
    # Open and parse the config the same way load_universe does.
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    # Build a plain {name: description} dict — callers just want a summary,
    # not the full tickers list, so we only surface the description field.
    return {
        name: details["description"]  # every universe entry must have a description
        for name, details in config["universes"].items()
    }
