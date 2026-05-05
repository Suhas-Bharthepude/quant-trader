# refresh_sp500_universe.py — maintenance script to sync the sp500 universe in
# config/universe.yaml with the current S&P 500 constituents from Wikipedia.
#
# This is NOT part of the daily trading pipeline.  Run it manually once at
# setup, then again every few months when the index adds or removes companies.
# It is safe to run repeatedly — it overwrites the tickers list in place and
# leaves all other config untouched.
#
# Usage: uv run python scripts/refresh_sp500_universe.py

import io             # io.StringIO wraps a string as a file-like object for pandas
import logging        # timestamps and severity levels to stdout
import sys            # sys.exit() with specific exit codes
import urllib.request  # low-level HTTP so we can set a custom User-Agent header

import pandas as pd  # pandas.read_html parses the HTML string into DataFrames
import yaml          # read and write config/universe.yaml in block style

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Wikipedia page that lists current S&P 500 constituents.
# pandas.read_html fetches the HTML and parses all <table> elements — the
# first table (index 0) is the constituents table with a "Symbol" column.
WIKIPEDIA_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Path to the universe config relative to the repo root (where uv run runs from).
CONFIG_PATH: str = "config/universe.yaml"

# ---------------------------------------------------------------------------
# LOGGING — same pattern as other scripts in this repo.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # logger scoped to this module


def fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 ticker symbols from Wikipedia.

    Returns a sorted list of cleaned ticker strings.
    Raises on any network or parse failure so the caller can handle exit codes.
    """
    log.info("Fetching S&P 500 constituents from Wikipedia …")

    # Wikipedia rejects the default Python urllib User-Agent ("Python-urllib/3.x")
    # with HTTP 403.  We send a real browser UA so the request looks legitimate.
    req = urllib.request.Request(
        WIKIPEDIA_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )

    # Fetch the page and decode the response body to a plain HTML string.
    # timeout=30 prevents the script from hanging indefinitely on a slow network.
    with urllib.request.urlopen(req, timeout=30) as response:
        html: str = response.read().decode("utf-8")  # Wikipedia serves UTF-8

    # Wrap the HTML string in a file-like object before passing to read_html.
    # Passing a raw string causes pandas to print the HTML to stdout as a side
    # effect; io.StringIO suppresses that by making it look like an open file.
    tables: list[pd.DataFrame] = pd.read_html(io.StringIO(html))

    # The first table on the page is the constituents table.
    constituents: pd.DataFrame = tables[0]

    log.info("Parsed %d rows from the constituents table.", len(constituents))

    # Extract the "Symbol" column — each cell is a ticker string like "AAPL".
    raw_symbols: pd.Series = constituents["Symbol"]

    # Clean each ticker:
    #   1. str.strip()    — remove any leading/trailing whitespace the HTML may carry
    #   2. .replace(".", "-") — Yahoo Finance uses BRK-B; Wikipedia uses BRK.B
    cleaned: list[str] = [
        str(symbol).strip().replace(".", "-")
        for symbol in raw_symbols
    ]

    # Sort alphabetically so diffs in version control are stable and readable.
    cleaned.sort()

    log.info("Cleaned and sorted %d ticker symbols.", len(cleaned))
    return cleaned


def update_config(tickers: list[str]) -> int:
    """Write tickers into universes['sp500']['tickers'] in the YAML config.

    Returns the previous ticker count so the caller can print the summary.
    """
    # Read the existing config so we only touch the sp500 tickers list and
    # leave every other key (description, other universes, etc.) intact.
    with open(CONFIG_PATH, "r") as fh:
        config = yaml.safe_load(fh)  # parse into a plain Python dict

    # Record the old count before overwriting so we can report "was M".
    previous_tickers: list[str] = config["universes"]["sp500"]["tickers"]
    previous_count: int = len(previous_tickers)  # may be 0 on first run

    # Overwrite only the tickers list; every other field stays as-is.
    config["universes"]["sp500"]["tickers"] = tickers

    # Write back in block style (default_flow_style=False) so the YAML remains
    # human-readable and produces clean diffs instead of a single long line.
    with open(CONFIG_PATH, "w") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False)

    log.info("Wrote updated config to %s.", CONFIG_PATH)
    return previous_count


def main() -> None:
    """Entry point — fetch, update, summarise.  Exits 1 on any fetch failure."""
    # Attempt the Wikipedia fetch inside a try/except so we can give a clean
    # error message and a non-zero exit code instead of a raw traceback.
    try:
        tickers: list[str] = fetch_sp500_tickers()
    except Exception as exc:  # noqa: BLE001 — intentional broad catch at boundary
        # Print to stderr so the error stands out from INFO logs on stdout.
        print(f"ERROR: failed to fetch S&P 500 constituents: {exc}", file=sys.stderr)
        sys.exit(1)  # exit code 1 signals failure to any calling shell script

    # Update the YAML config and get back the previous count for the summary.
    previous_count: int = update_config(tickers)

    # Human-readable summary so a manual run gives immediate confirmation.
    print(f"Refreshed sp500 universe: {len(tickers)} tickers (was {previous_count})")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
