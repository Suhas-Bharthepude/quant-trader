# tests/test_universe.py

"""
Unit tests for src/data/universe.py.

These tests are pure unit tests — no network, no DuckDB, no filesystem state
beyond the tmp_path fixture that pytest provides and cleans up automatically.
All fake configs are written with yaml.safe_dump so the serialised YAML always
matches the exact structure that load_universe expects.
"""

# pathlib.Path lets us construct the config path from tmp_path cleanly.
from pathlib import Path

# pytest is the test runner; pytest.raises is the context manager for
# asserting that a specific exception type is raised.
import pytest

# yaml.safe_dump serialises the fake config dict to a YAML file — using
# safe_dump ensures the output is identical to what refresh_sp500_universe.py
# writes, so our tests exercise the real round-trip format.
import yaml

# The function under test — load_universe reads a YAML config and returns a
# list of ticker strings for the named universe.
from src.data.universe import load_universe


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A reusable config structure used by most tests.  Both universes have the
# full schema (description + tickers) so load_universe never hits a KeyError
# on a missing "description" field.
TWO_UNIVERSE_CONFIG: dict = {
    "universes": {
        "alpha": {
            "description": "Test universe alpha",  # required field in the schema
            "tickers": ["A", "B", "C"],            # three tickers to verify list length
        },
        "beta": {
            "description": "Test universe beta",   # required field in the schema
            "tickers": ["X", "Y"],                 # two tickers — different from alpha
        },
    }
}


def write_config(tmp_path: Path, config: dict) -> str:
    """Write a config dict to a temp YAML file and return its path as a string."""
    # Place the file inside tmp_path so pytest deletes it when the session ends.
    config_file: Path = tmp_path / "universe.yaml"

    # safe_dump produces block-style YAML identical to what yaml.safe_dump
    # writes in refresh_sp500_universe.py — this exercises the real format.
    with config_file.open("w") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False)

    # load_universe accepts a str path, not a Path object, so we convert here.
    return str(config_file)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_universe_returns_tickers(tmp_path: Path) -> None:
    """load_universe returns the correct ticker list for each named universe."""
    # Write the shared two-universe config to a temp file.
    config_path: str = write_config(tmp_path, TWO_UNIVERSE_CONFIG)

    # Alpha should return exactly the three tickers defined above.
    result_alpha: list[str] = load_universe("alpha", config_path=config_path)
    assert result_alpha == ["A", "B", "C"]  # order must be preserved from YAML

    # Beta should return its two tickers independently of the alpha lookup.
    result_beta: list[str] = load_universe("beta", config_path=config_path)
    assert result_beta == ["X", "Y"]  # separate call — no shared state


def test_load_universe_raises_on_unknown_name(tmp_path: Path) -> None:
    """load_universe raises KeyError with a helpful message for unknown names."""
    # Same two-universe config — "nonexistent" is intentionally absent.
    config_path: str = write_config(tmp_path, TWO_UNIVERSE_CONFIG)

    # pytest.raises captures the exception so we can inspect its message.
    with pytest.raises(KeyError) as exc_info:
        load_universe("nonexistent", config_path=config_path)

    # The error message should list the available universe names so the user
    # knows what valid options exist without opening the YAML manually.
    # str(exc_info.value) includes the repr quotes around the message string.
    error_message: str = str(exc_info.value)
    assert "alpha" in error_message  # both defined names must appear
    assert "beta" in error_message


def test_load_universe_raises_on_empty_universe(tmp_path: Path) -> None:
    """load_universe raises ValueError when the tickers list is empty."""
    # Config with a universe that has the correct schema but an empty ticker list.
    # This mirrors the sp500 universe before refresh_sp500_universe.py has run.
    empty_config: dict = {
        "universes": {
            "empty": {
                "description": "Universe with no tickers yet",  # valid schema
                "tickers": [],  # empty list — should trigger ValueError
            }
        }
    }

    config_path: str = write_config(tmp_path, empty_config)

    # An empty universe would silently produce no work downstream, so
    # load_universe is expected to raise rather than return [].
    with pytest.raises(ValueError):
        load_universe("empty", config_path=config_path)
