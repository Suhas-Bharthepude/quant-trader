# tests/test_alpaca_intraday.py

"""
Tests for the ADDITIVE AlpacaBroker.get_minute_bars method.

These live in a SEPARATE file (not tests/test_alpaca_broker.py) precisely to
demonstrate the change is purely additive: the existing broker test file and its
_FakeDataClient stay untouched.  This file brings its own extended fake data
client that implements get_stock_bars, since the existing no-op stub deliberately
does not.

No network: the Alpaca SDK clients are monkeypatched with fakes, mirroring the
hermetic pattern already used in tests/test_alpaca_broker.py.
"""

# datetime/timezone build the tz-aware UTC timestamps the mapper normalises.
from datetime import datetime, timezone

# The class under test, imported through the package like the sibling test file.
from src.brokers.alpaca_broker import AlpacaBroker


# ---------------------------------------------------------------------------
# Minimal fakes — a stub Bar and a data client that serves a fixed BarSet.
# ---------------------------------------------------------------------------


class _StubBar:
    """Duck-typed stand-in for alpaca.data.models.Bar (only fields we read)."""

    def __init__(self, timestamp, open_, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class _StubBarSet:
    """Stand-in for alpaca.data.models.BarSet: exposes .data as {symbol: [Bar]}."""

    def __init__(self, data):
        self.data = data


class _FakeDataClientWithBars:
    """Fake StockHistoricalDataClient that returns a pre-seeded BarSet.

    A class attribute holds the BarSet to serve, so each test seeds it before
    constructing the broker.  get_stock_bars records the request it received so
    tests can assert the symbol/timeframe were wired correctly.
    """

    barset = _StubBarSet({})
    captured_request = None

    def __init__(self, api_key: str, api_secret: str) -> None:
        # No network, no state beyond the class attributes above.
        pass

    def get_stock_bars(self, request):
        type(self).captured_request = request
        return type(self).barset


class _FakeTradingClient:
    """No-op TradingClient stub — get_minute_bars never touches trading."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        pass


def _build_broker_with_bars(monkeypatch, barset: "_StubBarSet") -> "AlpacaBroker":
    """Patch the SDK client names in the broker module, then build a faked broker."""
    _FakeDataClientWithBars.barset = barset
    _FakeDataClientWithBars.captured_request = None
    monkeypatch.setattr("src.brokers.alpaca_broker.TradingClient", _FakeTradingClient)
    monkeypatch.setattr(
        "src.brokers.alpaca_broker.StockHistoricalDataClient", _FakeDataClientWithBars
    )
    return AlpacaBroker(api_key="fake", api_secret="fake", paper=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_minute_bars_maps_fields_and_provenance(monkeypatch) -> None:
    """Alpaca Bars map to OHLCVBar with timeframe="1m", source="alpaca", adj_close==close."""
    ts = datetime(2024, 1, 2, 14, 31, tzinfo=timezone.utc)
    barset = _StubBarSet(
        {"SOXL": [_StubBar(ts, 30.0, 30.5, 29.8, 30.2, 12345)]}
    )
    broker = _build_broker_with_bars(monkeypatch, barset)

    bars = broker.get_minute_bars("SOXL", "2024-01-02", "2024-01-03")

    assert len(bars) == 1
    bar = bars[0]
    assert bar.symbol == "SOXL"
    assert bar.timestamp == ts
    assert (bar.open, bar.high, bar.low, bar.close) == (30.0, 30.5, 29.8, 30.2)
    assert bar.volume == 12345
    assert bar.timeframe == "1m"
    assert bar.source == "alpaca"
    # adj_close mirrors close for minute bars (no intraday split/dividend).
    assert bar.adj_close == bar.close


def test_get_minute_bars_sorts_ascending(monkeypatch) -> None:
    """Bars are returned sorted by timestamp ascending, regardless of input order."""
    early = datetime(2024, 1, 2, 14, 31, tzinfo=timezone.utc)
    late = datetime(2024, 1, 2, 14, 32, tzinfo=timezone.utc)
    # Seed OUT of order (late first) to prove the defensive sort.
    barset = _StubBarSet(
        {
            "SPY": [
                _StubBar(late, 470.0, 470.2, 469.9, 470.1, 100),
                _StubBar(early, 469.5, 469.8, 469.4, 469.7, 200),
            ]
        }
    )
    broker = _build_broker_with_bars(monkeypatch, barset)

    bars = broker.get_minute_bars("SPY", "2024-01-02", "2024-01-03")

    assert [b.timestamp for b in bars] == [early, late]


def test_get_minute_bars_absent_symbol_returns_empty(monkeypatch) -> None:
    """A symbol missing from the BarSet yields an empty list, not a KeyError."""
    barset = _StubBarSet({})  # no keys at all
    broker = _build_broker_with_bars(monkeypatch, barset)

    bars = broker.get_minute_bars("QQQ", "2024-01-02", "2024-01-03")

    assert bars == []


def test_get_minute_bars_wires_symbol_and_minute_timeframe(monkeypatch) -> None:
    """The request sent to Alpaca carries our symbol and the 1-minute timeframe."""
    barset = _StubBarSet({"SOXL": []})
    broker = _build_broker_with_bars(monkeypatch, barset)

    broker.get_minute_bars("SOXL", "2024-01-02", "2024-01-03")

    req = _FakeDataClientWithBars.captured_request
    assert req.symbol_or_symbols == "SOXL"
    # TimeFrame instances lack value-equality, so compare the string form ("1Min").
    assert str(req.timeframe) == "1Min"


def test_naive_timestamp_is_normalised_to_utc(monkeypatch) -> None:
    """A naive Alpaca timestamp is assumed UTC and made tz-aware."""
    naive = datetime(2024, 1, 2, 14, 31)  # no tzinfo
    barset = _StubBarSet({"SOXL": [_StubBar(naive, 30.0, 30.1, 29.9, 30.0, 10)]})
    broker = _build_broker_with_bars(monkeypatch, barset)

    bars = broker.get_minute_bars("SOXL", "2024-01-02", "2024-01-03")

    assert bars[0].timestamp.tzinfo is timezone.utc
