import os
from decimal import Decimal

import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.venues import venue

# Backtests run in-process under test, where the tests' stand-in price feeds apply; the server runs
# each in a process of its own (test_jobs.py covers that).
os.environ.setdefault("BACKTEST_ISOLATE", "0")


@pytest.fixture(scope="session")
def instrument():
    return venue("KRAKEN").instrument("BTC", "USD")


@pytest.fixture(scope="session")
def prices():
    return synthetic_ohlcv(days=900, seed=11)


@pytest.fixture(autouse=True)
def _no_swallowed_strategy_errors(capfd, request):
    """The engine catches exceptions raised in strategy callbacks: on_bar and timers it only logs,
    and order and market data handlers it drops without a word, so the strategy reports those itself
    (LongFlatStrategy._reporting). Either way a broken handler could pass a test that checks
    something else, so fail any test that logs one."""
    yield
    out, err = capfd.readouterr()
    if request.node.get_closest_marker("strategy_errors"):  # raised on purpose, and checked by the test
        return
    for line in (out + err).splitlines():
        if ("Python " in line and ("failed" in line or "raised exception" in line)) or "strategy handler " in line or "sleeve tick failed" in line:
            pytest.fail(f"a strategy callback raised inside the engine: {line}")


@pytest.fixture
def maker_on(monkeypatch):
    """Maker-first orders are switched off by default (strategies.base.maker_orders_enabled); the tests of
    the post-only path switch them on."""
    monkeypatch.setenv("SLEEVE_MAKER_ORDERS", "1")


@pytest.fixture
def full_margin(monkeypatch):
    """Every risk profile puts the strategy's whole equity up as margin, so a perpetual sizes at its full
    leverage cap (2x on balanced) and its liquidation price is near enough to test. The default margin
    cap (PM, 5 Oct 2026: 33% on balanced, so 0.66x notional) leaves liquidation out of reach."""
    import dataclasses

    from sleeve_fund import risk

    for name, p in list(risk.PROFILES.items()):
        monkeypatch.setitem(risk.PROFILES, name, dataclasses.replace(p, max_position_pct=1.0))


@pytest.fixture(autouse=True)
def _refusal_notes_in_tmp(monkeypatch, tmp_path_factory):
    """A study row that fails writes a note beside the tear sheets (rescore.refusal_note); never the repository's."""
    from sleeve_fund.research import rescore

    monkeypatch.setattr(rescore, "REFUSED", tmp_path_factory.mktemp("refused"))


@pytest.fixture(autouse=True)
def _open_risk_setup(monkeypatch, request):
    """Paper measures a stopless perp position's open risk from its pair's daily ATR in the history store
    (sleeve_fund.open_risk), which tests don't fill: a calm 2% day unless a test sets its own. A test of the paper
    engine's mechanics with a stopless model above 1x (paper-backtest parity, funding, liquidation) lifts the
    open-risk limit, which would refuse its entries in paper only."""
    if "real_daily_atr" not in request.keywords:
        from sleeve_fund import open_risk

        monkeypatch.setattr(open_risk, "history_atr_pct", lambda venue, pair, now, history=None: 0.02)
    if "no_open_risk_limit" in request.keywords:
        from sleeve_fund import open_risk

        monkeypatch.setattr(open_risk, "LIMIT", float("inf"))
    if "no_restart_safety_stop" in request.keywords:  # QA SG11: the marker, never a bare monkeypatch
        from sleeve_fund.strategies.base import LongFlatStrategy

        monkeypatch.setattr(LongFlatStrategy, "_safety_stop_on_restore", lambda self, book: None)


class _EveryPair(dict):
    """Every pair at one rate: a venue table standing in for the published tiers until they are kept (VS-1)."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def get(self, key, default=None):
        return self.table

    def __contains__(self, key):
        return True

    def __iter__(self):
        return iter(())


@pytest.fixture(autouse=True)
def _vs1_interim_tables(monkeypatch, request):
    """VS-1 (A12): a perp on a pair with no maintenance table is refused. Until the DA's published tables are kept on
    the venue profile, every test runs the perp venue at the one 0.5% rate it ran at before, so nothing else moves; a
    test marked no_maintenance_tables sees the venue as it is."""
    if request.node.get_closest_marker("no_maintenance_tables"):
        return
    from sleeve_fund.margin import flat

    monkeypatch.setattr(venue("BINANCE"), "maintenance_tiers", _EveryPair(flat(0.005)))


@pytest.fixture(autouse=True)
def _price_tick_read(monkeypatch, request):
    """B-36b, B-43: the start check and an instrument read the venue's live contract tick from its public listing; a
    build machine can't reach the venue, so every test reads the kept tick back as the live one (the check passes
    wherever one is kept). A test marked price_tick_live reads it as it is."""
    if request.node.get_closest_marker("price_tick_live"):
        return
    from sleeve_fund import markets, start_check

    monkeypatch.setattr(start_check, "_live_tick", lambda venue, pair: markets.price_tick(venue, pair))
    monkeypatch.setattr(venue("BINANCE"), "live_tick", lambda pair: markets.price_tick("BINANCE", pair))

    def kept_lot(pair):  # B-44: the kept lot read back as the live one; none kept, the contract limits stand
        kept = venue("BINANCE").lot_sizes.get(pair)
        if kept is None:
            raise ValueError("unreachable from a build machine")
        return tuple(Decimal(x) for x in kept)

    monkeypatch.setattr(venue("BINANCE"), "live_lot", kept_lot)
    monkeypatch.setattr(start_check, "_hub_lot", lambda v, pair: kept_lot(pair))  # the hub's contract, as kept


@pytest.fixture(autouse=True)
def _venue_terms_fetched(monkeypatch, request):
    """FEE-LIVE: paper refuses entries, and the start check fails, while an instrument's fetched fee rates or tiers
    are over 72 h old or were never fetched; only the supervisor fetches them. Every other test's store has never
    fetched, so it runs as if they were fresh, and its paper is charged the published VIP 0 (fees.resolve); a test
    marked venue_terms_live sees the rule as it is."""
    if request.node.get_closest_marker("venue_terms_live"):
        return
    from sleeve_fund import venue_terms

    monkeypatch.setattr(venue_terms, "entry_refusal", lambda *a, **k: None)
    monkeypatch.setattr(venue_terms, "live_terms", lambda *a, **k: [])


# TEC-BTWAIT (HoE, Drop B): how long a dashboard backtest request waits for its job before showing the queued-job page.
# Production waits 8 s (BACKTEST_WAIT_SECONDS); a loaded test box can take longer, and a cell expecting the finished
# run then reads the queued page instead (Drop A round, 14 Oct). A test that means the queued page sets its own.
TEST_BACKTEST_WAIT = 300.0


@pytest.fixture(autouse=True)
def _backtest_wait(monkeypatch):
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "BACKTEST_WAIT", TEST_BACKTEST_WAIT)


def pytest_collection_modifyitems(items):
    """Every test that opens the browser carries the browser mark, so CI runs it on the one shard that installs Chromium
    (and `-m "not browser"` everywhere else): FE's browser cells live in their own files, not only in test_browser.py."""
    for item in items:
        if "browser" in getattr(item, "fixturenames", ()) or item.path.name == "test_browser.py":
            item.add_marker(pytest.mark.browser)
