"""A9 end to end through the paper engine: inside the fund, a new Strategy enters at full size with the ramp OFF and
at the venue's minimum with it ON, and its stop still closes the whole position. Kept apart from test_min_size_ramp:
the exposure-gate harness's offline fixtures are autouse."""
from __future__ import annotations

import os

from decimal import Decimal as D

import pandas as pd
import pytest
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund import ramp
from sleeve_fund.store import Store


def _new_store(tmp_path, name):
    """Both engines: Postgres when TEST_DATABASE_URL is set (CI), else SQLite."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine, portfolio_gate=True)
    return Store(f"sqlite:///{tmp_path}/{name}.db", portfolio_gate=True)


def _e2e(tmp_path, monkeypatch, on: bool, minutes=6, price=None):
    import test_exposure_gate_xfails as h
    from test_portfolio_paper_entry import END

    from sleeve_fund.supervisor import Supervisor

    if on:
        monkeypatch.setenv(ramp.ENV, "1")
    else:
        monkeypatch.delenv(ramp.ENV, raising=False)
    store = _new_store(tmp_path, "e2e")
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params={**h.PERP, "stop_loss": h.STOP})
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 20_000, actor="pm", reason="QA: the fund's capital", rebase=True,
                                      now=before), before)
    Supervisor(store).mark_portfolio(END)
    kw = {} if price is None else {"price": price}
    plan = h.Plan(t0=h.STD_T0, minutes=minutes, windows=[(h.M(h.STD_T0, 2), h.M(h.STD_T0, minutes), 1)],
                  balance=200_000.0, **kw)
    h.run(tmp_path, store, plan, monkeypatch)
    return h, store


@pytest.mark.parametrize("on", [False, True])
def test_e2e_a_new_strategy_enters_at_full_size_when_off_and_at_the_venue_minimum_when_on(tmp_path, monkeypatch, on):
    h, store = _e2e(tmp_path, monkeypatch, on)
    entries = [o for o in h._orders(store) if o["intent"] == "entry"]
    assert entries and entries[0]["status"] == "filled", entries
    sig = entries[0]["signal"]
    if on:
        assert sig["sized_by"] == ramp.LIMIT and sig["ramp"].startswith("stays at minimum size"), sig
        # the venue's smallest order: 100 USDT (+1% headroom) at ~60,000 rounds up to two lots, never full size
        assert D(str(entries[0]["qty"])) == D("0.002"), entries[0]
        assert float(sig[ramp.UNRAMPED]) > 0.3 * 20_000, sig  # the size it would have had, for its track
    else:
        assert sig["sized_by"] != ramp.LIMIT and "ramp" not in sig and ramp.UNRAMPED not in sig, sig
        assert float(sig["budget"]) > 0.3 * 20_000, sig


def test_e2e_on_a_stop_still_closes_the_whole_position(tmp_path, monkeypatch):
    """Exits are never capped: the ramped position's stop takes it flat."""
    h, store = _e2e(tmp_path, monkeypatch, True, minutes=18,
                    price=lambda s: 60_000 + s * 0.01 if s < 720 else 58_000.0)
    book = store.journal_book(h.NAME, 0)
    buys = [f for f in h._fills(store) if f["side"] == "BUY"]
    sells = [f for f in h._fills(store) if f["side"] == "SELL"]
    assert buys and all(D(str(f["qty"])) <= D("0.002") for f in buys), buys  # ramped in
    assert sells and float(book["qty"]) == 0, (book, sells)  # and stopped out in full

