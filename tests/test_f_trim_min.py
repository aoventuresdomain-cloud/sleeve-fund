"""F-TRIM-MIN (HoQA, 15 Oct; covers B17): when the compliance tests trim an entry under the smallest order value the
venue takes, the gate refuses it in words naming the test, journaled once for the episode, and never sends it for the
venue to deny on every bar. HoQA's z8 (v2-p2/drop-4/test_g5_a9_boxa_qa_v3.py) is the end-to-end master; this is the
once-per-episode cell. Runs on Postgres too with TEST_DATABASE_URL, on an empty schema."""

from __future__ import annotations

import os
from decimal import Decimal as D

import pandas as pd
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund.store import Store


def _store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        return Store(f"sqlite:///{tmp_path}/ftrim.db", portfolio_gate=True)
    from sqlalchemy import text

    from sleeve_fund.store import make_engine, metadata

    engine = make_engine(url)
    with engine.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
        for f in ("fund_flows_write_once", "fill_instrument", "compliance_write_once", "gate_test_hit_known"):
            c.execute(text(f"DROP FUNCTION IF EXISTS {f}() CASCADE"))
    metadata.drop_all(engine)
    return Store(engine=engine, portfolio_gate=True)


def test_an_entry_trimmed_under_the_venue_minimum_is_refused_once_in_words_and_never_sent(tmp_path, monkeypatch):
    """The minimum-size ramp on and T1 at 0.25% of an 800 fund: T1 trims the one minimum order (0.002 at ~60,000) to
    0.001, 60 of value, under the venue's 100. Over ten bars that each want the entry: one refusal naming T1, no order
    sent, none denied."""
    import test_exposure_gate_xfails as h
    from test_portfolio_paper_entry import END

    from sleeve_fund import ramp
    from sleeve_fund.supervisor import Supervisor

    monkeypatch.setenv(ramp.ENV, "1")
    store = _store(tmp_path)
    c = store.compliance()
    tests = [dict(t) for t in c.tests]
    for t in tests:
        if t["test"] == "asset_stop_loss":
            t["limit"] = D("0.0025")
    store.add_compliance_version(tests, c.asset_types, expected_version=c.version, note="F-TRIM-MIN cell")
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params={**h.PERP, "stop_loss": h.STOP})
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 800, actor="pm", reason="F-TRIM-MIN: the fund's capital",
                                      rebase=True, now=before), before)
    Supervisor(store).mark_portfolio(END)
    plan = h.Plan(t0=h.STD_T0, minutes=12, windows=[(h.M(h.STD_T0, 2), h.M(h.STD_T0, 12), 1)], balance=200_000.0,
                  params={}, tag="ftrim")
    h.run(tmp_path, store, plan, monkeypatch)

    entries = [o for o in h._orders(store) if o["intent"] == "entry"]
    assert entries == [], entries  # never sent, so never denied
    refusals = [e for e in h._events(store) if e["kind"] == "entry_trimmed_below_minimum"]
    assert len(refusals) == 1, refusals  # once for the episode, not every bar
    msg = refusals[0]["message"]
    assert msg.startswith("Entry refused: T1 Per asset leaves 0.001") and "under the venue minimum 100" in msg, msg
