"""VS-1 Done when 4 (HoQA point 6): a paper liquidation at the venue's maintenance tiers books the same price whichever
journal it is written to: a recorded session gapping through its liquidation price, replayed through the paper runtime
into the journal on Postgres (TEST_DATABASE_URL, as CI runs) and into the replay's own journal in memory, sends the same
orders and books the liquidation at the same price to the cent, the price solved at the tier its notional lands in.
The replay harness runs the recorded spot-venue instrument (it has no market for a venue's own perp symbol yet), so the
tiered table is put on that instrument's perp terms: the same solve the venue's tiers take."""
from __future__ import annotations

from test_degraded_155_qa import _record_session, _session_meta
from test_exposure_gate_xfails import (
    _offline,  # noqa: F401 - the paper harness's offline venue (autouse)
)
from test_sleeve_runtime import (
    store,  # noqa: F401  (Postgres when TEST_DATABASE_URL is set)
)
from test_vs1_maintenance_tiers import TIERS

PERP = {"market": "perp", "allow_short": True, "rise": 0.01, "dip": 0.005, "stop_loss": 0.10}
GAP = [(5, 0.0), (20, 0.015), (0, 0.6), (5, 0.0)]  # a short (sold on the rise) gapped through


def test_a_paper_liquidation_books_the_same_tiered_price_on_postgres_as_in_memory(store, tmp_path, monkeypatch):  # noqa: F811
    import dataclasses

    from sleeve_fund import markets, risk
    from sleeve_fund.research.replay import comparable, replay
    from sleeve_fund.store import Store

    for nm, p in list(risk.PROFILES.items()):  # a 10% margin cap, as QA's guards-on cases (test_gap_liq)
        monkeypatch.setitem(risk.PROFILES, nm, dataclasses.replace(p, max_position_pct=0.1))
    real = markets.terms

    def tiered(params, venue=None, pair=None):  # TEC-TIERS3-A: R-28's assume_from is gone
        t = real(params, venue, pair)
        return dataclasses.replace(t, maintenance_margin=TIERS) if t is not None else t

    monkeypatch.setattr(markets, "terms", tiered)
    path = tmp_path / "gap.jsonl.gz"
    _record_session(path, _session_meta(10_000.0, PERP), GAP)
    on_pg, pg_fills = replay(path, with_fills=True, store=store)
    in_memory, mem_fills = replay(path, with_fills=True)
    assert comparable(on_pg) == comparable(in_memory)
    (liq_pg,) = [o for o in on_pg if o["intent"] == "liquidation"]
    (liq_mem,) = [o for o in in_memory if o["intent"] == "liquidation"]
    px = {f["order_id"]: f["price"] for f in pg_fills}, {f["order_id"]: f["price"] for f in mem_fills}
    assert round(px[0][liq_pg["order_id"]], 2) == round(px[1][liq_mem["order_id"]], 2)
    said = [e["message"] for e in store.events("ping-pong-test", limit=1000) if e["kind"] == "liquidation"]
    assert len(said) == 1, said
    # The tiers are in force: the same session at one flat 0.5% names another liquidation price
    monkeypatch.setattr(markets, "terms", real)
    flat = Store.in_memory()
    replay(path, store=flat)
    (said_flat,) = [e["message"] for e in flat.events("ping-pong-test", limit=1000) if e["kind"] == "liquidation"]
    assert _liq_px(said[0]) != _liq_px(said_flat), (said, said_flat)


def _liq_px(message: str) -> str:
    return message.split("the liquidation price ")[1].split(", ", 1)[0]
