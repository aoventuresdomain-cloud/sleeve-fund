"""F-B1-REPLAY (HoE Drop B MUST, HoQA b1): a multi-asset Strategy's paper path replayed offline, as the single-Strategy
replay harness does (research.replay.replay): research.replay.replay_legs runs every leg through the paper runtime in
one engine, as its one paper node does (paper.node.build_legs_node), each leg on its own venue clone with its own
journal, and the legs' same-close entries gated in one LegBatch pass, strongest idea first, then by the asset's place in
the list (spec §4). QT3's b1 compares it with the backtest (research.multi_asset.run_multi_asset)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from sleeve_fund.store import Store, gate_decisions_t, sleeves_t

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
PX = {"BTC": 60_000.0, "ETH": 3_000.0, "SOL": 150.0}
PARAMS = {"rise": 0.01, "dip": 0.005, "market": "perp", "allow_short": True, "stop_loss": 0.02}  # as paper_scenarios
INFO = {"symbols": [  # the venue's perpetual contracts, offline (as test_exposure_gate_xfails.INFO)
    {"symbol": f"{b}USDT", "pair": f"{b}USDT", "contractType": "PERPETUAL", "status": "TRADING", "baseAsset": b,
     "quoteAsset": "USDT", "pricePrecision": 2, "quantityPrecision": 3,
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                 {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                 {"filterType": "MIN_NOTIONAL", "notional": "5"}]} for b in PX]}
LEGS = [(5, 0.0), (20, 0.015), (20, -0.012), (20, 0.015)]  # every asset moves alike: same-close ideas
NAME = "tx"


def _start():
    from test_replay import START

    return datetime.fromtimestamp(START / 1e9, tz=UTC)


def _store():
    store = Store.in_memory(portfolio_gate=True)
    at = _start() - timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 10_000_000, actor="pm", reason="test", rebase=True, now=at), at)
    store.create_legs(name=NAME, strategy="ping_pong", bar_spec="1-MINUTE-LAST-INTERNAL", assets=ASSETS,
                      params=PARAMS, starting_balance=0, universe_date="2025-10-02", venue="BINANCE")
    for leg in [NAME, *store._asset_legs(NAME)]:
        if store.joining(leg):
            store.land_move(leg, at)
    return store


def _record(path, row, asset, minutes_legs):
    """One leg's recorded session (test_long_short._record's feed, on its own asset): quotes and trades every second."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.paper.recorder import Recorder
    from sleeve_fund.venues import venue
    from test_replay import START

    base = asset.split("/")[0]
    inst = venue("binance").instrument(base, "USDT")
    rec = Recorder(path)
    rec.meta = {"balances": ["1000000.00 USDT"],
                "sleeve": {"name": row, "strategy": "ping_pong", "instrument": asset,
                           "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 0, "risk_profile": "balanced",
                           "params": PARAMS, "maker_fee": "0.0002", "taker_fee": "0.0005", "tick_seconds": 30}}
    rec.start(inst)
    px, s, p = PX[base], 0, inst.price_precision  # the instrument's own tick (B-43: BTC's kept 0.1)
    for minutes, move in minutes_legs:
        step = (1 + move) ** (1 / (minutes * 60)) if minutes else 1 + move
        for _ in range(minutes * 60 or 1):
            px *= step
            t = START + s * 1_000_000_000
            half = px * 0.00001
            rec.quote(QuoteTick(inst.id, Price(px - half, p), Price(px + half, p), Quantity(100, 3), Quantity(100, 3),
                                t, t + 1000))
            rec.trade(TradeTick(inst.id, Price(px, p), Quantity(50, 3),
                                AggressorSide.BUY if s % 2 else AggressorSide.SELL, TradeId(f"{base}{s}"), t + 2000,
                                t + 3000))
            s += 1
    rec.close()


def _run(tmp_path, tag=""):
    from sleeve_fund.research.replay import replay_legs
    from sleeve_fund.supervisor import Supervisor

    store = _store()
    rows = [NAME, *store._asset_legs(NAME)]
    paths = []
    for row, asset in zip(rows, ASSETS):
        paths.append(tmp_path / f"{row.replace('~', '-')}{tag}.jsonl.gz")
        _record(paths[-1], row, asset, LEGS)
    Supervisor(store).mark_portfolio(_start() + timedelta(hours=2))  # one mark covers the run (cut-A e2e pattern)
    return store, rows, replay_legs(paths, store=store)


def _decisions(store):
    """Every gate decision, in the order the gate made them: (bar, leg, intent, outcome)."""
    d, s = gate_decisions_t, sleeves_t
    q = (select(d.c.bar_ts, s.c.name, d.c.intent, d.c.outcome).join(s, s.c.id == d.c.sleeve_id)
         .where(d.c.stage == "submit").order_by(d.c.seq))
    with store.engine.connect() as c:
        return [tuple(r) for r in c.execute(q)]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """The venue's perpetuals offline, as test_drop_b_multi_asset_qa3.timer_run: each contract from a fixed listing and a
    published 0.01% funding rate stored for every settlement around the run (no funding hold)."""
    import json

    import pandas as pd

    from sleeve_fund import funding as funding_mod
    from sleeve_fund.venues import binance_contract, venue

    monkeypatch.setattr(funding_mod, "DEFAULT_ROOT", tmp_path / "funding")
    bv = venue("binance")
    monkeypatch.setattr(bv, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    rates = [(int(t.timestamp() * 1000), 0.0001) for t in pd.date_range(
        _start() - timedelta(days=2), _start() + timedelta(days=2), freq="8h").floor("8h")]
    for asset in ASSETS:
        fp = funding_mod._path("BINANCE", asset)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps({"rates": rates}))
    monkeypatch.setattr(bv, "funding_loader", lambda pair, since: [r for r in rates if r[0] >= since])


@pytest.fixture
def strength(monkeypatch):
    """Each leg's idea strength by its asset: SOL strongest, then ETH, then BTC, the reverse of the list's order, so
    the pass's order shows it ranks by strength, not by place or by which leg's bar came first."""
    from sleeve_fund.strategies.ping_pong import PingPong

    rank = {"BTC": 1.0, "ETH": 2.0, "SOL": 3.0}
    monkeypatch.setattr(PingPong, "idea_strength", lambda self: rank[str(self.instrument.id.symbol)[:3]],
                        raising=False)


RANK = {NAME: 1, f"{NAME}~ETH": 2, f"{NAME}~SOL": 3}


def test_a_multi_asset_strategy_replays_through_the_paper_runtime_one_pass_per_close(tmp_path, strength):
    """Every leg trades its own asset under its own row; on every close where legs have ideas, the gate takes them in
    one pass strongest first (SOL, ETH, BTC), the first entries and the flips' opens alike: a flip opens only after
    its close fills, and the pass waits for it, as the backtest's CloseBatch does."""
    store, rows, orders = _run(tmp_path)
    assert list(orders) == rows
    for row in rows:
        assert [(o["side"], o["intent"]) for o in orders[row]][:3] == [
            ("BUY", "entry"), ("SELL", "exit"), ("SELL", "entry")], (row, orders[row][:3])
        assert all(o["filled_qty"] > 0 for o in orders[row][:3])
    per_bar = {}
    for bar, leg, intent, outcome in _decisions(store):
        per_bar.setdefault(bar, []).append(leg)
    together = {b: legs for b, legs in per_bar.items() if len(legs) > 1}
    flips = {o["ts"].replace(second=0, microsecond=0, tzinfo=None) for o in orders[NAME][2:] if o["intent"] == "entry"}
    together = {b.replace(tzinfo=None): legs for b, legs in together.items()}
    assert len(together) >= 3 and flips <= set(together), ("set-up: closes where every leg has an idea", per_bar)
    for bar, legs in together.items():
        assert legs == sorted(legs, key=lambda n: -RANK[n]), (bar, legs)


def test_a_multi_asset_replay_is_deterministic(tmp_path, strength):
    from sleeve_fund.research.replay import comparable

    a_store, rows, a = _run(tmp_path, "a")
    b_store, _, b = _run(tmp_path, "b")
    assert {r: comparable(a[r]) for r in rows} == {r: comparable(b[r]) for r in rows}
    assert _decisions(a_store) == _decisions(b_store)


def test_the_recordings_must_be_the_strategys_legs_in_its_order(tmp_path):
    from sleeve_fund.research.replay import replay_legs

    store = _store()
    rows = [NAME, *store._asset_legs(NAME)]
    paths = []
    for row, asset in zip(reversed(rows), reversed(ASSETS)):
        paths.append(tmp_path / f"{row.replace('~', '-')}.jsonl.gz")
        _record(paths[-1], row, asset, [(2, 0.0)])
    with pytest.raises(ValueError, match="legs in its asset list's order"):
        replay_legs(paths, store=store)
