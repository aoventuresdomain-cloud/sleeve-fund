"""The market data hub's live side (v2 P1-1): the stream, the fan-out, gaps and their refill."""

import json
import socket
import time

import pandas as pd
import pytest

from nautilus_trader.model import AggressorSide, Bar, BarType, InstrumentId, Price, Quantity, QuoteTick, TradeId, TradeTick

from sleeve_fund.hub import protocol
from sleeve_fund.hub.relay import MINUTE_NS, Gaps, HubRelay, HubRelayConfig, refill_bars, store_sink
from sleeve_fund.hub.server import Fanout

BTC, ETH = "BTCUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE"
T0 = int(pd.Timestamp("2026-10-05 12:00", tz="UTC").value)


def _bar(iid, close_ns, c="60000.10"):
    return Bar(BarType.from_str(f"{iid}-1-MINUTE-LAST-INTERNAL"), Price.from_str("60000.00"), Price.from_str("60010.00"),
               Price.from_str("59990.00"), Price.from_str(c), Quantity.from_str("1.250"), close_ns, close_ns)


def test_ticks_and_bars_travel_as_the_venues_own_decimals():
    t = TradeTick(InstrumentId.from_str(BTC), Price.from_str("60000.10"), Quantity.from_str("0.005"),
                  AggressorSide.from_str("BUY"), TradeId("42"), T0, T0)
    q = QuoteTick(InstrumentId.from_str(BTC), Price.from_str("60000.00"), Price.from_str("60000.10"),
                  Quantity.from_str("3.1"), Quantity.from_str("0.7"), T0, T0)
    msgs = [protocol.trade(t, T0 + 5), protocol.quote(q, T0 + 6), protocol.bar_from_nautilus(_bar(BTC, T0), T0 + 7)]
    back = [protocol.decode(protocol.encode(m)) for m in msgs]
    assert back == msgs
    assert back[0] == {"t": "trade", "id": BTC, "px": "60000.10", "qty": "0.005", "side": "BUY", "tid": "42",
                       "ts": T0, "recv": T0 + 5}
    assert back[1]["bid"] == "60000.00" and back[1]["ask_qty"] == "0.7"
    assert back[2]["c"] == "60000.10" and back[2]["ts"] == T0 and back[2]["refilled"] is False


@pytest.mark.parametrize("msg", [{"v": 2, "sub": [BTC]}, {"v": 1, "sub": []}, {"v": 1}, {"v": 1, "sub": [3]}])
def test_a_subscription_in_another_version_or_without_instruments_is_refused(msg):
    with pytest.raises(ValueError):
        protocol.check_subscription(msg)


def _connect(port, ids):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(protocol.encode(protocol.subscription(ids)))
    f = s.makefile("rb")
    return s, f


def _until(f, kind):
    while True:
        m = json.loads(f.readline())
        if m["t"] == kind:
            return m


@pytest.fixture
def fanout():
    wanted = []
    fo = Fanout("BINANCE", known=lambda: {BTC}, want=wanted.extend, heartbeat=0.2, log=lambda *_: None,
                instruments=lambda ids: [{"type": "CryptoPerpetual", "id": i} for i in sorted(ids & {BTC})])
    fo.wanted = wanted
    fo.start(host="127.0.0.1")
    yield fo
    fo.stop()


def test_each_client_gets_only_its_instruments_and_every_heartbeat(fanout):
    a, fa = _connect(fanout.port, [BTC])
    b, fb = _connect(fanout.port, [ETH])
    ha, hb = _until(fa, "hello"), _until(fb, "hello")
    assert ha["pending"] == [] and hb["pending"] == [ETH]
    assert [i["id"] for i in ha["instruments"]] == [BTC] and hb["instruments"] == []  # what each can be sent
    assert fanout.wanted == [ETH]  # not relayed yet: the relay is asked to add it
    deadline = time.time() + 5
    while len(fanout.clients) < 2 and time.time() < deadline:
        time.sleep(0.01)
    fanout.publish({"t": "trade", "id": BTC, "px": "1"})
    fanout.publish({"t": "trade", "id": ETH, "px": "2"})
    assert _until(fa, "trade")["px"] == "1" and _until(fb, "trade")["px"] == "2"
    assert "venue_up" in _until(fa, "hb") and "venue_up" in _until(fb, "hb")
    a.close(), b.close()


def test_a_client_speaking_another_version_is_told_and_closed(fanout):
    s = socket.create_connection(("127.0.0.1", fanout.port), timeout=5)
    s.sendall(protocol.encode({"v": 9, "sub": [BTC]}))
    f = s.makefile("rb")
    m = json.loads(f.readline())
    assert m["t"] == "error" and "version 9" in m["message"]
    assert f.readline() == b""  # closed
    s.close()


def test_a_client_that_falls_behind_is_cut_off_and_the_rest_carry_on(fanout, monkeypatch):
    from sleeve_fund.hub import server

    monkeypatch.setattr(server, "MAX_QUEUE", 5)
    slow, _ = _connect(fanout.port, [BTC])  # never reads
    slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    fast, ff = _connect(fanout.port, [BTC])
    _until(ff, "hello")
    deadline = time.time() + 5
    while len(fanout.clients) < 2 and time.time() < deadline:
        time.sleep(0.01)
    for i in range(20_000):
        fanout.publish({"t": "trade", "id": BTC, "px": str(i), "pad": "x" * 200})
    deadline = time.time() + 10
    while fanout.cut_off == 0 and time.time() < deadline:
        ff.readline()
    assert fanout.cut_off >= 1
    fast.close(), slow.close()


def test_a_skipped_minute_is_a_gap_and_a_late_bar_changes_nothing():
    g = Gaps()
    assert g.see(BTC, T0) is None and g.see(BTC, T0 + MINUTE_NS) is None
    assert g.see(BTC, T0 + 4 * MINUTE_NS) == (T0 + 2 * MINUTE_NS, T0 + 3 * MINUTE_NS)
    assert g.see(BTC, T0 + 2 * MINUTE_NS) is None  # a refilled bar arriving after: no new gap
    assert g.see(ETH, T0 + 9 * MINUTE_NS) is None  # instruments are tracked apart


def _candles(start, n):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    c = 100.0 + pd.Series(range(n), index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 2.0}, index=idx)


def test_a_gap_is_refilled_from_the_venues_closed_candles_only():
    recent = lambda pair, minutes: _candles("2026-10-05 11:50", 15)  # noqa: E731 - 11:50 to 12:04, 12:04 forming
    bars = refill_bars(recent, "BTC/USDT", BTC, T0 + MINUTE_NS, T0 + 10 * MINUTE_NS, T0 + 11 * MINUTE_NS)
    # Candles opening 12:00 to 12:03 close 12:01 to 12:04; 12:04's is still forming and left out.
    assert [b["ts"] for b in bars] == [T0 + k * MINUTE_NS for k in (1, 2, 3, 4)]
    assert all(b["refilled"] and b["id"] == BTC for b in bars) and bars[0]["o"] == "110.0"


class _Fan:
    def __init__(self):
        self.sent = []

    def publish(self, m):
        self.sent.append(m)


class _Now:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *a):
        self.calls.append(a)
        fn(*a)


def _relay(last_close=None, since=None):
    stored = []
    r = HubRelay(HubRelayConfig(instrument_ids=(BTC,)))
    recent = lambda pair, minutes: _candles("2026-10-05 11:50", 30)  # noqa: E731
    r.attach(_Fan(), sink=stored.extend, pairs={BTC: "BTC/USDT"}, recent=recent, last_close=last_close)
    r._refill = r._store = _Now()
    r._since[BTC] = since if since is not None else T0 - 10 * MINUTE_NS
    return r, stored


def test_bars_go_to_clients_and_the_store_and_a_missed_stretch_is_refilled_and_flagged():
    r, stored = _relay(last_close={BTC: T0 - 3 * MINUTE_NS})  # the store ends three minutes back: hub was down
    r.on_bar(_bar(BTC, T0))
    gap = next(m for m in r.fanout.sent if m["t"] == "gap")
    assert (gap["since"], gap["until"]) == (T0 - 2 * MINUTE_NS, T0 - MINUTE_NS)
    refilled = [m for m in stored if m["refilled"]]
    assert [m["ts"] for m in refilled] == [T0 - 2 * MINUTE_NS, T0 - MINUTE_NS]
    live = [m for m in stored if not m["refilled"]]
    assert [m["ts"] for m in live] == [T0] and live[0] in r.fanout.sent


def test_the_first_bar_after_subscribing_is_a_part_bar_and_the_venues_candle_stands_in():
    r, stored = _relay(since=T0 - 30_000_000_000)  # subscribed half way through the minute that closes at T0
    r.on_bar(_bar(BTC, T0, c="60001.00"))
    assert all(m.get("c") != "60001.00" for m in r.fanout.sent)  # the part bar is never published
    assert [(m["ts"], m["refilled"]) for m in stored] == [(T0, True)]
    r.on_bar(_bar(BTC, T0 + MINUTE_NS, c="60002.00"))  # the next is whole
    assert stored[-1]["c"] == "60002.00" and stored[-1]["refilled"] is False


def test_bars_reach_the_stores_write_path_as_open_time_and_source_a_refill_in_one_call(tmp_path, monkeypatch):
    from sleeve_fund.history import HistoryStore

    store, calls, said = HistoryStore(tmp_path), [], []
    real = store.append_bars
    monkeypatch.setattr(store, "append_bars", lambda *a: calls.append(a) or real(*a))
    sink = store_sink(store, "BINANCE", {BTC: "BTC/USDT"}, log=said.append)
    sink([protocol.bar_from_nautilus(_bar(BTC, T0), T0)])
    # A refill of the same minute with another close, and the next minute: one call, the stored bar kept.
    sink([{**protocol.bar_from_nautilus(_bar(BTC, T0 + k * MINUTE_NS, c="60005.00"), T0), "refilled": True}
          for k in (0, 1)])
    sink([protocol.bar_from_nautilus(_bar(ETH, T0), T0)])  # not one of the hub's instruments: not stored
    bar = lambda close, c: (close - MINUTE_NS, 60000.0, 60010.0, 59990.0, c, 1.25)  # noqa: E731
    assert calls == [("BINANCE", "BTC/USDT", [bar(T0, 60000.1)], "live"),
                     ("BINANCE", "BTC/USDT", [bar(T0, 60005.0), bar(T0 + MINUTE_NS, 60005.0)], "refill")]
    assert len(said) == 1 and "1 refill bar(s) differ from the stored" in said[0]
    cov = store.coverage("BINANCE", "BTC/USDT")
    assert cov.last == pd.Timestamp(T0, unit="ns", tz="UTC")  # both minutes stored, the first one not overwritten


def test_an_instrument_added_while_the_hub_runs_is_picked_up_on_the_next_pass():
    r, _ = _relay()
    r.discover = lambda: {BTC: "BTC/USDT", ETH: "ETH/USDT"}
    relayed = []
    r._relay = relayed.append  # the node's subscription itself
    r.relayed = {BTC}
    r._refresh()
    assert relayed == [ETH] and r.pairs[ETH] == "ETH/USDT"


def test_a_bar_with_no_volume_is_checked_against_the_venues_candle():
    r, stored = _relay()
    r.on_bar(_bar(BTC, T0))
    flat = Bar(BarType.from_str(f"{BTC}-1-MINUTE-LAST-INTERNAL"), *(Price.from_str("60000.10"),) * 4,
               Quantity.from_str("0.000"), T0 + MINUTE_NS, T0 + MINUTE_NS)
    r.on_bar(flat)  # no trades reached the hub that minute: a dropped feed or a quiet market
    assert stored[-1]["ts"] == T0 + MINUTE_NS and stored[-1]["refilled"] is True
    assert not any(m["t"] == "bar" and m["ts"] == T0 + MINUTE_NS and not m["refilled"] for m in r.fanout.sent)


def test_the_hub_runs_the_store_backfill_in_its_own_process(monkeypatch):
    """The store takes one writer per venue: the REST backfill runs in the hub, not beside it."""
    from types import SimpleNamespace

    from sleeve_fund import history
    from sleeve_fund.hub.__main__ import collector

    calls = []
    monkeypatch.setattr(history, "main", lambda argv: calls.append(argv) or 0)
    collector(SimpleNamespace(name="BINANCE")).join(5)
    assert calls == [["run", "--venue", "binance"]]


def test_trades_arriving_after_their_minutes_bar_are_counted_for_the_parity_report(tmp_path):
    r, _ = _relay()
    r.late_path = tmp_path / "hub-late-BINANCE.json"
    trade = lambda ts: TradeTick(InstrumentId.from_str(BTC), Price.from_str("60000.10"),  # noqa: E731
                                 Quantity.from_str("0.005"), AggressorSide.from_str("BUY"), TradeId(str(ts)), ts, ts)
    r.on_trade(trade(T0 - 1))
    r.on_bar(_bar(BTC, T0))
    r.on_trade(trade(T0 - 2))  # in the minute just built: late
    r.on_trade(trade(T0 + 1))
    r._write_late()
    assert json.loads(r.late_path.read_text()) == {"BTC/USDT": [1, 3]}
