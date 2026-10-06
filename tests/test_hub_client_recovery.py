"""PR #140's QA round (quant-review/v2-p1/hub-client-140.md): what the hub client does about minutes it missed
(P1-C2), the bar under way when it starts (P1-C5), a stream that keeps breaking (P1-C6), the venue's own candles
(P1-C7), data that isn't the hub's (P1-C8) and a hub that is down or has lost its venue (P1-C9)."""

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_dashboard import AUTH, SAME, client  # noqa: F401 - the dashboard fixture

from sleeve_fund.history import HistoryStore
from sleeve_fund.paper import hub_client
from sleeve_fund.paper.hub_client import Decoder, HubDataClient, HubStatus
from sleeve_fund.paper.node import stored_minutes

IID = "BTCUSDT-PERP.BINANCE"
M = 60_000_000_000
S = 1_000_000_000
T0 = 1_791_223_200_000_000_000  # 18:00 UTC on 5 Oct 2026: a whole hour


def _minutes(n: int, seed: int = 5) -> pd.DataFrame:
    """n minutes from 18:00 by open time, prices to 2 dp and volumes to 3, as the venue sends them."""
    rng = np.random.default_rng(seed)
    close = np.round(60_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, n))), 2)
    open_ = np.round(np.r_[60_000.0, close[:-1]], 2)
    return pd.DataFrame({"open": open_, "high": np.round(np.maximum(open_, close) + 5, 2),
                         "low": np.round(np.minimum(open_, close) - 5, 2), "close": close,
                         "volume": np.round(rng.uniform(0.5, 5, n), 3)},
                        index=pd.to_datetime(T0 + np.arange(n) * M, unit="ns", utc=True))


def _msg(df, i):
    r, close = df.iloc[i], int(df.index[i].value) + M
    return {"t": "bar", "id": IID, "o": f"{r.open:.2f}", "h": f"{r.high:.2f}", "l": f"{r.low:.2f}",
            "c": f"{r.close:.2f}", "v": f"{r.volume:.3f}", "ts": close, "recv": close, "refilled": False}


def _store(tmp_path, df) -> HistoryStore:
    """The hub's store: every minute it relayed, written as it relayed it."""
    hs = HistoryStore(tmp_path / "history")
    hs.append_bars("BINANCE", "BTC/USDT", [(int(ts.value), *r) for ts, r in df.iterrows()], "live")
    return hs


def _feed(d, df, rows, lag=2 * S):
    out = []
    for i in rows:
        b = d(_msg(df, i), int(df.index[i].value) + M + lag)
        if b is not None:
            out.append(b)
    return out


def _tup(b):
    return (int(b.ts_event), *(round(float(x), 2) for x in (b.open, b.high, b.low, b.close)), round(float(b.volume), 3))


def _store_bars(hs, minutes, end=None):
    out = hs.read("BINANCE", "BTC/USDT", minutes, end=end)
    return [(int(ts.value), *(round(float(x), 2) for x in (r.open, r.high, r.low, r.close)), round(float(r.volume), 3))
            for ts, r in out.iterrows()]


def _recorder():
    said = []
    return said, lambda level, kind, message: said.append((level, kind, message))


def test_a_minute_the_client_missed_while_the_hub_kept_it_comes_from_the_store(tmp_path):
    """P1-C2: the client was away 18:09:50-18:10:05 while the hub stayed up; the minute it missed went to the
    store, so the bar is the store's (the backtest's), not one short."""
    df = _minutes(30)
    hs = _store(tmp_path, df)
    said, report = _recorder()
    d = Decoder("15-MINUTE-LAST-EXTERNAL", report, stored_minutes("BINANCE", "BTC/USDT", hs))
    out = _feed(d, df, [i for i in range(30) if i != 9])
    assert [_tup(b) for b in out] == _store_bars(hs, 15) and said == []


def test_the_closing_minute_missed_is_recovered_and_the_bar_still_sent_on_time(tmp_path):
    df = _minutes(30)
    hs = _store(tmp_path, df)
    d = Decoder("15-MINUTE-LAST-EXTERNAL", recover=stored_minutes("BINANCE", "BTC/USDT", hs))
    out = _feed(d, df, [i for i in range(30) if i != 14])  # 18:14-18:15, the bar's last minute, missed
    assert [_tup(b) for b in out] == _store_bars(hs, 15)  # sent with 18:16, 62 s after its close: in time


def test_a_minute_missed_and_not_in_the_store_either_leaves_the_bar_short_and_told(tmp_path):
    df = _minutes(30)
    hs = _store(tmp_path, df.drop(df.index[9]))  # the hub missed it too
    said, report = _recorder()
    d = Decoder("15-MINUTE-LAST-EXTERNAL", report, stored_minutes("BINANCE", "BTC/USDT", hs))
    out = _feed(d, df, [i for i in range(30) if i != 9])
    assert len(out) == 2 and [k for _, k, _ in said] == ["bar_incomplete"] and "missing 1 of its 15" in said[0][2]


def test_on_one_minute_bars_a_missed_minute_is_told_and_kept_as_history_not_decided_on(tmp_path):
    df = _minutes(6)
    hs = _store(tmp_path, df)
    said, report = _recorder()
    d = Decoder("1-MINUTE-LAST-EXTERNAL", report, stored_minutes("BINANCE", "BTC/USDT", hs))
    out = _feed(d, df, [0, 1, 4, 5])  # 18:03 and 18:04 missed
    assert [b.ts_event for b in out] == [T0 + M, T0 + 2 * M, T0 + 5 * M, T0 + 6 * M]
    assert said == [("warning", "hub_gap", f"{IID}: the feed missed 2 minutes closing 05 Oct 18:03 UTC to 05 Oct "
                     "18:04 UTC, so they weren't decided on; 2 recovered from the history store for the record")]
    # A refill of one landing afterwards changes nothing: it was told already.
    assert d(_msg(df, 2), T0 + 6 * M + 5 * S) is None and len(said) == 1


def _as_refilled(m):
    """A minute as the hub refills it from the venue's candles: its numbers as str(float), 60000.1 for 60000.10."""
    return {**m, **{k: str(float(m[k])) for k in "ohlcv"}, "refilled": True}


def test_a_bar_built_from_live_and_refilled_minutes_is_written_at_the_instruments_decimals(tmp_path):
    """Code review on #140: the refilled minute holds the bar's high, written 61000.1; the bar is 61000.10, as
    every other price of the instrument, and the store's bar to the cent."""
    df = _minutes(15)
    df.iloc[7, df.columns.get_loc("high")] = 61_000.10
    hs = _store(tmp_path, df)
    d = Decoder("15-MINUTE-LAST-EXTERNAL")
    d.precision[IID] = (2, 3)
    out = [d(_as_refilled(_msg(df, i)) if i == 7 else _msg(df, i), int(df.index[i].value) + M + 2 * S)
           for i in range(15)]
    (b,) = [x for x in out if x is not None]
    assert str(b.high) == "61000.10" and {p.precision for p in (b.open, b.high, b.low, b.close)} == {2}
    assert b.volume.precision == 3 and _tup(b) == _store_bars(hs, 15)[0]


def test_minutes_recovered_after_a_refilled_one_keep_their_cents(tmp_path):
    """Code review on #140: the minute that shows the gap is a refill written 60000.0; the minutes taken from
    the store for the gap keep the instrument's decimals rather than that message's one."""
    df = _minutes(15)
    df.iloc[8, df.columns.get_loc("close")] = 60_000.0
    df.iloc[8, df.columns.get_loc("high")] = max(df.high.iloc[8], 60_000.0)
    df.iloc[8, df.columns.get_loc("low")] = min(df.low.iloc[8], 60_000.0)
    hs = _store(tmp_path, df)
    d = Decoder("15-MINUTE-LAST-EXTERNAL", recover=stored_minutes("BINANCE", "BTC/USDT", hs))
    d.precision[IID] = (2, 3)
    for i in (0, 1, 2, 3, 4, 8):
        d(_as_refilled(_msg(df, i)) if i == 8 else _msg(df, i), int(df.index[i].value) + M + 2 * S)
    kept = d.building[IID].minutes
    assert [kept[int(df.index[i].value) + M]["h"] for i in (5, 6, 7)] == [f"{df.high.iloc[i]:.2f}" for i in (5, 6, 7)]
    out = [d(_msg(df, i), int(df.index[i].value) + M + 2 * S) for i in range(9, 15)]
    assert _tup(out[-1]) == _store_bars(hs, 15)[0]


def test_a_decoder_without_the_definition_rounds_nothing():
    d = Decoder("15-MINUTE-LAST-EXTERNAL")
    df = _minutes(15)
    out = [d(_as_refilled(_msg(df, i)) if i == 3 else _msg(df, i), int(df.index[i].value) + M + 2 * S)
           for i in range(15)]
    (b,) = [x for x in out if x is not None]
    assert {p.precision for p in (b.open, b.high, b.low, b.close)} == {2} and b.volume.precision == 3


def test_a_store_that_cant_be_read_leaves_the_minutes_missing_and_says_why():
    df = _minutes(20)
    said, report = _recorder()

    def broken(*_):
        raise OSError("disk gone")

    d = Decoder("15-MINUTE-LAST-EXTERNAL", report, broken)
    out = _feed(d, df, [i for i in range(16) if i != 9])
    assert len(out) == 1 and [k for _, k, _ in said] == ["hub_gap", "bar_incomplete"] and "disk gone" in said[0][2]


def test_after_a_start_mid_bar_the_warm_up_and_the_live_bars_cover_every_bar_the_store_has(tmp_path):
    """P1-C5: started at 20:07 on 15-minute bars. The warm-up holds the stored bars to 20:00; the bar under way
    (20:00-20:15) is completed from the store and sent, so the indicators see every bar a backtest sees."""
    df = _minutes(3 * 60)
    k = 2 * 60 + 7
    hs = _store(tmp_path, df.iloc[:k])  # stored to 20:07 when the node starts
    warm = _store_bars(hs, 15)
    d = Decoder("15-MINUTE-LAST-EXTERNAL", recover=stored_minutes("BINANCE", "BTC/USDT", hs))
    hs.append_bars("BINANCE", "BTC/USDT", [(int(ts.value), *r) for ts, r in df.iloc[k:].iterrows()], "live")
    sent = [_tup(b) for b in _feed(d, df, range(k, len(df)))]
    assert warm + sent == _store_bars(hs, 15)


def test_a_bar_under_way_at_start_the_store_cant_complete_is_not_sent_and_told(tmp_path):
    df = _minutes(45)
    hs = _store(tmp_path, df.iloc[3:7])  # the store lacks 18:00-18:03
    said, report = _recorder()
    d = Decoder("15-MINUTE-LAST-EXTERNAL", report, stored_minutes("BINANCE", "BTC/USDT", hs))
    out = _feed(d, df, range(7, 45))
    assert [b.ts_event for b in out] == [T0 + 30 * M, T0 + 45 * M]
    assert said == [("warning", "warmup", f"{IID}: the bar closing 05 Oct 18:15 UTC, under way when this node "
                     "started, lacks 3 of its minutes before the first one received, even from the history store: "
                     "not sent, so the indicators skip it")]


def test_a_node_started_on_a_bars_first_minute_asks_the_store_for_nothing():
    asked = []
    d = Decoder("15-MINUTE-LAST-EXTERNAL", recover=lambda *a: asked.append(a) or [])
    _feed(d, _minutes(16), range(16))
    assert asked == []


class _Client(SimpleNamespace):
    """The real _open, _read and _connect on a stand-in node: data reaching the node in .got, reports in .said."""

    def __init__(self, port, spec="1-MINUTE-LAST-EXTERNAL", status=None):
        super().__init__(cfg=SimpleNamespace(host="127.0.0.1", port=port, instrument_ids=(IID,), status=status),
                         _writer=None, connects=0, _closing=False, last_heartbeat_ns=0, venue_up=False, got=[],
                         said=[], writers=[], defined=[])
        self.clock = SimpleNamespace(timestamp_ns=time.time_ns)
        self.decode = Decoder(spec, self.report)

    def report(self, level, kind, message):
        self.said.append((level, kind, message))

    def _handle_data(self, d):
        self.got.append(d)

    def _handle_instrument(self, i):
        self.defined.append(i)

    def create_task(self, coro, name=None):
        coro.close()

    async def _read(self, reader):
        pass

    async def _open(self):
        r = await HubDataClient._open(self)
        self.writers.append(self._writer)
        return r


def test_a_stream_that_breaks_after_every_message_backs_off_is_told_once_and_closes_its_sockets(monkeypatch):
    """P1-C6: a hub that sends one good line then a broken one, every time."""
    sleeps, real_sleep = [], asyncio.sleep

    async def fake_sleep(d, *a, **k):
        sleeps.append(d)
        await real_sleep(0.02)

    monkeypatch.setattr(hub_client.asyncio, "sleep", fake_sleep)

    async def serve(reader, writer):
        await reader.readline()
        writer.write(b'{"t":"hello","v":1,"venue":"BINANCE","pending":[],"instruments":[]}\n')
        writer.write(b'{"t":"hb","ts":1,"venue_up":true}\n{"t":"bar","id":"' + IID.encode() + b'"}\n')
        await writer.drain()
        await real_sleep(30)

    async def run():
        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        c = _Client(server.sockets[0].getsockname()[1])
        reader, _ = await c._open()
        task = asyncio.create_task(HubDataClient._read(c, reader))
        await real_sleep(0.6)
        c._closing = True
        await asyncio.wait_for(task, 5)
        server.close()
        return c

    c = asyncio.run(run())
    assert len(sleeps) >= 4 and sleeps[:4] == [1, 2, 5, 5]  # the backoff keeps growing: never back for long enough
    assert [k for _, k, _ in c.said] == ["hub"] and "Lost" in c.said[0][2]  # told once, not twice a cycle
    assert all(w.is_closing() for w in c.writers[:-1])


def test_a_node_started_before_its_hub_listens_waits_for_it(monkeypatch):
    """P1-C9: the hub restarting or still starting is retried, not left to a restart of the node."""
    import socket

    from sleeve_fund.hub.server import Fanout

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setattr(hub_client, "RECONNECT_SECONDS", (0.1,))
    monkeypatch.setattr(hub_client, "instrument_from", lambda d: SimpleNamespace(**d))
    hub = Fanout("BINANCE", known=lambda: {IID}, instruments=lambda ids: [{"id": IID, "price_precision": 2,
                                                                           "size_precision": 3}],
                 log=lambda *_: None)
    c = _Client(port)

    async def run():
        async def late_hub():
            await asyncio.sleep(0.3)
            hub.start("127.0.0.1", port)

        starter = asyncio.create_task(late_hub())
        await asyncio.wait_for(HubDataClient._connect(c), 10)
        await starter

    asyncio.run(run())
    hub.stop()
    assert c.connects == 1 and [i.id for i in c.defined] == [IID]
    assert c.decode.precision == {IID: (2, 3)}  # the bars it builds are written at the instrument's decimals


def test_a_hub_that_never_comes_gives_up_at_the_connect_timeout(monkeypatch):
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setattr(hub_client, "RECONNECT_SECONDS", (0.05,))
    monkeypatch.setattr(hub_client, "CONNECT_TIMEOUT", 0.2)
    with pytest.raises(OSError):
        asyncio.run(HubDataClient._connect(_Client(port)))


def test_heartbeats_keep_the_shared_status_and_it_tells_a_hub_without_its_venue():
    status = HubStatus()
    c = _Client(1, status=status)

    async def run():
        r = asyncio.StreamReader()
        r.feed_data(b'{"t":"hb","ts":1,"venue_up":false}\n')
        r.feed_eof()
        c._open = lambda: (_ for _ in ()).throw(OSError("stopped"))
        c._closing = False

        async def stop_after_first(*_a, **_k):
            c._closing = True

        hub_client.asyncio.sleep, real = stop_after_first, hub_client.asyncio.sleep
        try:
            await HubDataClient._read(c, r)
        finally:
            hub_client.asyncio.sleep = real

    asyncio.run(run())
    now = c.last_heartbeat_ns
    assert status.heartbeat_ns == now and status.venue_up is False
    assert status.venue_down(now + 10 * S) and not status.venue_down(now + 31 * S)  # a dead hub is not "venue down"
    status.venue_up = True
    assert not status.venue_down(now + S)


def test_the_price_watchdog_waits_while_the_hub_is_up_but_has_lost_its_venue():
    """P1-C9: 15 minutes without prices restarts a node, unless its hub is alive and says the venue is gone:
    then a restart can't help. A dead hub still means a restart."""
    from sleeve_fund.strategies.base import STALE_PRICE_RESTART_MINUTES
    from sleeve_fund.strategies.trend_filter import TrendFilter, TrendFilterConfig
    from sleeve_fund.venues import venue
    from nautilus_trader.model import BarType

    now = T0 + (STALE_PRICE_RESTART_MINUTES + 1) * M

    class Probe(TrendFilter):
        clock = property(lambda self: SimpleNamespace(timestamp_ns=lambda: now))

    inst = venue("binance").instrument("BTC", "USDT")
    s = Probe(TrendFilterConfig(instrument_id=inst.id, assumed_taker_fee=0.0005,
                                bar_type=BarType.from_str(f"{inst.id}-15-MINUTE-LAST-EXTERNAL")))
    events = []
    s._last_market_ns, s._price = T0, lambda: 60_000.0
    s.runtime = SimpleNamespace(name="s1", now=lambda: now, backtest=False,
                                store=SimpleNamespace(event=lambda *a, **k: events.append(a[2])))
    s.hub_status = HubStatus()
    s.hub_status.heartbeat_ns, s.hub_status.venue_up = now - 3 * S, False
    assert s._feed_dead() is False and "hub_venue_down" in events
    s.hub_status.heartbeat_ns = now - 60 * S  # the hub itself silent
    assert s._feed_dead() is True and "feed_dead" in events
    del s
    _drop_here()


@pytest.mark.parametrize("spec", ["1-HOUR-LAST-EXTERNAL", "4-HOUR-LAST-EXTERNAL", "1-DAY-LAST-EXTERNAL"])
def test_the_venues_own_candles_are_refused_up_front_on_a_hub_fed_venue(spec, tmp_path):
    """P1-C7: refused where a strategy is made (a seed file, the dashboard's form), with what to choose instead,
    rather than failing to start; elsewhere they are still offered."""
    from sleeve_fund.paper.config import check_hub_bar_spec, load_sleeve

    with pytest.raises(ValueError, match="venue's own .* candles aren't available there; choose 1, 5 or 15-minute"):
        check_hub_bar_spec("binance", spec)
    check_hub_bar_spec("kraken", spec)
    check_hub_bar_spec("binance", "15-MINUTE-LAST-INTERNAL")
    f = tmp_path / "s.toml"
    f.write_text(f'[sleeve]\nname = "x"\nstrategy = "trend_filter"\ninstrument = "BTC/USDT"\nvenue = "binance"\n'
                 f'bar_spec = "{spec}"\nstarting_balance = 1000\n[params]\nmarket = "perp"\n')
    with pytest.raises(ValueError, match="aren't available there"):
        load_sleeve(f)


def test_the_dashboard_refuses_a_hub_venue_strategy_on_the_venues_own_candles(client):  # noqa: F811
    from urllib.parse import parse_qs, urlparse

    c, store = client
    form = {"name": "daily-b", "strategy": "trend_filter", "instrument": "BTC/USDT", "venue": "binance",
            "bar_spec": "1-DAY-LAST-EXTERNAL", "starting_balance": "5000", "risk_profile": "conservative",
            "warmup_bars": "0", "reason": "test", "market": "perp"}
    r = c.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    error = parse_qs(urlparse(r.headers["location"]).query)["error"][0]
    assert "venue's own 1-day candles aren't available there" in error and store.sleeves() == []
    assert "strategies on this market" in error and "Binance" not in error  # no venue name on the dashboard


def test_a_hub_fed_node_takes_no_data_from_the_venue(monkeypatch):
    """P1-C8: no venue candles to top up its warm-up, and no gap loader that could put a venue candle in place of
    the hub's bar. It shares the hub's status with its watchdog."""
    from sleeve_fund.paper import node as node_mod
    from sleeve_fund.paper.config import SleeveConfig

    loaders = []
    real = node_mod.history_loader
    monkeypatch.setattr(node_mod, "history_loader", lambda *a, **k: loaders.append(k) or real(*a, **k))
    added = []
    monkeypatch.setattr(node_mod.LiveNode, "add_strategy", lambda self, st: added.append(st), raising=False)
    cfg = SleeveConfig(name="t", strategy="trend_filter", instrument="BTC/USDT", venue="binance",
                       bar_spec="15-MINUTE-LAST-INTERNAL", starting_balance=1000.0, params={"market": "perp"})
    n = node_mod.build_node(cfg, log_level="ERROR", asset_fetch=dict, hub=("127.0.0.1", 1))
    (s,) = added
    assert loaders == [{"recent": None}] and s.gap_loader is None
    assert s.hub_fed and isinstance(s.hub_status, HubStatus)
    _drop_here(n, added)


def _drop_here(*objs):
    """Nautilus objects must be freed on the thread that made them, not by a later test's server thread."""
    import gc

    for o in objs:
        if hasattr(o, "dispose"):
            o.dispose()
        if isinstance(o, list):
            o.clear()
    gc.collect()
