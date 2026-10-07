"""P1-1-DELAY: the hub builds each minute from trades by their venue time, BAR_GRACE_SECONDS after the minute ends,
so a trade from a minute's last moments that reaches the hub just after the boundary lands in its own minute (as
the venue's candle counts it), and one from the next minute's first moments never lands in this one."""

import random

import pandas as pd

from nautilus_trader.model import AggressorSide, InstrumentId, Price, Quantity, TradeId, TradeTick

from sleeve_fund.hub.relay import BAR_GRACE_SECONDS, MINUTE_NS, HubRelay, HubRelayConfig, Minutes
from tests.test_hub import BTC, T0, _relay

MS = 1_000_000
GRACE = BAR_GRACE_SECONDS * 1_000_000_000


def _trade(ts, px="60000.10", qty="0.005"):
    return TradeTick(InstrumentId.from_str(BTC), Price.from_str(px), Quantity.from_str(qty),
                     AggressorSide.from_str("BUY"), TradeId(str(ts)), ts, ts)


def _live(r):
    return [m for m in r.fanout.sent if m["t"] == "bar" and not m["refilled"]]


def test_a_late_boundary_trade_lands_in_its_own_minute_and_an_early_next_one_in_the_next():
    """Done-when (3): a trade stamped 11:59:59.900 that reaches the hub at 12:00:00.400 is in the minute closing 12:00;
    one stamped 12:00:00.100 is in the minute closing 12:01, though both arrive before 12:00's bar is built."""
    r, stored = _relay(last_close={BTC: T0 - MINUTE_NS})
    r.on_trade(_trade(T0 - 30_000 * MS, px="60000.00", qty="1.000"))
    r.on_trade(_trade(T0 + 100 * MS, px="60009.00", qty="0.200"))  # arrives first: the next minute's opening trade
    r.on_trade(_trade(T0 - 100 * MS, px="60005.50", qty="0.010"))  # arrives at 12:00:00.400, inside the grace
    r._build(now_ns=T0 + GRACE)
    (bar,) = _live(r)
    assert (bar["ts"], bar["o"], bar["h"], bar["l"], bar["c"], bar["v"]) == (
        T0, "60000.00", "60005.50", "60000.00", "60005.50", "1.010")
    assert stored == [bar]
    r.on_trade(_trade(T0 + 50_000 * MS, px="60001.00", qty="0.300"))
    r._build(now_ns=T0 + MINUTE_NS + GRACE)
    nxt = _live(r)[-1]
    assert (nxt["ts"], nxt["o"], nxt["h"], nxt["l"], nxt["c"], nxt["v"]) == (
        T0 + MINUTE_NS, "60009.00", "60009.00", "60001.00", "60001.00", "0.500")


def test_a_trade_after_its_minute_is_built_is_counted_late_and_never_put_in_a_later_minute(tmp_path):
    r, _ = _relay(last_close={BTC: T0 - MINUTE_NS})
    r.late_path = tmp_path / "hub-late-BINANCE.json"
    r.on_trade(_trade(T0 - 10_000 * MS))
    r._build(now_ns=T0 + GRACE)
    r.on_trade(_trade(T0 - 1 * MS, px="61000.00", qty="9.000"))  # 12:00's bar is out: late, by over the grace
    r.on_trade(_trade(T0 + 10_000 * MS))
    r._build(now_ns=T0 + MINUTE_NS + GRACE)
    nxt = _live(r)[-1]
    assert nxt["ts"] == T0 + MINUTE_NS and nxt["h"] == "60000.10" and nxt["v"] == "0.005"
    r._write_late()
    assert r.late_path.read_text() == '{"BTC/USDT": [1, 3]}'


def test_a_minute_is_built_exactly_the_grace_after_it_ends_and_not_before():
    """Done-when (2): the venue-time bars add BAR_GRACE_SECONDS to when a bar reaches the strategies, and nothing more."""
    r, _ = _relay(last_close={BTC: T0 - MINUTE_NS})
    r.on_trade(_trade(T0 - 10_000 * MS))
    r._build(now_ns=T0 + GRACE - 1)
    assert _live(r) == []
    r._build(now_ns=T0 + GRACE)
    assert [m["ts"] for m in _live(r)] == [T0]
    r._build(now_ns=T0 + GRACE + 30_000 * MS)  # building again changes nothing
    assert [m["ts"] for m in _live(r)] == [T0]


def test_the_build_timer_fires_the_grace_after_every_minute():
    set_ns = []

    class _Clock:
        def timestamp_ns(self):
            return T0 + 17_250 * MS

        def set_timer(self, *a, **k):
            pass

        def set_timer_ns(self, name, interval_ns, start_time_ns=None, callback=None, **k):
            set_ns.append((name, interval_ns, start_time_ns, callback))

    class _Relay(HubRelay):
        clock = _Clock()

    r = _Relay(HubRelayConfig())
    r.on_start()
    assert set_ns == [("hub-bars", MINUTE_NS, T0 + MINUTE_NS + GRACE, r._build)]
    assert BAR_GRACE_SECONDS == 2  # the 1-2 s graded by the Advisor (P1-1-DELAY), at its upper end


def test_a_minute_with_no_trades_comes_out_flat_and_the_venues_candle_stands_in():
    r, stored = _relay(last_close={BTC: T0 - MINUTE_NS})
    r.on_trade(_trade(T0 - 10_000 * MS, px="60003.00"))
    r._build(now_ns=T0 + MINUTE_NS + GRACE)  # nothing in the minute closing 12:01
    assert [m["ts"] for m in _live(r)] == [T0]
    gap = next(m for m in r.fanout.sent if m["t"] == "gap")
    assert (gap["since"], gap["until"]) == (T0 + MINUTE_NS, T0 + MINUTE_NS)
    assert [(m["ts"], m["refilled"]) for m in stored] == [(T0, False), (T0 + MINUTE_NS, True)]


def test_bars_built_from_trades_arriving_up_to_the_grace_late_match_the_venues_candles():
    """Done-when (1), offline: trades reaching the hub in the venue's order, each up to just under BAR_GRACE_SECONDS
    after its venue time, give exactly the candles bucketed by venue time. Nautilus's arrival-time bars miss this
    whenever a trade crosses a boundary in flight."""
    rng = random.Random(7)
    trades, arrive = [], 0  # (venue ts, arrival ts, px, qty)
    for ts in sorted(T0 + rng.randrange(0, 30 * MINUTE_NS) for _ in range(3000)):
        arrive = max(arrive, ts + rng.randrange(0, GRACE - MS))  # one stream: never overtakes an earlier trade
        trades.append((ts, arrive, Price(60000 + rng.randrange(-500, 500) / 100, 2),
                       Quantity(rng.randrange(1, 2000) / 1000, 3)))
    venue: dict[int, list] = {}
    for ts, _, px, qty in trades:
        close = (ts // MINUTE_NS + 1) * MINUTE_NS
        b = venue.get(close)
        venue[close] = [px, px, px, px, qty] if b is None else [b[0], max(b[1], px), min(b[2], px), px, b[4] + qty]
    m, built, ticks = Minutes(), [], T0 + GRACE
    for ts, arrive, px, qty in trades:
        while ticks <= arrive:  # the build timer, between arrivals
            built += m.close(ticks, GRACE)
            ticks += MINUTE_NS
        assert m.add(BTC, ts, px, qty)
    built += m.close(T0 + 30 * MINUTE_NS + GRACE, GRACE)
    assert {b[1]: [str(x) for x in b[2:]] for b in built} == {c: [str(x) for x in b] for c, b in venue.items()}
    assert len(built) == 30 and pd.Series([b[1] for b in built]).is_monotonic_increasing


def test_no_bar_reaches_a_strategy_before_its_minute_ends_plus_the_grace():
    """Advisor 17:05 UK: decisions on a hub bar act at boundary + grace at the earliest. Bars leave the hub only
    from _build, so every bar built at `now` must close no later than now - grace, however early, late or often the
    build timer fires."""
    rng = random.Random(11)
    m = Minutes()
    for ts in sorted(T0 + rng.randrange(0, 20 * MINUTE_NS) for _ in range(500)):
        m.add(BTC, ts, Price(60000, 2), Quantity(1, 3))
    now, closes = T0, []
    while now < T0 + 22 * MINUTE_NS:
        now += rng.choice([1, GRACE - 1, GRACE, MINUTE_NS - 1, MINUTE_NS, MINUTE_NS + 5 * GRACE])
        for _, close, *_ in m.close(now, GRACE):
            assert close + GRACE <= now
            closes.append(close)
    # Every minute up to the last one over by the grace is built, once each and in order (quiet ones flat).
    assert closes == list(range(T0 + MINUTE_NS, (now - GRACE) // MINUTE_NS * MINUTE_NS + 1, MINUTE_NS))


def test_the_parity_report_flags_an_instrument_whose_late_trades_pass_the_widen_the_grace_trigger():
    """The Advisor's trigger: late trades above 0.05% of an instrument's trades over the window."""
    from sleeve_fund.parity import LATE_ALERT_RATE, Parity, markdown

    start = pd.Timestamp("2026-10-05 12:00", tz="UTC")
    at = Parity("BTC/USDT", start, start + pd.Timedelta("24h"), late_trades=(5, 10_000))  # exactly 0.05%: no alert
    over = Parity("ETH/USDT", start, start + pd.Timedelta("24h"), late_trades=(6, 10_000))
    quiet = Parity("SOL/USDT", start, start + pd.Timedelta("24h"))  # the hub sent no counts
    md = markdown("the venue", [at, over, quiet])
    assert LATE_ALERT_RATE == 0.0005
    assert "**ETH/USDT**: LATE TRADES ABOVE 0.05% (0.060%)" in md
    assert "**BTC/USDT**: LATE" not in md and "**SOL/USDT**: LATE" not in md
