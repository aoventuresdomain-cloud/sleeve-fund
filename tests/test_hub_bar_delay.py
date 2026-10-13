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
    from sleeve_fund.parity import late_counts

    assert late_counts(type("S", (), {"root": tmp_path})(), "BINANCE") == {"BTC/USDT": (1, 3)}


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


H = 3600 * 10**9


def _clock(monkeypatch, at: list[int]):
    """The relay's wall clock, read from at[0]: the hour each trade is counted in and the time each write is stamped."""
    import time
    import types

    monkeypatch.setattr("sleeve_fund.hub.relay.time", types.SimpleNamespace(time_ns=lambda: at[0], time=time.time))


def _store(tmp_path):
    return type("S", (), {"root": tmp_path})()


def test_a_hub_restart_carries_the_late_counts_on_and_the_report_counts_only_the_time_it_ran(tmp_path, monkeypatch):
    """HUB-LATE-PERSIST: every deploy restarts the hub, so the counts carry on from what the last run wrote. The time
    covered is the time the hub ran, summed over its runs: 20 h, then 10 h down, then 6 h is 26 h, not 36 h. The
    report reads the latest 24 h of that up-time, so the first run's opening 2 h (and the late trade in them) drop out."""
    from sleeve_fund.hub.relay import late_since_path
    from sleeve_fund.parity import Parity, late_counts, late_since, markdown

    path = tmp_path / "hub-late-BINANCE.json"
    late_since_path(path).write_text("{}")  # #239's window file: replaced by the counts' own
    now = [T0 + 30 * MINUTE_NS]
    _clock(monkeypatch, now)
    first, _ = _relay(last_close={BTC: T0 - MINUTE_NS}, since=T0)
    first.late_path = path
    first.on_trade(_trade(T0 + 1 * MS))
    first.on_trade(_trade(T0 - 2 * MINUTE_NS))  # its minute already stored: late, counted in the hour from T0
    now[0] = T0 + 20 * H
    first.on_trade(_trade(T0 + 20 * H - 1 * MS))
    first._write_late()
    assert late_counts(_store(tmp_path), "BINANCE") == {"BTC/USDT": (1, 3)}
    assert late_since(_store(tmp_path), "BINANCE")["BTC/USDT"] == (
        pd.Timestamp(T0, unit="ns", tz="UTC"), pd.Timestamp(T0 + 20 * H, unit="ns", tz="UTC"), pd.Timedelta("20h"))
    assert not late_since_path(path).exists()

    now[0] = T0 + 36 * H
    restarted, _ = _relay(last_close={BTC: T0 - MINUTE_NS}, since=T0 + 30 * H)  # a deploy restarts the hub, 10 h on
    restarted.late_path = path
    restarted.on_trade(_trade(T0 + 36 * H - 1 * MS))
    restarted._write_late()
    restarted._write_late()  # each write brings this run's counts up to date, never adding them twice
    assert late_counts(_store(tmp_path), "BINANCE") == {"BTC/USDT": (0, 2)}  # the latest 24 h of up-time
    since, until, counted = late_since(_store(tmp_path), "BINANCE")["BTC/USDT"]
    assert since == pd.Timestamp(T0 + 2 * H, unit="ns", tz="UTC") and counted == pd.Timedelta("24h")
    assert until == pd.Timestamp(T0 + 36 * H, unit="ns", tz="UTC")

    end = pd.Timestamp(T0 + 40 * H, unit="ns", tz="UTC")
    md = markdown("the venue", [Parity("BTC/USDT", end - pd.Timedelta("132h"), end, late_trades=(0, 30_000),
                                       late_since=since, late_until=until, late_counted=counted)])
    assert "0 of 30000 (0.000%) over 24.0 h |" in md and "**BTC/USDT**: late trades counted" not in md
    short = Parity("BTC/USDT", end - pd.Timedelta("132h"), end, late_trades=(1, 30_000), late_since=since,
                   late_until=until, late_counted=pd.Timedelta("16h"))
    assert "late trades counted for 16.0 h, since 05 Oct 14:00 (while the hub ran): under 24 h" in markdown(
        "the venue", [short])


def test_a_rise_in_late_trades_today_is_not_diluted_by_a_quiet_week_before_it(tmp_path, monkeypatch):
    """PE2 / HoE: the share is the latest 24 h of up-time, not everything since the hub began, so a bad day after a
    clean week still crosses the alert (the Advisor's rule is per instrument over 24 h)."""
    import json

    from sleeve_fund.parity import LATE_ALERT_RATE, late_counts

    path = tmp_path / "hub-late-BINANCE.json"
    now = T0 + 7 * 24 * H
    week = {str(T0 + h * H): [1, 10_000, H] for h in range(7 * 24)}  # 0.01% late, every hour of a week
    week.update({str(now - h * H): [20, 10_000, H] for h in range(1, 25)})  # then 0.2% for the latest day
    path.write_text(json.dumps({"BTC/USDT": {"grace": BAR_GRACE_SECONDS, "last": now, "hours": week}}))
    late, total = late_counts(_store(tmp_path), "BINANCE")["BTC/USDT"]
    assert (late, total) == (480, 240_000) and late > LATE_ALERT_RATE * total


def test_hourly_counts_older_than_a_week_are_dropped(tmp_path, monkeypatch):
    import json

    path = tmp_path / "hub-late-BINANCE.json"
    _clock(monkeypatch, [T0 + 200 * H])
    r, _ = _relay(last_close={BTC: T0 - MINUTE_NS}, since=T0)
    r.late_path = path
    r.on_trade(_trade(T0 + 1 * MS))
    r._write_late()
    hours = sorted(int(h) for h in json.loads(path.read_text())["BTC/USDT"]["hours"])
    assert hours[0] == T0 + (200 - 7 * 24) * H and hours[-1] == T0 + 200 * H and len(hours) == 7 * 24 + 1


def test_counts_taken_under_another_grace_or_by_an_older_hub_start_afresh(tmp_path, monkeypatch):
    """A grace change makes earlier counts a different measure, and #239's {pair: [late, total]} has no up-time:
    either way the hub starts counting afresh from its own start rather than mix them."""
    import json

    from sleeve_fund.parity import late_counts, late_since

    path = tmp_path / "hub-late-BINANCE.json"
    now = [T0 + 2 * H]
    _clock(monkeypatch, now)
    other = {"grace": BAR_GRACE_SECONDS + 1, "last": T0 - H, "hours": {str(T0 - 2 * H): [7, 9_000, H]}}
    for kept in ({"BTC/USDT": other}, {"BTC/USDT": [7, 9_000]}):
        path.write_text(json.dumps(kept))
        r, _ = _relay(last_close={BTC: T0 - MINUTE_NS}, since=T0)
        r.late_path = path
        r.on_trade(_trade(T0 + 1 * MS))
        r._write_late()
        assert late_counts(_store(tmp_path), "BINANCE") == {"BTC/USDT": (0, 1)}
        assert late_since(_store(tmp_path), "BINANCE")["BTC/USDT"] == (
            pd.Timestamp(T0, unit="ns", tz="UTC"), pd.Timestamp(T0 + 2 * H, unit="ns", tz="UTC"), pd.Timedelta("2h"))


def test_unreadable_late_counts_start_afresh_and_say_so(tmp_path, monkeypatch):
    """A file cut short (the disk filled mid-write) never stops the hub: it counts afresh and logs why, once."""
    from sleeve_fund.parity import late_counts

    path = tmp_path / "hub-late-BINANCE.json"
    path.write_text('{"BTC/USDT": {"grace": 2, "last": 9')
    _clock(monkeypatch, [T0 + H])
    r, _ = _relay(last_close={BTC: T0 - MINUTE_NS}, since=T0)
    r.late_path = path
    warned = []
    monkeypatch.setattr(type(r), "log", property(lambda self: type("L", (), {"warning": warned.append})()))
    r.on_trade(_trade(T0 + 1 * MS))
    r._write_late()
    r._write_late()
    assert late_counts(_store(tmp_path), "BINANCE") == {"BTC/USDT": (0, 1)}
    assert len([w for w in warned if "can't read the late counts kept" in w]) == 1


def test_an_instrument_not_relayed_since_the_restart_keeps_its_counts_and_its_last_write(tmp_path, monkeypatch):
    """The counts of an instrument this run hasn't subscribed to yet stay as written, last write and all, so the
    report doesn't count the time since as covered."""
    from sleeve_fund.parity import late_counts, late_since

    path = tmp_path / "hub-late-BINANCE.json"
    now = [T0 + 3 * H]
    _clock(monkeypatch, now)
    first, _ = _relay(last_close={BTC: T0 - MINUTE_NS}, since=T0)
    first.late_path = path
    first.on_trade(_trade(T0 + 1 * MS))
    first._write_late()
    now[0] = T0 + 9 * H
    restarted, _ = _relay(last_close={BTC: T0 - MINUTE_NS}, since=T0 + 8 * H)
    restarted._since.clear()  # not subscribed again yet
    restarted.late_path = path
    restarted._write_late()
    assert late_counts(_store(tmp_path), "BINANCE") == {"BTC/USDT": (0, 1)}
    assert late_since(_store(tmp_path), "BINANCE")["BTC/USDT"] == (
        pd.Timestamp(T0, unit="ns", tz="UTC"), pd.Timestamp(T0 + 3 * H, unit="ns", tz="UTC"), pd.Timedelta("3h"))


def test_a_trade_counted_before_its_subscription_was_recorded_still_gets_a_window(tmp_path, monkeypatch):
    """PE2: counts written with no window would read as not proven without saying why; the first trade starts it."""
    from sleeve_fund.parity import late_since

    path = tmp_path / "hub-late-BINANCE.json"
    now = [T0 + H]
    _clock(monkeypatch, now)
    r, _ = _relay(last_close={BTC: T0 - MINUTE_NS})
    r._since.clear()
    r.late_path = path
    r.on_trade(_trade(T0 + 1 * MS))
    now[0] = T0 + 3 * H
    r._write_late()
    assert late_since(_store(tmp_path), "BINANCE")["BTC/USDT"][2] == pd.Timedelta("2h")


def test_a_late_share_counted_over_24_hours_is_reported_without_the_not_proven_mark():
    from sleeve_fund.parity import Parity, markdown

    end = pd.Timestamp("2026-10-14 15:00", tz="UTC")
    day = Parity("BTC/USDT", end - pd.Timedelta("132h"), end, late_trades=(2, 180_000),
                 late_since=end - pd.Timedelta("26h"))
    unknown = Parity("ETH/USDT", end - pd.Timedelta("132h"), end, late_trades=(0, 790))  # an older hub: no window
    md = markdown("the venue", [day, unknown])
    assert "2 of 180000 (0.001%) over 26.0 h |" in md and "**BTC/USDT**: late trades counted" not in md
    assert "0 of 790 (0.000%), counting window unknown: not proven" in md
    assert "**ETH/USDT**: late trades counted for an unknown time" in md


def test_counts_last_written_long_before_the_report_cover_only_until_that_write():
    """PE2 (HUB-LATE-COVER): a hub that stopped 2 h after it started, read 30 h later, covered 2 h, not 30."""
    from sleeve_fund.parity import Parity, markdown

    end = pd.Timestamp("2026-10-14 15:00", tz="UTC")
    since = end - pd.Timedelta("30h")
    stale = Parity("BTC/USDT", end - pd.Timedelta("132h"), end, late_trades=(0, 200_000), late_since=since,
                   late_until=since + pd.Timedelta("2h"))
    md = markdown("the venue", [stale])
    assert "0 of 200000 (0.000%) over 2.0 h: not proven" in md
    assert "**BTC/USDT**: late trades counted for 2.0 h" in md and "under 24 h, so the late share is not proven" in md


def test_a_day_with_too_few_trades_to_show_the_bar_is_not_proven():
    """PE2: with none late in n trades the 95% bound is 3/n, under 0.05% only from 6,000 trades."""
    from sleeve_fund.parity import Parity, markdown

    end = pd.Timestamp("2026-10-14 15:00", tz="UTC")
    quiet = Parity("SUI/USDT", end - pd.Timedelta("132h"), end, late_trades=(0, 5_000),
                   late_since=end - pd.Timedelta("26h"), late_until=end)
    md = markdown("the venue", [quiet])
    assert "0 of 5000 (0.000%) over 26.0 h: not proven" in md
    assert "over 5,000 trades: under 6,000 trades, so the late share is not proven" in md


def test_time_counted_after_the_report_ends_is_not_counted_as_covered():
    """A report read up to a point before the hub's last write covers less: the time after its end comes off."""
    from sleeve_fund.parity import Parity, late_hours

    end = pd.Timestamp("2026-10-14 15:00", tz="UTC")
    p = Parity("BTC/USDT", end - pd.Timedelta("132h"), end, late_trades=(0, 200_000),
               late_since=end - pd.Timedelta("40h"), late_until=end + pd.Timedelta("3h"), late_counted=pd.Timedelta("30h"))
    assert late_hours(p) == 27.0
