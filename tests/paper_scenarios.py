"""Shared paper scenarios for tests (stop-safety, halt clearing, RAL): a real liquidation in paper, and a frozen clock.

Lifted from Head of QA's ral-xfails harness (`_liquidate`, `_dashboard_clock`) so engineering tests and QA's use the
same set-up; the API was agreed with QA (6 Oct, ~20:56). Two rules from that agreement:
- `liquidate_in_paper` never lifts a production guard on its own. A set-up that needs one lifted names it in
  `guards_off`, so the caller's test says so; any global it touches is restored before it returns.
- The liquidation is real code: a recorded paper session replayed through the paper runtime (as test_long_short does),
  never a hand-written journal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from test_long_short import PERP, _meta, _record

NAME = "ping-pong-test"  # the recorded session's strategy (test_long_short._meta)
# ping_pong on a perp, balanced: 2x with a 33% margin cap, and a 2% stop-loss. The stop lets it start under the
# stop-safety gate (stopless strategies are capped at 1x) and keeps its open risk under the 5%-of-book limit; the 60%
# gap goes through the stop and the liquidation price together.
PARAMS = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.02, **PERP}
GAP_PX = 97_440.0  # 60,900 x 1.6: where the gap leaves the price
LEGS = [(5, 0.0), (20, 0.015), (0, 0.6), (5, 0.0)]  # (minutes, move): flat, a 1.5% rise (it goes short), the gap

# Production guards a set-up may lift, by name: (module, attribute, value while lifted). None yet; a set-up that
# needs one adds it here and passes its name in guards_off, so the test that relies on it says so.
GUARD_LIFTS: dict[str, tuple[str, str, object]] = {}


def record_at(path, meta, legs, px, hours):
    """test_long_short._record, starting `hours` after test_replay.START, so a restart is later than what it follows."""
    import test_replay

    start = test_replay.START
    test_replay.START = start + int(hours * 3600) * 1_000_000_000
    try:
        _record(path, meta, legs, px=px)
    finally:
        test_replay.START = start


def replay_into(store, path, name=NAME):
    """The recorded session replayed through the paper runtime into `store`; the strategy may exist already."""
    from sleeve_fund.research.replay import replay

    if any(s.name == name for s in store.sleeves()):
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
    try:
        return replay(path, store=store)
    finally:
        store.__dict__.pop("create_sleeve", None)


def _gap_closed(orders, side):
    """Set-up check: the gap closed the position more than 50% past its entry (past a 2x liquidation, ~49%), by the
    liquidation or by the stop it went through. Returns (the closing order, the entry it closed)."""
    close = orders[-1]
    assert close["intent"] in ("liquidation", "stop_loss") and close["side"] == side, [(o["side"], o["intent"])
                                                                                       for o in orders]
    entry = [o for o in orders if o["intent"] == "entry" and o["side"] != side][-1]
    move = close["avg_px"] / entry["avg_px"] - 1
    assert (move if side == "BUY" else -move) > 0.5, (close["avg_px"], entry["avg_px"])
    return close, entry


def _gap_event(store, name, after_id=0):
    """The engine's "liquidation" event or, on a head that books a stop filled through the liquidation price as a
    stop-loss, the halt the gap caused on the same tick."""
    for kinds in (("liquidation",), ("risk_halt",)):
        ev = store.last_event(name, kinds)
        if ev is not None and ev["id"] > after_id:
            return ev
    raise AssertionError("the gap left no liquidation or halt event")


def _position(store, name, close_id, start):
    """The fills of the position the order `close_id` closed: from the fill that opened it from flat (or crossed
    through flat) up to the close. Returns (equity just after the opening fill, from the journal; the position's fills;
    the closing order's fills)."""
    from sleeve_fund.store import replay_book

    fills = sorted(store.fills(name, limit=100_000), key=lambda x: (x["ts"], x["id"]))
    end = next(i for i, x in enumerate(fills) if x["order_id"] == close_id)
    held, first = 0.0, None
    for i, x in enumerate(fills[:end]):
        new = held + (x["qty"] if x["side"] == "BUY" else -x["qty"])
        new = 0.0 if abs(new) < 1e-9 else new
        if new and (not held or (new > 0) != (held > 0)):
            first = i
        held = new
    assert first is not None and held, "set-up: no open position before the close"
    book = replay_book(fills[:first + 1], start)
    return (book["cash"] + book["qty"] * fills[first]["price"], fills[first:end],
            [x for x in fills[end:] if x["order_id"] == close_id])


def liquidate_in_paper(tmp_path, store, *, name=NAME, params=None, start=10_000.0, guards_off=()):
    """A paper short at about 60,629 with a 2% stop, then a 60% gap up through the stop and the liquidation price: the
    position closes past bankruptcy and the strategy halts (on #155 3be572a it keeps about 6,733 of 10,000). Real code
    throughout: a recorded session replayed through the paper runtime.

    guards_off: names from GUARD_LIFTS to lift for this set-up only (the caller's test must say why).

    Returns: rem (cash left), peak, liq (the liquidation or halt event), before (marked equity just before it), close
    and entry (orders), orders (how many), reason (the halt reason), at_entry (equity at the position's first entry
    fill, Advisor 20:37), entry_fee, x (isolated margin + entry fees + the closing fee, Advisor 18:17) and y_at_entry
    (100 x / at_entry, uncapped)."""
    from sleeve_fund import markets, risk

    unknown = set(guards_off) - set(GUARD_LIFTS)
    if unknown:
        raise ValueError(f"no such guard to lift: {sorted(unknown)}; known: {sorted(GUARD_LIFTS)}")
    if name != NAME:
        raise ValueError(f"the recorded session's strategy is {NAME!r}")
    import importlib

    profiles = dict(risk.PROFILES)  # the original profile objects, restored as they were (identity kept)
    lifted = []
    try:
        for g in guards_off:
            mod, attr, value = GUARD_LIFTS[g]
            m = importlib.import_module(mod)
            lifted.append((m, attr, getattr(m, attr)))
            setattr(m, attr, value)
        path = tmp_path / "liq.jsonl.gz"
        record_at(path, _meta(start, params or PARAMS), LEGS, 60_000.0, 0)
        orders = replay_into(store, path, name)
    finally:
        for m, attr, old in reversed(lifted):
            setattr(m, attr, old)
        risk.PROFILES.clear()
        risk.PROFILES.update(profiles)
    close, entry = _gap_closed(orders, "BUY")
    s = store.sleeve(name)
    assert s.status == "halted", (s.status, s.status_reason)
    book = store.journal_book(name, start)
    assert book["qty"] == 0
    liq = _gap_event(store, name)
    before = store.equity_at_or_before(name, liq["ts"] - timedelta(seconds=1))["equity"]
    at_entry, held, closing = _position(store, name, close["order_id"], start)
    qty = sum(f["qty"] if f["side"] == "SELL" else -f["qty"] for f in held)  # the short, before the close
    opened = [f for f in held if f["side"] == "SELL"]
    avg = sum(f["qty"] * f["price"] for f in opened) / sum(f["qty"] for f in opened)
    lev = risk.PROFILES[s.risk_profile].max_leverage
    x = (markets.isolated_margin(qty, avg, lev) + sum(f["fee"] for f in opened)
         + sum(f["fee"] for f in closing))
    return SimpleNamespace(rem=book["cash"], peak=store.peak_equity(name), liq=liq, before=before, close=close,
                           entry=entry, orders=len(orders), reason=s.status_reason, at_entry=at_entry,
                           entry_fee=held[0]["fee"], x=x, y_at_entry=100 * x / at_entry)


def frozen_clock(monkeypatch, at: datetime, *where: str):
    """Freeze the clock at `at`. By default every loaded sleeve_fund module attribute bound to store.utcnow, under
    any name (journal, riskops, the dashboard, the supervisor, research.runner's _utcnow, ...), so nothing reads live
    time for pauses, restarts and reconciles. Named modules are imported and must have a utcnow (else ValueError).
    Returns a setter: `clock(later)` moves it. A module imported after this call isn't patched (unless it reads
    store.utcnow at call time), so import what the test needs first. A SleeveRuntime built with the default `now` was
    bound at import, so pass `now=` to it as the tests already do."""
    import importlib
    import sys

    from sleeve_fund import store

    if at.tzinfo is None:
        raise ValueError("frozen_clock needs an aware UTC time")
    state = {"at": at}
    real = store.utcnow

    def frozen():
        return state["at"]

    if where:
        for mod in where:
            m = importlib.import_module(mod)
            if not hasattr(m, "utcnow"):
                raise ValueError(f"{mod} has no utcnow to freeze")
            monkeypatch.setattr(m, "utcnow", frozen)
    else:
        for mod_name, m in list(sys.modules.items()):
            if m is None or not (mod_name == "sleeve_fund" or mod_name.startswith("sleeve_fund.")):
                continue
            for attr, value in list(vars(m).items()):
                if value is real:
                    monkeypatch.setattr(m, attr, frozen)

    def move(to: datetime) -> datetime:
        state["at"] = to
        return to

    return move


NEXT_DAY = datetime(2025, 10, 4, 1, 0, tzinfo=timezone.utc)  # the recording is 3 Oct 00:00-00:30 UTC
