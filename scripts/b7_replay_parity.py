"""B7 (venue-swap plan §B; pe1-plans/b-smoke-runbook.md): after the B3 round trip, read-only.

1. Replay parity (P2-7): paper's gate decisions over B3's window, as the journal holds them (DbLedger.replayable()),
   against the portfolio backtest's (run_portfolio) over the stored bars of the same window, through
   research.replay_parity.compare; and the fill-drift line (replay_parity.drift) over the window's opening fills.
2. Dashboard = journal: the Strategy page's own figures (dashboard.metrics.sleeve_summary and dashboard.book) against
   the journal: fees, the B3 trip's P&L, and equity.

The replay starts on paper's first decided bar, so both sides open the cycle on the same close, and it is fed what
paper had: the Strategy's own cash before B3 (the journal's), the fund's equity at that decision (its gate row's
book_equity; the rest of the fund is held as unallocated cash), the Strategy's settings and risk profile, and the
venue's fees and the measured spread. Other Strategies' positions are not replayed: when paper's row shows any, the
check says so beside the verdict, since a limit can then bite differently.

Runs over stdin in the supervisor container (.github/workflows/final-smoke.yml), from /app:
    python -u - --strategy ping-pong-ls-binance --since ISO [--until ISO]
--since is the round trip's own "since" line; --until is by default the last fill since then. Prints its figures and
one line "B7 replay parity: PASS|FAIL (why)" and one "B7 dashboard = journal: PASS|FAIL (why)"; exits 1 on any FAIL.
Nothing is written: selects, a read of the history store and an in-memory backtest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal

CENT = Decimal("0.01")


def verdict(check: str, fails: list[str], notes: list[str] = ()) -> int:
    tail = f" [{'; '.join(notes)}]" if notes else ""
    print((f"{check}: PASS" if not fails else f"{check}: FAIL ({'; '.join(fails)})") + tail)
    return 1 if fails else 0


def _figures(store, name: str, since: datetime) -> dict:
    """Per gate decision of `name` since `since`, by intent id: the book its gate read (book_equity, open_risk, gross)
    and its order row's side, signal, filled qty and average price, when one was sent."""
    from sqlalchemy import select

    from sleeve_fund.store import gate_decisions_t as d
    from sleeve_fund.store import orders_t as o
    from sleeve_fund.store import sleeves_t as s

    q = (select(d.c.intent_id, d.c.stage, d.c.book_equity, d.c.open_risk, d.c.gross, d.c.profile_version, o.c.side,
                o.c.signal, o.c.filled_qty, o.c.avg_px)
         .join(s, s.c.id == d.c.sleeve_id).outerjoin(o, o.c.order_id == d.c.order_id)
         .where(s.c.name == name, d.c.bar_ts >= since, d.c.outcome != "error"))
    with store.engine.connect() as c:
        return {(r.intent_id, r.stage): r._asdict() for r in c.execute(q)}


def journal_side(store, name: str, since: datetime, until: datetime):
    """Paper's side: (replayable rows of `name` in the window, their figures, the unbarred intent ids)."""
    from sleeve_fund.gate_ledger import DbLedger

    rows = [r for r in DbLedger(store, lambda: None).replayable()
            if r.strategy == name and since <= r.bar_ts <= until]
    figures = _figures(store, name, since)
    # Sent outside a bar decision: its order row has no signal.bar (replay_parity's `unbarred`).
    unbarred = {i for (i, _), f in figures.items() if f["signal"] is not None and "bar" not in (f["signal"] or {})}
    return rows, figures, unbarred


def bars(history, venue: str, pair: str, minutes: int, start: datetime, end: datetime, warm_days: int):
    """(warm-up, window) bars of `minutes`, stamped at their close: the window from the bar closing at `start` to the
    one closing at or before `end`; the warm-up the warm_days before it."""
    import pandas as pd

    lo = pd.Timestamp(start) - pd.Timedelta(days=warm_days)
    got = history.read(venue, pair, minutes, start=lo, end=pd.Timestamp(end))
    got = got[(got.index >= lo) & (got.index <= pd.Timestamp(end))]
    return got[got.index < pd.Timestamp(start)], got[got.index >= pd.Timestamp(start)]


def replay(store, s, start: datetime, end: datetime, capital: Decimal, equity: Decimal, history=None,
           instrument=None, run=None):
    """The portfolio backtest of `s` alone over [start, end], fed what paper had. Returns (result, why): why is set
    when the stored bars can't cover the window, and the run isn't made."""
    from sleeve_fund import markets, spreads
    from sleeve_fund.data import spec_minutes
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.instruments import history_price_decimals
    from sleeve_fund.open_risk import ATR_DAYS
    from sleeve_fund.research.portfolio_run import LegSpec, run_portfolio
    from sleeve_fund.venues import venue

    profile = venue((s.venue or "").lower())
    history = history or HistoryStore(os.environ.get("HISTORY_DIR"))
    minutes = spec_minutes(s.bar_spec)
    warm, window = bars(history, profile.name, s.instrument, minutes, start, end, ATR_DAYS + 1)
    if window.empty or window.index[0].to_pydatetime() != start:
        return None, f"the history store has no {minutes}-minute bar closing at {start:%H:%M:%S} UTC"
    exec_prices = window if minutes == 1 else bars(history, profile.name, s.instrument, 1, start, end, 0)[1]
    if instrument is None:
        base, quote = s.instrument.split("/")
        instrument = profile.instrument(base, quote, fees=profile.fees,
                                        price_precision=history_price_decimals(window["close"]))
    spec = LegSpec(s.name, s.strategy, instrument, window, exec_prices, warm, exec_minutes=1, bar_minutes=minutes,
                   params=dict(s.params), capital=float(capital),
                   half_spread=spreads.series(profile.name, s.instrument, store),
                   fees=markets.fees_for(s.params, profile.fees, s.venue), risk_profile=s.risk_profile)
    return (run or run_portfolio)([spec], unallocated=float(equity - capital)), None


def parity(store, name: str, since: datetime, until: datetime | None = None, history=None, instrument=None,
           run=None) -> int:
    from sleeve_fund.research import portfolio
    from sleeve_fund.research.replay_parity import (
        compare,
        drift,
        from_backtest,
        from_journal,
    )
    from sleeve_fund.risk import PORTFOLIO

    s = store.sleeve(name)
    fills = sorted((f for f in store.fills(name, limit=100_000) if f["ts"] >= since), key=lambda f: f["ts"])
    until = until or (fills[-1]["ts"] if fills else datetime.now(UTC))
    rows, figures, unbarred = journal_side(store, name, since, until)
    submits = [r for r in rows if r.stage == "submit"]
    print(f"strategy {name}, window {since:%Y-%m-%d %H:%M:%S} to {until:%H:%M:%S} UTC: {len(submits)} gate "
          f"decision(s) journaled, {len(fills)} fill(s)")
    first = next((r for r in submits if r.intent_id not in unbarred), None)
    if first is None:
        return verdict("B7 replay parity", ["no gate decision on a bar journaled in the window: nothing to replay"])
    fig = figures[(first.intent_id, "submit")]
    if fig["book_equity"] is None:
        return verdict("B7 replay parity", [f"the gate row at {first.bar_ts:%H:%M} holds no book: it can't be replayed"])
    before = [f for f in store.fills(name, limit=100_000) if f["ts"] < since]
    held = sum((Decimal(str(f["qty"])) * (1 if f["side"] == "BUY" else -1) for f in before), Decimal(0))
    if held:
        return verdict("B7 replay parity", [f"held {held} before the window: the replay starts only from flat"])
    capital = Decimal(str(s.starting_balance)) + store.strategy_cash_pnl(name, at=since - timedelta(microseconds=1))
    equity = Decimal(str(fig["book_equity"]))
    notes = []
    if Decimal(str(fig["open_risk"] or 0)) or Decimal(str(fig["gross"] or 0)):
        notes.append(f"other Strategies held gross {fig['gross']}, open risk {fig['open_risk']} at the first "
                     "decision; the replay holds none")
    if fig["profile_version"] != PORTFOLIO.version:
        notes.append(f"paper decided under portfolio profile v{fig['profile_version']}, the replay v{PORTFOLIO.version}")
    start, end = first.bar_ts, until.replace(second=0, microsecond=0)
    print(f"replay from the first decided bar {start:%H:%M} to {end:%H:%M} UTC: capital {capital:.2f}, fund equity "
          f"{equity:.2f}, risk profile {s.risk_profile}")
    result, why = replay(store, s, start, end, capital, equity, history, instrument, run)
    if why:
        return verdict("B7 replay parity", [why], notes)
    report = compare(from_journal(rows, unbarred), from_backtest(result.decisions), start, end)
    print(f"decisions: paper {report.paper}, backtest {report.backtest}, matched {report.matched}, "
          f"mismatched {len(report.mismatches)} {report.counts() or ''}")
    for m in report.mismatches:
        p, b = m.paper, m.backtest
        print(f"  {m.bar_ts:%H:%M} {', '.join(m.what)}: paper "
              f"{'-' if p is None else f'{p.outcome} {p.requested}->{p.approved} {p.limit or ''}'} | backtest "
              f"{'-' if b is None else f'{b.outcome} {b.requested}->{b.approved} {b.limit or ''}'}")
    for u in report.unbarred:
        print(f"  not decided on a bar (listed, not compared): {u.bar_ts:%H:%M:%S} {u.outcome} {u.approved}")
    # Drift: each opening order paper filled, from the price its gate was handed (with any act drift taken back out).
    opened = []
    for r in submits:
        f = figures.get((r.intent_id, "submit")) or {}
        if f.get("avg_px") and Decimal(str(f.get("filled_qty") or 0)) > 0:
            side = 1 if f["side"] == "BUY" else -1
            close = Decimal(str((f["signal"] or {}).get("close", r.price)))
            expected = r.price - close * side * Decimal(str(portfolio.ACT_DRIFT_BP)) / 10_000
            opened.append((side, expected, Decimal(str(f["avg_px"]))))
    d = drift(opened)
    print(f"fill drift: {d.n} opening fill(s), mean adverse "
          f"{'-' if d.mean_bp is None else f'{d.mean_bp:.2f}'} bp; trigger {'MET' if d.act else 'not met'}")
    fails = [f"{len(report.mismatches)} decision(s) differ"] if not report.clean else []
    fails += ["the ACT-DRIFT trigger is met (a decision for the HoE and the Advisor)"] if d.act else []
    return verdict("B7 replay parity", fails, notes)


def dashboard(store, name: str, since: datetime) -> int:
    """The Strategy page's fees, the B3 trip's P&L and equity, from the page's own functions, against the journal."""
    from sleeve_fund import markets
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard.metrics import sleeve_summary
    from sleeve_fund.research.metrics import trades

    s = store.sleeve(name)
    x = bookm.sleeve_extras(store, sleeve_summary(store, s), bookm.daily(store, name))
    fills = store.fills(name, limit=100_000)  # newest first, as the page reads them
    perp = markets.is_perp(s.params)
    fails: list[str] = []
    fees = sum((Decimal(str(f["fee"])) for f in fills), Decimal(0))
    print(f"fees: page {Decimal(str(x['fees'])):.8f}, journal {fees:.8f} over {len(fills)} fill(s)")
    if round(Decimal(str(x["fees"])), 8) != round(fees, 8):
        fails.append("fees differ")
    trips = [t for t in trades(list(reversed(fills)), perp, store.funding(name, limit=None) if perp else None,
                               store.insurance(name, limit=None) if perp else None)
             if t["opened"] is not None and t["opened"] >= since]
    moved = store.strategy_cash_pnl(name, since=since - timedelta(microseconds=1))
    page = sum((Decimal(repr(t["pnl"])) for t in trips), Decimal(0))
    print(f"B3 trip(s): {len(trips)}, P&L on the page {page:.2f}, cash the journal moved since {moved:.2f}")
    if not trips:
        fails.append("no closed trip since the window opened on the page")
    elif round(page, 2) != round(moved, 2):
        fails.append(f"trip P&L {page:.2f} differs from the journal's {moved:.2f}")
    jb = store.journal_book(name, s.starting_balance)
    last = store.last_equity(name)
    journal_eq = Decimal(str(jb["cash"])) + Decimal(str(jb["qty"])) * Decimal(str(last["price"] if last else 0))
    print(f"equity: page {x['equity']:.2f} (mark {last['ts'] if last else 'none'}), journal {journal_eq:.2f}")
    if fills and (last is None or last["ts"] < fills[0]["ts"]):
        fails.append("the page's equity is from a mark before the last fill")
    elif round(Decimal(str(x["equity"])), 2) != round(journal_eq, 2):
        fails.append(f"equity {x['equity']:.2f} differs from the journal's {journal_eq:.2f}")
    return verdict("B7 dashboard = journal", fails)


def main(argv: list[str]) -> int:
    from sleeve_fund.store import Store

    ap = argparse.ArgumentParser(prog="b7_replay_parity")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--since", required=True, type=lambda t: datetime.fromisoformat(t).astimezone(UTC))
    ap.add_argument("--until", type=lambda t: datetime.fromisoformat(t).astimezone(UTC))
    a = ap.parse_args(argv)
    print(json.dumps({"check": "b7", "at": datetime.now(UTC).isoformat(timespec="seconds")}))
    store = Store()
    rc = parity(store, a.strategy, a.since, a.until)
    return dashboard(store, a.strategy, a.since) | rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
