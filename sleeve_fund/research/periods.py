"""R-13 and R-15 (HoE, PM 21 Oct 2026): a tear sheet's returns by year and month against buy-and-hold, and every
trade with the equity after it, over the scored window.

The scored window is the full research period's default-settings run, as the trading table reads, until R-11's
continuous window lands; `scored` is the one place that says so. Every figure here is read off that run's own
equity and fills, so the months compound to its end equity and the trades sum to its end minus its start, both to
the cent. Pure: no I/O, no runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

import pandas as pd

from sleeve_fund.research.metrics import daily_closes, fills_to_rows, max_drawdown, summary, trades

MIN_SHARPE_DAYS = 60
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def scored(r) -> tuple[pd.Series, pd.Series]:
    """The scored window's equity, the Strategy's and buy-and-hold's: the full research period's runs (R-11 moves
    this to its continuous window)."""
    return r.full_period.equity, r.full_period_benchmark.equity


@dataclass
class Period:
    label: str  # "2021", or "2021-03" for a month
    ret: float
    partial: str = ""  # "" for a whole period, else "from 02 Jan" / "to 24 Jul"
    sharpe: float | None = None
    max_dd: float | None = None
    trades: int = 0


def _days(equity: pd.Series) -> pd.Series:
    """The curve's value at each day's close, labelled by the day it closes (closes are stamped at the next
    midnight), with the starting value first, labelled by the day the curve starts."""
    d = daily_closes(equity)
    idx = d.index.tz_convert(None) if d.index.tz is not None else d.index
    day = (idx - pd.Timedelta(1, "ns")).normalize()
    d = pd.Series(d.to_numpy(dtype=float), index=day)
    d = d[~d.index.duplicated(keep="last")]
    start = equity.index[0]
    start = start.tz_convert(None) if start.tzinfo is not None else start
    first = pd.Series([float(equity.iloc[0])], index=[start.normalize() - pd.Timedelta(days=1)])
    return pd.concat([first, d[d.index >= first.index[0] + pd.Timedelta(days=1)]])


def span(equity: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The first and last day whose close the curve's returns cover, as the period tables label them."""
    days = _days(equity)
    return days.index[0] + pd.Timedelta(days=1), days.index[-1]


def periods(equity: pd.Series, freq: str, trips: list[dict] | None = None) -> list[Period]:
    """Each calendar year ("Y") or month ("M") the curve covers: its return, end on end, so the periods compound to
    the curve's end over its start exactly. The first and last are partial unless the curve covers them whole.
    Years also carry their Sharpe (daily returns inside the year), max drawdown and the trades closed in them."""
    days = _days(equity)
    key = days.index.to_period(freq)
    out, prev = [], float(days.iloc[0])
    first_day, last_day = span(equity)
    for p in key[1:].unique():
        values = days[key == p]
        if p == key[0]:  # the starting value's own period: its days after the start
            values = values.iloc[1:]
        if values.empty:
            continue
        end = float(values.iloc[-1])
        partial = ""
        if first_day > p.start_time.normalize() and first_day.to_period(freq) == p:
            partial = f"from {first_day:%d %b}"
        if last_day < p.end_time.normalize():
            partial = f"{partial}, to {last_day:%d %b}" if partial else f"to {last_day:%d %b}"
        row = Period(label=str(p), ret=end / prev - 1, partial=partial)
        if freq == "Y":
            path = pd.concat([pd.Series([prev]), values.reset_index(drop=True)], ignore_index=True)
            rets = path.pct_change().dropna()
            # A partial year's Sharpe on under MIN_SHARPE_DAYS days is noise, so n/a (Advisor (f), 21 Oct 2026).
            row.sharpe = summary(rets)["sharpe"] if len(rets) >= (MIN_SHARPE_DAYS if partial else 2) else None
            row.max_dd = max_drawdown(path)
            row.trades = sum(1 for t in trips or [] if t.get("closed") is not None
                             and pd.Timestamp(t["closed"]).year == p.year)
        out.append(row)
        prev = end
    return out


def compounded(rows: list[Period]) -> float:
    return math.prod(1 + p.ret for p in rows) - 1


@dataclass
class TradeRow:
    n: int
    side: str
    opened: pd.Timestamp
    entry_px: float
    closed: pd.Timestamp | None  # None: still open at the end, marked to market there
    exit_px: float | None
    qty: float
    notional: float
    held: pd.Timedelta
    fees: float
    funding: float
    pnl: float
    pnl_pct: float  # of the equity before the trade (the account's view)
    ret: float  # of the entry notional
    reason: str
    equity_after: float
    drawdown: float  # at the trade's close: the equity after against its daily marked-to-market peak so far
    # M1 item 11 (PM's trade book, 30 Oct): why the trade opened and closed, and the indicator values the rule read then
    entry_rule: str = ""
    entry_signal: str = ""
    exit_rule: str = ""
    exit_signal: str = ""


def _naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert(None) if t.tzinfo is not None else t


def exit_reason(decision: dict | None) -> str:
    """What closed a trip, from its exit order's decision: a resting stop, the venue's liquidation, a target, the
    time stop, or the strategy's own signal."""
    d = decision or {}
    intent, words = d.get("intent"), f"{d.get('reason', '')} {(d.get('signal') or {}).get('rule', '')}".lower()
    if intent == "liquidation":
        return "liquidation"
    if intent == "stop_loss" or ".stop" in words or "past the stop" in words:
        return "stop"
    if intent == "take_profit" or ".target" in words or "past the target" in words:
        return "target"
    if "time stop" in words or "time_stop" in words:
        return "time stop"
    return "signal"


def signal_values(decision: dict | None) -> str:
    """The indicator values a rule-builder order's rule read at its decision ("fast=101.2; slow=100.9", by name), from
    its lineage payload; "" for an order that carries none (a hand-coded model's, a stop resting at the venue)."""
    blocks = ((decision or {}).get("signal") or {}).get("blocks") or {}
    return "; ".join(f"{k}={v!r}" for k, v in sorted(blocks.items()))


def _rule(decision: dict | None) -> str:
    d = decision or {}
    return str((d.get("signal") or {}).get("rule") or d.get("reason") or "")


def trade_rows(run) -> list[TradeRow]:
    """Every trip of the run, oldest first, after fees and funding, as the trading table counts them (R-16), and a
    trip still open at the end as the last row, marked to market at the end: so the rows' P&L sums to the end
    equity minus the start and the last row's equity after is the end equity."""
    eq = run.equity
    idx = pd.DatetimeIndex([_naive(t) for t in eq.index])
    values = eq.to_numpy(dtype=float)
    start, end = float(values[0]), float(values[-1])
    trips = trades(fills_to_rows(run.fills), run.shorts, run.funding, getattr(run, "insurance", None), open_trip=True)
    decisions = getattr(run, "decisions", {}) or {}
    rows, booked = [], start
    for n, t in enumerate(trips, 1):
        opened = _naive(t["opened"])
        before_at = int(idx.searchsorted(opened, side="left")) - 1
        before = float(values[before_at]) if before_at >= 0 else start
        if t["closed"] is None:  # open at the end: what the equity made since the last close
            closed, after_at, pnl = None, len(values) - 1, end - booked
        else:
            closed = _naive(t["closed"])
            after_at = min(int(idx.searchsorted(closed, side="left")), len(values) - 1)
            pnl = float(t["pnl"])
            booked += pnl
        after = end if closed is None else float(values[after_at])
        peak = float(values[: after_at + 1].max())
        rows.append(TradeRow(
            n=n, side="long" if t["side"] > 0 else "short", opened=opened, entry_px=float(t["entry_px"]),
            closed=closed, exit_px=None if closed is None else float(t["exit_px"]), qty=float(t["qty"]),
            notional=float(t["cost"]), held=(closed if closed is not None else idx[-1]) - opened,
            fees=float(t["fees"]), funding=float(t.get("funding", 0.0)), pnl=pnl,
            pnl_pct=pnl / before if before else 0.0, ret=pnl / float(t["cost"]) if t["cost"] else 0.0,
            reason="open at end, marked to market" if closed is None else exit_reason(decisions.get(t["exit_order"])),
            equity_after=after, drawdown=after / peak - 1 if peak > 0 else 0.0,
            entry_rule=_rule(decisions.get(t.get("entry_order"))),
            entry_signal=signal_values(decisions.get(t.get("entry_order"))),
            exit_rule="" if closed is None else _rule(decisions.get(t["exit_order"])),
            exit_signal="" if closed is None else signal_values(decisions.get(t["exit_order"]))))
    return rows


CSV_COLUMNS = ("trade", "side", "entry_time", "entry_price", "exit_time", "exit_price", "size", "notional",
               "holding_hours", "fees", "funding", "pnl", "pnl_pct_of_equity", "return_on_notional", "exit_reason",
               "equity_after", "drawdown_at_trade_close", "entry_rule", "entry_signal_values", "exit_rule",
               "exit_signal_values")
# The test-window trade list (M1 item 9): each window's trades, as its own run booked them, and whether the sheet's
# out-of-sample trade count counts it (opened and closed inside the window; one still open at its end is marked there).
OOS_COLUMNS = ("window", "window_start", "window_end", "counted", *CSV_COLUMNS)


def _cell(x) -> str:
    if x is None:
        return ""
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, float):
        return repr(x)
    return '"' + str(x).replace('"', '""') + '"' if any(c in str(x) for c in ',"\n') else str(x)


def _fields(r: TradeRow) -> tuple:
    return (r.n, r.side, r.opened, r.entry_px, r.closed, r.exit_px, r.qty, r.notional, r.held / pd.Timedelta(hours=1),
            r.fees, r.funding, r.pnl, r.pnl_pct, r.ret, r.reason, r.equity_after, r.drawdown, r.entry_rule,
            r.entry_signal, r.exit_rule, r.exit_signal)


def trades_csv(rows: list[TradeRow]) -> str:
    lines = [",".join(CSV_COLUMNS)]
    lines += [",".join(_cell(x) for x in _fields(r)) for r in rows]
    return "\n".join(lines) + "\n"


def oos_trades_csv(windows: list[tuple[pd.Timestamp, pd.Timestamp, list[TradeRow]]]) -> str:
    """Every test window's trades (OOS_COLUMNS), window by window: when it starts trading (the close before its first
    bar), its last bar's close, and its rows."""
    lines = [",".join(OOS_COLUMNS)]
    for k, (first, last, rows) in enumerate(windows, 1):
        for r in rows:
            counted = r.closed is not None and r.opened >= _naive(first)
            lines.append(",".join(_cell(x) for x in (k, _naive(first), _naive(last), "yes" if counted else "no",
                                                     *_fields(r))))
    return "\n".join(lines) + "\n"


def equity_csv(strategy: pd.Series, hold: pd.Series) -> str:
    s, h = _days(strategy), _days(hold)
    both = pd.concat([s.rename("strategy"), h.rename("buy_and_hold")], axis=1)
    return "day,strategy,buy_and_hold\n" + "".join(
        f"{d:%Y-%m-%d},{'' if math.isnan(a) else repr(float(a))},{'' if math.isnan(b) else repr(float(b))}\n"
        for d, a, b in zip(both.index, both["strategy"], both["buy_and_hold"]))


def equity_svg(strategy: pd.Series, hold: pd.Series, title: str) -> str:
    """The two curves at their daily closes as a self-contained SVG: a viewBox, no script, every text escaped."""
    s, h = _days(strategy), _days(hold)
    w, ht, pad = 800, 300, 40
    lo, hi = min(s.min(), h.min()), max(s.max(), h.max())
    span = (hi - lo) or 1.0
    t0, t1 = s.index[0], max(s.index[-1], h.index[-1])
    days = max((t1 - t0).days, 1)

    def path(series: pd.Series) -> str:
        pts = (f"{pad + (d - t0).days / days * (w - 2 * pad):.1f},{ht - pad - (v - lo) / span * (ht - 2 * pad):.1f}"
               for d, v in series.items() if not math.isnan(v))
        return "M" + " L".join(pts)

    label = escape(title)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {ht}" role="img" aria-label="{label}">'
            f'<title>{label}</title><rect width="{w}" height="{ht}" fill="#fff"/>'
            f'<path d="{path(h)}" fill="none" stroke="#999" stroke-width="1.5"/>'
            f'<path d="{path(s)}" fill="none" stroke="#1f5fbf" stroke-width="2"/>'
            f'<text x="{pad}" y="20" font-size="12" fill="#1f5fbf">Strategy</text>'
            f'<text x="{pad + 70}" y="20" font-size="12" fill="#777">Buy and hold</text>'
            f'<text x="{pad}" y="{ht - 10}" font-size="11" fill="#555">{t0:%d %b %Y}</text>'
            f'<text x="{w - pad}" y="{ht - 10}" font-size="11" fill="#555" text-anchor="end">{t1:%d %b %Y}</text>'
            f'<text x="4" y="{pad}" font-size="11" fill="#555">{hi:,.0f}</text>'
            f'<text x="4" y="{ht - pad}" font-size="11" fill="#555">{lo:,.0f}</text></svg>\n')
