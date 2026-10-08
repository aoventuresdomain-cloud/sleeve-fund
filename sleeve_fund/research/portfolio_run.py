"""Portfolio backtest runner (v2 P2-7 M4): several Strategies in ONE NautilusTrader run, each on its own clone of its
venue with its own account, every opening order through P2-2's gate over one MemoryLedger (research.portfolio), the
whole fund marked on an hourly grid.

The Independent Quant Advisor's rulings (8 Oct 03:05 UK) on the run's risk choices:
- R1: the halt and the pause are detected on the marks, and the marks are never coarser than one hour (canonical
  candles), whatever the Strategies trade: paper marks every few seconds, so a daily mark would understate halts and
  overstate returns. The worst intrabar drawdown is reported beside them, with the days on which an intrabar move
  crossed the halt or the pause but no mark did ("intrabar halt risk").
- R2: a halt flattens every Strategy at the next tradable price, its close + 1 ns market exit through the exit path,
  with no extra haircut (ACT-DRIFT governs); what that flatten cost is reported on its own line.
- R3: after a halt the fund stays flat to the window's end (the PM's Resume has no backtest equivalent); the result
  names the halt and its drawdown, and every figure covers the whole window, the flat part included.
- R4: the daily pause flattens nothing: entries wait for 00:00 UTC; exits and stops carry on.
- R5: an order the gate can't measure (a stopless position before its daily ATR exists) is refused; every leg loads
  at least ATR_DAYS of warm-up before the window, and a refusal inside the window is reported as a data gap."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
from nautilus_trader.common import DataActor, DataActorConfig

from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.open_risk import ATR_DAYS
from sleeve_fund.research import metrics, trials
from sleeve_fund.research.portfolio import (
    BATCH_DELAY_NS,
    CloseBatch,
    PortfolioGate,
    clone_instrument,
    clone_venue,
    strategy_order,
)
from sleeve_fund.research.runner import (
    BacktestResult,
    add_leg,
    feed_legs,
    leg_result,
    new_engine,
    plan_run,
    release_leg,
)
from sleeve_fund.risk import PORTFOLIO, PortfolioProfile

HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
MARK_MINUTES = 60  # the coarsest mark grid there may be (Advisor R1 MUST)
HALTED = "Portfolio halted"  # what every halt-flatten order's reason starts with (LongFlatStrategy.portfolio_halt)


@dataclass(frozen=True)
class LegSpec:
    """One Strategy of a portfolio run. prices: its decision bars over the window, stamped at their close.
    exec_prices: bars of exec_minutes (at most an hour) over the same window: the engine fills against them and the
    book is marked at each hour's close from them. warmup_prices: decision bars before the window, at least ATR_DAYS
    of them (R5). capital: what the Strategy's own account opens with."""

    name: str
    strategy: str
    instrument: Any
    prices: pd.DataFrame
    exec_prices: pd.DataFrame
    warmup_prices: pd.DataFrame
    exec_minutes: int = 60
    bar_minutes: int = 1440
    params: dict = field(default_factory=dict)
    capital: float = 10_000.0
    half_spread: float | None = None
    fees: FeeSchedule | None = None


@dataclass
class PortfolioResult:
    legs: dict[str, BacktestResult]  # each Strategy's own record, as run_backtest gives it, over the whole window
    book: pd.Series  # the fund's equity at every mark (each hour's close), the whole window
    decisions: list[dict]  # every gate decision, in order: strategy, ts, outcome, approved, requested, limit, reason
    halt: dict | None  # {"ts", "drawdown", "reason"} when the drawdown halt fired
    halt_flatten_cost: float  # the fees and half spread the halt's flatten paid (R2)
    # {"ts", "reason", "until"} for each daily pause. By design (Advisor 03:25 UK) a pause lifts at 00:00 UTC, when a
    # daily Strategy decides, so it can't block a daily Strategy's entry; it bites on Strategies deciding intraday.
    pauses: list[dict]
    worst_intrabar_drawdown: float  # the deepest the book went inside an hour, from its high-water mark (R1)
    intrabar_halt_risk: dict  # {"halt": days, "pause": days}: crossed inside an hour, never on a mark (R1)
    data_gaps: list[dict]  # orders refused inside the window because a figure couldn't be measured (R5)
    unallocated: float = 0.0
    # Every refusal, by Strategy and why: a limit, "halt", "pause", "portfolio_state_stale", or "unmeasurable" (the
    # gate couldn't measure the order, e.g. a stopless entry with no daily ATR yet: CR F220-1, R5).
    refusals: dict = field(default_factory=dict)
    profile: PortfolioProfile = PORTFOLIO

    def summary(self) -> dict:
        """The fund's headline metrics from the book's daily closes, over the whole window: after a halt the flat
        days count too, never cut at the halt (R3)."""
        return metrics.summary(metrics.daily_returns(self.book))

    def trials_rows(self, dataset: str, member_trial_ids: dict[str, str] | None = None) -> list[dict]:
        """The run for the trials register (Data Architect 8 Oct): one row, source 'backtest', kind 'portfolio_run',
        whose settings list each member Strategy with its own trial id. member_trial_ids: the members' rows already
        counted; any not given gets its own row here, returned after the run's. Whether a portfolio run counts in
        the deflated Sharpe's N is the Advisor's call."""
        ids, rows = dict(member_trial_ids or {}), []
        start, end = self.book.index[0].to_pydatetime(), self.book.index[-1].to_pydatetime()
        for name, r in self.legs.items():
            if name not in ids:
                row = trials.model_run_row(strategy=r.strategy, params=r.params, dataset=dataset, source="backtest",
                                           setup={"portfolio_member": name}, data_start=start, data_end=end,
                                           trades=0 if r.fills is None else len(r.fills))
                ids[name] = row["id"]
                rows.append(row)
        members = [{"name": n, "strategy": r.strategy, "params": r.params, "trial_id": ids[n]}
                   for n, r in self.legs.items()]
        settings = {"members": members, "profile": asdict(self.profile), "unallocated": self.unallocated}
        key = {k: v for k, v in settings.items() if k != "members"} | {
            "members": [[m["name"], trials.legacy_definition_hash(m["strategy"], m["params"])] for m in members]}
        try:
            sharpe = self.summary()["sharpe"]
        except ValueError:  # under two days of book
            sharpe = None
        ideas = sorted(trials.legacy_idea_hash(m["strategy"]) for m in members)
        run = trials.trial_row(
            definition_hash=trials.content_hash(key), idea_hash=trials.content_hash({"portfolio": ideas}),
            name="portfolio:" + "+".join(self.legs), family="portfolio", settings=settings, dataset=dataset,
            stage="in_sample", source="backtest", sharpe=sharpe,
            trades=sum(0 if r.fills is None else len(r.fills) for r in self.legs.values()),
            data_start=start, data_end=end)
        return [{**run, "kind": "portfolio_run"}, *rows]


class _MarkerConfig(DataActorConfig):
    def __init__(self, *, first_ns: int, last_ns: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.first_ns, self.last_ns = first_ns, last_ns


class _Marker(DataActor):
    """Marks the fund at every hour's close + BATCH_DELAY_NS (R1), entries or not."""

    def __init__(self, config: _MarkerConfig):
        super().__init__(config)
        self._mark, self._first, self._last = None, config.first_ns, config.last_ns

    def on_start(self) -> None:
        first, last = self._first, self._last
        now = self.clock.timestamp_ns()  # the run's first event: its first hour's close, at the earliest
        start = max(first, -(-max(now - BATCH_DELAY_NS, 0) // HOUR_NS) * HOUR_NS) + BATCH_DELAY_NS
        if start <= last + BATCH_DELAY_NS:
            # A timer's first event comes one interval after its start: the first hour has its own alert.
            self.clock.set_time_alert_ns("portfolio-mark-first", start, callback=self._on_timer)
            if start + HOUR_NS <= last + BATCH_DELAY_NS:
                self.clock.set_timer_ns("portfolio-mark", HOUR_NS, start_time_ns=start,
                                        stop_time_ns=last + BATCH_DELAY_NS, callback=self._on_timer)

    def _on_timer(self, event) -> None:
        self._mark(event.ts_event - BATCH_DELAY_NS)


def _closes(exec_prices: pd.DataFrame) -> pd.DataFrame:
    """Hourly bars from bars of an hour or less stamped at their close: each hour's high, low and last close."""
    period = pd.Timedelta(minutes=MARK_MINUTES)
    hour = (exec_prices.index - pd.Timedelta(1, "ns")).floor(period) + period
    return exec_prices[["high", "low", "close"]].groupby(hour).agg({"high": "max", "low": "min", "close": "last"})


def _check(specs: list[LegSpec]) -> None:
    strategy_order(s.name for s in specs)
    for s in specs:
        if s.exec_prices is None or s.exec_prices.empty or s.exec_minutes > MARK_MINUTES:
            raise ValueError(f"{s.name}: a portfolio run marks the fund every hour (Advisor R1), so each Strategy "
                             f"needs bars of an hour or less over the window (exec_prices), not "
                             f"{s.exec_minutes} minutes")
        warm = s.warmup_prices
        before = None if warm is None or warm.empty else s.prices.index[0] - warm.index[0]
        if before is None or before < pd.Timedelta(days=ATR_DAYS):
            raise ValueError(f"{s.name}: load at least {ATR_DAYS} days of warm-up before the window, so the daily ATR "
                             f"a stopless position is measured by exists from its first day (Advisor R5); it has "
                             f"{before or pd.Timedelta(0)}")


def run_portfolio(specs: list[LegSpec], profile: PortfolioProfile = PORTFOLIO, unallocated: float = 0.0,
                  log_level: str = "ERROR") -> PortfolioResult:
    """Run every Strategy in `specs` together, in their order (the gate's fixed order), each on its own venue clone,
    under the portfolio limits in `profile`. unallocated: the fund's cash held by no Strategy, counted in the book
    (Done-when 3)."""
    _check(specs)
    order = tuple(s.name for s in specs)
    plans = [plan_run(s.strategy, s.prices, s.instrument, s.params, s.capital, bar_minutes=s.bar_minutes,
                      exec_prices=s.exec_prices, exec_minutes=s.exec_minutes, half_spread=s.half_spread, fees=s.fees,
                      warmup_prices=s.warmup_prices) for s in specs]
    hours = {s.name: _closes(s.exec_prices) for s in specs}
    grid = sorted(set().union(*(h.index for h in hours.values())))
    window = int(min(s.prices.index[0] for s in specs).value)
    legs: list = []
    marks: dict[int, tuple[float, list[tuple[str, float, float]]]] = {}  # hour -> (book, [(name, qty, close)])
    actions: list[tuple[int, str, str, object]] = []  # (ts, "halt" | "pause", why, paused until)
    last: dict = {}
    running = {"on": True}  # the engine can still take orders

    def book(ts_ns: int):
        total, held, seen = Decimal(repr(float(unallocated))), [], []
        for leg, spec in zip(legs, specs):
            h = hours[spec.name]
            at = h.index.searchsorted(pd.Timestamp(ts_ns, tz="UTC"), side="right") - 1
            if at < 0:
                continue
            close = float(h["close"].iloc[at])
            atr = leg.strategy._daily_atr.get((ts_ns - 1) // DAY_NS * DAY_NS)
            equity, holding = leg.strategy.portfolio_book(spec.name, close, atr)
            total += equity
            if holding is not None:
                held.append(holding)
            seen.append((spec.name, float(holding.notional) / close if holding is not None else 0.0, close))
        last["book"] = (float(total), seen)
        return total, tuple(held)

    def mark(hour_ns: int) -> None:
        """The hourly mark (R1): the gate marks the book, and the run keeps it for the result."""
        gate.mark(hour_ns + BATCH_DELAY_NS)
        marks[hour_ns] = last["book"]

    def act(ts_ns: int, what: str) -> None:
        st = gate.ledger.state()
        actions.append((ts_ns, what, st.halted if what == "halt" else st.paused, st.paused_until))
        if what == "halt" and running["on"]:
            why = st.halted
            for leg in legs:  # every Strategy, in the run's order, in this one pass (Done-when 3)
                leg.strategy.portfolio_halt(why)

    gate = PortfolioGate(book, profile, on_action=act)
    batch = CloseBatch(order, gate)
    engine = new_engine(log_level)
    try:
        for n, (spec, plan) in enumerate(zip(specs, plans), start=1):
            clone = clone_instrument(spec.instrument, clone_venue(spec.instrument.id.venue, n))
            join = lambda s, name=spec.name: s.join_portfolio(batch, name)  # noqa: E731
            legs.append(add_leg(engine, plan, clone, before_add=join, order_id_tag=f"{n:03d}"))
        first, end = int(grid[0].value), int(grid[-1].value)
        marker = _Marker(_MarkerConfig(first_ns=first, last_ns=end))
        marker._mark = mark
        engine.add_actor(marker)
        feed_legs(engine, legs)
        running["on"] = False
        if end not in marks:
            mark(end)  # the window's last close comes after the engine's last event
        results = {spec.name: leg_result(engine, leg) for spec, leg in zip(specs, legs)}
        spread = {spec.name: dict(leg.fee_model.spread_paid) for spec, leg in zip(specs, legs)}
    finally:
        for leg in legs:
            release_leg(leg)
        engine.dispose()
    return _result(results, spread, marks, actions, gate, hours, window, profile, unallocated)


def _flatten_cost(results: dict[str, BacktestResult], legs_spread: dict[str, dict]) -> float:
    cost = 0.0
    for name, r in results.items():
        if r.fills is None or r.fills.empty:
            continue
        for oid, row in r.fills.iterrows():
            d = r.decisions.get(oid, {})
            if d.get("intent") == "risk_halt" and str(d.get("reason", "")).startswith(HALTED):
                fee = row["commissions"]
                fee = sum(float(str(m).split()[0]) for m in (fee if isinstance(fee, (list, tuple)) else [fee]))
                cost += fee + float(legs_spread[name].get(oid, 0.0))
    return cost


def _result(results, spread, marks, actions, gate, hours, window, profile, unallocated) -> PortfolioResult:
    ts = sorted(marks)
    book = pd.Series([marks[t][0] for t in ts], index=pd.to_datetime(ts, utc=True), name="book")
    halt = next((a for a in actions if a[1] == "halt"), None)
    halt_info = None
    if halt is not None:
        at = pd.Timestamp(halt[0] - BATCH_DELAY_NS, tz="UTC")
        ref = book[book.index <= at]
        halt_info = {"ts": at, "drawdown": 1 - float(ref.iloc[-1]) / float(ref.max()), "reason": halt[2]}
    # A pause set by the mark at exactly 00:00 belongs to the day just ended and lifts at once (gate.mark_book):
    # it blocks nothing, so it isn't listed.
    pauses = [{"ts": pd.Timestamp(t - BATCH_DELAY_NS, tz="UTC"), "reason": why, "until": until}
              for t, w, why, until in actions
              if w == "pause" and until is not None and until > pd.Timestamp(t, tz="UTC")]
    worst, risk = _intrabar(book, marks, hours, profile, halt_info, pauses)
    decisions = [{"strategy": r.strategy, "ts": r.at, "outcome": r.decision.outcome,
                  "approved": r.decision.approved_qty, "requested": r.decision.requested_qty,
                  "limit": r.decision.limit_hit, "reason": r.decision.reason} for r in gate.ledger.records]
    gaps = [{"ts": at, "strategy": msg.split(":", 1)[0], "why": msg.split(":", 1)[1].strip()}
            for kind, msg, at in gate.ledger.alerts
            if kind == "portfolio_check_failed" and int(pd.Timestamp(at).value) >= window]
    refusals: dict[str, dict[str, int]] = {}
    for d in decisions:
        if d["outcome"] == "rejected":
            by = refusals.setdefault(d["strategy"], {})
            by[d["limit"] or "unmeasurable"] = by.get(d["limit"] or "unmeasurable", 0) + 1
    for kind, msg, _ in gate.ledger.alerts:
        if kind == "portfolio_check_failed":
            by = refusals.setdefault(msg.split(":", 1)[0], {})
            by["unmeasurable"] = by.get("unmeasurable", 0) + 1
    return PortfolioResult(results, book, decisions, halt_info, _flatten_cost(results, spread), pauses, worst, risk,
                           gaps, unallocated, refusals, profile)


def _intrabar(book: pd.Series, marks: dict, hours: dict, profile: PortfolioProfile, halt: dict | None,
              pauses: list[dict]) -> tuple[float, dict]:
    """R1: the book at each hour's worst price, each position held from the mark before at that hour's low (a long)
    or high (a short), against the high-water mark and the day's start the marks kept; and the UTC days on which
    that crossed the halt or the pause while no mark did. Fills inside the hour aren't replayed: the positions are
    those of the mark before."""
    ts = sorted(marks)
    # never shallower than what the marks themselves fell to (a halt's flatten included)
    hwm, worst_dd, start_of_day, day = None, float((1 - book / book.cummax()).max()) if len(book) else 0.0, None, None
    crossed = {"halt": set(), "pause": set()}
    marked = {"halt": {halt["ts"].normalize()} if halt else set(), "pause": {p["ts"].normalize() for p in pauses}}
    for prev, now in zip(ts, ts[1:]):
        equity_prev, held = marks[prev]
        stamp = pd.Timestamp(now, tz="UTC")
        hwm = max(hwm or equity_prev, equity_prev)
        d = (stamp - pd.Timedelta(1, "ns")).normalize()  # a mark at 00:00 is the day before's last
        if d != day:
            day, start_of_day = d, equity_prev
        if halt is not None and stamp > halt["ts"]:
            break
        low = equity_prev
        for name, qty, close in held:
            if not qty:
                continue
            h = hours[name]
            if stamp not in h.index:
                continue
            px = float(h.at[stamp, "low"] if qty > 0 else h.at[stamp, "high"])
            low += qty * (px - close)
        worst_dd = max(worst_dd, 1 - low / hwm)
        if low <= hwm * (1 - profile.drawdown):
            crossed["halt"].add(d)
        if low <= start_of_day * (1 - profile.daily_loss):
            crossed["pause"].add(d)
    return worst_dd, {k: len(crossed[k] - marked[k] - (marked["halt"] if k == "pause" else set())) for k in crossed}
