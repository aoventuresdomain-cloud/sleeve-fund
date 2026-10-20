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
from sleeve_fund.money import scale
from sleeve_fund.open_risk import ATR_DAYS
from sleeve_fund.portfolio import netting as nt
from sleeve_fund.portfolio.regime_weight import Weighting
from sleeve_fund.research import metrics, trials
from sleeve_fund.research.portfolio import (
    BATCH_DELAY_NS,
    CloseBatch,
    PortfolioGate,
    _utc,
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
    risk_profile: str | None = None  # paper's: its position cap and leverage, as run_backtest's (B7 replays paper)


@dataclass
class PortfolioResult:
    legs: dict[str, BacktestResult]  # each Strategy's own record, as run_backtest gives it, over the whole window
    book: pd.Series  # the fund's equity at every mark (each hour's close), the whole window
    decisions: list[dict]  # every gate decision, in order: strategy, ts, bar_ts, outcome, approved, requested, limit,
    # reason, side, price (the expected fill the gate priced it at); bar_ts keys it against the paper journal (parity)
    halt: dict | None  # {"ts", "drawdown", "reason"} when the drawdown halt fired
    halt_flatten_cost: float  # the fees and half spread the halt's flatten paid (R2)
    # {"ts", "reason", "until"} for each daily pause. By design (Advisor 03:25 UK) a pause lifts at 00:00 UTC, when a
    # daily Strategy decides, so it can't block a daily Strategy's entry; it bites on Strategies deciding intraday.
    pauses: list[dict]
    worst_intrabar_drawdown: float  # the deepest the book went inside an hour, from its high-water mark (R1)
    intrabar_halt_risk: dict  # {"halt": days, "pause": days}: crossed inside an hour, never on a mark (R1)
    # Inside the window, what couldn't be measured (R5, CR F227-1): {ts, strategy, why, what}, what "entry" for an order
    # refused, "open_risk" for an hour whose open risk wasn't marked (a held stopless position's daily ATR missing).
    data_gaps: list[dict]
    unallocated: float = 0.0
    # Every refusal, by Strategy and why: a limit, "halt", "pause", "portfolio_state_stale", or "unmeasurable" (the
    # gate couldn't measure the order, e.g. a stopless entry with no daily ATR yet: CR F220-1, R5).
    refusals: dict = field(default_factory=dict)
    profile: PortfolioProfile = PORTFOLIO
    portfolio_runs_tried: int | None = None  # set by record(): every portfolio run in the project, this one included
    # With run_portfolio(netting=True): step B in shadow beside this run (P2-3, netting_shadow), else None. It never
    # changes anything above: every figure there is step A's.
    netting: dict | None = None
    # Hourly marks whose open risk couldn't be measured (Advisor 13 Oct 16:20 UK, MUST): their equity, drawdown halt
    # and daily pause were still marked; only open risk wasn't, and every entry was refused meanwhile.
    open_risk_unmarked_hours: int = 0

    def summary(self) -> dict:
        """The fund's headline metrics from the book's daily closes, over the whole window: after a halt the flat
        days count too, never cut at the halt (R3)."""
        return metrics.summary(metrics.daily_returns(self.book))

    def trials_rows(self, dataset: str, member_trial_ids: dict[str, str], counted_ids: set[str]) -> list[dict]:
        """The run for the trials register (Data Architect 8 Oct): one row, source 'backtest', kind 'portfolio_run',
        whose settings list each member Strategy with its own trial id. It never counts in the deflated Sharpe's N,
        and it is never evidence for one member's edge: each member's edge is judged on its own row (Advisor 8 Oct
        04:05 UK). So every member must already have a counted row of its own (counted_ids, the register's
        counted_ids()), and a run with a member that hasn't is refused (MUST)."""
        missing = [n for n in self.legs if member_trial_ids.get(n) not in counted_ids]
        if missing:
            raise ValueError(f"a portfolio run is recorded only when every member has a counted trial of its own; "
                             f"none for {', '.join(missing)}")
        start, end = self.book.index[0].to_pydatetime(), self.book.index[-1].to_pydatetime()
        members = [{"name": n, "strategy": r.strategy, "params": r.params, "trial_id": member_trial_ids[n]}
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
        return [{**run, "kind": "portfolio_run"}]

    def record(self, register: trials.TrialsRegister, dataset: str, member_trial_ids: dict[str, str]) -> str:
        """Write the run to the trials register (see trials_rows) and set portfolio_runs_tried: how many portfolio
        runs the project has tried, this one included, since choosing members is itself selection (Advisor 8 Oct
        04:05 UK). Returns the run's trial id."""
        (row,) = self.trials_rows(dataset, member_trial_ids, register.counted_ids())
        register.store.add_trials([row])
        self.portfolio_runs_tried = register.portfolio_runs()
        return row["id"]


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
                  log_level: str = "ERROR", netting: bool = False, regime: Weighting | None = None,
                  fund_sized: bool = False) -> PortfolioResult:
    """Run every Strategy in `specs` together, in their order (the gate's fixed order), each on its own venue clone,
    under the portfolio limits in `profile`. unallocated: the fund's cash held by no Strategy, counted in the book
    (Done-when 3). netting: also report step B in shadow (P2-3, netting_shadow) on the result; off by default, and
    it never changes the run. regime: the P2-5 weight on each new entry, for the counted regime variant only; None
    (always, in G1) runs unweighted. fund_sized: every Strategy sizes its entries off the fund's book as the gate last
    read it (BRIDGE-3, as paper sizes off the supervisor's last mark), not its own account: a multi-asset Strategy's
    run (research.multi_asset); off, every run sizes as before."""
    _check(specs)
    if netting:
        _netting_taker(specs)  # F265-1: a taker mismatch is refused before the run, not after it
    order = tuple(s.name for s in specs)
    # A Strategy deciding on bars as short as its execution bars is fed its decision bars alone (run_backtest's
    # own rule); its exec_prices still mark it every hour.
    plans = [plan_run(s.strategy, s.prices, s.instrument, s.params, s.capital, bar_minutes=s.bar_minutes,
                      exec_prices=None if s.exec_minutes == s.bar_minutes else s.exec_prices,
                      exec_minutes=s.exec_minutes, half_spread=s.half_spread, fees=s.fees,
                      warmup_prices=s.warmup_prices, risk_profile=s.risk_profile) for s in specs]
    hours = {s.name: _closes(s.exec_prices) for s in specs}
    grid = sorted(set().union(*(h.index for h in hours.values())))
    window = int(min(s.prices.index[0] for s in specs).value)
    legs: list = []
    marks: dict[int, tuple[float, list[tuple[str, float, float]]]] = {}  # hour -> (book, [(name, qty, close)])
    actions: list[tuple[int, str, str, object]] = []  # (ts, "halt" | "pause", why, paused until)
    last: dict = {}
    unmarked: set[int] = set()  # hours whose open risk wasn't marked
    running = {"on": True}  # the engine can still take orders

    def book(ts_ns: int):
        """The fund at ts_ns: equity and every held position. A held stopless position whose daily ATR is missing still
        counts its equity, so the drawdown halt and the daily pause are marked every hour; only its open risk can't be
        measured, so it is left out of the holdings and named in last["missing"], and every entry is refused until the
        figure exists (Advisor 13 Oct 16:20 UK, MUST; CR F227-1, F237-1)."""
        total, held, seen, missing = Decimal(repr(float(unallocated))), [], [], []
        for leg, spec in zip(legs, specs):
            h = hours[spec.name]
            at = h.index.searchsorted(pd.Timestamp(ts_ns, tz="UTC"), side="right") - 1
            if at < 0:
                # Before the window's first hourly close (a window opening inside an hour, as B7's replay of paper):
                # its last execution bar's close, so the Strategy's own equity is in the book from its first decision.
                h = spec.exec_prices
                at = h.index.as_unit("ns").searchsorted(pd.Timestamp(ts_ns, tz="UTC"), side="right") - 1
                if at < 0:
                    continue
            close = float(h["close"].iloc[at])
            atr = leg.strategy._daily_atr.get((ts_ns - 1) // DAY_NS * DAY_NS)
            try:
                equity, holding = leg.strategy.portfolio_book(spec.name, close, atr)
                qty = float(holding.notional) / close if holding is not None else 0.0
            except ValueError as e:  # its open risk can't be measured; its equity still can
                eq, _, qty, _ = leg.strategy._mark(close)
                equity, holding = Decimal(repr(float(eq))), None
                missing.append(f"{spec.name}: {e}")
            total += equity
            if holding is not None:
                held.append(holding)
            seen.append((spec.name, qty, close))
        last["book"], last["missing"] = (float(total), seen), missing
        return total, tuple(held)

    def unmeasured() -> str | None:
        """Why the book just read has open risk it couldn't measure, for the gate to refuse an entry on."""
        return "; ".join(last.get("missing") or []) or None

    def mark(hour_ns: int) -> None:
        """The hourly mark (R1): the gate marks the book, and the run keeps it for the result. An hour whose open risk
        can't be measured is still marked, so its drawdown halt and daily pause act (Advisor 13 Oct 16:20 UK, MUST);
        it is counted as open risk unmarked and each Strategy short of its figure is reported as a data gap."""
        gate.mark(hour_ns + BATCH_DELAY_NS)
        marks[hour_ns] = last["book"]
        for why in last["missing"]:
            unmarked.add(hour_ns)
            gate.ledger.alert("portfolio_open_risk_unmarked", why, _utc(hour_ns + BATCH_DELAY_NS))

    def act(ts_ns: int, what: str) -> None:
        st = gate.ledger.state()
        actions.append((ts_ns, what, st.halted if what == "halt" else st.paused, st.paused_until))
        if what == "halt" and running["on"]:
            why = st.halted
            for leg in legs:  # every Strategy, in the run's order, in this one pass (Done-when 3)
                leg.strategy.portfolio_halt(why)

    gate = PortfolioGate(book, profile, on_action=act, unmeasured=unmeasured, regime=regime)
    batch = CloseBatch(order, gate)
    engine = new_engine(log_level)
    try:
        for n, (spec, plan) in enumerate(zip(specs, plans), start=1):
            clone = clone_instrument(spec.instrument, clone_venue(spec.instrument.id.venue, n))
            join = lambda s, name=spec.name: s.join_portfolio(batch, name)  # noqa: E731
            legs.append(add_leg(engine, plan, clone, before_add=join, order_id_tag=f"{n:03d}"))
        if fund_sized:
            last["book"] = (float(sum(s.capital for s in specs)) + float(unallocated), [])  # before the first read
            for leg in legs:
                leg.strategy.fund_equity = lambda: last["book"][0]
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
    result = _result(results, spread, marks, actions, gate, hours, window, profile, unallocated)
    result.open_risk_unmarked_hours = sum(1 for t in unmarked if t >= window)
    if netting:
        result.netting = netting_shadow(results, specs, hours, unallocated)
    return result


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
    # bar_ts: the deciding bar's close, as paper's journal keys a decision: both gates tag it (tagged_intent).
    decisions = [{"strategy": r.strategy, "ts": r.at, "bar_ts": r.intent.bar_ts, "kind": r.intent.kind,
                  "outcome": r.decision.outcome, "approved": r.decision.approved_qty,
                  "requested": r.decision.requested_qty, "limit": r.decision.limit_hit, "reason": r.decision.reason,
                  "side": r.intent.side, "price": r.intent.price, "regime_weight": r.intent.regime_weight,
                  "regime_state": r.intent.regime_state,
                  "bar_decided": r.intent.intent_id not in gate.unbarred} for r in gate.ledger.records]
    gaps = [{"ts": at, "strategy": msg.split(":", 1)[0], "why": msg.split(":", 1)[1].strip(),
             "what": "entry" if kind == "portfolio_check_failed" else "open_risk"}
            for kind, msg, at in gate.ledger.alerts
            if kind in ("portfolio_check_failed", "portfolio_open_risk_unmarked")
            and int(pd.Timestamp(at).value) >= window]
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
        if profile.drawdown is not None and low <= hwm * (1 - profile.drawdown):  # None: switched off (B-32)
            crossed["halt"].add(d)
        if profile.daily_loss is not None and low <= start_of_day * (1 - profile.daily_loss):
            crossed["pause"].add(d)
    return worst_dd, {k: len(crossed[k] - marked[k] - (marked["halt"] if k == "pause" else set())) for k in crossed}


def _fee(commissions) -> float:
    return sum(float(str(m).split()[0]) for m in (commissions if isinstance(commissions, (list, tuple)) else [commissions]))


def _exact(x: float) -> Decimal:
    return scale(1, float(x))


def netting_shadow(results: dict[str, BacktestResult], specs: list[LegSpec], hours: dict[str, pd.DataFrame],
                   unallocated: float = 0.0) -> dict:
    """Step B in shadow over a portfolio run's legs (P2-3 step 2; Advisor 13 Oct 19:10 UK, specs-p2-3-p2-8-p2-5.md §2):
    what one account per venue would have traded had each bar's intents in an instrument gone out as one net order,
    from what step A recorded, nothing re-simulated. Strategies on the same instrument net together whatever venue
    clone they ran on (each leg's clone stands for its own account; the shadow asks what one account would do).

    - Fills: each leg's own, at the price and fee it booked, stamped when they filled (intents decided on one bar fill
      together, at its close + 1 ns), so they group by bar.
    - Where intents cross, the net order is priced as step A prices a market order: the decision mark (the signal's
      close) moved by the half spread that bar's market fills paid.
    - Funding: each leg's own payments; the account pays the settlement's rate on the net position. Linear in the
      position, so there is no funding saving line: the difference is rounding, left to the residual (Advisor 13 Oct
      20:25 UK), but for the baseline for a missing rate, which has its own line (below).

    Returns:
    - saving: the book's lines by kind (crossed, fee), each + where the account saved;
    - account, sub_ledger_total: the account's cash move and the sub-ledgers' summed; funding: the account's funding
      and the sub-ledgers', which reconcile to within the residual once the next line is taken out;
    - conservative_funding_baseline: on settlements charged the baseline for a missing rate (every side pays), the
      sub-ledgers' charges less the account's on its net position, by formula (Advisor 13 Oct 20:55 UK); 0 when every
      rate is present;
    - crossed_notional: the volume matched inside the account instead of sent, at its mark;
    - orders_sent / orders_alone: net orders with something to send, against step A's fills;
    - residual: reconcile() of the sub-ledgers, the book's lines and the baseline line against the account, from the
      fund's starting equity: here the funding's rounding only (the account is built from the same fills); on paper it bites against
      the venue's own account;
    - sub_ledger_vs_step_a: per Strategy, its sub-ledger P&L (cash plus the open position at the last close) less its
      step A equity change, which should be nil but for float rounding;
    - opposed_exposure: per Strategy pair, the share of their gross notional that cancels, at each day's close;
    - notional_peak: the peak gross notional of the virtual positions summed (what step A's accounts margin) and of
      the net positions (what one account would margin), across instruments, at each day's close. Information only:
      the margin netting frees is never P&L (Advisor 13 Oct 20:25 UK)."""
    fills: list[nt.Fill] = []
    marks: dict[tuple[str, int], Decimal] = {}
    halves: dict[tuple[str, int], float] = {}
    for spec in specs:
        r, inst = results[spec.name], str(spec.instrument.id)
        if r.fills is None or r.fills.empty:
            continue
        for oid, f in r.fills.iterrows():
            qty = float(f["filled_qty"])
            if not qty:
                continue
            # The bar's close, to the microsecond: an exit fills at it and the flip's entry 1 ns after, one decision
            ts, px = int(pd.Timestamp(f["ts_last"]).value) // 1_000 * 1_000, float(f["avg_px"])
            side = 1 if str(f["side"]).endswith("BUY") else -1
            fills.append(nt.Fill(spec.name, inst, ts, side * _exact(qty), _exact(px), _exact(_fee(f["commissions"]))))
            close = ((r.decisions.get(oid) or {}).get("signal") or {}).get("close")
            if close and (inst, ts) not in marks:
                marks[(inst, ts)] = _exact(close)
                halves[(inst, ts)] = abs(px / float(close) - 1)
    funding = _funding_rows(results, specs, fills)
    first = {}
    for f in fills:
        first.setdefault((f.instrument, f.ts), f)

    def mark(inst: str, ts: int) -> Decimal:  # a cross with no decision close (stops only): the first fill's price
        return marks.get((inst, ts), first[(inst, ts)].price)

    def half(inst: str, ts: int) -> float:
        return halves.get((inst, ts), next(float(s.half_spread or results[s.name].half_spread) for s in specs
                                           if str(s.instrument.id) == inst))

    sh = nt.shadow(fills, funding, mark, half, _netting_taker(specs))
    saving = sh.saving()
    sub = sh.strategy_cash()
    equity = sum((_exact(s.capital) for s in specs), _exact(unallocated))
    # The residual is measured without the conservative funding baseline line (Advisor 13 Oct 20:55 UK)
    residual = nt.reconcile(sum(sub.values(), Decimal(0)), sum(saving.values(), Decimal(0)) + sh.baseline_total(),
                            sh.account, equity)
    # Just after each day's close (1 us), so a flip at the close counts as the position it left
    days = sorted({int(t.value) + 1_000 for h in hours.values() for t in h.index if t.value % DAY_NS == 0})
    price = _closes_at(specs, hours)
    return {
        "saving": {k: float(v) for k, v in saving.items()},
        "account": float(sh.account), "sub_ledger_total": float(sum(sub.values(), Decimal(0))),
        "conservative_funding_baseline": float(sh.baseline_total()),
        "funding": {"account": float(sh.account_funding),
                    "sub_ledgers": float(sum((r.amount for r in funding), Decimal(0)))},
        "crossed_notional": float(sum((o.crossed * mark(o.instrument, o.ts) for o in sh.orders), Decimal(0))),
        "orders_sent": sum(1 for o in sh.orders if o.qty), "orders_alone": len(fills),
        "residual": {k: (float(v) if isinstance(v, Decimal) else v) for k, v in residual.items()},
        "sub_ledger_vs_step_a": _vs_step_a(results, specs, sub, fills, funding),
        "opposed_exposure": {"/".join(k): v for k, v in nt.opposed_exposure(fills, price, days).items()},
        "notional_peak": _notional_peak(fills, price, days),
    }


def _netting_taker(specs: list[LegSpec]) -> Decimal:
    """The one taker fee the shadow's account pays: every Strategy's, each from its own fee schedule (or its market's).
    Raises ValueError when they differ, before anything runs (F265-1)."""
    taker = {s.fees.taker if s.fees is not None else _instrument_taker(s.instrument) for s in specs}
    if len(taker) > 1:
        raise ValueError("netting_shadow: the Strategies on one account must pay the same taker fee")
    return next(iter(taker)) if taker else Decimal(0)


def _instrument_taker(instrument) -> Decimal:
    return Decimal(str(instrument.taker_fee))


def _funding_rows(results, specs, fills: list[nt.Fill]) -> list[nt.Funding]:
    """Each leg's funding payments as netting.Funding rows. A run records the amount, not the rate, so the settlement's
    rate x mark is read back from a leg holding a position then (-amount / position) and given to every row of that
    settlement: the legs pay one rate on one mark."""
    held: dict[tuple[str, str], list[nt.Fill]] = {}
    for f in fills:
        held.setdefault((f.strategy, f.instrument), []).append(f)

    def position(name: str, inst: str, ts: int) -> Decimal:
        return sum((f.qty for f in held.get((name, inst), ()) if f.ts < ts), Decimal(0))

    raw = []
    for spec in specs:
        inst = str(spec.instrument.id)
        for row in results[spec.name].funding:
            ts = int(pd.Timestamp(row["ts"]).value)
            both = row.get("kind") == "baseline"  # a missing rate: every side pays it (Advisor 6 Oct, D9)
            raw.append((spec.name, inst, ts, _exact(row["amount"]), position(spec.name, inst, ts), both))
    rate: dict[tuple[str, int], Decimal] = {}
    for _, inst, ts, amount, pos, both in raw:
        if pos and (inst, ts) not in rate:
            rate[(inst, ts)] = abs(amount / pos) if both else -amount / pos
    return [nt.Funding(name, inst, ts, rate.get((inst, ts), Decimal(0)), Decimal(1), amount, both)
            for name, inst, ts, amount, _, both in raw]


def _closes_at(specs, hours):
    """The instrument's last hourly close at or before `ts`, as a Decimal (the marks the book used)."""
    by = {str(s.instrument.id): hours[s.name]["close"] for s in specs}

    def at(inst: str, ts: int) -> Decimal:
        h = by[inst]
        i = h.index.searchsorted(pd.Timestamp(ts, tz="UTC"), side="right") - 1
        return _exact(h.iloc[max(i, 0)])

    return at


def _notional_peak(fills: list[nt.Fill], price, days: list[int]) -> dict:
    """The peak, over each day's close, of the summed gross notional of the virtual positions and of the net ones."""
    names, insts = sorted({f.strategy for f in fills}), sorted({f.instrument for f in fills})
    peak_sum = peak_net = Decimal(0)
    for t in days:
        total = net = Decimal(0)
        for inst in insts:
            m = price(inst, t)
            pos = [sum((f.qty for f in fills if f.strategy == n and f.instrument == inst and f.ts <= t), Decimal(0))
                   for n in names]
            total += sum((abs(p) for p in pos), Decimal(0)) * m
            net += abs(sum(pos, Decimal(0))) * m
        peak_sum, peak_net = max(peak_sum, total), max(peak_net, net)
    return {"sum": float(peak_sum), "net": float(peak_net)}


def _vs_step_a(results, specs, sub: dict, fills: list[nt.Fill], funding: list[nt.Funding]) -> dict[str, float]:
    """Per Strategy: its sub-ledger P&L (cash plus the open position at its last decision close, plus any insurance
    the venue's fund took) less the change in its step A equity over the run. Nil but for float rounding (MUST)."""
    out = {}
    for spec in specs:
        r = results[spec.name]
        if r.equity is None or r.equity.empty:
            continue
        pos = sum((f.qty for f in fills if f.strategy == spec.name), Decimal(0))
        last = spec.prices["close"].iloc[-1]
        insurance = sum((_exact(i["amount"]) for i in r.insurance), Decimal(0))
        pnl = sub.get(spec.name, Decimal(0)) + pos * _exact(last) + insurance
        out[spec.name] = float(pnl) - (float(r.equity.iloc[-1]) - float(r.starting_capital))
    return out
