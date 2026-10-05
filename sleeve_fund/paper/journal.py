"""The journal a backtest's runtime writes to: the paper journal's calls, held in memory.

A backtest marks, guards and journals once a bar, as paper does every few seconds. On a database
that is three round trips a bar, most of a long minute-bar run; here it is a list append. When the
run ends, save() copies it into the real journal under a backtest name, so the run's orders, fills,
positions and reasons show in the same screens as paper's.
"""

from __future__ import annotations

import itertools
from datetime import datetime

from sleeve_fund.store import INTENTS, LEVELS, ORDER_STATUSES, STATUSES, Sleeve, exact_sum, utcnow

_FINISHED = ("filled", "canceled", "rejected", "denied", "expired")
KEEP_ALL_MARKS = 5000  # a run with at most this many marks saves every one


class MemoryJournal:
    """The subset of Store a SleeveRuntime uses, in memory, for one sleeve."""

    def __init__(self) -> None:
        self.sleeve_row: Sleeve | None = None
        self.equity: list[dict] = []
        self.fills_: list[dict] = []
        self.funding_: list[dict] = []
        self.insurance_: list[dict] = []
        self.orders_: dict[str, dict] = {}
        self.events_: list[dict] = []
        self.exit_plans_: dict[str, list[dict]] = {}
        self._ids = itertools.count(1)
        self._peak: float | None = None
        # The deepest drawdown over every mark, as (peak mark, trough mark): kept through thinning so the
        # saved run's drawdown is the one the risk guard saw, not the one left between hourly marks.
        self._peak_mark: dict | None = None
        self._worst: tuple[dict, dict] | None = None
        self._hour: tuple | None = None  # the hour of the latest mark, and where its marks start
        self._hour_start = 0
        self._thinned = False

    # --- the sleeve -----------------------------------------------------------------

    def create_sleeve(self, *, name: str, strategy: str, instrument: str, bar_spec: str, starting_balance: float,
                      params: dict | None = None, risk_profile: str = "balanced", warmup_bars: int = 0,
                      desired_state: str = "running") -> Sleeve:
        ts = utcnow()
        self.sleeve_row = Sleeve(id=1, name=name, strategy=strategy, instrument=instrument, bar_spec=bar_spec,
                                 params=dict(params or {}), starting_balance=starting_balance,
                                 risk_profile=risk_profile, warmup_bars=warmup_bars, desired_state=desired_state,
                                 status="starting", status_reason="", paused_until=None, heartbeat_at=None,
                                 created_at=ts, updated_at=ts)
        return self.sleeve_row

    def sleeve(self, name: str) -> Sleeve:
        if self.sleeve_row is None or self.sleeve_row.name != name:
            raise KeyError(f"no strategy {name!r}")
        return self.sleeve_row

    def set_status(self, name: str, status: str, reason: str = "", paused_until: datetime | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        row = self.sleeve(name)
        row.status, row.status_reason, row.paused_until = status, reason, paused_until

    def heartbeat(self, name: str) -> None:
        pass  # nobody watches a backtest's pulse

    def pending_commands(self, sleeve: str) -> list[dict]:
        return []  # a backtest takes no PM commands

    def mark_applied(self, command_id: int) -> None:
        pass

    def record_spread(self, *args, **kwargs) -> None:
        pass  # replayed quotes are not a fresh measurement

    # --- journal --------------------------------------------------------------------

    def record_equity(self, sleeve: str, *, equity: float, cash: float, qty: float, price: float,
                      benchmark: float, ts: datetime | None = None) -> None:
        ts = ts or utcnow()
        hour = (ts.year, ts.month, ts.day, ts.hour)
        if hour != self._hour:
            # Past KEEP_ALL_MARKS only the last mark of each hour is ever saved (marks_to_keep), so a
            # finished hour shrinks to that one now: five years of minutes would otherwise hold 2.6
            # million marks in memory. The first mark of the run always stays.
            if len(self.equity) > KEEP_ALL_MARKS:
                start = max(self._hour_start, 1)
                if len(self.equity) - start > 1:
                    del self.equity[start:-1]
                    self._thinned = True
            self._hour, self._hour_start = hour, len(self.equity)
        mark = {"ts": ts, "equity": equity, "cash": cash, "qty": qty, "price": price, "benchmark": benchmark}
        self.equity.append(mark)
        if self._peak is None or equity > self._peak:
            self._peak, self._peak_mark = equity, mark
        elif self._peak > 0 and (self._worst is None or equity / self._peak < self._worst[1]["equity"] / self._worst[0]["equity"]):
            self._worst = (self._peak_mark, mark)

    def record_fill(self, sleeve: str, *, side: str, qty: float, price: float, fee: float, order_id: str,
                    trade_id: str, ts: datetime | None = None) -> None:
        self.fills_.append({"id": next(self._ids), "sleeve": sleeve, "ts": ts or utcnow(), "side": side, "qty": qty,
                            "price": price, "fee": fee, "order_id": order_id, "trade_id": trade_id})

    def record_order(self, sleeve: str, *, order_id: str, side: str, qty: float, intent: str, reason: str,
                     signal: dict | None = None, order_type: str = "MARKET", ts: datetime | None = None) -> None:
        if intent not in INTENTS:
            raise ValueError(f"bad intent {intent!r}")
        now = ts or utcnow()
        self.orders_[order_id] = {"id": next(self._ids), "sleeve": sleeve, "order_id": order_id, "ts": now,
                                  "updated_at": now, "side": side, "order_type": order_type, "qty": qty,
                                  "status": "submitted", "filled_qty": 0.0, "avg_px": None, "fee": 0.0,
                                  "intent": intent, "reason": reason, "signal": signal or {}, "message": ""}

    def update_order(self, order_id: str, *, status: str | None = None, message: str | None = None,
                     fill_qty: float = 0.0, fill_px: float | None = None, fee: float = 0.0,
                     qty: float | None = None) -> None:
        if status is not None and status not in ORDER_STATUSES:
            raise ValueError(f"bad order status {status!r}")
        row = self.orders_.get(order_id)
        if row is None:
            return
        if qty is not None:
            row["qty"] = qty
        if fill_qty:
            filled = exact_sum(row["filled_qty"], fill_qty)
            row["avg_px"] = ((row["avg_px"] or 0.0) * row["filled_qty"] + fill_qty * fill_px) / filled
            row["filled_qty"] = filled
            row["fee"] += fee
            row["status"] = "filled" if filled >= row["qty"] - 1e-12 else "partially_filled"
        if status is not None and row["status"] not in _FINISHED:
            row["status"] = status
        if message:
            row["message"] = message
        if self.equity:
            row["updated_at"] = self.equity[-1]["ts"]

    def event(self, sleeve: str | None, level: str, kind: str, message: str, ts: datetime | None = None) -> None:
        if level not in LEVELS:
            raise ValueError(f"bad level {level!r}")
        self.events_.append({"id": next(self._ids), "sleeve": sleeve, "ts": ts or utcnow(), "level": level,
                             "kind": kind, "message": message})

    # --- reads, newest first like Store -------------------------------------------------

    def events(self, sleeve: str | None = None, limit: int = 100, min_level: str = "info") -> list[dict]:
        levels = LEVELS[LEVELS.index(min_level):]
        return [e for e in reversed(self.events_) if e["level"] in levels][:limit]

    def last_event(self, sleeve: str, kinds: tuple[str, ...]) -> dict | None:
        return next((e for e in reversed(self.events_) if e["kind"] in kinds), None)

    def sleeve_events_since(self, sleeve: str, kinds: tuple[str, ...], after_id: int = 0) -> list[dict]:
        return [e for e in self.events_ if e["kind"] in kinds and e["id"] > after_id]

    # Exit plans set after entry (see Store.set_exit_plan). A backtest starts flat and its settings don't
    # change mid-run, so it rarely has any; they are kept for the same calls.
    def set_exit_plan(self, sleeve: str, entry_order: str, **plan) -> None:
        self.exit_plans_.setdefault(entry_order, []).append({"entry_order": entry_order, **plan})

    def exit_plan(self, sleeve: str, entry_order: str) -> dict | None:
        return (self.exit_plans_.get(entry_order) or [None])[-1]

    def exit_plans(self, sleeve: str) -> dict[str, dict]:
        return {k: v[-1] for k, v in self.exit_plans_.items() if v}

    def orders(self, sleeve: str | None = None, statuses: tuple[str, ...] | None = None, limit: int = 500) -> list[dict]:
        rows = sorted(self.orders_.values(), key=lambda o: (o["ts"], o["id"]), reverse=True)
        return [o for o in rows if not statuses or o["status"] in statuses][:limit]

    def fills(self, sleeve: str | None = None, limit: int = 200) -> list[dict]:
        return list(reversed(self.fills_))[:limit]

    def peak_equity(self, sleeve: str, since: datetime | None = None) -> float | None:
        if since is None:
            return self._peak
        return max((float(m["equity"]) for m in self.equity if m["ts"] >= since), default=None)

    def equity_at_or_before(self, sleeve: str, ts: datetime) -> dict | None:
        return next((m for m in reversed(self.equity) if m["ts"] <= ts), None)

    def first_equity(self, sleeve: str) -> dict | None:
        return self.equity[0] if self.equity else None

    def day_open_equity(self, sleeve: str, day_start) -> float | None:
        before = [m for m in self.equity if m["ts"] < day_start]
        if before:
            return float(before[-1]["equity"])
        return next((float(m["equity"]) for m in self.equity if m["ts"] >= day_start), None)

    def last_equity(self, sleeve: str) -> dict | None:
        return self.equity[-1] if self.equity else None

    def equity_series(self, sleeve: str, limit: int = 5000) -> list[dict]:
        return self.equity[-limit:]

    def journal_book(self, sleeve: str, starting_balance: float) -> dict:
        from sleeve_fund.store import replay_book

        return replay_book(self.fills_, starting_balance, self.funding_total(sleeve), self.insurance_total(sleeve))

    def record_funding(self, sleeve: str, *, qty: float, price: float, rate: float, amount: float,
                       ts: datetime | None = None) -> None:
        self.funding_.append({"sleeve": sleeve, "ts": ts, "qty": qty, "price": price, "rate": rate, "amount": amount})

    def funding(self, sleeve: str, limit: int = 1000) -> list[dict]:
        return list(reversed(self.funding_))[:limit]

    def funding_total(self, sleeve: str) -> float:
        return float(sum(f["amount"] for f in self.funding_))

    def record_insurance(self, sleeve: str, *, price: float, amount: float, ts: datetime | None = None) -> None:
        self.insurance_.append({"sleeve": sleeve, "ts": ts, "price": price, "amount": amount})

    def insurance(self, sleeve: str, limit: int = 1000) -> list[dict]:
        return list(reversed(self.insurance_))[:limit]

    def insurance_total(self, sleeve: str) -> float:
        return float(sum(f["amount"] for f in self.insurance_))

    # --- into the real journal ---------------------------------------------------------

    def marks_to_keep(self) -> list[dict]:
        """Equity marks worth saving: every mark of a short run; the last of each hour (or, past
        90 days, of each day) of a long one, which is what the screens chart anyway. The first and
        last marks are always kept, so the run's start and end values are exact, and so are the peak
        and trough of the deepest drawdown, so the maximum drawdown is exact too."""
        marks = self.equity
        if len(marks) <= KEEP_ALL_MARKS and not self._thinned:
            return list(marks)
        span = marks[-1]["ts"] - marks[0]["ts"]
        fmt = "%Y%m%d%H" if span.days <= 90 else "%Y%m%d"
        keys = [m["ts"].strftime(fmt) for m in marks]
        kept = [marks[0]] + [m for i, m in enumerate(marks) if i and (i + 1 == len(marks) or keys[i + 1] != keys[i])]
        if self._worst:
            ids = {id(m) for m in kept}
            kept += [m for m in self._worst if id(m) not in ids]
            kept.sort(key=lambda m: m["ts"])  # stable: marks sharing a time keep their order
        return kept

    def max_drawdown(self, sleeve: str | None = None) -> float:
        """The deepest fall from a peak over every mark of the run, thinned or not."""
        return 1 - self._worst[1]["equity"] / self._worst[0]["equity"] if self._worst else 0.0
