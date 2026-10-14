"""P2-8 two legs in the engine: a two-leg Strategy runs as two strategies in one engine, one per leg, each on its own
venue account with its own runtime (its own sleeves row), so every money path (fills, fees, marks, funding,
liquidation, the journal's book) works per leg unchanged. A Pair couples them for what spans both legs (spec
pe2-plans/p2-8-spec.md; Independent Quant Advisor, 13 Oct 23:55 UK):

- The lead leg's model decides; the follower (LegFollower) has no signal of its own and only trades with it.
- An entry opens both legs, the less liquid one first (by default the perp leg). The other is sent once the first has
  filled, sized to what filled at the hedge ratio. Every order of the trade carries one parent_trade_id.
- If the second leg doesn't fill it is sent once more; if that fails too, the first leg is unwound (intent
  leg_unwind) with a warning alert (two_leg_broken), and the lead takes no new entry until its signal has moved off
  its side, as after a stop. If the second leg fills in part, the first leg is trimmed to the ratio.
- While one leg is held without the other (a legged state) no new entry is taken (ruling A).
- When either leg exits, for any reason, the other is closed with it: a lone leg is never held."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from types import SimpleNamespace

from nautilus_trader.model import Bar, ClientOrderId, OrderSide

from sleeve_fund.data import bar_minutes
from sleeve_fund import router
from sleeve_fund.margin import entry_liquidation
from sleeve_fund.strategies.base import OPENING_INTENTS, LongFlatStrategy

DELAY_NS = 1  # the first leg goes after every venue's bar for the close it was decided on, as a portfolio gate pass
RETRIES = 1  # the second leg is sent once more before the first is unwound (Advisor C)
REHEDGE = 0.10  # the hedge leg is brought back to the ratio once it has drifted more than this (Advisor D)
BROKEN = "two_leg_broken"  # the alert raised when a pair can't be completed and its first leg is unwound
CLAMP = 0.5  # a perp leg's distance to liquidation below this share of its entry distance closes the pair (Advisor B)
CLAMPED = "two_leg_liq_distance"  # the alert raised when it does
LEG_TIMEOUT_NS = 60_000_000_000  # a legged state is unwound after one bar or this, whichever is first (Advisor C)
# Ruling B's top-up (margin added from the perp leg's free cash) needs a venue profile that allows it; none does yet,
# so the close is the path that runs (Advisor 14 Oct 05:35 UK).


class LegFollower(LongFlatStrategy):
    """A two-leg Strategy's other leg: no signal of its own. It trades only when its Pair sends an order through it,
    and keeps every per-leg duty of a strategy (its marks, funding, guards and journal)."""

    def want_long(self, bar: Bar) -> bool | None:
        return None

    def want_side(self, bar: Bar) -> int | None:
        return None

    def target_weight(self, bar: Bar) -> float | None:
        return None

    def __init__(self, config) -> None:
        super().__init__(config)
        self.two_leg = True  # never opens on its own: only through its Pair, and nothing new without one (HoE pin)

    def explain(self, bar: Bar, target) -> tuple[str, dict]:
        return "The second leg trades with its Strategy's first leg", {}


def _allows_top_up(venue_name: str) -> bool:
    """The venue profile's allows_margin_top_up (a portfolio or two-leg clone reads its real venue's); a venue with no
    profile (a test's) allows none."""
    from sleeve_fund.venues import venue

    try:
        return venue(venue_name).allows_margin_top_up
    except ValueError:
        return False


def _utc(ns: int):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _side(s: int):
    return OrderSide.BUY if s > 0 else OrderSide.SELL


class Pair:
    """The two leg strategies of one two-leg Strategy, coupled. lead_side and follow_side: each leg's configured side
    (+1 long, -1 short; the lead's model opens its side, the follower takes its own). ratio: the follower's notional
    per unit of the lead's. first_out: which leg is sent first, "follow" or "lead" (default: the perp leg, the less
    liquid one, when exactly one leg is a perp; else the lead)."""

    def __init__(self, lead: LongFlatStrategy, follow: LongFlatStrategy, *, lead_side: int, follow_side: int,
                 ratio: Decimal | float | str = 1, first_out: str | None = None, name: str | None = None) -> None:
        self.lead, self.follow = lead, follow
        self.sides = {id(lead): lead_side, id(follow): follow_side}
        self.ratio = Decimal(str(ratio))
        if first_out is None:
            first_out = "follow" if follow._cfg.perp is not None and lead._cfg.perp is None else "lead"
        if first_out not in ("lead", "follow"):
            raise ValueError(f"first_out is 'lead' or 'follow', not {first_out!r}")
        self.first_out = first_out
        self.name = name
        for leg in (lead, follow):
            if leg._cfg.perp is not None and _allows_top_up(str(leg._cfg.instrument_id.venue)):
                raise ValueError("ruling B's margin top-up isn't built: a venue that allows it can't run a two-leg "
                                 "Strategy until it is, so the close path is the one that runs")
        lead.pair = follow.pair = self
        # One price feed for both legs (a perp simulated on the spot's prices): the other leg is marked at this leg's
        # price, so the summed book the guard judges has one price level.
        self.same_prices = str(lead._cfg.instrument_id.symbol) == str(follow._cfg.instrument_id.symbol)
        for leg in (lead, follow):
            if leg.runtime is not None:
                leg.runtime.paired = True
        self.trade_id: str | None = None
        self.trades = 0
        self._state = "flat"  # flat, legging (one leg sent or held without the other), hedged, closing
        self.deadline_ns: int | None = None  # while legging: when the legged state is unwound if still legged
        # Called with the pair's state (state, trade_id, deadline_ns) whenever it changes: paper keeps it in the journal
        # so a restart recovers it (Advisor M3); a backtest has none.
        self.on_state = None
        # Paper with a journal: False until recover() has run, on the first tick both legs are back and priced (ready).
        self.recovered = True
        self.second: dict | None = None  # the leg still to send: leg, qty, reason, signal, tries
        self.sending: LongFlatStrategy | None = None  # the leg whose own _send_entry the pair is calling now
        self.closing_from: LongFlatStrategy | None = None
        self._measured: int | None = None  # the last decision close the hedge was measured at
        self._entry_dist: dict[int, float] = {}  # a perp leg's distance to liquidation as a share of its entry
        self._clamped: int | None = None  # the bar the pair was last closed for its liquidation distance
        self.log: list[tuple[int, str, str]] = []  # (ts, what, detail): the pair's own steps, for tests and reports

    # --- state -------------------------------------------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        if value == self._state:
            return
        self._state = value
        if value in ("legging", "closing"):
            # A legged or closing pair always has its trade and a time it must be flat by (pair_state's shape).
            self.trade_id = self.trade_id or f"{self._name()}-{self._now()}"
            if value == "closing" or self.deadline_ns is None:
                self.deadline_ns = self._now() + self._leg_timeout()
        else:
            self.deadline_ns = None
        if self.on_state is not None:
            self.on_state({"state": value, "trade_id": self.trade_id, "deadline_ns": self.deadline_ns})

    # --- the journal's copy (Advisor M3; DA 14 Oct 05:02 UK) ------------------------------------------------------

    def journal(self, store, parent: str) -> None:
        """Paper: keep the pair's state in pair_state (`store`, the journal) for `parent`, the first leg's row, by
        compare-and-set from the state last written; and read it back now, for recover()."""
        self._store, self._parent = store, parent
        row = store.pair_state(parent)
        if row is None:
            store.set_pair_state(parent, expected=None, state="flat", ts=_utc(self._now()))
            row = store.pair_state(parent)
        self._written = row["state"], row["trade_id"]
        self._saved = row
        self.on_state = self._write
        self.recovered = False

    def ready(self) -> None:
        """Paper, on each leg's tick until it has run: recover() once both legs hold what the journal says (no restore
        pending) and have a price to unwind at."""
        if self.recovered:
            return
        if any(leg._restore is not None or not leg._price() > 0 for leg in (self.lead, self.follow)):
            return
        self.recover()

    def _write(self, st: dict) -> None:
        deadline = _utc(st["deadline_ns"]) if st["deadline_ns"] is not None else None
        trade = st["trade_id"] if st["state"] != "flat" else None
        for _ in range(2):  # lost a race: re-read what is there and move from that
            try:
                if self._store.set_pair_state(self._parent, expected=self._written[0], state=st["state"],
                                              trade_id=trade, deadline=deadline, ts=_utc(self._now()),
                                              expected_trade_id=self._written[1]):
                    self._written = st["state"], trade
                    return
            except ValueError:  # what is there can't move to this state: re-read it, then alert below if still so
                pass
            row = self._store.pair_state(self._parent)
            self._written = (row["state"], row["trade_id"]) if row is not None else (None, None)
            if self._written[1] not in (None, trade):
                break  # another trade's row: never moved from here (DA, stale writer)
        self._alert("pair_state_unsaved", f"Two-leg trade {self.trade_id}: its state ({st['state']}) couldn't be "
                                          "kept in the journal; a restart reads the fills instead")

    def recover(self) -> None:
        """Paper, once both legs hold what the journal says they hold (their restore filled): a pair the journal left
        legged or closing is unwound now, whatever is held sold at market (the fills are the truth, the state only the
        hint, DA); a hedged one with a leg missing likewise. Both legs flat: the state goes back to flat, no order."""
        self.recovered = True
        saved, self._saved = getattr(self, "_saved", None), None  # once: what the journal said at start
        if saved is None or saved["state"] == "flat":
            return
        self.trade_id = saved["trade_id"]
        self._state = saved["state"]  # as the journal left it; the moves below are written from it
        held = [leg for leg in (self.lead, self.follow) if not self._flat(leg)]
        if saved["state"] == "hedged" and len(held) == 2:
            return
        if not held:
            self._note("recovered_flat", f"{self.trade_id}: both legs flat at restart")
            self._end()
            return
        late = saved["deadline"] is not None and saved["deadline"] < _utc(self._now())
        self._alert(BROKEN, f"Two-leg trade {self.trade_id} was {saved['state']} when the process stopped"
                            f"{' and its deadline has passed' if late else ''}; the leg held is unwound at market")
        self.state = "closing"
        for leg in held:
            self.closing_from = leg
            try:
                leg._sell_all("leg_unwind", "Unwound at restart: the trade was left with one leg")
            finally:
                self.closing_from = None
        self._maybe_flat()

    def _leg_timeout(self) -> int:
        return min(bar_minutes(self.lead._cfg.bar_type) * 60_000_000_000, LEG_TIMEOUT_NS)

    def _timed_out(self, trade_id: str) -> None:
        """The legged state's deadline: still one leg without the other, so whatever is held is unwound and anything
        still working is cancelled (ruling C: unwound within one bar or 60 s)."""
        if self.state == "legging" and self.trade_id == trade_id:
            self._broken(f"it was still legged {self._leg_timeout() / 1e9:.0f} s after its entry")

    # --- helpers -----------------------------------------------------------------------------------------------------

    def other(self, leg: LongFlatStrategy) -> LongFlatStrategy:
        return self.follow if leg is self.lead else self.lead

    def _name(self) -> str:
        if self.name:
            return self.name
        rt = getattr(self.lead, "runtime", None)
        return rt.name if rt is not None else "pair"

    def _now(self) -> int:
        return self.lead.clock.timestamp_ns()

    def _note(self, what: str, detail: str) -> None:
        self.log.append((self._now(), what, detail))

    def _alert(self, kind: str, message: str) -> None:
        self._note(kind, message)
        for leg in (self.lead, self.follow):
            if leg.runtime is not None:
                leg.runtime.store.event(leg.runtime.name, "warning", kind, message, ts=leg.runtime.now())

    def other_book(self, leg: LongFlatStrategy, price: float) -> tuple[float, float, float]:
        """The other leg's (equity, cash, qty), marked at `price` when both legs trade one price feed, else at its own
        last price: what the summed guard adds to this leg's (Advisor E)."""
        equity, cash, qty, _ = self.other(leg)._mark(price if self.same_prices else None)
        return equity, cash, qty

    def held(self, leg: LongFlatStrategy) -> Decimal:
        return leg._position_qty() if leg.cache is not None and leg.instrument is not None else Decimal(0)

    def _flat(self, leg: LongFlatStrategy) -> bool:
        q = self.held(leg)
        return q <= 0 or q < leg._min_qty()

    def _sized(self, to: LongFlatStrategy, qty: Decimal, frm: LongFlatStrategy, px_from: float) -> Decimal:
        """`to`'s quantity for `qty` of `frm` at the hedge ratio, matched in notional at each leg's last price."""
        ratio = self.ratio if to is self.follow else 1 / self.ratio
        px_to = to._price() or px_from
        if not px_to:
            return Decimal(0)
        return (qty * ratio * Decimal(repr(px_from)) / Decimal(repr(px_to))).quantize(to._lot(), rounding=ROUND_DOWN)

    def follower_entry(self, qty: Decimal, px: float) -> tuple[float, float, float | None]:
        """What a lead entry of `qty` at `px` adds on the second leg: its signed quantity, price and stop (None without
        one), for the open-risk check, which counts both legs in full with no hedge credit (Advisor M6)."""
        f = self.follow
        side, f_px = self.sides[id(f)], f._price() or px
        stop = f_px * (1 - side * f._stop_frac) if f._stop_frac else None
        return side * float(self._sized(f, qty, self.lead, px)), f_px, stop

    def blocks_entry(self) -> str | None:
        """Why the lead may not open now: a pair already open, being sent, or held on one leg only (ruling A), or one
        not yet recovered after a restart."""
        if not self.recovered:
            return "the pair is still being checked after the restart: no new entry until it is"
        if self.state == "legging":
            return "one leg is held without the other: no new entry until it is hedged or unwound"
        if self.state != "flat":
            return None if self.state == "hedged" else "the pair is closing"
        if not (self._flat(self.lead) and self._flat(self.follow)):
            return "one leg is held without the other: no new entry until it is hedged or unwound"
        return None

    # --- entry -------------------------------------------------------------------------------------------------------

    def entry(self, leg: LongFlatStrategy, side, qty: Decimal, reason: str, signal: dict, what: str = "entry",
              sent=None) -> None:
        """The lead decided to open (its _send_entry, sized on its own equity): open the pair. Only a lead entry on its
        configured side starts a pair; anything else is refused with a note."""
        s = 1 if side == OrderSide.BUY else -1
        if leg is not self.lead or what != "entry":
            leg._note("pair_entry_refused", f"Entry refused: a two-leg Strategy opens only through its first leg "
                                            f"({what} on the {'first' if leg is self.lead else 'second'} leg)")
            return
        if s != self.sides[id(self.lead)]:
            leg._note("pair_entry_refused", "Entry refused: the first leg's model asked for the side opposite the "
                                            "one this pair is defined with")
            return
        if self.state == "hedged":
            return  # both legs already held: an add is not part of a pair (a re-hedge is its own order)
        if (why := self.blocks_entry()) is not None:
            leg._note("pair_entry_held", f"Entry held: {why}")
            return
        if self.lead._portfolio is not None or self.follow._portfolio is not None:
            leg._note("pair_entry_refused", "Entry refused: a two-leg Strategy doesn't run in a portfolio run yet")
            return
        px = float(signal.get("close") or leg._price())
        f_qty = self._sized(self.follow, qty, self.lead, px)
        lead_part = {"leg": self.lead, "qty": qty, "reason": reason, "signal": signal, "tries": 0}
        follow_part = {"leg": self.follow, "qty": f_qty, "reason": f"Second leg of: {reason}",
                       "signal": {**signal, "close": self.follow._price() or px, "pair_ratio": str(self.ratio)},
                       "tries": 0}
        if not self._gate_both(lead_part, follow_part):
            return
        self.trades += 1
        self.trade_id = f"{self._name()}-{self._now()}"
        first, self.second = (follow_part, lead_part) if self.first_out == "follow" else (lead_part, follow_part)
        self.deadline_ns = self._now() + self._leg_timeout()
        self.state = "legging"
        trade_id = self.trade_id
        self.lead.clock.set_time_alert_ns(f"pair-deadline-{self.trades}", self.deadline_ns,
                                          callback=lambda _e: self._timed_out(trade_id))
        self._note("entry", f"{self.trade_id}: {first['leg'] is self.lead and 'lead' or 'follow'} first")
        # After every venue's bar for this close (as a portfolio gate pass), so each leg fills on its own latest bar.
        self.lead.clock.set_time_alert_ns(f"pair-entry-{self.trades}", self._now() + DELAY_NS,
                                          callback=lambda _e, part=first: self._send_first(part))

    def _gate_both(self, lead_part: dict, follow_part: dict) -> bool:
        """Paper with the portfolio gate (Advisor M1): both legs checked at once under the one lock, one share for both
        or neither, each reserved under its own row before either is sent; each part keeps its leg's answer (`gated`)
        and is cut to what it allows. Elsewhere (a backtest) there is nothing to check. False when refused."""
        gate = self.lead._paper_gated()
        if gate is None or self.follow._paper_gated() is None:
            return True
        intents = []
        for part in (lead_part, follow_part):
            leg = part["leg"]
            part["signal"], intent = leg._gate_intent(_side(self.sides[id(leg)]), part["qty"], part["signal"], "entry")
            if intent["atr_pct"] is None and not leg._stop_frac:
                self.lead._note("entry_refused_by_portfolio", "Entry refused: a leg's open risk can't be measured "
                                "(no stop, and the daily ATR isn't known), so neither leg is sent")
                return False
            intents.append(intent)
        answers = gate.pair((self.lead.runtime.name, self.follow.runtime.name), tuple(intents), self._now())
        if any(a.approved_qty <= 0 for a in answers):
            for a in answers:
                a.release("cancel")
            self.lead._note("entry_refused_by_portfolio", answers[0].decision.reason or "Entry refused by the "
                            "portfolio limits, for both legs")
            return False
        self.lead._noted.discard("entry_refused_by_portfolio")
        for part, answer in zip((lead_part, follow_part), answers):
            leg = part["leg"]
            part["qty"] = min(part["qty"], Decimal(str(answer.approved_qty))).quantize(leg._lot(),
                                                                                         rounding=ROUND_DOWN)
            part["gated"], part["cap"] = answer, part["qty"]
        return True

    def _release(self, part: dict | None) -> None:
        """A checked part never sent, or a second leg's held for its retry: its reservation goes back to the book."""
        if part is None:
            return
        if part.get("gated") is not None:
            part.pop("gated").release("cancel")
        if part.get("held") is not None:
            part["leg"].runtime.store.release_held(part.pop("held"))

    def _retry_refusal(self, leg: LongFlatStrategy) -> str | None:
        """Why the second leg's retry is not sent and the pair unwinds instead (Advisor, leg-2 retry MUST 3): a halt,
        pause or stale data on either leg, its venue pin not matching, or the coordinator gone: on paper, also when the
        journal no longer holds this trade legging (another writer moved it, so this coordinator lost it)."""
        for each in (self.lead, self.follow):
            if each.pair is not self:
                return "its coordinator is gone"
            rt = each.runtime
            if rt is None:
                continue
            if rt.backtest:
                if not rt.can_open():
                    return f"a leg is {rt.status}"
                continue
            # CHOKE on each leg, not can_open alone: stale data, a degraded candle or the portfolio's block on the
            # other leg unwinds the pair too (CR F283-4).
            blocked, why = rt.entry_blocked()
            if blocked or not rt.can_open():
                return why or f"a leg is {rt.status}"
        if getattr(self, "_store", None) is not None:
            row = self._store.pair_state(self._parent)
            if row is None or (row["state"], row["trade_id"]) != ("legging", self.trade_id):
                return "its coordinator lost the trade: the journal no longer holds it legging"
        if leg.runtime is not None and not leg.runtime.backtest:
            try:
                router.check(leg.runtime.name, str(leg._cfg.instrument_id), leg.runtime.venue)
            except ValueError as exc:
                return str(exc)
        return None

    def _carry_reservation(self, part: dict) -> bool:
        """Paper: the retry draws on what the pair's check reserved for this leg, never a check of its own (Advisor,
        MUST 1, 2): at most what is left of it. False when nothing is left."""
        if part.get("held") is None:
            return True  # no gate (a backtest): nothing reserved to carry
        got = part["leg"].runtime.store.hand_on(part.pop("held"))
        if got is None:
            return False
        rid, left = got
        part["qty"] = min(part["qty"], left).quantize(part["leg"]._lot(), rounding=ROUND_DOWN)
        store = part["leg"].runtime.store
        part["gated"] = SimpleNamespace(approved_qty=left, checked=SimpleNamespace(reservation=rid),
                                        release=lambda reason, rid=rid: store.release_reservation(rid, reason))
        return True

    def _send_first(self, part: dict) -> None:
        """The first leg's order: when none was placed (too small, or its strategy declined it), nothing is held or
        working, so the pair ends here and the next signal can open it, rather than staying legged (CR P2-8 1)."""
        if self.state != "legging" or self._send(part):
            return
        why = "its first leg's order couldn't be placed"
        self._note("first_unsent", why)
        part["leg"]._note("pair_first_unsent", f"Entry not opened: {why}", level="info")
        self._end()

    def _send(self, part: dict) -> bool:
        leg = part["leg"]
        qty = part["qty"]
        if qty <= 0 or qty < leg._min_qty():
            self._note("too_small", f"{qty} is below the smallest order the venue takes ({leg._min_qty()})")
            self._release(part)
            return False
        part["tries"] += 1
        before = getattr(leg, "_last_submitted", None)
        signal = {**part["signal"], **self._margin_signal(leg, qty)}
        gated = part.pop("gated", None)
        self.sending = leg
        try:
            leg._send_entry(_side(self.sides[id(leg)]), qty, part["reason"], signal, gated=gated)
        finally:
            self.sending = None
        coid = getattr(leg, "_last_submitted", None)
        if coid is None or coid == before:
            return False
        part["coid"] = coid
        store = getattr(leg.runtime, "store", None)
        if gated is not None and part is self.second and hasattr(store, "hold_for_retry"):
            # Its reject or cancel keeps what is left reserved for the retry (Advisor, leg-2 retry MUST 1).
            store.hold_for_retry(coid)
            part["held"] = coid
        self._note("sent", f"{'lead' if leg is self.lead else 'follow'} {qty} ({coid})")
        return True

    def _margin_signal(self, leg: LongFlatStrategy, qty: Decimal) -> dict:
        """A perp leg's entry: its liquidation price at its own leverage and its leverage on its equity, journaled with
        the order as a single perp entry's are (LongFlatStrategy._open)."""
        if not leg._margin:
            return {}
        equity, cash, _, _ = leg._mark()
        px = leg._price()
        if equity <= 0 or px <= 0:
            return {}
        side = self.sides[id(leg)]
        liq, _ = entry_liquidation(cash, float(qty), px, side, leg._cfg.assumed_taker_fee,
                                   leg._cfg.perp.maintenance_margin, leg._max_leverage())
        out = {"close": px, "leverage": round(float(qty) * px / equity, 4)}
        if liq is not None:
            out["liquidation_px"] = round(liq, 8)
        return out

    # --- what each leg's orders came to ------------------------------------------------------------------------------

    def order_closed(self, leg: LongFlatStrategy, coid: str) -> None:
        """A leg's order closed (filled, cancelled, rejected, denied or expired): carry the pair on."""
        order = leg.cache.order(ClientOrderId(coid))
        intent = (leg.decisions.get(coid) or {}).get("intent")
        filled = order.filled_qty.as_decimal() if order is not None else Decimal(0)
        if intent in OPENING_INTENTS and self.state == "legging":
            self._entry_closed(leg, coid, order, filled)
        elif intent not in OPENING_INTENTS and intent is not None:
            if (filled > 0 and self.state in ("legging", "hedged") and self.closing_from is None and self._flat(leg)
                    and not self._flat(self.other(leg))):
                # A resting order closed this leg at the venue (a stop, a target, a liquidation) without its own
                # sell-all: the other leg goes with it.
                self.leg_closing(leg, intent, (leg.decisions.get(coid) or {}).get("reason") or intent)
            self._maybe_flat()

    def _entry_closed(self, leg, coid, order, filled: Decimal) -> None:
        second = self.second
        if second is not None and second["leg"] is not leg:
            # The first leg's order has closed: send the second for what filled, or end the pair if nothing did.
            if filled <= 0:
                self._note("first_unfilled", f"{coid} closed with nothing filled: no pair opened")
                self._end()
                return
            px = float(order.avg_px) if order is not None and order.avg_px else leg._price()
            second["qty"] = self._sized(second["leg"], filled, leg, px)
            if "cap" in second:  # never more than the pair's check reserved for it
                second["qty"] = min(second["qty"], second["cap"])
            if not self._send(second):
                self._broken(f"its second leg couldn't be sent ({second['qty']} is below the venue's minimum)")
            return
        if second is None or second["leg"] is not leg:
            return
        sent = order.quantity.as_decimal() if order is not None else second["qty"]
        if filled >= sent:
            self.state, self.second = "hedged", None
            self._note("hedged", self.trade_id or "")
            return
        rest = sent - filled
        if second["tries"] <= RETRIES and rest >= leg._min_qty():
            second["qty"] = rest.quantize(leg._lot(), rounding=ROUND_DOWN)
            if (why := self._retry_refusal(leg)) is not None:
                self._note("retry_refused", why)
            elif not self._carry_reservation(second):
                self._note("retry_refused", "nothing is left of the pair's reservation for it")
            elif self._send(second):
                return
        self._release(second)
        if not self._flat(leg):  # what the earlier tries filled is held: trim to it, don't unwind (CR P2-8 6)
            self._trim_first(leg)
            return
        self._broken(f"its second leg was not filled after {second['tries']} tries")

    def _trim_first(self, second_leg) -> None:
        """The second leg filled in part: cut the first leg back to the hedge ratio, and hold the pair so."""
        first = self.other(second_leg)
        want = self._sized(first, self.held(second_leg), second_leg, second_leg._price())
        cut = (self.held(first) - want).quantize(first._lot(), rounding=ROUND_DOWN)
        self.state, self.second = "hedged", None
        if cut > 0 and cut >= first._min_qty():
            side = _side(-self.sides[id(first)])
            first._submit(side, cut, "leg_unwind", f"Trimmed to the hedge ratio: the second leg filled "
                                                   f"{self.held(second_leg)} of what was sent", {"pair": self.trade_id})
        self._note("trimmed", f"first leg cut by {cut}")

    def _broken(self, why: str) -> None:
        """The pair can't be completed: unwind whatever leg is held, with a warning alert (two_leg_broken)."""
        self._alert(BROKEN, f"Two-leg trade {self.trade_id} broken: {why}; the leg held is unwound at market, and no "
                            "new entry is taken until the first leg's signal has moved off its side")
        self._release(self.second)
        self.state, self.second = "closing", None
        # As after a stop: the model's signal must reset before the pair is tried again, not every bar while it holds.
        self.lead._exit_lock = self.sides[id(self.lead)] if self.lead._margin else True
        for leg in (self.lead, self.follow):
            if not self._flat(leg) or leg._working():  # a leg's entry still working is cancelled with it
                self.closing_from = leg
                try:
                    leg._sell_all("leg_unwind", f"Unwound: {why}")
                finally:
                    self.closing_from = None
        self._maybe_flat()

    # --- the liquidation-distance clamp (ruling B, amended 14 Oct 05:35 UK) -------------------------------------------

    def distance_breached(self, leg: LongFlatStrategy, price: float, liq: float | None) -> bool:
        """Whether a perp leg at `price` is nearer its liquidation price `liq` than CLAMP of the distance it was
        entered at. Measured on every mark and fill of the leg (M5); the entry distance is taken at its first one."""
        if self.state not in ("legging", "hedged") or liq is None or not price or leg._entry_px is None:
            return False
        entry = self._entry_dist.setdefault(id(leg), abs(leg._entry_px - liq) / leg._entry_px)
        return entry > 0 and abs(price - liq) / price < CLAMP * entry

    def check_distance(self, leg: LongFlatStrategy, price: float, liq: float | None) -> bool:
        """Ruling B: below CLAMP of the entry distance, close the pair (no venue allows the top-up yet), with ungated
        leg_unwind orders, at most once a bar. Unlike a broken pair it may re-enter on its model's next signal, gated
        and freshly sized. True when it closed."""
        if not self.distance_breached(leg, price, liq):
            return False
        bar = self._now() // (bar_minutes(self.lead._cfg.bar_type) * 60_000_000_000)  # the lead's decision bar
        if self._clamped == bar:
            return False
        self._clamped = bar
        entry = self._entry_dist[id(leg)]
        self._alert(CLAMPED, f"Two-leg trade {self.trade_id}: the perp leg came within {abs(price - liq) / price:.1%} "
                             f"of its liquidation price, under half the {entry:.1%} it was entered at; both legs "
                             "were closed at market")
        self._release(self.second)
        self.state, self.second = "closing", None
        for each in (self.lead, self.follow):
            if not self._flat(each):
                self.closing_from = each
                try:
                    each._sell_all("leg_unwind", "Closed: the perp leg came within half its entry distance of "
                                                 "its liquidation price")
                finally:
                    self.closing_from = None
        self._maybe_flat()
        return True

    def on_mark(self, leg: LongFlatStrategy, price: float) -> None:
        """Paper, a fill of either leg (Advisor M5): each leg's guards (the summed halt and pause, the liquidation
        guard and ruling B's clamp) judged at once, as on a trade, not left to the next 30-second tick."""
        for each in (leg, self.other(leg)):
            if each.runtime is None or each._pending_exit is not None:
                continue
            if each._guard_breached(price if each is leg else each._price()):
                each._on_tick()

    # --- the hedge ----------------------------------------------------------------------------------------------------

    def bar_closed(self, leg: LongFlatStrategy, bar: Bar) -> None:
        """The lead's decision bar closed: measure the hedge after every venue's bar for this close (ruling D)."""
        if leg is not self.lead or self.state != "hedged" or bar.ts_event == self._measured:
            return
        self._measured = bar.ts_event
        self.lead.clock.set_time_alert_ns(f"pair-hedge-{bar.ts_event}", self._now() + DELAY_NS,
                                          callback=lambda _e: self.rehedge())

    def hedge_drift(self) -> float | None:
        """|the hedge leg's notional / its target - 1|, the target the ratio times the lead's notional; None when
        either leg has no price or the lead is flat."""
        lead_px, follow_px = self.lead._price(), self.follow._price()
        lead_q = self.held(self.lead)
        if not lead_px or not follow_px or lead_q <= 0:
            return None
        target = float(lead_q) * lead_px * float(self.ratio)
        return abs(float(self.held(self.follow)) * follow_px / target - 1)

    def rehedge(self) -> None:
        """Ruling D: at a bar close, at most once a bar, bring the hedge leg back to the ratio when it has drifted more
        than REHEDGE from it. An increase is a gated order (an add, intent rebalance); a reduction never is."""
        # A resting exit or the backtest's risk stop doesn't block it; an order on its way to the venue does.
        if self.state != "hedged" or self.lead._busy() or self.follow._busy() or self.follow._pending_exit is not None:
            return
        drift = self.hedge_drift()
        if drift is None or drift <= REHEDGE:
            return
        want = self._sized(self.follow, self.held(self.lead), self.lead, self.lead._price())
        have = self.held(self.follow)
        delta = (want - have).quantize(self.follow._lot(), rounding=ROUND_DOWN) if want > have else \
            (have - want).quantize(self.follow._lot(), rounding=ROUND_DOWN)
        if delta <= 0 or delta < self.follow._min_qty():
            return
        side = self.sides[id(self.follow)] * (1 if want > have else -1)
        reason = (f"Re-hedged: the second leg had drifted {drift:.1%} from its ratio to the first; "
                  f"{'added' if want > have else 'cut'} {delta} to bring it back")
        self._note("rehedge", reason)
        signal = {"pair": self.trade_id, "drift": round(drift, 6), "close": self.follow._price()}
        if want > have:
            self.sending = self.follow
            try:
                self.follow._send_entry(_side(side), delta, reason, {**signal, **self._margin_signal(self.follow,
                                                                                                    delta)},
                                        what="rebalance")
            finally:
                self.sending = None
        else:
            self.follow._submit(_side(side), delta, "rebalance", reason, signal)

    # --- exits -------------------------------------------------------------------------------------------------------

    def leg_closing(self, leg: LongFlatStrategy, intent: str, reason: str) -> None:
        """A leg is selling all (its model's exit, a stop, a halt, a liquidation cut): close the other leg too, in the
        same step. A lone leg is never held."""
        if self.closing_from is not None:
            return  # this is the pair's own close of the other leg
        other = self.other(leg)
        self.state = "closing" if self.state != "flat" or not self._flat(other) else self.state
        self._release(self.second)
        self.second = None
        if self._flat(other) and not other._working():
            return
        self.closing_from = leg
        try:
            other._sell_all("leg_unwind" if intent in ("liquidation", "liquidation_cut") else intent,
                            f"The other leg closed: {reason}")
        finally:
            self.closing_from = None
        if intent in ("liquidation", "liquidation_cut"):
            self._alert(BROKEN, f"Two-leg trade {self.trade_id}: a leg was liquidated, so the other was closed at "
                                "market; no new entry until the first leg's signal has moved off its side")
            self.lead._exit_lock = self.sides[id(self.lead)] if self.lead._margin else True

    def _maybe_flat(self) -> None:
        if self.state in ("closing", "hedged") and self._flat(self.lead) and self._flat(self.follow) \
                and not self.lead._working() and not self.follow._working():
            self._end()

    def _end(self) -> None:
        self._release(self.second)
        self._note("flat", self.trade_id or "")
        self.second, self.trade_id = None, None
        self.state = "flat"
        self._entry_dist.clear()

