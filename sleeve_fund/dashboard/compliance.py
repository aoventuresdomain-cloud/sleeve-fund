"""Portfolio Settings (Bridge Drop A): the five compliance tests the gate applies to every idea, their use now, the
asset-group map, an edit with a confirm step and the history of versions. The rules (keys, words, bounds) are
sleeve_fund.compliance_rules'; the rows are the DA's versioned compliance tables, read and written through the Store
only. Use now is the entry gate's own book (as withdrawal.limits_in_use): display only."""

from __future__ import annotations

from datetime import UTC
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from sleeve_fund import compliance_rules as rules
from sleeve_fund.dashboard.riskops import NEAR_LIMIT
from sleeve_fund.portfolio.limits import day_return
from sleeve_fund.store import Store

UK = ZoneInfo("Europe/London")
ACTION_WORDS = {"trim": "Refuse or trim", "refuse": "Refuse", "pause_day": "Pause new trades", "halt": "Flatten and halt"}
# B-31: what a row switched off stops checking, in the review step, before the PM's confirm tick. A pause or halt in
# force when its brake goes off stays in force (Advisor B31-BRAKE-SWITCH, gate state kept by QD's 75f96a6).
OFF_WORDS = {
    "asset_stop_loss": "new trades are no longer trimmed to a loss per asset if every stop is hit.",
    "type_stop_loss": "new trades are no longer trimmed to a loss per asset group if every stop is hit, and an asset "
                      "with no group can take a trade.",
    "fund_gross": "new trades are no longer trimmed to a gross size for the fund.",
    "asset_net": "new trades are no longer trimmed to a net size on one asset.",
    "fund_margin": "new trades are no longer trimmed to the margin the fund may post.",
    "fund_stop_loss": "new trades are no longer trimmed to a loss for the whole fund if every stop is hit.",
    "fund_day_loss": "today's loss no longer pauses new trades. A pause already in force stays until the next day "
                     "begins.",
    "fund_drawdown": "a fall from the peak no longer closes everything and halts. A halt already in force stays until "
                     "you restart the fund.",
}
# B-31: T3 and the T5 brakes switch off for paper testing only (Store.add_compliance_version refuses it otherwise).
BRAKES = ("fund_stop_loss", "fund_day_loss", "fund_drawdown")
PAPER_ONLY = "Switching the whole-fund test or a loss brake off is for paper testing only."
# The rules every order keeps whatever the compliance tests (PM capital model, 14 Oct).
ORDER_RULES = ("The order rules still apply to every trade: a stop, or 1x leverage without one; a stop within half the "
               "distance to liquidation; and the leverage cap.")
# Portfolio Settings' fixed lines: the order rules are the only rules with no off switch (Advisor, 20 Oct 17:00 UK).
NO_OFF_SWITCH = ("The order rules apply to every trade and have no off switch: a stop, or 1x leverage without one; a "
                 "stop within half the distance to liquidation; and the leverage cap.")


def brake_switches(old: dict[str, dict], new: dict[str, dict]) -> list[tuple[str, bool]]:
    """(key, switched on) for each of T3 and the T5 brakes the new set switches off or back on (B-31)."""
    return [(k, bool(new[k].get("enabled"))) for k in BRAKES
            if k in new and bool(old.get(k, {}).get("enabled", True)) != bool(new[k].get("enabled"))]


def rebase_words(store: Store, old: dict[str, dict], new: dict[str, dict]) -> list[str]:
    """B-31 (Advisor MUST 2): before a T5 brake goes back on, what it will judge from. The gate re-bases it to the book
    at that moment (gate.rebase_brakes), so the PM reads the level and, for the halt, the true fall from the peak, which
    is kept. Display only: the figures are the gate's last mark."""
    back_on = [k for k, on in brake_switches(old, new) if on and k in ("fund_drawdown", "fund_day_loss")]
    if not back_on:
        return []
    from sleeve_fund.paper.portfolio_gate import PaperGate

    try:
        st = PaperGate(store).ledger.state()
    except Exception:  # noqa: BLE001 - the words say so; the gate re-bases on its own either way
        st = None
    out = []
    for k in back_on:
        if st is None or st.equity is None or st.equity <= 0:
            out.append(f"{rules.name(k)} back on: the fund's book has no mark yet, so it measures from the first mark.")
        elif k == "fund_drawdown":
            below = 1 - st.equity / st.hwm if st.hwm else Decimal(0)
            out.append(f"{rules.name(k)} back on: it will measure from {st.equity:,.2f}, the book now. The book is "
                       f"{below:.1%} below its peak of {st.hwm:,.2f}, which is kept.")
        else:
            down = 1 - st.equity / st.day_start_equity if st.day_start_equity else Decimal(0)
            out.append(f"{rules.name(k)} back on: today's loss will count from {st.equity:,.2f}. The day is "
                       f"{down:.1%} down now.")
    return out


def brakes_off(store: Store) -> list[dict]:
    """B-31 (Advisor MUST 4): T3 and each T5 brake switched off in the version in force, with when it went off (the
    first version of the run that has it off) and that version's reason, for the banner that stays on the fund pages
    until it is back on."""
    try:
        return _brakes_off(store)
    except Exception:  # noqa: BLE001 - every fund page reads this: say it can't be read, never a broken page (CR)
        return [{"key": "unreadable", "unreadable": True}]


def _brakes_off(store: Store) -> list[dict]:
    current = as_set(store.compliance())[0]
    off = [k for k in BRAKES if k in current and not current[k]["enabled"]]
    if not off:
        return []
    out, sets = [], {}
    versions = store.compliance_versions()  # newest first
    for k in off:
        first = None
        for v in versions:
            if v["version"] not in sets:
                sets[v["version"]] = as_set(store.compliance(v["version"]))[0]
            if sets[v["version"]].get(k, {}).get("enabled", True):
                break
            first = v
        at = first["created_at"] if first else None
        if at is not None and at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        out.append({"key": k, "name": rules.name(k), "since": f"{at.astimezone(UK):%d %b %H:%M} UK" if at else None,
                    "reason": (first or {}).get("note") or ""})
    return out


def switched_off(old: dict[str, dict], new: dict[str, dict]) -> list[str]:
    """One line per test the new set switches off, 'T4 Gross off: new trades are no longer ...' (B-31)."""
    return [f"{rules.name(k)} off: {OFF_WORDS.get(k, 'new trades are no longer checked against it.')}"
            for k in rules.KEYS if k in new and old.get(k, {}).get("enabled") and not new[k].get("enabled")]


def as_set(compliance) -> tuple[dict[str, dict], dict[str, str]]:
    """A stored version as the rules read it: ({key: {limit, enabled}}, {underlying: group})."""
    tests = {t["test"]: {"limit": Decimal(str(t["limit"])), "enabled": bool(t["enabled"])} for t in compliance.tests}
    return tests, dict(compliance.asset_types)


def assets(store: Store) -> list[str]:
    """Every underlying the fund can trade (the instruments table), for the group map."""
    return sorted({r["underlying"] for r in store.instruments()})


def _gate_book(store: Store):
    """(the entry gate's Book, its state) at its last mark, or a reason it can't be read: 'unmarked' or 'unreadable'."""
    from sleeve_fund.paper.portfolio_gate import PaperGate
    from sleeve_fund.portfolio.limits import Book

    try:
        ledger = PaperGate(store).ledger
        state = ledger.state()
        if state.equity is None or state.equity <= 0:
            return "unmarked"
        _, held = ledger.positions()
        return Book(state.equity, (*held, *(r.holding for r in ledger.reservations()))), state
    except Exception as exc:  # noqa: BLE001 - a book the gate can't read leaves the use out, said so
        print(f"portfolio settings: can't read the entry gate's book: {exc!r}")
        return "unreadable"


def fund_halt(store: Store) -> str | None:
    """Why the fund is halted (the gate's own state, until the PM resumes), or None; None too when it can't be read."""
    from sleeve_fund.paper.portfolio_gate import PaperGate

    try:
        return PaperGate(store).ledger.state().halted
    except Exception:  # noqa: BLE001 - unread is not halted; the page's banners say when the book can't be read
        return None


def engine_use(store: Store, tests: dict[str, dict]) -> dict | None:
    """C2: the engine's own use of each test (PE2's supervisor.compliance_use, Drop B) in this page's shape, or None
    until the engine has it. The engine gives money: used of the limit it allows. The page shows the limit as its share
    of the book, so use is shown as used/limit x that share, and the bar fills used/limit exactly as the gate counts
    it. A test that is off has no limit, so its use is shown as a share of the book. Per asset (T1, net) and per group
    (T2), the largest is the headline and the rest are on hover; net is signed, so its size is shown."""
    try:
        from sleeve_fund.supervisor import compliance_use
    except ImportError:
        return None
    try:
        got = compliance_use(store)
    except Exception as exc:  # noqa: BLE001 - unread is shown as unreadable, never as room
        print(f"portfolio settings: can't read the engine's use of the compliance tests: {exc!r}")
        return {"source": "unreadable"}
    eq = got.get("equity")
    if eq is None or eq <= 0:
        return {"source": "unmarked"}
    out: dict = {"source": "gate"}
    for key, t in (got.get("tests") or {}).items():
        if key not in rules.BY_KEY:
            continue
        limit, pct = t.get("limit"), Decimal(str(tests.get(key, {}).get("limit", 0)))

        def shown(money: Decimal, limit=limit, pct=pct) -> Decimal:
            return money / limit * pct if limit else money / eq

        used = t.get("used")
        if isinstance(used, dict):
            parts = {(k.capitalize() if key == "type_stop_loss" else k): abs(Decimal(v)) for k, v in used.items()}
            order = sorted(parts.items(), key=lambda kv: (-kv[1], kv[0]))
            what, amount = order[0] if order else ("", Decimal(0))
            rest = " · ".join(f"{k} {float(v / limit * 100):.0f}% of its limit" if limit
                              else f"{k} {rules.show(key, shown(v))}" for k, v in order)
        else:
            what, amount, rest = "", abs(Decimal(used or 0)), ""
        out[key] = {"used": shown(amount), "what": what, "rest": rest}
    return out


def use_now(store: Store, groups: dict[str, str]) -> dict:
    """{key: {used, what, tone}} for each test against the entry gate's last mark, or {} with `source` unmarked /
    unreadable. Open risk by asset and by group is the gate's own per-holding risk summed; an unmapped asset is in no
    group (the gate takes no new entry on it while the per-group test is on)."""
    got = _gate_book(store)
    if isinstance(got, str):
        return {"source": got}
    book, state = got
    eq = book.equity
    by_asset: dict[str, Decimal] = {}
    by_group: dict[str, Decimal] = {}
    for h in book.holdings:
        by_asset[h.underlying] = by_asset.get(h.underlying, Decimal(0)) + h.risk
        if (g := groups.get(h.underlying)) is not None:
            by_group[g] = by_group.get(g, Decimal(0)) + h.risk
    nets = {u: abs(book.net(u)) for u in by_asset}

    def top(d: dict[str, Decimal]) -> tuple[Decimal, str]:
        return max(((v, k) for k, v in d.items()), default=(Decimal(0), ""))

    risk_a, asset = top(by_asset)
    risk_g, group = top(by_group)
    net, net_asset = top(nets)
    reference = state.halt_reference or state.hwm or eq
    day_start = state.day_start_equity or eq
    used = {
        "asset_stop_loss": (risk_a / eq, asset),
        "type_stop_loss": (risk_g / eq, group.capitalize() if group else ""),
        "fund_stop_loss": (book.open_risk() / eq, ""),
        "fund_gross": (book.gross() / eq, ""),
        "asset_net": (net / eq, net_asset),
        "fund_margin": (book.margin() / eq, ""),
        # DAY-RET: the daily pause's own figure (limits.day_return), never worked out again here
        "fund_day_loss": (max(Decimal(0), -(day_return(eq, day_start) or Decimal(0))), ""),
        "fund_drawdown": (max(Decimal(0), 1 - eq / reference) if reference else Decimal(0), ""),
    }
    return {"source": "gate", **{k: {"used": share, "what": what} for k, (share, what) in used.items()}}


def day_figure(store: Store, now=None) -> dict:
    """Today's return for the pages (DAY-RET): gate.day_figure on the fund's state, with today's T5 re-base if any
    (portfolio_brake_rebased). FE shows this; it never works the figure out again."""
    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.portfolio.gate import day_figure as figure
    from sleeve_fund.store import utcnow

    now = now or utcnow()
    # by kind, so a busy day's other events never push it out; gate.day_figure drops one from before today's open
    e = next((e for e in store.events_of(("portfolio_brake_rebased",), limit=20)
              if str(e["message"]).startswith("T5 daily loss pause")), None)
    rebased = e["ts"] if e else None
    if rebased is not None and rebased.tzinfo is None:
        rebased = rebased.replace(tzinfo=UTC)
    return figure(DbLedger(store, lambda: ()).state(), now, rebased)


DAY_OPEN_UNRECONCILED = "day_open_unreconciled"
UNRECONCILED_OVER = Decimal(1)  # USDT: the day's P&L and the Strategies' sum may differ by rounding, never by more


def day_open(store: Store, now=None, day_pnl=None) -> dict | None:
    """DAY-RET (A), Advisor option 2: "Opened today at X; deposits/withdrawals today Y; P&L today Z", read on demand
    from the book's minute marks (book_marks) and the fund's flows; no column of its own. The day is the brake's,
    from 00:00 UTC, shown in UK time. A display and check only: the T5 brake keeps day_figure's scaled start.
    day_pnl: Home's day P&L (the Strategies' sum). Z more than 1 USDT from it is unreconciled (a cost or flow booked
    outside the Strategies): shown, and journaled as a warning once a UTC day. None before the first mark.
    shortcut: read from the minute marks, kept whole today; if ever thinned within a day, a column (option 1)."""
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    from sleeve_fund.portfolio.gate import next_utc_midnight
    from sleeve_fund.store import utcnow

    now = now or utcnow()
    opened = next_utc_midnight(now) - timedelta(days=1)
    first, last = store.book_mark_at_or_before(opened), store.book_mark_at_or_before(now)
    if first is None or last is None:
        return None
    since, upto = _utc(first["ts"]), _utc(last["ts"])
    # [since, upto): the poll writes its mark first and lands its flows after it at the same ts, so a flow at a mark's
    # own ts is in the next mark's equity, not that one's. A pending flow (no applied_at) is in neither.
    flows = sum((Decimal(str(f["amount"])) * (-1 if f["kind"] == "withdraw" else 1) for f in store.fund_flows()
                 if f["applied_at"] is not None and since <= _utc(f["applied_at"]) < upto), Decimal(0))
    pnl = Decimal(str(last["equity"])) - Decimal(str(first["equity"])) - flows
    uk = ZoneInfo("Europe/London")
    gap = opened - since > timedelta(minutes=1)
    off = None if day_pnl is None else pnl - Decimal(str(day_pnl))
    off = off if off is not None and abs(off) > UNRECONCILED_OVER else None
    if off is not None and not any(_utc(e["ts"]) >= opened for e in store.events_of((DAY_OPEN_UNRECONCILED,), limit=1)):
        store.event(None, "warning", DAY_OPEN_UNRECONCILED,
                    f"Today's P&L from the book ({pnl:,.2f}) and the Strategies' ({Decimal(str(day_pnl)):,.2f}) differ "
                    f"by {off:,.2f}: a cost or flow is booked outside the Strategies", ts=now)
    return {"open": Decimal(str(first["equity"])), "opened": f"{opened.astimezone(uk):%H:%M} UK",
            "last_mark": f"{since.astimezone(uk):%H:%M} UK" if gap else None, "flows": flows, "pnl": pnl,
            "unreconciled": off}


def _utc(ts):
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


def _tone(used: Decimal, limit: Decimal) -> str:
    return "loss" if used >= limit else "warn" if used >= limit * Decimal(repr(NEAR_LIMIT)) else ""


def page(store: Store) -> dict:
    """Everything the page shows: the current version's tests grouped T1-T5 with their use now, the group map and
    the history, each version with what it changed from the one before."""
    current = store.compliance()
    tests, groups = as_set(current)
    use = engine_use(store, tests) or use_now(store, groups)
    rows = []
    for r in rules.RULES:
        t = tests.get(r.key, {"limit": r.default, "enabled": False})
        u = use.get(r.key)
        rows.append({"key": r.key, "test": r.test, "label": r.label, "limit": t["limit"], "enabled": t["enabled"],
                     "shown": rules.show(r.key, t["limit"]),
                     "edit": rules.edit(r.key, t["limit"]), "low": rules.show(r.key, r.low),
                     "high": rules.show(r.key, r.high), "unit": r.unit, "action": ACTION_WORDS.get(r.action, r.action),
                     "used": rules.show(r.key, u["used"]) if u else None, "what": u["what"] if u else "",
                     "rest": u.get("rest", "") if u else "",
                     "tone": _tone(u["used"], t["limit"]) if u and t["enabled"] else "",
                     # The bar's fill, % of the limit used (not capped: past 100 is past the limit), None with no use.
                     "fill": float(u["used"] / t["limit"] * 100) if u and t["limit"] else None})
    grouped = [{"test": k, "name": v, "rows": [x for x in rows if x["test"] == k]} for k, v in rules.TESTS.items()]
    names = sorted(set(assets(store)) | set(groups))
    return {"version": current.version, "tests": grouped, "source": use.get("source"),
            # Unmapped first, flagged (Advisor BRIDGE-A-FE-2).
            "groups": [{"asset": a, "group": groups.get(a)} for a in sorted(names, key=lambda a: (a in groups, a))],
            "group_names": rules.GROUPS, "unmapped_words": rules.UNMAPPED,
            "t2_on": tests.get("type_stop_loss", {}).get("enabled", False), "history": history(store),
            "refused": refused(store, limit=50), "current": {"tests": tests, "groups": groups},
            "no_off_switch": NO_OFF_SWITCH}


def history(store: Store, limit: int = 30) -> list[dict]:
    """The versions, newest first, each with its changes from the one before in the PM's words."""
    out = []
    for v in store.compliance_versions(limit):
        new_t, new_g = as_set(store.compliance(v["version"]))
        if v["version"] > 1:
            old_t, old_g = as_set(store.compliance(v["version"] - 1))
            lines = rules.changes(old_t, new_t, old_g, new_g)
        else:
            lines = ["The compliance tests as they started"]
        out.append({**v, "changes": lines})
    return out


def from_form(form) -> tuple[dict[str, dict], dict[str, str], list[str]]:
    """The tests and the group map a submitted form asks for, and why it can't be read. Limits are typed in the
    page's own unit (% or x); stored as a share of the fund's equity."""
    tests, errors = {}, []
    for r in rules.RULES:
        raw = (form.get(f"{r.key}.limit") or "").strip().rstrip("%x").strip()
        try:
            v = Decimal(raw)
        except InvalidOperation:
            errors.append(f"{rules.name(r.key)}: '{raw}' isn't a number.")
            continue
        tests[r.key] = {"limit": v / 100 if r.unit == "%" else v,
                        # As asked: every test has its checkbox since B-31 (F-LOCKED-OFF: never quietly put back on).
                        "enabled": form.get(f"{r.key}.enabled") in ("on", "1", "true")}
    # Advisor cell f: a switch for anything but a compliance test (the order rules have none) is refused, never ignored.
    unknown = sorted({k.rsplit(".", 1)[0] for k in form if k.endswith((".enabled", ".limit"))} - set(rules.KEYS))
    errors += [f"{k} isn't a compliance test, and the order rules have no off switch." for k in unknown]
    groups = {}
    for key in form:
        if key.startswith("group.") and form.get(key) in rules.GROUPS:
            groups[key.removeprefix("group.")] = form.get(key)
    return tests, groups, errors


def to_form(tests: dict[str, dict], groups: dict[str, str]) -> list[tuple[str, str]]:
    """The hidden fields that carry a reviewed set to the apply step unchanged."""
    out = []
    for k, t in tests.items():
        r = rules.BY_KEY[k]
        v = t["limit"] * 100 if r.unit == "%" else t["limit"]
        out.append((f"{k}.limit", f"{Decimal(v).normalize():f}"))
        if t["enabled"]:
            out.append((f"{k}.enabled", "on"))
    out += [(f"group.{a}", g) for a, g in sorted(groups.items())]
    return out


def rows(tests: dict[str, dict]) -> list[dict]:
    """A set of tests as the DA's compliance_tests rows (Store.add_compliance_version)."""
    return [{"test": r.key, "label": r.label, "scope": r.scope, "measure": r.measure, "limit": tests[r.key]["limit"],
             "action": r.action, "enabled": tests[r.key]["enabled"]} for r in rules.RULES if r.key in tests]


def strategy_bars(store: Store, underlying: str, risk_to_stop: float | None, risk_pct: float | None) -> list[dict]:
    """The Strategy page's three bars under its positions (C8): its risk to stop against its risk per trade, in money;
    the fund's net on this asset against T4; the fund's loss if every stop is hit on this asset against T1. The fund's
    figures and limits are the gate's book and the version in force; nothing is worked out in the browser. A bar with
    no figure yet says why."""
    got = _gate_book(store)
    book = None if isinstance(got, str) else got[0]
    why = {"unmarked": "The fund's book has no mark yet", "unreadable": "The fund's book can't be read"}.get(
        got if isinstance(got, str) else "", "")
    tests = {t["test"]: t for t in store.compliance().tests}

    def bar(key, label, used, limit, title):
        if used is None or limit is None or limit <= 0:
            return {"key": key, "label": label, "fill": None, "why": why or "Not known yet", "title": title}
        return {"key": key, "label": label, "fill": float(used / limit), "used": f"{float(used):,.2f}",
                "limit": f"{float(limit):,.2f}", "title": title, "why": ""}

    eq = book.equity if book is not None else None
    per_trade = Decimal(str(risk_pct)) * eq if eq is not None and risk_pct is not None else None
    out = [bar("risk_per_trade", "Risk to stop", Decimal(str(risk_to_stop)) if risk_to_stop is not None else None,
               per_trade, "This Strategy's loss if its stops are hit, against its risk per trade")]
    net = abs(book.net(underlying)) if book is not None else None
    t4 = tests.get("asset_net")
    out.append(bar("asset_net", f"Net on {underlying}", net,
                   t4["limit"] * eq if t4 and t4["enabled"] and eq is not None else None,
                   f"The fund's long less short on {underlying}, every Strategy, against T4 net per asset"))
    risk = sum((h.risk for h in book.holdings if h.underlying == underlying), Decimal(0)) if book is not None else None
    t1 = tests.get("asset_stop_loss")
    out.append(bar("asset_stop_loss", f"{underlying} loss if stops hit", risk,
                   t1["limit"] * eq if t1 and t1["enabled"] and eq is not None else None,
                   f"The fund's loss on {underlying} if every stop is hit, every Strategy, against T1 per asset"))
    return out


def refused(store: Store, sleeve: str | None = None, since=None, limit: int = 200) -> list[dict]:
    """C6: the ideas the tests refused or trimmed, newest first (every Strategy's, or one's): each with the test by
    name, with its numbers when the gate journaled what the idea would have reached (would_be, a share of the book,
    against the limit of the version that decided it), and that version. Display only: no figure is computed here."""
    limits: dict[int, dict] = {}
    out = []
    for r in store.refused_ideas(sleeve, since, limit):
        hit = r.get("test_hit") or r.get("limit_hit")
        key = hit if hit in rules.BY_KEY else rules.LIMIT_HIT.get(hit)
        words = rules.refused_by(hit)
        v, would = r.get("compliance_version"), r.get("would_be")
        if key and v is not None and would is not None:
            if v not in limits:
                limits[v] = as_set(store.compliance(v))[0]
            lim = limits[v].get(key, {}).get("limit")
            if lim is not None:
                words = f"{words}: {rules.show(key, would)} would exceed {rules.show(key, lim)}"
        side = r.get("side")
        out.append({**r, "test": words,
                    "side": {1: "Long", -1: "Short"}.get(int(side)) if side is not None else None})
    return out


def fund_state(store: Store) -> dict | None:
    """The engine's fund state (PE2, Drop B): {state, since, until, reason}, or None until the engine has it. Display
    only: nothing here works out a state on its own."""
    try:
        from sleeve_fund.supervisor import fund_state as engine  # PE2's, once it lands
    except ImportError:
        return None
    try:
        return engine(store)
    except Exception:  # noqa: BLE001 - an unreadable state shows no banner here; the gate still refuses on its own
        return None


def fund_banner(store: Store) -> dict | None:
    """C7: the one banner at the top of Portfolio Settings, a Strategy page and Portfolio, only when it applies:
    the daily brake, the drawdown halt, a failed book check or a fund value out of date."""
    s = fund_state(store)
    if not s or s.get("state") in (None, "normal"):
        return None
    reason = s.get("reason")
    state = s["state"]
    if state == "brake":
        until = s.get("until")
        if until is not None and until.tzinfo is None:  # the engine's times are UTC, stored naive or aware
            until = until.replace(tzinfo=UTC)
        text = "New trades paused until the next day begins" + (f" ({until.astimezone(UK):%H:%M} UK)" if until else "")
        return {"state": state, "tone": "warn", "text": text, "reason": reason}
    if state == "halt":
        return {"state": state, "tone": "loss", "text": "The fund is flattened and halted", "reason": reason,
                "restart": s.get("restart_url")}
    if state == "book_check":
        return {"state": state, "tone": "loss", "text": "The book check failed, so new trades are blocked",
                "reason": reason, "resume_off": reason or "The fund's book doesn't match the account"}
    if state == "stale":
        return {"state": state, "tone": "warn",
                "text": "The fund's value is out of date, so new trades are refused until it updates", "reason": reason}
    return None
