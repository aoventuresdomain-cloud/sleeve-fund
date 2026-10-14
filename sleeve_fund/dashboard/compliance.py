"""Portfolio Settings (Bridge Drop A): the five compliance tests the gate applies to every idea, their use now, the
asset-group map, an edit with a confirm step and the history of versions. The rules (keys, words, bounds, what can't be
switched off) are sleeve_fund.compliance_rules'; the rows are the DA's versioned compliance tables, read and written
through the Store only. Use now is the entry gate's own book (as withdrawal.limits_in_use): display only."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from sleeve_fund import compliance_rules as rules
from sleeve_fund.dashboard.riskops import NEAR_LIMIT
from sleeve_fund.store import Store

ACTION_WORDS = {"trim": "Refuse or trim", "refuse": "Refuse", "pause_day": "Pause new trades", "halt": "Flatten and halt"}


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
        "fund_day_loss": (max(Decimal(0), 1 - eq / day_start) if day_start else Decimal(0), ""),
        "fund_drawdown": (max(Decimal(0), 1 - eq / reference) if reference else Decimal(0), ""),
    }
    return {"source": "gate", **{k: {"used": share, "what": what} for k, (share, what) in used.items()}}


def _tone(used: Decimal, limit: Decimal) -> str:
    return "loss" if used >= limit else "warn" if used >= limit * Decimal(repr(NEAR_LIMIT)) else ""


def page(store: Store) -> dict:
    """Everything the page shows: the current version's tests grouped T1-T5 with their use now, the group map and
    the history, each version with what it changed from the one before."""
    current = store.compliance()
    tests, groups = as_set(current)
    use = use_now(store, groups)
    rows = []
    for r in rules.RULES:
        t = tests.get(r.key, {"limit": r.default, "enabled": False})
        u = use.get(r.key)
        rows.append({"key": r.key, "test": r.test, "label": r.label, "limit": t["limit"], "enabled": t["enabled"],
                     "shown": rules.show(r.key, t["limit"]), "locked": r.locked, "low": rules.show(r.key, r.low),
                     "high": rules.show(r.key, r.high), "unit": r.unit, "action": ACTION_WORDS.get(r.action, r.action),
                     "used": rules.show(r.key, u["used"]) if u else None, "what": u["what"] if u else "",
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
            "current": {"tests": tests, "groups": groups}}


def history(store: Store, limit: int = 30) -> list[dict]:
    """The versions, newest first, each with its changes from the one before in the PM's words."""
    out = []
    for v in store.compliance_versions(limit):
        new_t, new_g = as_set(store.compliance(v["version"]))
        if v["version"] > 1:
            old_t, old_g = as_set(store.compliance(v["version"] - 1))
            lines = rules.changes(old_t, new_t, old_g, new_g)
        else:
            lines = ["The tests as they started"]
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
                        "enabled": r.locked or form.get(f"{r.key}.enabled") in ("on", "1", "true")}
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
