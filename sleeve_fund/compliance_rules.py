"""The portfolio's compliance tests as the PM sees and edits them (Bridge Drop A; Advisor BRIDGE-3 and BRIDGE-A-FE,
14 Oct 2026). One source for the keys, the PM's words, the edit bounds and the tests that can't be switched off, read by
the Portfolio Settings form and by the store's write alike. Every limit is a share of the fund's equity (0.02 = 2%)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Rule:
    key: str
    test: str  # the PM's test number, T1 to T5; T4 and T5 group several rules
    label: str  # the PM's words
    scope: str  # fund, asset or asset_type (the DA's compliance_tests.scope)
    measure: str  # open_risk, gross, net, margin, day_loss or drawdown
    action: str  # what the gate does at the limit
    default: Decimal
    low: Decimal
    high: Decimal
    unit: str = "%"  # "%" or "x": how the page shows the limit
    locked: bool = False  # can be changed within its bounds but never switched off


RULES = (
    Rule("asset_stop_loss", "T1", "Per asset: loss if every stop is hit", "asset", "open_risk", "trim",
         Decimal("0.02"), Decimal("0.0025"), Decimal("0.05")),
    Rule("type_stop_loss", "T2", "Per asset group: loss if every stop is hit", "asset_type", "open_risk", "trim",
         Decimal("0.03"), Decimal("0.005"), Decimal("0.10")),
    Rule("fund_stop_loss", "T3", "Whole fund: loss if every stop is hit", "fund", "open_risk", "trim",
         Decimal("0.05"), Decimal("0.01"), Decimal("0.10"), locked=True),
    Rule("fund_gross", "T4", "Gross: every position's size added up", "fund", "gross", "trim",
         Decimal("1.5"), Decimal("0.1"), Decimal("3"), unit="x"),
    Rule("asset_net", "T4", "Net per asset: long less short on one asset", "asset", "net", "trim",
         Decimal("0.5"), Decimal("0.05"), Decimal("1"), unit="x"),
    Rule("fund_margin", "T4", "Margin posted", "fund", "margin", "trim",
         Decimal("0.5"), Decimal("0.10"), Decimal("0.80")),
    Rule("fund_day_loss", "T5", "Today's loss pauses new trades until 00:00 UTC", "fund", "day_loss", "pause",
         Decimal("0.03"), Decimal("0.01"), Decimal("0.05"), locked=True),
    Rule("fund_drawdown", "T5", "Drawdown from the peak closes everything and halts", "fund", "drawdown", "halt",
         Decimal("0.15"), Decimal("0.05"), Decimal("0.25"), locked=True),
)
BY_KEY = {r.key: r for r in RULES}
KEYS = tuple(BY_KEY)
LOCKED = frozenset(r.key for r in RULES if r.locked)
TESTS = {"T1": "Per asset", "T2": "Per asset group", "T3": "Whole fund", "T4": "Size limits", "T5": "Loss brakes"}

GROUPS = ("large", "mid", "small")
STRICTEST = "small"  # an asset with no group counts as this one (Advisor BRIDGE-3)


def show(key: str, limit: Decimal | float) -> str:
    """A limit in the PM's words: 2.0% or 1.5x."""
    v = Decimal(str(limit))
    if BY_KEY[key].unit == "x":
        return f"{v.normalize():f}x"
    return f"{(v * 100).quantize(Decimal('0.01')).normalize():f}%"


SHORT = {"fund_gross": "Gross", "asset_net": "Net per asset", "fund_margin": "Margin",
         "fund_day_loss": "Daily loss pause", "fund_drawdown": "Drawdown halt"}


def name(key: str) -> str:
    """The test as the PM reads it in a message: 'T1 Per asset', 'T4 Margin'."""
    r = BY_KEY[key]
    return f"{r.test} {SHORT.get(key, TESTS[r.test])}"


def validate(tests: dict[str, dict]) -> list[str]:
    """Why a set of tests can't be saved, in the PM's words; empty when it can. `tests`: key → {limit, enabled}. Every
    key must be there: a version is the whole set, never a change to the last one."""
    errors = []
    missing = [k for k in KEYS if k not in tests]
    if missing:
        errors.append("Every test must be in the new version; missing: " + ", ".join(name(k) for k in missing) + ".")
    unknown = sorted(set(tests) - set(KEYS))
    if unknown:
        errors.append("Not a compliance test: " + ", ".join(unknown) + ".")
    for k in KEYS:
        if k not in tests:
            continue
        r, t = BY_KEY[k], tests[k]
        try:
            limit = Decimal(str(t["limit"]))
        except (KeyError, ArithmeticError, ValueError, TypeError):
            errors.append(f"{name(k)}: the limit isn't a number.")
            continue
        if not r.low <= limit <= r.high:
            errors.append(f"{name(k)} must be between {show(k, r.low)} and {show(k, r.high)}.")
        if r.locked and not t.get("enabled", False):
            errors.append(f"{name(k)} can be changed but not switched off.")
    return errors


def warnings(tests: dict[str, dict]) -> list[str]:
    """Settings that can't lose money but would imply room that isn't there (Advisor BRIDGE-A-FE): a narrower test set
    above a wider one, so the wider one binds first. Shown in the confirm step; never a refusal."""
    out = []
    chain = [k for k in ("asset_stop_loss", "type_stop_loss", "fund_stop_loss") if tests.get(k, {}).get("enabled")]
    for a, b in zip(chain, chain[1:]):
        la, lb = Decimal(str(tests[a]["limit"])), Decimal(str(tests[b]["limit"]))
        if la > lb:
            out.append(f"{TESTS[BY_KEY[a].test]} ({show(a, la)}) is above {TESTS[BY_KEY[b].test].lower()} "
                       f"({show(b, lb)}); the {TESTS[BY_KEY[b].test].lower()} test will bind first.")
    return out


def changes(old: dict[str, dict] | None, new: dict[str, dict], old_groups: dict[str, str] | None = None,
            new_groups: dict[str, str] | None = None) -> list[str]:
    """What a new version changes, one line per field: 'T1 Per asset: 2% -> 1.5%', 'T4 Margin posted: on -> off',
    'SOL group: none -> mid'."""
    out = []
    for k in KEYS:
        if k not in new:
            continue
        o, n = (old or {}).get(k), new[k]
        if o is None:
            out.append(f"{name(k)}: {show(k, n['limit'])}{'' if n.get('enabled') else ', off'} (new)")
            continue
        if Decimal(str(o["limit"])) != Decimal(str(n["limit"])):
            out.append(f"{name(k)}: {show(k, o['limit'])} → {show(k, n['limit'])}")
        if bool(o.get("enabled")) != bool(n.get("enabled")):
            out.append(f"{name(k)}: {'on' if o.get('enabled') else 'off'} → {'on' if n.get('enabled') else 'off'}")
    og, ng = old_groups or {}, new_groups or {}
    for asset in sorted(set(og) | set(ng)):
        if og.get(asset) != ng.get(asset):
            out.append(f"{asset} group: {og.get(asset) or 'none'} → {ng.get(asset) or 'none'}")
    return out


def defaults() -> dict[str, dict]:
    """Version 1 as the DA seeds it: the built tests on, per asset and per group off until the PM switches them on."""
    return {r.key: {"limit": r.default, "enabled": r.key not in ("asset_stop_loss", "type_stop_loss")} for r in RULES}
