"""The PM's portfolio compliance tests (bridge A, Advisor BRIDGE-3 and BRIDGE-A-FE): the fixed keys, what each one
measures, the bounds an edit must stay within, and which tests can't be switched off. One source for the Portfolio
Settings form and the store's write (Store.add_compliance_version), so the two never disagree.

Stand-in written by the Data Architect so the store builds; Front-end owns this file and replaces it."""

from __future__ import annotations

from decimal import Decimal

# key: (scope, measure, action). Fixed: an edit changes a limit or switches a test on or off, never what it is.
SHAPE = {
    "asset_stop_loss": ("asset", "open_risk", "trim"),  # T1
    "type_stop_loss": ("asset_type", "open_risk", "trim"),  # T2, one limit for every group
    "fund_stop_loss": ("fund", "open_risk", "trim"),  # T3
    "fund_gross": ("fund", "gross", "trim"),  # T4a
    "asset_net": ("asset", "net", "trim"),  # T4b
    "fund_margin": ("fund", "margin", "trim"),  # T4c
    "fund_day_loss": ("fund", "day_loss", "pause_day"),  # T5a
    "fund_drawdown": ("fund", "drawdown", "halt"),  # T5b: flatten and halt; only the PM restarts
}
KEYS = tuple(SHAPE)
LABELS = {
    "asset_stop_loss": "Loss per asset if every stop is hit",
    "type_stop_loss": "Loss per asset group if every stop is hit",
    "fund_stop_loss": "Loss for the whole fund if every stop is hit",
    "fund_gross": "Total position size (gross)",
    "asset_net": "Net position per asset",
    "fund_margin": "Margin in use",
    "fund_day_loss": "Daily loss that pauses new entries until 00:00 UTC",
    "fund_drawdown": "Drawdown that closes everything and halts",
}
# Inclusive, as shares of fund equity (gross and net: times the book). Advisor BRIDGE-A-DATA and BRIDGE-A-FE.
BOUNDS = {
    "asset_stop_loss": (Decimal("0.0025"), Decimal("0.05")),
    "type_stop_loss": (Decimal("0.005"), Decimal("0.10")),
    "fund_stop_loss": (Decimal("0.01"), Decimal("0.10")),
    "fund_gross": (Decimal("0.1"), Decimal("3")),
    "asset_net": (Decimal("0.05"), Decimal("1")),
    "fund_margin": (Decimal("0.10"), Decimal("0.80")),
    "fund_day_loss": (Decimal("0.01"), Decimal("0.05")),
    "fund_drawdown": (Decimal("0.05"), Decimal("0.25")),
}
LOCKED = frozenset({"fund_stop_loss", "fund_day_loss", "fund_drawdown"})  # always on
# Version 1 (HoE, PM DEFAULTS-2): every test on.
DEFAULTS = {
    "asset_stop_loss": Decimal("0.04"), "type_stop_loss": Decimal("0.06"), "fund_stop_loss": Decimal("0.10"),
    "fund_gross": Decimal("1.5"), "asset_net": Decimal("0.5"), "fund_margin": Decimal("0.5"),
    "fund_day_loss": Decimal("0.03"), "fund_drawdown": Decimal("0.15"),
}
DEFAULT_GROUPS = {"BTC": "large", "ETH": "large", "SOL": "mid", "XRP": "mid"}
OTHER_GROUP = "small"  # every other underlying in the instruments table when v1 is seeded


def _pct(x: Decimal) -> str:
    return f"{x * 100:.2f}".rstrip("0").rstrip(".") + "%"


def validate(tests: list[dict]) -> list[str]:
    """What is wrong with a whole set of tests, in the PM's words; empty when it can be saved."""
    errors, seen = [], set()
    for t in tests:
        key = t.get("test")
        if key not in SHAPE:
            errors.append(f"unknown test {key!r}")
            continue
        if key in seen:
            errors.append(f"{LABELS[key]} is listed twice")
        seen.add(key)
        if (t.get("scope"), t.get("measure"), t.get("action")) != SHAPE[key]:
            errors.append(f"{LABELS[key]}: what it measures and does can't change")
        lo, hi = BOUNDS[key]
        try:
            limit = Decimal(str(t.get("limit")))
        except Exception:
            errors.append(f"{LABELS[key]}: the limit is not a number")
            continue
        if not lo <= limit <= hi:
            show = (lambda x: f"{x:g}x") if SHAPE[key][1] in ("gross", "net") else _pct
            errors.append(f"{LABELS[key]} must be between {show(lo)} and {show(hi)}")
        if key in LOCKED and not t.get("enabled"):
            errors.append(f"{LABELS[key]} can't be switched off")
    errors += [f"{LABELS[k]} is missing" for k in KEYS if k not in seen]
    return errors


def default_tests() -> list[dict]:
    return [{"test": k, "label": LABELS[k], "scope": s, "measure": m, "limit": DEFAULTS[k], "action": a,
             "enabled": True} for k, (s, m, a) in SHAPE.items()]
