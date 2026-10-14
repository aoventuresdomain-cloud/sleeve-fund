"""The portfolio compliance tests (Bridge Drop A; Advisor COMPLIANCE-5 and BRIDGE-3, 14 Oct 2026): the one set of rows
backtest, portfolio_run and paper all read, as a risk.PortfolioProfile the gate (limits.decide) applies.

The rows are the Data Architect's (store.compliance / store.add_compliance_version: versioned, whole, write-once); this
module maps them to and from the profile, and holds the bounds a PM edit is checked against before it is saved.

Tests and their keys (gate_decisions.test_hit, the FE's T1..T5b):
  T1  asset_open_risk   loss if every stop is hit, per asset            trim
  T2  group_open_risk   the same, per asset group (the PM's group map)  trim
  T3  open_risk         the same, on the whole fund                     trim
  T4a gross             sum of every position's notional               trim
  T4b net_instrument    long less short notional in one asset          trim
  T4c margin            posted isolated margin                         trim
  T5a daily_loss        loss since 00:00 UTC: no new entries that day   pause
  T5b drawdown          from the high-water mark: flatten and halt      halt
"""

from __future__ import annotations

from dataclasses import fields, replace

from sleeve_fund.risk import PortfolioProfile

# (scope, measure, action) of each test, as the DA's compliance_tests rows hold them
SHAPE = {
    "asset_open_risk": ("asset", "open_risk", "trim"),
    "group_open_risk": ("asset_type", "open_risk", "trim"),
    "open_risk": ("fund", "open_risk", "trim"),
    "gross": ("fund", "gross", "trim"),
    "net_instrument": ("asset", "net", "trim"),
    "margin": ("fund", "margin", "trim"),
    "daily_loss": ("fund", "day_loss", "pause"),
    "drawdown": ("fund", "drawdown", "halt"),
}
LABELS = {
    "asset_open_risk": "Loss per asset if every stop is hit",
    "group_open_risk": "Loss per asset group if every stop is hit",
    "open_risk": "Loss for the whole fund if every stop is hit",
    "gross": "Gross exposure",
    "net_instrument": "Net exposure per asset",
    "margin": "Margin in use",
    "daily_loss": "Daily loss that pauses new trades",
    "drawdown": "Drawdown that halts the fund",
}
# What the PM may set each to, inclusive, as a share of fund equity (x book for gross and net). T3 and T5 bounds are the
# Advisor's (BRIDGE-3); the rest are wide guards against a slip of the finger, the Advisor's to tighten.
BOUNDS = {
    "asset_open_risk": (0.005, 0.10),
    "group_open_risk": (0.005, 0.15),
    "open_risk": (0.01, 0.10),
    "gross": (0.1, 3.0),
    "net_instrument": (0.05, 1.5),
    "margin": (0.05, 1.0),
    "daily_loss": (0.01, 0.05),
    "drawdown": (0.05, 0.25),
}
LOCKED = ("open_risk", "daily_loss", "drawdown")  # T3 and T5 can't be switched off (BRIDGE-3)
TESTS = tuple(SHAPE)
# The PM's defaults (14 Oct ~22:45 UK): T1 4%, T2 6%, T3 10%; T4 and T5 as the accepted portfolio limits. The first
# compliance version is seeded with these (the DA's migration), and a run with no journal of its own reads them.
SEEDED = replace(PortfolioProfile(), asset_open_risk=0.04, group_open_risk=0.06, open_risk=0.10)


def validate_tests(profile: PortfolioProfile) -> None:
    """Refuse a test set the PM may not save, with plain words: a locked test switched off, a value outside its bounds,
    or a daily pause not below the drawdown halt."""
    for key in TESTS:
        value = getattr(profile, key)
        if value is None:
            if key in LOCKED:
                raise ValueError(f"{LABELS[key]} can't be switched off")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise ValueError(f"{LABELS[key]} must be a number, not {value!r}")
        low, high = BOUNDS[key]
        if not low <= value <= high:
            raise ValueError(f"{LABELS[key]} must be between {low:g} and {high:g}, not {value:g}")
    if profile.daily_loss >= profile.drawdown:
        raise ValueError("the daily loss pause must be below the drawdown halt")


def _get(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def from_rows(version: int, tests, asset_types) -> PortfolioProfile:
    """A PortfolioProfile from one version's rows (tests: each with test, limit, enabled; asset_types: {underlying:
    group} or rows of underlying, asset_type). A test the version doesn't hold, or holds disabled, is off."""
    values = {key: None for key in TESTS}
    for row in tests:
        key = _get(row, "test")
        if key in values and _get(row, "enabled"):
            values[key] = float(str(_get(row, "limit")))
    groups = (dict(asset_types) if isinstance(asset_types, dict)
              else {_get(r, "underlying"): _get(r, "asset_type") for r in asset_types})
    return PortfolioProfile(version=version, groups=groups, **values)


def to_rows(profile: PortfolioProfile) -> tuple[list[dict], dict[str, str]]:
    """The rows a version of `profile` is saved as: every test (a switched-off one kept, disabled, at its last bound
    so its history reads) and the group map."""
    tests = []
    for key in TESTS:
        value = getattr(profile, key)
        scope, measure, action = SHAPE[key]
        tests.append({"test": key, "label": LABELS[key], "scope": scope, "measure": measure, "action": action,
                      "limit": value if value is not None else BOUNDS[key][1], "enabled": value is not None})
    return tests, dict(profile.groups or {})


def load(store, mode: str = "paper", version: int | None = None) -> PortfolioProfile:
    """The compliance tests in force (the latest version, or the one named) as the gate applies them. Every run mode
    reads the same rows (BRIDGE-3); `mode` is the reader's name, for its error only. Raises LookupError when the
    journal holds none, so no caller falls back to tests nobody accepted."""
    reader = getattr(store, "compliance", None)
    if reader is None:  # a journal from before the compliance tables: the PM's portfolio limits, T1 and T2 off
        p = store.portfolio_profile(version)
        return PortfolioProfile(**{f.name: getattr(p, f.name) for f in fields(p)
                                   if f.name not in ("asset_open_risk", "group_open_risk")},
                                asset_open_risk=None, group_open_risk=None)
    got = reader(version)
    if got is None:
        raise LookupError(f"no compliance tests recorded for {mode}")
    return from_rows(_get(got, "version"), _get(got, "tests"), _get(got, "asset_types"))


def save(store, profile: PortfolioProfile, changed_by: str, note: str | None = None) -> int:
    """Record `profile` as the next version, after validate_tests; returns the version. PM only (the DA's write path
    refuses anyone else)."""
    validate_tests(profile)
    tests, groups = to_rows(profile)
    return store.add_compliance_version(tests, groups, created_by=changed_by, note=note)
