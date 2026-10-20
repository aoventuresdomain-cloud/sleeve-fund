"""The portfolio compliance tests (Bridge Drop A; Advisor COMPLIANCE-5 and BRIDGE-3, 14 Oct 2026) as the gate applies
them: one version of the Data Architect's rows (Store.compliance, versioned, whole, write-once), which backtest,
portfolio_run and paper all read, mapped to a risk.PortfolioProfile for limits.decide.

The keys, labels, bounds and locks are compliance_rules' (the one source the Portfolio Settings form and the store's
write share); this module only maps each key to the profile field the gate checks:
  T1  asset_stop_loss  asset_open_risk   loss if every stop is hit, per asset            trim
  T2  type_stop_loss   group_open_risk   the same, per asset group (the version's map)   trim
  T3  fund_stop_loss   open_risk         the same, on the whole fund                     trim
  T4a fund_gross       gross             sum of every position's notional               trim
  T4b asset_net        net_instrument    long less short notional in one asset          trim
  T4c fund_margin      margin            posted isolated margin                         trim
  T5a fund_day_loss    daily_loss        loss since 00:00 UTC: no new entries that day   pause_day
  T5b fund_drawdown    drawdown          from the high-water mark: flatten and halt      halt
"""

from __future__ import annotations

from dataclasses import fields, replace
from decimal import Decimal

from sleeve_fund import compliance_rules as rules
from sleeve_fund.risk import PortfolioProfile

# profile field -> the test's key (gate_decisions.test_hit, compliance_tests.test)
KEY = {
    "asset_open_risk": "asset_stop_loss",
    "group_open_risk": "type_stop_loss",
    "open_risk": "fund_stop_loss",
    "gross": "fund_gross",
    "net_instrument": "asset_net",
    "margin": "fund_margin",
    "daily_loss": "fund_day_loss",
    "drawdown": "fund_drawdown",
}
FIELD = {k: f for f, k in KEY.items()}
TESTS = tuple(KEY)
# Each key's (default, low, high) from compliance_rules, the Front-end's one source (Rule rows; or the DA's stand-in's
# DEFAULTS and BOUNDS dicts).
_RULE = {k: ((r.default, r.low, r.high) if (r := getattr(rules, "BY_KEY", {}).get(k)) is not None
             else (rules.DEFAULTS[k], *rules.BOUNDS[k])) for k in KEY.values()}
LABELS = {f: rules.LABELS[k] for f, k in KEY.items()}
BOUNDS = {f: (float(_RULE[k][1]), float(_RULE[k][2])) for f, k in KEY.items()}
LOCKED = tuple(f for f, k in KEY.items() if k in rules.LOCKED)  # none since B-31: every test can be switched off (PM)
# The seeded version 1's tests (PM DEFAULTS-2: T1 4%, T2 6%, T3 10%; T4 and T5 the accepted portfolio limits). Its
# asset-group map lives with the version in the journal (compliance_rules.DEFAULT_GROUPS), not here.
SEEDED = replace(PortfolioProfile(), **{f: float(_RULE[k][0]) for f, k in KEY.items()})
# A refusal because the fund is halted or paused for the day is T5's (compliance_rules.LIMIT_HIT)
_OUTCOME = {"halt": "fund_drawdown", "pause": "fund_day_loss"}


def test_key(limit_hit: str | None) -> str | None:
    """The compliance test a gate outcome names (limits.decide's limit_hit), or None for an outcome that isn't a test
    (lock_error, portfolio_state_stale, below_min). A halt or a day's pause in force is T5's."""
    return (KEY.get(limit_hit) or _OUTCOME.get(limit_hit)) if limit_hit else None


def validate_tests(profile: PortfolioProfile) -> None:
    """Refuse a test set the PM may not save, in plain words: a locked test switched off, a value outside its bounds,
    or a daily pause not below the drawdown halt."""
    for f in TESTS:
        value = getattr(profile, f)
        if value is None:
            if f in LOCKED:
                raise ValueError(f"{LABELS[f]} can't be switched off")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)) or value != value:
            raise ValueError(f"{LABELS[f]} must be a number, not {value!r}")
        low, high = BOUNDS[f]
        if not low <= float(value) <= high:
            raise ValueError(f"{LABELS[f]} must be between {low:g} and {high:g}, not {float(value):g}")
    if profile.daily_loss is not None and profile.drawdown is not None and profile.daily_loss >= profile.drawdown:
        raise ValueError("the daily loss pause must be below the drawdown halt")


def from_compliance(comp) -> PortfolioProfile:
    """A PortfolioProfile from one Store.compliance() version. A test the version holds disabled is off (None)."""
    values = {f: None for f in TESTS}
    for t in comp.tests:
        f = FIELD.get(t["test"])
        if f is not None and t["enabled"]:
            values[f] = float(str(t["limit"]))
    return PortfolioProfile(version=comp.version, groups=dict(comp.asset_types), **values)


def to_tests(profile: PortfolioProfile, current=None) -> list[dict]:
    """The rows a version of `profile` is saved as: every test, a switched-off one kept disabled at its last limit (or
    its default) so its history reads."""
    out = []
    for f, k in KEY.items():
        value = getattr(profile, f)
        scope, measure, action = rules.SHAPE[k]
        was = current.test(k) if current is not None else None
        limit = (Decimal(str(value)).quantize(Decimal("0.0001")) if value is not None
                 else (was["limit"] if was is not None else _RULE[k][0]))
        out.append({"test": k, "label": rules.LABELS[k], "scope": scope, "measure": measure, "limit": limit,
                    "action": action, "enabled": value is not None})
    return out


def load(store, mode: str = "paper", version: int | None = None, conn=None) -> PortfolioProfile:
    """The compliance tests in force (the latest version, or the one named) as the gate applies them. Every run mode
    reads the same rows (BRIDGE-3); `mode` names the reader in its error. conn: the gate's own, inside its lock.
    Raises (never falls back to tests nobody accepted) when the journal holds none or they can't be read."""
    reader = getattr(store, "compliance", None)
    if reader is None:  # a store from before the compliance tables: the PM's portfolio limits, T1 and T2 off
        p = store.portfolio_profile(version)
        return PortfolioProfile(**{f.name: getattr(p, f.name) for f in fields(p)
                                   if f.name not in ("asset_open_risk", "group_open_risk")})
    try:
        comp = reader(version, conn=conn) if conn is not None else reader(version)
    except Exception as e:  # noqa: BLE001 - said in words; the gate refuses on it
        raise LookupError(f"the compliance tests couldn't be read for {mode}: {e}") from e
    return from_compliance(comp)


def save(store, profile: PortfolioProfile, changed_by: str, note: str | None = None) -> int:
    """Record `profile` (its tests and its group map) as the next version, after validate_tests; returns the version.
    A profile the latest version already holds, value for value, writes nothing and returns that version. PM only
    (the store's write refuses anyone else)."""
    validate_tests(profile)
    current = store.compliance()
    tests = to_tests(profile, current)
    groups = dict(profile.groups if profile.groups is not None else current.asset_types)
    same = ({(t["test"], Decimal(str(t["limit"])), t["enabled"]) for t in tests}
            == {(t["test"], Decimal(str(t["limit"])), t["enabled"]) for t in current.tests})
    if same and groups == dict(current.asset_types):
        return current.version
    return store.add_compliance_version(tests, groups, expected_version=current.version,
                                        note=note or f"compliance tests set by {changed_by}", created_by=changed_by)
