"""P2-5 acceptance test for regime weighting (Advisor spec quant-review/specs-p2-3-p2-8-p2-5.md section 4, MUST): the
weighted run against two nulls, the unweighted run, double fees and a minimum of entries per state. A fail means the
weight is 1.0 everywhere; nothing here re-tunes a mapping to rescue it (a new mapping is a new counted idea).

Primary null: the regime state series circularly shifted by a random offset of at least MIN_SHIFT_DAYS per draw, which
keeps how long regimes last and how often each weight is used but breaks their timing. Secondary null: each entry's
weight drawn independently with the frequencies the regime run used. DRAWS seeded draws of each; the seed is fixed and
reported.

How a draw is made (Advisor 13 Oct 19:45 UK, method A): the engine runs unweighted and regime-weighted for real. A
trade's size scales with its entry weight, fixed for the trade's life (an add carries it), so a draw's P&L is each
trade's P&L at weight 1, from the unweighted run, times its weight. The regime run is rebuilt the same way and ranked
against the rebuilt nulls, so the approximation sits on both sides; the real runs decide the other lines (at least the
unweighted Sharpe, drawdown, double fees). The rebuild must be faithful or the result is NOT JUDGED: the rebuilt
regime Sharpe within FIDELITY_SHARPE of the real regime run's, and the two runs' entries the same but for at most
FIDELITY_ENTRIES of them (a smaller weight can let a limit or a halt bind differently)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from sleeve_fund.portfolio.regime_weight import DIRECTION, weight
from sleeve_fund.research import metrics

DRAWS = 1000
MIN_SHIFT_DAYS = 60
PCTL = 95.0
MAX_DD_WORSE = 0.10  # relative: the weighted run's max drawdown at most 10% deeper than the unweighted run's
MIN_ENTRIES_PER_STATE = 30
FIDELITY_SHARPE = 0.05
FIDELITY_ENTRIES = 0.02
SEED = 20261013
STATES = ("low", "mid", "high")
NOT_JUDGED, PASS, FAIL = "NOT JUDGED", "PASS", "FAIL"


def shifted_states(states: pd.Series, rng: np.random.Generator, min_shift: int = MIN_SHIFT_DAYS) -> pd.Series:
    """The primary null's state series: `states` (one per day, in date order) rolled by an offset in
    [min_shift, len - min_shift], wrapping round, on the same dates."""
    n = len(states)
    if n < 2 * min_shift + 1:
        raise ValueError(f"a shifted null needs more than {2 * min_shift} days of regime states, got {n}")
    k = int(rng.integers(min_shift, n - min_shift + 1))
    return pd.Series(np.roll(states.to_numpy(), k), index=states.index)


def iid_weights(used: Sequence, rng: np.random.Generator) -> list:
    """The secondary null's weights: one per entry, drawn independently with the frequencies of `used` (the weights
    the regime run gave its entries)."""
    values, counts = np.unique(np.asarray([float(w) for w in used]), return_counts=True)
    return list(rng.choice(values, size=len(used), p=counts / counts.sum()))


def daily_sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe of daily net returns (365 days: the venue trades every day)."""
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(365)) if len(r) > 1 and sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    """Deepest fall of the compounded curve from its high-water mark, as a positive fraction."""
    curve = (1 + returns.fillna(0)).cumprod()
    return float((1 - curve / curve.cummax()).max()) if len(curve) else 0.0


@dataclass(frozen=True)
class Verdict:
    result: str  # PASS, FAIL or NOT JUDGED
    sharpe: float
    unweighted_sharpe: float
    primary_p95: float
    secondary_p95: float
    drawdown: float
    unweighted_drawdown: float
    double_fees: bool
    entries: dict
    reasons: tuple[str, ...]
    rebuilt_sharpe: float = float("nan")
    seed: int | None = None


def judge(weighted: pd.Series, unweighted: pd.Series, primary: Sequence[float], secondary: Sequence[float],
          double_fees_pass: bool, entries_per_state: dict, rebuilt_sharpe: float, unfaithful: Sequence[str] = (),
          seed: int | None = None) -> Verdict:
    """Section 4.4 on the out-of-sample daily net returns of the real runs, with the nulls ranked against the rebuilt
    regime Sharpe: every condition must hold. An unfaithful rebuild, fewer than MIN_ENTRIES_PER_STATE entries in any
    state, or a null that isn't complete is NOT JUDGED, never a pass. The holdout confirm is a separate single run
    after a pass."""
    sr, base, rb = daily_sharpe(weighted), daily_sharpe(unweighted), rebuilt_sharpe
    p1 = float(np.percentile(primary, PCTL)) if len(primary) else float("nan")
    p2 = float(np.percentile(secondary, PCTL)) if len(secondary) else float("nan")
    dd, base_dd = max_drawdown(weighted), max_drawdown(unweighted)
    entries = {s: int(entries_per_state.get(s, 0)) for s in STATES}
    short = [s for s, n in entries.items() if n < MIN_ENTRIES_PER_STATE]
    args = (sr, base, p1, p2, dd, base_dd, double_fees_pass, entries)
    if unfaithful:
        return Verdict(NOT_JUDGED, *args, tuple(unfaithful), rb, seed)
    if short:
        why = f"fewer than {MIN_ENTRIES_PER_STATE} entries in {', '.join(short)}"
        return Verdict(NOT_JUDGED, *args, (why,), rb, seed)
    if len(primary) < DRAWS or len(secondary) < DRAWS or not np.isfinite([sr, base, rb, p1, p2]).all():
        return Verdict(NOT_JUDGED, *args, (f"the nulls need {DRAWS} draws each and every Sharpe measured",), rb, seed)
    fails = []
    if not rb > p1:
        fails.append(f"rebuilt Sharpe {rb:.2f} not above the shifted-regime null's 95th percentile {p1:.2f}")
    if not rb > p2:
        fails.append(f"rebuilt Sharpe {rb:.2f} not above the random-weight null's 95th percentile {p2:.2f}")
    if sr < base:
        fails.append(f"Sharpe {sr:.2f} below unweighted {base:.2f}")
    if dd > base_dd * (1 + MAX_DD_WORSE):
        fails.append(f"max drawdown {dd:.1%} more than 10% worse than unweighted {base_dd:.1%}")
    if not double_fees_pass:
        fails.append("fails at double fees")
    return Verdict(FAIL if fails else PASS, *args, tuple(fails), rb, seed)


@dataclass(frozen=True)
class Trades:
    """The unweighted run cut into trades: per Strategy, its equity change over each of its bars and the opening
    entry (an index into `entries`) whose trade that change belongs to, -1 while flat. entries: (strategy, bar_ts) of
    every approved opening entry, in order."""

    start: float  # the book at the window's start: every Strategy's equity plus the unallocated cash
    changes: dict  # strategy -> pd.DataFrame(index=bar close, columns=["change", "entry"])
    entries: tuple


def trades(run) -> Trades:
    """Cut an unweighted PortfolioResult into trades. A bar's equity change belongs to the latest opening entry decided
    before the bar's close, while a position was held at either end of the bar or that entry was decided at its start;
    when flat it moves nothing."""
    opening = [d for d in run.decisions if d.get("kind") == "entry" and d["approved"] > 0]
    entries, changes, start = [], {}, float(run.unallocated)
    for name, leg in run.legs.items():
        mine = [d for d in opening if d["strategy"] == name]
        first = len(entries)
        entries += [(name, d["bar_ts"]) for d in mine]
        eq = leg.equity.astype(float)
        start += float(eq.iloc[0])
        held = leg.exposure.reindex(eq.index).fillna(0.0).to_numpy() != 0
        held = held | np.concatenate([[False], held[:-1]])
        at = pd.DatetimeIndex([pd.Timestamp(d["bar_ts"]) for d in mine])
        last = at.searchsorted(eq.index, side="left") - 1  # decided strictly before this bar's close
        # A trade opened and closed inside one bar is held at neither end: its entry was decided at the bar's start.
        bar_open = pd.DatetimeIndex([eq.index[0], *eq.index[:-1]])
        opened = ((last >= 0) & (at[np.maximum(last, 0)] >= bar_open) if len(at) else np.zeros(len(eq), bool))
        entry = np.where((held | opened) & (last >= 0), last + first, -1)
        changes[name] = pd.DataFrame({"change": eq.diff().fillna(0.0).to_numpy(), "entry": entry}, index=eq.index)
    return Trades(start, changes, tuple(entries))


def rebuilt_returns(t: Trades, weights: Sequence[float]) -> pd.Series:
    """Daily returns of the book with each trade's P&L scaled by its entry's weight (weights: one per t.entries)."""
    w = np.append(np.asarray(weights, dtype=float), 1.0)  # index -1: flat, weight 1
    pnl = [(c["change"] * w[c["entry"].to_numpy()]).cumsum().rename(name) for name, c in t.changes.items()]
    book = t.start + pd.concat(pnl, axis=1).sort_index().ffill().fillna(0.0).sum(axis=1)
    return metrics.daily_returns(book)


def regime_weights(t: Trades, state_on: Callable[[date], str | None], families: Mapping[str, str]) -> list[float]:
    """Each trade's weight from its entry day's regime state (the row dated that day), as Weighting gives it."""
    return [float(weight(families.get(s, ""), state_on(pd.Timestamp(ts).date()))) for s, ts in t.entries]


def oos(returns: pd.Series, folds: Sequence[tuple]) -> pd.Series:
    """The out-of-sample days only: those inside any (start, end) fold, both ends included."""
    keep = np.zeros(len(returns), dtype=bool)
    for a, b in folds:
        keep |= (returns.index >= pd.Timestamp(a)) & (returns.index <= pd.Timestamp(b))
    return returns[keep]


def _opening(run) -> set:
    return {(d["strategy"], d["bar_ts"]) for d in run.decisions if d.get("kind") == "entry" and d["approved"] > 0}


def acceptance(unweighted, weighted, states: pd.Series, families: Mapping[str, str], folds: Sequence[tuple],
               double_fees: tuple | None, seed: int = SEED, draws: int = DRAWS) -> Verdict:
    """Section 4 on two real PortfolioResults over the same Strategies, windows and fee model: `weighted` ran with
    Weighting over `states` (the regime state per date, a datetime.date index). double_fees: the same pair at double
    fees, judged the same way with the same seed (it passes when its own verdict passes); None when not yet run, and
    then the result is at best NOT JUDGED."""
    v = _verdict(unweighted, weighted, states, families, folds, seed, draws, None)
    if double_fees is None:
        return v if v.result != PASS else replace(v, result=NOT_JUDGED, reasons=("the double-fee pair hasn't run",))
    twice = _verdict(*double_fees, states, families, folds, seed, draws, None).result == PASS
    return _verdict(unweighted, weighted, states, families, folds, seed, draws, twice)


def _verdict(unweighted, weighted, states, families, folds, seed, draws, twice: bool | None) -> Verdict:
    rng = np.random.default_rng(seed)
    t = trades(unweighted)
    w = regime_weights(t, states.get, families)
    rebuilt = daily_sharpe(oos(rebuilt_returns(t, w), folds))
    real_w = oos(metrics.daily_returns(weighted.book), folds)
    real_u = oos(metrics.daily_returns(unweighted.book), folds)
    unfaithful = []
    if not abs(rebuilt - daily_sharpe(real_w)) <= FIDELITY_SHARPE:
        unfaithful.append(f"the rebuild is off the real regime run: Sharpe {rebuilt:.3f} vs {daily_sharpe(real_w):.3f}")
    a, b = _opening(unweighted), _opening(weighted)
    if len(a ^ b) > FIDELITY_ENTRIES * max(len(a), 1):
        unfaithful.append(f"{len(a ^ b)} of {len(a)} entries differ between the weighted and unweighted runs")
    weighed = [families.get(s, "") in DIRECTION for s, _ in t.entries]  # unregistered families stay at 1.0
    used = [x for x, f in zip(w, weighed) if f]
    primary, secondary = [], []
    for _ in range(draws):
        shifted = shifted_states(states, rng)
        primary.append(daily_sharpe(oos(rebuilt_returns(t, regime_weights(t, shifted.get, families)), folds)))
        drawn = iter(iid_weights(used, rng) if used else [])
        secondary.append(daily_sharpe(oos(rebuilt_returns(t, [next(drawn) if f else x for x, f in zip(w, weighed)]),
                                          folds)))
    per_state: dict = {}
    for d in weighted.decisions:
        if d.get("kind") == "entry" and d["approved"] > 0:
            per_state[d["regime_state"]] = per_state.get(d["regime_state"], 0) + 1
    return judge(real_w, real_u, primary, secondary, bool(twice), per_state, rebuilt, unfaithful, seed)
