"""P2-5 acceptance test for regime weighting (Advisor spec quant-review/specs-p2-3-p2-8-p2-5.md section 4, MUST): the
weighted run against two nulls, the unweighted run, double fees and a minimum of entries per state. A fail means the
weight is 1.0 everywhere; nothing here re-tunes a mapping to rescue it (a new mapping is a new counted idea).

Primary null: the regime state series circularly shifted by an offset in [L, N - L], L = max(MIN_SHIFT_DAYS,
SPELL_MULTIPLE x the mean spell length of the run's states), which keeps how long regimes last and how often each weight
is used but breaks their timing (a shift near N is a small shift the other way, so the top of the range is cut too).
When fewer than DRAWS shifts are admissible, every one is used once instead of drawing (Advisor 14 Oct 08:45 UK).
Secondary null: each entry's weight drawn independently with the frequencies the regime run used. DRAWS seeded draws;
the seed is fixed and reported. A pass needs both nulls beaten at the 95th percentile and the weighted book's Sharpe at
least the unweighted book's.

The decision is on the book the PM would run (its Sharpe and drawdown); the per-family figures are reported, never
judged. The verdict reads the development period only: the runs must end before the holdout starts, and the holdout is
read once, after a pass, as its own run.

How a draw is made (Advisor 13 Oct 19:45 UK, method A): the engine runs unweighted and regime-weighted for real. A
trade's size scales with its entry weight, fixed for the trade's life (an add carries it), so a draw's P&L is each
trade's P&L at weight 1, from the unweighted run, times its weight. The regime run is rebuilt the same way and ranked
against the rebuilt nulls, so the approximation sits on both sides; the real runs decide the other lines (at least the
unweighted Sharpe, drawdown, double fees). The rebuild must be faithful or the result is NOT JUDGED: the rebuilt
regime Sharpe within FIDELITY_SHARPE of the real regime run's, and the two runs' entries the same but for at most
FIDELITY_ENTRIES of them (a smaller weight can let a limit or a halt bind differently).

A second, exact fidelity check (Advisor 14 Oct 08:45 UK, M3): the regime run with every weight forced to 1.0 (a
Weighting with no family registered) must be the unweighted run itself, the same entries and the book the same to the
cent, and the rebuild at weight 1.0 must be the unweighted book to the cent. Any difference is a bug, so NOT JUDGED."""

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
SPELL_MULTIPLE = 3  # the shift is at least this many mean regime spells, so a shifted state is not the same spell
CENT = 0.005  # half a cent: two books are the same to the cent when no mark differs by this much
PCTL = 95.0
MAX_DD_WORSE = 0.10  # relative: the weighted run's max drawdown at most 10% deeper than the unweighted run's
MIN_ENTRIES_PER_STATE = 30
FIDELITY_SHARPE = 0.05
FIDELITY_ENTRIES = 0.02
SEED = 20261013
STATES = ("low", "mid", "high")
NOT_JUDGED, PASS, FAIL = "NOT JUDGED", "PASS", "FAIL"


def mean_spell(states: pd.Series) -> float:
    """The mean length in days of the runs of one unchanged state in `states` (in date order)."""
    v = states.to_numpy()
    return len(v) / (1 + int((v[1:] != v[:-1]).sum())) if len(v) else 0.0


def min_shift(states: pd.Series) -> int:
    """L, the smallest shift the primary null may use: MIN_SHIFT_DAYS or SPELL_MULTIPLE mean spells, the larger."""
    return max(MIN_SHIFT_DAYS, int(np.ceil(SPELL_MULTIPLE * mean_spell(states))))


def shifts(states: pd.Series, rng: np.random.Generator, draws: int = DRAWS) -> list[int]:
    """The primary null's offsets: `draws` drawn from [L, N - L], or every one of them once when there are fewer."""
    n, low = len(states), min_shift(states)
    if n < 2 * low + 1:
        raise ValueError(f"a shifted null needs more than {2 * low} days of regime states, got {n}")
    every = n - 2 * low + 1
    if every < draws:
        return list(range(low, n - low + 1))
    return [int(k) for k in rng.integers(low, n - low + 1, size=draws)]


def shifted_states(states: pd.Series, k: int) -> pd.Series:
    """The primary null's state series: `states` (one per day, in date order) rolled by k days, wrapping round, on the
    same dates."""
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
    families: dict = None  # family -> {"sharpe", "unweighted_sharpe", "drawdown", "unweighted_drawdown"}: reported only


def judge(weighted: pd.Series, unweighted: pd.Series, primary: Sequence[float], secondary: Sequence[float],
          double_fees_pass: bool, entries_per_state: dict, rebuilt_sharpe: float, unfaithful: Sequence[str] = (),
          seed: int | None = None, primary_complete: int = DRAWS) -> Verdict:
    """Section 4.4 on the out-of-sample daily net returns of the real runs, with the nulls ranked against the rebuilt
    regime Sharpe: every condition must hold. An unfaithful rebuild, fewer than MIN_ENTRIES_PER_STATE entries in any
    state, or a null that isn't complete is NOT JUDGED, never a pass. primary_complete: how many primary draws make it
    complete (DRAWS, or every admissible shift when there are fewer). The holdout confirm is a separate single run
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
    short_null = len(primary) < min(primary_complete, DRAWS) or len(secondary) < DRAWS
    if short_null or not np.isfinite([sr, base, rb, p1, p2]).all():
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


def rebuilt_book(t: Trades, weights: Sequence[float]) -> pd.Series:
    """The book with each trade's P&L scaled by its entry's weight (weights: one per t.entries)."""
    w = np.append(np.asarray(weights, dtype=float), 1.0)  # index -1: flat, weight 1
    pnl = [(c["change"] * w[c["entry"].to_numpy()]).cumsum().rename(name) for name, c in t.changes.items()]
    return t.start + pd.concat(pnl, axis=1).sort_index().ffill().fillna(0.0).sum(axis=1)


def rebuilt_returns(t: Trades, weights: Sequence[float]) -> pd.Series:
    """Daily returns of rebuilt_book."""
    return metrics.daily_returns(rebuilt_book(t, weights))


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


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def development_only(runs: Sequence, folds: Sequence[tuple], holdout_start) -> None:
    """Raise unless every run and fold ends before the holdout (None only when the data keeps no holdout): the verdict
    never reads the held-back period, which is read once, after a pass (Advisor 14 Oct 08:45 UK, M5)."""
    if holdout_start is None:
        return
    held = _utc(holdout_start)
    for _, end in folds:
        if _utc(end) >= held:
            raise ValueError(f"the regime verdict never reads the held-back period: a fold to {_utc(end):%d %b %Y} "
                             f"reaches the holdout from {held:%d %b %Y}")
    for run in runs:
        if len(run.book) and _utc(run.book.index[-1]) >= held:
            raise ValueError(f"the regime verdict never reads the held-back period: a run to "
                             f"{_utc(run.book.index[-1]):%d %b %Y} reaches the holdout from {held:%d %b %Y}")


def _sized(run) -> list:
    return sorted((d["strategy"], d["bar_ts"], d["approved"]) for d in run.decisions
                  if d.get("kind") == "entry" and d["approved"] > 0)


def exact(unweighted, ones) -> list[str]:
    """Why the run with every weight forced to 1.0 isn't the unweighted run, or [] when it is: the same opening
    entries at the same sizes and the book the same to the cent at every mark (M3)."""
    if ones is None:
        return ["the run with every weight forced to 1.0 hasn't run"]
    why = []
    if _sized(ones) != _sized(unweighted):
        why.append("with every weight forced to 1.0 the entries differ from the unweighted run's")
    a, b = unweighted.book.astype(float), ones.book.astype(float)
    if not a.index.equals(b.index) or (a - b).abs().max() >= CENT:
        why.append("with every weight forced to 1.0 the book differs from the unweighted run's")
    return why


def _family_books(run, families: Mapping[str, str]) -> dict:
    books: dict = {}
    for name, leg in run.legs.items():
        f = families.get(name, "") or "unregistered"
        books[f] = leg.equity.astype(float) if f not in books else books[f].add(leg.equity.astype(float),
                                                                                  fill_value=0.0)
    return books


def per_family(unweighted, weighted, families: Mapping[str, str], folds: Sequence[tuple]) -> dict:
    """Each family's Sharpe and drawdown, weighted and unweighted, out of sample: reported, never judged (M4)."""
    u, w = _family_books(unweighted, families), _family_books(weighted, families)
    out = {}
    for f in sorted(u):
        ru, rw = oos(metrics.daily_returns(u[f]), folds), oos(metrics.daily_returns(w.get(f, u[f])), folds)
        out[f] = {"sharpe": daily_sharpe(rw), "unweighted_sharpe": daily_sharpe(ru), "drawdown": max_drawdown(rw),
                  "unweighted_drawdown": max_drawdown(ru)}
    return out


def acceptance(unweighted, weighted, states: pd.Series, families: Mapping[str, str], folds: Sequence[tuple],
               double_fees: tuple | None, *, ones, holdout_start, seed: int = SEED, draws: int = DRAWS) -> Verdict:
    """Section 4 on two real PortfolioResults over the same Strategies, windows and fee model: `weighted` ran with
    Weighting over `states` (the regime state per date, a datetime.date index). double_fees: the same pair at double
    fees, judged the same way with the same seed (it passes when its own verdict passes); None when not yet run, and
    then the result is at best NOT JUDGED. ones: the same run with every weight forced to 1.0 (Weighting with no
    family), which must be the unweighted run exactly; None is NOT JUDGED. holdout_start: where the held-back period
    begins; a run or fold reaching it is refused."""
    development_only([unweighted, weighted, ones, *(double_fees or ())], folds, holdout_start)
    v = _verdict(unweighted, weighted, ones, states, families, folds, seed, draws, None)
    if double_fees is None:
        return v if v.result != PASS else replace(v, result=NOT_JUDGED, reasons=("the double-fee pair hasn't run",))
    twice = _verdict(*double_fees, None, states, families, folds, seed, draws, None, exact_check=False).result == PASS
    return _verdict(unweighted, weighted, ones, states, families, folds, seed, draws, twice)


def _verdict(unweighted, weighted, ones, states, families, folds, seed, draws, twice: bool | None,
             exact_check: bool = True) -> Verdict:
    rng = np.random.default_rng(seed)
    t = trades(unweighted)
    w = regime_weights(t, states.get, families)
    rebuilt = daily_sharpe(oos(rebuilt_returns(t, w), folds))
    real_w = oos(metrics.daily_returns(weighted.book), folds)
    real_u = oos(metrics.daily_returns(unweighted.book), folds)
    unfaithful = exact(unweighted, ones) if exact_check else []
    at_one = metrics.daily_closes(rebuilt_book(t, [1.0] * len(t.entries)))
    book_u = metrics.daily_closes(unweighted.book.astype(float))
    # The book is marked hourly from the window's start; a leg's first mark can come later, so the rebuild is checked
    # on its own days, each of which must be a day of the book.
    if not at_one.index.isin(book_u.index).all() or (at_one - book_u.reindex(at_one.index)).abs().max() >= CENT:
        unfaithful.append("the rebuild at weight 1.0 isn't the unweighted book to the cent")
    if not abs(rebuilt - daily_sharpe(real_w)) <= FIDELITY_SHARPE:
        unfaithful.append(f"the rebuild is off the real regime run: Sharpe {rebuilt:.3f} vs {daily_sharpe(real_w):.3f}")
    a, b = _opening(unweighted), _opening(weighted)
    if len(a ^ b) > FIDELITY_ENTRIES * max(len(a), 1):
        unfaithful.append(f"{len(a ^ b)} of {len(a)} entries differ between the weighted and unweighted runs")
    weighed = [families.get(s, "") in DIRECTION for s, _ in t.entries]  # unregistered families stay at 1.0
    used = [x for x, f in zip(w, weighed) if f]
    try:
        offsets = shifts(states, rng, draws)
    except ValueError as e:  # too few days for the regime spells seen: there is no primary null, so no verdict
        offsets, unfaithful = [], [*unfaithful, str(e)]
    primary = [daily_sharpe(oos(rebuilt_returns(t, regime_weights(t, shifted_states(states, k).get, families)), folds))
               for k in offsets]
    secondary = []
    for _ in range(draws):
        drawn = iter(iid_weights(used, rng) if used else [])
        secondary.append(daily_sharpe(oos(rebuilt_returns(t, [next(drawn) if f else x for x, f in zip(w, weighed)]),
                                          folds)))
    per_state: dict = {}
    for d in weighted.decisions:
        if d.get("kind") == "entry" and d["approved"] > 0:
            per_state[d["regime_state"]] = per_state.get(d["regime_state"], 0) + 1
    v = judge(real_w, real_u, primary, secondary, bool(twice), per_state, rebuilt, unfaithful, seed,
              primary_complete=max(len(offsets), 1))
    return replace(v, families=per_family(unweighted, weighted, families, folds))
