"""A G1 study on the venue history store, run the way the backtest page and paper run: the same fees
(the connected account's, else the published schedule), the same measured spread, the strategy's
risk profile with its halt and pause, and resting orders matched on shorter bars between decisions.

The command line (python -m sleeve_fund study --store) and the Research page both run studies here,
so a tear sheet means the same whichever started it."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
# Where the server keeps research it writes: outside the code, which each deploy replaces.
LEDGER = Path(os.environ.get("IDEA_LEDGER", ROOT / "research" / "idea_ledger.jsonl"))
TEARSHEETS = Path(os.environ.get("TEARSHEET_DIR", ROOT / "research" / "tearsheets"))

# Bar lengths a study can decide on. 5-minute bars take about three times as long as 15-minute ones
# (a 1,000-day study: 149 s and 0.46 GB against 47 s and 0.38 GB). 1-minute bars take hours and more
# memory than a job has (review round 7: a 2-year study ran past 90 minutes and 1.9 GB), so 1-minute
# settings are tried on the backtest page; a study still matches resting orders on shorter bars.
STUDY_MINUTES = (1440, 240, 60, 15, 5)
# The most execution bars one backtest in a study replays: resting orders and the risk guard are
# matched on the shortest bars that keep each run within this (as the backtest page does).
EXEC_BAR_BUDGET = 150_000
EXEC_STEPS = (1, 5, 15, 60)
# Stored history ending longer ago than this is still being backfilled; a study says so.
STALE_HISTORY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class StudyRequest:
    strategy: str
    pair: str  # BASE/QUOTE
    venue: str | None = None
    minutes: int = 1440
    risk_profile: str | None = "balanced"  # None: uncapped and unguarded
    train_days: int = 3 * 365
    test_days: int = 365
    holdout_days: int = 365
    use_holdout: bool = False
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_per_trade: float | None = None
    stop_atr: float | None = None
    stop_swing_bars: int | None = None
    atr_bars: int | None = None
    take_profit_r: float | None = None

    def exits(self) -> dict:
        """The exits every run of the study trades with, as the strategy takes them."""
        keys = ("stop_loss", "take_profit", "risk_per_trade", "stop_atr", "stop_swing_bars", "take_profit_r")
        out = {k: getattr(self, k) for k in keys if getattr(self, k) is not None}
        if self.stop_atr is not None and self.atr_bars is not None:
            out["atr_bars"] = self.atr_bars
        return out

    def validate(self) -> None:
        if self.minutes not in STUDY_MINUTES:
            raise ValueError(f"studies decide on {', '.join(_bars(m) for m in STUDY_MINUTES)} bars; "
                             f"{_bars(self.minutes)} bars would take hours, so try them on the backtest page")
        for name in ("train_days", "test_days"):
            if getattr(self, name) < 30:
                raise ValueError(f"{name.replace('_', ' ')} must be at least 30")
        if self.holdout_days < 0:
            raise ValueError("holdout days can't be negative")
        if "/" not in self.pair:
            raise ValueError(f"instrument {self.pair!r} should look like BTC/USD")


def _bars(minutes: int) -> str:
    return "daily" if minutes == 1440 else f"{minutes // 60}-hour" if minutes % 60 == 0 else f"{minutes}-minute"


def exec_step(span: pd.Timedelta, minutes: int) -> int | None:
    """The execution-bar length for a run of `span` deciding on `minutes` bars, or None when the
    decision bars are already as short as the budget allows."""
    step = next((m for m in EXEC_STEPS if span / pd.Timedelta(minutes=m) <= EXEC_BAR_BUDGET), None)
    return step if step is not None and step < minutes and minutes % step == 0 else None


def spec_of(strategy: str):
    try:
        return importlib.import_module(f"sleeve_fund.strategies.{strategy}").SPEC
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(f"no strategy module with a SPEC called {strategy!r}") from exc


def dataset_name(venue: str, pair: str, minutes: int) -> str:
    base, quote = pair.split("/")
    return f"{venue}-{base}{quote}-store".lower() + (f"-{minutes}m" if minutes != 1440 else "")


def run_store_study(req: StudyRequest, store=None, progress=None, ledger_path: Path | None = None,
                    out_dir: Path | None = None, history=None) -> Path:
    """Run the study and write its tear sheet; returns the tear sheet's path. store: the journal
    (for the connected account's fees and the measured spread), if there is one."""
    from sleeve_fund import spreads
    from sleeve_fund.fees import resolve as resolve_fees
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.instruments import history_price_decimals
    from sleeve_fund.research.holdout import HoldoutLocks
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.venues import venue as venue_profile

    req.validate()
    spec = spec_of(req.strategy)
    profile = venue_profile(req.venue)
    history = history or HistoryStore()
    try:
        prices = history.read(profile.name, req.pair, req.minutes)
    except KeyError:
        raise ValueError(f"no stored history for {req.pair} on this venue; ask for it under Stored history on "
                         "the Research page") from None
    if prices.empty:
        raise ValueError(f"the stored history for {req.pair} has no {_bars(req.minutes)} bars yet")
    fees = resolve_fees(profile.name, store)
    spread = spreads.resolve(profile.name, req.pair, store)
    base, quote = req.pair.split("/")
    # Price decimals from the stored prices, as the backtest page does: the default 2 rounded every
    # sub-$10 instrument to the cent (review round 9, B9-2).
    instrument = profile.instrument(base, quote, fees=fees.fees,
                                    price_precision=history_price_decimals(prices["close"]))
    exec_prices = None
    step = exec_step(prices.index[-1] - prices.index[0], req.minutes)
    if step is not None:
        exec_prices = history.read(profile.name, req.pair, step, start=prices.index[0] - pd.Timedelta(minutes=req.minutes))
    ledger = IdeaLedger(ledger_path or LEDGER)
    register = TrialsRegister(store) if store is not None else None
    dataset = dataset_name(profile.name, req.pair, req.minutes)
    result = run_study(
        spec, prices, instrument, dataset=dataset, ledger=ledger, holdout_days=req.holdout_days,
        train_days=req.train_days, test_days=req.test_days, use_holdout=req.use_holdout,
        exits=req.exits(),
        risk_profile=req.risk_profile, exec_prices=exec_prices, half_spread=spread.half_spread, progress=progress,
        register=register,
        locks=HoldoutLocks(store) if store is not None else None)
    # The fees the strategy's runs paid: a perpetual market's own when it trades one, not the venue's spot rates (RE-COST).
    paid = result.fee_basis
    fee_words = fees.text if paid is None or paid == fees.fees else \
        f"{float(paid.maker):.2%} maker, {float(paid.taker):.2%} taker (the perpetual market's)"
    result.fee_note = f"{fee_words}; the maker rate on post-only orders only; spread: {spread.text}"
    if result.breakeven:
        result.fee_note += f"; break-even: {result.breakeven}"
    cov = history.coverage(profile.name, req.pair)
    if cov is not None and pd.Timestamp.now(tz="UTC") - cov.last > STALE_HISTORY:
        result.notes.append(f"The stored history ends {cov.last:%d %b %Y %H:%M} UTC: the collector is still catching "
                            "up on this instrument, so the study ends there.")
    # Every run keeps its own sheet: a re-run with other exits, profile or windows is new evidence, not a
    # replacement for the old (review round 8, R8-M3). The newest sheet per instrument and bars decides G1.
    stem = f"{spec.name}_{dataset}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out = (out_dir or TEARSHEETS) / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while out.exists():  # two runs in the same second
        n += 1
        out = out.with_name(f"{stem}-{n}.md")
    out.write_text(render(result, ledger, register), encoding="utf-8")
    return out


def seed(ledger: Path | None = None, tearsheets: Path | None = None) -> None:
    """On the server, research the dashboard writes lives outside the code (IDEA_LEDGER, TEARSHEET_DIR),
    which each deploy replaces. Bring in the repository's record: every ledger line not there yet (the
    counter only ever grows), and every tear sheet that is missing or older there."""
    import shutil

    ledger, tearsheets = ledger or LEDGER, tearsheets or TEARSHEETS
    repo_ledger, repo_sheets = ROOT / "research" / "idea_ledger.jsonl", ROOT / "research" / "tearsheets"
    if ledger.resolve() != repo_ledger.resolve() and repo_ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        have = set(ledger.read_text(encoding="utf-8").splitlines()) if ledger.exists() else set()
        new = [line for line in repo_ledger.read_text(encoding="utf-8").splitlines() if line.strip() and line not in have]
        if new:
            with ledger.open("a", encoding="utf-8") as f:
                f.write("".join(line + "\n" for line in new))
    if tearsheets.resolve() != repo_sheets.resolve() and repo_sheets.is_dir():
        tearsheets.mkdir(parents=True, exist_ok=True)
        for src in repo_sheets.glob("*.md"):
            dst = tearsheets / src.name
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
