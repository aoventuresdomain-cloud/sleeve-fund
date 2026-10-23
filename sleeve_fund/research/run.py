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
from decimal import Decimal
from pathlib import Path

import pandas as pd

from sleeve_fund import markets
from sleeve_fund.instruments import pct

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
# The out-of-sample windows and the holdout (with the training bars that warm each window up) run on 1-minute bars for
# G1 (P1-D13): a few runs, not one per setting and fee, so their cap is higher. About six and a half years of minutes
# (a study's research period plus holdout reached 6 years on 22 Oct); past it they run on 5-minute bars, a one-way
# test (study.ONE_WAY_MINUTES). When they fit, every run of the study is matched on them (PM 22 Oct: 15-minute
# decisions, 1-minute execution); a 6-year 1-minute run took about 75 s and 0.5 GB (HoE, 22 Oct).
OOS_EXEC_BUDGET = 3_500_000
# Stored history ending longer ago than this is still being backfilled; a study says so.
STALE_HISTORY = pd.Timedelta(days=1)
BASKET_POOL = ("BTC", "ETH", "SOL", "XRP")  # STUDY-ASSET-LIST: the research pool an asset list draws from (PM 21 Oct)


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
    # A rule-builder study (the re-score workflow): the `rules` model's params, to_params(definition) plus its market,
    # and the definition's name for its tear sheet. None: the model's own defaults, as the Research page's form runs.
    params: dict | None = None
    label: str | None = None
    # B-37: a study list's pinned dates (research.pinned.Pinned): the study reads those days only, whatever day it runs.
    pinned: object | None = None
    # RESEARCH-SPOT-FEES (HoE, 21 Oct): the study's market, whatever its params or risk profile say: "spot-lowfee" for
    # the research spot market, "perp" for the perpetual. None: as today (the params' market, else the venue's).
    market: str | None = None
    # R-11: "walk_forward" (as before) or "continuous" (pinned settings only: a warm-up, then scored to the research end)
    score_window: str = "walk_forward"
    # R-12, this round only: also run the old walk-forward on the same settings (never counted as a try), and pass
    # only when both windows pass
    old_window_too: bool = False
    # R-24: a filtered variant's parent definition (a rule-builder definition), whose entries a placebo blocks at
    # random in the share the variant's filter blocked; None: no placebo
    placebo_parent: dict | None = None
    starting_capital: float = 10_000.0  # R-15: the account every run starts from (the PM's $1,000 run)
    # STUDY-ASSET-LIST (PM 21 Oct): the assets one definition runs on as one basket (research.basket_run), `pair`
    # their first; None: `pair` alone, as before
    pairs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.pairs is not None:
            object.__setattr__(self, "pairs", tuple(self.pairs))

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
        if self.old_window_too and self.score_window != "continuous":
            raise ValueError("the old walk-forward verdict is shown beside a continuous window's only")
        if self.placebo_parent is not None and self.score_window != "continuous":
            raise ValueError("the placebo runs on a continuous scoring window only")
        if not 0 < self.starting_capital < float("inf"):
            raise ValueError("the starting capital must be above 0")
        if self.market is not None and not isinstance(self.market, str):
            raise ValueError(f"a study's market is a name, not {self.market!r}")
        if self.pairs is not None:
            self._validate_pairs()

    def _validate_pairs(self) -> None:
        bases = [p.split("/")[0] if "/" in p else None for p in self.pairs]
        if None in bases:
            raise ValueError(f"every listed asset should look like BTC/USDT, not {self.pairs!r}")
        if not 2 <= len(self.pairs) <= len(BASKET_POOL):
            raise ValueError(f"an asset list holds 2 to {len(BASKET_POOL)} assets, not {len(self.pairs)}")
        if len(set(bases)) != len(bases):
            raise ValueError(f"an asset is listed twice: {', '.join(bases)}")
        if outside := [b for b in bases if b not in BASKET_POOL]:
            raise ValueError(f"{', '.join(outside)} is not in the research pool ({', '.join(BASKET_POOL)})")
        if len({p.split("/")[1] for p in self.pairs}) != 1:
            raise ValueError("the listed assets are quoted in different currencies")
        if self.pair != self.pairs[0]:
            raise ValueError(f"the request's pair {self.pair} is not its list's first asset {self.pairs[0]}")
        if self.minutes != 1440:
            raise ValueError("an asset list is studied on daily bars only")
        if self.placebo_parent is not None or self.old_window_too:
            raise ValueError("an asset list runs without the placebo or the old window beside it")


def study_market(req: StudyRequest) -> dict:
    """The params overlay for the study's market (RESEARCH-SPOT-FEES, DA's to wire): {} for None."""
    if req.market is None:
        return {}
    raise ValueError(f"a study's market setting ({req.market}) isn't wired yet")


def study_market(req: StudyRequest) -> dict:
    """RESEARCH-SPOT-FEES (HoE, 21 Oct): the params overlay for a study's own market setting, whatever its params or
    risk profile say: {} for None (as today), {"market": "perp"} or {"market": "spot-lowfee"}; anything else refused.
    The single and basket study paths both apply it."""
    if req.market is None:
        return {}
    if req.market not in (markets.PERP, markets.SPOT_LOWFEE):
        raise ValueError(f"a study's market is {markets.PERP} or {markets.SPOT_LOWFEE}, not {req.market!r}")
    return {"market": req.market}


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


def execution_bars(req: StudyRequest, prices: pd.DataFrame, profile, history):
    """The bars a store study matches its resting orders on: (exec_prices, oos_exec, oos_step). A pinned list's study
    deciding on intraday bars gets 1-minute bars for its whole window or is refused (rescore's dry run calls this too,
    to prove it before a run)."""
    # The execution bars are sized on the days the study reads: a list's pinned window, not the whole store, which
    # can reach back years before a list's data start (the 22 Oct 23:52 BTC run was sized on the whole store, went
    # over the budget and fell back to 5-minute bars).
    if req.pinned is not None:
        from sleeve_fund.research import pinned as pinning

        window = pinning.cut(prices, req.pinned)
    else:
        window = prices
    first, last = window.index[0], window.index[-1]
    start = first - pd.Timedelta(minutes=req.minutes)
    exec_prices = None
    step = exec_step(last - first, req.minutes)
    if step is not None:
        exec_prices = history.read(profile.name, req.pair, step, start=start, end=last)
    oos_exec = None
    for oos_step in (1, 5):
        if oos_step < req.minutes and (last - start) / pd.Timedelta(minutes=oos_step) <= OOS_EXEC_BUDGET:
            oos_exec = exec_prices if step == oos_step else history.read(profile.name, req.pair, oos_step,
                                                                          start=start, end=last)
            break
    if req.pinned is not None and 1 < req.minutes < 1440:
        # A list's intraday study runs on 1-minute bars or not at all (PM 23 Oct): no quiet fall back to coarser
        # fills. A daily row still falls back, and its sheet says on which bars.
        if oos_exec is None or oos_step != 1:
            raise ValueError(f"the pinned window's {_bars(1)} bars don't fit the study's budget: no 1-minute run")
        if not len(oos_exec) or oos_exec.index[0] > start + pd.Timedelta(days=1):
            raise ValueError(f"the stored {_bars(1)} bars don't cover the pinned window from {first:%d %b %Y}: "
                             "no 1-minute run")
    return exec_prices, oos_exec, oos_step


def run_store_study(req: StudyRequest, store=None, progress=None, ledger_path: Path | None = None,
                    out_dir: Path | None = None, history=None, notes: tuple[str, ...] = ()) -> Path:
    """Run the study and write its tear sheet; returns the tear sheet's path. store: the journal
    (for the connected account's fees and the measured spread), if there is one. notes: caveats the sheet adds."""
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
    if req.market is not None:  # the study's own market over the params' (a short side is refused on spot)
        from dataclasses import replace as _replace

        req = _replace(req, params={**(req.params or {}), **study_market(req)})
    if req.pairs is not None:  # STUDY-ASSET-LIST: one basket of the listed assets
        from sleeve_fund.research.basket_run import run_store_basket

        return run_store_basket(req, store, out_dir=out_dir, history=history)
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
    if profile.fee_bounds:  # FEE-LIVE: never below the published VIP 0, with the account's own rate said beside it
        from dataclasses import replace

        from sleeve_fund import venue_terms

        floor = venue_terms.research_fees(store, profile, req.pair)
        fees = replace(fees, fees=floor.fees, words=floor.text)
    spread = spreads.series(profile.name, req.pair, store)  # each run charges the one in force at each time
    base, quote = req.pair.split("/")
    # Price decimals from the stored prices, as the backtest page does: the default 2 rounded every
    # sub-$10 instrument to the cent (review round 9, B9-2).
    instrument = profile.instrument(base, quote, fees=fees.fees,
                                    price_precision=history_price_decimals(prices["close"]))
    exec_prices, oos_exec, oos_step = execution_bars(req, prices, profile, history)
    if oos_exec is not None and len(oos_exec) and oos_step == 1:
        exec_prices = oos_exec

    def minutes_between(lo, hi):
        return history.read(profile.name, req.pair, 1, start=lo, end=hi)

    ledger = IdeaLedger(ledger_path or LEDGER)
    register = TrialsRegister(store) if store is not None else None
    dataset = dataset_name(profile.name, req.pair, req.minutes)
    def study(window: str, ledger, register, progress):
        return run_study(
            spec, prices, instrument, dataset=dataset, ledger=ledger, default_params=req.params,
            holdout_days=req.holdout_days,
            train_days=req.train_days, test_days=req.test_days, use_holdout=req.use_holdout,
            exits=req.exits(),
            risk_profile=req.risk_profile, exec_prices=exec_prices, half_spread=spread, progress=progress,
            oos_exec_prices=oos_exec, minute_loader=minutes_between, register=register,
            locks=HoldoutLocks(store) if store is not None else None, pinned=req.pinned, score_window=window,
            placebo_parent=req.placebo_parent if window == req.score_window else None,
            starting_capital=req.starting_capital)

    result = study(req.score_window, ledger, register, progress)
    old = None
    if req.old_window_too:  # R-12: the same settings on the old window, its runs counted nowhere (no new try)
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            old = study("walk_forward", IdeaLedger(Path(scratch) / "old_window.jsonl"), None, None)
    # The fees the strategy's runs paid: a perpetual market's own when it trades one, not the venue's spot rates (RE-COST).
    paid = result.fee_basis
    fee_words = fees.text if paid is None or paid == fees.fees else \
        f"{pct(paid.maker)} maker, {pct(paid.taker)} taker ({markets.SPOT_LOWFEE_WORDS})" \
        if markets.market_of(req.params) == markets.SPOT_LOWFEE else \
        f"{pct(paid.maker)} maker, {pct(paid.taker)} taker (the perpetual market's)"
    span = int((prices.index[0] - pd.Timedelta(minutes=req.minutes)).value), int(prices.index[-1].value)
    result.fee_note = f"{fee_words}; the maker rate on post-only orders only; spread: {spread.text(*span)}"
    if result.breakeven:
        result.fee_note += f"; break-even: {result.breakeven}"
    cov = history.coverage(profile.name, req.pair)
    if cov is not None and pd.Timestamp.now(tz="UTC") - cov.last > STALE_HISTORY:
        result.notes.append(f"The stored history ends {cov.last:%d %b %Y %H:%M} UTC: the collector is still catching "
                            "up on this instrument, so the study ends there.")
    result.notes.extend(notes)
    # Every run keeps its own sheet: a re-run with other exits, profile or windows is new evidence, not a
    # replacement for the old (review round 8, R8-M3). The newest sheet per instrument and bars decides G1.
    stem = f"{req.label or spec.name}_{dataset}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out = (out_dir or TEARSHEETS) / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while out.exists():  # two runs in the same second
        n += 1
        out = out.with_name(f"{stem}-{n}.md")
    out.write_text(render(result, ledger, register, old=old, files=True), encoding="utf-8")
    write_results(result, out)
    write_returns(result, out)
    write_oos_returns(result, out)
    write_beside(result, out)
    return out


BESIDE = ("trades.csv", "equity.csv", "equity.svg")  # R-15: what sits beside a sheet as <stem>-<kind>, all it may


def write_beside(result, sheet: Path) -> None:
    """R-15: the sheet's every trade, its daily equity and buy-and-hold's, and their curve, beside it, never trimmed."""
    from sleeve_fund.research import periods

    eq, hold = periods.scored(result)
    files = {"trades.csv": periods.trades_csv(periods.trade_rows(result.full_period)),
             "equity.csv": periods.equity_csv(eq, hold),
             "equity.svg": periods.equity_svg(eq, hold, f"{sheet.stem}: equity, Strategy and buy and hold")}
    for kind in BESIDE:
        sheet.with_name(f"{sheet.stem}-{kind}").write_text(files[kind], encoding="utf-8")


def results_path(sheet: Path) -> Path:
    """B-22: the stored figures beside a tear sheet, for the page to read as fields rather than words."""
    return sheet.with_suffix(".results.json")


def returns_path(sheet: Path) -> Path:
    """R-20: the daily returns a tear sheet's "Risk, trades and costs" section is computed on, beside it."""
    return sheet.with_suffix(".returns.csv")


def write_returns(result, sheet: Path) -> Path:
    """R-20 (HoE, 21 Oct): the strategy's daily returns over the sheet's window (the full research period until R-11's
    scored window), as CSV (date, return), written in the same step as the sheet, for the fund summary's correlation
    matrix. Never a holdout day: a series running past the research period is refused, not written."""
    from sleeve_fund.research.metrics import daily_returns

    ret = daily_returns(result.full_period.equity)
    if len(ret) and ret.index[-1] > result.research_end:
        raise ValueError(f"the returns run to {ret.index[-1]}, past the research period's end {result.research_end}: "
                         "a holdout day would be in them")
    path = returns_path(sheet)
    path.write_text("date,return\n" + "".join(f"{t:%Y-%m-%d},{v!r}\n" for t, v in ret.items()), encoding="utf-8")
    return path


def oos_returns_path(sheet: Path) -> Path:
    """(d) FINALIST-CORR: the out-of-sample daily returns and the asset's price return on those days, beside the sheet."""
    return sheet.with_suffix(".oos-returns.csv")


def write_oos_returns(result, sheet: Path) -> Path:
    """(d) FINALIST-CORR (HoE, 21 Oct): the strategy's out-of-sample daily returns (the walk-forward test windows,
    net of costs, a day out of the market a 0) and the asset's plain price return each of those days, as CSV
    (date, return, price), for the fund summary's finalist correlation. Never a holdout day, as write_returns."""
    from sleeve_fund.research.metrics import daily_returns

    ret = result.oos_returns
    if len(ret) and ret.index[-1] > result.research_end:
        raise ValueError(f"the out-of-sample returns run to {ret.index[-1]}, past the research period's end "
                         f"{result.research_end}: a holdout day would be in them")
    price = daily_returns(result.research_closes).reindex(ret.index)
    path = oos_returns_path(sheet)
    path.write_text("date,return,price\n" + "".join(f"{t:%Y-%m-%d},{v!r},{p!r}\n"
                                                    for t, v, p in zip(ret.index, ret, price)), encoding="utf-8")
    return path


def write_results(result, sheet: Path) -> Path:
    """B-22 (Advisor B22-DOUBLE-FEES): the study's stored results beside its tear sheet (results_path), as JSON:
    {"double_fees": its fields (study._double_fees) or null}. Fees as decimal strings, times as ISO 8601."""
    import json

    d = result.double_fees

    def plain(v):
        return str(v) if isinstance(v, Decimal) else v.isoformat() if isinstance(v, pd.Timestamp) else v

    fields = None if d is None else {k: plain(v) for k, v in d.items()}
    path = results_path(sheet)
    path.write_text(json.dumps({"double_fees": fields}, indent=1, sort_keys=True), encoding="utf-8")
    return path


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
