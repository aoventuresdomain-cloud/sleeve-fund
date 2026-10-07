"""GAP-LIQ-CAP: strict xfails written before the build (Head of QA, 7 Oct; Advisor 23:42 and 23:44). Owner: PE1,
stacked on PE2's stop-safety PR.

The ruling (advisor-rulings.md, "23:42 6 Oct"): on isolated margin a liquidation books a loss of exactly
X = posted margin + entry fee + liquidation fee (as in RAL), gapped or not. The excess past the bankruptcy price is a
diagnostic only, "covered by the venue's insurance fund" with its amount, never P&L. Same rule in backtest and paper;
RAL's Y unchanged. Any liquidation in a strategy's out-of-sample windows or holdout is a G1 finding shown to the PM,
whatever the P&L. Register perp backtests that contain a liquidation are re-run, and both figures are recorded.

The Advisor's answers to QA's edges (advisor-rulings.md, "GAP-LIQ-CAP / SPREAD-PIT edges", 7 Oct ~00:19-00:22):
(1) the liquidation fill is booked at the bankruptcy price, the liquidation fee its own fee line, totalling X; insurance
money never enters any P&L line; the gapped market price is kept as a diagnostic field on the trade. (2) A liquidation
short of bankruptcy also books exactly X, at the bankruptcy price. (3) Liquidation fee = qty x liquidation (trigger)
price x rate. (4) Random-entry benchmark draws use the same engine and cap. (5) A liquidation in out-of-sample or the
holdout: caused by a gap past a correctly placed stop, it is shown and a G1 pass needs the PM's explicit
acknowledgement; with the stop missing or beyond the half-distance-to-liquidation rule, it is a G1 FAIL. (6) A register
re-run replaces the row, which is marked superseded; N is unchanged.

Marks:
- LIQ_CAP: `xfail(strict=True, raises=AssertionError, reason="GAP-LIQ-CAP ...")`, not built on any head tested.
- LIQ_CAP_UNLESS_D3: the same mark, but only on a head WITHOUT #155's P1-D3 margin cap (`markets.gap_loss_cap`). On
  main (no #155) these are xfails; on the stop-safety stack (#155 underneath) they are plain tests that pass and must
  keep passing. They pin the money side of the ruling, which #155 already gives in backtest and paper.
- Tests named `test_control_...` carry no mark: today's behaviour is already right there.

Every test checks its set-up first, on the market path and the journal, never on the outcome under test, so a broken
set-up fails as a plain error in the set-up check (an AssertionError with "set-up" in it), and an xfail can only come
from the final assertion. A missing interface fails as AssertionError("not built: ..."), never as an ImportError,
AttributeError or TypeError.

Run (from a checkout, this file copied into tests/):
  BACKTEST_ISOLATE=0 PYTHONPATH=<wt>:<wt>/tests python -B -m pytest -q -p no:cacheprovider
    tests/test_gap_liq_cap_xfails.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REASON = "GAP-LIQ-CAP: not built yet (Advisor 23:42)"
LIQ_CAP = pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON)


@pytest.fixture(autouse=True)
def _funding_judged(monkeypatch):
    """Set-up only (HoQA 7 Oct, with #163; DA option A; and #178 below): the harness's studies run a SIMULATED perp,
    whose funding #163 leaves NOT JUDGED (Advisor 6 Oct), which keeps the holdout closed and puts only the unjudged
    names in g1_verdict's failed list. These cells judge liquidations alone, so funding is lifted to PASS here; its
    own pins are in the O17 masters. A no-op on a head without study._funding_check."""
    try:
        from sleeve_fund.research import study
    except Exception:  # noqa: BLE001
        return
    if hasattr(study, "_funding_check"):
        monkeypatch.setattr(study, "_funding_check",
                            lambda r: ("PASS", "funding lifted by the GAP-LIQ-CAP harness: liquidations judged alone"))
    # Same for #178's P1-D13 rule (HoQA 7 Oct, CR): these studies run daily bars with a resting exit (the liquidation
    # price), which D13 leaves NOT JUDGED for want of 1-minute execution data. Only that reason is lifted; every other
    # not-judged reason (errors, a blind guard, a one-way 5-minute result) still stands.
    no_1m = getattr(study, "NO_1M", None)
    prop = getattr(getattr(study, "StudyResult", None), "not_judged", None)
    if no_1m and isinstance(prop, property):
        def _not_judged(self, _get=prop.fget):
            why = _get(self)
            return "" if why.startswith(no_1m) else why

        monkeypatch.setattr(study.StudyResult, "not_judged", property(_not_judged))


def _has_d3_cap() -> bool:
    try:
        from sleeve_fund import markets

        return hasattr(markets, "gap_loss_cap")  # #155's P1-D3 cap (3be572a): a gap loses the margin and no more
    except Exception:  # noqa: BLE001
        return False


D3 = _has_d3_cap()
LIQ_CAP_UNLESS_D3 = pytest.mark.xfail(
    not D3, strict=True, raises=AssertionError,
    reason=REASON + "; the money side is #155's P1-D3 cap, absent on this head (plain on the stop-safety stack)")

START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC (tests/test_replay.py START, the recorder's clock)
S = 1_000_000_000
M = 60 * S
BASE = 60_000.0
STOP = 0.10  # a real stop, within half the distance to liquidation at 2x and 3x (the stop-safety gate needs one)
MARGIN_CAP = 0.10  # margin cap 10% of equity: open risk stays inside the interim 5%-of-book limit at a 10% stop
NAME = "qa-gap-cap"
INSURANCE_WORDS = "covered by the venue's insurance fund"

# (side, risk profile, leverage): the two cases of the hub-146 "4bf4f97 delta" Advisor item 1
CASES = {"3x-short": (-1, "aggressive", 3.0), "2x-long": (1, "balanced", 2.0)}
CASE_IDS = list(CASES)
MODES = ["backtest", "paper"]


# --------------------------------------------------------------------------------------------- the probe strategy


def _probe_classes():
    """Holds `side` for the 1-minute bars closing in [enter, leave) minutes after START, flat otherwise (as
    tests/test_hub_146_qa.py's probe), so every order is placed by design and only the stop, the liquidation or the
    venue can close it."""
    from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy

    class QAGapProbeConfig(LongFlatConfig):
        def __init__(self, *, enter: int = 5, leave: int = 10**6, side: int = 1, **kwargs) -> None:
            super().__init__(**kwargs)
            self.enter, self.leave, self.side = enter, leave, side

    class QAGapProbe(LongFlatStrategy):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.enter, self.leave, self.side = config.enter, config.leave, config.side

        def _k(self, bar) -> int:
            return int((bar.ts_event - START) // M)

        def want_long(self, bar):
            return self.enter <= self._k(bar) < self.leave

        def want_side(self, bar):
            return self.side if self.enter <= self._k(bar) < self.leave else 0

    return QAGapProbe, QAGapProbeConfig


@contextlib.contextmanager
def _patched():
    """The probe registered and every risk profile's margin cap at 10%, put back afterwards."""
    from sleeve_fund import risk
    from sleeve_fund.strategies import REGISTRY

    saved_profiles, saved_probe = dict(risk.PROFILES), REGISTRY.get("qa_gap_probe")
    try:
        for nm, p in list(risk.PROFILES.items()):
            risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=MARGIN_CAP)
        REGISTRY["qa_gap_probe"] = _probe_classes()
        yield
    finally:
        risk.PROFILES.clear()
        risk.PROFILES.update(saved_profiles)
        if saved_probe is None:
            REGISTRY.pop("qa_gap_probe", None)
        else:
            REGISTRY["qa_gap_probe"] = saved_probe


def _params(side: int) -> dict:
    return {"enter": 5, "leave": 10**6, "side": side, "stop_loss": STOP, "market": "perp", "allow_short": True}


# ---------------------------------------------------------------------------------------------------- the runs


def _minute_bars(gap_factor: float, gap_minute: int = 20, n: int = 40) -> pd.DataFrame:
    """1-minute bars by close time: flat at BASE, then from `gap_minute` flat at BASE x gap_factor. Each bar opens at
    its own price, so the bar after the jump OPENS there: a gap no resting order can trade inside."""
    px = np.where(np.arange(n) < gap_minute, BASE, BASE * gap_factor)
    idx = pd.to_datetime(START + (np.arange(n) + 1) * M, unit="ns", utc=True)
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1e6}, index=idx)


def _recording(path: Path, side: int, profile: str, gap_factor: float) -> None:
    """A recorded paper session (tests/test_long_short.py _record): one trade and quote a second, flat at BASE for 20
    minutes, one trade at BASE x gap_factor (the gap), then flat there for 20 minutes."""
    import test_long_short as tls

    meta = {"balances": ["10000.00 USD"],
            "sleeve": {"name": NAME, "strategy": "qa_gap_probe", "instrument": "BTC/USD",
                       "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000, "risk_profile": profile,
                       "params": _params(side), "maker_fee": "0.0002", "taker_fee": "0.0005", "tick_seconds": 30}}
    tls._record(path, meta, [(20, 0.0), (0, gap_factor - 1.0), (20, 0.0)], px=BASE)


def _collect(st, name: str, *, mode: str, side: int, profile: str, lev: float, gap_px: float, gap_ts,
             curve=None, extra=None, flows=None, trip_rows=None) -> dict:
    """What every check reads, from a store or an in-memory journal (the same API). flows: a backtest's own
    (funding, insurance) lists and its fills report rows, as the backtest page hands them to the trade list
    (dashboard/preview.py: trades(fills_to_rows(res.fills), res.shorts, res.funding, res.insurance))."""
    from sleeve_fund.research.metrics import trades

    orders = sorted(st.orders(name, limit=100_000), key=lambda o: (o["ts"], o.get("id") or 0))
    fills = sorted(st.fills(name, limit=100_000), key=lambda f: (f["ts"], f.get("id") or 0))
    entry = next((o for o in orders if o["intent"] == "entry"), None)
    opened = [f for f in fills if entry is not None and f["order_id"] == entry["order_id"]]
    closing_ids = [o["order_id"] for o in orders if entry is not None and o["order_id"] != entry["order_id"]]
    closed = [f for f in fills if f["order_id"] in closing_ids]
    filled_ids = {f["order_id"] for f in closed}
    closing_ids = [i for i in closing_ids if i in filled_ids]  # orders that closed it: a resting stop never hit is not
    q_open = sum(f["qty"] for f in opened)
    entry_px = sum(f["qty"] * f["price"] for f in opened) / q_open if q_open else float("nan")
    margin = q_open * entry_px / lev if q_open else float("nan")
    entry_fees = sum(f["fee"] for f in opened)
    close_fees = sum(f["fee"] for f in closed)
    rate = entry_fees / (q_open * entry_px) if q_open else float("nan")  # the run's taker rate, read off the entry
    x = margin + entry_fees + close_fees  # RAL's X (Advisor 18:17 point 4, 21:05): margin + entry fee + liquidation fee
    bk = entry_px * (1 - side / lev)  # bankruptcy: where the price loss equals the posted margin
    q_close = sum(f["qty"] for f in closed)
    close_px = sum(f["qty"] * f["price"] for f in closed) / q_close if q_close else float("nan")
    funding = float(st.funding_total(name))
    if flows is not None:
        funding_rows, insurance = list(flows[0] or []), list(flows[1] or [])
    else:
        funding_rows, insurance = list(st.funding(name)), list(st.insurance(name))
    rows = trip_rows if trip_rows is not None else [dict(f) for f in fills]
    trips = trades(rows, True, funding_rows, insurance)
    marks = [m["equity"] for m in st.equity_series(name, limit=100_000)]
    events = [(e["kind"], e["message"]) for e in st.events(name, limit=100_000)]
    sleeve = st.sleeve(name)
    closing_orders = [o for o in orders if o["order_id"] in closing_ids]
    return {
        "mode": mode, "side": side, "profile": profile, "lev": lev, "orders": orders, "fills": fills,
        "entry": entry, "opened": opened, "closed": closed, "closing_intents": [o["intent"] for o in orders
                                                                              if o["order_id"] in closing_ids],
        "q": q_open, "entry_px": entry_px, "margin": margin, "x": x, "bk": bk, "close_px": close_px,
        "entry_fees": entry_fees, "close_fees": close_fees, "rate": rate, "closing_orders": closing_orders,
        "liq": ((entry or {}).get("signal") or {}).get("liquidation_px"), "gap_px": gap_px, "gap_ts": gap_ts,
        "funding": funding, "insurance": insurance, "trips": trips, "marks": marks, "events": events,
        "status": sleeve.status, "reason": sleeve.status_reason or "",
        "cash_after": float(st.journal_book(name, 10_000.0)["cash"]), "curve": curve, **(extra or {}),
    }


@functools.lru_cache(maxsize=None)
def _backtest(case: str, gap_factor: float) -> dict:
    from sleeve_fund.research.metrics import fills_to_rows
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    side, profile, lev = CASES[case]
    with _patched():
        inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
        res = run_backtest("qa_gap_probe", _minute_bars(gap_factor), inst, params=_params(side),
                           starting_capital=10_000, risk_profile=profile, bar_minutes=1, half_spread=0.0)
        j = res.journal
        name = j.sleeve_row.name
        return _collect(j, name, mode="backtest", side=side, profile=profile, lev=lev, gap_px=BASE * gap_factor,
                        gap_ts=pd.Timestamp(START + 20 * M, unit="ns", tz="UTC"), curve=res.equity,
                        extra={"handler_errors": list(res.handler_errors)}, flows=(res.funding, res.insurance),
                        trip_rows=fills_to_rows(res.fills))


@functools.lru_cache(maxsize=None)
def _paper(case: str, gap_factor: float) -> dict:
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store

    side, profile, lev = CASES[case]
    with _patched(), tempfile.TemporaryDirectory(prefix="qa-gapcap-") as d:
        path = Path(d) / "gap.jsonl.gz"
        _recording(path, side, profile, gap_factor)
        store = Store.in_memory()
        replay(path, store=store)
        return _collect(store, NAME, mode="paper", side=side, profile=profile, lev=lev, gap_px=BASE * gap_factor,
                        gap_ts=pd.Timestamp(START + 20 * M, unit="ns", tz="UTC"))


def _run(mode: str, case: str, gap_factor: float) -> dict:
    return (_backtest if mode == "backtest" else _paper)(case, gap_factor)


def _gap_through(case: str) -> float:
    """60% against the position: past the stop, the liquidation price and the bankruptcy price."""
    side = CASES[case][0]
    return 1 - side * 0.60


def _gap_short_of_bankruptcy(case: str) -> float:
    """Past the liquidation price but short of the bankruptcy price (inside the maintenance buffer): 3x short to
    79,800 (liq ~79,602, bankruptcy 80,000); 2x long to 30,075 (liq ~30,150.8, bankruptcy 30,000)."""
    return {"3x-short": 1.33, "2x-long": 0.50125}[case]


def _gap_short_of_liquidation(case: str) -> float:
    """Past the 10% stop, well short of the liquidation price: 20% against."""
    side = CASES[case][0]
    return 1 - side * 0.20


# ------------------------------------------------------------------------------------------------ set-up checks


def _setup_common(d: dict) -> None:
    assert not d.get("handler_errors"), ("set-up: the strategy raised", d.get("handler_errors"))
    assert d["entry"] is not None and d["opened"] and d["fills"][0]["order_id"] == d["entry"]["order_id"], (
        "set-up: the position is the run's first fill", d["fills"][:2])
    assert abs(d["entry_px"] / BASE - 1) < 1e-4, ("set-up: entered at about 60,000", d["entry_px"])
    assert d["closed"], ("set-up: something closed the position", d["closing_intents"])
    assert abs(sum(f["qty"] for f in d["closed"]) - d["q"]) < 1e-8, ("set-up: closed in full", d["closed"])
    assert min(pd.Timestamp(f["ts"]) for f in d["closed"]) >= d["gap_ts"], ("set-up: still held at the gap",
                                                                             d["closed"])
    assert d["funding"] == 0, ("set-up: no funding settled while it was held (00:05 to 00:20)", d["funding"])
    assert abs(d["margin"] - MARGIN_CAP * 10_000) < 1.0, ("set-up: margin about 10% of 10,000", d["margin"])


def _setup_gap_past_bankruptcy(d: dict) -> None:
    _setup_common(d)
    s, liq = d["side"], d["liq"]
    stop_px = d["entry_px"] * (1 - s * STOP)
    assert liq is not None and STOP <= 0.5 * abs(1 - liq / d["entry_px"]), ("set-up: stop within half way to liq",
                                                                              liq)
    assert s * (d["gap_px"] - stop_px) < 0 and s * (d["gap_px"] - liq) < 0 and s * (d["gap_px"] - d["bk"]) < 0, (
        "set-up: the market gapped past the stop, the liquidation price and the bankruptcy price",
        d["gap_px"], stop_px, liq, d["bk"])


def _setup_gap_short_of_bankruptcy(d: dict) -> None:
    _setup_common(d)
    s, liq = d["side"], d["liq"]
    assert liq is not None and s * (d["gap_px"] - liq) < 0 < s * (d["gap_px"] - d["bk"]), (
        "set-up: the market gapped past the liquidation price but not the bankruptcy price", d["gap_px"], liq, d["bk"])


def _setup_gap_short_of_liquidation(d: dict) -> None:
    _setup_common(d)
    s, liq = d["side"], d["liq"]
    stop_px = d["entry_px"] * (1 - s * STOP)
    assert liq is not None and s * (d["gap_px"] - stop_px) < 0 < s * (d["gap_px"] - liq), (
        "set-up: the market gapped past the stop but not the liquidation price", d["gap_px"], stop_px, liq)


def _money(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\.\d{2}\b", text)]


def _excess(d: dict) -> float:
    """What the market took past the bankruptcy price: |gap price - bankruptcy price| x quantity."""
    return d["side"] * (d["bk"] - d["gap_px"]) * d["q"]


def _loss(d: dict) -> float:
    return 10_000.0 - d["cash_after"] + d["funding"]


# ===================================================================================== the gapped liquidation


@LIQ_CAP_UNLESS_D3
@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_a_gapped_liquidation_books_a_loss_of_exactly_x(mode, case):
    """Gapped 60% past the bankruptcy price: booked as a liquidation, and the journal's cash after it is the cash
    before the entry less exactly X (to the cent): posted margin + entry fee + liquidation fee."""
    d = _run(mode, case, _gap_through(case))
    _setup_gap_past_bankruptcy(d)
    got = {"booked_as": d["closing_intents"], "loss": round(_loss(d), 2)}
    want = {"booked_as": ["liquidation"] * len(d["closing_intents"]), "loss": round(d["x"], 2)}
    assert got["booked_as"] == want["booked_as"] and abs(_loss(d) - d["x"]) <= 0.011, (got, want)


@LIQ_CAP_UNLESS_D3
@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_equity_never_goes_below_what_x_allows_and_returns_and_drawdown_carry_x_only(mode, case):
    """Every equity mark (paper: the journal's marks; backtest: the curve and the journal's marks) stays at or above
    10,000 - X, and ends there. The drawdown read from the curve and the trip's return are those of a loss of X:
    the excess past bankruptcy is in neither."""
    from sleeve_fund.research.metrics import max_drawdown

    d = _run(mode, case, _gap_through(case))
    _setup_gap_past_bankruptcy(d)
    floor = 10_000.0 - d["x"]
    series = list(d["marks"]) + (list(d["curve"].values) if d["curve"] is not None else [])
    trip = d["trips"][-1]
    got = {"lowest": round(min(series), 2), "last": round(series[-1] if d["curve"] is None else
                                                        float(d["curve"].iloc[-1]), 2),
           "trip_ret": round(trip["ret"], 6)}
    want = {"lowest": round(floor, 2), "last": round(floor, 2), "trip_ret": round(-d["x"] / trip["cost"], 6)}
    ok = min(series) >= floor - 0.011 and abs(got["last"] - floor) <= 0.011 and abs(
        trip["ret"] + d["x"] / trip["cost"]) <= 0.011 / trip["cost"]
    if d["curve"] is not None:
        peak = float(d["curve"].cummax().max())
        got["max_drawdown"], want["max_drawdown"] = round(max_drawdown(d["curve"]), 6), round(floor / peak - 1, 6)
        ok = ok and abs(max_drawdown(d["curve"]) - (floor / peak - 1)) <= 0.011 / peak
    assert ok, (got, want)


@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_the_liquidation_books_at_the_bankruptcy_price_and_the_excess_is_never_pnl(mode, case):
    """Advisor 00:19 (1): the journal books the liquidation fill at the bankruptcy price, never past it, so the trip's
    own price loss is the posted margin to the cent, and its P&L is -X with no insurance-fund credit netted into it
    ("insurance money never enters any P&L line"; the stop-safety stack's netted +800 / +200 credit is wrong)."""
    d = _run(mode, case, _gap_through(case))
    _setup_gap_past_bankruptcy(d)
    trip = d["trips"][-1]
    price_loss = d["side"] * (d["entry_px"] - d["close_px"]) * d["q"]
    got = {"booked_px": round(d["close_px"], 2), "price_loss": round(price_loss, 2), "trip_pnl": round(trip["pnl"], 2),
           "insurance_in_pnl": round(trip.get("insurance", 0.0), 2)}
    want = {"booked_px": round(d["bk"], 2), "price_loss": round(d["margin"], 2), "trip_pnl": round(-d["x"], 2),
            "insurance_in_pnl": 0.0}
    assert (abs(price_loss - d["margin"]) <= 0.011 and abs(trip["pnl"] + d["x"]) <= 0.011
            and trip.get("insurance", 0.0) == 0.0), (got, want)


@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_the_excess_is_a_diagnostic_covered_by_the_venues_insurance_fund_with_its_amount(mode, case):
    """One journal event says the excess was "covered by the venue's insurance fund" and names its amount,
    |gap price - bankruptcy price| x quantity (3x short: 16,000 x 0.05 = 800.00; 2x long: 6,000 x 0.0333 = 200.00),
    to the cent (0.1% for the fill's spread)."""
    d = _run(mode, case, _gap_through(case))
    _setup_gap_past_bankruptcy(d)
    excess = _excess(d)
    tol = max(0.011, 1e-3 * excess)
    said = [m for _, m in d["events"] if INSURANCE_WORDS in m.lower()]
    named = [m for m in said if any(abs(v - excess) <= tol for v in _money(m))]
    assert named, {"want": f"an event saying {INSURANCE_WORDS!r} with {excess:,.2f}",
                   "insurance_events": [m for k, m in d["events"] if "insurance" in (k + m).lower()][:4]}


@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_the_liquidation_fee_is_its_own_line_qty_x_trigger_price_x_rate(mode, case):
    """Advisor 00:19 (1) and (3): the liquidation fee is its own fee line on the liquidation fill, and it is
    qty x the liquidation (trigger) price x the rate (the run's taker rate, read off the entry fill), to the cent:
    never charged on the gapped market price, nor on the bankruptcy price the fill is booked at. With the fill at the
    bankruptcy price, margin + entry fee + this fee is X. (Today's fee is the rate on the price the fill was booked
    at, the gapped one. Gapped short of bankruptcy, the market's price sits within 0.5% of the trigger, a fee
    difference under a cent, so only the gap past bankruptcy can tell them apart.)"""
    d = _run(mode, case, _gap_through(case))
    _setup_gap_past_bankruptcy(d)
    fee = d["q"] * d["liq"] * d["rate"]
    got = {"liq_fee": round(d["close_fees"], 2), "booked_px": round(d["close_px"], 2),
           "fill_fee_lines": [round(f["fee"], 2) for f in d["closed"]]}
    want = {"liq_fee": round(fee, 2), "booked_px": round(d["bk"], 2), "rate": d["rate"], "trigger_px": d["liq"]}
    assert abs(d["close_fees"] - fee) <= 0.011 and all(f["fee"] > 0 for f in d["closed"]), (got, want)


def _market_px(d: dict) -> list:
    """Every `market_px` (ASSUMED name: markets.LiquidationBooking.market_px, PE1's 1da0dd5) the trade carries: on
    the liquidation's order signal, on its fill rows, or on the trip row."""
    out = [(o.get("signal") or {}).get("market_px") for o in d["closing_orders"]]
    out += [f.get("market_px") for f in d["closed"]]
    out += [d["trips"][-1].get("market_px")] if d["trips"] else []
    return [float(v) for v in out if v is not None]


@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_the_gapped_market_price_is_kept_as_a_diagnostic_on_the_trade(mode, case):
    """Advisor 00:19 (1): the fill is booked at the bankruptcy price, and the price the market actually gave (the
    gap, 96,000 for the 3x short, 24,000 for the 2x long) stays on the trade as a diagnostic field, `market_px`
    (assumed name), within 0.1% (paper's quote spread)."""
    d = _run(mode, case, _gap_through(case))
    _setup_gap_past_bankruptcy(d)
    seen = _market_px(d)
    got = {"market_px": seen, "booked_px": round(d["close_px"], 2)}
    want = {"market_px": f"about {d['gap_px']:,.2f}", "booked_px": round(d["bk"], 2)}
    assert (any(abs(v / d["gap_px"] - 1) <= 1e-3 for v in seen)
            and abs(d["close_px"] - d["bk"]) <= 0.011), (got, want)


@LIQ_CAP_UNLESS_D3
@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_ral_y_is_unchanged(mode, case):
    """RAL's text stays "Position margin lost (liquidated): X, Y% of strategy equity at entry" (Advisor 17:57, 20:37),
    X now equal to the loss booked (to the cent) and Y = X over the equity at the first entry fill (10,000 here: the
    position is the run's first fill), to the printed rounding."""
    d = _run(mode, case, _gap_through(case))
    _setup_gap_past_bankruptcy(d)
    texts = [d["reason"]] + [m for k, m in d["events"] if k == "risk_halt"]
    m = next((re.search(r"Position margin lost \(liquidated\): ([\d,]+\.\d\d), (\d+(?:\.\d+)?)% of strategy equity "
                        r"at entry", t) for t in texts if "Position margin lost" in t), None)
    x_text = float(m.group(1).replace(",", "")) if m else None
    y_text = m.group(2) if m else None
    y_ok = y_text is not None and abs(float(y_text) - 100 * x_text / 10_000.0) <= 0.5 * 10 ** -len(
        y_text.partition(".")[2]) + 1e-9
    got = {"x_text": x_text, "x_matches_loss": x_text is not None and abs(x_text - _loss(d)) <= 0.011, "y_ok": y_ok}
    want = {"x_text": round(d["x"], 2), "x_matches_loss": True, "y_ok": True}
    assert got["x_matches_loss"] and got["y_ok"] and abs((x_text or 0) - d["x"]) <= 0.011, (got, want, texts[:2])


@pytest.mark.parametrize("case", CASE_IDS)
def test_gap_liq_cap_backtest_equals_paper_for_the_same_gap(case):
    """The same minutes, the same entry minute and the same 60% gap: both book a liquidation of exactly their own X
    at their own bankruptcy price, and the two X agree within 0.10 (the entries differ by the paper quote's half
    spread only)."""
    bt, pp = _backtest(case, _gap_through(case)), _paper(case, _gap_through(case))
    _setup_gap_past_bankruptcy(bt)
    _setup_gap_past_bankruptcy(pp)

    def view(d):
        return {"booked_as": sorted(set(d["closing_intents"])), "loss_minus_x": round(_loss(d) - d["x"], 2),
                "booked_px_minus_bk": round(d["close_px"] - d["bk"], 2)}

    got = {"backtest": view(bt), "paper": view(pp), "x_gap": round(abs(bt["x"] - pp["x"]), 2)}
    want_one = {"booked_as": ["liquidation"], "loss_minus_x": 0.0, "booked_px_minus_bk": 0.0}
    assert (all(abs(v["loss_minus_x"]) <= 0.011 and abs(v["booked_px_minus_bk"]) <= 0.011
                and v["booked_as"] == ["liquidation"] for v in (got["backtest"], got["paper"]))
            and got["x_gap"] <= 0.10), (got, {"backtest": want_one, "paper": want_one, "x_gap": "<= 0.10"})


# --------------------------------------------------------------------------------- the outage replay (#146 NA-3)

OUTAGE_CASES = [pytest.param(-1, "aggressive", 3.0, 0.40, 0.10, id="3x-short"),
                pytest.param(1, "balanced", 2.0, 0.60, 0.30, id="2x-long")]


def _hub_harness():
    try:
        import test_hub_146_qa as h  # #146's QA harness (tests/ on main since fac1a93)

        from sleeve_fund.paper import hub_client  # noqa: F401  (#146's hub-fed node)
    except ImportError:
        pytest.skip("needs #146's outage replay and its QA harness (tests/test_hub_146_qa.py); not on this head")
    return h


@pytest.mark.parametrize("side,profile,lev,gap,back", OUTAGE_CASES)
def test_gap_liq_cap_the_outage_replay_books_exactly_x_at_bankruptcy_and_names_the_excess(side, profile, lev, gap,
                                                                                         back):
    """The hub-146 "4bf4f97 delta" repro (Advisor item 1): a journaled position of 0.25 at about 60,000 (margin
    about 5,000 at 3x, 7,500 at 2x), the process down 00:06-00:12, the minute to 00:08 OPENING 40% (3x short) or 60%
    (2x long) against it, past the bankruptcy price. Today the replay books it at that open: 3x short about 6,018
    lost, 2x long about 9,011. The replay books exactly X, at the bankruptcy price, never below 10,000 - X on any
    mark, and names the excess as covered by the venue's insurance fund."""
    from sleeve_fund import risk
    from sleeve_fund.strategies import REGISTRY

    h = _hub_harness()
    saved = REGISTRY.get("probe")
    REGISTRY["probe"] = h._probe_classes()
    try:
        p = h.flat_prices(30)
        p = h.shape(h.shape(p, 7.0, 8.0, h.adverse(side, gap)), 8.0, 30, h.adverse(side, back))
        run = h.restart(p, 6, 12, side=side, perp=True, profile=profile, stop=0.01, qty=0.25)
    finally:
        if saved is None:
            REGISTRY.pop("probe", None)
        else:
            REGISTRY["probe"] = saved
    assert risk.profile(profile).max_leverage == lev, "set-up: the profile's leverage"
    held = [f for f in run.fills if f["order_id"] == "O-held"]
    closed = [f for f in run.fills if f["order_id"] != "O-held"]
    q = sum(f["qty"] for f in held)
    entry_px = sum(f["qty"] * f["price"] for f in held) / q
    gap_open = h.price_at(p, 7.0)
    bk = entry_px * (1 - side / lev)
    assert abs(q - 0.25) < 1e-9 and closed and abs(sum(f["qty"] for f in closed) - q) < 1e-8, (
        "set-up: 0.25 held over the restart, closed in full", held, closed)
    assert side * (gap_open - bk) < 0, ("set-up: the minute to 00:08 opened past the bankruptcy price", gap_open, bk)
    assert run.store.funding_total("q146") == 0, "set-up: no funding settled"
    x = q * entry_px / lev + sum(f["fee"] for f in held) + sum(f["fee"] for f in closed)
    cash = float(run.store.journal_book("q146", 10_000)["cash"])
    close_px = sum(f["qty"] * f["price"] for f in closed) / q
    excess = side * (bk - gap_open) * q
    marks = [m["equity"] for m in run.store.equity_series("q146", limit=100_000)]
    said = [e["message"] for e in run.events if INSURANCE_WORDS in str(e.get("message", "")).lower()
            and any(abs(v - excess) <= max(0.011, 1e-3 * excess) for v in _money(e["message"]))]
    intents = sorted({o["intent"] for o in run.orders if o["order_id"] != "O-held"})
    got = {"booked_as": intents, "loss": round(10_000 - cash, 2), "booked_px": round(close_px, 2),
           "lowest_mark": round(min(marks), 2) if marks else None, "insurance_said": bool(said)}
    want = {"booked_as": ["liquidation"], "loss": round(x, 2), "booked_px": round(bk, 2),
            "lowest_mark": f">= {10_000 - x:,.2f}", "insurance_said": True}
    assert (intents == ["liquidation"] and abs(10_000 - cash - x) <= 0.011 and abs(close_px - bk) <= 0.011
            and marks and min(marks) >= 10_000 - x - 0.011 and said), (got, want, f"excess {excess:,.2f}")


# ------------------------------------------------------------------------- a liquidation short of bankruptcy


@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_gap_liq_cap_a_liquidation_short_of_bankruptcy_also_books_exactly_x(mode, case):
    """Advisor 00:19 (2). Gapped past the liquidation price but NOT past the bankruptcy price: booked as a
    liquidation at the bankruptcy price, and the loss is exactly X; the maintenance margin left between the market's
    price and bankruptcy is forfeited to the venue (3x short about 10.00, 2x long about 2.50), with no insurance
    diagnostic. Today it books the market's price and loses X less that margin (main books it as a stop-loss)."""
    d = _run(mode, case, _gap_short_of_bankruptcy(case))
    _setup_gap_short_of_bankruptcy(d)
    got = {"booked_as": d["closing_intents"], "loss": round(_loss(d), 2), "booked_px": round(d["close_px"], 2)}
    want = {"booked_as": ["liquidation"] * len(d["closing_intents"]), "loss": round(d["x"], 2),
            "booked_px": round(d["bk"], 2)}
    assert (got["booked_as"] == want["booked_as"] and abs(_loss(d) - d["x"]) <= 0.011
            and abs(d["close_px"] - d["bk"]) <= 0.011), (got, want)


@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_control_a_liquidation_short_of_bankruptcy_records_no_insurance_cover_and_loses_no_more_than_x(mode, case):
    """The part of "unchanged" that holds today and must keep holding: nothing went past bankruptcy, so no insurance
    row, no insurance event, and the loss is at most X."""
    d = _run(mode, case, _gap_short_of_bankruptcy(case))
    _setup_gap_short_of_bankruptcy(d)
    assert not d["insurance"], d["insurance"]
    assert not [m for k, m in d["events"] if k == "insurance_fund" or INSURANCE_WORDS in m.lower()]
    assert _loss(d) <= d["x"] + 0.011, (_loss(d), d["x"])


@pytest.mark.parametrize("case", CASE_IDS)
@pytest.mark.parametrize("mode", MODES)
def test_control_a_gap_through_the_stop_short_of_liquidation_stays_a_stop_loss_below_x(mode, case):
    """20% against: through the 10% stop, nowhere near the liquidation price. A stop-loss at the gapped price, no
    insurance, a loss well under X. GAP-LIQ-CAP must not touch it."""
    d = _run(mode, case, _gap_short_of_liquidation(case))
    _setup_gap_short_of_liquidation(d)
    assert set(d["closing_intents"]) == {"stop_loss"}, d["closing_intents"]
    assert not d["insurance"]
    loss = _loss(d)
    expected = d["side"] * (d["entry_px"] - d["close_px"]) * d["q"] + sum(f["fee"] for f in d["opened"] + d["closed"])
    assert abs(loss - expected) <= 0.011 and loss < 0.7 * d["x"], (loss, expected, d["x"])


# ============================================================================== G1: out-of-sample and holdout

DAY0 = pd.Timestamp("2024-01-02", tz="UTC")


def _sched_classes(plan: tuple):
    """A daily strategy that holds side s on the bars closing on days [a, b) of `plan` ((a, b, s), ...), counted
    from DAY0, and is flat otherwise."""
    from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy

    class QASchedConfig(LongFlatConfig):
        def __init__(self, *, pad: int = 0, **kwargs) -> None:
            super().__init__(**kwargs)
            self.pad = pad

    class QASched(LongFlatStrategy):
        def _side(self, bar) -> int:
            k = int((pd.Timestamp(bar.ts_event, unit="ns", tz="UTC") - DAY0) / pd.Timedelta(days=1))
            return next((s for a, b, s in plan if a <= k < b), 0)

        def want_long(self, bar):
            return self._side(bar) > 0

        def want_side(self, bar):
            return self._side(bar)

    return QASched, QASchedConfig


def _path(n: int, legs: list[tuple[int, int, float, float]]) -> pd.DataFrame:
    """Daily closes, flat at 100 unless a leg (a, b, from, to) moves it linearly over days [a, b], then held. Every
    bar opens at its own close (a change between bars is a gap)."""
    c = np.full(n, 100.0)
    for a, b, lo, hi in legs:
        c[a:b + 1] = np.linspace(lo, hi, b - a + 1)
        c[b + 1:] = hi
    idx = pd.date_range(DAY0, periods=n, freq="D")
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e9}, index=idx)


def _spec(name: str):
    from sleeve_fund.strategies.base import IdeaSpec

    return IdeaSpec(name=name, family="test", idea="QA GAP-LIQ-CAP schedule", rules="holds a fixed schedule",
                    param_grid={"market": ["perp"], "allow_short": [True]}, default_params={})


ACK_NOTE = "PM: liquidated by a gap past a correctly placed stop; seen and accepted"


def _g1_checks(result, ledger, ack):
    """ASSUMED interface (one place to change): tearsheet.g1_checks(result, ledger, liquidation_ack=<the PM's note>)
    takes the PM's explicit acknowledgement of a liquidation finding; without it (None) nothing is acknowledged.
    Returns (checks, failed) as tearsheet.g1_verdict reads them, or the "not built" text."""
    from sleeve_fund.research.tearsheet import g1_checks, g1_verdict

    try:
        checks = g1_checks(result, ledger) if ack is None else g1_checks(result, ledger, liquidation_ack=ack)
    except TypeError as exc:
        return f"not built: tearsheet.g1_checks(result, ledger, liquidation_ack=<PM's note>): {exc}"
    return checks, g1_verdict(checks)[1]


@functools.lru_cache(maxsize=None)
def _study(which: str):
    """OOS: 120 days, walk-forward 30/30 (fold 1 tests days 30-59). A 3x long days 32-45 doubles (100 -> 200, about
    +3,000), then a 3x short from day 47 is gapped 60% on day 50 (200 -> 320): past its stop, liquidation and
    bankruptcy, INSIDE fold 1's test window. Overall P&L positive on every head (the short loses 1,800-2,400 today).
    "oos" runs with the 10% stop (correctly placed: inside half the distance to liquidation), "oos-nostop" with no
    stop.
    HOLDOUT: 150 days, 30-day holdout opened; a small long through the research period (so every fold scores), then
    the same long-then-gapped-short inside the holdout (days 122-145, gap on day 140).
    Returns (result, sheet, prices, full run (holdout only), {ack: G1 checks}) with the ledger still open."""
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.venues import venue

    stop = None if which == "oos-nostop" else STOP
    if which.startswith("oos"):
        plan = ((32, 45, 1), (47, 56, -1))
        prices = _path(120, [(32, 45, 100.0, 200.0), (50, 50, 320.0, 320.0)])
        prices.iloc[46:50, :4] = 200.0
        kw = dict(holdout_days=0, use_holdout=False)
    else:
        plan = ((2, 118, 1), (122, 135, 1), (137, 146, -1))
        prices = _path(150, [(2, 118, 100.0, 110.0), (122, 135, 110.0, 220.0), (140, 140, 352.0, 352.0)])
        prices.iloc[136:140, :4] = 220.0
        kw = dict(holdout_days=30, use_holdout=True)
    name = f"qa_gapcap_{which.replace('-', '_')}"
    inst = venue("KRAKEN").instrument("BTC", "USD")
    exits = {} if stop is None else {"stop_loss": stop}
    with _patched(), tempfile.TemporaryDirectory(prefix="qa-gapcap-study-") as d:
        REGISTRY[name] = _sched_classes(plan)
        try:
            ledger = IdeaLedger(Path(d) / "ledger.jsonl")
            result = run_study(_spec(name), prices, inst, dataset="qa-gapcap", ledger=ledger, train_days=30,
                               test_days=30, exits=exits, risk_profile="aggressive", half_spread=0.0, **kw)
            from sleeve_fund.research.tearsheet import render

            try:
                sheet = render(result, ledger)
            except Exception as exc:  # noqa: BLE001 - the sheet is read by the final assertion only
                sheet = f"<render failed: {exc!r}>"
            checks = {ack: _g1_checks(result, ledger, ack) for ack in (None, ACK_NOTE)}
            full = None
            if which == "holdout":  # the study's holdout run, as it ran it: the chosen settings on every bar
                from sleeve_fund.research.runner import run_backtest

                full = run_backtest(name, prices, inst, {"market": "perp", "allow_short": True, "stop_loss": STOP},
                                    risk_profile="aggressive", half_spread=0.0)
            return result, sheet, prices, full, checks
        finally:
            REGISTRY.pop(name, None)


def _closes_past_liquidation(res, start, end) -> list:
    """Closing fills of a short, inside [start, end], at or past that short's entry liquidation price (any intent:
    main books it as a stop-loss)."""
    j = res.journal
    orders = {o["order_id"]: o for o in j.orders_.values()}
    out, liq = [], None
    for f in sorted(j.fills_, key=lambda f: f["ts"]):
        o = orders.get(f["order_id"], {})
        if o.get("intent") == "entry" and f["side"] == "SELL":
            liq = (o.get("signal") or {}).get("liquidation_px")
        elif f["side"] == "BUY" and liq is not None and f["price"] >= liq and start <= pd.Timestamp(f["ts"]) <= end:
            out.append((o.get("intent"), f["price"], liq, f["ts"]))
    return out


def _findings(result, checks) -> list[str]:
    """The G1 findings shown to the PM: StudyResult.findings / g1_findings (assumed names) and the G1 checks' rows
    ("name: detail") that name a liquidation."""
    got = getattr(result, "g1_findings", None)
    if got is None:
        got = getattr(result, "findings", None)
    texts = [f if isinstance(f, str) else str(f.get("text", f) if isinstance(f, dict) else getattr(f, "text", f))
             for f in (got or [])]
    rows = checks.get(None)
    if isinstance(rows, tuple):
        texts += [f"{name}: {detail}" for name, _, detail in rows[0] if "liquidat" in f"{name} {detail}".lower()]
    if got is None and not texts:
        raise AssertionError("not built: a G1 finding for a liquidation in out-of-sample or the holdout (StudyResult."
                             "findings, or a G1 check naming it)")
    return texts


def _on_sheet(text: str, sheet: str) -> bool:
    return text in sheet or all(part.strip() in sheet for part in text.split(": ", 1))


def _oos_setup(result, which: str) -> list:
    assert not result.errors, ("set-up: the strategy raised", result.errors[:1])
    f1 = result.folds[0]
    test_start = f1.train_end + pd.Timedelta(days=1)
    hits = _closes_past_liquidation(result.full_period, test_start, f1.test_end)
    assert hits, f"set-up ({which}): a short closed at or past its liquidation price inside fold 1's test window"
    j = result.full_period.journal
    orders = {o["order_id"]: o for o in j.orders_.values()}
    shorts = [f for f in sorted(j.fills_, key=lambda f: f["ts"]) if f["side"] == "SELL"
              and orders.get(f["order_id"], {}).get("intent") == "entry"]
    assert shorts, f"set-up ({which}): the short's entry fill"
    entry, liq = shorts[-1]["price"], hits[0][2]
    half = 0.5 * abs(liq / entry - 1)
    # the short's protective stops: buy orders with a trigger short of liquidation (the stop-safety stack journals a
    # stop that filled past liquidation with intent "liquidation", so the intent can't tell)
    stops = [o for o in orders.values() if o.get("side") == "BUY" and (o.get("signal") or {}).get("trigger")
             is not None and (o.get("signal") or {})["trigger"] < liq * (1 - 1e-6)]
    if which == "oos":
        assert STOP <= half and stops, ("set-up: a stop placed within half the distance to liquidation", STOP, half)
    else:
        assert not stops, ("set-up: the short had no stop", stops)
    return hits


def _liq_rows(checks, ack):
    got = checks[ack]
    if isinstance(got, str):
        raise AssertionError(got)
    rows, failed = got
    return [(name, verdict, detail) for name, verdict, detail in rows
            if "liquidat" in f"{name} {detail}".lower()], failed


def test_gap_liq_cap_a_liquidation_in_an_out_of_sample_window_is_a_g1_finding_even_with_positive_pnl():
    """Advisor 23:42 guard: ANY liquidation in a strategy's out-of-sample windows is a G1 finding shown to the PM,
    whatever the P&L. The study's overall P&L and the window's are positive; a finding names the liquidation and the
    out-of-sample window, and the tear sheet shows it."""
    result, sheet, prices, _, checks = _study("oos")
    _oos_setup(result, "oos")
    f1 = result.folds[0]
    test_start = f1.train_end + pd.Timedelta(days=1)
    eq = result.full_period.equity
    assert float(eq.iloc[-1]) > 10_000, ("set-up: overall P&L positive", float(eq.iloc[-1]))
    win = eq[(eq.index >= test_start - pd.Timedelta(days=1)) & (eq.index <= f1.test_end)]
    assert float(win.iloc[-1]) > float(win.iloc[0]), ("set-up: the window's own P&L positive", win.iloc[[0, -1]])
    texts = _findings(result, checks)
    found = [t for t in texts if "liquidat" in t.lower() and ("out-of-sample" in t.lower()
                                                             or "test window" in t.lower())]
    assert found and _on_sheet(found[0], sheet), {"findings": texts, "on_sheet": bool(found) and _on_sheet(
        found[0], sheet)}


def test_gap_liq_cap_a_liquidation_in_the_holdout_is_a_g1_finding():
    """The same guard for the holdout: the holdout is opened, it holds a liquidation, and a G1 finding names the
    liquidation and the holdout, on the tear sheet."""
    result, sheet, prices, full, checks = _study("holdout")
    assert not result.errors, ("set-up: the strategy raised", result.errors[:1])
    assert result.holdout is not None, ("set-up: the holdout was opened", result.holdout_withheld)
    h_start = prices.index[-30]
    hits = _closes_past_liquidation(full, h_start, prices.index[-1])
    assert hits, "set-up: a short closed at or past its liquidation price inside the holdout"
    texts = _findings(result, checks)
    found = [t for t in texts if "liquidat" in t.lower() and "holdout" in t.lower()]
    assert found and _on_sheet(found[0], sheet), {"findings": texts}


def test_gap_liq_cap_g1_a_gap_past_a_correctly_placed_stop_passes_only_with_the_pms_acknowledgement():
    """Advisor 00:19 (5), first cause: the out-of-sample liquidation came from a gap past a stop placed within half
    the distance to liquidation. Without the PM's acknowledgement a G1 check naming the liquidation is among the
    checks that stop a pass (tearsheet.g1_verdict's failed list); with it (ASSUMED: g1_checks(...,
    liquidation_ack=<the PM's note>)) the check is still shown and no longer stops a pass."""
    result, sheet, prices, _, checks = _study("oos")
    _oos_setup(result, "oos")
    before, failed_before = _liq_rows(checks, None)
    blocks = [r for r in before if r[0] in failed_before]
    after, failed_after = _liq_rows(checks, ACK_NOTE)
    got = {"without_ack": before, "blocks_pass": bool(blocks), "with_ack": after,
           "still_blocks": [r for r in after if r[0] in failed_after]}
    want = {"without_ack": "a check naming the liquidation", "blocks_pass": True, "with_ack": "the same check, shown",
            "still_blocks": []}
    assert blocks and after and not got["still_blocks"], (got, want)


def test_gap_liq_cap_g1_a_liquidation_with_the_stop_missing_is_a_fail_the_pm_cannot_override():
    """Advisor 00:19 (5), second cause: the same out-of-sample liquidation with no stop: a G1 FAIL. The check naming
    it reads FAIL and stops the pass with and without the PM's acknowledgement. (A stop beyond half the distance to
    liquidation is the same FAIL, but the engine refuses such an entry ("Entry refused: its 25.00% stop is more than
    50% of the way to the liquidation price"), so no study run can hold one; not pinned.)"""
    result, sheet, prices, _, checks = _study("oos-nostop")
    _oos_setup(result, "oos-nostop")
    before, failed_before = _liq_rows(checks, None)
    fails = [r for r in before if r[1] == "FAIL" and r[0] in failed_before]
    after, failed_after = _liq_rows(checks, ACK_NOTE)
    still = [r for r in after if r[1] == "FAIL" and r[0] in failed_after]
    assert fails and still, ({"without_ack": before, "with_ack": after}, "a FAIL naming the liquidation, both ways")


def test_gap_liq_cap_the_excess_is_not_in_the_random_entry_benchmark():
    """Advisor 00:19 (4). The random-entry benchmark (C3) compares the strategy's own out-of-sample trade returns with
    random draws. The liquidated short's own return there is its loss X over its notional on the benchmark's cost
    basis (the posted 1/3 at 3x + costs), not the unlevered close-to-close move past bankruptcy (-60% less costs), so
    the strategy's compounded return is (1 + the long's) x (1 + the short's) - 1, both on one cost per side."""
    result, sheet, prices, _, _ = _study("oos")
    assert not result.errors, ("set-up: the strategy raised", result.errors[:1])
    re_ = result.random_entry
    assert re_ is not None and re_.trades == 2, ("set-up: both trades of fold 1's window reach the benchmark",
                                                  getattr(re_, "trades", None))
    j = result.full_period.journal
    orders = {o["order_id"]: o for o in j.orders_.values()}
    fills = sorted(j.fills_, key=lambda f: f["ts"])
    short_in = [f for f in fills if orders.get(f["order_id"], {}).get("intent") == "entry" and f["side"] == "SELL"]
    short_out = [f for f in fills if f["side"] == "BUY" and short_in and pd.Timestamp(f["ts"]) > pd.Timestamp(
        short_in[-1]["ts"])]
    assert short_in and short_out, "set-up: the short opened and closed"
    q = sum(f["qty"] for f in short_in)

    # One fee basis for both trades (QA amendment 01:05, PE1's finding): the benchmark's own cost per side, which the
    # study passes as instrument.taker_fee + the half spread (0 here). The short's X over its notional is then the
    # posted margin (1/3 at 3x) + the entry's cost + the liquidation fee on the trigger price (Advisor 00:19 (1)), about
    # 4/3 of the entry at 3x; the trigger's exact place moves the result by < 1e-3, inside the tolerance. If RE-COST
    # changes the benchmark's basis to the strategy's market fee, change `cost` here only.
    # Re-pinned 7 Oct (HoQA, after #181 RE-COST merged in 6a9256a): the benchmark now charges the run's own market fee,
    # here the study's perp at 0.05% taker (+ 0 half spread), not the venue's 0.80% spot taker. Set-up only: the
    # formula and the 2e-3 tolerance are unchanged. (Was: venue("KRAKEN").instrument("BTC", "USD").taker_fee.)
    cost = 0.0005 + 0.0
    r_long = 200.0 / 100.0 - 1 - 2 * cost
    r_short = -(1 / 3.0) - cost - cost * (4 / 3.0)
    want = (1 + r_long) * (1 + r_short) - 1
    assert q > 0 and abs(re_.strategy_return - want) <= 2e-3, {"strategy_return": round(re_.strategy_return, 4),
                                                                "want": round(want, 4)}


def _draws_capped(closes, trades, windows, cost, lev):
    """ASSUMED interface (one place to change): the benchmark is told each trade's leverage, either on the trade
    (random_entry.Trade(entry, exit, side, leverage=...)) or for the run (random_entry(..., leverage=...)), so its
    draws book a liquidation as the engine does."""
    from sleeve_fund.research import random_entry as rmod

    try:
        placed = [rmod.Trade(e, x, sd, leverage=lev) for e, x, sd in trades]
        return rmod.random_entry(closes, placed, windows, cost, draws=200)
    except TypeError as first:
        try:
            return rmod.random_entry(closes, [rmod.Trade(e, x, sd) for e, x, sd in trades], windows, cost, draws=200,
                                     leverage=lev)
        except TypeError as exc:
            raise AssertionError(f"not built: the random-entry draws told the trade's leverage ({first}; {exc})") \
                from None


def test_gap_liq_cap_random_entry_draws_book_a_gap_past_bankruptcy_as_the_engine_does():
    """Advisor 00:19 (4): the draws use the same engine and cap. One 3x short holding its whole 5-bar window, so every
    draw enters on bar 0 and is out on bar 5, across a 60% gap up on bar 3: each draw loses X over its notional
    (1/3 of it plus the fees, about -33.4%), never the unlevered -60% less costs it is charged today."""
    from sleeve_fund.research import random_entry as rmod

    closes = [100.0, 100.0, 100.0, 160.0, 160.0, 160.0]
    cost = 0.0005
    plain = rmod.random_entry(closes, [rmod.Trade(0, 5, -1)], [(0, 5)], cost, draws=200)
    assert plain.trades == 1 and abs(plain.median_random_return - (-0.6 - 2 * cost)) < 1e-9, (
        "set-up: every draw is the one placement, across the gap", plain.median_random_return)
    res = _draws_capped(closes, [(0, 5, -1)], [(0, 5)], cost, 3.0)
    lo, hi = -(1 / 3 + 3 * cost), -1 / 3
    assert lo - 1e-9 <= res.median_random_return <= hi + 1e-9 and lo - 1e-9 <= res.strategy_return <= hi + 1e-9, {
        "median_draw": round(res.median_random_return, 5), "strategy": round(res.strategy_return, 5),
        "want": f"between {lo:.5f} and {hi:.5f}"}


# ======================================================================================= the register re-run


@LIQ_CAP
def test_gap_liq_cap_register_rerun_replaces_the_row_marks_it_superseded_and_leaves_n_unchanged():
    """Advisor 23:42: register perp backtests that contain a liquidation are re-run, and both figures recorded;
    00:19 (6): the re-run replaces the row, which is marked superseded, and N is unchanged.
    Assumed interface (PE1 may rename): `sleeve_fund.research.trials.rerun_liquidated(store, run)`, where
    `run(trial_row)` re-runs one trial's backtest and returns its BacktestResult. It re-runs every counted perp trial
    whose saved backtest holds a liquidation (here one of two), and only those; returns one row per re-run with
    `before` and `after`, each with `sharpe` and `liquidation_loss`; marks the original trials row superseded
    (assumed column `superseded_by` = the new row's id), leaving its figures as they were; adds the re-run as a new
    row with `rerun_of` = the original's id, in the same variant; N (the register's variant count) does not move; and
    the variant's Sharpe in the spread (TrialsRegister.sharpes) is the re-run's, never the superseded -1.5."""
    from sleeve_fund.research import trials as trials_mod
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.research.trials import TrialsRegister, model_run_row
    from sleeve_fund.store import Store
    from sleeve_fund.venues import venue

    store = Store.in_memory()
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    side, profile, _ = CASES["3x-short"]
    saved = {}
    with _patched():
        for run_id, factor in (("qaliq", _gap_through("3x-short")), ("qacalm", 1.0)):
            res = run_backtest("qa_gap_probe", _minute_bars(factor), inst, params=_params(side),
                               starting_capital=10_000, risk_profile=profile, bar_minutes=1, half_spread=0.0)
            trial = model_run_row(strategy="qa_gap_probe", params={**_params(side), "run": run_id},
                                  dataset="kraken-btcusd-1m", source="backtest", setup={"risk_profile": profile},
                                  sharpe=-1.5 if run_id == "qaliq" else 0.4, backtest_id=run_id, trades=1)
            bt_name = store.save_backtest(res.journal, run_id=run_id, key=run_id, title=run_id, query="", result={},
                                          trial=trial)
            saved[run_id] = (trial, bt_name)
    def closes_past_liq(bt_name):
        orders = {o["order_id"]: o for o in store.orders(bt_name, limit=100_000)}
        fills = sorted(store.fills(bt_name, limit=100_000), key=lambda f: f["ts"])
        entry = next(orders[f["order_id"]] for f in fills if orders[f["order_id"]]["intent"] == "entry")
        liq = (entry.get("signal") or {}).get("liquidation_px")
        return [f for f in fills if f["side"] == "BUY" and liq is not None and f["price"] >= liq]

    assert closes_past_liq(saved["qaliq"][1]), "set-up: the saved gapped run closed its short past liquidation"
    assert not closes_past_liq(saved["qacalm"][1]) and len(store.fills(saved["qacalm"][1], limit=100)) == 1, (
        "set-up: the saved calm run only entered")
    before_rows = {t["id"]: dict(t) for t in store.trials()}
    variants = TrialsRegister(store).counts()["variants"]
    calls = []

    def rerun(trial_row):
        calls.append(trial_row["id"])
        with _patched():
            factor = _gap_through("3x-short") if trial_row["backtest_id"] == "qaliq" else 1.0
            return run_backtest("qa_gap_probe", _minute_bars(factor), inst, params=_params(side),
                                starting_capital=10_000, risk_profile=profile, bar_minutes=1, half_spread=0.0)

    fn = getattr(trials_mod, "rerun_liquidated", None)
    if fn is None:
        raise AssertionError("not built: sleeve_fund.research.trials.rerun_liquidated(store, run) (re-run register "
                             "perp backtests that contain a liquidation and record both figures)")
    try:
        out = fn(store, rerun)
    except TypeError as exc:
        raise AssertionError(f"not built: rerun_liquidated(store, run) as assumed: {exc}") from exc
    liq_id = saved["qaliq"][0]["id"]
    rows = list(out or [])
    after_rows = list(store.trials())
    new = [t for t in after_rows if t["id"] not in before_rows]
    figures_ok = (len(rows) == 1 and rows[0].get("trial_id") == liq_id
                  and all(isinstance(rows[0].get(k), dict) and {"sharpe", "liquidation_loss"} <= set(rows[0][k])
                          for k in ("before", "after"))
                  and rows[0]["before"]["sharpe"] == -1.5)
    # Supersession metadata only (DA 0007/0008 columns, QA 01:40); every other column of a kept row must not change.
    marks = {"superseded_by", "superseded", "superseded_reason", "superseded_at"}
    kept = {t["id"]: t for t in after_rows if t["id"] in before_rows}
    sharpes = TrialsRegister(store).sharpes()
    got = {"rerun_calls": calls, "rows": rows,
           "figures_kept": all({k: v for k, v in dict(t).items() if k not in marks}
                               == {k: v for k, v in before_rows[i].items() if k not in marks} for i, t in kept.items()),
           "superseded_by": {i: t.get("superseded_by") for i, t in kept.items()},
           "new_rows": [(t.get("rerun_of"), t["definition_hash"] == saved["qaliq"][0]["definition_hash"]) for t in new],
           "variants_unchanged": TrialsRegister(store).counts()["variants"] == variants,
           "superseded_sharpe_in_spread": -1.5 in sharpes}
    want = {"rerun_calls": [liq_id], "rows": "one row: trial_id, before/after with sharpe and liquidation_loss",
            "figures_kept": True, "superseded_by": {liq_id: new[0]["id"] if len(new) == 1 else "<the re-run's id>",
                                                     saved["qacalm"][0]["id"]: None},
            "new_rows": [(liq_id, True)], "variants_unchanged": True, "superseded_sharpe_in_spread": False}
    assert (calls == [liq_id] and figures_ok and got["figures_kept"] and got["new_rows"] == [(liq_id, True)]
            and got["superseded_by"] == want["superseded_by"] and got["variants_unchanged"]
            and not got["superseded_sharpe_in_spread"]), (got, want)
