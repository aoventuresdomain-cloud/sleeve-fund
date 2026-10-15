"""G5 end to end through the paper engine: a Model set to size by strength that gives no strength never enters; the
entry is skipped with an alert, never sized at full size. Kept apart from test_signal_strength: the exposure-gate
harness's offline fixtures are autouse."""
from __future__ import annotations

import os

import pandas as pd
import pytest
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund.store import Store


def _new_store(tmp_path, name):
    """Both engines: Postgres when TEST_DATABASE_URL is set (CI), else SQLite."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine, portfolio_gate=True)
    return Store(f"sqlite:///{tmp_path}/{name}.db", portfolio_gate=True)


def test_e2e_a_model_with_strength_on_but_no_strength_in_its_signal_never_enters_and_alerts(tmp_path, monkeypatch):
    import test_exposure_gate_xfails as h
    from test_portfolio_paper_entry import END

    from sleeve_fund.supervisor import Supervisor

    store = _new_store(tmp_path, "g5")
    params = {**h.PERP, "stop_loss": h.STOP, "signal_strength": {"mapping": "linear", "floor": 0.3}}
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params=params)
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 20_000, actor="pm", reason="QA: the fund's capital", rebase=True,
                                      now=before), before)
    Supervisor(store).mark_portfolio(END)
    plan = h.Plan(t0=h.STD_T0, minutes=6, windows=[(h.M(h.STD_T0, 2), h.M(h.STD_T0, 6), 1)], balance=200_000.0,
                  params={"signal_strength": params["signal_strength"]})
    h.run(tmp_path, store, plan, monkeypatch)
    assert not [o for o in h._orders(store) if o["intent"] == "entry"]
    alerts = [e for e in h._events(store) if e["kind"] == "entry_skipped_strength"]
    assert alerts and alerts[0]["level"] == "error" and "isn't a number from 0 to 1" in alerts[0]["message"], alerts


# ---- Done-when 4: a forced restart replays the same strengths and sizes; the backtest decides the same strengths ------
def _strength_model(monkeypatch):
    """Egx, with a strength from the decision bar's close only (its tenths digit: 0.6, 0.2, 0.8, 0.4, 0.0, ... as the
    harness's price climbs 0.6 a minute), as a Model would read a closed bar's indicator."""
    import test_exposure_gate_xfails as h

    from sleeve_fund.strategies import REGISTRY

    class EgxS(h.Egx):
        def explain(self, bar, target):
            reason, values = super().explain(bar, target)
            return reason, {**values, "strength": (round(bar.close.as_double() * 10) % 10) / 10}

    monkeypatch.setitem(REGISTRY, "egx", (EgxS, h.EgxConfig))
    return h


LINEAR = {"mapping": "linear", "floor": 0.3}


def _windows(h, t0):
    return [(h.M(t0, a), h.M(t0, b), 1) for a, b in ((2, 4), (5, 7), (9, 11), (12, 14), (15, 17))]


def _paper(tmp_path, monkeypatch, splits):
    """The session in one process (splits = [18]) or restarted, flat, at each split."""
    from test_portfolio_paper_entry import END

    from sleeve_fund.supervisor import Supervisor

    h = _strength_model(monkeypatch)
    store = _new_store(tmp_path, f"g5-{len(splits)}")
    params = {**h.PERP, "stop_loss": h.STOP, "signal_strength": LINEAR}
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params=params)
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 20_000, actor="pm", reason="QA: the fund's capital", rebase=True,
                                      now=before), before)
    Supervisor(store).mark_portfolio(END)
    start = 0
    for i, end in enumerate(splits):
        t0 = h.M(h.STD_T0, start)
        h.run(tmp_path, store, h.Plan(t0=t0, minutes=end - start, windows=_windows(h, h.STD_T0), balance=200_000.0,
                                      params={"signal_strength": LINEAR}, tag=f"s{len(splits)}{i}",
                                      price=lambda s, off=start: 60_000 + (s + off * 60) * 0.01), monkeypatch)
        start = end
    entries = [o for o in h._orders(store) if o["intent"] == "entry"]
    _paper.last = (store, entries)  # the parity cell reads the gate's trims and the signals from it
    return [(o["ts"], o["signal"].get("strength"), float(o["qty"])) for o in entries], h._events(store)


def test_parity_a_forced_restart_replays_the_same_strengths_and_sizes(tmp_path, monkeypatch):
    once, ev_once = _paper(tmp_path, monkeypatch, [18])
    restarted, ev = _paper(tmp_path, monkeypatch, [8, 18])  # restarted flat at minute 8, between two windows
    assert [e["kind"] for e in ev].count("start") == 2 and "restore_position" not in {e["kind"] for e in ev}
    assert len(once) >= 3 and all(w and w.startswith("signal strength") for _, w, _ in once), once
    assert restarted == once
    skipped = [e["message"] for e in ev_once if e["kind"] == "entry_skipped_strength"]
    assert any("below its floor" in m for m in skipped), skipped  # a 0.0 or 0.2 bar is skipped, journaled


def test_parity_the_backtest_decides_the_same_strengths_as_paper(tmp_path, monkeypatch):
    """The same closed bars into the backtest give the same strength on each entry bar, and the same share of the
    fund (the backtest's 20,000 of its own, paper the fund's 20,000)."""
    import numpy as np

    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    paper, _ = _paper(tmp_path, monkeypatch, [18])
    h = _strength_model(monkeypatch)
    idx = pd.date_range(h.STD_T0, periods=18, freq="1min", tz="UTC") + pd.Timedelta(minutes=1)
    close = np.round(60_000 + 0.6 * np.arange(1, 19), 1)
    prices = pd.DataFrame({"open": close - 0.6, "high": close, "low": close - 0.6, "close": close, "volume": 1e9},
                          index=idx)
    params = {**h.PERP, "stop_loss": h.STOP, "signal_strength": LINEAR,
              "windows": [[h._ns(o), h._ns(c), s] for o, c, s in _windows(h, h.STD_T0)]}
    r = run_backtest("egx", prices, venue("binance").instrument("BTC", "USDT"), params, starting_capital=20_000,
                     risk_profile="balanced", bar_minutes=1, half_spread=0)
    bt = sorted((o["ts"], (o.get("signal") or r.decisions.get(o["order_id"], {}).get("signal") or {}).get("strength"),
                 float(o["qty"])) for o in r.journal.orders_.values() if o["intent"] == "entry")
    # strength and the strength-scaled share of the fund; the final size can differ by the caps that apply after it
    # (paper's margin account sizes off the fund's virtual funding), never by the strength
    assert [(t, w) for t, w, _ in paper] == [(t, w) for t, w, _ in bt], (paper, bt)

    # Advisor: no divergence hides behind the trim. Paper's size before the gate is the strength-scaled share of the
    # fund it sized off, to the lot; its final size is the smaller of that and what the gate's trim allowed.
    import re
    from decimal import ROUND_DOWN
    from decimal import Decimal as D

    store, entries = _paper.last
    trims = {(r["decided_at"], round(float(r["requested_qty"]), 3)): r for r in store.refused_ideas(h.NAME)
             if r["outcome"] == "trimmed"}
    trimmed = 0
    for o in entries:
        sig = o["signal"]
        share = D(re.search(r"= ([0-9.]+)% of fund", sig["strength"]).group(1)) / 100
        want = (share * D(str(sig["fund_equity"])) / D(str(sig["close"]))).quantize(D("0.001"), rounding=ROUND_DOWN)
        trim = next((r for (at, req), r in trims.items() if abs((at - o["ts"]).total_seconds()) < 60), None)
        if trim is not None:
            trimmed += 1
            assert D(str(trim["requested_qty"])).quantize(D("0.001")) == want, (sig, trim)
            assert trim["test_hit"] == "asset_net", trim  # T4 net per asset
            assert D(str(o["qty"])) == min(want, D(str(trim["approved_qty"])).quantize(D("0.001"))), (o, trim)
        else:
            assert D(str(o["qty"])) == want, (o["qty"], want, sig)
    assert trimmed, "setup: the T4 trim acted on at least one entry"



# ---- the parity cells kill planted mutants (HoQA: written after the hook, so each must turn red on a mutant) ----------
def _lagged(monkeypatch, paper_only: bool):
    """A planted mutant: the hook sizes off the strength of the previous decision, kept on the Strategy (so a restart
    loses it); paper_only: only where a runtime is attached outside a backtest (paper reads it differently)."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    real = LongFlatStrategy._strength_limit

    def mutant(self, limits, values, base):
        rt = getattr(self, "runtime", None)
        if paper_only and (rt is None or rt.backtest):
            return real(self, limits, values, base)
        prev = self.__dict__.get("_qa_prev_strength", 0.95)
        self.__dict__["_qa_prev_strength"] = values.get("strength")
        return real(self, limits, {**values, "strength": prev}, base)

    monkeypatch.setattr(LongFlatStrategy, "_strength_limit", mutant)


def test_mutant_a_strength_kept_in_process_state_turns_the_restart_parity_red(tmp_path, monkeypatch):
    _lagged(monkeypatch, paper_only=False)
    with pytest.raises(AssertionError) as red:
        test_parity_a_forced_restart_replays_the_same_strengths_and_sizes(tmp_path, monkeypatch)
    assert "restarted == once" in str(red.traceback[-1].statement), red.value  # red on the parity, not on the setup


def test_mutant_paper_reading_strength_differently_from_the_backtest_turns_the_backtest_parity_red(tmp_path,
                                                                                                   monkeypatch):
    _lagged(monkeypatch, paper_only=True)
    with pytest.raises(AssertionError) as red:
        test_parity_the_backtest_decides_the_same_strengths_as_paper(tmp_path, monkeypatch)
    assert "(paper, bt)" in str(red.traceback[-1].statement), red.value  # red on the strength comparison itself
