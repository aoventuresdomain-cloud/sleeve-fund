"""R-13 and R-15 (HoE, PM 21 Oct 2026): every tear sheet has returns by year and month against buy-and-hold, and
every trade with the equity after it, over the scored window; the months compound and the trades sum to the end
equity, both to the cent; the starting capital is a study input; the trade list, the daily equity and its curve sit
beside the sheet, and the dashboard serves exactly those three to the PM."""
from __future__ import annotations

import dataclasses
import re

import pandas as pd
import pytest

from sleeve_fund.research import periods
from test_r04_short_leg_sheet import LONG, SHORT, _study, perp_venue  # noqa: F401


def _known() -> pd.Series:
    """A daily curve stamped at each close (the next midnight): 1,000 at the close of 15 Jan 2021, 1,100 at the
    close of 31 Jan, 990 at the close of 28 Feb, 1,039.50 at the close of 10 Mar, straight lines between."""
    knots = {"2021-01-16": 1000.0, "2021-02-01": 1100.0, "2021-03-01": 990.0, "2021-03-11": 1039.5}
    idx = pd.date_range("2021-01-16", "2021-03-11", freq="D", tz="UTC")
    s = pd.Series(knots, dtype=float)
    s.index = pd.DatetimeIndex(s.index).tz_localize("UTC")
    return s.reindex(idx).interpolate()


def test_months_and_years_on_a_curve_with_known_returns():
    rows = periods.periods(_known(), "M")
    assert [(p.label, round(p.ret, 10), p.partial) for p in rows] == [
        ("2021-01", 0.1, "from 16 Jan"), ("2021-02", -0.1, ""), ("2021-03", 0.05, "to 10 Mar")]
    (year,) = periods.periods(_known(), "Y", trips=[{"closed": pd.Timestamp("2021-02-03")}, {"closed": None}])
    assert (year.label, round(year.ret, 10), year.partial, year.trades) == ("2021", 0.0395, "from 16 Jan, to 10 Mar", 1)
    assert year.max_dd == pytest.approx(990 / 1100 - 1)
    assert year.sharpe is None, "a partial year under 60 days has no Sharpe (Advisor (f))"
    assert round(1000 * (1 + periods.compounded(rows)), 2) == 1039.50


def test_trade_rows_and_the_trading_table_book_the_insurance_funds_cover(monkeypatch):
    """Advisor (g): every flow the equity carries is booked to the trade it fell in, the insurance fund's cover of
    a liquidation's shortfall too, so the trades still sum to the equity."""
    import types

    from sleeve_fund.research import study

    seen = []
    monkeypatch.setattr(periods, "trades", lambda rows, shorts, funding, insurance, open_trip: seen.append(insurance) or [])
    monkeypatch.setattr(study, "trades", lambda rows, shorts, funding, insurance: seen.append(insurance) or [])
    cover = [{"ts": pd.Timestamp("2021-01-02"), "amount": 12.5}]
    run = types.SimpleNamespace(equity=_known(), fills=None, shorts=True, funding=[], insurance=cover, decisions={})
    periods.trade_rows(run)
    study.StudyResult.trade_stats.fget(types.SimpleNamespace(full_period=run))
    assert seen == [cover, cover]


def test_a_curve_that_starts_on_the_first_of_a_month_has_that_month_whole():
    idx = pd.date_range("2021-01-01", "2021-03-01", freq="D", tz="UTC")  # the first close: 31 Dec 2020
    eq = pd.Series(range(1000, 1000 + len(idx)), index=idx, dtype=float)
    rows = periods.periods(eq, "M")
    assert [(p.label, p.partial) for p in rows] == [("2021-01", ""), ("2021-02", "")]


@pytest.mark.parametrize("defn", [LONG, SHORT], ids=["long", "short"])
def test_the_sheets_tables_reconcile_to_its_end_equity_to_the_cent(tmp_path, perp_venue, defn):
    r, _, sheet = _study(tmp_path, perp_venue, defn)
    eq, hold = periods.scored(r)
    start, end = float(eq.iloc[0]), float(eq.iloc[-1])
    rows = periods.trade_rows(r.full_period)
    assert rows and round(sum(t.pnl for t in rows), 2) == round(end - start, 2)
    assert round(rows[-1].equity_after, 2) == round(end, 2)
    closed = [t for t in rows if t.closed is not None]
    assert len(closed) == r.trade_stats["trades"] and round(sum(t.pnl for t in closed), 2) == round(r.trade_stats["pnl"], 2)
    assert all(t.drawdown <= 0 and t.held > pd.Timedelta(0) for t in rows)
    months, years = periods.periods(eq, "M"), periods.periods(eq, "Y", trips=[vars(t) for t in closed])
    assert round(start * (1 + periods.compounded(months)), 2) == round(end, 2)
    assert round(start * (1 + periods.compounded(years)), 2) == round(end, 2)
    assert sum(y.trades for y in years) == len(closed)
    for title in ("## Returns by year", "## Returns by month", "## Every trade"):
        assert title in sheet
    order = [sheet.index(t) for t in ("## Results after fees", "## Returns by year", "## Returns by month",
                                      "## Trading and costs", "## Every trade", "## Cost ladder")]
    assert order == sorted(order), "the section layout DA builds on"
    assert f"| {years[0].label} ({years[0].partial}) |" in sheet
    listed = sheet[sheet.index("## Every trade"):sheet.index("## Cost ladder")]
    assert sum(1 for line in listed.splitlines() if re.match(r"\| \d+ \|", line)) == min(len(rows), 50)
    if len(rows) > 50:
        assert f"50 of {len(rows)} trades shown" in sheet


def test_a_long_trade_list_is_never_trimmed_silently(tmp_path, perp_venue):
    from sleeve_fund.research import tearsheet

    r, ledger, _ = _study(tmp_path, perp_venue, LONG)
    rows = periods.trade_rows(r.full_period)
    sheet = tearsheet.render(r, ledger, files="qr_x", shown_trades=2)
    assert f"2 of {len(rows)} trades shown; every trade is in [qr_x-trades.csv](qr_x/file/trades.csv)" in sheet
    assert "[qr_x-equity.svg](qr_x/file/equity.svg)" in sheet and "[qr_x-equity.csv](qr_x/file/equity.csv)" in sheet
    csv = periods.trades_csv(rows)
    assert csv.count("\n") == len(rows) + 1 and csv.startswith("trade,side,entry_time")


def test_the_starting_capital_is_a_study_input_and_scales_the_money(tmp_path, perp_venue):
    from sleeve_fund.research.run import StudyRequest

    assert {f.name: f.default for f in dataclasses.fields(StudyRequest)}["starting_capital"] == 10_000.0
    big, _, _ = _study(tmp_path / "big", perp_venue, LONG)
    small, _, sheet = _study(tmp_path / "small", perp_venue, LONG, starting_capital=1_000.0)
    assert float(small.full_period.equity.iloc[0]) == 1_000.0
    b, s = periods.trade_rows(big.full_period), periods.trade_rows(small.full_period)
    assert len(b) == len(s)
    for x, y in zip(b, s):  # the same trades at a tenth of the size, within the venue's lot
        assert (x.opened, x.closed, x.side) == (y.opened, y.closed, y.side)
        assert y.qty == pytest.approx(x.qty / 10, abs=0.001)
    assert "Starting capital: 1,000." in sheet


def test_the_files_beside_a_sheet(tmp_path, perp_venue):
    from sleeve_fund.research.run import write_beside

    r, _, _ = _study(tmp_path, perp_venue, LONG)
    out = tmp_path / "sheets" / "qr_x.md"
    out.parent.mkdir()
    write_beside(r, out)
    names = sorted(p.name for p in out.parent.iterdir())
    assert names == ["qr_x-equity.csv", "qr_x-equity.svg", "qr_x-trades.csv"]
    svg = (out.parent / "qr_x-equity.svg").read_text()
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox=') and "<script" not in svg
    assert periods.equity_svg(r.full_period.equity, r.full_period_benchmark.equity, "<b>&") .count("&lt;b&gt;&amp;") == 2


def test_the_dashboard_serves_exactly_the_three_files_beside_a_sheet_to_the_pm(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    (sheets / "qr_x.md").write_text("# Tear sheet\n")
    for name, body in (("qr_x-trades.csv", "trade\n1\n"), ("qr_x-equity.csv", "day\n"), ("qr_x-equity.svg", "<svg/>"),
                       ("qr_x-notes.txt", "secret")):
        (sheets / name).write_text(body)
    (tmp_path / "outside-trades.csv").write_text("secret")
    monkeypatch.setattr(app_mod, "TEARSHEETS", sheets)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    c = TestClient(app_mod.create_app(Store(f"sqlite:///{tmp_path}/t.db")))
    auth = ("pm", "test-pw")
    for kind, mime in (("trades.csv", "text/csv"), ("equity.csv", "text/csv"), ("equity.svg", "image/svg+xml")):
        got = c.get(f"/research/qr_x/file/{kind}", auth=auth)
        assert got.status_code == 200 and got.headers["content-type"].startswith(mime), kind
        assert got.headers["x-content-type-options"] == "nosniff"
        assert c.get(f"/research/qr_x/file/{kind}").status_code == 401
        assert c.get(f"/research/qr_x/file/{kind}", auth=("pm", "nope")).status_code == 401
    assert c.get("/research/qr_x/file/trades.csv", auth=auth).text == "trade\n1\n"
    for bad in ("notes.txt", "..%2Foutside-trades.csv", "trades.csv%00", "TRADES.CSV", "equity.svg.bak"):
        assert c.get(f"/research/qr_x/file/{bad}", auth=auth).status_code == 404, bad
    for sheet in ("..%2Foutside", "..%2F..%2Fetc%2Fpasswd", "nope"):
        assert c.get(f"/research/{sheet}/file/trades.csv", auth=auth).status_code == 404, sheet
