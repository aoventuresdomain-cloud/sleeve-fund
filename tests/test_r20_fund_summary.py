"""R-20 2a (HoE, 21 Oct): every tear sheet gets its daily returns beside it (.returns.csv, written in the same step,
never a holdout day), and the fund summary reads the sheets and those files: one row per strategy with the sheet's
own printed figures, ranked G1 PASS, then NOT JUDGED, then FAIL, by out-of-sample Sharpe, ties by deflated Sharpe, and
the correlation of the strategies' daily returns over their common days. A sheet without its file, or a file without
its sheet, is refused. Expected values are worked out here from the fixtures, not read from the code under test."""
from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund.research import fund_summary as fs
from sleeve_fund.research.run import returns_path, write_returns


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import tearsheet
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import venue

    d = tmp_path_factory.mktemp("r20")
    ledger = IdeaLedger(d / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1900, seed=3), venue("KRAKEN").instrument("BTC", "USD"), dataset="syn",
                  ledger=ledger, synthetic=True, holdout_days=200, train_days=730, test_days=365)
    sheet = d / "trend_filter_syn.md"
    sheet.write_text(tearsheet.render(r, ledger), encoding="utf-8")
    write_returns(r, sheet)
    return r, sheet


def _read(path):
    return pd.read_csv(path, index_col="date", parse_dates=["date"])["return"]


# The returns file
def test_the_returns_file_ends_inside_the_research_period_never_on_a_holdout_day(study):
    r, sheet = study
    ret = _read(returns_path(sheet))
    assert len(ret) > 1000  # set-up: about 1,700 research days
    assert ret.index[-1] <= r.research_end.tz_localize(None)
    holdout_first_day = (r.research_end + pd.Timedelta(days=1)).floor("1D").tz_localize(None)
    assert ret.index[-1] < holdout_first_day


def test_compounding_the_file_reproduces_the_sheets_printed_total_return(study):
    _, sheet = study
    ret = _read(returns_path(sheet))
    printed = re.search(r"^Total return over the window: ([+-][\d.]+)% \((\S+) to (\S+)\)", sheet.read_text(), re.M)
    assert printed, "the sheet prints its total return"
    compounded = math.prod(1 + v for v in ret) - 1
    assert f"{compounded * 100:+.2f}" == printed.group(1)
    assert f"{ret.index[-1]:%Y-%m-%d}" == printed.group(3)


def test_a_series_running_past_the_research_period_is_refused_and_nothing_written(tmp_path):
    idx = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    r = SimpleNamespace(full_period=SimpleNamespace(equity=pd.Series([100.0, 101, 102, 103, 104], index=idx)),
                        research_end=idx[3])  # the 5th day is the holdout's
    with pytest.raises(ValueError, match="holdout"):
        write_returns(r, tmp_path / "s.md")
    assert not (tmp_path / "s.returns.csv").exists()


def test_the_file_holds_each_days_return_exactly(tmp_path):
    idx = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    r = SimpleNamespace(full_period=SimpleNamespace(equity=pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)),
                        research_end=idx[-1])
    write_returns(r, tmp_path / "s.md")
    want = [110.0 / 100.0 - 1, 99.0 / 110.0 - 1, 0.0]  # repr round-trips: the file holds the float itself
    assert (tmp_path / "s.returns.csv").read_text() == ("date,return\n" + "".join(
        f"2026-01-0{d},{w!r}\n" for d, w in zip((2, 3, 4), want)))


# The summary
SHEET = """# Tear sheet: {strategy}

Tested on `{inst}` at 1440-minute bars on `BINANCE`

G1 rules: 2026-10-13

Dataset `real` · research period 01 Jan 2021 to 31 Dec 2025 · holdout: last 365 days ({opened}) · fees: 0.020% maker, 0.055% taker

## G1 checks

**G1: {g1}**

| Check | Result | Evidence |
| --- | --- | --- |
| Runs complete enough to judge | {run} | x |
| G1 test: out-of-sample Sharpe clearly beats benchmark after fees | {sharpe_check} | Sharpe {oos} vs 0.50 |
| Enough out-of-sample trades to judge | PASS | {trades} closed in the 4 walk-forward test windows |

## Results after fees

| Period | Strategy CAGR | Benchmark CAGR | Strategy Sharpe | Benchmark Sharpe | Strategy max DD | Benchmark max DD |
| --- | --- | --- | --- | --- | --- | --- |
| Walk-forward out-of-sample (4 folds, 1460 days) | {cagr} | +20.0% | {oos} | 0.50 | -20.0% | -50.0% |

**Profitable at double fees:** {double}

## Risk, trades and costs

Total return over the window: {total} (2021-01-01 to 2021-01-06), from the daily closes.

## Idea counter

{budget}{dsr}
"""

DAYS = pd.date_range("2021-01-02", periods=5, freq="1D")


def _put(d, name, g1="PASS", oos="1.20", dsr="80", returns=(0.01, -0.02, 0.03, 0.0, 0.01), days=DAYS, inst="BTC/USDT",
         total="+2.97%", cagr="+31.0%", trades="120", double="PASS, net +3.1% over the out-of-sample period at 2x the "
         "venue's fees (re-run)", file=True, tries=4, opened=False, strategy=None):
    run = "NOT JUDGED" if g1 == "NOT JUDGED" else "PASS"
    sharpe_check = "FAIL" if g1 == "FAIL" else "PASS"
    sheet = d / f"{name}.md"
    dsr = ("- Deflated Sharpe: can't be computed here" if dsr is None else
           f"- Deflated Sharpe: {dsr}% probability the out-of-sample Sharpe is real rather than the best of many tries "
           "(higher is better; 95% is a strong bar).")
    budget = "" if tries is None else f"- Budget: {tries} of 20 configurations used for this idea family.\n"
    sheet.write_text(SHEET.format(strategy=strategy or name, inst=inst, g1=g1, run=run, sharpe_check=sharpe_check,
                                  oos=oos, dsr=dsr, budget=budget, opened="opened" if opened else "untouched",
                                  total=total, cagr=cagr, trades=trades, double=double), encoding="utf-8")
    if file:
        returns_path(sheet).write_text("date,return\n" + "".join(f"{t:%Y-%m-%d},{v!r}\n" for t, v in
                                                                 zip(days, returns)))
    return sheet


def _names(sections):
    return {k: [r["name"] for r in v] for k, v in sections.items()}


def test_within_a_section_pass_then_not_judged_then_fail_by_deflated_sharpe_ties_by_oos_sharpe(tmp_path):
    sheets = [_put(tmp_path, "fail_hi", g1="FAIL", oos="2.50", dsr="99"),
              _put(tmp_path, "nj", g1="NOT JUDGED", oos="1.90", dsr="98"),
              _put(tmp_path, "pass_lucky", oos="2.40", dsr="55"),  # the best raw Sharpe, the least likely real
              _put(tmp_path, "pass_tie_lo", oos="1.30", dsr="70"),
              _put(tmp_path, "pass_tie_hi", oos="1.40", dsr="70"),
              _put(tmp_path, "pass_zero", oos="0.10", dsr="0"),  # a 0% deflated Sharpe is still a tracked figure
              _put(tmp_path, "pass_no_dsr", oos="3.00", dsr=None),
              _put(tmp_path, "pass_untracked", oos="2.90", dsr="99", tries=None)]  # no count: no honest deflation
    assert _names(fs.rows(sheets)) == {"BTC only": ["pass_tie_hi", "pass_tie_lo", "pass_lucky", "pass_zero",
                                                    "pass_no_dsr", "pass_untracked", "nj", "fail_hi"]}


def test_a_row_whose_tries_are_not_tracked_says_so_in_its_g1_cell(tmp_path):
    sheets = [_put(tmp_path, "p", tries=None), _put(tmp_path, "nj", g1="NOT JUDGED", tries=None)]
    page = fs.page(sheets)
    assert "| PASS (tries not tracked) |" in page and "| *NOT JUDGED: x; tries not tracked* |" in page


def test_among_passes_a_clean_holdout_ranks_above_a_spent_or_leaked_one_whatever_its_sharpe(tmp_path):
    sheets = [_put(tmp_path, "spent_by_sheet", dsr="99", opened=True),
              _put(tmp_path, "leaked", dsr="98", strategy="multi_speed"),
              _put(tmp_path, "clean", dsr="60"),
              _put(tmp_path, "fail_leaked", g1="FAIL", dsr="10", strategy="multi_speed")]
    sections = fs.rows(sheets, {"multi_speed": _mark("leaked"), "spent_by_sheet": _mark("leaked")})
    assert _names(sections) == {"BTC only": ["clean", "spent_by_sheet", "leaked", "fail_leaked"]}
    got = {r["name"]: r["holdout"] for r in sections["BTC only"]}
    # the sheet says opened: spent; a mark can only make it worse (leaked), never clean
    assert got == {"clean": "clean", "spent_by_sheet": "leaked", "leaked": "leaked", "fail_leaked": "leaked"}
    assert fs.rows([sheets[0]])["BTC only"][0]["holdout"] == "spent"
    with pytest.raises(ValueError, match="never"):
        fs.rows(sheets, {"multi_speed": _mark("clean")})
    for half in ({"status": "leaked", "reason": "seen"}, {"status": "leaked", "date": "2026-10-21", "reason": " "}):
        with pytest.raises(ValueError, match="date .* and its reason"):
            fs.rows(sheets, {"multi_speed": half})


def _mark(status):
    return {"status": status, "date": "2026-10-21", "reason": "holdout year read before the lock"}


def test_the_page_names_the_holdout_file_and_says_the_total_includes_in_sample(tmp_path):
    sheets = [_put(tmp_path, "a")]
    status = tmp_path / "holdout-status.json"
    status.write_text('{"a": {"status": "spent", "date": "2026-10-21", "reason": "opened by hand"}}')
    out = tmp_path / "summary.md"
    fs.main([str(sheets[0]), "--out", str(out), "--holdout-status", str(status)])
    page = out.read_text()
    assert f"spent or leaked as recorded in `{status}`" in page and "| spent |" in page
    assert "| Total return over the window (includes in-sample) |" in page
    assert "Holdout status: from each sheet only." in fs.page(sheets)


def test_pooled_and_each_asset_are_sections_never_ranked_against_each_other(tmp_path):
    sheets = [_put(tmp_path, "eth", inst="ETH/USDT", dsr="99"), _put(tmp_path, "btc", dsr="50"),
              _put(tmp_path, "pool", inst="BTC/USDT+ETH/USDT+SOL/USDT+XRP/USDT", dsr="40")]
    assert _names(fs.rows(sheets)) == {"Pooled (one try)": ["pool"], "BTC only": ["btc"], "ETH only": ["eth"]}
    page = fs.page(sheets)
    assert page.index("## Pooled (one try)") < page.index("## BTC only") < page.index("## ETH only")
    assert sum(line.startswith("| 1 | [") for line in page.splitlines()) == 3  # each section ranks from 1


def test_the_page_shows_each_sheets_own_strings(tmp_path):
    sheets = [_put(tmp_path, "a", oos="1.25", dsr="81", total="+12.34%", cagr="+27.5%", trades="140", tries=7),
              _put(tmp_path, "b", g1="FAIL", oos="0.40", dsr="12", total="-3.21%", cagr="-4.0%", trades="9",
                   double="FAIL, net -1.2% over the out-of-sample period at 2x the venue's fees (x)", tries=12)]
    page = fs.page(sheets)
    row_a = next(line for line in page.splitlines() if line.startswith("| 1 |"))
    row_b = next(line for line in page.splitlines() if line.startswith("| 2 |"))
    for want in ("[a](a.md)", "BTC/USDT", "PASS", "clean", "1.25", "0.50", "+27.5%", "+12.34%", "81%", "PASS, +3.1%",
                 "7 of 20", "140"):
        assert f" {want} |" in row_a, want
    for want in ("[b](b.md)", "FAIL", "0.40", "-4.0%", "-3.21%", "12%", "FAIL, -1.2%", "12 of 20", "9"):
        assert f" {want} |" in row_b, want
    # string for string: every figure is a substring of its own sheet (the 2x-fee and tries cells are re-joined)
    for row, sheet in ((row_a, sheets[0]), (row_b, sheets[1])):
        text = sheet.read_text()
        for cell in [c.strip() for c in row.strip("|").split("|")][2:]:
            if cell.startswith(("PASS, ", "FAIL, ")):
                verdict, net = cell.split(", ")
                assert f"{verdict}, net {net} over" in text
            elif cell.endswith(" of 20"):
                assert f"- Budget: {cell} configurations used" in text
            elif cell not in ("clean", "n/a"):
                assert cell in text, cell


def test_a_not_judged_row_is_muted_and_says_why(tmp_path):
    sheets = [_put(tmp_path, "nj", g1="NOT JUDGED", oos="1.90", dsr="98")]
    row = next(line for line in fs.page(sheets).splitlines() if line.startswith("| 1 |"))
    assert "| *NOT JUDGED: x* |" in row  # the reason is the not-judged check's evidence
    assert "| *1.90* |" in row and "| *98%* |" in row and "| 1.90 |" not in row


def test_the_window_total_is_last_close_over_first(tmp_path):
    from sleeve_fund.research.analysis_pack import window_total

    eq = pd.Series([100.0, 125.0, 80.0, 110.0], index=pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"))
    assert window_total(eq) == pytest.approx(0.10)


def test_the_sheet_prints_its_familys_tries_against_the_budget(study, tmp_path):
    import dataclasses

    from sleeve_fund.research import tearsheet
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.store import Store

    r, _ = study
    register = TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))
    for h, ds in (("d1", "syn"), ("d2", "syn"), ("d3", "syn"), ("d3", "other")):  # 3 configurations, 4 variants
        register.record(definition_hash=h, idea_hash="fam", name="x", family="trend", settings={"h": h},
                        dataset=ds, stage="in_sample", source="backtest", sharpe=0.2)
    register.record(definition_hash="o1", idea_hash="other", name="y", family="trend", settings={}, dataset="syn",
                    stage="in_sample", source="backtest", sharpe=0.2)
    sheet = tearsheet.render(dataclasses.replace(r, idea_hash="fam"), IdeaLedger(tmp_path / "l.jsonl"), register)
    assert register.counts("fam")["variants"] == 4  # set-up: the variant count differs from the configurations
    assert "- Budget: 3 of 20 configurations used for this idea family" in sheet


def test_the_correlation_is_over_the_common_days_only(tmp_path):
    a = (0.01, -0.02, 0.03, 0.0, 0.01)
    b = (0.02, -0.01, 0.01, 0.02, -0.03)  # its days start one later: 4 in common with a
    sheets = [_put(tmp_path, "a", returns=a),
              _put(tmp_path, "b", returns=b, days=pd.date_range("2021-01-03", periods=5, freq="1D"))]
    x, y = a[1:], b[:4]  # 3, 4, 5, 6 Jan
    mx, my = sum(x) / 4, sum(y) / 4
    want = (sum((p - mx) * (q - my) for p, q in zip(x, y))
            / math.sqrt(sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)))
    corr, common = fs.correlation(sheets)
    assert (common.min(), common.max(), len(common)) == (pd.Timestamp("2021-01-03"), pd.Timestamp("2021-01-06"), 4)
    assert corr.loc["a", "b"] == pytest.approx(want)
    page = fs.page(sheets)
    assert f"| 1 a | *1.00* | *{want:.2f}* |" in page and "4 common days, 2021-01-03 to 2021-01-06" in page
    assert "**Too short to read:** under 90 days" in page


@pytest.mark.parametrize("days, short", [(89, True), (90, False)])
def test_the_matrix_is_greyed_under_90_common_days(tmp_path, days, short):
    a = [0.01 * ((i * 7) % 5 - 2) for i in range(days)]
    b = [0.01 * ((i * 3) % 4 - 1.5) for i in range(days)]
    idx = pd.date_range("2021-01-02", periods=days, freq="1D")
    page = fs.page([_put(tmp_path, "a", returns=a, days=idx), _put(tmp_path, "b", returns=b, days=idx)])
    assert ("Too short to read" in page) is short
    assert ("| 1 a | *1.00* |" in page) is short and ("| 1 a | 1.00 |" in page) is not short


def test_a_sheet_without_its_returns_file_is_refused(tmp_path):
    sheets = [_put(tmp_path, "a"), _put(tmp_path, "b", file=False)]
    with pytest.raises(fs.Unmatched, match=r"b\.md.*no returns file"):
        fs.page(sheets)


def test_a_returns_file_without_its_sheet_is_refused(tmp_path):
    sheets = [_put(tmp_path, "a")]
    (tmp_path / "gone.returns.csv").write_text("date,return\n2021-01-02,0.01\n")
    with pytest.raises(fs.Unmatched, match=r"gone\.returns\.csv.*no tear sheet"):
        fs.page(sheets)


def test_the_command_writes_the_page_and_prints_counts_only(tmp_path, capfd):
    sheets = [_put(tmp_path, "a"), _put(tmp_path, "b", oos="0.80")]
    out = tmp_path / "summary.md"
    assert fs.main([str(s) for s in sheets] + ["--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == fs.page(sheets)
    said = capfd.readouterr().out
    assert said == f"fund summary: 2 strategies, 5 common days: {out}\n"
