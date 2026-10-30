"""M1 items 9 and 11 (PM 30 Oct, next strategy round): every test window's trades and every basket leg's beside the
sheet, each trade with why it opened and closed and the values its rule read, and scripts/recompute_sheet.py, which
rebuilds the sheet's figures from those files alone (accuracy check 2)."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

import pytest

from sleeve_fund.research import periods
from sleeve_fund.research.run import BESIDE, write_beside, write_oos_returns, write_results
from test_r04_short_leg_sheet import LONG, _study, perp_venue  # noqa: F401
from test_study_asset_list import ran  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recompute_sheet.py"


def _recompute(sheet: Path) -> subprocess.CompletedProcess:
    """The script as anyone would run it: isolated (-I), so it can't import the engine even by accident."""
    return subprocess.run([sys.executable, "-I", str(SCRIPT), str(sheet)], capture_output=True, text=True, timeout=120)


@pytest.fixture
def written(tmp_path, perp_venue):
    r, _, _ = _study(tmp_path, perp_venue, LONG)
    sheet = tmp_path / "sheets" / "qr_x.md"
    sheet.parent.mkdir()
    sheet.write_text("# Tear sheet\n")
    write_results(r, sheet)
    write_oos_returns(r, sheet)
    write_beside(r, sheet)
    return r, sheet


def _csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text())))


def test_every_test_windows_trades_sit_beside_the_sheet_and_count_as_the_sheet_does(written):
    r, sheet = written
    rows = _csv(sheet.with_name("qr_x-oos-trades.csv"))
    assert list(rows[0])[:4] == ["window", "window_start", "window_end", "counted"]
    traded = [f for f in r.folds if f.trade_from is not None]
    assert traded and {int(x["window"]) for x in rows} <= set(range(1, len(traded) + 1))
    assert sum(x["counted"] == "yes" for x in rows) == r.oos_trades > 0
    for k, f in enumerate(traded, 1):  # each window's own run: its trades sum to its change, to the cent
        mine = [x for x in rows if int(x["window"]) == k]
        assert sum(x["counted"] == "yes" for x in mine) == f.test_trades
        if mine:
            pnl = sum(float(x["pnl"]) for x in mine)
            start = float(mine[-1]["equity_after"]) - pnl
            assert pnl / start == pytest.approx(f.test["total_return"], abs=1e-9)


def test_each_trade_says_why_it_opened_and_closed_and_what_its_rule_read(written):
    _, sheet = written
    for name in ("qr_x-trades.csv", "qr_x-oos-trades.csv"):
        rows = _csv(sheet.with_name(name))
        closed = [x for x in rows if x["exit_time"]]
        assert closed, name
        for x in closed:
            assert x["entry_rule"] == "long.entry" and x["exit_rule"] == "long.exit", (name, x)
            assert "fast=" in x["entry_signal_values"] and "slow=" in x["entry_signal_values"]
            assert "fast=" in x["exit_signal_values"]
    assert periods.CSV_COLUMNS[:17] == (  # the columns before stay where the PM's book reads them
        "trade", "side", "entry_time", "entry_price", "exit_time", "exit_price", "size", "notional", "holding_hours",
        "fees", "funding", "pnl", "pnl_pct_of_equity", "return_on_notional", "exit_reason", "equity_after",
        "drawdown_at_trade_close")


def test_a_trade_without_a_rule_leaves_its_reasons_blank():
    assert periods.signal_values(None) == "" and periods.signal_values({"signal": {"reduce_only": True}}) == ""
    assert periods.signal_values({"signal": {"blocks": {"b": 2.0, "a": 1.5}}}) == "a=1.5; b=2.0"
    assert periods._cell("a=1.5; b=2.0") == "a=1.5; b=2.0" and periods._cell('x,"y"') == '"x,""y"""'


def test_the_recompute_rebuilds_a_studys_sheet_from_its_files_alone(written):
    _, sheet = written
    got = _recompute(sheet)
    assert got.returncode == 0, got.stdout + got.stderr
    assert "0 mismatch(es)" in got.stdout and "MISMATCH" not in got.stdout
    for what in ("full period: closed trades", "out of sample: counted trades", "out of sample: worst drop",
                 "out of sample: Sharpe", "test window 1 "):
        assert what in got.stdout


def test_the_recompute_catches_a_trade_that_doesnt_add_up(written):
    _, sheet = written
    path = sheet.with_name("qr_x-oos-trades.csv")
    lines = path.read_text().splitlines()
    head = lines[0].split(",")
    cells = lines[1].split(",")
    cells[head.index("pnl")] = repr(float(cells[head.index("pnl")]) + 5.0)
    path.write_text("\n".join([lines[0], ",".join(cells), *lines[2:]]) + "\n")
    got = _recompute(sheet)
    assert got.returncode == 1 and "MISMATCH test window 1" in got.stdout


def test_the_recompute_says_so_when_a_sheet_has_no_files():
    got = subprocess.run([sys.executable, "-I", str(SCRIPT), "/nonexistent/old.md"], capture_output=True, text=True)
    assert got.returncode == 2 and "MISSING" in got.stdout


def test_a_baskets_legs_and_pooled_days_sit_beside_its_sheet_and_recompute(ran):  # noqa: F811
    out, _, _, _ = ran
    trades = _csv(out.with_name(f"{out.stem}-basket-trades.csv"))
    assert {x["asset"] for x in trades} <= {"BTC", "ETH", "SOL"} and trades
    assert {x["test_day"] for x in trades} <= {"yes", "no"}
    pooled = _csv(out.with_name(f"{out.stem}-pooled.csv"))
    assert list(pooled[0]) == ["date", "pnl_BTC", "pnl_ETH", "pnl_SOL", "equity_before", "pooled_return"]
    got = _recompute(out)
    assert got.returncode == 0, got.stdout + got.stderr
    assert "pooled: net" in got.stdout and "0 mismatch(es)" in got.stdout


def test_the_dashboard_may_serve_each_new_file_by_its_exact_name():
    assert set(BESIDE) == {"trades.csv", "equity.csv", "equity.svg", "oos-trades.csv", "basket-trades.csv",
                           "pooled.csv"}
