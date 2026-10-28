"""The study workflow's runner (sleeve_fund.research.rescore): each row of the list is one G1 study of the `rules`
model on its definition, on the perpetual, with the holdout locked; it writes research records only (never a
Strategy, order, fill or fund row), and its public log prints counts and ran/failed only."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from sqlalchemy import func, select

from sleeve_fund.research import rescore

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qr-rescore-list.md"
WORKFLOW = ROOT / ".github" / "workflows" / "study.yml"


def test_the_list_parses_to_its_twenty_one_rows_with_their_candles():
    rows = rescore.rows_of(LIST.read_text())
    assert [n for n, _, _ in rows] == list(range(1, 22))
    assert rows[0] == (1, "qr01_trend_ema_long", 240) and rows[4][2] == 1440 and rows[11][2] == 15
    assert all((rescore.DEFINITIONS / f"{name}.toml").is_file() for _, name, _ in rows)


@pytest.mark.parametrize("spec, want", [("", 21), ("1-8", 8), ("9-16", 8), ("3", 1), ("1,4,7", 3)])
def test_rows_pick(spec, want):
    assert len(rescore.pick(rescore.rows_of(LIST.read_text()), spec)) == want


@pytest.mark.parametrize("spec", ["0", "22", "1-23", "a", "1;rm -rf /", "1-"])
def test_bad_rows_are_refused(spec):
    with pytest.raises(ValueError):
        rescore.pick(rescore.rows_of(LIST.read_text()), spec)


def test_each_row_is_the_rules_model_on_its_definition_on_the_perpetual_with_the_holdout_locked():
    from sleeve_fund.strategies.definitions import load_definition_file, to_params

    for _, name, minutes in rescore.rows_of(LIST.read_text()):
        req = rescore.request(name, minutes, "binance", "BTC/USDT")
        defn = load_definition_file(rescore.DEFINITIONS / f"{name}.toml")
        assert req.strategy == "rules" and req.minutes == minutes and req.label == name
        assert req.params["market"] == "perp" and req.params["definition"] == defn
        assert req.params["definition_hash"] == to_params(defn, "BINANCE")["definition_hash"]
        assert req.params.get("allow_short", False) == ("short" in defn)
        assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
        assert req.risk_profile is None
        req.validate()


def test_the_log_prints_counts_and_ran_or_failed_only(capfd, tmp_path):
    seen = []

    def run(req, store=None):
        seen.append(req)
        if req.label == "qr02_dip_short":
            raise ValueError("Sharpe 1.234 at 0.055% fees, host api.example.com")  # nothing of this may print
        return tmp_path / f"{req.label}_binance-btcusdt-store-15m_20261024-153413.md"

    code = rescore.main([str(LIST), "--rows", "1-8"], run=run, journal=lambda: None)
    out = capfd.readouterr().out
    assert code == 1 and len(seen) == 8 and all(not r.use_holdout for r in seen)
    assert "row  8 qr02_dip_short: FAILED (ValueError), note refused-" in out and "done: 7 ran, 1 failed" in out
    # A row that ran names its sheet's stem, the sheet export's input; a failed row names none.
    stem = "qr01_trend_ema_long_binance-btcusdt-store-15m_20261024-153413"
    assert f"row  1 qr01_trend_ema_long: ran, sheet {stem}\n" in out
    assert "qr02_dip_short_binance" not in out
    assert "1.234" not in out and "0.055" not in out and "example.com" not in out and "Sharpe" not in out


def test_a_dry_run_runs_nothing(capfd):
    called = []
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: called.append(1), journal=lambda: None) == 0
    assert called == [] and "dry run: 21 checked, 0 run" in capfd.readouterr().out


@pytest.mark.parametrize("argv", [["--venue", "Binance;x"], ["--pair", "btc/usdt"], ["--rows", "99"]])
def test_bad_inputs_are_refused_before_anything_runs(argv, capfd):
    called = []
    assert rescore.main([str(LIST), *argv], run=lambda *a, **k: called.append(1), journal=lambda: None) == 2
    assert called == [] and capfd.readouterr().out.startswith("refused:")


def _counts(store) -> dict[str, int]:
    from sleeve_fund.store import metadata

    with store.engine.connect() as c:
        return {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in metadata.sorted_tables}


def test_a_real_study_writes_research_records_only_never_a_strategy_order_fill_or_fund_row(tmp_path, monkeypatch):
    """The guard: a real store study of a list row, on a real journal, changes no table but the research ones."""
    from test_binance import INFO
    from test_research import _stored_minutes

    from sleeve_fund import funding
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research.run import run_store_study
    from sleeve_fund.store import Store
    from sleeve_fund.venues import binance_contract, venue

    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    store = Store(f"sqlite:///{tmp_path}/j.db")
    store.create_sleeve(name="timed-test", strategy="candle_step", instrument="BTC/USDT",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, desired_state="stopped",
                        venue="binance")
    hist = HistoryStore(tmp_path / "hist")
    hist.append("BINANCE", "BTC/USDT", _stored_minutes(130), cursor="x")
    before = _counts(store)
    req = dataclasses.replace(rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT"),
                              train_days=60, test_days=30, holdout_days=30)  # the list's windows need years of bars
    sheet = run_store_study(req, store=store, ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path / "ts", history=hist)
    assert sheet.name.startswith("qr01_trend_ema_long_binance-btcusdt-store-240m_")
    after = _counts(store)
    changed = {t for t in before if before[t] != after[t]}
    assert changed and changed <= {"trials", "trial_runs", "research_trials", "holdout_locks"}, changed
    for table in ("sleeves", "orders", "fills", "equity", "fund_flows", "events", "gate_decisions", "demo_mirror"):
        if table in before:
            assert before[table] == after[table], table


def test_the_workflow_is_dispatch_only_never_opens_the_holdout_and_runs_the_list_in_a_one_off_container():
    text = WORKFLOW.read_text()
    on = text.split("\non:", 1)[1].split("\njobs:", 1)[0]
    assert "workflow_dispatch:" in on and "push" not in on and "schedule" not in on and "pull_request" not in on
    assert "use-holdout" not in text and "use_holdout" not in text
    assert "python -u -m sleeve_fund.research.rescore" in text and "--rm" in text
    assert "github.ref == 'refs/heads/main'" in text and "DEPLOY_ENABLED" in text


def test_the_default_venue_is_the_research_venue_as_before(monkeypatch):
    """The --venue default comes from the registry (no venue named in code) and resolves to the venue it always was,
    so a study run without --venue reads the same stored history as before."""
    seen, real = [], rescore.request
    monkeypatch.setattr(rescore, "request", lambda name, minutes, venue, pair, *a: seen.append(venue) or real(
        name, minutes, venue, pair, *a))
    rescore.main([str(LIST), "--rows", "1", "--dry-run"], run=lambda *a, **k: None, journal=lambda: None)
    assert seen == ["binance"]


def test_a_failed_rows_message_and_traceback_go_to_a_note_the_sheet_export_reads_never_the_log(capfd, tmp_path,
                                                                                                monkeypatch):
    monkeypatch.setattr(rescore, "REFUSED", tmp_path / "refused")

    def run(req, store=None):
        raise ValueError("refused: the legs held 1.37x the fund's equity at once")

    assert rescore.main([str(LIST), "--rows", "1"], run=run, journal=lambda: None) == 1
    out = capfd.readouterr().out
    (note,) = (tmp_path / "refused").glob("refused-*-row1.md")
    assert f"row  1 qr01_trend_ema_long: FAILED (ValueError), note {note.stem}\n" in out
    assert "1.37" not in out and "legs held" not in out
    text = note.read_text()
    assert "ValueError: refused: the legs held 1.37x the fund's equity at once" in text and "Traceback" in text
    assert "qr-rescore-list.md" in text and "BTC/USDT" in text
    export = (ROOT / ".github" / "workflows" / "sheet-export.yml").read_text()
    assert "refused-*) cd refused" in export


def test_a_note_that_cant_be_written_never_hides_the_rows_failure(capfd, tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("")
    monkeypatch.setattr(rescore, "REFUSED", blocked / "refused")  # under a file: mkdir fails
    assert rescore.main([str(LIST), "--rows", "1"], run=lambda *a, **k: (_ for _ in ()).throw(ValueError("x")),
                        journal=lambda: None) == 1
    assert "row  1 qr01_trend_ema_long: FAILED (ValueError)\n" in capfd.readouterr().out
