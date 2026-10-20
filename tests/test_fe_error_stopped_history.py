"""HoQA NOTE 2 (HoE, 20 Oct), on the D3 pages: a Strategy left in error that the PM has stopped, flat, reads as stopped
on History's Test accounts list too, as it does everywhere else (fe-error-stopped). Display only."""

from test_ccy_8dp_ui import AUTH, _client


def test_stopped_after_an_error_reads_stopped_on_historys_test_accounts(tmp_path, monkeypatch):
    c, store = _client(tmp_path, monkeypatch)
    store.set_status("tst", "error", "node exited 1")
    store.set_desired_state("tst", "stopped")
    group = c.get("/history", auth=AUTH).text.split('id="test-accounts"')[1].split("</details>")[0]
    assert 'class="chip stopped">Stopped' in group and "chip error" not in group, group
    assert store.sleeve("tst").status == "error"  # the stored status is untouched


def test_stopped_after_an_error_reads_stopped_on_the_strategy_areas_runs_tab(tmp_path, monkeypatch):
    """CR on 0a5e5a36: the runs-on table on the Strategy area reads the shown status too."""
    from test_b16_strategy_area import DEFS

    from sleeve_fund.strategies.definitions import load_definition_file

    c, store = _client(tmp_path, monkeypatch)
    defn = load_definition_file(DEFS / "qr02_dip_long.toml")
    store.create_sleeve(name="dip-on-fund", strategy="rules", instrument="BTC/USDT", bar_spec="4-HOUR-LAST-EXTERNAL",
                        starting_balance=1000, params={"definition": defn}, desired_state="stopped")
    store.set_status("dip-on-fund", "error", "node exited 1")
    runs = c.get("/strategy?s=qr02-dip-long&tab=runs", auth=AUTH).text
    row = runs.split('<td><a href="/sleeves/dip-on-fund"')[1].split("</tr>")[0]
    assert 'class="chip stopped">Stopped' in row and "chip error" not in row, row
