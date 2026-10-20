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
