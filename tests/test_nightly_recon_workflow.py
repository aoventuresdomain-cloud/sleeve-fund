"""NIGHTLY-RECON's two gates (HoE, 15 Oct): (1) a by-hand read-only run of the night against the server journal
(.github/workflows/nightly-recon.yml), its lines in the job log, nothing written, no alert; (2) a scheduled night's FAIL
reaches the PM only once the repository variable NIGHTLY_ALERT is on (deploy.yml -> compose -> alerts.Forwarder), off by
default."""

from __future__ import annotations

import pathlib

import yaml
from test_nightly_recon import NIGHT, _events, _forwarder

from sleeve_fund.store import Store
from tests.test_start_check import _journal

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "nightly-recon.yml"


def _wf():
    return yaml.safe_load(WF.read_text())


def test_the_by_hand_run_is_read_only_by_default_and_refuses_anything_else():
    wf = _wf()
    inp = wf[True]["workflow_dispatch"]["inputs"]  # YAML 1.1 reads "on" as true
    assert list(inp) == ["read_only"] and inp["read_only"]["default"] is True and inp["read_only"]["type"] == "boolean"
    cond = wf["jobs"]["recon"]["if"]
    for gate in ("vars.DEPLOY_ENABLED == 'true'", "github.ref == 'refs/heads/main'", "inputs.read_only"):
        assert gate in cond
    assert wf[True].keys() == {"workflow_dispatch"}  # never on a schedule or a push


def test_it_runs_the_read_only_night_into_a_scratch_folder_and_no_input_reaches_a_run_line():
    steps = _wf()["jobs"]["recon"]["steps"]
    assert [s["name"] for s in steps] == ["Configure SSH", "Run the night read-only"]
    line = steps[1]["run"]
    assert "python -m sleeve_fund.nightly_recon --dir /tmp/recon-preview" in line
    assert "/data/recon" not in line and "LogLevel=QUIET" in line
    for s in steps:
        assert "${{" not in s["run"], s["name"]  # inputs, secrets and vars reach a shell line through env only


def test_deploy_passes_nightly_alert_to_the_supervisor_off_by_default():
    deploy = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert "NIGHTLY_ALERT: ${{ vars.NIGHTLY_ALERT || 'off' }}" in deploy
    assert "NIGHTLY_ALERT=%q" in deploy and '"$NIGHTLY_ALERT"' in deploy and ",NIGHTLY_ALERT," in deploy
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert compose["services"]["supervisor"]["environment"]["NIGHTLY_ALERT"] == "${NIGHTLY_ALERT:-off}"


def test_with_the_alert_off_a_fail_is_journaled_as_info_and_sent_to_no_one(tmp_path):
    url, _ = _journal(tmp_path, mark_cash=3_990.0)
    sent = []
    from sleeve_fund.alerts import Forwarder

    fwd = Forwarder(Store(url), environ={"RECON_DIR": str(tmp_path / "recon"), "ALERT_WEBHOOK_URL": "https://h.example/x"},
                    send=lambda u, text: sent.append(text), now=lambda: NIGHT)
    fwd.step()
    fwd.step()
    [ev] = _events(url, "recon_failed")
    assert ev.level == "info" and ev.message.endswith("(not sent: NIGHTLY_ALERT is off)")
    assert not any("nightly recon" in t for t in sent)
    assert (tmp_path / "recon" / f"recon-{NIGHT:%Y-%m-%d}.txt").read_text().startswith("FAIL\n")


def test_with_the_alert_on_a_fail_is_a_warning_and_is_sent(tmp_path):
    url, _ = _journal(tmp_path, mark_cash=3_990.0)
    sent, now = [], [NIGHT]
    fwd = _forwarder(url, tmp_path / "recon", now, sent)
    fwd.step()
    fwd.step()
    [ev] = _events(url, "recon_failed")
    assert ev.level == "warning" and "not sent" not in ev.message
    assert any("nightly recon FAILED" in t for t in sent)
