"""B-16: the Strategy area (was Development). The 16 built rule files, the hand-coded Models and the two examples
are listed with their status; each opens with its reason and plain-English rules; a file that fails its check shows
why instead of crashing; Run is refused while research is on hold; and no page writes anything or acts on the fund."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from sleeve_fund.dashboard import strategy_area as sa
from sleeve_fund.research.candidates import idea_hash
from sleeve_fund.research.trials import trial_row
from sleeve_fund.store import Store, metadata
from sleeve_fund.strategies.definitions import definition_hash, load_definition_file

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
DEFS = Path(__file__).resolve().parents[1] / "configs" / "definitions"
QR_IDS = sorted(load_definition_file(p)["name"] for p in DEFS.glob("qr*.toml"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    from sleeve_fund.dashboard import app as app_mod

    sheets = tmp_path / "sheets"
    sheets.mkdir()
    monkeypatch.setattr(app_mod, "TEARSHEETS", sheets)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _counts(store) -> dict:
    with store.engine.connect() as c:
        return {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in metadata.sorted_tables}


def test_nav_says_strategy(client):
    c, _ = client
    html = c.get("/", auth=AUTH).text
    assert '<a href="/strategy"' in html and "<span>Strategy</span>" in html
    assert "<span>Development</span>" not in html


def test_list_holds_every_built_strategy(client):
    c, _ = client
    html = c.get("/strategy", auth=AUTH).text
    assert len(QR_IDS) == 16
    for fid in QR_IDS + ["rsi-bands", "rsi-pullback"]:
        assert f"/strategy?s={fid}" in html
    for model in sa.MODELS:
        assert f"/strategy?s=model%3A{model}" in html
    assert html.count(">Built, not tested</span>") >= 16
    assert "Built Strategies" in html and "Models (hand-coded)" in html and "Earlier examples" in html


@pytest.mark.parametrize("fid", QR_IDS)
def test_each_strategy_page(client, fid):
    c, _ = client
    defn = load_definition_file(next(p for p in DEFS.glob("*.toml") if load_definition_file(p)["name"] == fid))
    html = c.get(f"/strategy?s={fid}", auth=AUTH).text
    first_line = defn["reason"].strip().splitlines()[0]
    assert first_line.replace('"', "&#34;").replace("'", "&#39;") in html or first_line in html
    assert "Rules in plain English" in html and "couldn't be put into words" not in html
    assert "Indicators used" in html and "Settings to test" in html and "Show the file" in html
    assert definition_hash(defn)[:7] in html
    for tab in ("test", "history", "runs"):
        assert c.get(f"/strategy?s={fid}&tab={tab}", auth=AUTH).status_code == 200


def test_empty_test_history(client):
    c, _ = client
    html = c.get("/strategy?s=qr02-dip-long&tab=history", auth=AUTH).text
    assert "Not tested yet. Research is on hold until your go." in html and "Tries used 0 of 20" in html
    assert "holdout unused" in html


def test_test_tab_run_is_disabled_and_refused(client):
    c, store = client
    html = c.get("/strategy?s=qr02-dip-long&tab=test", auth=AUTH).text
    assert "Changing the asset list uses a new try." in html
    assert 'disabled title="Research is on hold until your go">Run test</button>' in html
    before = _counts(store)
    r = c.post("/strategy/test", data={"key": "qr02-dip-long", "assets": "x:BTC/USDT"}, auth=AUTH, headers=SAME)
    assert r.status_code == 409 and "on hold" in r.json()["detail"]
    assert _counts(store) == before


def test_refused_file_shows_why(client, tmp_path, monkeypatch):
    c, _ = client
    defs = tmp_path / "defs"
    shutil.copytree(DEFS, defs)
    (defs / "zz_broken.toml").write_text('version = 1\nname = "zz-broken"\nreason = "r"\n[long]\nentry = { left = "nope", '
                                         'op = ">", right = 1 }\n')
    (defs / "zz_unreadable.toml").write_text("this is = = not toml")
    monkeypatch.setattr(sa, "DEFINITIONS", defs)
    html = c.get("/strategy?s=zz-broken", auth=AUTH).text
    assert "This file fails its check" in html and "unknown reference" in html and ">Refused</span>" in html
    assert c.get("/strategy?s=zz-unreadable", auth=AUTH).status_code == 200
    assert c.get("/strategy", auth=AUTH).status_code == 200


def test_unknown_strategy_is_404(client):
    c, _ = client
    assert c.get("/strategy?s=nope", auth=AUTH).status_code == 404


def test_tried_and_on_paper(client):
    """Status follows the store: a counted trial row makes it Tested, with its tries; a Strategy on the fund trading
    the same definition makes it On paper, and Archive is off."""
    c, store = client
    defn = load_definition_file(DEFS / "qr02_dip_long.toml")
    store.add_trials([trial_row(definition_hash=definition_hash(defn), idea_hash=idea_hash(defn), name="qr02-dip-long",
                                family="rules", settings={}, dataset="x-btcusdt-store-240m", stage="in_sample",
                                source="study", sharpe=0.5, trades=40)])
    html = c.get("/strategy?s=qr02-dip-long&tab=history", auth=AUTH).text
    assert ">Tested</span>" in html and "Tries used 1 of 20" in html and "x-btcusdt-store-240m" in html
    store.create_sleeve(name="dip-on-fund", strategy="rules", instrument="BTC/USDT", bar_spec="4-HOUR-LAST-EXTERNAL",
                        starting_balance=1000, params={"definition": defn}, desired_state="stopped")
    html = c.get("/strategy?s=qr02-dip-long", auth=AUTH).text
    assert ">On paper</span>" in html and "Stop and archive it on the fund first" in html
    assert '<details class="sa-confirm">' not in html.split('id="sa-h"')[1].split("sa-tabs")[0]
    runs = c.get("/strategy?s=qr02-dip-long&tab=runs", auth=AUTH).text
    assert "/sleeves/dip-on-fund" in runs


def test_pages_write_nothing_and_files_unchanged(client):
    c, store = client
    files = {p.name: p.read_bytes() for p in DEFS.glob("*.toml")}
    before = _counts(store)
    for url in ("/strategy", "/strategy?s=qr03-squeeze-long", "/strategy?s=qr03-squeeze-long&tab=test",
                "/strategy?s=model:trend_filter", "/strategy/results", "/strategy/coverage", "/strategy/archived"):
        assert c.get(url, auth=AUTH).status_code == 200, url
    assert _counts(store) == before
    assert {p.name: p.read_bytes() for p in DEFS.glob("*.toml")} == files


def test_no_route_acts_on_the_fund(client):
    """The area's routes: GET pages, plus Archive and Restore (the archive row and the decision log only) and the
    refused Run. None is under /sleeves or the supervisor's commands."""
    c, _ = client
    mine = {(r.path, m) for r in c.app.routes if getattr(r, "path", "").startswith("/strategy")
            and not r.path.startswith("/strategies") for m in r.methods}
    assert {p for p, m in mine if m == "POST"} == {"/strategy/archive", "/strategy/restore", "/strategy/test"}
    assert {p for p, m in mine if m == "GET"} == {"/strategy", "/strategy/results", "/strategy/coverage",
                                                   "/strategy/archived"}


def test_pm_only(client):
    c, _ = client
    assert c.get("/strategy").status_code == 401
    assert c.post("/strategy/test", headers=SAME).status_code == 401


def _new(c, name):
    form = {"name": name, "strategy": "trend_filter", "instrument": "BTC/USDT", "venue": "binance", "market": "perp", "bar_spec": "1-HOUR-LAST-INTERNAL",
            "starting_balance": "5000", "risk_profile": "conservative", "warmup_bars": "0", "p_trend_filter__fast": "10",
            "p_trend_filter__slow": "30", "reason": "first test"}
    r = c.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "error" not in r.headers["location"], r.headers["location"]


def test_history_lists_archived_strategies(client):
    """Part 2: a Strategy archived on the fund (its own route) is listed on History, linking to its page; read only."""
    c, store = client
    _new(c, "put-away")
    store.set_desired_state("put-away", "stopped")
    r = c.post("/sleeves/put-away/archive", data={"action": "archive", "reason": "done"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303
    before = _counts(store)
    html = c.get("/history", auth=AUTH).text
    assert "Archived strategies (1)" in html and 'href="/sleeves/put-away"' in html
    assert '<a href="/history"' in html and "<span>History</span>" in html  # in the main menu
    assert _counts(store) == before
    assert c.get("/sleeves/put-away", auth=AUTH).status_code == 200


def test_history_lists_the_previous_book(client, tmp_path):
    from sleeve_fund.supervisor import clear

    c, store = client
    _new(c, "old-one")
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-04"\nreason = "new book"\n')
    clear(store, str(path))
    html = c.get("/history", auth=AUTH).text
    assert "Previous book (1)" in html and 'href="/sleeves/old-one"' in html


def test_history_page_empty(client):
    c, _ = client
    assert "No old Strategies" in c.get("/history", auth=AUTH).text
