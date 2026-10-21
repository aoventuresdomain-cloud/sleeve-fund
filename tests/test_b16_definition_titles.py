"""B-16 (HoE 20 Oct): each QR definition carries a display `title`, which the dashboard shows, and adding it changes
no definition_hash or idea_hash, so no built Strategy's trials count or history resets. The hashes pinned in
configs/definition_pins.toml are main's (274ecd88) before any title existed, and each later definition's own."""

import tomllib
from pathlib import Path

import pytest

from sleeve_fund.research.candidates import idea_hash
from sleeve_fund.strategies.definitions import check_definition, definition_hash, load_definition_file, to_params
from sleeve_fund.venues import VENUES

DEFS = Path(__file__).resolve().parent.parent / "configs" / "definitions"

# file id: (definition_hash, idea_hash) on main, before the titles, and every definition since (R-29: one data file)
PINS = tomllib.loads((DEFS.parent / "definition_pins.toml").read_text())
BEFORE = {fid: (pin["definition_hash"], pin["idea_hash"]) for fid, pin in PINS.items()}


def test_every_qr_file_is_pinned():
    assert sorted(p.stem for p in DEFS.glob("qr*.toml")) == sorted(BEFORE)


@pytest.mark.parametrize("fid", sorted(BEFORE))
def test_the_title_changes_no_hash(fid):
    defn = load_definition_file(DEFS / f"{fid}.toml")
    assert isinstance(defn.get("title"), str) and defn["title"].strip()  # every QR file has its display name
    check_definition(defn)
    assert (definition_hash(defn), idea_hash(defn)) == BEFORE[fid]
    untitled = {k: v for k, v in defn.items() if k != "title"}
    for venue in (None, *VENUES):  # the venue's day anchor joins the hash; the title never does
        assert definition_hash(defn, venue) == definition_hash(untitled, venue)
    assert to_params(defn)["definition_hash"] == BEFORE[fid][0]


def test_a_retitle_is_the_same_variant():
    defn = load_definition_file(DEFS / "qr02_dip_long.toml")
    assert definition_hash({**defn, "title": "Something else entirely"}) == definition_hash(defn)


@pytest.mark.parametrize("bad", ["", "   ", 7, ["a"]])
def test_a_title_is_a_line_of_text(bad):
    defn = {**load_definition_file(DEFS / "qr02_dip_long.toml"), "title": bad}
    with pytest.raises(ValueError, match="title"):
        check_definition(defn)


def test_a_rules_strategys_params_keys_ignore_the_title_too():
    """CR B-16 follow-up: a rules Strategy's params carry its whole definition (to_params), and the run, study,
    portfolio-member and failed-run keys hash those params; the title stays out of them as well. Params with no
    titled definition hash as before."""
    from sleeve_fund.research.trials import failed_row, legacy_definition_hash

    defn = load_definition_file(DEFS / "qr02_dip_long.toml")
    titled = to_params(defn)
    untitled = to_params({k: v for k, v in defn.items() if k != "title"})
    assert titled != untitled
    setup = {"risk_profile": "balanced", "windows": None, "fee": 0.0006}
    for s in (None, setup):
        assert legacy_definition_hash("rules", titled, s) == legacy_definition_hash("rules", untitled, s)
    assert (failed_row(strategy="rules", params=titled, source="backtest", error="x")["definition_hash"]
            == failed_row(strategy="rules", params=untitled, source="backtest", error="x")["definition_hash"])
    plain = {"fast": 10, "slow": 30}
    assert legacy_definition_hash("trend_filter", plain) == legacy_definition_hash("trend_filter", dict(plain))
