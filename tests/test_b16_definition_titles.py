"""B-16 (HoE 20 Oct): each QR definition carries a display `title`, which the dashboard shows, and adding it changes
no definition_hash or idea_hash, so no built Strategy's trials count or history resets. The hashes below are main's
(274ecd88) before any title existed."""

from pathlib import Path

import pytest

from sleeve_fund.research.candidates import idea_hash
from sleeve_fund.strategies.definitions import check_definition, definition_hash, load_definition_file, to_params
from sleeve_fund.venues import VENUES

DEFS = Path(__file__).resolve().parent.parent / "configs" / "definitions"

# file id: (definition_hash, idea_hash) on main, before the titles
BEFORE = {
    "qr01_trend_ema_long": ("db887a6fa3a01f1b", "4a7eb5ddd831d115"),
    "qr01_trend_ema_short": ("8ff28a74df7b0372", "ae95f6b39c909bfd"),
    "qr01b_trend_multispeed_long": ("e122ddf1851edf73", "de5a829d3c0285d2"),
    "qr01b_trend_multispeed_short": ("ebeb628bca57ab35", "130dd41dd09d7583"),
    "qr01c_trend_donchian_long": ("a64268b388237a66", "755f51c7b13eea57"),
    "qr01c_trend_donchian_short": ("170c176d9440913e", "c467a46247a2def8"),
    "qr02_dip_long": ("ce126dc7f0c4275c", "0e13cf32e1775235"),
    "qr02_dip_short": ("d6a770a1cc522b77", "11201be8f25dd86e"),
    "qr03_squeeze_long": ("95545de0d400e19f", "7c296e19aa018b1c"),
    "qr03_squeeze_short": ("8bb0e50ac6fd3c0f", "da6a2f9869be5fdd"),
    "qr10_capitulation_long": ("62a1ee8da26544c1", "6115b1bf66edcbc1"),
    "qr11_exhaustion_long": ("8ad588bcd437b26a", "32783815f699c062"),
    "qr11_exhaustion_short": ("daa1bea4ae612937", "d7f2ac6ecd8cd817"),
    "qr12_overnight_breakout_long": ("aa3a666c41d9381c", "4a60499278cd6443"),
    "qr13_level_continuation_long": ("18e1ac88db4b05b8", "f1f453cf366a222e"),
    "qr13_level_continuation_short": ("8ee488792d38fdd4", "cd26d663530515cb"),
}


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
