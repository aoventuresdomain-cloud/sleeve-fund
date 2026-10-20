"""B-22: the tear sheet's 'profitable at double fees' row beside the break-even fee. The re-run itself is research code
(a real run at exactly 2x the venue's maker and taker fees, same out-of-sample period, never the holdout); the page
only shows what the sheet stored: PASS or FAIL with the net figure, or 'not run' when the sheet has none. Built
against the line the sheet is expected to carry until the research side's field names land (STUB_LINE)."""

import pytest
from test_development import (  # noqa: F401  (the fixtures' sheet writer and the app client)
    AUTH,
    SAME,
    _sheet,
    client,
)

from sleeve_fund.dashboard import development as dev

STUB_LINE = ("**Profitable at double fees:** {verdict}, net {net} over the out-of-sample period at 2x the venue's fees "
             "(0.04% maker, 0.11% taker; 90 days, 41 round trips; at the venue's fees +4.0%)\n")
NOT_TESTED = "**Profitable at double fees:** not tested: no out-of-sample window chose settings to re-run\n"


def _with(tmp_path, verdict=None, net="+3.1%", name="b22"):
    text = _sheet() + ("\n" + STUB_LINE.format(verdict=verdict, net=net) if verdict else "")
    path = tmp_path / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("verdict, net", [("PASS", "+3.1%"), ("FAIL", "-2.4%")])
def test_sheet_reads_the_stored_double_fee_result(tmp_path, verdict, net):
    s = dev.read_sheet(_with(tmp_path, verdict, net))
    assert s["double_fees"] == {"verdict": verdict, "net": net}


def test_sheet_without_it_is_not_run(tmp_path):
    assert dev.read_sheet(_with(tmp_path))["double_fees"] is None


@pytest.mark.parametrize("verdict, word", [("PASS", "Profitable at 2× fees"), ("FAIL", "Not profitable at 2× fees")])
def test_result_and_tearsheet_pages_show_it_beside_the_break_even_fee(client, tmp_path, verdict, word):  # noqa: F811
    c, _, _ = client
    _with(tmp_path, verdict, "+3.1%")
    for url in ("/results/b22", "/research/b22"):
        r = c.get(url, auth=AUTH)
        if r.status_code == 404:
            continue
        assert word in r.text and "+3.1%" in r.text, url
    assert word in c.get("/results/b22", auth=AUTH).text


def test_pages_say_not_run_when_the_sheet_has_no_result(client, tmp_path):  # noqa: F811
    c, _, _ = client
    _with(tmp_path)
    html = c.get("/results/b22", auth=AUTH).text
    assert "2× fees: not run" in html and "Profitable at 2× fees" not in html


def test_not_tested_line_is_its_own_state(client, tmp_path):  # noqa: F811
    c, _, _ = client
    (tmp_path / "b22.md").write_text(_sheet() + "\n" + NOT_TESTED, encoding="utf-8")
    assert dev.double_fees(NOT_TESTED) == {"verdict": "NOT TESTED", "net": None}
    html = c.get("/results/b22", auth=AUTH).text
    assert "2× fees: not tested" in html and "not run" not in html.split('id="double-fees"')[1].split("</p>")[0]
