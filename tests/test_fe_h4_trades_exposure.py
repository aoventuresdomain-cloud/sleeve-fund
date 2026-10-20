"""Head 4 step 6 gap 1 (QD2's read-back on 48c30ff6): on Positions and trades a fund-held position's exposure read
715.21x against the ledger's 0.25x, because the cell divided the position's value by the Strategy's own equity, which
inside the fund is its P&L from 0. Inside the fund it is over the fund's equity, as Home, Risk and the Strategy page
show it (metrics.sleeve_summary, B-20); a Strategy with its own capital reads as before. On Postgres too
(TEST_DATABASE_URL)."""

import html
import re

import pytest
from test_fe_b20_fund_held_wording import (  # noqa: F401
    AUTH,
    MARK,
    QTY,
    _hold,
    _long,
    _summaries,
    c,
)
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

from sleeve_fund import supervisor

CELL = re.compile(r'<span class="muted" title="(Exposure:[^"]*)">([\d.]+)×</span>')


def _cell(c):  # noqa: F811
    r = c.get("/trades", auth=AUTH)
    assert r.status_code == 200
    cells = CELL.findall(r.text)
    assert len(cells) == 1, cells
    return html.unescape(cells[0][0]), float(cells[0][1])


def test_fund_held_the_positions_exposure_is_over_the_funds_equity(c, store):  # noqa: F811
    _hold(store)
    _long(store)
    title, shown = _cell(c)
    fund = float(supervisor.fund_equity(store))
    assert shown == pytest.approx(round(QTY * MARK / fund, 2))  # 0.06x, not 57.11x over a P&L of -103.30
    assert "the fund's equity" in title and "strategy's equity" not in title
    summaries, _ = _summaries(store)
    assert shown == pytest.approx(round(abs(summaries[0]["exposure"]), 2))  # the figure Home and Risk show


def test_not_held_the_positions_exposure_is_over_its_own_equity(c, store):  # noqa: F811
    _long(store, "own-long", balance=10_000)
    title, shown = _cell(c)
    summaries, _ = _summaries(store)
    assert shown == pytest.approx(round(QTY * MARK / summaries[0]["equity"], 2))
    assert "the strategy's equity" in title
