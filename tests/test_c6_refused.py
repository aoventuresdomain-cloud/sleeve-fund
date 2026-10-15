"""C6 (ui-v2/compliance-build-spec.md): the ideas the compliance tests refused or trimmed, on Portfolio Settings for
every Strategy and on a Strategy's Refused ideas tab for its own; each with the test by name, its numbers when the
gate journaled what the idea would have reached, and the version that decided it."""

from __future__ import annotations

from decimal import Decimal as D

from sqlalchemy import insert
from test_attribution import _traded
from test_dashboard import AUTH, client  # noqa: F401 - the fixture

from sleeve_fund.store import gate_decisions_t, sleeves_t, utcnow


def _versioned(store, would_be=None, side=None):
    with store.engine.begin() as conn:
        sid = conn.execute(sleeves_t.select().where(sleeves_t.c.name == "btc-test")).first().id
        conn.execute(insert(gate_decisions_t), [{
            "sleeve_id": sid, "bar_ts": utcnow(), "intent": "entry", "underlying": "BTC", "profile_version": 1,
            "price": D(50000), "stage": "submit", "seq": 9, "intent_id": "i9", "outcome": "rejected",
            "test_hit": "asset_stop_loss", "compliance_version": 1, "requested_qty": D("0.2"), "approved_qty": D(0),
            "decided_at": utcnow(), "would_be": would_be, "side": side}])


def test_portfolio_settings_lists_every_refused_idea_with_its_strategy_test_and_version(client):  # noqa: F811
    c, store = client
    _traded(c, store)
    _versioned(store)
    page = c.get("/portfolio-settings", auth=AUTH).text
    table = page.split('id="refused-h"')[1].split("</section>")[0]
    assert table.count("<tr>") == 1 + 3  # the head and three refused or trimmed ideas; the approved one isn't here
    assert 'href="/sleeves/btc-test">btc-test</a>' in table and " UK</td>" in table
    assert "T1 Per asset" in table and "T4 Net per asset" in table and "Trimmed to 0.45" in table
    first = table.split("<tbody>")[1].split("</tr>")[0]  # newest first: the versioned refusal
    assert ">Refused<" in first and "T1 Per asset" in first and '<td class="num">1</td>' in first
    assert first.count("<td>–</td>") == 1  # no side journaled yet: a dash, never a guess


def test_a_refusal_shows_the_tests_numbers_and_side_the_gate_journaled(client):  # noqa: F811
    c, store = client
    _traded(c, store)
    _versioned(store, would_be=D("0.046"), side=-1)
    t1 = next(t for t in store.compliance(1).tests if t["test"] == "asset_stop_loss")["limit"]
    from sleeve_fund import compliance_rules as rules

    want = f"T1 Per asset: 4.6% would exceed {rules.show('asset_stop_loss', t1)}"
    page = c.get("/portfolio-settings", auth=AUTH).text
    first = page.split('id="refused-h"')[1].split("<tbody>")[1].split("</tr>")[0]
    assert want in first and "<td>Short</td>" in first
    tab = c.get("/sleeves/btc-test", auth=AUTH).text.split('id="tab-refused"')[1]
    assert want in tab and "<th scope=\"col\">Strategy</th>" not in tab.split("</table>")[0]



def test_a_refusal_with_no_test_or_version_shows_the_gates_reason_not_a_bare_refused(client):  # noqa: F811
    """F-XA12-TESTNAME (DA 05:15 UK): a T1 or T2 refusal always carries its test and version (the store ties
    test_hit to a version), so the row names both; one with neither, a reason the gate gave before any test was
    reached or a limit named the old way, shows that reason in words and a dash for the version."""
    from sleeve_fund import compliance_rules as rules

    c, store = client
    _traded(c, store)
    with store.engine.begin() as conn:
        sid = conn.execute(sleeves_t.select().where(sleeves_t.c.name == "btc-test")).first().id
        conn.execute(insert(gate_decisions_t), [{
            "sleeve_id": sid, "bar_ts": utcnow(), "intent": "entry", "underlying": "BTC", "profile_version": 1,
            "price": D(50000), "stage": "submit", "seq": 9 + i, "intent_id": f"i{9 + i}", "outcome": "rejected",
            "limit_hit": hit, "requested_qty": D("0.2"), "approved_qty": D(0), "decided_at": utcnow()}
            for i, hit in enumerate(("below_min", "open_risk"))])
    rows = c.get("/sleeves/btc-test", auth=AUTH).text.split('id="tab-refused"')[1].split("<tbody>")[1]
    rows = rows.split("</tbody>")[0].split("</tr>")
    shown = [r for r in rows if "<td>–</td>" in r or '<td class="num">–</td>' in r]
    assert any(f"<td>{rules.NOT_A_TEST['below_min']}</td>" in r and '<td class="num">–</td>' in r for r in shown)
    assert any(f"<td>{rules.name('fund_stop_loss')}</td>" in r and '<td class="num">–</td>' in r for r in shown)
    assert not any("<td>Refused</td>" in r for r in rows)
