"""QA regression cells from the #155 final round (QA worker p155z, 7 Oct): liquidation probes on the merged code.

Ported from /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_155_final_p155z.py (QA-only
test file). The findings these cells pinned as strict xfails are fixed, so the marks are gone; each cell is now a plain
regression test. Assertions, expected values and set-up are QA's, unchanged.

Built on the QA master tests/test_degraded_155_qa.py (its harness and autouse _reg / _guard_marks fixtures) and, for
the hub-fed outage replay, #146's QA harness tests/test_hub_146_qa.py. Both are imported by their bare module names, as
the harness itself does (`from test_replay import START`), so the module that the harness reads START from is the one
these cells move.
"""
import dataclasses
import re

import pytest

import test_degraded_155_qa as q
from test_degraded_155_qa import _reg  # noqa: F401  (fixture: the master's harness)
try:  # a head before stop safety has no guard marks to apply
    from test_degraded_155_qa import _guard_marks  # noqa: F401  (fixture: applies the guard-lift marks below)
except ImportError:
    pass
from sleeve_fund import risk
from sleeve_fund.store import Store
from sleeve_fund.strategies import REGISTRY

PERP = q.PERP
# Set-up only (HoQA 7 Oct, after #182): every probe here is about liquidation mechanics with a stopless perp above 1x,
# which #182's 5% open-risk limit refuses at entry and its restore guard closes with a safety stop. The repo's marks lift
# both, saying why; no assertion changes. (The stopless-above-1x refusal at start is not lifted: PERP carries a stop.)
GUARDS_OFF = "guards off: liquidation mechanics only"
pytestmark = [pytest.mark.no_open_risk_limit(reason=GUARDS_OFF), pytest.mark.no_restart_safety_stop(reason=GUARDS_OFF)]
NAME = "ping-pong-test"
H = 3600 * 10**9
RULED = "Position margin lost (liquidated): "


def _caps(pct):
    for nm, p in list(risk.PROFILES.items()):
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=pct)


def _short_then_restart_past_liq(tmp_path, store, params, start0, restart_at_h=6, legs=None, px=None):
    import test_replay
    from sleeve_fund.research.replay import replay
    s1 = tmp_path / "s1.jsonl.gz"
    q._record_session(s1, q._session_meta(10_000, params), [(5, 0.0), (20, 0.015), (2, 0.0)])
    replay(s1, store=store)
    book = store.journal_book(NAME, 10_000)
    assert book["qty"] < 0, book  # harness: short held
    store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
    test_replay.START = start0 + int(restart_at_h * H)
    s2 = tmp_path / "s2.jsonl.gz"
    bal = book["cash"] + book["qty"] * book["entry_px"]
    q._record_session(s2, q._session_meta(bal, params), legs or [(3, 0.0), (3, -0.02), (3, 0.03), (4, -0.02)],
                      px=px or 60_000.0 * 1.015 * 1.6)
    return replay(s2, store=store), book


# ---- F-2b (P1-D25) -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["reconnect", "restart"])
@pytest.mark.parametrize("setup", [0, 1, 2], ids=["perp-2x-long", "perp-3x-long", "perp-3x-short"])
def test_f2b_the_replayed_liquidations_incident_names_the_equity_the_journal_leaves(monkeypatch, path, setup):
    """After a liquidation found by the outage replay, the incident's 'equity left' is the journal's cash, not the
    engine's mark (the replayed fill's fee as journaled, not the market-on-return fee).

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_155_final_p155z.py; finding
    P1-D25 (MINOR)."""
    import test_hub_146_qa as qa
    monkeypatch.setitem(REGISTRY, "probe", qa._probe_classes())
    label, perp, profile, side = qa.LIQ_SETUPS[setup]
    p = qa.shape(qa.flat_prices(30), 7.0, 8.0, qa.adverse(side, qa.LIQ_DEPTH[profile]))
    run = qa._outage(path, p, side=side, perp=perp, profile=profile, stop=0.10, qty=0.25)
    assert qa.first_exit(run.sequence())[0] == "liquidation"  # harness
    (inc,) = run.kinds("incident")
    m = re.search(r"; ([\d,]+\.\d\d) of equity left", inc["message"])
    left = run.store.journal_book("q146", 10_000)["cash"]
    assert m and float(m.group(1).replace(",", "")) == pytest.approx(max(left, 0.0), abs=0.011), (inc["message"], left)


# ---- F-7 (GAP-LIQ-CAP) ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["restart-past-liq", "live-gap"])
def test_f7_the_loss_booked_on_a_liquidation_is_never_more_than_x(tmp_path, variant):
    """10% isolated margin at 2x: the loss booked on a liquidation (from the equity once the short was open, its entry
    fee already paid, funding kept out) is never more than X, after a restart past liquidation or on a live gap.

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_155_final_p155z.py; finding
    GAP-LIQ-CAP."""
    import test_replay
    _caps(0.1)
    params = {"rise": 0.01, "dip": 0.005, **PERP}
    store = Store(f"sqlite:///{tmp_path}/f7.db")
    start0 = test_replay.START
    try:
        if variant == "restart-past-liq":
            _, book = _short_then_restart_past_liq(tmp_path, store, params, start0)
        else:
            _, book = _short_then_restart_past_liq(tmp_path, store, params, start0,
                                                   legs=[(2, 0.0), (0, 0.6), (5, 0.0)], px=60_000.0 * 1.015)
    finally:
        test_replay.START = start0
    s = store.sleeve(NAME)
    assert s.status == "halted" and s.status_reason.startswith(RULED), s.status_reason  # harness: liquidated
    end = store.journal_book(NAME, 10_000)
    # the liquidation's loss from the equity once the short was open (its entry fee already paid), funding kept out
    loss = book["cash"] + book["qty"] * book["entry_px"] + end["funding"] - end["cash"]
    assert loss <= q._x_of(s.status_reason) + 0.01, (loss, s.status_reason)
