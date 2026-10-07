"""FD-FOLLOW (HoE 7 Oct): QA's FUNDING-DEADLINE delta MINORs FD-F7, F9 and F10 (F8 is its own PR), pinned by QA's own probes
(tests/fdz_delta_probe.py = quant-review/v2-p1/funding-deadline-scripts/test_fdz_delta_probe.py, copied unchanged and
named so it isn't collected whole: its FD-F4 probes stay QA's, with P1-L16/L17). Each probe here fails on main b4b88bb.

- FD-F7: a restart between a settlement and its booking, with a fill after it or a flat book, still charges the
  position held at it (on_start resumes from the opening fill, not the last one).
- FD-F9: the strategy's own fill after a restart doesn't hide the position held at an unbooked settlement.
- FD-F10: a replayed stop in the minute ending at the settlement: a cost is paid, a credit is not, as the backtest.
"""

from fdz_delta_probe import (  # noqa: F401  (fixtures are used by name; the probes are collected here)
    DAY,
    STOP,
    _booked,
    _o17win,
    _probe,
    binance,
    charges,
    qa,
    restarted,
    utc,
    SETTLE,
    test_a_plain_exit_after_the_settlement_then_restart_charges_08_00_once,
    test_a_replayed_stop_in_the_minute_ending_at_the_settlement_matches_the_backtest,
    test_a_restart_after_a_reduce_after_the_settlement_charges_the_quantity_held_at_it,
    test_a_restart_after_a_reduce_and_a_reversal_after_the_settlement_charges_the_long_held_at_it,
    test_a_restart_after_an_exit_journaled_at_the_settlement_instant_still_charges_the_long,
    test_restart_journal_path_with_an_exit_filled_while_the_rate_is_awaited_charges_the_long,
)


def test_a_restart_after_a_replayed_exit_filled_after_the_settlement_charges_nothing(monkeypatch):
    """QA's control (a replayed close, then a restart after the exit filled) with the replayed close journaled on its
    order, as A writes it since 2cc3c3e: B resumes from the opening fill (FD-F7) and still charges nothing for 08:00,
    which the venue's stop closed before."""
    p = qa.shape(qa.flat_prices(60), *STOP, qa.adverse(1, 0.02))
    sig = {"price_source": "replay_model", "replayed_close": utc(f"{DAY} 07:59").isoformat(), "stop_frac": 0.01}
    run, _ = restarted(monkeypatch, back_min=18, heartbeat_min=16, seen_min=16, prices=p, leave=55,
                       fills=[("SELL", 0.05, 15 + 5 / 60)], orders=[("O-rx", "SELL", 0.05, "stop_loss", 15, sig)])
    assert charges(run.store.funding("q146"), SETTLE) == [], run.store.funding("q146")
    assert not run.kinds("funding_charged_while_flat")
