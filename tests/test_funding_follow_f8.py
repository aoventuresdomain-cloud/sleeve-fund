"""FD-FOLLOW, FD-F8 (HoE 7 Oct): a sparse trade landing just before each minute's hub bar re-armed the missed-minutes
hold (_awaiting) for ever, so funding waited until the position closed and the live stop and target waited with it.
The hold now lifts once the bar covers all but a stretch shorter than UNSEEN_GAP_NS. Pinned by QA's own probe
(tests/fdz_delta_probe.py = quant-review/v2-p1/funding-deadline-scripts/test_fdz_delta_probe.py, copied unchanged and
named so it isn't collected whole), which fails on main b4b88bb."""

from fdz_delta_probe import (  # noqa: F401  (fixtures are used by name; the probe is collected here)
    _booked,
    _o17win,
    _probe,
    binance,
    test_sparse_trades_landing_before_the_minute_bar_still_book_by_the_bound,
)
