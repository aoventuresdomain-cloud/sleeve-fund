"""QA #203 probe (HoQA, 7 Oct): the previous process's last heartbeat came AFTER a candle closed but BEFORE that
candle's bar was delivered and decided (the tick timer stamps the heartbeat every 30 s on its own; the hub's minute
arrives seconds after the close and may be held up to HOLD_SECONDS). The 00:15 close says exit. A restart must
treat 00:15 as missed and exit once, "Late exit, missed candle 00:15", before the next close."""

from datetime import timezone

import pytest
from test_hub_146_qa import _guard_marks, _probe  # noqa: F401


def _exits(run):
    return [o for o in run.orders if o["intent"] == "exit"]


@pytest.mark.parametrize("perp, side", [(False, 1), (True, -1)])
@pytest.mark.parametrize("back", [15 + 40 / 60, 20.5])
def test_heartbeat_after_the_close_before_its_decision_still_exits_on_that_candle(perp, side, back):
    from test_hub_146_qa import M, START, flat_prices, restart

    run = restart(flat_prices(30), 10, back, heartbeat=15 + 5 / 60, side=side, perp=perp, leave=15, warmup=30,
                  stop=None)
    ex = _exits(run)
    assert ex, "no exit at all"
    first = ex[0]
    sent = first["ts"] if first["ts"].tzinfo else first["ts"].replace(tzinfo=timezone.utc)
    assert first["reason"] == "Late exit, missed candle 00:15", [(o["reason"], o["ts"]) for o in ex]
    assert sent.timestamp() * 1e9 < START + (int(back) + 1) * M  # before the next close
