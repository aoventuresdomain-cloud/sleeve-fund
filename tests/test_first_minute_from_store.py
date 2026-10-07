"""QA P1-D19 (Advisor and HoE restart ruling, R1): paper on its own feed, restarted part way through a minute, builds
that minute's share of its first bar from the history store's minute, not only from the trades it saw after the
start. Synthetic data only, no venue called."""
import pandas as pd
import pytest

import qa155_lib as lib
import test_degraded_155_qa as q
from sleeve_fund import funding


@pytest.fixture(autouse=True)
def _reg(monkeypatch, tmp_path):
    lib.stub_contract(monkeypatch)
    monkeypatch.setitem(q.REGISTRY, "qa_t", (q.T, q.TConfig))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    monkeypatch.setattr(funding, "fetch", lambda *a, **k: (_ for _ in ()).throw(OSError("no venue calls")))
    q.SEEN.clear()


def test_a_restart_mid_minute_takes_that_minutes_range_from_the_store(tmp_path, monkeypatch):
    """Restarted at 02:52:20, inside the 15-minute bar closing 03:00, whose high and low both fall in 02:52 before the
    start: the first bar is the store's 15-minute bar, high and low included (s20's spike case)."""
    seen = []
    real = q.T.update_indicators

    def record(self, bar):
        seen.append((pd.Timestamp(bar.ts_event, tz="UTC"), bar.open.as_double(), bar.high.as_double(),
                     bar.low.as_double(), bar.close.as_double()))
        return real(self, bar)

    monkeypatch.setattr(q.T, "update_indicators", record)
    cut = int(pd.Timestamp("2025-01-01 02:52:20", tz="UTC").value)
    ticks = q.ticks_from_bars
    monkeypatch.setattr(q, "ticks_from_bars", lambda *a, **k: [t for t in ticks(*a, **k) if t.ts_event >= cut])
    inst = q.binance_inst()
    m1 = q.synth_1m(days=1, seed=8, vol_day=0.01)
    spike = pd.Timestamp("2025-01-01 02:53", tz="UTC")  # the 02:52 minute, by its close
    m1.loc[spike, "high"] += 80.0
    m1.loc[spike, "low"] -= 80.0
    q.write_funding(m1.index[0], m1.index[-1], root=tmp_path / "funding")
    hs = q.holed_store(tmp_path / "h", m1, [])
    close = pd.Timestamp("2025-01-01 03:00", tz="UTC")
    stored = hs.read("BINANCE", "BTC/USDT", 15).loc[close]
    m1x = m1[m1.index - pd.Timedelta(minutes=1) >= pd.Timestamp("2025-01-01 02:52", tz="UTC")]
    params = {**q.PERP, "at": close.value, "side": 1, "tag": "d19"}
    q.quiet(lambda: q._paper_from(inst, m1x, params, hs, False))
    (first,) = [b for b in seen if b[0] == close]
    assert first[1:] == pytest.approx((stored.open, stored.high, stored.low, stored.close)), (first, stored)
    assert stored.high == pytest.approx(m1.loc[spike, "high"])  # the spike is what the store adds
