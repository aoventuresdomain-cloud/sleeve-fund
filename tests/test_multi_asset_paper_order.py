"""Multi-asset Strategies on paper: same-close ideas wait for every leg, or close + 2 s, then gate strongest first and by
asset place (multi-asset spec §4, §8 cells 2, 8 and 9; BRIDGE-3-QA 2). A leg deciding after its close's pass gates at
once, after its group, and is counted late."""

from __future__ import annotations

from sleeve_fund.paper.leg_batch import WAIT_NS, LegBatch

NAMES = ("t", "t~ETH", "t~SOL")
CLOSE = 60 * 1_000_000_000


class Clock:
    def __init__(self):
        self.now, self.alerts = CLOSE + 10_000_000, []

    def timestamp_ns(self):
        return self.now

    def set_time_alert_ns(self, name, at, callback):
        self.alerts.append((at, callback))

    def fire(self):
        for at, cb in self.alerts:
            cb(None)
        self.alerts.clear()


def _post(batch, clock, gated, name, strength=0.0, close=CLOSE):
    batch.post(clock, close, name, strength, lambda: gated.append(name))


def test_once_every_leg_has_decided_the_pass_runs_by_strength_then_asset_place():
    batch, clock, gated = LegBatch(NAMES), Clock(), []
    _post(batch, clock, gated, "t~SOL", 1.0)
    _post(batch, clock, gated, "t", 1.0)
    batch.decide(CLOSE, "t~SOL")
    batch.decide(CLOSE, "t")
    assert gated == []  # t~ETH hasn't decided yet
    batch.decide(CLOSE, "t~ETH")  # no idea of its own: the pass runs now
    assert gated == ["t", "t~SOL"]  # equal strength: the asset list's order, whatever arrived first


def test_the_stronger_idea_is_gated_first():
    batch, clock, gated = LegBatch(NAMES), Clock(), []
    _post(batch, clock, gated, "t", 0.5)
    _post(batch, clock, gated, "t~SOL", 2.0)
    for n in NAMES:
        batch.decide(CLOSE, n)
    assert gated == ["t~SOL", "t"]


def test_a_slow_leg_never_holds_the_others_past_close_plus_2_s_and_then_gates_after_its_group():
    batch, clock, gated = LegBatch(NAMES), Clock(), []
    _post(batch, clock, gated, "t~ETH")
    batch.decide(CLOSE, "t~ETH")
    assert clock.alerts[0][0] == CLOSE + WAIT_NS
    clock.fire()  # close + 2 s: t and t~SOL never decided
    assert gated == ["t~ETH"]
    _post(batch, clock, gated, "t")  # decided late on that close: gated at once, after its group
    assert gated == ["t~ETH", "t"] and batch.late == {"t": 1}


def test_each_close_has_its_own_pass():
    batch, clock, gated = LegBatch(NAMES), Clock(), []
    _post(batch, clock, gated, "t~SOL")
    for n in NAMES:
        batch.decide(CLOSE, n)
    _post(batch, clock, gated, "t", close=CLOSE * 2)
    assert gated == ["t~SOL"]
    for n in NAMES:
        batch.decide(CLOSE * 2, n)
    assert gated == ["t~SOL", "t"]
