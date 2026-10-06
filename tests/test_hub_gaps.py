"""Minutes the market data feed missed that the hub hasn't refilled (QA P1-L17, and the condition HoE and QA set on
deferring P1-L16 to the Data Architect): no new entries until they are refilled and out of the indicators' reach, an
alert names them, and exits are never held. Uses Head of QA's hub-fed paper harness (test_hub_146_qa).

Two ways the minutes are known lost: the hub announces the gap and its refill doesn't come (the decoder's flush
releases the held live minute late; the strategy sees it), or the hub restarted without its gap state and announces
a gap that starts after them (only the hub client can tell; it shares them through HubStatus, as the node wires it).
Minutes no trade happened in are neither: decided past, as a backtest does."""

import pytest

import test_hub_146_qa as qa
from sleeve_fund.paper.hub_client import HubStatus
from test_hub_146_qa import M, S, START, _probe, flat_prices, minute, paper  # noqa: F401 - _probe registers the probe

GONE = frozenset(range(6 * 60, 10 * 60))  # no trade 00:06 to 00:10 reaches the client


class TimelineStatus(HubStatus):
    """A HubStatus for Head of QA's harness, which decodes the whole feed before the run: it answers as the hub
    client's would have at the strategy's clock time (the decoder records its lost minutes after each message)."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[tuple[int, dict]] = []
        self.strategy = None

    def record(self, now_ns: int) -> None:
        self.history.append((now_ns, {k: set(v) for k, v in self.lost.items()}))

    def unfilled(self, iid: str) -> list[int]:
        now, snap = self.strategy.clock.timestamp_ns(), {}
        for at, lost in self.history:
            if at > now:
                break
            snap = lost
        return sorted(snap.get(iid, ()))


def _restart_sched(refill_at):
    """The hub restarts without its gap state (QA P1-L16): down 00:06:20 to 00:08:30, it announces and refills only the
    stand-in minute to 00:09, then the minute to 00:10 comes live. 00:07 and 00:08 come only at refill_at, if at all
    (refilled by something other than this hub's gap, e.g. once the DA's fix lands)."""

    def sched(mins, iid, lag=S // 2, *_, **__):
        stand_in = START + 9 * M
        out = []
        for ts, r in mins.iterrows():
            close = int(ts.value) + M
            if close <= START + 6 * M or close > stand_in:
                out.append((close + lag, qa.hub_msg(iid, close, r), close))
            elif close == stand_in:
                out.append((stand_in + lag + 2 * S, qa.hub_msg(iid, close, r, refilled=True), close))
            elif refill_at is not None:
                out.append((refill_at, qa.hub_msg(iid, close, r, refilled=True), close))
        span = {"id": iid, "since": stand_in, "until": stand_in}
        out += [(stand_in + lag, {"t": "gap", **span}, stand_in - 1),
                (stand_in + lag + 2 * S, {"t": "filled", **span}, stand_in + 1)]
        out += [(t + 3, {"t": "hb", "ts": t, "venue_up": True}, -1)
                for t in range(START + 8 * M + 30 * S, START + len(mins) * M, 5 * S)]
        return [(at, m) for at, m, _ in sorted(out, key=lambda x: (x[0], x[2]))]

    return sched


def _hub(monkeypatch, *, since, refill_at=None, status=None):
    """The hub loses the venue 00:06-00:10 and announces a gap from `since` (00:07, or 00:09 after a restart that
    forgot the earlier minutes) just before the live minute to 00:11. The minutes to 00:07-00:10 come only at
    refill_at (refilled, then "filled"), if at all. Heartbeats every 5 s flush held minutes, as HubDataClient does.
    status: a TimelineStatus shared by the decoder and the strategy, as paper.node wires a HubStatus."""
    from sleeve_fund.paper import hub_client
    from sleeve_fund.strategies import base

    class HbDecoder(hub_client.Decoder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k, status=status)

        def __call__(self, m, now_ns):
            out = (self.flush(now_ns) or None) if m.get("t") == "hb" else super().__call__(m, now_ns)
            if status is not None:
                status.record(now_ns)
            return out

    def sched(mins, iid, lag=S // 2, *_, **__):
        a, b, live = START + 6 * M, START + 10 * M, START + 11 * M
        out = []
        for ts, r in mins.iterrows():
            close = int(ts.value) + M
            if not a < close <= b:
                out.append((close + lag, qa.hub_msg(iid, close, r), close))
            elif refill_at is not None:
                out.append((refill_at, qa.hub_msg(iid, close, r, refilled=True), close))
        out.append((live + lag, {"t": "gap", "id": iid, "since": since, "until": b}, live - 1))
        if refill_at is not None:
            out.append((refill_at, {"t": "filled", "id": iid, "since": since, "until": b}, b + 1))
        out += [(t + 3, {"t": "hb", "ts": t, "venue_up": True}, -1) for t in range(START, START + len(mins) * M, 5 * S)]
        return [(at, m) for at, m, _ in sorted(out, key=lambda x: (x[0], x[2]))]

    monkeypatch.setattr(hub_client, "Decoder", HbDecoder)
    monkeypatch.setattr(qa, "schedule", sched)
    if status is not None:
        attach = base.LongFlatStrategy.attach_runtime

        def wired(self, runtime):
            self.hub_status, status.strategy = status, self
            return attach(self, runtime)

        monkeypatch.setattr(base.LongFlatStrategy, "attach_runtime", wired)


def _entries(run):
    return [r for r in run.sequence() if r[0] == "entry"]


@pytest.mark.parametrize("warmup", [0, 5], ids=["no-warm-up", "five-bar-warm-up"])
def test_an_announced_gap_never_refilled_blocks_entries_and_is_reported(warmup, monkeypatch):
    """The entry is due from the minute to 00:11. Minutes 00:07 to 00:10 never come, not even refilled (no store):
    nothing is opened, and the gap is named once."""
    _hub(monkeypatch, since=START + 7 * M)
    run = paper(flat_prices(30), enter=11, leave=25, stop=0.01, gone=GONE, warmup=warmup)
    assert not _entries(run), run.sequence()
    (gap,) = run.kinds("data_gap")
    assert "4 minute(s) closing 00:07 to 00:10" in gap["message"]
    assert run.kinds("entry_skipped_gap") and not run.kinds("gap_cleared")


@pytest.mark.parametrize("warmup, first", [(0, 15), (5, 16)], ids=["no-warm-up", "five-bar-warm-up"])
def test_entries_resume_once_the_gap_is_refilled_and_out_of_the_indicators_reach(warmup, first, monkeypatch):
    """The refill lands at 00:14:30, after the flush released the minute to 00:11: entries resume on the next minute
    (00:15), or once the warm-up no longer reaches a minute the indicators never got (00:16 with five bars)."""
    _hub(monkeypatch, since=START + 7 * M, refill_at=START + 14 * M + 30 * S)
    run = paper(flat_prices(30), enter=11, leave=25, stop=0.01, gone=GONE, warmup=warmup)
    assert _entries(run) and _entries(run)[0][2] == minute(first), run.sequence()
    assert run.kinds("data_gap") and run.kinds("gap_cleared")


def test_an_unfilled_gap_never_holds_the_stop(monkeypatch):
    """In a position when the gap opens: the stop crossed after it is still taken on the trade that crosses it."""
    _hub(monkeypatch, since=START + 7 * M)
    p = qa.shape(flat_prices(30), 12.5, 30, qa.adverse(1, 0.02))
    run = paper(p, enter=2, leave=25, stop=0.01, gone=GONE, warmup=5)
    assert [r[0] for r in run.sequence()][:2] == ["entry", "stop_loss"], run.sequence()
    assert run.sequence()[1][2] <= minute(13)


@pytest.mark.parametrize("refilled", [False, True], ids=["never-refilled", "refilled-at-00:14:30"])
def test_after_a_hub_restart_that_forgot_its_gap_no_entry_until_the_minutes_before_it_are_refilled(refilled,
                                                                                                    monkeypatch):
    """The L16 condition: the hub announces a gap of its stand-in minute (00:09) only, so 00:07 and 00:08 will never
    be refilled by it, and the minute to 00:09 comes on time. Only the hub client can tell: it says so (an alert
    naming them) and the strategy, due to enter from 00:10, opens nothing until they come."""
    status = TimelineStatus()
    _hub(monkeypatch, since=START + 9 * M, status=status)
    monkeypatch.setattr(qa, "schedule", _restart_sched(START + 14 * M + 30 * S if refilled else None))
    run = paper(flat_prices(30), enter=10, leave=25, stop=0.01, gone=frozenset(range(6 * 60 + 20, 8 * 60 + 30)))
    (told,) = [e["message"] for e in run.kinds("hub_gap") if "announced a gap from" in e["message"]]
    assert "00:09" in told and "2 minutes closing" in told and "00:07" in told and "00:08" in told, told
    assert run.kinds("data_gap") and run.kinds("entry_skipped_gap")
    if refilled:
        assert _entries(run) and _entries(run)[0][2] == minute(15), run.sequence()
        assert run.kinds("gap_cleared") and not status.lost[next(iter(status.lost))]
    else:
        assert not _entries(run), run.sequence()


def test_minutes_with_no_trades_are_never_held_as_lost(monkeypatch):
    """No gap announced: minutes with no trade are decided past, as the backtest does (QA's hole parity)."""
    status = TimelineStatus()
    _hub(monkeypatch, since=START + 7 * M, status=status)
    monkeypatch.setattr(qa, "schedule", lambda mins, iid, lag=S // 2, *_, **__: [
        (int(ts.value) + M + lag, qa.hub_msg(iid, int(ts.value) + M, r)) for ts, r in mins.iterrows()])
    run = paper(flat_prices(30), enter=11, leave=25, stop=0.01, holes=(7, 8))
    assert _entries(run) and _entries(run)[0][2] == minute(11), run.sequence()
    assert not any(status.lost.values())
