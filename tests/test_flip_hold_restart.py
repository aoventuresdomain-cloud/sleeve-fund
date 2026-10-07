"""FLIP-HOLD end to end (QA #161 delta, HoE 7 Oct 01:28): restarted holding a perp long with its slower candles
short of history, the long's target trades and the signal has turned short. The target still sells, and no reversal
opens until the warm-up is met (_entry_held). In paper the target trades tick by tick (_check_exits), so the turn is
decided flat on the next candle and held there; _flip_after_target (a backtest's bar target) holds it the same way."""


def test_a_restart_short_of_slower_history_takes_the_target_but_opens_no_reversal_after_it(tmp_path, monkeypatch):
    from sleeve_fund.strategies.rsi_cross import RsiCross
    from test_p1_4_xfails import _restart_holding_a_long, _says_short

    def want_side(self, bar):  # long until the target has traded, then the signal turns short
        took = any(d.get("intent") == "take_profit" for d in self.decisions.values())
        return -1 if took else 1

    monkeypatch.setattr(RsiCross, "want_side", want_side)
    orders, events = _restart_holding_a_long(tmp_path, monkeypatch, [(5, 0.0), (5, 0.03), (20, 0.0)],
                                             {"take_profit": 0.02, "stop_loss": 0.05, "time_stop_bars": 0,
                                              "allow_short": True})
    assert _says_short(events)
    assert ("SELL", "take_profit") in [(o["side"], o["intent"]) for o in orders], orders
    assert not [o for o in orders if o["intent"] == "entry"], orders  # no short opened on short history
    assert "entry_held" in {e["kind"] for e in events}
