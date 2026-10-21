"""B-23 (Advisor B23-ACCOUNT-CAPITAL): the account's marked equity against the book, the gap, and "Size from the account
figure", which ships OFF. Every venue answer here is a recorded response shaped as the wallet-balance call returns it.
CI also runs these against Postgres via TEST_DATABASE_URL."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from sleeve_fund import account_capital as ac
from sleeve_fund.store import Store, make_engine, metadata

T = datetime(2026, 10, 20, 19, 0, tzinfo=timezone.utc)
ON = {ac.SETTING_ENV: "on"}


def wallet(wallet="100000", unrealised="0", available="12345.67", others=()):
    """A recorded wallet-balance answer: one unified wallet, USDT plus any other coins."""
    coins = [{"coin": "USDT", "walletBalance": wallet, "unrealisedPnl": unrealised, "equity": "0",
              "availableToWithdraw": available, "usdValue": wallet}]
    coins += [{"coin": c, "walletBalance": "0.5", "unrealisedPnl": "0", "equity": "0.5"} for c in others]
    return {"list": [{"accountType": "UNIFIED", "totalEquity": "1", "totalAvailableBalance": available,
                      "totalWalletBalance": "1", "coin": coins}]}


class Recorded:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def call(self, method, path, params=None):
        self.calls.append((method, path, params))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield Store(engine=eng)
    _clean(eng)


def _kinds(store):
    return [e["kind"] for e in reversed(store.events_of((ac.GAP_ALERT, ac.GAP_CLOSED, ac.GAP_REFUSED, ac.READ_FAILED,
                                                        ac.READ_BACK)))]


def _live_read(store, account, equity, book, at):
    """A live read as it will be stored after G2 (before G2 fetch_account never reads a live account)."""
    store.record_account_read(account, ok=True, at=at, payload={"coin": "USDT", "wallet": equity, "unrealised": "0",
                                                                "equity": equity, "other": [], "book": book})


# --- the read --------------------------------------------------------------------------------------------------------

def test_marked_equity_is_wallet_plus_unrealised_and_the_available_balance_is_never_used():
    p = ac.wallet_payload(wallet("100000", "-250.5", available="1.00"))
    assert p["equity"] == "99749.5" and p["wallet"] == "100000" and p["unrealised"] == "-250.5"
    assert "1.00" not in p.values()


def test_other_assets_are_left_out_of_the_figure_and_flagged():
    p = ac.wallet_payload(wallet("1000", others=("BTC",)))
    assert p["equity"] == "1000" and p["other"] == ["BTC"]


@pytest.mark.parametrize("bad", [{"list": []}, {"list": [{"coin": [{"coin": "USDT", "walletBalance": 5.0}]}]},
                                 {"list": [{"coin": [{"coin": "USDT", "walletBalance": "x"}]}]},
                                 {"list": [{"coin": [{"coin": "BTC", "walletBalance": "1"}]}]}])
def test_a_malformed_answer_is_refused_never_guessed(bad):
    with pytest.raises(ValueError):
        ac.wallet_payload(bad)


@pytest.mark.parametrize("field", ["walletBalance", "unrealisedPnl"])
@pytest.mark.parametrize("how", ["missing", "empty", "not a number"])
def test_must_a_missing_empty_or_unreadable_figure_fails_the_read_never_zero(store, field, how):
    answer = wallet("100000", "-10")
    coin = answer["list"][0]["coin"][0]
    if how == "missing":
        coin.pop(field)
    else:
        coin[field] = "" if how == "empty" else "n/a"
    with pytest.raises(ValueError):
        ac.wallet_payload(answer)
    ac.fetch_account(store, "paper", client=Recorded(answer), live=False, book=D("100000"), now=T)
    row = store.account_reads("paper")[0]
    assert not row["ok"] and row["payload"] is None and store.account_reads("paper", good=True) == []


def test_the_read_is_a_get_on_the_wallet_only(store):
    client = Recorded(wallet())
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100000"), now=T)
    assert client.calls == [("GET", ac.WALLET_PATH, {"accountType": "UNIFIED"})]


def test_a_live_account_is_never_read_before_g2(store):
    client = Recorded(wallet())
    ac.fetch_account(store, "live-1", client=client, live=True, book=D("100000"), now=T)
    assert client.calls == [] and store.account_reads("live-1")[0]["reason"] == ac.NOT_BEFORE_G2


def test_reads_are_append_only(store):
    rid = store.record_account_read("paper", ok=False, reason="timeout", at=T)
    with pytest.raises((IntegrityError, DBAPIError)):
        with store.engine.begin() as c:
            c.execute(text(f"DELETE FROM account_capital_fetch WHERE id = {rid}"))


# --- Advisor cells ---------------------------------------------------------------------------------------------------

def test_cell_paper_read_failure_does_not_refuse_and_warns_once(store):
    client = Recorded(TimeoutError("no answer"), TimeoutError("no answer"))
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100000"), now=T)
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100000"), now=T + timedelta(minutes=5))
    assert ac.entry_capital(store, "paper", D("100000"), T, live=False, environ=ON) == (D("100000"), None)
    assert ac.entry_capital(store, "paper", D("100000"), T, live=False, environ={}) == (D("100000"), None)
    assert _kinds(store) == [ac.READ_FAILED]


def test_cell_live_stale_read_refuses_entries_not_exits(store):
    _live_read(store, "live-1", "100000", "100000", T)
    equity, why = ac.entry_capital(store, "live-1", D("100000"), T + timedelta(minutes=6), live=True, environ=ON)
    assert why.startswith("Entry refused: no good read") and "exits, stops and flatten still run" in why
    # inside 5 minutes it isn't refused
    assert ac.entry_capital(store, "live-1", D("100000"), T + timedelta(minutes=4), live=True, environ=ON)[1] is None


def test_cell_live_sizes_from_the_lower_of_account_and_book(store):
    _live_read(store, "live-1", "99800", "100000", T)
    assert ac.entry_capital(store, "live-1", D("100000"), T, live=True, environ=ON) == (D("99800"), None)
    _live_read(store, "live-1", "100100", "100000", T + timedelta(minutes=1))
    assert ac.entry_capital(store, "live-1", D("100000"), T + timedelta(minutes=1), live=True,
                            environ=ON) == (D("100000"), None)


def test_cell_a_0_6_percent_gap_alerts_only(store):
    client = Recorded(wallet("100000"), wallet("100600"))
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100000"), now=T)
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100000"), now=T + timedelta(minutes=5))
    v = ac.view(store, "paper", T + timedelta(minutes=5))
    assert v.gap == D("600") and v.alerting and _kinds(store) == [ac.GAP_ALERT]
    assert ac.entry_capital(store, "paper", D("100000"), T, live=False, environ=ON) == (D("100000"), None)
    _live_read(store, "live-1", "100600", "100000", T)
    assert ac.entry_capital(store, "live-1", D("100000"), T, live=True, environ=ON) == (D("100000"), None)


def test_cell_a_2_1_percent_gap_refuses_live_entries_until_back_inside_and_acknowledged(store):
    _live_read(store, "live-1", "97900", "100000", T)
    v = ac.view(store, "live-1", T, live=True)
    assert v.over_refuse and v.alerting
    told = store.event(None, "error", ac.GAP_REFUSED, "gap over 2%", ts=T)  # as fetch_account journals it
    assert ac.entry_capital(store, "live-1", D("100000"), T, live=True, environ=ON)[1].startswith("Entry refused")
    # Advisor Q3: an acknowledgement while the gap is still over 2% does not lift the refusal
    store.ack(store.last_event(None, (ac.GAP_REFUSED,))["id"], "pm")  # stamped at the real clock
    later = datetime.now(timezone.utc) + timedelta(minutes=1)
    _live_read(store, "live-1", "100000", "100000", later)  # reconciled after the ack
    store.event(None, "info", ac.GAP_INSIDE, "back inside", ts=later)
    assert "acknowledgement" in ac.entry_capital(store, "live-1", D("100000"), later, live=True, environ=ON)[1]
    # the same 2.1% on paper never refuses
    assert ac.entry_capital(store, "paper", D("100000"), T, live=False, environ=ON) == (D("100000"), None)


def test_cell_an_acknowledgement_once_back_inside_lifts_the_refusal_and_is_journaled_with_who_and_when(store):
    _live_read(store, "live-1", "97900", "100000", T)
    store.event(None, "error", ac.GAP_REFUSED, "gap over 2%", ts=T)
    later = T + timedelta(minutes=1)
    _live_read(store, "live-1", "100000", "100000", later)
    store.event(None, "info", ac.GAP_INSIDE, "back inside", ts=later)
    told = store.last_event(None, (ac.GAP_REFUSED,))
    store.ack(told["id"], "pm")
    ack = store.ack_of(told["id"])
    assert ack["actor"] == "pm" and ack["ts"] is not None
    now = datetime.now(timezone.utc)  # the ack is stamped at the real clock
    _live_read(store, "live-1", "100000", "100000", now)
    assert ac.entry_capital(store, "live-1", D("100000"), now, live=True, environ=ON) == (D("100000"), None)


def test_cell_a_paper_deposit_of_10000_re_anchors_and_raises_no_gap_alert(store):
    client = Recorded(wallet("50000"), wallet("50000"))
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100000"), now=T)
    fid = store.fund_flow("deposit", "10000", actor="pm", reason="top up", now=T + timedelta(minutes=1))
    store.settle_flow(fid, T + timedelta(minutes=2))
    ac.fetch_account(store, "paper", client=client, live=False, book=D("110000"), now=T + timedelta(minutes=5))
    v = ac.view(store, "paper", T + timedelta(minutes=5))
    assert v.gap == 0 and not v.alerting and ac.GAP_ALERT not in _kinds(store)
    assert [r["anchor"] for r in store.account_reads("paper", limit=5)] == [1, 1]
    (said,) = store.events_of((ac.ANCHORED,))
    assert "fund deposit of 10,000.00" in said["message"] and "was account 50,000.00 vs book 100,000.00" in said["message"]
    assert "now account 50,000.00 vs book 110,000.00" in said["message"]


def test_a_stale_read_says_its_real_age(store):
    ac.fetch_account(store, "paper", client=Recorded(wallet()), live=False, book=D("100000"), now=T)
    assert "last good read 7 min ago" in ac.view(store, "paper", T + timedelta(minutes=7)).words


def test_cell_paper_gap_is_drift_since_the_anchor_so_a_starting_difference_is_not_a_gap(store):
    # the demo account starts at 50,000 while the book holds 100,000: no gap, no alert
    client = Recorded(wallet("50000"), wallet("50100"), wallet("49000"))
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100000"), now=T)
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100100"), now=T + timedelta(minutes=5))
    v = ac.view(store, "paper", T + timedelta(minutes=5))
    assert v.gap == 0 and not v.alerting and "drift since the anchor" in v.basis
    assert [r["anchor"] for r in store.account_reads("paper", limit=5)] == [0, 1]
    ac.fetch_account(store, "paper", client=client, live=False, book=D("100100"), now=T + timedelta(minutes=10))
    v = ac.view(store, "paper", T + timedelta(minutes=10))
    assert v.gap == D("-1100") and v.alerting and _kinds(store) == [ac.GAP_ALERT]


# --- the setting ---------------------------------------------------------------------------------------------------

def test_the_setting_ships_off_and_on_is_refused_in_words_on_paper():
    assert ac.setting_words({}) == (False, "Size from the account figure: Off. Entries are sized from the book")
    on, words = ac.setting_words(ON, live=False)
    assert not on and words.startswith("Size from the account figure: refused")
    assert ac.setting_words(ON, live=True)[0]


def test_off_sizes_from_the_book_even_on_a_live_account_with_no_read(store):
    assert ac.entry_capital(store, "live-1", D("100000"), T, live=True, environ={}) == (D("100000"), None)


# --- the supervisor's pass, through the demo client, on the recorded answer ------------------------------------------

def test_the_supervisor_reads_the_paper_account_from_the_demo_host_every_5_minutes(store, monkeypatch):
    import json
    from pathlib import Path

    from sleeve_fund import mirror, supervisor

    answer = json.loads((Path(__file__).parent / "data" / "b23" / "wallet-balance-unified.json").read_text())
    urls = []

    def http(method, url, headers, body=None):
        urls.append((method, url))
        return answer

    real = mirror.fee_client
    monkeypatch.setattr(mirror, "fee_client", lambda env: mirror.BybitDemo(real(env)._creds, http=http))
    monkeypatch.setattr(supervisor, "fund_equity", lambda s: D("101000"))
    env = {mirror.BYBIT_KEY_ENV: "k", mirror.BYBIT_SECRET_ENV: "s"}
    sup = supervisor.Supervisor(store)
    sup.check_account_capital(env, now=T)
    sup.check_account_capital(env, now=T + timedelta(minutes=1))  # not due yet
    assert urls == [("GET", "https://api-demo.bybit.com/v5/account/wallet-balance?accountType=UNIFIED")]
    row = store.account_reads("paper")[0]
    assert row["ok"] and row["anchor"] and row["payload"]["equity"] == "100987.65"  # 101,000.00 - 12.35, not available
    assert row["payload"]["other"] == [] and row["payload"]["book"] == "101000"


def test_with_no_demo_key_the_read_fails_once_said_and_trading_goes_on(store, monkeypatch):
    from sleeve_fund import supervisor

    monkeypatch.setattr(supervisor, "fund_equity", lambda s: D("100000"))
    supervisor.Supervisor(store).check_account_capital({}, now=T)
    assert store.account_reads("paper")[0]["reason"] == "this server has no read-only key for the account"
    assert "trading goes on" in store.events_of((ac.READ_FAILED,))[0]["message"]


# --- the start check ------------------------------------------------------------------------------------------------

def test_the_start_check_notes_the_account_and_the_setting_on_paper(store):
    from sleeve_fund.start_check import FAIL, NOTE, check_account_capital

    sec = check_account_capital(store, ON, now=T)
    assert not sec.failed and [s for s, _ in sec.lines] == [NOTE, NOTE]
    assert sec.lines[0][1].startswith("Account capital: never read")
    assert sec.lines[1][1].startswith("Size from the account figure: refused")
    assert FAIL not in [s for s, _ in check_account_capital(store, {}, now=T).lines]


def test_0033_downgrade_refuses_while_reads_are_kept(tmp_path):
    from alembic import command

    from sleeve_fund import schema

    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/m.db")
    _clean(eng)
    with eng.begin() as conn:  # built at 0033 itself: from head, "-1" runs the newest migration's downgrade (TEC, DA 0034)
        command.upgrade(schema._config(conn), "0033")
    with eng.begin() as c:
        c.execute(text("INSERT INTO account_capital_fetch (account, fetched_at, ok, reason, anchor) "
                       "VALUES ('paper', :t, 0, 'timeout', 0)"), {"t": T})
    with pytest.raises(NotImplementedError, match="holds reads: they are never dropped"), eng.begin() as conn:
        command.downgrade(schema._config(conn), "-1")
    with eng.begin() as c:
        c.execute(text("DROP TABLE account_capital_fetch"))
    _clean(eng)


# --- CR2 (21:05 UK) ------------------------------------------------------------------------------------------------

def test_cr2_must1_an_unmarked_book_is_a_failed_read_said_once_not_every_pass(store, monkeypatch):
    from sleeve_fund import mirror, supervisor

    def unmarked(s):
        raise RuntimeError("BTC/USDT has no mark")

    monkeypatch.setattr(supervisor, "fund_equity", unmarked)
    env = {mirror.BYBIT_KEY_ENV: "k", mirror.BYBIT_SECRET_ENV: "s"}
    sup = supervisor.Supervisor(store)
    for minutes in (0, 5, 10):
        sup.check_account_capital(env, now=T + timedelta(minutes=minutes))
    rows = store.account_reads("paper", limit=5)
    assert len(rows) == 3 and not any(r["ok"] for r in rows)
    assert rows[0]["reason"].startswith("the book can't be marked yet (RuntimeError: BTC/USDT has no mark")
    assert _kinds(store) == [ac.READ_FAILED] and store.events_of(("account_capital_pass_failed",)) == []


def test_cr2_must2_the_alert_line_is_never_below_50_usdt(store):
    client = Recorded(wallet("5000"), wallet("5040"), wallet("5060"))
    ac.fetch_account(store, "paper", client=client, live=False, book=D("5000"), now=T)
    ac.fetch_account(store, "paper", client=client, live=False, book=D("5000"), now=T + timedelta(minutes=5))
    v = ac.view(store, "paper", T + timedelta(minutes=5))
    assert v.alert_at == D("50") and v.gap == D("40") and not v.alerting
    ac.fetch_account(store, "paper", client=client, live=False, book=D("5000"), now=T + timedelta(minutes=10))
    assert ac.view(store, "paper", T + timedelta(minutes=10)).alerting and _kinds(store) == [ac.GAP_ALERT]


def test_cr2_boundaries_on_the_line_is_inside(store):
    # follow-up a: exactly on the alert line, exactly 2%, exactly 5 minutes old are all inside
    _live_read(store, "live-1", "99500", "100000", T)
    v = ac.view(store, "live-1", T + timedelta(minutes=5), live=True)
    assert v.gap == D("-500") and not v.alerting and v.state == ac.FRESH
    _live_read(store, "live-1", "98000", "100000", T + timedelta(minutes=1))
    assert not ac.view(store, "live-1", T + timedelta(minutes=1), live=True).over_refuse


def test_cr2_the_start_check_fails_when_the_setting_is_on_and_a_live_account_holds_a_strategy(store):
    # follow-up b
    from sleeve_fund.start_check import check_account_capital

    store.create_account("live-1", "live", venue="BINANCE")
    store.create_sleeve(name="a", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    store.assign_account("a", "live-1")
    sec = check_account_capital(store, ON, now=T)
    assert sec.failed and "isn't wired into the entry check" in sec.lines[-1][1]
    assert not check_account_capital(store, {}, now=T).failed
