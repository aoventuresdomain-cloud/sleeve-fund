"""B-23 (Advisor B23-ACCOUNT-CAPITAL, 20 Oct 2026): the fund's capital as the venue account holds it, next to the
book, and the gap between them.

- The read: the account's marked equity, wallet plus unrealised P&L in USDT (the Book.equity basis), from the signed,
  read-only wallet-balance call. Never the available balance. Other assets are ignored and flagged. Every read, good or
  failed, is one row in account_capital_fetch with the book at that moment.
- PAPER: Book.equity is the only sizing source. The read is shown and alerted and never refuses or sizes an entry. The
  gap is drift since a journaled anchor (the first good read), so a fixed starting difference between the demo account
  and the book is not a gap.
- LIVE (after G2): only with "Size from the account figure" ON. No good read in the last 5 minutes refuses entries;
  sizing uses the lower of account and book; a gap above 2% refuses entries until it is back inside and the PM has
  acknowledged the alert. Exits, stops and flatten never read this.
- Both: a gap above max(0.5% of book, 50 USDT) is a warning alert.
- Before G2 a live account is never read (demo and paper keys only): its reads are kept as failed, saying so.

The words here reach the dashboard, the start check and the alerts: they never name the venue."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from sleeve_fund.venue_terms import uk

SETTING_ENV = "SIZE_FROM_ACCOUNT"
SETTING_NAME = "Size from the account figure"
WALLET_PATH = "/v5/account/wallet-balance"
QUOTE = "USDT"
PAPER_EVERY = timedelta(minutes=5)  # paper only shows and alerts
LIVE_EVERY = timedelta(minutes=1)  # live must have a good read inside FRESH_FOR
FRESH_FOR = timedelta(minutes=5)
ALERT_SHARE, ALERT_FLOOR = Decimal("0.005"), Decimal("50")
REFUSE_SHARE = Decimal("0.02")
FRESH, STALE, NEVER = "fresh", "stale", "never"
GAP_ALERT, GAP_CLOSED, GAP_REFUSED = "account_gap", "account_gap_closed", "account_gap_refused"
GAP_INSIDE, ANCHORED = "account_gap_inside", "account_anchor"
READ_FAILED, READ_BACK = "account_read_failed", "account_read_back"
NOT_BEFORE_G2 = "a live account's capital is not read before G2: demo and paper keys only"


def setting_on(environ=None) -> bool:
    """The PM's switch, OFF unless SIZE_FROM_ACCOUNT=on on the server."""
    env = os.environ if environ is None else environ
    return (env.get(SETTING_ENV) or "").strip().lower() == "on"


def setting_words(environ=None, live: bool = False) -> tuple[bool, str]:
    """(whether the setting takes effect, what to say). It ships OFF; ON is refused in words on paper, where the book
    is the only sizing source."""
    if not setting_on(environ):
        return False, f"{SETTING_NAME}: Off. Entries are sized from the book"
    if not live:
        return False, (f"{SETTING_NAME}: refused. It is On, but the fund trades on paper, where entries are sized "
                       "from the book only; it takes effect on a live account after G2")
    return True, f"{SETTING_NAME}: On. Entries are sized from the lower of the account and the book"


def _dec(x, what: str) -> Decimal:
    if not isinstance(x, str):
        raise ValueError(f"the account's {what} isn't a decimal string: {x!r}")
    try:
        d = Decimal(x)
    except InvalidOperation:
        raise ValueError(f"the account's {what} isn't a number: {x!r}") from None
    if not d.is_finite():
        raise ValueError(f"the account's {what} isn't a number: {x!r}")
    return d


def wallet_payload(result: dict) -> dict:
    """The wallet-balance call's result as the stored payload: the USDT wallet and unrealised P&L as the venue wrote
    them, and their sum, the marked equity. The available balance is never read. Other assets held are named in
    "other" and left out of the figure. Either figure missing, empty or not a number fails the read: never read as 0
    (Advisor MUST: a zero would show a 100% gap on paper and size at nothing live)."""
    rows = (result or {}).get("list") or []
    if len(rows) != 1:
        raise ValueError(f"the account's answer has {len(rows)} wallets, not one")
    coins = rows[0].get("coin") or []
    usdt = [c for c in coins if c.get("coin") == QUOTE]
    if len(usdt) > 1:
        raise ValueError(f"the account's answer lists {QUOTE} twice")
    if not usdt:
        raise ValueError(f"the account's answer has no {QUOTE} balance")
    w, u = usdt[0].get("walletBalance"), usdt[0].get("unrealisedPnl")
    wallet, unrealised = _dec(w, "wallet balance"), _dec(u, "unrealised P&L")
    other = sorted(str(c.get("coin")) for c in coins if c.get("coin") != QUOTE and _nonzero(c))
    return {"coin": QUOTE, "wallet": w, "unrealised": u, "equity": str(wallet + unrealised), "other": other}


def _nonzero(coin: dict) -> bool:
    for k in ("walletBalance", "equity"):
        try:
            if Decimal(str(coin.get(k) or "0")) != 0:
                return True
        except InvalidOperation:
            return True  # unreadable is flagged, not hidden
    return False


@dataclass(frozen=True)
class View:
    """The account against the book, as the dashboard, start check and alerts read it."""

    account: str
    live: bool
    state: str  # FRESH (a good read inside FRESH_FOR), STALE (an older good read), NEVER
    read_at: datetime | None
    account_equity: Decimal | None
    book: Decimal | None  # the book at the read, so both sides are the same moment
    gap: Decimal | None  # account less book (live), or that drift since the anchor (paper)
    share: Decimal | None  # |gap| / book
    alert_at: Decimal | None  # max(0.5% of book, 50 USDT)
    basis: str  # how the gap is measured, in words
    other: tuple[str, ...]
    failure: str | None  # the newest read's reason, when it failed
    age_min: int | None = None  # whole minutes since the read

    @property
    def age_words(self) -> str:
        return "" if self.age_min is None else f"{self.age_min} min ago"

    @property
    def alerting(self) -> bool:
        return self.gap is not None and abs(self.gap) > self.alert_at

    @property
    def over_refuse(self) -> bool:
        """Above 2% of book: refuses live entries only (paper never)."""
        return self.share is not None and self.share > REFUSE_SHARE

    @property
    def words(self) -> str:
        if self.state == NEVER:
            why = f" (last attempt: {self.failure})" if self.failure else ""
            return f"Account capital: never read{why}"
        head = (f"Account {_m(self.account_equity)} {QUOTE} vs book {_m(self.book)} {QUOTE} at {uk(self.read_at)}; "
                f"gap {_signed(self.gap)} {QUOTE} ({self.share * 100:.2f}% of book, {self.basis})")
        if self.state == STALE:
            head += f"; last good read {self.age_words}"
        if self.failure:
            head += f"; last attempt failed: {self.failure}"
        if self.other:
            head += f"; other assets in the account ignored: {', '.join(self.other)}"
        return head


def _m(x: Decimal) -> str:
    return f"{x:,.2f}"


def _signed(x: Decimal) -> str:
    return f"{x:+,.2f}"


def _utc(t: datetime) -> datetime:
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def view(store, account: str, now: datetime, live: bool = False) -> View:
    """`account` against the book at its newest good read."""
    newest = store.account_reads(account)
    failure = newest[0]["reason"] if newest and not newest[0]["ok"] else None
    good = store.account_reads(account, good=True)
    if not good:
        return View(account, live, NEVER, None, None, None, None, None, None, "", (), failure)
    row = good[0]
    p = row["payload"]
    acct, book = Decimal(p["equity"]), Decimal(p["book"])
    if live:
        gap, basis = acct - book, "account less book"
    else:
        a = store.account_reads(account, anchor=True)
        a0, b0 = (Decimal(a[0]["payload"]["equity"]), Decimal(a[0]["payload"]["book"])) if a else (acct, book)
        at0 = a[0]["fetched_at"] if a else row["fetched_at"]
        gap, basis = (acct - a0) - (book - b0), f"drift since the anchor read at {uk(at0)}"
    share = abs(gap) / book if book > 0 else Decimal(1)
    alert_at = max(book * ALERT_SHARE, ALERT_FLOOR)
    age = _utc(now) - row["fetched_at"]
    state = FRESH if age <= FRESH_FOR else STALE
    return View(account, live, state, row["fetched_at"], acct, book, gap, share, alert_at, basis,
                tuple(p.get("other") or ()), failure, max(int(age.total_seconds() // 60), 0))


def due(store, account: str, now: datetime, live: bool) -> bool:
    newest = store.account_reads(account)
    return not newest or _utc(now) - newest[0]["fetched_at"] >= (LIVE_EVERY if live else PAPER_EVERY)


def fetch_account(store, account: str, *, client, live: bool, book: Decimal, now: datetime | None = None) -> None:
    """One read of `account`'s marked equity (client: a signed, read-only V5 client's call(method, path, params), or
    None when this server has no key for it), kept with the book at that moment. A paper account's first good read is
    the anchor. A live account is never read before G2. Then the alerts: a gap past the alert line, a gap past 2% on a
    live account (which the PM acknowledges), and a read failing, each once when it starts and once when it ends."""
    at = now or datetime.now(timezone.utc)
    before = view(store, account, at, live)
    before_failed = bool(store.account_reads(account)) and not store.account_reads(account)[0]["ok"]
    reason = None
    if live:
        reason = NOT_BEFORE_G2
    elif client is None:
        reason = "this server has no read-only key for the account"
    if reason is None:
        try:
            payload = wallet_payload(client.call("GET", WALLET_PATH, {"accountType": "UNIFIED"}))
        except Exception as e:  # noqa: BLE001 - every failed read is kept, with why
            reason = f"{type(e).__name__}: {e}"[:500]
    if reason is not None:
        store.record_account_read(account, ok=False, reason=reason, at=at)
        if not before_failed:
            store.event(None, "warning", READ_FAILED,
                        f"Account capital read failed ({reason}); "
                        + ("live entries are refused while it lasts; exits and stops still run" if live else
                           "paper sizing is from the book, so trading goes on"))
        return
    payload["book"] = str(book)
    old = None if live else store.account_reads(account, anchor=True)
    why = None
    if old == []:
        why = "the first good read"
    elif old:
        why = _reanchor_reason(store, old[0])
    if why:
        payload["anchor_reason"] = why
    store.record_account_read(account, ok=True, payload=payload, at=at, anchor=bool(why))
    if why and old:
        o = old[0]["payload"]
        store.event(None, "info", ANCHORED, f"Account gap re-anchored ({why}): was account {_m(Decimal(o['equity']))} "
                    f"vs book {_m(Decimal(o['book']))} at {uk(old[0]['fetched_at'])}, now account "
                    f"{_m(Decimal(payload['equity']))} vs book {_m(book)} at {uk(at)}")
    if before_failed:
        store.event(None, "info", READ_BACK, "Account capital read works again")
    after = view(store, account, at, live)
    if after.alerting and not before.alerting:
        store.event(None, "warning", GAP_ALERT, f"{after.words}; over the alert line of {_m(after.alert_at)} {QUOTE}"
                    + ("" if live else "; paper sizing is from the book, so trading goes on"))
    elif before.alerting and not after.alerting:
        store.event(None, "info", GAP_CLOSED, f"Account and book agree again: {after.words}")
    if live and after.over_refuse and not before.over_refuse:
        store.event(None, "error", GAP_REFUSED, f"{after.words}; over 2% of book, so new entries are refused until "
                    "the gap is reconciled and you acknowledge this alert; exits and stops still run")
    elif live and before.over_refuse and not after.over_refuse:
        store.event(None, "info", GAP_INSIDE, f"Account gap back inside 2%: {after.words}; entries wait for your "
                    "acknowledgement of the gap alert")


def _reanchor_reason(store, anchor: dict) -> str | None:
    """Why a paper anchor no longer holds, or None (Advisor Q2): a deposit or withdrawal booked to the fund since (a
    REFRESH books one too), or a clean slate. The demo account receives neither, so the drift would jump by it."""
    at = anchor["fetched_at"]
    for f in store.fund_flows():
        if f.get("applied_at") is not None and _utc(f["applied_at"]) > at:
            return f"a fund {f['kind']} of {_m(Decimal(str(f['amount'])))} booked at {uk(_utc(f['applied_at']))}"
    start = store.book_start()
    if start is not None and _utc(start) > at:
        return f"the book was started again at {uk(_utc(start))}"
    return None


def entry_capital(store, account: str, book: Decimal, now: datetime, live: bool,
                  environ=None) -> tuple[Decimal, str | None]:
    """The capital an entry is sized from, and why it is refused (or None). OFF, or on paper: the book, never refused
    here. ON on a live account: no good read in 5 minutes refuses; a gap over 2%, or its alert not yet acknowledged,
    refuses; else the lower of account and book. Only entries ask: exits, stops and flatten never do."""
    takes_effect, _ = setting_words(environ, live)
    if not takes_effect:
        return book, None
    v = view(store, account, now, live=True)
    tail = "; exits, stops and flatten still run"
    if v.state != FRESH:
        return book, f"Entry refused: no good read of the account's capital in the last 5 minutes{tail}"
    if v.over_refuse:
        return book, f"Entry refused: {v.words}; over 2% of book until reconciled and acknowledged{tail}"
    told = store.last_event(None, (GAP_REFUSED,))
    if told is not None:
        # Advisor Q3: only an acknowledgement made once the gap is back inside 2% lifts the refusal
        inside = store.last_event(None, (GAP_INSIDE,))
        ack = store.ack_of(told["id"])
        back = inside["ts"] if inside is not None and inside["id"] > told["id"] else None
        if back is None or ack is None or _utc(ack["ts"]) < _utc(back):
            return book, (f"Entry refused: the account gap alert of {uk(_utc(told['ts']))} waits for your "
                          f"acknowledgement, given once the gap is back inside 2%{tail}")
    return min(v.account_equity, book), None
