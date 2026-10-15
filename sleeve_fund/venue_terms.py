"""FEE-LIVE (Advisor FEE-LIVE-DW, 15 Oct 2026): the paper venue's fee rates and margin tiers come from the venue, not
from the code, and how old they are decides whether entries may go out.

- Fees per instrument from the account's fee-rate endpoint (signed, read-only), tiers per instrument daily from the
  public risk-limit endpoint. The supervisor fetches; every attempt, good or failed, is one row in the DA's
  venue_terms_fetch (Store.record_venue_fetch), and the newest good row is last-known-good.
- Age: up to 24 h fresh; past 24 h stale (alerted, once a day, and trading goes on); past 72 h, or never fetched,
  refused (the start check fails and entries are refused). Exits, stops, the liquidation cut and Stop always run: only
  the entry path reads entry_refusal.
- A fetched fee outside the profile's published [lowest, VIP 0] range, or a tier table the engine's margin.tiers
  refuses, is kept as failed and is missing: the last good value stays, and it is never a wider range.
- A demo account with no fee endpoint: paper is charged the published VIP 0, a good row labelled "published, not
  account" written at each attempt, so it ages like any other if the attempts stop. A live account never uses it.
- The tier table the PM pasted on 15 Oct (hand-checked by the DA) counts as a fetch dated 15 Oct, until the daily pull
  lands.
- Research never charges below the published VIP 0 (research_fees).

The words here reach the start check, alerts and the dashboard: they never name the venue."""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.store import PUBLISHED_FEES, margin_tiers_of

ALERT_AFTER = timedelta(hours=24)
REFUSE_AFTER = timedelta(hours=72)
FRESH, STALE, REFUSED = "fresh", "stale", "refused"
FEE_RATE, RISK_LIMIT, PASTED = "fee-rate", "risk-limit", "hand-checked-pasted"
FEE_RATE_PATH = "/v5/account/fee-rate"
# The paper venue's public V5 risk-limit page: public market data, read-only, no key.
RISK_LIMIT_URL = "https://api.bybit.com/v5/market/risk-limit?category=linear&symbol={symbol}"
KIND_WORDS = {"fee": "fee rates", "tiers": "margin tiers"}
SOURCE_WORDS = {FEE_RATE: "the account", PUBLISHED_FEES: "the published VIP 0 schedule (published, not account)",
                RISK_LIMIT: "the public risk-limit table", PASTED: "the table pasted on 15 Oct, hand-checked"}


def symbol_of(instrument: str) -> str:
    """The venue's own symbol for an engine pair: BTC/USDT -> BTCUSDT."""
    return instrument.replace("/", "").upper()


def _row_for(result: dict, symbol: str) -> dict:
    rows = [r for r in (result or {}).get("list") or [] if r.get("symbol") == symbol]
    if not rows:
        raise ValueError(f"the venue's answer has no {symbol} in it")
    return rows


def fee_payload(result: dict, symbol: str) -> dict:
    """The fee-rate endpoint's result for `symbol` as the stored payload, the venue's own decimal strings."""
    r = _row_for(result, symbol)[0]
    maker, taker = r.get("makerFeeRate"), r.get("takerFeeRate")
    if not isinstance(maker, str) or not isinstance(taker, str):
        raise ValueError(f"the venue's fee rates for {symbol} aren't decimal strings: {maker!r}, {taker!r}")
    Decimal(maker), Decimal(taker)  # noqa: B018 - raises on a value that isn't a number
    return {"symbol": symbol, "maker": maker, "taker": taker}


def tier_payload(result: dict, symbol: str) -> dict:
    """The risk-limit endpoint's result for `symbol` as the stored payload: its tiers by rising ceiling, each the
    venue's own strings, a blank deduction kept as "0" (vs1-tiers.md)."""
    rows = sorted(_row_for(result, symbol), key=lambda r: Decimal(r["riskLimitValue"]))
    return {"symbol": symbol, "tiers": [
        {"ceiling": r["riskLimitValue"], "mmr": r["maintenanceMargin"], "imr": r["initialMargin"],
         "max_leverage": r["maxLeverage"], "deduction": r.get("mmDeduction") or "0"} for r in rows]}


def _failed(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:500] or "failed"


def _strings(fees: FeeSchedule, symbol: str) -> dict:
    return {"symbol": symbol, "maker": str(fees.maker), "taker": str(fees.taker)}


def fetch_fees(store, profile, instrument: str, *, client, account: str, live: bool, now: datetime | None = None) -> None:
    """One attempt at the account's fee rates for `instrument` (client: a signed V5 client's call(method, path,
    params)). The answer is kept, checked against the profile's published bounds; a refusal or a bad answer is kept as
    failed. On paper a failed attempt also keeps the published VIP 0 as a good row labelled published, not account
    (Advisor HEAD3, FEE-LIVE-DW (a)); live never does, and its entries are refused until the account answers."""
    symbol, at = symbol_of(instrument), now or datetime.now(timezone.utc)
    try:
        payload = fee_payload(client.call("GET", FEE_RATE_PATH, {"category": "linear", "symbol": symbol}), symbol)
    except Exception as e:  # noqa: BLE001 - every failed attempt is kept, with why
        store.record_venue_fetch(profile.name, "fee", instrument, source=FEE_RATE, ok=False, reason=_failed(e),
                                 account=account, at=at)
        if not live:
            store.record_venue_fetch(profile.name, "fee", instrument, source=PUBLISHED_FEES, ok=True,
                                     payload=_strings(profile.fees, symbol), account=account, at=at,
                                     bounds=profile.fee_bounds)
        return
    store.record_venue_fetch(profile.name, "fee", instrument, source=FEE_RATE, ok=True, payload=payload,
                             account=account, at=at, bounds=profile.fee_bounds)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=20) as r:
        return json.load(r)


def fetch_tiers(store, profile, instrument: str, *, get=_get_json, now: datetime | None = None) -> None:
    """One attempt at the public margin tiers for `instrument`, kept good (if margin.tiers accepts it) or failed."""
    symbol, at = symbol_of(instrument), now or datetime.now(timezone.utc)
    try:
        answer = get(RISK_LIMIT_URL.format(symbol=symbol))
        if not isinstance(answer, dict) or answer.get("retCode") != 0:
            raise ValueError(f"the venue refused: {(answer or {}).get('retMsg', answer)}")
        payload = tier_payload(answer.get("result"), symbol)
    except Exception as e:  # noqa: BLE001 - every failed attempt is kept, with why
        store.record_venue_fetch(profile.name, "tiers", instrument, source=RISK_LIMIT, ok=False, reason=_failed(e),
                                 at=at)
        return
    store.record_venue_fetch(profile.name, "tiers", instrument, source=RISK_LIMIT, ok=True, payload=payload, at=at)


def _num(x: float) -> str:
    return format(Decimal(repr(float(x))).normalize(), "f")


def seed_pasted_tiers(store, profile) -> None:
    """The tables kept in the profile (the PM's 15 Oct paste, hand-checked) as good fetches dated the day they were read,
    once per instrument, where no tiers row exists yet (Advisor FEE-LIVE-DW (b)). The venue's last ceiling stands in
    for the open last tier, as the venue publishes it."""
    if not profile.maintenance_read_on:
        return
    at = datetime.combine(profile.maintenance_read_on, datetime.min.time(), tzinfo=timezone.utc)
    for pair, table in profile.maintenance_tiers.items():
        if store.latest_venue_fetch(profile.name, "tiers", pair) is not None:
            continue
        last = profile.maintenance_limits.get(pair)
        rows = [{"ceiling": _num(last if math.isinf(t.ceiling) and last else t.ceiling), "mmr": _num(t.rate),
                 "imr": "", "max_leverage": "", "deduction": _num(t.amount)} for t in table]
        store.record_venue_fetch(profile.name, "tiers", pair, source=PASTED, ok=True, at=at,
                                 payload={"symbol": symbol_of(pair), "tiers": rows})


@dataclass(frozen=True)
class Term:
    """One instrument's fee rates or margin tiers as the engine holds them now."""

    kind: str  # "fee" or "tiers"
    instrument: str
    state: str  # FRESH, STALE or REFUSED
    source: str | None  # the good row's source; None when never fetched
    fetched_at: datetime | None
    account: str | None
    value: object  # (maker, taker) Decimals for fees, the margin.Tier table for tiers; None when never fetched
    words: str  # one line for the start check, alerts and the dashboard; never the venue's name

    @property
    def hours(self) -> int | None:
        return None if self.fetched_at is None else self._hours

    _hours: int = 0


def _utc(t: datetime) -> datetime:
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def term(store, profile, kind: str, instrument: str, now: datetime, live: bool = False) -> Term:
    """The last-known-good fee rates or tiers for `instrument` and how old they are (FRESH, STALE, REFUSED)."""
    what = f"{instrument} {KIND_WORDS[kind]}"
    if kind == "fee" and not profile.fee_bounds:
        return Term(kind, instrument, REFUSED, None, None, None, None,
                    f"{what}: no published range to check a fetched rate against, so none is used")
    row = store.latest_good_venue_fetch(profile.name, kind, instrument)
    if row is None:
        return Term(kind, instrument, REFUSED, None, None, None, None,
                    f"{what}: never fetched, so entries are refused; exits and stops still run")
    if live and row["source"] == PUBLISHED_FEES:
        return Term(kind, instrument, REFUSED, row["source"], _utc(row["fetched_at"]), row.get("account"), None,
                    f"{what}: the live account's own rate hasn't been fetched and the published rate is never used "
                    "live, so entries are refused; exits and stops still run")
    at = _utc(row["fetched_at"])
    age = _utc(now) - at
    hours = max(int(age.total_seconds() // 3600), 0)
    value = ((Decimal(row["payload"]["maker"]), Decimal(row["payload"]["taker"])) if kind == "fee"
             else margin_tiers_of(row))
    src = SOURCE_WORDS.get(row["source"], row["source"])
    when = f"fetched {at.day} {at:%b %H:%M} UTC from {src}"
    priced = profile.maintenance_tiers.get(instrument) if kind == "tiers" else None
    if kind == "tiers" and priced is not None and tuple(priced) != tuple(value):
        # Liquidation is priced with the profile's table (Advisor question (b), PE1's default): a pulled table that
        # differs is never used silently; entries wait until it is re-pasted and checked in.
        state, words = REFUSED, (f"{what}: the table {when} differs from the table liquidation is priced with, so "
                                 "entries are refused until it is re-pasted and checked (DA-11); exits and stops "
                                 "still run")
    elif age > REFUSE_AFTER:
        state, words = REFUSED, (f"{what}: last {when}, {hours} hours ago, over 72 hours, so entries are refused "
                                 "until they are fetched again; exits and stops still run")
    elif age > ALERT_AFTER:
        state, words = STALE, f"{what}: last {when}, {hours} hours ago, over 24 hours; trading goes on"
    else:
        state, words = FRESH, f"{what}: {when}"
    t = Term(kind, instrument, state, row["source"], at, row.get("account"), value, words)
    object.__setattr__(t, "_hours", hours)
    return t


def entry_refusal(store, profile, instrument: str, now: datetime, live: bool = False) -> str | None:
    """Why an entry on `instrument` is refused for its fee rates or tiers (over 72 h old, or never fetched), or None.
    Only entries read this: exits, stops, the liquidation cut and Stop never do."""
    for kind in ("fee", "tiers"):
        t = term(store, profile, kind, instrument, now, live)
        if t.state == REFUSED:
            return f"Entry refused: {t.words}"
    return None


class _NoClient:
    """Stands in for an account with no fee client on this server: every attempt is kept as failed, saying why."""

    def __init__(self, why: str):
        self.why = why

    def call(self, *a, **k):
        raise RuntimeError(self.why)


TIER_EVERY = timedelta(hours=20)  # a tier table younger than this isn't pulled again: daily, with room before 24 h


def fetch_due(store, now: datetime, client_for, get=_get_json) -> None:
    """The supervisor's hourly pass (Supervisor.check_fees): for every running perp Strategy's instrument on a FEE-LIVE
    venue, the pasted tables seeded once, the account's fee rates fetched (client_for(account, live) gives its signed
    client, or None when this server has none for it), and the public tiers pulled when the last good table is 20 h
    old or more. Each attempt is one row, so the 24 h and 72 h rules read the last good one."""
    from sleeve_fund.venues import venue as venue_profile

    seen = set()
    for s in store.sleeves():
        if s.lifecycle == "retired" or (s.params or {}).get("market") != "perp":
            continue
        profile = venue_profile(s.venue)
        if not profile.fee_bounds:
            continue
        account = store.account_of(s.name)
        live = store.account_kind(account) == "live"
        if (profile.name, s.instrument, account) in seen:
            continue
        seen.add((profile.name, s.instrument, account))
        seed_pasted_tiers(store, profile)
        client = client_for(account, live)
        fetch_fees(store, profile, s.instrument, account=account, live=live, now=now,
                   client=client or _NoClient("this server has no fee-rate client for the account"))
        good = store.latest_good_venue_fetch(profile.name, "tiers", s.instrument)
        if good is None or now - _utc(good["fetched_at"]) >= TIER_EVERY:
            fetch_tiers(store, profile, s.instrument, get=get, now=now)


def live_terms(store, now: datetime) -> list[Term]:
    """Every running perp Strategy's instrument on a FEE-LIVE venue, its fee rates then its tiers, once per instrument
    and account kind: what the start check, the alerts and the dashboard read (FE's reader: live_terms(store, now))."""
    from sleeve_fund.venues import venue as venue_profile

    out, seen = [], set()
    for s in store.sleeves():
        if s.lifecycle == "retired" or (s.params or {}).get("market") != "perp":
            continue
        profile = venue_profile(s.venue)
        if not profile.fee_bounds:
            continue
        live = store.account_kind(store.account_of(s.name)) == "live"
        if (profile.name, s.instrument, live) in seen:
            continue
        seen.add((profile.name, s.instrument, live))
        out += [term(store, profile, kind, s.instrument, now, live) for kind in ("fee", "tiers")]
    return out


@dataclass(frozen=True)
class ResearchFees:
    fees: FeeSchedule
    text: str


def research_fees(store, profile, instrument: str | None = None) -> ResearchFees:
    """What research charges: never below the published VIP 0, maker and taker each (Advisor FEE-LIVE (4)), with the
    account's own rate beside it when one has been fetched, so a result shows both."""
    vip0 = profile.fees
    row = store.latest_good_venue_fetch(profile.name, "fee", instrument) if store is not None and instrument else None
    own = (Decimal(row["payload"]["maker"]), Decimal(row["payload"]["taker"])) if row else None
    fees = FeeSchedule(max(vip0.maker, own[0]) if own else vip0.maker, max(vip0.taker, own[1]) if own else vip0.taker)
    text = (f"Research fees: {float(fees.maker):.3%} maker, {float(fees.taker):.3%} taker (published VIP 0, never "
            "below it)")
    if own:
        text += f"; the account's own: {float(own[0]):.3%} maker, {float(own[1]):.3%} taker"
    return ResearchFees(fees, text)
