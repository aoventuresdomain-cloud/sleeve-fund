"""B-23 display (Advisor B23-ACCOUNT-CAPITAL): the account's marked equity against the book, and the gap, on Portfolio
and Risk & health. Display only, and the engine's own reading (account_capital.view, which the start check and the
alerts also read), so the page can't disagree with them. On paper it never refuses or sizes anything."""

from __future__ import annotations

from datetime import datetime

from sleeve_fund import account_capital as ac
from sleeve_fund.accounts import PAPER


def account_view(store, now: datetime, environ=None) -> dict:
    v = ac.view(store, PAPER, now)
    tone = "loss" if v.alerting else ("warn" if v.state != ac.FRESH or v.failure else "")
    return {"never": v.state == ac.NEVER, "account": v.account_equity, "book": v.book, "gap": v.gap,
            "share": v.share, "alerting": v.alerting, "alert_at": v.alert_at, "basis": v.basis, "read_at": v.read_at,
            "stale": v.state == ac.STALE, "age": v.age_words, "failure": v.failure, "other": v.other, "words": v.words, "tone": tone,
            "setting": ac.setting_words(environ)[1]}
