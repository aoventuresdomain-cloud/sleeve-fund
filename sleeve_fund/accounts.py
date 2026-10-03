"""Venue accounts a sleeve can trade on: the shared paper account, or a named Kraken
(sub-)account for live trading.

Keys never touch the database or the dashboard. A live account's key lives only on
the server, in kraken.env, which only the supervisor reads. The supervisor reports
whether a key is present (never its value) so the dashboard can show it.
"""

from __future__ import annotations

import os
import re

PAPER = "paper"
KINDS = ("paper", "live")
NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,40}")


def env_names(account: str, venue: str | None = None) -> tuple[str, str]:
    """The two environment variables that hold a live account's key and secret for its venue."""
    from sleeve_fund.venues import venue as venue_profile

    return venue_profile(venue).key_env(account)


def key_present(account: str, environ=None, venue: str | None = None) -> bool:
    env = os.environ if environ is None else environ
    return all(env.get(n, "").strip() for n in env_names(account, venue))


def credentials(account: str, venue: str | None = None, environ=None) -> tuple[str, str] | None:
    """Supervisor only: the account's key and secret from the server environment, or None."""
    env = os.environ if environ is None else environ
    key, secret = (env.get(n, "").strip() for n in env_names(account, venue))
    return (key, secret) if key and secret else None
