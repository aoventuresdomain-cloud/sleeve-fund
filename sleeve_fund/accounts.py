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


def env_names(account: str) -> tuple[str, str]:
    """The two environment variables that hold a live account's Kraken key and secret."""
    suffix = account.upper().replace("-", "_")
    return f"KRAKEN_API_KEY__{suffix}", f"KRAKEN_API_SECRET__{suffix}"


def key_present(account: str, environ=None) -> bool:
    env = os.environ if environ is None else environ
    return all(env.get(n, "").strip() for n in env_names(account))
