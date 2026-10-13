"""Fail-closed checks that keep the paper process keyless and unable to trade."""

from __future__ import annotations

import os
import re

# Any credential-looking variable for a venue we connect to. The paper process
# needs none; if one is present the environment is wrong, so refuse to start.
_CREDENTIAL_ENV = re.compile(r"^(KRAKEN|BINANCE|COINBASE|IB|BYBIT|OKX|DERIBIT)_.*(KEY|SECRET|PASSPHRASE|TOKEN)", re.I)
# Live account keys for any venue, as VenueProfile.key_env names them: <VENUE>_API_KEY__<ACCOUNT>.
_ACCOUNT_KEY_ENV = re.compile(r"^[A-Z0-9]+_API_(KEY|SECRET)__", re.I)


def credential_var(name: str) -> bool:
    return bool(_CREDENTIAL_ENV.match(name) or _ACCOUNT_KEY_ENV.match(name))


class PaperSafetyError(RuntimeError):
    pass


def assert_keyless(environ: dict[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    found = sorted(k for k, v in env.items() if v and credential_var(k))
    if found:
        raise PaperSafetyError(
            "paper trading refuses to start with venue credentials in the environment: "
            + ", ".join(found)
            + ". Unset them; paper needs public data only."
        )


def assert_portfolio_gate(store) -> None:
    """Paper and live run under the portfolio gate (Advisor 06:10, 8 Oct): a journal opened without it, the opt-out
    kept for single-strategy research backtests, is refused at startup."""
    if not getattr(store, "portfolio_gate", False):
        raise PaperSafetyError("paper trading refuses to start on a journal without the portfolio gate: that opt-out is "
                               "for single-strategy research backtests only")
