"""Fail-closed checks that keep the paper process keyless and unable to trade."""

from __future__ import annotations

import os
import re

# Any credential-looking variable for a venue we connect to. The paper process
# needs none; if one is present the environment is wrong, so refuse to start.
_CREDENTIAL_ENV = re.compile(r"^(KRAKEN|BINANCE|COINBASE|IB|BYBIT|OKX)_.*(KEY|SECRET|PASSPHRASE|TOKEN)", re.I)


class PaperSafetyError(RuntimeError):
    pass


def assert_keyless(environ: dict[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    found = sorted(k for k, v in env.items() if v and _CREDENTIAL_ENV.match(k))
    if found:
        raise PaperSafetyError(
            "paper trading refuses to start with venue credentials in the environment: "
            + ", ".join(found)
            + ". Unset them; paper needs public data only."
        )
