"""Where the exchange order client (EXEC-CLIENT) would send orders, read from the environment: the one reader the start
check's key and host line and the client's own guard share (head 3; PE3 agreed 15 Oct ~12:30 UK). Off by default: with
EXEC_MODE unset the client is the simulated paper one and nothing here applies.

    EXEC_MODE   demo or live (unset: off)
    EXEC_VENUE  the venue (BYBIT when unset)
    EXEC_HOST   the bare API host, e.g. api-demo.bybit.com
    and the venue's key and secret for that mode (KEYS), e.g. BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET in demo mode

A mismatch is refused, named: a production host in demo mode, a demo host in live mode, a host off the closed list, a
key or secret missing, the other mode's key set alongside. The client calls assert_ok and builds its URL from nothing
else. Whether a key is trade-only is asked of the venue when the client goes on (PE3, drill prep), so until then it is
noted as not yet checked, never shown as verified. Gates (drill go, the G2 lock) are the client's guard's, not this.

Pure: it imports nothing that opens a connection or places an order, so the read-only start check can use it. Key values
are never kept or printed: only whether each is set."""

from __future__ import annotations

from dataclasses import dataclass

MODE_ENV, VENUE_ENV, HOST_ENV = "EXEC_MODE", "EXEC_VENUE", "EXEC_HOST"
DEFAULT_VENUE = "BYBIT"
MODES = ("demo", "live")
# The closed list of hosts: (venue, mode) -> the hosts allowed in that mode.
HOSTS = {
    ("BYBIT", "demo"): ("api-demo.bybit.com",),
    ("BYBIT", "live"): ("api.bybit.com",),
}
# (venue, mode) -> (key variable, secret variable). The demo pair is the one the demo mirror already uses.
KEYS = {
    ("BYBIT", "demo"): ("BYBIT_DEMO_API_KEY", "BYBIT_DEMO_API_SECRET"),
    ("BYBIT", "live"): ("BYBIT_TRADE_API_KEY", "BYBIT_TRADE_API_SECRET"),
}
OFF = "off (simulated paper client)"
SCOPE_UNCHECKED = "key scope not yet checked: trade-only is verified with the venue when the order client goes on"


class ExecTargetRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    mode: str
    venue: str
    host: str
    key_var: str | None
    secret_var: str | None
    key_set: bool
    secret_set: bool
    other_set: tuple[str, ...]  # the other mode's key variables that are set (names only)


def _set(environ: dict, name: str | None) -> bool:
    return bool(name) and bool((environ.get(name) or "").strip())


def read(environ: dict) -> Target | None:
    """The order client's target from the environment, or None when it is off (EXEC_MODE unset)."""
    mode = (environ.get(MODE_ENV) or "").strip().lower()
    if not mode:
        return None
    venue = ((environ.get(VENUE_ENV) or "").strip() or DEFAULT_VENUE).upper()
    host = (environ.get(HOST_ENV) or "").strip().lower()
    key_var, secret_var = KEYS.get((venue, mode), (None, None))
    others = [v for m in MODES if m != mode for v in KEYS.get((venue, m), ())]
    return Target(mode=mode, venue=venue, host=host, key_var=key_var, secret_var=secret_var,
                  key_set=_set(environ, key_var), secret_set=_set(environ, secret_var),
                  other_set=tuple(v for v in others if _set(environ, v)))


def problems(t: Target | None) -> list[str]:
    """Everything wrong with the target, in plain words; none when it is off or sound."""
    if t is None:
        return []
    if t.mode not in MODES:
        return [f"unknown mode {t.mode!r} ({MODE_ENV} is demo or live)"]
    if not any(v == t.venue for v, _ in HOSTS):
        return [f"unknown venue {t.venue!r} for the order client"]
    out = []
    other = "live" if t.mode == "demo" else "demo"
    if not t.host:
        out.append(f"no host set ({HOST_ENV})")
    elif any(c in t.host for c in "/:@ "):
        out.append(f"{HOST_ENV} {t.host!r} isn't a bare host (no scheme, path or port)")
    elif t.host in HOSTS.get((t.venue, other), ()):
        kind = "a production" if other == "live" else "a demo"
        out.append(f"{kind} host ({t.host}) in {t.mode} mode")
    elif t.host not in HOSTS.get((t.venue, t.mode), ()):
        out.append(f"host {t.host} isn't a known {t.venue} host")
    if not t.key_set:
        out.append(f"the {t.mode} key isn't set ({t.key_var})")
    if not t.secret_set:
        out.append(f"the {t.mode} secret isn't set ({t.secret_var})")
    for v in t.other_set:
        out.append(f"a {other} key is set alongside the {t.mode} one ({v})")
    return out


def describe(t: Target) -> str:
    """'demo key, host api-demo.bybit.com'."""
    return f"{t.mode} key, host {t.host}"


def assert_ok(environ: dict) -> Target | None:
    """The target the client may use, or ExecTargetRefused naming every problem. None when off."""
    t = read(environ)
    found = problems(t)
    if found:
        raise ExecTargetRefused("order client refused: " + "; ".join(found))
    return t
