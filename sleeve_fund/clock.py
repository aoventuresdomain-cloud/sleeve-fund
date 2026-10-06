"""The server clock against a time server (v2 P1-2): every decision, order and fill is stamped by this clock,
so close-to-fill times and the venue's own timestamps only compare if it keeps time. The server syncs it
(chrony); this checks that it does.

One SNTP query (RFC 4330) to CLOCK_SERVER (set in docker-compose.yml to time.aws.com, the time service AWS
servers sync to; unset, nothing is checked). Its error is at most half the round trip, a few milliseconds, far
below the alert threshold."""

from __future__ import annotations

import os
import socket
import struct
import time

MAX_OFFSET_MS = 250  # above this, a warning: venue and server timestamps no longer compare
NTP_EPOCH = 2_208_988_800  # seconds from 1900 (NTP) to 1970 (UNIX)


def server(environ=None) -> str | None:
    return ((os.environ if environ is None else environ).get("CLOCK_SERVER") or "").strip() or None


def _ntp_time(raw: bytes) -> float:
    secs, frac = struct.unpack("!II", raw)
    return secs - NTP_EPOCH + frac / 2**32


def offset_ms(host: str, port: int = 123, timeout: float = 2.0, now=time.time) -> float:
    """How far this clock is behind the time server, in milliseconds (negative: ahead). Raises OSError when
    the server can't be reached or answers with nonsense."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        t1 = now()
        s.sendto(b"\x1b" + 47 * b"\0", (host, port))  # version 3, client mode
        data, _ = s.recvfrom(512)
        t4 = now()
    if len(data) < 48 or data[1] == 0:  # stratum 0: a "kiss of death", not a time
        raise OSError(f"{host} sent no usable time")
    t2, t3 = _ntp_time(data[32:40]), _ntp_time(data[40:48])
    return ((t2 - t1) + (t3 - t4)) / 2 * 1000
