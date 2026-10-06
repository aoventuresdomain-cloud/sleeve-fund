import re

import pytest

from sleeve_fund.wording import no_venues

VENUE = re.compile(r"bybit|binance|kraken|deribit", re.IGNORECASE)


@pytest.mark.parametrize(("text", "reads"), [
    # Any case (P1-U23): venues name themselves as they like.
    ("binance rejected the order", "the perpetual venue rejected the order"),
    ("ByBit Demo said no", "the demo account said no"),
    ("BYBIT DEMO said no", "the demo account said no"),
    ("kraken is down", "the spot venue is down"),
    ("deribit timeout", "the demo account timeout"),
    # Hosts and URLs are the venue, never a mangled host.
    ("GET api-demo.bybit.com failed", "GET the demo account failed"),
    ("fapi.binance.com: 503", "the perpetual venue: 503"),
    ("https://api.Binance.com/fapi/v1/order?x=1, retrying", "the perpetual venue, retrying"),
    ("api.kraken.com and test.deribit.com", "the spot venue and the demo account"),
    # Code identifiers stay readable.
    ("BinanceClientError: -2021", "VenueClientError: -2021"),
    ("KrakenSpot adapter", "VenueSpot adapter"),
    ("BybitDemo copy", "the demo account copy"),
    # Instrument ids keep their symbol; a bare suffix still names nothing.
    ("BTCUSDT-PERP.BINANCE: the feed missed 2 minutes", "BTCUSDT-PERP: the feed missed 2 minutes"),
    ("InstrumentId('BTC/USD.kraken')", "InstrumentId('BTC/USD')"),
    (".BINANCE answered", "the perpetual venue answered"),
])
def test_no_venue_name_in_any_case_host_or_identifier(text, reads):
    assert no_venues(text) == reads
    assert not VENUE.search(reads)


@pytest.mark.parametrize("text", [
    "kraken-live: KRAKEN_API_KEY missing",  # the PM's own account and key names
    "BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET",
    "1,234.56 62,850.00 0.0004 -2021 1.5e-3 BTCUSDT-PERP BTC/USD XBTUSD BTC-PERPETUAL",
])
def test_labels_keys_numbers_and_symbols_are_left_alone(text):
    assert no_venues(text) == text
