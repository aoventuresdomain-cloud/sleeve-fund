from decimal import Decimal



def test_a_round_trip_at_xrp_sizes_closes_despite_float_residue():
    """Six-decimal fills in the thousands sum to 1.4e-12 in floats, which used to keep the trip open and
    merge it with the next one (review round 10, M10-4). The position is now summed in Decimal."""
    from sleeve_fund.dashboard.trading import open_lot
    from sleeve_fund.research.metrics import trades

    buys = [3356.872962, 4406.203882, 2854.315447, 3835.633353]
    sells = [925.448019, 7617.729411, 1160.749841, 1254.793923, 3494.30445]
    rows = [{"side": "BUY", "qty": q, "price": 0.5, "fee": 0.0, "order_id": "b"} for q in buys]
    rows += [{"side": "SELL", "qty": q, "price": 0.51, "fee": 0.0, "order_id": "s"} for q in sells]
    rows += [{"side": "BUY", "qty": 100.0, "price": 0.5, "fee": 0.0, "order_id": "b2"},
             {"side": "SELL", "qty": 100.0, "price": 0.49, "fee": 0.0, "order_id": "s2"}]
    trips = trades(rows)
    assert len(trips) == 2 and trips[1]["entry_order"] == "b2"
    assert open_lot(list(reversed(rows[:-1])))["order_id"] == "b2"  # newest first; the second trip is open


def test_xrp_sized_trips_close_with_quantities_as_the_engine_reports_them():
    """Review round 11, M10-4 reopened: the engine's Quantity.as_double() is not always the nearest float to
    the lot's decimal (1015.315545 reads 1015.3155449999999), so the noise survived the Decimal sum and
    trips merged. Fills are journaled from the quantity's own decimal; trips built that way all close."""
    import random

    from nautilus_trader.model import Quantity

    from sleeve_fund.research.metrics import trades

    rng = random.Random(4)
    off = 0
    rows = []
    for n in range(40):
        lots = [Quantity(rng.randint(1, 9_000_000_000) / 1e6, 6) for _ in range(rng.randint(1, 4))]
        off += any(Decimal(repr(q.as_double())) != q.as_decimal() for q in lots)
        total = sum(q.as_decimal() for q in lots)
        rows += [{"side": "BUY", "qty": float(q.as_decimal()), "price": 0.5, "fee": 0.0, "order_id": f"b{n}"}
                 for q in lots]
        rows.append({"side": "SELL", "qty": float(total), "price": 0.51, "fee": 0.0, "order_id": f"s{n}"})
    assert off  # as_double() would have carried noise into some of these
    assert len(trades(rows)) == 40


def test_a_backtest_journals_each_fill_at_its_lot_decimal(prices):
    """At XRP prices every journaled quantity prints with no more decimals than the instrument's lot
    (as_double() journaled 5557.105068 as 5557.105068000001 here)."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("XRP", "USD")
    p = prices.copy()
    scale = 0.32 / float(p["close"].iloc[0])
    for c in ("open", "high", "low", "close"):
        p[c] = (p[c] * scale).round(5)
    p["volume"] = p["volume"] / scale
    res = run_backtest("trend_filter", p, inst, {}, risk_profile="balanced")
    assert res.journal.fills_
    for f in res.journal.fills_:
        assert -Decimal(repr(f["qty"])).as_tuple().exponent <= inst.size_precision, f
