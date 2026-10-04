

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
