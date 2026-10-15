"""F-RX-1 (DA sign-off): an order the order-state check marks "unfilled_unknown" releases its portfolio-gate room as
a rejected one does ('reject'), so room held for an order that never reached the book is not held for ever."""

from test_p21b_wiring_qa import _approved, _res, store  # noqa: F401


def test_unfilled_unknown_releases_the_reservation_as_reject(store):
    rid = _approved(store, oid="o-9")
    assert rid is not None and _res(store, rid).released_at is None
    store.update_order("o-9", status="unfilled_unknown", message="not found at the venue; nothing filled")
    assert _res(store, rid).release_reason == "reject"
