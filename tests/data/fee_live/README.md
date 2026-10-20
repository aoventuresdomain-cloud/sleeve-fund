Recorded venue responses for the FEE-LIVE cells (no live calls).

- risk-limit-BTCUSDT-2026-10-15.json: the paper venue's public risk-limit page for BTCUSDT as the PM pasted it on
  15 Oct 2026 (research/venue-data, hand-checked by the DA).
- fee-rate-BTCUSDT.json: the account fee-rate response in the venue's V5 shape, at the published VIP 0 perp rates
  (0.02% maker, 0.055% taker). Not yet captured from the demo account: replace it with the server's first real
  response once the supervisor has fetched one.
- instruments-info-BTCUSDT.json: the paper venue's public instruments-info for linear BTCUSDT in its V5 shape, tick 0.10
  (B-43, the tick the venue's tiers are priced in, 0.1 as kept since B-36b). Written to the documented shape from a build
  machine that can't reach the venue, not captured: replace it with the server's first real response.
