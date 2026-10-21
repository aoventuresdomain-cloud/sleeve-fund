# Research definitions: the merge rule (R-29)

A study run from the study workflow with `configs_from=main` (the default) reads `configs/definitions/*.toml`,
`configs/definitions/variants/*.toml` and `configs/studies/*.md` from main at the commit it was dispatched on, with no
deploy. A merged definition can therefore be studied the same day. Until R-25 counts tries by idea family in code,
the count is kept by hand, and every such sheet says "Tries counted by hand until R-25."

## What a definitions PR needs before the HoE merges it

1. **Pre-registration in the same commit as the TOML, or an earlier one.** A pre-registration entry from the
   Researcher, with its settings, hypothesis and pass rule fixed before any run, in the style of the trend-variants
   pre-registration. It also names the idea's **family**: the parent idea it counts against. The family goes in the
   pre-registration and the PR, not in the TOML; the definition loader refuses keys it doesn't know.
2. **The Advisor's OK, recorded in the PR.**
3. **The family's tries so far, written in the PR.** Every file in a family counts, new files too (a qr01e is a try
   of qr01's family, not a fresh idea), against the family's budget of 20.

No new family until R-25 lands, unless the pre-registration file listed it before the merge. The HoE refuses any
definitions PR without all three.

## What a config-only merge does and doesn't change

- The deploy skips the restart when main moved only in those paths (`deploy/research_config_only.sh`); the server's
  deployed-commit record stays where it was.
- The study container alone sees the new configs, mounted read-only. The dashboard keeps the image's own, so the
  **Strategy tab keeps showing the image's definitions** until the next code deploy (CR F2). A new definition is
  studied from the workflow, not created as a Strategy, until then.
- `configs_from=image` runs on the image's configs as before, with no hand-count line.
- A file under `configs/definitions` or `configs/studies` that is not named as a research config, or is not a plain
  file, fails the study run, by name.
