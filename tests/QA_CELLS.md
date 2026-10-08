# QA cells ported into the repo (QA-CELLS)

The Head of QA's list: `quant-review/v2-p2/qa-cells-list.md` (22:58 UK, 7 Oct). Each cell below was a strict xfail in a
QA-only script whose bug is fixed, so it is now a plain regression check. Assertions are unchanged from the source;
only names, imports and helper paths were adapted. Source paths are under
`/mnt/project-files/sleeve-fund/quant-review/` (nothing in tests/ imports from there).

## How the reds were recorded

- Each old SHA was checked out in its own detached worktree and only the ported test file was copied in.
- pytest ran through a runner that removes the venv's editable-install finder from `sys.meta_path`. Without that, a
  `sleeve_fund` submodule missing from the old checkout loads silently from the current checkout, and an old run can
  pass or fail against today's code. Every run confirmed `sleeve_fund` loaded from the old worktree.
- Squash-merged PR heads were fetched as `refs/pull/<n>/head` (#151, #155, #182, #193), so the pre-fix commits inside a
  PR are reachable: eddc1bc, 81e6d6f, 998631d, 3732498, 7bac37b.
- SQLite. Every red is an `AssertionError` (or the cell's own `AssertionError` subclass) at the cell's own assertion,
  never an ImportError, AttributeError or TypeError. One exception: QA-193-F3 (p6b) is red with the `ValueError` its
  source's strict mark names, raised where the cell sends the reset again.
- Line numbers refer to the ported file.

## Cells

| test id | source script | finding | red on | failing assertion |
|---|---|---|---|---|
| test_qa_gapliq_adversarial.py::test_adv7_…[isolated-10pct-guard-path-ping_pong-resume] | v2-p1/gate-stop-choke-gapliq-scripts/test_155_adv_v2_adapted_sgz.py | P1-U34 | 81e6d6f (#155 final head; fix 3a2e439) | :121 `raise RunningForATick(...)`: running on the liquidating tick, -0.0331 |
| test_qa_gapliq_adversarial.py::test_adv7_…[isolated-10pct-guard-path-always-short-resume] | same | P1-U34 | 81e6d6f | :121 (same): -0.0333 |
| test_qa_gapliq_adversarial.py::test_adv8_…[isolated-10pct] | same | P1-D22 | 998631d (#155 follow-up head; fix 45e2a4f); also 81e6d6f | :159 `assert li, intents`: a daily-loss pause, no liquidation |
| test_qa_gapliq_adversarial.py::test_adv9_a_reset_requested_before_a_liquidation_does_not_clear_the_liquidation_halt | same | P1-D24 | 998631d; green on deb5f25 (the fix) and main | :260 `assert not new`: the halt was cleared and the strategy traded on the next day's process with no resume. Set-up re-pinned by the Head of QA (8 Oct 01:20 UK): the reset is asked while paused and holding, before the liquidation; the source asked it on the liquidating tick, where main now refuses it |
| test_qa_gapliq_adversarial.py::test_adv10_…[flatten-waiting] | same | P1-D23 | 3732498 (pr/182, parent of fix deb5f25); also 998631d | :212 `assert s.desired_state == "stopped"`: Stop refused, "a flatten is still waiting" |
| test_qa_liquidation_final.py::test_f2b_…[perp-2x-long-reconnect] | v2-p1/degraded-155-scripts/test_155_final_p155z.py | P1-D25 | 1582c3a (#155 as merged) | :79 equity left 6,032.78 vs journal 6,034.75 |
| test_qa_liquidation_final.py::test_f2b_…[perp-2x-long-restart] | same | P1-D25 | 1582c3a | :79 984.80 vs 989.30 |
| test_qa_liquidation_final.py::test_f2b_…[perp-3x-long-reconnect] | same | P1-D25 | 1582c3a | :79 3,983.61 vs 3,986.60 |
| test_qa_liquidation_final.py::test_f2b_…[perp-3x-long-restart] | same | P1-D25 | 1582c3a | :79 3,984.93 vs 3,987.92 |
| test_qa_liquidation_final.py::test_f2b_…[perp-3x-short-reconnect] | same | P1-D25 | 1582c3a | :79 3,983.26 vs 3,980.26 |
| test_qa_liquidation_final.py::test_f2b_…[perp-3x-short-restart] | same | P1-D25 | 1582c3a | :79 3,984.57 vs 3,981.58 |
| test_qa_liquidation_final.py::test_f7_…[restart-past-liq] | same | GAP-LIQ-CAP | 1582c3a; also b7368b5 (parent of fix #189) | :109 `assert loss <= X + 0.01`: 1,218.23 vs 1,004.50 |
| test_qa_liquidation_final.py::test_f7_…[live-gap] | same | GAP-LIQ-CAP | 1582c3a; also b7368b5 | :109 1,016.92 vs 1,004.50 |
| test_qa_funding.py::test_the_lag_is_the_publication_lag_not_an_outage | v2-p1/degraded-155-scripts/test_funding_155_qa.py | P1-O4 | eddc1bc (#151 head; parent of fix a5297ef) | :70 lag 12:02:15 > 5 min |
| test_qa_funding.py::test_as_of_never_returns_a_live_snapshot_first_seen_after_t | same | P1-O6 | eddc1bc | :90 first seen 13:30 > at 12:32 |
| test_qa_funding.py::test_a_nan_snapshot_is_not_served | same | P1-O8 | eddc1bc | :103 as_of returned `[nan, 1000000.0]` |
| test_qa_funding.py::test_the_alert_is_keyed_on_first_seen | same | P1-O7 | eddc1bc | :118 `at_risk(...) is None`: an alert was raised |
| test_qa_funding.py::test_a_week_old_series_is_raised_even_when_the_price_refresh_fails | same | P1-O5 | eddc1bc | :141 no alert raised |
| test_qa_funding.py::test_an_open_interest_conflict_does_not_break_the_price_series | same | P1-O3 | eddc1bc | :174 `errors == []`: KeyError 'minute' in append_bars and parity |
| test_qa_re_cost.py (10 cells) | re-cost-xfails/test_re_cost_xfails.py | RE-COST DW1-DW4, HOLD1-HOLD3 | see "Re-cost" below | |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[kill-a2] | v2-p1/kill-reset-188-scripts/test_kr188_probes.py | P1-KR-2 | a83de76 (round's PR base); also 018bb23 (parent of fix #188) | :241 the PM's Kill was lost |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[kill-b] | same | P1-KR-1 | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[kill-c] | same | m13-U5 (fixed by #188) | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[pause-a2] | same | P1-KR-2 | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[pause-b] | same | m13-U5 | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[pause-c] | same | m13-U5 | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[stop-b] | same | P1-KR-3 | a83de76 | :239 the PM's Stop was lost |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[stop-c] | same | P1-KR-3 | a83de76 | :239 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[flatten-a2] | same | P1-KR-2 | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[flatten-b] | same | P1-KR-1 | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_pm_intent_survives_the_reset[flatten-c] | same | m13-U5 | a83de76 | :241 |
| test_qa_kill_reset.py::test_matrix_flat_strategy_…[kill] | same | fixed by #188 | a83de76 | :269 |
| test_qa_kill_reset.py::test_matrix_flat_strategy_…[pause] | same | fixed by #188 | a83de76 | :269 |
| test_qa_kill_reset.py::test_matrix_flat_strategy_…[stop] | same | P1-KR-3 | a83de76 | :265 |
| test_qa_kill_reset.py::test_matrix_flat_strategy_…[flatten] | same | fixed by #188 | a83de76 | :267 |
| test_qa_kill_reset.py::test_kill_switch_route_acts_on_a_strategy_whose_reset_flatten_is_queued | same | P1-KR-1 | a83de76 | :287 the kill switch skipped it |
| test_qa_kill_reset.py::test_per_strategy_flatten_route_while_the_reset_flatten_is_queued | same | P1-KR-1 | a83de76 | :300 "a flatten is already waiting" |
| test_qa_kill_reset.py::test_two_resets_kill_switch_between_them_before_the_fresh_first_tick | same | P1-KR-2 | a83de76 | :321 kill switch lost across the second reset |
| test_qa_ral_probes.py::test_p1_ral_on_a_strategy_never_liquidated_is_refused_and_a_raw_row_changes_nothing[running] | v2-p1/ral-193-scripts/test_qa_ral_193_probes.py | QA-193-F1 (Postgres only) | 7bac37b on Postgres, recorded by the Head of QA; passes on SQLite | :82 `_tick(...)`: StringDataRightTruncation, varchar(32) on `events.kind` |
| test_qa_ral_probes.py::test_p1_ral_on_a_strategy_never_liquidated_is_refused_and_a_raw_row_changes_nothing[drawdown_halt] | v2-p1/ral-193-scripts/test_qa_ral_193_probes.py | QA-193-F1 (Postgres only) | 7bac37b on Postgres, recorded by the Head of QA; passes on SQLite | :82 `_tick(...)`: StringDataRightTruncation, varchar(32) on `events.kind` |
| test_qa_ral_probes.py::test_p1_ral_on_a_strategy_never_liquidated_is_refused_and_a_raw_row_changes_nothing[daily_pause] | v2-p1/ral-193-scripts/test_qa_ral_193_probes.py | QA-193-F1 (Postgres only) | 7bac37b on Postgres, recorded by the Head of QA; passes on SQLite | :82 `_tick(...)`: StringDataRightTruncation, varchar(32) on `events.kind` |
| test_qa_ral_probes.py::test_p7_…[stop_then_ral] | v2-p1/ral-193-scripts/test_qa_ral_193_probes.py | QA-193-F2 | 7bac37b (pr/193; parent of fix 746bcae) | :120 Start refused, RAL still pending |
| test_qa_ral_probes.py::test_p7_…[ral_then_stop] | same | QA-193-F2 | 7bac37b | :120 Start refused, Stop dropped the RAL |
| test_qa_ral_probes.py::test_p7c_stopped_and_liquidated_then_the_note_then_ral_then_start_trades_again | same | QA-193-F2 | 7bac37b | :149 Start refused, RAL never applied |
| test_qa_ral_f3f4.py::test_p5b_the_process_applies_a_ral_row_only_with_its_noted_incident | same | QA-193-F4 | 60d49e7 (main before this fix) | :54 the raw row was carried out: running, not halted |
| test_qa_ral_f3f4.py::test_p6b_a_failure_inside_the_ral_step_leaves_a_way_to_reset | same | QA-193-F3 | 60d49e7 (main before this fix) | :78 `ValueError` "already reset for this liquidation" (the source's strict mark raises ValueError): the incident was used up |
| test_qa_ral_f3f4.py::test_w1_…[before_halt_lifts] | v2-p2/pr217/test_217_ral_window_probes.py | QA F217-1 | d7875ef (#217 before this fix) | :192 the reset asked before the liquidation still pending |
| test_qa_ral_f3f4.py::test_w1_…[before_marked_applied] | same | QA F217-1 | d7875ef | :192 the reset asked before the liquidation still pending |
| test_qa_ral_f3f4.py::test_w0_a_normal_ral_lapses_the_earlier_reset | same | QA F217-1 (control) | none: passes on 60d49e7 and d7875ef | the Head of QA's control for W1, kept as a guard |
| test_qa_ir_parity.py::test_ir1_…[1] | v2-p1/integration-1709cd9-scripts/test_ir_parity.py | ir1 | 1918f8b | :34 fees 780.68 vs 771.05 |
| test_qa_ir_parity.py::test_ir1_…[15] | same | ir1 | 1918f8b | :34 fees 203.58 vs 201.07 |
| test_qa_ir_l6_restart.py::test_ir3_…[spot-long] | v2-p1/integration-1709cd9-scripts/test_ir_l6_restart.py | ir3 | 1918f8b | :33 restart fill not one seeded half spread from the backtest |
| test_qa_ir_l6_restart.py::test_ir3_…[perp-1x-long] | same | ir3 | 1918f8b | :33 |
| test_qa_ir_l6_restart.py::test_ir3_…[perp-3x-long] | same | ir3 | 1918f8b | :33 |
| test_qa_ir_l6_restart.py::test_ir3_…[perp-1x-short] | same | ir3 | 1918f8b | :33 |
| test_qa_ir_l6_restart.py::test_ir3_…[perp-3x-short] | same | ir3 | 1918f8b | :33 |
| test_qa_ir_glc_bars.py::test_ir4_…[3x-short-0.4] | v2-p1/integration-1709cd9-scripts/test_ir_glc_bars.py | ir4 / IR-1 (GAP-LIQ-CAP #189) | b7368b5 (parent of #189); passes on 1918f8b | :56 loss 6,018.42 vs X 5,018.15 |
| test_qa_ir_glc_bars.py::test_ir4_…[2x-short-0.6] | same | ir4 | b7368b5 | :56 9,020.05 vs 7,519.72 |
| test_qa_ir_glc_bars.py::test_ir4_…[3x-short-0.3317] | same | ir4 | b7368b5 | :56 4,993.39 vs 5,017.64 |

`tests/qa_ir_glc189_lib.py`, `tests/qa_ir_hub146_lib.py` and `tests/qa_ir_hub_path_parity_lib.py` are verbatim copies
of the QA harnesses the G scripts loaded through `sys.path` (the repo's `test_hub_146_qa.py` now seeds the restart
entry at the touch, which would change ir3's set-up). They have no `test_` prefix, so pytest does not collect them.

## Re-cost (list D)

Source: `re-cost-xfails/test_re_cost_xfails.py` (QA Tester 2's master), findings RE-COST DW1-DW4 and HOLD1-HOLD3.
Red on **769fe22**, main's parent of the RE-COST merge #181 (1d81c09), run **with the run cache on**: 10 of 10 red.

| test id | finding | failing assertion on 769fe22 |
|---|---|---|
| test_dw1_the_benchmark_costs_the_fee_the_run_pays_plus_half_the_spread[perp] | DW1 | :135 benchmark cost per side 0.0083 vs perp fee + half spread 0.0008 |
| test_dw2_the_runs_fee_rate_is_the_benchmarks_cost_with_no_spread | DW2 | :164 "and so must the benchmark": 0.008 |
| test_dw2_the_benchmarks_strategy_return_recomputes_from_the_perp_fee | DW2 | :176 strategy_return 0.1473 vs 0.1800 recomputed at the perp fee |
| test_dw3_the_fee_note_shows_the_fee_paid_and_the_break_even_fee | DW3 | :185 the note shows the spot fees (0.40% / 0.80%), not the fee paid |
| test_dw4_each_ladder_rung_carries_the_random_entry_return_at_its_own_cost | DW4 | :40 (`_need`) not built: `LadderRung.random_return` |
| test_dw4_the_rendered_ladder_shows_the_random_entry_return | DW4 | :218 not built: no ladder text |
| test_hold1_the_headline_hold_on_a_perp_study_pays_perp_funding | HOLD1 | :242 the headline hold is the spot hold (2.2516 vs perp-like 1.6231) |
| test_hold2_spot_is_a_secondary_line_labelled_holding_spot_instead | HOLD2 | :40 (`_need`) not built: `StudyResult.spot_hold_returns` |
| test_hold3_a_window_with_a_missing_settlement_is_left_out_never_zero_filled | HOLD3 | :296 "the uncovered days must leave the comparison" |
| test_hold3_under_half_the_days_covered_is_no_verdict_not_a_pass | HOLD3 | :40 (`_need`) not built: `StudyResult.hold_insufficient` |

Three cells go red at the master's own `_need` assertion (:40), because the field they read did not exist before #181.
That is the master's assertion, unchanged, but it is not the value check further down the cell.

**Run cache (Head of QA, 7 Oct 00:06 UK).** A study takes about 87 s, so cells whose study inputs are all equal share
one run inside a process. The key is every input; each cell gets a deep copy of the result, of what its fee spy
recorded and of the run's idea ledger. A run with `run_backtest` patched is never shared, and DW4's spy cell always
runs its own study.

| key | inputs | cells |
|---|---|---|
| K1 | perp spec, default prices (hash), instrument, spread 0.0003, windows 0/365/365, no extra args | DW1 (with its spy), DW2 strategy_return, DW3, DW4 rendered ladder, HOLD1, HOLD2 |
| K1, uncached | as K1 | DW4 each rung (keeps its own run) |
| K2 | as K1 but spread 0 | DW2 fee rate |
| K3 | native funding history kept from 1900 (full), prices hash, instrument, windows | both HOLD3 cells |
| K4, K5 | native funding history kept from each cell's own cut | one per HOLD3 cell |

Run time on main, one process: 10 passed in 635 s; the slowest cell is 256 s, under the 600 s per-test timeout.
CI's splitter may put the cells on different shards, where each runs its own study, so `.test_durations` records each
cell's uncached time (about 87 s; 256 s for each HOLD3 cell). That keeps the five shards balanced.

## The D13 masters (not cells)

`test_qa_d13_master.py` (master 74b6d107) and `test_qa_d13_parity_pins.py`: repo-ready copies that import nothing from
the project folder. Marks are as they stand (2 strict xfails remain open). Every assertion inside a def is equal to the
master as a multiset (137 = 137, 44 = 44); the only removed asserts are the two module-level folder guards that broke
collection in #178.

## Not ported

Never red at their own assertion on any pre-fix SHA; kept in QA masters as guards (Head of QA, 8 Oct), not obsolete:

- **ADV-7, 10 params** (every guard-path `flatten`/`start`, every full-margin param): no finding mark; pass on 81e6d6f,
  3a2e439, 998631d and 45e2a4f.
- **ADV-8 [full-margin]**: no finding mark; passes on 81e6d6f, 998631d and 45e2a4f.
- **ADV-10 [resume-waiting]**: TypeError on 81e6d6f (no `liquidating` argument yet); passes on 3a2e439, 998631d,
  45e2a4f, 3732498 and deb5f25.
- **test_matrix_pm_intent_survives_the_reset, 9 params** (kill-a1, kill-d, pause-a1, pause-d, stop-a1, stop-a2, stop-d,
  flatten-a1, flatten-d): pass on a83de76 and 018bb23; older heads fail only at import. Excluded through `NOT_RED`.
- **ir2** and **ir5**: pass on 1918f8b, 018bb23, a83de76, 1582c3a, 9ae1f8a and 00f4b5a.

Left out for another reason:

- **P1-O7, second half** (`test_a_series_never_kept_is_raised_across_restarts`): obsolete (Head of QA, 8 Oct). It
  asserts nothing beyond `test_open_interest.py::test_never_kept_counts_from_the_first_try_across_restarts`.
