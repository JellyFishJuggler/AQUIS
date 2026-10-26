# Phase 3a revised — named-station gate PASSED; build pending

Status: **NAMED-STATION GATE PASSED (2026-10-26). The cleaned `data_v2` build, purge check, and contact-sheet review are not yet complete; this file will be updated when they are.**

This revision resolves the previous Block Office, Kurara gate failure. The failure was a detector logic error in the M2 repeated-exact path, not evidence that Kurara's data must be masked. The no-change requirement for Block Office, Kurara is unchanged at **zero M1/M2 rows**.

## Named-station gate (corrected detector)

| Requirement | M1/M2 rows | Result |
|---|---:|---|
| Must fix: Primary School, Kalan Bar Devran | 76 | PASS |
| Must fix: Primary School/ Composite, Mehmoodpur | 5 | PASS |
| Must not change: Mathura Chaumuha Navagaon-11 | 0 | PASS |
| Must not change: Block Office, Kurara | 0 | PASS |
| Must not change: Hamirpur Sarila Purani | 0 | PASS |
| Must not change: Primary School, Evani | 0 | PASS |
| Must not change: Primary School, Garher | 0 | PASS |
| Residual: Hamirpur Kurara Jalokhar | 0 | Flag-only, as amended |

Gate artifacts: `named_gate.csv`, `gate_summary.json`, `gate_only_run.log`, `station_detector_summary.csv`, `masked_rows_by_station.csv`, `masked_rows_by_month.csv`.

## Block Office, Kurara: evidence and decision

The previous run reported 22 M2 rows at this control station. Reproducing the detector showed that all 22 were **interior points of two flat, carried-forward plateaus**, not isolated excursions:

- 11 rows at `2025-07-02 06:00`–`2025-07-04 18:00`, all value `-10.293`;
- 11 rows at `2025-07-08 06:00`–`2025-07-10 18:00`, all value `-9.863`;
- every one of the 22 rows has `gwl_obs=0`, i.e. it is a 6-hour grid forward-fill, not a new observation;
- each plateau starts with one observed reading and is then carried for up to 12 slots (the documented `04b_align_6h` limit).

The old repeated-exact condition required only `v[i-1] == v[i+1]` plus a 6-hour cadence. Inside any constant run that condition is trivially true, so the detector flagged plateau interiors as if each were an isolated `A-B-A` spike. It never flagged the single observed start of either plateau because one neighbour is missing. This was therefore a **detector logic error**, not a Kurara quality event, and the correct gate outcome is zero masked rows. The `-10.293`/`-9.863` episodes remain unmasked; they also do not meet the M2 physical path (about 6 m departure, not >30 m, and longer than the 12-slot cap). They are left as-is and are not used to exclude the station.

### Rule change (physical/instrumentation rationale, not threshold tuning)

The M2 repeated-exact path now additionally requires:

1. `v[i] != v[i-1]` — the value must actually depart from the preceding slot; a flat run is not an excursion.
2. `gwl_obs=1` at `i-1`, `i`, and `i+1` — all three slots must be real readings. A forward-filled carry is not a new reading and cannot create or confirm a repeated extreme.

No distance, count, window, or robustness threshold was changed. The pre-cutoff definition, minimum of three pre-cutoff occurrences, 8-MAD envelope, exact-return requirement, and 12-slot physical cap are unchanged. The change was checked against every named station and against the full station population.

Impact of the correction across all 600 stations:

- M2 rows: 2,439 before → 635 after (1,804 removed).
- All 1,804 removed rows are constant-run interiors: 1,798 have `gwl_obs=0`; the other 6 are observed but still have `v[i] == v[i-1]`, so they are not isolated excursions.
- Must-fix stations remain fixed: Kalan Bar Devran 87 → 76; Mehmoodpur 5 → 5.
- All five no-change controls go to zero; Hamirpur Kurara Jalokhar remains zero.
- No rows were added by the correction.

Evidence files: `inspect_block_office.py`, `inspect_block_office.log`, `block_office_m2_plateau_evidence.csv`, `evaluate_block_office_fix.py`, `evaluate_block_office_fix.log`, `detector_old_vs_new_summary.csv`, `m2_removed_by_detector_fix.csv`, `named_station_m2_rows_corrected.csv`.

## Rules evaluated

- **GWL convention:** raw datum-referenced water level. Negative and positive values are both meaningful; no sign-only test is used, and no station is rejected for sign.
- **M1 placeholder prefix:** at least 28 identical six-hour slots, or at least four slots for a known sentinel; continue through the second plateau until two adjacent 28-slot windows each contain at least 20 non-sentinel observations, eight distinct values, and a 5th–95th percentile spread no greater than 10 m. Instrument/continuation rules, not tuned to model error.
- **M2 physical short excursion:** 1–12 slots, >30 m from both surrounding 7-day medians, the medians agree within 5 m, and the series returns to the prior level.
- **M2 repeated exact extreme:** at least three pre-cutoff occurrences outside median ±8 robust MAD (or sentinels), exact immediate return at 6-hour spacing; now also requires an actual departure and observed slots as described above.
- **M3 persistent level transition:** detected as a candidate and recorded in `m3_breaks.csv`; values are never edited. The feature build excludes a 30-day post-break lag warm-up and anchor/target pairs straddling a break.
- **Stuck, flip-flop, extreme burst:** flags only. They do not delete values or exclude stations.
- **Early-era review:** Fugana_1, Bhadarwara P_1, Ranipur_2, and Murwa_1 remain flagged where present. No absolute implausibility claim or M1 early-era mask is made without surveyed well elevation/logger datum metadata.

## Build and review status

The gate passed before any `data_v2` write. The previous `data_v2` contents were stale partial artifacts and were explicitly marked DO NOT USE. Their exact inventory is recorded and they are archived under `outputs/diagnostics_phase3a/stale_data_v2_20261026/` before a fresh build; nothing is consumed from them.

Remaining required steps:

1. Build a fresh `data_v2` from `data/aligned/table_6h.parquet` using the corrected masks.
2. Run the target-time purge/leakage check for main and trajectory splits.
3. Render and inspect the 29+12 contact sheets.
4. Mark the build complete only if all of the above pass.

No original `data/`, `models/`, or `data/refresh/` paths are written by this work.
