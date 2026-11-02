# Why the fleet scan shows 111 `decline (high)` but the 2026 hindcast only 4

Read-only investigation on the 2026-11-01 data (before the fleet CSV was
regenerated). Inputs: `ml/outputs/fleet_forecast.csv` (scan of 2026-11-01
12:19), the rolling hindcast parquet (9 origins, 990 rows) and the aligned
6-hourly table. No model, data or fleet file was modified.

## Headline numbers

| set | rows | `change <= -0.6 m` | `change <= -0.3 m` | median change |
|---|---|---|---|---|
| fleet scan (pre-refresh) | 543 | 118 (111 in the `decline (high)` category) | 255 | -0.219 m |
| hindcast, evaluable rows | 937 | 4 | 137 | -0.139 m |

The hindcast denominator is the causal/evaluable set: DL and XGB values, an
actual within ±1 day of origin+30d, and anchor lag ≤7 days (391 distinct
stations).

## (a) Change distribution is split by anchor month, not by model

Fleet scan by anchor month (the anchor is the station's last pre-refresh
observation):

| anchor month | n | median 30d change | n ≤ -0.6 |
|---|---|---|---|
| 2025-04 and older | 30 | ~0.00 | 0 |
| 2026-08 | 14 | +0.403 | 0 |
| 2026-09 | 123 | +0.104 | 7 |
| 2026-10 | **370** | **-0.478** | **111** |

All 111 high declines are anchored in October 2026 (97 of them at exactly
2026-10-10 18:00). The hindcast per-origin high-decline counts are
`0, 0, 0, 0, 1, 0, 0, 0, 3` for 2026-01-10 → 2026-09-07 and its latest anchor
month is September: **no hindcast origin can score an October anchor**, because
`origin + 30d` must be ≤ the pre-refresh last observation (2026-10-10 18:00).
The next origin after 2026-09-07 would be 2026-10-07, whose target
(2026-11-06) is past the data.

So 111 vs 4 is first of all a window mismatch: the fleet cohort is one month
later than anything the hindcast can evaluate. It is also a seasonal mismatch:
the fleet's 30-day window (2026-10-10 → 2026-11-09) is the post-monsoon
recession, while the latest evaluable hindcast window is 2026-09-07 → 2026-10-07
(actual median +0.082 m) and the August origin was still in monsoon recharge
(actual median +0.991 m). The hindcast simply never observes the recession
month.

## (b) Anchor age and time: the scan ran on pre-refresh data

- Pre-refresh backup: `common.parquet` and `table_6h.parquet` both end at
  **2026-10-10 18:00**, with 382 stations reporting that slot.
- The fleet CSV/snapshot mtime is **2026-11-01 12:19**, before the refresh
  finished (~20:00), so `age_days = 21` for every row in the October batch.
- Fleet anchor clock hours: 429 at 18:00, 77 at 00:00, 27 at 12:00, 10 at 06:00.
  The high-decline cohort is a single late-evening telemetry batch
  (age 21 days, 111/111).
- The hindcast anchors at the fixed origin (median anchor lag 0 d, 931/988 at
  00:00) and picks stations that also have an actual at origin+30d.
- Fleet also carries 165 stations whose anchor is older than 45 days (99 at
  48 days, 24 at 556 days, 1 at 778 days). The scan's plausibility guard labels
  exactly those rows `unreliable`; the benchmark excludes anchor lag >7 days
  instead, so the two universes differ by more than the 111/4 question.

## (c) Same prediction path — the difference is the anchor date

Both the fleet scan and the hindcast take the +30d q50 endpoint of the same
deployed DL trajectory (`_traj_dispatch` → `_dl.trajectory_forecast`); they do
not use different models. Re-running the DL on high-decline stations from the
post-refresh anchors reproduces declines of the same order, so the October
signal is not an artifact of the scan code:

| station | fleet 30d | current anchor | rerun 30d |
|---|---|---|---|
| Navrasia | -0.741 | 2026-10-27 12:00 | -0.559 |
| Rajapur Sikraur | -0.676 | 2026-10-27 12:00 | -0.535 |
| Singhara Pr. Vid. | -0.644 | 2026-10-27 12:00 | -0.612 |
| Chorwa Bhari Prathmik Vidyalaya | -0.719 | 2026-10-27 12:00 | -0.672 |
| Niwabhari Prathmik Vidyalaya | -0.785 | 2026-10-24 12:00 | -0.768 |
| Lagdi Gullar Prathmik Vidyalaya | -0.835 | 2026-10-27 12:00 | -0.808 |

Correlation between the fleet change and the hindcast's latest-origin DL change
for the same station is only 0.044 (n=411) — expected when one side is anchored
2026-09-07 and the other 2026-10-10, with 30 days of weather in between. Of the
111 high declines, 85 are in the hindcast sample and their median hindcast DL
change at the Jan–Sep origins was ≈ +0.06 m.

## (d) District pattern is a coherent eastern-UP + Bundelkhand cluster

23 of 37 districts have at least one high decline (20.4% of the fleet):

| district | n | high | share |
|---|---|---|---|
| GORAKHPUR | 39 | 25 | 64% |
| SONBHADRA | 25 | 9 | 36% |
| DEORIA | 18 | 8 | 44% |
| SHAHJAHANPUR | 15 | 8 | 53% |
| LALITPUR | 22 | 7 | 32% |
| HAMIRPUR | 15 | 6 | 40% |
| KUSHINAGAR | 15 | 6 | 40% |
| SIDDHARTHNAGAR | 8 | 6 | 75% |
| JHANSI | 29 | 5 | 17% |
| MIRZAPUR | 22 | 5 | 23% |
| MAHOBA | 11 | 5 | 45% |

Western UP is essentially clear: MUZAFFARNAGAR 0/29, BAGHPAT 0/28,
SAHARANPUR 0/27, MEERUT 2/24, GHAZIABAD 0/6. The flagged cluster is eastern UP
(Gorakhpur, Deoria, Kushinagar, Maharajganj, Siddharthnagar, Shrawasti) plus
Bundelkhand (Sonbhadra, Lalitpur, Hamirpur, Jhansi, Mahoba, Banda, Jalaun),
which is consistent with a post-monsoon recession rather than a random data
failure.

## Partial validation against the refreshed data (to 2026-10-29)

The refresh adds observations after the scan anchor, which allows a partial
(18-day, not 30-day) check of the cohort:

- high-decline stations: median observed change **-0.362 m** over 18.25 days,
  **74%** actually fell;
- all other categories: median **0.000 m**, 36% fell.

So the October cohort did decline directionally, but the 30-day target
(2026-11-09) is still beyond the data and the magnitude cannot be verified yet.
An 8-station spot check of the DL 18-day point (h=72) predicted -0.38…-0.77 m
against observed -0.02…+1.16 m; 5 of 8 observed rises were still predicted as
declines, so station-level magnitude looks pessimistic.

## Takeaway

1. The 111-vs-4 gap is mostly a **time-window artifact**: the hindcast cannot
   evaluate the October anchor batch, and it samples 391 stations vs the
   fleet's 543.
2. The fleet scan is **stale** (pre-refresh anchors, one Oct-10 18:00 batch).
   Regenerating it on the refreshed data moves anchors to Oct 24–27; the
   declines largely persist (5/5 rerun spot checks stay ≤ -0.6).
3. Treat the 111 as **unverified 30-day declines**: the 18-day partial check
   supports the direction in aggregate, not the station-level magnitude.
   Re-evaluate once observations reach 2026-11-09.
