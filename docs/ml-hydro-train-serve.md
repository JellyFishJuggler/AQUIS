# `rain_api` train/serve mismatch — definition, evidence, and what must change

**Status:** the hydrology block (`HYDRO_COLS`) is **carried but untrained**, and this
document explains one concrete reason it cannot be promoted fairly yet. Nothing
described here has been changed in the pipeline; this is an analysis of the code
as it stands.

Scope: `rain_api7/22/60` and `rain_obs7/22/60` as built by `_hydrology.rain_memory_block`.

---

## 1. Where rainfall comes from

| Layer | File | What it does to rainfall |
|---|---|---|
| Normalise | `03_merge_normalize.py` | numeric coercion; drops NaN rows. **No fill.** |
| Daily align | `04_align.py:111-123` | IDW (1/d²) over up to 3 nearest gauges within 150 km, merged onto wells with `how="left"` (`:212`) |
| 6 h align | `04b_align_6h.py:116-127` | 6 h **sum** per gauge, then IDW to the well; `how="left"` (`:125`) → an unobserved slot stays **NaN** |
| Feature build | `06_features.py:166-180` | mask = `np.isfinite(rain)` |

**Consequence:** rainfall is never forward-filled or interpolated anywhere in the
pipeline. In a batch frame a finite value *is* a real observation, and a missing
slot is NaN. That is why the batch `rain_obs*` mask can be `isfinite` — and why
the code builds it explicitly rather than assuming it, so the meaning survives any
imputation step added upstream later.

---

## 2. Definition used for **training** (batch)

`_hydrology.rain_memory_block(rain_values, mask)` called once per station over the
station's **observed history only** — `06_features.py:166-180`:

```
rain_api{τ}[t] = λ·rain_api{τ}[t-1] + rain[t],   λ = exp(-(6/24)/τ)
rain_obs{τ}[t] = (# of finite rain slots in [t-τ·4+1 .. t]) / (τ·4)
```

* **`rain_api{τ}`** — exponential memory of rainfall **known at or before the row's
  own timestamp**. For a training row anchored at `t`, that is rainfall observed
  up to `t`. No future value can enter: the row's `rain` series simply has no
  entries after `t`.
* **`rain_obs{τ}`** — observed-rain fraction of the trailing τ·4-slot window
  (28 / 88 / 240 slots for τ = 7 / 22 / 60 d).

Causality is therefore satisfied in the batch path.

---

## 3. Definition used by **serving** (future mode)

`refresh.future_drivers._rain_memory_steps:329-370`, reached only when
`feature_mode == "future"` (currently **not** active — see §5):

```
series = concat(observed_history_rain, forecast_rain[0..steps-1])
mask   = concat(isfinite(observed_history_rain), zeros(steps))   # future = UNOBSERVED
blk    = rain_memory_block(series, mask)
step k -> blk[...][n_hist + k]
```

* **`rain_api{τ}`** — the recursion is **advanced across the forecast window**. For
  forecast step *k*, the value depends on precipitation at
  `anchor + 6h, anchor + 12h, … anchor + (k+1)·6h`, i.e. rainfall **after the
  anchor**. The values come from
  `future_weather_matrix` (`future_drivers.py:437-459`): Open-Meteo forecast
  within 16 days, climatology beyond, NaN otherwise.
* **`rain_obs{τ}`** — correctly unaffected: future slots are masked unobserved, so
  coverage counts observed history only and **decays** across the horizon.

So the two families are correctly separated, but `rain_api{τ}` at serving time is
**not the same function of the anchor** as `rain_api{τ}` at training time.

---

## 4. The mismatch, precisely

For a row anchored at `t`, horizon step `k` (forecast time `t + (k+1)·6h`):

| | Training feature | Serving feature |
|---|---|---|
| Information used | rainfall ≤ `t` | rainfall ≤ `t` **plus forecast rainfall in `(t, t+(k+1)·6h]`** |
| Depends on `k`? | **No** — constant across horizons | **Yes** — advances per step |
| `rain_obs` | observed fraction ≤ `t` | identical semantics, decays per step |

**Two distinct problems, and they must not be conflated:**

1. **Schema inconsistency.** In training the column is horizon-invariant; in
   serving it varies with `k`. A model fitted on the training definition is being
   served a different function. This alone makes the feature unfair.
2. **Leakage risk if "fixed" naively.** The obvious repair — build training
   `rain_api` from what *would have been* forecast at `t` — requires
   point-in-time forecast rainfall. Using the Open-Meteo **archive** API for that
   is the trap the code already contains: `refresh/future_drivers.py:14` labels it
   "Open-Meteo (archive for backtest / forecast <=16 d for live)", and
   `ml/20_openmeteo_fetch.py:85-86` calls that archive for *past* dates. For a
   backtest that is defensible, because ERA5 at `t` is a fair mirror of an
   operational forecast. For **training** it is not: ERA5 is *realized* rainfall,
   so a training row would be fitted on rainfall that had not yet happened.
   **Using realized future rainfall as a training feature is disallowed.**

**Verdict:** the two definitions are not interchangeable, and the safe repair is
not "make training look like serving" — it is to pick a definition that is
legitimately available at both times.

---

## 5. Current exposure

`data/refresh/runtime.json` reports `"feature_mode": "flat"`, which means the
serving path replicates the anchor row across all 120 horizons
(`_trajectory.py:201-205`) and **never calls** `_rain_memory_steps`. The mismatch
is therefore **latent, not live**: the `future` path is gated behind a backtest
precisely so it cannot ship unvalidated.

It becomes live the moment `feature_mode` is flipped to `"future"`, or the moment
`HYDRO_COLS` is promoted into a trained `num_cols` and the serving feature vector
is assembled from `build_step_features`.

---

## 6. Does a suitable archive exist?

**Partly — and not sufficiently.**

* **Archived forecast JSONs exist.** `ml/data/refresh/forecast_archive/` holds
  ~7,187 files of the form `<station_key>-<anchor>.json`, written each publish
  cycle by `refresh/publish.py:145-158`.
* **But they do not contain driver values.** Each archived record stores only the
  forecast GWL quantiles per horizon, the confidence labels, and a
  `driver_source` **label** string (`"Open-Meteo forecast (<=16d)"`,
  `"climatology (beyond Open-Meteo forecast)"`). Example:
  `forecast_archive/29bd98b3d1bf483f-2026-09-06T120000.json:6-16`.
  **The numeric rainfall that produced the forecast is not persisted.**
* **No point-in-time driver store exists.** `ml/data/cfs/openmeteo_weather_daily.parquet`
  is a rolling overwrite (the fetch re-queries archive + forecast and rewrites the
  file), so past *forecast* rainfall is not recoverable from it.
* **Coverage would be thin even if it were.** The archive spans roughly Aug–Sep
  2026 in the sampled filenames, and rainfall observations exist for only ~8
  districts / ~50 gauges, so most archived forecasts have no ground truth to be
  scored against.

**Conclusion: there is no archive suitable for training `rain_api` in its
serving definition.**

---

## 7. What must change before these features can be trained and promoted fairly

Ordered; (A) is sufficient to train the block, (B) is needed to keep the
forecast-rain signal.

### Option A — pin the definition to anchor-only information *(recommended first)*

Make serving compute `rain_api{τ}` from **observed rainfall up to the anchor
only**, identical to batch.

* Change: in `_rain_memory_steps`, stop concatenating `fw["rain"]` into the API
  series (or keep the concatenation for a *separate, explicitly named* column, e.g.
  `rain_api_fwd{τ}`, and never let it carry the trained name).
* Cost: forfeits the forecast-rain signal, which was the point of "future" mode.
  The `flat` production path already accepts that loss, so this is not a
  regression against what ships today.
* Benefit: training and serving become the *same function of the anchor*;
  `rain_api` becomes horizon-invariant and the schema is consistent.

### Option B — persist a point-in-time driver archive *(enables the forecast signal)*

* Change: extend `refresh/publish.py` (and the `future_drivers` build step) to
  persist, per anchor, the **numeric per-step driver matrix actually used**
  (rain at minimum, ideally all `DRIVER_COLS`) alongside the existing forecast
  JSON.
* Then training can consume archived forecast rainfall from the same vintage the
  model saw — legitimately, with no realized-future contamination.
* Cost: a data-engineering task plus months of accumulation before there is enough
  history to train distance-to-anchor features. **Until it exists, state plainly
  that the forecast-rain form of `rain_api` is not trainable.**

### Whichever option is taken

1. **Implement the chosen definition in the shared helper**, not per-caller, so
   batch and serving cannot drift again (`_hydrology.rain_memory_block` is already
   the shared choke point).
2. **Add a parity test** asserting a serving row at the anchor equals the batch row
   at the same timestamp for every `rain_api*` / `rain_obs*` column. The existing
   `ml/tests/test_hydro_rain_obs.py` covers `rain_obs` semantics and the
   `rain_api`-uses-forcing rule; a train/serve parity assertion does not yet exist.
3. **Re-run the backtest** on the corrected definition before any promotion, on the
   honest anchor set, with the calibration fitted and reported on disjoint anchors.
4. **Only then** append `HYDRO_COLS` to `num_cols` and retrain — the refresh gate
   refuses a candidate whose `feature_cols` differ from production
   (`refresh/model_update.py:282`), so this is a deliberate, coordinated change and
   not something that can happen by accident.

---

## 8. What is *not* in question

* `rain_obs7/22/60` semantics are correct and consistent in both paths: only actual
  gauge observations count, over a fixed τ·4-slot denominator, and
  forecast/climatology values never count as observed.
* The other hydrology columns do not have an analogous train/serve problem:
  `gwl_dev*`, `gwl_slope*`, `gwl_obs*` and `gwl_imputed` are computed from
  history ≤ the anchor in **both** paths and are pinned at the anchor in serving.

---

## 9. Summary table

| Feature | Training definition | Serving definition | Fair to train? |
|---|---|---|---|
| `rain_api{τ}` | rainfall ≤ anchor, constant across h | rainfall ≤ anchor **+ forecast in-window**, varies with h | **No** — mismatch; no point-in-time forecast archive |
| `rain_obs{τ}` | observed fraction ≤ anchor | observed fraction ≤ anchor, decays with h | **Semantics correct**; still untrained |
| `gwl_dev{7,30}` | history ≤ anchor | history ≤ anchor, pinned | Yes |
| `gwl_slope{7,30,90}` | history ≤ anchor | history ≤ anchor, pinned | Yes |
| `gwl_obs{7,30,90}` | history ≤ anchor | history ≤ anchor, pinned | Yes |
| `gwl_imputed` | anchor row provenance | anchor row provenance | Yes |
