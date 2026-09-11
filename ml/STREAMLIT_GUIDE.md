# AQUIS dashboard — Streamlit component guide

Every Streamlit widget in the AQUIS groundwater dashboard, explained at the
same depth as the Correlation page below. The app is a multi-page Streamlit
app under `ml/`; every page is a plain script that runs top-to-bottom on each
rerun.

**Live navigation (4 pages):** Assistant (default) · Correlation · Forecast ·
Sources — declared in `app.py` via `st.navigation`. The older explorer pages
(Overview, Drivers, Stations, Model, Fleet) still exist under `app_pages/` and
are documented below but are **not wired into the nav**; the **Verification**
page (`app_pages/verification.py`) reads the refresh pipeline's realisations.

**Run it:** `cd ml && streamlit run app.py` (opinionated dark theme is in
`ml/.streamlit/config.toml`).

**Convention used throughout:** `GWL` (groundwater level) is a water-table
level in *metres* — a rising level means the value goes **up**. "Rising" name =
value up, "declining" = value down.

---

## Table of contents

1. Application entry (`app.py`)
2. Theme (`.streamlit/config.toml`)
3. Page-by-page, widget-by-widget
   - 3.1 Overview (deactivated)
   - 3.2 Correlation (active — the "depth template")
   - 3.3 Drivers (deactivated)
   - 3.4 Stations / Station explorer (deactivated)
   - 3.5 Model (deactivated)
   - 3.6 Forecast (active — trajectory v2)
   - 3.7 Fleet (deactivated)
   - 3.8 Assistant (active — default page)
   - 3.9 Sources (active)
   - 3.10 Verification (deactivated)
4. Cross-page component reference
5. Under the hood (data loaders, labels, outputs)

---

## 1. Application entry — `ml/app.py`

```python
st.set_page_config(page_title="AQUIS — correlation explorer",
                   page_icon=":material/analytics:",
                   layout="wide")
```

- **`st.set_page_config`** — global page options. `layout="wide"` removes the
  narrow centered column and uses the full browser width. `page_icon` uses a
  **Material Symbols** name (`:material/analytics:`) — Streamlit renders these
  without shipping an image asset. This must be the *first* Streamlit command
  in a script.

```python
page = st.navigation([...])   # list of st.Page definitions
page.run()
```

- **`st.navigation`** — declares the app's pages in one place (the modern
  replacement for `st.Page`-based multipage apps). Each item is an
  **`st.Page("app_pages/<file>.py", title="…", icon=":material/…:")`**. Order in
  this list = order in the sidebar. `page.run()` renders the currently selected
  page.

| # | Page file                | Title        | Icon                          | In nav |
|---|--------------------------|--------------|-------------------------------|--------|
| 1 | `app_pages/assistant.py`         | Assistant   | `:material/smart_toy:`        | ✅ (default) |
| 2 | `app_pages/correlation.py`       | Correlation | `:material/query_stats:`      | ✅ |
| 3 | `app_pages/forecast.py`          | Forecast    | `:material/troubleshoot:`     | ✅ |
| 4 | `app_pages/data_sources.py`      | Sources     | `:material/database:`         | ✅ |
| — | `app_pages/overview.py`          | Overview    | `:material/home:`             | off |
| — | `app_pages/drivers.py`           | Drivers     | `:material/science:`          | off |
| — | `app_pages/station_explorer.py`  | Stations    | `:material/monitoring:`       | off |
| — | `app_pages/model.py`             | Model       | `:material/model_training:`   | off |
| — | `app_pages/fleet.py`             | Fleet       | `:material/query_stats:`      | off |
| — | `app_pages/verification.py`      | Verification| `:material/verified:`         | off |

---

## 2. Theme — `ml/.streamlit/config.toml`

```toml
[theme]
primaryColor = "#4ecca3"             # teal — buttons, focus, chart accents
backgroundColor = "#0a0a0a"          # nearly-black page background
secondaryBackgroundColor = "#1a1a2e" # cards / alternate panels
textColor = "#e0e0e0"
font = "monospace"

[server]
headless = true

[browser]
gatherUsageStats = false
```

What the theme gives you: every widget picks up `primaryColor` for its active
state, `secondaryBackgroundColor` for bordered containers and inputs, and the
monospace font everywhere. The charts in `app_charts.py` and `snapshot.py`
reuse these exact hex values so inline charts, Altair charts and the exported
PNG all look the same.

---

## 3. Page-by-page, widget-by-widget

Each widget is documented as: **what it is**, **its options**, **what changes
when you use it**, and **which backend data/function it reads**. Repeated
helper patterns (e.g. `st.caption` explaining a number) are described once per
page to avoid noise.

### 3.1 Overview — `app_pages/overview.py`

Top-level sanity screen. Reads `ml/data/meta/manifest.json` (per-source row
counts) and `ml/data/meta/selected_gwl_stations.csv`.

| Element | Code | What it is / does |
|---|---|---|
| `st.title` | `st.title("AQUIS — groundwater correlation explorer")` | Page heading, largest text element. |
| `st.caption` | caption under title | Small muted note: correlation-first validation, sandbox, not production. |
| `st.container(horizontal=True)` | `with st.container(horizontal=True):` | **Layout** element — stacks the metrics that come below in a **single horizontal row** (instead of one per line). |
| `st.metric` ×4 | `st.metric("Selected GWL stations", f"{...nunique()}", f"{...} districts", border=True)` | **KPI card**: label + value + optional delta line + `border=True` draws a card outline. Sources: `manifest.json` rows for `gwl` source; driver rows = total rows − GWL rows; station count from `load_selected_gwl()`. |
| `st.header` | "Driver signal (median correlation vs groundwater)" | Section divider heading (larger than `subheader`, smaller than `title`). |
| `st.metric` ×2 | Strongest / Weakest driver | Biggest and smallest **\|raw Spearman\|** correlation from `outputs/correlation_report.csv` (`mode=="raw"`, `metric=="spearman"`), delta shows the r value and # stations. |
| `st.bar_chart` | `st.bar_chart(df, x="label", y="corr", color="#4ecca3")` | **Native Chart** — server-rendered bar chart. `x/y` are column names; `color` overrides the bar color. No Altair needed for simple ranking views. |
| `st.bar_chart` | `df` of source→station coverage | Same component, different frame: per-driver station coverage from manifest. |
| `st.caption` | footer note | Where the data comes from (local `ml/outputs/` + `ml/data/`). |

### 3.2 Correlation — `app_pages/correlation.py`

*This is the depth template used for every other page.*

Goal: which environmental drivers move groundwater, measured per-station and
aggregated (median) across stations.

| Element | Code | What it is / does |
|---|---|---|
| `st.title` / `st.caption` | "Correlation report" | Heading + a one-line caption summarising the three modes. |
| `st.segmented_control` | `st.segmented_control("Correlation mode", ["raw", "deseason", "diff"], default="raw")` | **Segmented button row** — a single-select control (modern toggle instead of dropdown). Choosing a mode re-filters the whole page below. |
| `st.stop` | `if not mode: st.stop()` | **Guard**: if the user *clears* the segmented control (possible — clicking an option again deselects it) `mode` is `None`; `st.stop()` halts the script so the rest of the page doesn't render with a broken filter. |

**What each mode actually means (backend in `ml/05_correlate.py`):**

- **`raw`** — full per-station history correlation of `driver` vs `gwl`
  (Spearman or Pearson). Requires ≥ 20 jointly-observed days per station; else
  that station's correlation is `NaN`. Stores the **median across stations** and
  how many stations were valid (`stations_ok`). Mixed signal: contains both the
  station's absolute base level and its seasonal cycle.
- **`deseason`** — each series is turned into an **anomaly**: subtract the
  per-`(station, day-of-year)` mean from every day (`deseason()` in
  `05_correlate.py`). Only computed for *level* columns. Isolates the
  **non-seasonal** co-movement, so a rain spike → GWL dip is visible, while
  both series' long seasonal envelopes cancel out. Interpretation: "when the
  *unusual* part of rainfall moves, does the *unusual* part of the level move?"
- **`diff`** — **first-difference** of both series (day-to-day change) before
  Spearman. Only for level columns and `rain_*` windows. **Datum-free**: it can
  never be dominated by one station sitting 5 m deeper than another, because
  absolute level is removed. Interpretation: "does today's increment in the
  driver predict the increment in the level on the same/aligned day?"

| Element | Code | What it is / does |
|---|---|---|
| `st.segmented_control` | `st.segmented_control("Metric", ["spearman", "pearson"], default="spearman")` | Second axis of the filter: **rank**-based (`spearman`, robust to outliers / non-linear monotonic) vs **linear** (`pearson`, sensitive to scale of movement). Note the report only computes *both* for `raw`; the other modes are Spearman-only. |
| `st.stop` | `if not metric: st.stop()` | Same deselection guard. |
| `st.columns` | `col1, col2 = st.columns([2, 3])` | Splits the row **2:3** — narrower table, wider chart. Children are drawn with `with col1: …`. |
| `st.container(border=True)` | `with col1: with st.container(border=True):` | **Bordered box** — visually groups the mini-table as a card on the dark background. |
| `st.markdown` | `st.markdown(f"**{metric} · {mode}**")` | Lightweight text (supports `**bold**` and inline Markdown). |
| `st.dataframe` | `st.dataframe(sub[[...]].rename(...), hide_index=True)` | **Interactive table**. `hide_index=True` drops the row-number column (the driver name becomes the key of each row). Sorting by correlation descending. |
| `st.bar_chart` | `st.bar_chart(chart, x="label", y="corr", color="#4ecca3")` | Bar ranking of driver correlations; labels come from `_utils.nice()` (human driver names like "Rain, 7 days"). |
| `st.header` | "Recharge lag — GWL vs rainfall" | Second section. |
| `st.line_chart` | `st.line_chart(lag, x="lag", y="pearson_med")` | Line over `outputs/lag_curves.csv`. Each point = median (across stations) Pearson of `rain (shifted by lag days)` vs `gwl`; **the peak marks when rainfall historically shows up in the water level** (the recharge lag). |
| `st.success` | `st.success(f"Best rainfall recharge lag: **{…} days** …")` | **Green status callout**. Here it summarises the *argmax \|r\|* row (`lag.iloc[lag["pearson_med"].abs().idxmax()]`) into a single headline number. |
| `st.expander` | `with st.expander("Full report table", icon=":material/table_chart:"):` | **Collapsed section** — content hidden until toggled (keeps the page short). `icon=` adds a Material icon to the toggle row. |
| `st.dataframe` | inside expander, `height=420` | The *entire* report (driver × metric × mode), fixed 420 px height → its rows scroll inside the table. |
| `st.dataframe` / `st.altair_chart` | "Static features vs station groundwater" section | Soil-texture block: merges per-station GWL summary (`mean/std/median`) with ISRIC soil features, then Spearman of each soil fraction vs mean GWL and vs GWL spread. Left: `st.dataframe` of the ρ values; right: `st.altair_chart` bar of ρ vs mean GWL (teal). `st.caption` caveat: **exploratory only — soil is NOT fed to the model**. |
| same for LULC | "LULC class share vs district GWL" | Reads `outputs/correlation_lulc.csv` (28 districts). Left table = Pearson r per class; right chart = top-8 classes (rose `#e67676`). Same "not fed to the model" caption. |

### 3.3 Drivers — `app_pages/drivers.py`

Goal: overlay raw drivers against a chosen station's GWL, both normalised, one
chart per driver.

| Element | Code | What it is / does |
|---|---|---|
| `st.title` / `st.caption` | "Drivers vs groundwater level" | Heading + instructions ("Pick a station to see the aligned daily drivers…"). |
| `st.selectbox` | `st.selectbox("Select GWL station", stations)` | **Dropdown**. Options are station names sorted by **data recency** (most recent first), from `load_table()` grouped by `date.max()`. Changing it re-runs the page for that station. |
| `st.stop` | `if not stn: st.stop()` | If the selectbox is somehow empty, stop before indexing. |
| `st.metric` | `st.metric("Station", stn, f"{district}")` | KPI card showing the chosen station + its district as the delta line. |
| `st.caption` | soil summary line | One-line ISRIC soil texture summary for the station (`load_soil()`), e.g. "sand 62% / silt 24% / clay 14%". |
| `st.multiselect` | `st.multiselect("Drivers to overlay", driver_cols, default=[rain_7d, rain_30d, temp, river_level, pressure])` | **Multi-select dropdown**. `default=` pre-toggles the 5 most informative drivers if they exist for this station. Options exclude metadata (`Station/date/District/Tehsil`) and helper columns (`*_n`, `*_dist_km`). |
| `st.altair_chart` ×N | one **z-score overlay line chart per selected driver** | For each driver: take `[date, driver, gwl]` rows with both present, **z-normalise both series** (`(x−mean)/std`), melt to long form (series = "GWL (z)" / "<Driver> (z)"), draw an interactive line chart (grey `#e0e0e0` GWL, teal `#4ecca3` driver) with tooltip. Chart height 220, `interactive()` = pan/zoom. **What changes:** each tick of the multiselect adds/removes exactly one overlay chart. |
| `st.caption` | "No overlapping data for `{d}`…" | Per-driver empty-state note (drops if the pair has no co-observed days). |
| `st.caption` | `else:` "Choose at least one driver to compare." | Appears when the multiselect is emptied. |
| `st.expander` + `st.dataframe` | "Raw today layout" | Last 200 rows of `[date, gwl] + chosen drivers`, `hide_index=True`, `height=360` — the *unscaled* numbers behind the charts. |

### 3.4 Stations — `app_pages/station_explorer.py`

Goal: per-station browsing — driver coverage, a GWL+driver timeseries, median
trend and static soil.

| Element | Code | What it is / does |
|---|---|---|
| `st.title` / `st.caption` | "Station explorer" | Heading + subtitle "Browse per-station groundwater trends and driver coverage." |
| `st.sidebar` | `with st.sidebar:` | **Layout**: everything inside renders in the sidebar panel instead of the main column. |
| `st.selectbox` (sidebar) | "District" | Options `["All"] + sorted(unique districts)`; filters the whole page below. |
| `st.selectbox` (sidebar) | "Driver to inspect" | Which driver to measure coverage for; default `rain_7d` if present. Options reuse `driver_cols` logic from Drivers. |
| `st.columns(3)` + `st.metric` ×3 | "Stations (district)", "Rows", "Stations with {nice(driver)}" | Three bordered KPIs describing the filtered selection (`sub`) and the driver's non-NaN subset (`rec`). |
| `st.warning` + `st.stop` | `if day_counts.empty: st.warning(...); st.stop()` | **Yellow status callout** then halt. Triggers when the filtered selection has no station with the chosen driver. |
| `st.subheader` | "Top stations by {driver} day coverage" | Small section heading (smallest of the 3 heading levels). |
| `st.bar_chart` | `st.bar_chart(day_counts.head(15))` | Top-15 stations by # days the driver is present, descending. |
| `st.subheader` | "Timeseries — {top station}" | Section for the best-covered station. |
| `st.altair_chart` | melt of `[gwl, driver]` over date, `interactive()` | Two-line timeseries (grey GWL, teal driver), `width="stretch"` so it spans the column. |
| `st.caption` + `st.stop` | "No driver data for top station" | Guard before melting if the top station has none. |
| `st.subheader` + `st.line_chart` | "GWL trend (median surface)" | Daily **median** GWL across all stations in the current district filter, drawn as a line. |
| `st.subheader` | "Soil texture — {top station} (static, ISRIC)" | Static-soil block. |
| `st.container(horizontal=True)` + `st.metric` ×4 | Sand / Silt / Clay / Bulk density 0–30 cm | Four unit-carrying KPIs in one horizontal row (e.g. `f"{r['sand_0_30']:.0f}%"`). |
| `st.altair_chart` | faceted texture bars | Sand/silt/clay % at 0–30 and 60–100 cm, `column=` faceting creates two small sub-charts side by side, `legend=None`. `width=80` per facet. |
| `st.caption` | SoilGrids caveat | ISRIC SoilGrids v2, 250 m, point query; **gated out of the model** (redundant with station ordinals). |
| `st.caption` | "No soil profile mapped yet." | Empty-state. |

### 3.5 Model — `app_pages/model.py`

Goal: model pipeline config, honest evaluation, ablation, diagnostics and
feature importance.

| Element | Code | What it is / does |
|---|---|---|
| `st.set_page_config` | `page_title="AQUIS — model", page_icon=":material/model_training:", layout="wide"` | Per-page config (allowed because each page is its own script and it's the first command there). |
| `st.title` / `st.caption` | "Model — 30-day groundwater forecast" | Heading + the modelling contract: target is **GWL(t+120) − GWL(t)** on a 6-hourly grid; today's GWL is an anchor feature, never predicted. |
| `st.segmented_control` | `st.segmented_control("Metric basis", options=["level","delta"], format_func=…, default="level", key="model_basis")` | Toggles the whole page between absolute **level** outcome and 30-day **change** outcome. `format_func` maps `level`→"GWL level (m)", `delta`→"Change (m)". `key=` gives this widget a stable, named widget state slot. |
| `st.columns(3)` + `st.metric` ×3 | 3 headline KPIs | (a) "Models benchmarked" (unique model count) — `metadata` above with `caption` "30-day horizon (120 six-hour steps)"; (b) "Honest 30-d RMSE (non-overlap windows)" or fallback "Best RMSE (30 d)" — delta = vs persistence; (c) "XGBoost within ±0.5 m" — delta = percentage-point lead over persistence. |
| `st.subheader` | "Test-set metrics, 30 days horizon" | Table section. |
| `st.dataframe` | renamed model-metrics table | Columns Model / RMSE (m) / MAE (m) / R² / ±0.25 m / ±0.5 m / ±1.0 m / Rows; `use_container_width=True`, `hide_index=True`. |
| `st.subheader` + `st.altair_chart` | "RMSE by model" | Horizontal bar (teal), `y` sorted `-x` (best on top), tooltip rmse/mae/within_05. |
| `st.expander` | "Honest validation — spatial CV, overlap & residual autocorrelation" | Big-collapsed validation block. Inside: caption (why naive rows share ~99% of the same 30-day window, so they re-score on stride windows); 3-col metric row (Raw RMSE / Non-overlap RMSE / Independent 30-d windows); honest table dataframe; Spatial CV dataframe + caption (geographically held-out station blocks); 2-column split of residual checks. |
| `st.metric` ×2 + `st.caption` | Moran's I / Geary's C | Spatial-autocorrelation of per-station mean errors; `delta=` shows p-value ("p=0.021"); caption explains p<0.05 ⇒ spatial structure under-modelled. |
| `st.dataframe` | pooled ACF table | 6-hour-lag autocorrelation of residuals + effective-sample-size caption. |
| `st.write` | "…missing — run `07_evaluate.py`." | Plain-text fallback shown when an optional CSV is absent (component stays honest, no crash). |
| `st.expander` | "Feature ablation — is every driver paying its way?" | Table of configs with `Δ RMSE vs baseline`; caption: positive Δ = dropping the group HURTS (adds value). |
| `st.expander` | "Diagnostics — collinearity (VIF), feature influence & sensitivity" | Three-metric top row — **VIF>10 count** (collinearity redline, with ranked caption), **Max single-driver swing** (mean \|Δ in 30-day change\| moving 2.5→97.5th pct, delta = worst feature name), **Most important at stride** (permutation top feature). Below: permutation-importance bar, full VIF dataframe, sensitivity bar — each guarded by `if not vifd.empty:` / `if dperm` etc. |
| `st.expander` | "Feature importances / coefficients" | `st.radio("Source", ["xgboost gain", "ridge \|coef\|"], horizontal=True, key="imp_kind")` — **horizontal radio buttons** switch between tree gain vs absolute linear coefficients; one shared `st.altair_chart` bar below renders the top-20 of whichever source is active. |
| `st.expander` | "Pipeline configuration" | `st.json({...})` — **raw JSON viewer** (syntax-coloured, collapsible) of `traj/cfg` summary: horizons, steps/day, feature/station/district counts, train/val/test rows, extra features, build time; caption explains the 17 static soil/LULC columns are fetched but gated out (redundant with ordinals, per README items 6–7). |

### 3.6 Forecast — `app_pages/forecast.py` (trajectory v2)

Goal: one clean dark card presenting the **120 genuine 6-hour forecast
points** of the multi-horizon trajectory engine. Observed history stops at the
anchor; the forecast starts 6 h later and never re-predicts the anchor value.
This is the *only* forecast shown here.

| Element | Code | What it is / does |
|---|---|---|
| `st.set_page_config` / `st.title` / `st.caption` | heading block | icon `:material/troubleshoot:`; caption: "120 genuine 6-hour forecast points · multi-horizon trajectory model anchored on the latest observed reading". |
| `st.markdown(…, unsafe_allow_html=True)` | **CSS injection** | Tiny `<style>` block (bordered `stVerticalBlockBorderWrapper`, `.aquis-legend`, `.aquis-dir`). `unsafe_allow_html` is the deliberate escape hatch — the string is developer-authored, not user input. |
| `st.selectbox` | `st.selectbox("Station", stations, key="fc_station")` | All stations present in `outputs/predictions.csv`, recency-first. |
| `st.caption` | "Latest reading: **…** — the forecast anchor" | Reads `station_recency()` for the anchor timestamp. |
| `st.caption` ×2 | refresh-pipeline freshness | `refresh.publish.freshness_for(station)`: 🟢/🟠/⚪ data status, forecast-generated time, model version, anchor, staleness reason. Wrapped in `try/except` — **decorative, never breaks the page**. |
| `st.container(border=True)` | the whole trajectory card | Single bordered container holds everything below (the rounded-corner card). |
| `st.markdown` + `st.caption` | header | `"#### 30-day groundwater outlook"` + caption (120 points · anchor ↦ forecast range). |
| `st.markdown` | manual **compact legend** HTML | Recreates the chart legend by hand (observed line, q50 line, bright 90% band, q05·q95 edges) because the Altair chart hides its official legend for a cleaner card. |
| `st.altair_chart` | `trajectory_chart(pts, tail, anchor_t, height=480)`, `width="stretch"` | The card's centerpiece (`app_charts.py`): observed tail (grey, dashed) → anchor rule **"Forecast starts"** → 120-point q50 line (teal, wide) with a **bright q05–q95 band** (`COL_BAND`, 30% opacity) + bright edge lines (`COL_QEDGE`), continuous lines — **no dot markers**. X-domain = anchor−15 d → +30 d. |
| `st.markdown` | direction banner | Big ↑ / ↓ / → glyph + "Rising/Falling/Stable" + q50 change over 30 d (+ sign-accuracy when available), coloured per `DIR_GLYPH`. |
| `st.columns(6)` + `st.metric` ×6 | metric strip | **Anchor GWL (observed)** (never predicted), **+24 h**, **+7 d**, **+30 d** (the genuine 120th point, delta = change), **30 d change** (q50 at +30 d − anchor, delta = sign-accuracy), **Confidence** (label from `CONF_STYLE`; tooltip = reason + guidance). |
| `st.warning` ×2 | trajectory failure paths | "Trajectory engine unavailable: {e}" or the backend's `error` string — the page degrades to a warning instead of crashing. |

**Caching (visible behaviour):** `_trajectory_cached(station, version)` caches the
trajectory run keyed by a content fingerprint (`_traj_version()` hashes model
files, calibration/reliability tables, the aligned 6h table and today's date),
so the forecast refreshes exactly when data/weights change or daily.

### 3.7 Fleet — `app_pages/fleet.py`

Goal: whole-fleet 30-day outlook scan from `ml/11_fleet.py`'s
`outputs/fleet_forecast.csv`; headline change is the **q50 median** estimate so
degenerate pooled-regression outputs can't fake a decline.

| Element | Code | What it is / does |
|---|---|---|
| `st.set_page_config` / `st.title` / `st.caption` | heading block | icon `:material/query_stats:`; caption explains the q50 design choice. |
| `st.warning` + `st.stop` | `if df.empty: st.warning("No fleet snapshot found — run ml/11_fleet.py first."); st.stop()` | Missing-data guard: yellow callout, then halt the page. |
| `st.columns(6)` + `st.metric` ×6 | KPI strip | Stations scored / Districts / Median 30-d change / Declining (≤−0.3 m) / Recovering (≥+0.3 m) / Quality-gated out (`unreliable`). All over the `good` subset (`category != "unreliable"`). |
| `st.caption` | bucket caveat | "…point-estimate flags, not statistically significant moves." |
| `st.subheader` + `st.info` / `st.warning` + `st.dataframe` | "Significant movers" | If no station moved beyond its calibrated 90% band → blue `st.info` note; otherwise → red-tinted `st.warning` "{N} station(s)…" then a dataframe of the movers sorted by Δ. |
| `st.subheader` + `st.altair_chart` + `st.caption` | "Distribution of forecast 30-day change" | Histogram of q50 changes (`bin=alt.Bin(maxbins=40)`), **overlaid with two dashed `mark_rule` lines at ±0.3 m** (`strokeDash`), teal bars; caption identifies the dashed lines. |
| `st.subheader` + `st.dataframe` | "District rollup" | Per-district counts + median Δ + decline share (renamed columns). |
| `st.subheader` + `st.columns(3)` | "Station-level scan" filters | `c1.multiselect("Category", …)` (key `fleet_cat`), `c2.selectbox("District", ["All"]+…)` (key `fleet_dist`), `c3.checkbox("Movers only (|Δ| ≥ 0.3 m)")` (key `fleet_movers`). **Checkbox** = boolean toggle adding a filter. All three combine to filter the big table. |
| `st.dataframe` | the scan table | Renamed columns + a computed `"Δ > band"` boolean column; caption defines it. |
| `st.expander` + `st.markdown` | "Method & caveats" | Bullet list: pipeline path, why q50 beats pooled delta (pooled saturates on empty lag/driver columns), quality-gate rules, real-move signature, thresholds, refresh command. |

### 3.8 Assistant — `app_pages/assistant.py`

Goal: station-locked natural-language Q&A. Facts are computed deterministically
from `table_6h` + the 30-day model; `llama3.2:3b` via Ollama (local) only
phrases them. The instant data panel works even without Ollama.

| Element | Code | What it is / does |
|---|---|---|
| `st.set_page_config` / `st.title` | heading | icon `:material/smart_toy:`. |
| `st.warning` | "Ollama server not reachable…" | shown only when the server is down; stresses the instant panel still works. |
| `st.info` + `st.code` | "Ollama reachable, but model not pulled" + `st.code(f"ollama pull {model_name}", language="bash")` | Blue info box + a **copyable code block** (bash). |
| `st.expander` + `st.json` + `st.caption` | "Model status" | Full status dict as JSON + "press R to re-check" note. |
| `st.columns([2,1])` + `st.selectbox` + `st.caption` | Station picker | Dropdown of stations by recency (key `as_station`); right column shows total count. |
| `st.columns(5)` + `st.metric` ×5 | fact KPIs | Latest level (delta = date), 30-day change, 7-day change, Obs. range (min–max), Samples (n_obs). |
| `st.caption` | district context line | District median across analysed stations. |
| `st.success` / `st.warning` / `st.info` | 30-day outlook status | Green success with `<b>30-day outlook (XGBoost)</b>` (value + band + anchor); yellow warning if the forecast is unstable; blue info if no data for the selection. |
| `st.columns(2)` + `st.expander` ×2 | two fact panels | Left: **"Factors driving this station"** — `st.dataframe` of per-factor Spearman r / p / pairs with `column_config={"Spearman r": st.column_config.NumberColumn(format="%+.3f"), "p-value": …("%.4f")}` (custom number formatting per column), plus markdown for recent rain and an annual-table dataframe. Right: **"Precautions & strategy"** — one colour-coded markdown block per advisory (🔵/🟡/🔴 + material icon), separated by `st.divider()`, with a caption giving example prompts. |
| `st.markdown("---")` | horizontal rule | Divides the deterministic panels from the chat. |
| `st.session_state` | station pinning | `st.session_state.get("as_pinned_station")` / `as_messages` — **widget-independent state** that survives reruns. When the station changes, it seeds a fresh conversation from `as_messages` and writes `as_pinned_station`. |
| `st.chat_message` + `st.markdown` | past-messages loop | `with st.chat_message(msg["role"]): st.markdown(msg["content"])` renders each prior turn as a chat bubble (avatar + styling differ for user/assistant). |
| `st.chat_input` | `st.chat_input(f"Ask about {station}")` | **Chat input box** anchored at the bottom of the page. Truthy prompt → append user message → render it → render assistant bubble. |
| `st.spinner` | `with st.spinner("Querying AQUIS data + generating answer…")` | Shows a temporary spinner while `StationAssistant.answer(prompt, station, history)` runs (last 6 messages as context). Wrapped in `try/except` → friendly warning instead of a traceback. |
| `st.expander` + `st.markdown` | "About this assistant" | Stack/method notes: LangChain + Ollama local, model override via `AQUIS_OLLAMA_MODEL`, data/forecast/factor/precaution sources, scope, and the "GWL is a level in metres" convention. |

**Design note:** the LLM only *phrases* pre-computed facts — no hallucinations of
numbers; the deterministic panels above the chat remain the source of truth.

### 3.9 Sources — `app_pages/data_sources.py`

Goal: per-source quality summary for the selected 29-district / 600-station set.

| Element | Code | What it is / does |
|---|---|---|
| `st.title` / `st.caption` | "Data sources" | Heading + scope note. |
| `st.columns(2)` + `st.container(border=True)` | source cards | Each source with rows>0 becomes a bordered card in a 2-column grid (cards alternate columns). Inside: `st.markdown` with the human name **plus a markdown hyperlink to the NWDP dataset** (`[NWDP dataset ↗](url)`), a bordered `st.metric("Rows", …)`, and a caption with stations · districts · span · cap-dropped count. A second caption marks Gate-1-only discharge series with an `:orange-badge[discharge]` pill. |
| `st.subheader` + `st.caption` | "Sources with no data in selected districts" | A single caption joining markdown links for empty sources, minus their explanation (canal discharge/barrage gauges lie outside the 29 districts). |
| `st.header` | "Static hydrogeology" | Two-col section: **Soil texture** card (metric "Stations mapped", ISPIC/SoilGrids links, feature description) and **Groundwater extraction** card (metric "Assessments", CGWB link). Missing-data states show `:orange-badge[not loaded]` captions. |
| `st.header` | "Land use / land cover (static, ISRO Bhuvan)" | 3-col row: `st.metric("Districts")` + cycles caption | stacked-bar `st.altair_chart` (area share per LULC class, 2011-12, `legend=alt.Legend(columns=2, symbolLimit=30)`) | dataframe of top-10 LULC↔GWL r. The `st.expander` holds the full 28-district table. |
| `st.header` + `st.write` | "Association method" | `st.write` — plain paragraph (IDW 1/d² over 3 nearest gauges; nearest-gauge joins; district proxy for river). |
| `st.expander` + `st.dataframe` | "Association link counts" | Counts well↔gauge links per driver from `meta/assoc_*.csv`. |
| `st.caption` | "All values from `ml/data/`. Nothing here is committed or production." | Footer honesty note. |

### 3.10 Verification — `app_pages/verification.py` (deactivated)

Goal: realised-forecast quality. Reads `refresh/verification.py`'s
`data/refresh/verification_summary.json` + the sign-accuracy ledger — archived
forecasts scored against readings that have since materialised.

| Element | Code | What it is / does |
|---|---|---|
| `st.title` / `st.caption` | "Forecast verification" | Heading + description of archive/realised scoring. |
| `st.info` | no-summary guard | "No verification summary yet — it appears after forecast cycles…" |
| `st.columns` + `st.metric` ×5 | KPI strip | `Ledger rows`, 24 h / 7 d / 30 d `Sign accuracy` (with `delta`), RMSE, coverage vs 90% target — each `border=True`. |
| `st.subheader` + `st.dataframe` | "Windows (rolling)" | Per-window RMSE/sign-accuracy rows from `windows`; caption when nothing is scored yet. |
| `st.subheader` + `st.dataframe` | "Ledger sample" | Latest 200 ledger rows (`target_time` desc), `width="stretch"`, `hide_index=True`. |

Not wired into `app.py` nav — add an `st.Page("app_pages/verification.py", …)` to enable it.

---

## 4. Cross-page component reference

Every distinct Streamlit element used in the app, where it appears, and what it
does — the quick lookup table.

| Component | Where used | Purpose |
|---|---|---|
| `st.set_page_config` | app entry + Model / Forecast / Fleet / Assistant pages | Global options (title, material icon, wide layout). |
| `st.navigation` + `st.Page` | `app.py` | Declares the sidebar structure (4 active pages + optional ones when wired); `page.run()` renders the active page. |
| `st.title` | every page | Page heading. |
| `st.header` | Correlation, Sources | Section heading (level 2). |
| `st.subheader` | Stations, Model, Fleet | Block heading (level 3). |
| `st.caption` | everywhere | Small muted annotation, empty states, methodology notes. |
| `st.markdown` | everywhere | Inline/brief rich text; also `unsafe_allow_html=True` for CSS + custom cards. |
| `st.write` | Model (fallbacks), Sources | Plain paragraphs / data-frames equivalent; safe fallback text. |
| `st.code` | Assistant | Copyable bash block (`ollama pull …`). |
| `st.divider` | Assistant | Horizontal separator between advisories. |
| `st.json` | Model, Forecast, Assistant | Syntax-coloured JSON detail viewer. |
| `st.metric` | Overview, Correlation, Drivers, Stations, Model, Forecast, Fleet, Assistant, Sources | KPI card (label, value, optional delta line, `border=True` card). |
| `st.selectbox` | Drivers, Stations (sidebar), Forecast, Fleet, Assistant | Single-choice dropdown. |
| `st.multiselect` | Drivers, Fleet | Multi-choice dropdown; each selection drives one chart or table row. |
| `st.segmented_control` | Correlation (×2), Model | Single-select segmented button toggle. |
| `st.radio` | Model | Horizontal choice to switch importance source. |
| `st.checkbox` | Fleet | Boolean filter toggle. |
| `st.chat_input` + `st.chat_message` | Assistant | Conversational input + bubble-styled messages. |
| `st.sidebar` | Stations | Moves its children into the sidebar. |
| `st.columns` | every page | Responsive vertical split; ratios like `[1,2]`, `[2,3]`, `[3,1]`, `[2,1]`, 3/5/6-way for metric strips. |
| `st.container(border=True)` | Correlation, Forecast, Sources | Bordered card grouping. |
| `st.container(horizontal=True)` | Overview, Stations | Forces its children into one horizontal row. |
| `st.expander` | Correlation, Drivers, Model, Forecast, Fleet, Assistant, Sources | Collapsible section (+ optional material `icon=`). |
| `st.dataframe` | Correlation, Drivers, Model, Forecast, Fleet, Assistant, Sources | Interactive table; `hide_index`, `width`/`height`, `use_container_width`, `column_config` formats. |
| `st.altair_chart` | Correlation, Drivers, Stations, Model, Forecast, Fleet, Sources | Full Vega-Lite/Altair chart (the only way to get bands, facets, rules, dashed lines, interactivity). |
| `st.bar_chart` | Overview, Correlation, Stations | Quick native bar chart (no Altair needed). |
| `st.line_chart` | Correlation, Stations | Quick native line chart. |
| `st.download_button` | Forecast | PNG snapshot download. |
| `st.spinner` | Assistant | Loading indicator around an expensive call. |
| `st.success` / `st.warning` / `st.info` | Correlation, Forecast, Fleet, Assistant, Sources | Status callouts (green / yellow / blue). |
| `st.stop` | Correlation, Drivers, Stations, Fleet | Early halt guard (missing data / deselected control). |
| `st.session_state` | Assistant | Cross-rerun state (pinned station + message history). |
| `st.cache_data` | `_utils`, forecast page | Decorator that caches loader results (TTL, max_entries) and expensive trajectory runs. |

---

## 5. Under the hood

### Data loaders (`ml/_utils.py`) — all `@st.cache_data(ttl="10m", …)`

| Loader | Reads | Returns |
|---|---|---|
| `load_manifest()` | `data/meta/manifest.json` | Per-source QC summary dict (rows, stations, districts, span, dropped). |
| `load_selected_gwl()` | `data/meta/selected_gwl_stations.csv` | The 29-district / ~600-station GWL selection. |
| `load_district_set()` | `data/meta/selected_districts.json` | The selected district names. |
| `load_report()` | `outputs/correlation_report.csv` | The 3-mode correlation report. |
| `load_lag_curves()` | `outputs/lag_curves.csv` | Rain-lead vs GWL lag curve. |
| `load_table()` | `data/aligned/table.parquet` (daily) | Daily aligned driver+GWL table. |
| `load_table_6h()` | `data/aligned/table_6h.parquet` (6-hourly) | The 6-hourly grid the forecast engine uses. |
| `station_recency()` | deriv. from `table_6h` | Most-recent reading per station (cached on top of `load_table_6h`). |
| `nice()`, `source_nice()` | `DRIVER_LABELS` / `SOURCE_LABELS` | Machine names → human labels ("rain_7d" → "Rain, 7 days"). |

### Supporting modules

- `ml/app_charts.py` — `trajectory_chart()`, colour constants
  (`COL_OBSERVED`, `COL_TRAJECTORY`, `COL_BAND`, `COL_QEDGE`) shared by the
  Forecast page; `snapshot.py` reuses the same palette for its PNG export
  (kept for the export tests).
- `ml/snapshot.py` — `trajectory_snapshot_png()`: matplotlib (Agg) render of the
  forecast card; no longer wired to a button on the Forecast page (exercised by tests).
- `ml/_model.py` — model-page loaders (`load_model_metrics`,
  `load_honest_metrics`, `load_spatial_cv_metrics`, `load_importance`,
  `load_coefficients`, `load_ablation`, `load_diagnostics`, `load_predictions`,
  fleet loaders…).
- `ml/_trajectory.py` — `trajectory_forecast(station)` — the multi-horizon
  engine behind the Forecast card and `/forecast/<slug>`.
- `ml/_assistant.py` — `StationAssistant` (facts + Ollama phrasing),
  `ollama_status`, station/district name lists.
- `ml/_soil.py`, `ml/_lulc.py` — static ISRIC soil + ISRO Bhuvan LULC loaders.
- `ml/refresh/` — the 6-hourly daemon: `publish.freshness_for()` (nav-level
  freshness caption), `verification.py` (Verification page), `future_drivers.py`
  (CWC, Open-Meteo) used by the trajectory engine.

### Key outputs read by the app

`ml/outputs/correlation_report.csv` · `lag_curves.csv` · `correlation_lulc.csv` ·
`fleet_forecast.csv` · `predictions.csv` · `ml/models/traj_*.json|joblib`.

> The dashboards' own conventions: status callouts (`success`=good,
> `warning`=degrade-but-continue, `info`=no-data hint), `st.stop()` guards all
> missing-data paths, and every static feature is explicitly captioned as
> "not fed to the model".