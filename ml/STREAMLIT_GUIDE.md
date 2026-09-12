# AQUIS dashboard — Streamlit component guide

Every Streamlit widget in the AQUIS groundwater dashboard, explained at the
same depth as the Correlation page below. The app is a multi-page Streamlit
app under `ml/`; every page is a plain script that runs top-to-bottom on each
rerun.

**Live navigation (4 pages):** Assistant (default) · Correlation · Forecast ·
Sources — declared in `app.py` via `st.navigation`. The **Verification**
page (`app_pages/verification.py`) reads the refresh pipeline's realisations
and is **not wired into the nav** (add an `st.Page` to enable it). The legacy
explorer pages (Overview, Drivers, Stations, Model, Fleet) were **removed**
— the fleet/model/correlate outputs they displayed are still produced by the
numbered pipeline (`11_fleet.py`, `07_evaluate.py`, `05_correlate.py`,
`10_ablate.py`, `12_diagnostics.py`) and covered below via the live pages.

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
   - 3.1 Correlation (active — the "depth template")
   - 3.2 Forecast (active — trajectory v2)
   - 3.3 Assistant (active — default page)
   - 3.4 Sources (active)
   - 3.5 Verification (not in nav)
4. Cross-page component reference
5. Under the hood (data loaders, labels, outputs)

---

## 1. Application entry — `ml/app.py`

```python
st.set_page_config(page_title="AQUIS — groundwater explorer",
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

### 3.1 Correlation — `app_pages/correlation.py`

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

### 3.2 Forecast — `app_pages/forecast.py` (trajectory v2)

Goal: one clean dark card presenting the **120 genuine 6-hour forecast
points** of the multi-horizon trajectory engine. Observed history stops at the
anchor; the forecast starts 6 h later and never re-predicts the anchor value.
This is the *only* forecast shown here.

| Element | Code | What it is / does |
|---|---|---|
| `st.set_page_config` / `st.title` / `st.caption` | heading block | icon `:material/troubleshoot:`; caption: "120 genuine 6-hour forecast points · multi-horizon trajectory model anchored on the latest observed reading". |
| `st.markdown(…, unsafe_allow_html=True)` | **CSS injection** | Tiny `<style>` block (bordered `stVerticalBlockBorderWrapper`, `.aquis-legend`, `.aquis-dir`). `unsafe_allow_html` is the deliberate escape hatch — the string is developer-authored, not user input. |
| `st.sidebar.selectbox` ×2 | `sidebar_station_picker(table6, stations)` (`_utils.py`) | **Shared sidebar pickers** (keys `sb_district`/`sb_station`, same on every page): District dropdown, then Station dropdown filtered to that district (recency-first, preds-backed). Selection persists across pages — Assistant follows along. Page shows a `**{station}** · {district}` caption instead of its own picker. |
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

### 3.3 Assistant — `app_pages/assistant.py`

Goal: station-locked natural-language Q&A. Facts are computed deterministically
from `table_6h` + the 30-day model; `llama3.2:3b` via Ollama (local) only
phrases them. The instant data panel works even without Ollama.

| Element | Code | What it is / does |
|---|---|---|
| `st.set_page_config` / `st.title` | heading | icon `:material/smart_toy:`. |
| `st.warning` | "Ollama server not reachable…" | shown only when the server is down; stresses the instant panel still works. |
| `st.info` + `st.code` | "Ollama reachable, but model not pulled" + `st.code(f"ollama pull {model_name}", language="bash")` | Blue info box + a **copyable code block** (bash). |
| `st.expander` + `st.json` + `st.caption` | "Model status" | Full status dict as JSON + "press R to re-check" note. |
| `st.sidebar.selectbox` ×2 | `sidebar_station_picker(df, stations)` (`_utils.py`) | **Shared sidebar pickers** — same `sb_district`/`sb_station` keys as Forecast, so both pages stay on one station. District dropdown + district-filtered Station dropdown (recency-first). |
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

### 3.4 Sources — `app_pages/data_sources.py`

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

### 3.5 Verification — `app_pages/verification.py` (not in nav)

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
| `st.set_page_config` | app entry + Forecast / Assistant pages | Global options (title, material icon, wide layout). |
| `st.navigation` + `st.Page` | `app.py` | Declares the sidebar structure (4 active pages + Verification when wired); `page.run()` renders the active page. |
| `st.title` | every page | Page heading. |
| `st.header` | Correlation, Sources | Section heading (level 2). |
| `st.subheader` | Verification | Block heading (level 3). |
| `st.caption` | everywhere | Small muted annotation, empty states, methodology notes. |
| `st.markdown` | everywhere | Inline/brief rich text; also `unsafe_allow_html=True` for CSS + custom cards. |
| `st.write` | Sources | Plain paragraphs / data-frames equivalent; safe fallback text. |
| `st.code` | Assistant | Copyable bash block (`ollama pull …`). |
| `st.divider` | Assistant | Horizontal separator between advisories. |
| `st.json` | Assistant | Syntax-coloured JSON detail viewer. |
| `st.metric` | Correlation, Forecast, Assistant, Sources, Verification | KPI card (label, value, optional delta line, `border=True` card). |
| `st.selectbox` | Forecast, Assistant | Single-choice dropdown. |
| `st.multiselect` | — (removed with the Drivers/Fleet pages) | Multi-choice dropdown pattern; retained here for reference. |
| `st.segmented_control` | Correlation (×2) | Single-select segmented button toggle. |
| `st.radio` | — (removed with the Model page) | Horizontal choice pattern; retained here for reference. |
| `st.checkbox` | — (removed with the Fleet page) | Boolean filter toggle pattern; retained here for reference. |
| `st.chat_input` + `st.chat_message` | Assistant | Conversational input + bubble-styled messages. |
| `st.sidebar` | — (removed with the Stations page) | Moves its children into the sidebar. |
| `st.columns` | every page | Responsive vertical split; ratios like `[1,2]`, `[2,3]`, `[3,1]`, `[2,1]`, 3/5/6-way for metric strips. |
| `st.container(border=True)` | Correlation, Forecast, Sources | Bordered card grouping. |
| `st.container(horizontal=True)` | — (removed with the Overview/Stations pages) | Forces its children into one horizontal row. |
| `st.expander` | Correlation, Forecast, Assistant, Sources | Collapsible section (+ optional material `icon=`). |
| `st.dataframe` | Correlation, Forecast, Assistant, Sources, Verification | Interactive table; `hide_index`, `width`/`height`, `use_container_width`, `column_config` formats. |
| `st.altair_chart` | Correlation, Forecast, Sources | Full Vega-Lite/Altair chart (the only way to get bands, facets, rules, dashed lines, interactivity). |
| `st.bar_chart` | Correlation | Quick native bar chart (no Altair needed). |
| `st.line_chart` | Correlation | Quick native line chart. |
| `st.download_button` | Forecast | PNG snapshot download. |
| `st.spinner` | Assistant | Loading indicator around an expensive call. |
| `st.success` / `st.warning` / `st.info` | Correlation, Forecast, Assistant, Sources, Verification | Status callouts (green / yellow / blue). |
| `st.stop` | Correlation, Forecast, Assistant, Verification | Early halt guard (missing data / deselected control). |
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
- `ml/_model.py` — loaders behind the live pages (`load_importance`,
  `load_predictions`) plus the shared forecast path (`forward_forecast`,
  quantile loaders) and the offline fleet/model loaders (`load_fleet_table`,
  `load_model_metrics`, `load_honest_metrics`, …) used by the numbered
  pipeline outputs.
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
`predictions_2026.parquet` · `fleet_forecast.csv` (offline scan output, read by
`_model.load_fleet_table`) · `ml/models/traj_*.json`.

> The dashboards' own conventions: status callouts (`success`=good,
> `warning`=degrade-but-continue, `info`=no-data hint), `st.stop()` guards all
> missing-data paths, and every static feature is explicitly captioned as
> "not fed to the model".