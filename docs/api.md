# AQUIS API Reference

Two HTTP services power the platform:

| Service | URL | Used by |
|---|---|---|
| **Node.js API** (Express) | `http://localhost:3000` | Frontend, mobile clients |
| **Python ML service** (Flask, headless) | `http://localhost:5000` | Backend only (proxied) |

The **frontend should talk to the Node API on `:3000`**. ML intelligence (recency‑sorted station/district lists, per‑station facts, XGBoost forecasts, fleet recovery ranking, and the LLM data assistant) is served by the Python service and proxied by the Node backend under `/ml/*`. `ML_SERVICE_URL` points the Node gateway at the Python service.

> Legacy numeric endpoints (`GET /ml/forecast/:stationId`, `/ml/risk/:unitId`, `/ml/anomalies/...`) are **kept for backward compatibility but deprecated** — they read from the DB model registry, not the live ML stack. Prefer the `/ml/live/*` endpoints below.

---

## Conventions

- **Encoding:** always JSON (`Content-Type: application/json`). Query parameters for GETs.
- **Auth:** none currently — services run on localhost/LAN. CORS is open on both services.
- **Errors:** every failure returns `{ "error": "...", "detail": "..." }` with an appropriate status code.
- **Slugs:** ML endpoints identify stations by **slug** — `Station_Agency_shortHash`, e.g. `ASHADHA PRATHMIK VIDYALAYA_UPGW_5f5b3671`. Slugs contain spaces; **URL‑encode them** (`encodeURIComponent` in JS), e.g. `ASHADHA%20PRATHMIK%20VIDYALAYA_UPGW_5f5b3671`. Always fetch the current slug from `/ml/stations` — never hand‑type one.

### Status codes

| Code | Meaning |
|---|---|
| 200 | OK |
| 400 | Missing/invalid request (e.g. no `question` on chat) |
| 404 | Unknown slug / no model trained / snapshot missing |
| 500 | Internal Python error |
| 502 | ML service down or unreachable (start `python ml/app.py`) |
| 503 | Assistant unavailable (Ollama not running) |

---

## Frontend endpoints (Node `:3000`)

### 1. `GET /ml/health`
Service status + model/Ollama health. Fast, call on app boot.

```json
{
  "available": true,
  "status": {
    "status": "ok", "service": "aquis-ml", "version": "2.0.0",
    "trained_stations": 92,
    "snapshot_age_seconds": 48724.09,
    "ollama": false,
    "dataset": "2026-09-05"
  }
}
```

### 2. `GET /ml/stations`
**Recency‑sorted** station list (newest reading first). This is the canonical station picker source.

| Param | Type | Default | Notes |
|---|---|---|---|
| `district` | string | — | Exact district name filter |
| `q` | string | — | Case‑insensitive substring on station/district name |
| `limit` | int | 200 | Max 2000 |

```json
{
  "count": 200,
  "stations": [
    {
      "slug": "ASHADHA PRATHMIK VIDYALAYA_UPGW_5f5b3671",
      "display": "ASHADHA PRATHMIK VIDYALAYA",
      "district": "KAUSHAMBI",
      "agency": "UPGW",
      "state": "Uttar Pradesh",
      "has_model": true,
      "last_ts": "2026-09-05 18:00:00",
      "last_epoch": 1788631200.0
    }
  ]
}
```

Use `has_model` to decide whether the station can produce a forecast.

### 3. `GET /ml/stations/:slug`
Latest telemetry facts (no forecast, no LLM) for one station.

```json
{
  "data": {
    "level": "station",
    "station": "ASHADHA PRATHMIK VIDYALAYA",
    "district": "KAUSHAMBI",
    "last": -0.785,
    "change_7d": -0.029, "change_30d": -0.029, "change_60d": -0.128, "change_180d": -0.519,
    "outliers": 0,
    "district_median": -0.41,
    "district_n_stations": 43,
    "slug": "ASHADHA PRATHMIK VIDYALAYA_UPGW_5f5b3671",
    "has_model": true
  }
}
```

`change_*` values are the observed level change over the window (m). Negative = falling/deepening.

### 4. `GET /ml/districts`
Districts sorted by most‑recent reading.

```json
{ "count": 34, "districts": [
  { "district": "AGRA", "n_stations": 43, "last_ts": "2026-09-05 18:00:00", "last_epoch": 1788631200.0 }
] }
```

### 5. `GET /ml/live/models`
Trained XGBoost stations (slugs + `trained_at`). `count` = number of trained models.

### 6. `GET /ml/live/forecast/:slug?days=30`
Daily XGBoost forecast from the station's **last observed reading** (recursive point + q05/q95 quantiles, calibration‑widened 90% interval, plausible‑check).

| Param | Type | Default | Range |
|---|---|---|---|
| `days` | int | 30 | 7–90 |

```json
{
  "trained": true,
  "slug": "ASHADHA PRATHMIK VIDYALAYA_UPGW_5f5b3671",
  "station": "ASHADHA PRATHMIK VIDYALAYA",
  "district": "KAUSHAMBI",
  "as_of": "2026-09-05",
  "last_obs": { "date": "2026-09-05", "value": -6.24 },
  "horizon_days": 30,
  "dates":   ["2026-09-06T00:00:00", "...", "2026-10-05T00:00:00"],
  "point":   [-6.262, "...", -6.293],
  "q50":     [-6.250, "...", -6.278],
  "lower":   [-6.746, "...", -6.910],
  "upper":   [-5.935, "...", -5.712],
  "change_7d_pred": -0.0115,
  "change_30d_pred": -0.0304,
  "change_60d_pred": null,
  "plausible": true,
  "obs_min": -12.51,
  "obs_max": -4.64
}
```

Field notes:
- `point` = best estimate; `lower`/`upper` = **calibrated** 90% interval (approx. q05/q95, widened to reach target coverage); `q50` = uncalibrated recursive median.
- `plausible=false` means the model's level drifted outside the station's observed range — render with a "directional only" warning.
- `404` `{error:"no model"}` when the station isn't trained.

### 7. `GET /ml/live/fleet/forecasts`
Precomputed fleet snapshot (built by `ml/agent/build_fleet_forecast.py`) — same shape the assistant answers "fleet forecast" questions from.

```json
{
  "computed_at": "2026-09-08 14:21:10",
  "stations": {
    "<slug>": { "station": "...", "change_30d_pred": ..., "change_60d_pred": ...,
                "day30_pred": ..., "day60_pred": ..., "plausible": true }
  }
}
```

### 8. `GET /ml/live/fleet/recovery`
District **recovery ranking** — median observed level change over the window (rising = recovery). ~8s on first call per window; memoized after.

| Param | Type | Default | Range |
|---|---|---|---|
| `window_days` | int | 60 | 7–365 |
| `top` | int | 5 | 1–20 |
| `min_stations` | int | 3 | — |

### 9. `GET /ml/live/fleet/scan`
Stations whose **forecast** predicts a decline ≥ `threshold` m at `horizon` days.

| Param | Type | Default |
|---|---|---|
| `district` | string | — (all) |
| `threshold` | float | 0.3 |
| `horizon` | int | 60 |

Returns `{ "available": true, "matches": [...], "unreliable": [...], "threshold": ... }`. Only stations with `plausible` forecasts are reported.

### 10. `POST /ml/assistant/chat`
Station‑locked natural‑language assistant (Ollama `llama3.2:3b` behind it). Answers are **grounded in live parquet + fleet facts** — the station is pinned by `station` or `station_slug` and refuses to change topic to another station.

```json
// Request
{ "question": "tell me about this station", "station": "ASHADHA PRATHMIK VIDYALAYA" }
// or
{ "question": "why is it falling?", "station_slug": "ASHADHA PRATHMIK VIDYALAYA_UPGW_5f5b3671", "model": "llama3.2:3b" }
```

```json
// Response
{ "answer": "...", "station": "ASHADHA PRATHMIK VIDYALAYA", "station_slug": "...", "facts": { ... } }
```

- `503` when Ollama is down (`{"hint": "is Ollama running?...", "error": "assistant failed"}`).

---

## Backend developers — Python service contract (`:5000`)

Internal service; the Node gateway proxies everything above. Run it with:

```bash
cd <repo-root>/ml
ml/venv/bin/python ml/app.py          # http://localhost:5000 (PORT env to change)
```

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | same payload as `/ml/health` |
| `/stations` | GET | `?district=&q=&limit=` |
| `/stations/:slug` | GET | facts |
| `/districts` | GET | recency‑sorted |
| `/forecast/:slug` | GET | `?days=` |
| `/fleet/forecasts` | GET | snapshot |
| `/fleet/recovery` | GET | `?window_days=&top=&min_stations=` |
| `/fleet/scan` | GET | `?district=&threshold=&horizon=` |
| `/models` | GET | trained slugs |
| `/assistant/chat` | POST | `{question, station, station_slug?, model?}` |

Gateway wiring (Node, `back-end/services/mlGateway.js`): `ML_SERVICE_URL` (default `http://localhost:5000`), timeout `ML_TIMEOUT_MS` (default 60000). The gateway now forwards **query strings** (`pathname + search`).

**Performance characteristics** (first call after server start):
- `/stations`, `/districts`, `/stations/:slug` — a few seconds (as they build the full index).
- `/forecast/:slug` — ~7–10 s first time per slug (feature pipeline + calibration); near‑instant after caching.
- `/ml/live/fleet/recovery` — ~8 s per new window; memoized.
- Everything else is sub‑second (snapshot, models, health).

---

## Quickstarts

**Find a station, then its forecast:**
```bash
curl "http://localhost:3000/ml/stations?q=vidyalaya&limit=5"
curl "http://localhost:3000/ml/live/forecast/ASHADHA%20PRATHMIK%20VIDYALAYA_UPGW_5f5b3671?days=30"
```

**Typical frontend flow:**
1. Boot → `GET /ml/health` (show offline banner if `available:false` / `ollama:false`).
2. Station picker → `GET /ml/stations?q=...` (sort is recency; render `last_ts`).
3. Detail view → `GET /ml/stations/:slug` for KPIs.
4. Forecast tab → `GET /ml/live/forecast/:slug?days=30`; chart `dates` vs `point` with `lower`/`upper` band.
5. Fleet view → `GET /ml/live/fleet/recovery?window_days=60` and/or `GET /ml/live/fleet/scan`.
6. Assistant → `POST /ml/assistant/chat`.