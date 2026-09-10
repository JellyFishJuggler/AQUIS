"""refresh/future_drivers.py — per-step future feature construction (mode="future").

For each of the MAX_H=120 six-hourly steps ahead of a forecast anchor the direct
multi-horizon model consumes a 33-column row (30 base + h/h_sin/h_cos). The
production builder replicates the anchor row to every horizon (mode="flat").
This module builds the genuinely-time-indexed alternative where the DRIVER and
CALENDAR features are reconstructed for each future timestamp while the
GWL-memory features (gwl, lag*, gwl_roll*) stay pinned to the anchor row —
the exact convention of the validated long-frame: memory describes information
available at time t, future forcing flows only through the driver features.

Source precedence for each step (never fabricates):
    weather (temp/humidity/solar/wind_speed/pressure) and rain
        -> Open-Meteo (archive for backtest / forecast <=16 d for live),
           one daily value held constant over the day's 4 six-hour slots,
           rain split uniformly over the 4 slots;
        -> else climatology (station day-of-year for weather,
           district doy+6h-slot for rain);
        -> else NaN (unavailable).
    river_level
        -> CWC / FMISC 7-day river forecast when the step lands inside the
           covered window AND the district has a CWC gauge: the forecast is
           injected as a delta over the station's own river-level baseline
           (station day-of-year climatology, else last observed level) so the
           units stay on the station's gauge datum and only the real CWC
           trend is borrowed (never raw metres from another gauge);
        -> else station day-of-year climatology;
        -> else last observed level (persistence);
        -> else NaN.
    canal_level -> station day-of-year climatology -> last observed -> NaN
        (canals are regulated; there is no public level forecast anywhere).
    calendar/diurnal          -> computed from the step's own timestamp.

Everything is float32 and column-ordered to match ``traj_config.feature_cols``,
so the matrix can be fed straight to the same DMatrix.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIM_PATH = ROOT / "data" / "meta" / "driver_climatology.parquet"
RAIN_CLIM_PATH = ROOT / "data" / "meta" / "district_rain_climatology.parquet"
OM_PATH = ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet"
RFC_PATH = ROOT / "data" / "cfs" / "river_forecast_cwc.parquet"

RFC_WINDOW_DAYS = 7            # CWC forecast window (level forecast horizon)

STEP_H = 6
FORECAST_STEPS = 64          # 16 d on the 6 h grid  (default forecast window)
DEFAULT_STEPS = 120          # 30 d horizon

H_COLS = ("h", "h_sin", "h_cos")

OM_COL_MAP = {
    "rain": "rain_mm", "temp": "temp_c", "humidity": "humidity_pct",
    "solar": "solar_wm2", "wind_speed": "wind_kmh", "pressure": "pressure_hpa",
}
WEATHER_COLS = ["temp", "humidity", "solar", "wind_speed", "pressure"]
RAIN_COLS = ["rain"]
RC_COLS = ["river_level", "canal_level"]
DRIVER_COLS = ["rain", "temp", "humidity", "solar", "wind_speed", "pressure",
               "river_level", "canal_level"]
ROLL_MEAN_PAIRS = (("temp_7d", "temp", 28), ("humidity_7d", "humidity", 28))   # out, src, win
RAIN_ROLL_PAIRS = (("rain_1d", 4), ("rain_7d", 28), ("rain_30d", 120))
GWL_MEAN_ROLLS = (("gwl_roll7_mean", 28), ("gwl_roll30_mean", 120))
GWL_STD_ROLLS = (("gwl_roll7_std", 28), ("gwl_roll30_std", 120))
GWL_LAGS = (("lag1", 1), ("lag4", 4), ("lag8", 8), ("lag28", 28), ("lag120", 120))
TAIL_BACK = 128              # six-hour steps of observed tail for roll windows


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _horizon_feats(h: int, max_h: int = DEFAULT_STEPS) -> dict:
    return {"h": float(h), "h_sin": float(np.sin(2 * np.pi * h / max_h)),
            "h_cos": float(np.cos(2 * np.pi * h / max_h))}


def _doy_hour_features(t: pd.Timestamp) -> dict:
    mo = t.month
    doy = t.dayofyear
    hr = t.hour + t.minute / 60.0
    return {
        "year": float(t.year),
        "month_sin": float(np.sin(2 * np.pi * mo / 12)),
        "month_cos": float(np.cos(2 * np.pi * mo / 12)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "hour_sin": float(np.sin(2 * np.pi * hr / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hr / 24)),
        "monsoon": float(doy - 152),
    }


def _sixh_tail(hist: pd.DataFrame, col: str, anchor_t: pd.Timestamp,
               n_back: int = TAIL_BACK) -> np.ndarray:
    """Contiguous 6h driver tail ending at ``anchor_t`` (ffill; rain->0)."""
    times = pd.date_range(anchor_t - pd.Timedelta(hours=n_back * STEP_H),
                          anchor_t, freq="6h")
    if col not in hist.columns:
        return np.full(len(times), np.nan)
    sub = hist[["time", col]].sort_values("time").drop_duplicates("time")
    merged = pd.DataFrame({"time": times}).merge(sub, on="time", how="left")
    if col == "rain":
        merged[col] = merged[col].ffill().fillna(0.0)
    else:
        merged[col] = merged[col].ffill()
    return merged[col].to_numpy(dtype=np.float64)


class ClimatologyIndex:
    """Keyed lookups for station/doy weather and district/doy/hour rain."""

    def __init__(self, weather: pd.DataFrame | None, rain: pd.DataFrame | None):
        self._w, self._r = weather, rain
        self._wo, self._ro = None, None
        if weather is not None and not weather.empty:
            w = weather.copy()
            w = w.drop_duplicates(subset=["Station", "doy"])
            idx = pd.MultiIndex.from_tuples(
                [(str(s), int(d)) for s, d in zip(w["Station"], w["doy"])],
                names=("Station", "doy"))
            self._wo = w.set_index(idx)
        if rain is not None and not rain.empty:
            r = rain.copy()
            r["District"] = r["District"].astype(str).str.upper()
            r = r.groupby(["District", "doy", "hour"], as_index=False)["rain"].mean()
            idx = pd.MultiIndex.from_tuples(
                [(d, int(o), int(h)) for d, o, h in zip(r["District"], r["doy"], r["hour"])],
                names=("District", "doy", "hour"))
            self._ro = r.set_index(idx)

    def weather(self, station: str, doy: int) -> pd.Series | None:
        if self._wo is None:
            return None
        try:
            i = self._wo.index.get_loc((station, int(doy)))
            return self._wo.iloc[i]
        except KeyError:
            return None

    def weather_present(self) -> bool:
        return self._wo is not None

    def rain(self, district: str, doy: int, hour: int) -> float | None:
        if self._ro is None:
            return None
        try:
            return float(self._ro.loc[(district.upper(), int(doy), int(hour)), "rain"])
        except KeyError:
            return None


class RiverForecastIndex:
    """District-level CWC day-1..7 river level forecast, as delta vs observed.

    Each CWC site records an observed (current) level plus 7 forecast levels
    on successive days. Levels live on that gauge's own datum, so the only
    transferable signal is the *delta* (forecast level minus current level) —
    ``delta(district, ts)`` returns the district-mean of those deltas. The
    caller anchors them onto the station's own river-level baseline.
    """

    def __init__(self, rfc: pd.DataFrame | None):
        self._dates: dict[str, dict[object, float]] = {}
        self._bounds: dict[str, tuple[object, object]] = {}
        self._keys = {}
        if rfc is None or rfc.empty:
            return
        if "d1_wl" in rfc.columns:
            g = rfc.dropna(subset=["observed_wl", "d1_wl"], how="all")
        else:
            g = rfc.dropna(subset=["observed_wl"])
        if g.empty:
            return
        for district, sub in g.groupby("district"):
            by_date: dict[object, list[float]] = {}
            for _, row in sub.iterrows():
                obs_dt = row.get("observed_dt")
                obs_wl = row.get("observed_wl")
                if pd.isna(obs_dt) or not np.isfinite(float(obs_wl)):
                    obs_dt = row.get("d1_dt")
                    obs_wl = row.get("d1_wl")
                    if pd.isna(obs_dt) or not np.isfinite(float(obs_wl)):
                        continue
                base = float(obs_wl)
                pairs = [(pd.Timestamp(obs_dt).normalize(), base)]
                for day in range(1, RFC_WINDOW_DAYS + 1):
                    dt = row.get(f"d{day}_dt")
                    wl = row.get(f"d{day}_wl")
                    if pd.isna(dt) or pd.isna(wl):
                        continue
                    pairs.append((pd.Timestamp(dt).normalize(), float(wl)))
                for date, wl in pairs:
                    by_date.setdefault(date, []).append(wl - base)
            self._dates[district] = {d: float(np.mean(v)) for d, v in by_date.items()}
            if by_date:
                self._bounds[district] = (min(by_date), max(by_date))
                self._keys[district] = sorted(by_date)

    def covers(self, district: str, t) -> bool:
        try:
            d = pd.Timestamp(t).normalize()
        except (TypeError, ValueError):
            return False
        if district not in self._bounds:
            return False
        lo, hi = self._bounds[district]
        return lo <= d <= hi

    def delta(self, district: str, t) -> float | None:
        try:
            d = pd.Timestamp(t).normalize()
        except (TypeError, ValueError):
            return None
        keys = self._keys.get(district)
        if not keys:
            return None
        i = bisect.bisect_right(keys, d) - 1
        if i < 0:
            return None
        v = self._dates[district][keys[i]]
        return float(v) if np.isfinite(v) else None


def _load_river_forecast() -> pd.DataFrame | None:
    """Latest CWC snapshot from the rolling cache (None when absent/empty)."""
    if not RFC_PATH.exists():
        return None
    df = pd.read_parquet(RFC_PATH)
    if df.empty or "fetched_at" not in df.columns:
        return None
    last = df["fetched_at"].max()
    out = df[df["fetched_at"] == last].reset_index(drop=True)
    return out if not out.empty else None


def _last_obs(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else None


def _series_get(cw, key: str) -> float:
    if cw is None:
        return float("nan")
    try:
        v = cw.get(key)
    except (KeyError, AttributeError):
        return float("nan")
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def _rc_values(cw, rfi: RiverForecastIndex | None, district: str, t,
               river_last: float | None, canal_last: float | None) -> tuple[float, float, str, str]:
    """Per-step river/canal values + source labels.

    river: CWC 7-day delta over station baseline -> climatology -> observed
        persistence -> NaN. canal: climatology -> persistence -> NaN.
    """
    clim_r = _series_get(cw, "river_level")
    clim_c = _series_get(cw, "canal_level")

    river = float("nan")
    river_src = "river_canal:unavailable"
    if rfi is not None and rfi.covers(district, t):
        delta = rfi.delta(district, t)
        baseline = clim_r if np.isfinite(clim_r) else (
            river_last if river_last is not None else np.nan)
        if delta is not None and np.isfinite(baseline):
            river = float(baseline + delta)
            river_src = "river_canal:river_cwc_forecast"
    if not np.isfinite(river) and np.isfinite(clim_r):
        river, river_src = float(clim_r), "river_canal:river_climatology"
    if not np.isfinite(river) and river_last is not None and np.isfinite(river_last):
        river, river_src = float(river_last), "river_canal:river_persistence"

    canal = float("nan")
    canal_src = "river_canal:unavailable"
    if np.isfinite(clim_c):
        canal, canal_src = float(clim_c), "river_canal:canal_climatology"
    elif canal_last is not None and np.isfinite(canal_last):
        canal, canal_src = float(canal_last), "river_canal:canal_persistence"
    return river, canal, river_src, canal_src


def _om_lookup(om_frame: pd.DataFrame, district: str) -> dict:
    if om_frame is None or om_frame.empty:
        return {}
    g = om_frame.copy()
    dists = g["District"].astype(str).str.upper().to_list()
    dates = pd.to_datetime(g["date"]).dt.normalize().to_list()
    return {tup: i for i, tup in enumerate(zip(dists, dates))}


# ---------------------------------------------------------------------------
# public builders
# ---------------------------------------------------------------------------
def future_weather_matrix(*, station: str, district: str, anchor_t: pd.Timestamp,
                          steps: int = DEFAULT_STEPS, om_frame: pd.DataFrame | None = None,
                          climatology: pd.DataFrame | None = None,
                          rain_clim: pd.DataFrame | None = None,
                          river_forecast: pd.DataFrame | None = None,
                          river_last: float | None = None, canal_last: float | None = None,
                          forecast_days: int = int(FORECAST_STEPS / 4)) -> tuple[pd.DataFrame, list[dict]]:
    """Per-step driver values + explicit per-step source resolution.

    Returns (df[steps, DRIVER_COLS], src_records[steps]).
    """
    hs = np.arange(1, steps + 1, dtype=np.int64)
    ts = anchor_t + pd.to_timedelta(hs * STEP_H, unit="h")
    om_idx = _om_lookup(om_frame, district)
    clim = ClimatologyIndex(climatology, rain_clim)
    rfi = RiverForecastIndex(river_forecast) if river_forecast is not None else None
    dkey = str(district).upper()

    driver_cols = {c: np.full(steps, np.nan) for c in DRIVER_COLS}
    srcs = []
    for k, t in enumerate(ts):
        rec = {"weather": "unavailable", "rain": "unavailable", "river_canal": "unavailable"}
        cw = None
        # One OM daily row covers the day's 4 six-hour slots: 06, 12, 18 and the
        # overnight 00:00 of the NEXT day. Bucket the 00:00 step under the prior date.
        dt = (t - pd.Timedelta(hours=STEP_H)).normalize()
        om_row = om_idx.get((dkey, dt))
        if om_row is not None:
            row = om_frame.iloc[om_row]
            for out_col, om_col in OM_COL_MAP.items():
                driver_cols[out_col][k] = float(row[om_col])
            driver_cols["rain"][k] = float(row["rain_mm"]) / 4.0   # uniform over 4 slots
            rec["weather"], rec["rain"] = "open-meteo", "open-meteo"
        else:
            cw = clim.weather(station, int(t.dayofyear)) if clim.weather_present() else None
            if cw is not None:
                for c in WEATHER_COLS:
                    driver_cols[c][k] = float(cw[c])
                rec["weather"] = "climatology"
            rv = clim.rain(dkey, int(t.dayofyear), int(t.hour))
            if rv is not None and np.isfinite(rv):
                driver_cols["rain"][k] = float(rv)
                rec["rain"] = "climatology"
        rv, cv, river_src, canal_src = _rc_values(cw, rfi, dkey, t, river_last, canal_last)
        driver_cols["river_level"][k] = rv
        driver_cols["canal_level"][k] = cv
        rec["river_canal"] = f"{river_src}|{canal_src}"
        srcs.append(rec)

    return (pd.DataFrame(driver_cols), srcs)


def build_step_features(*, station: str, district: str, anchor_t: pd.Timestamp,
                        hist: pd.DataFrame, fnames: list[str],
                        om_frame: pd.DataFrame | None = None,
                        climatology: pd.DataFrame | None = None,
                        rain_clim: pd.DataFrame | None = None,
                        river_forecast: pd.DataFrame | None = None,
                        forecast_days: int = int(FORECAST_STEPS / 4),
                        steps: int = DEFAULT_STEPS, mode: str = "future",
                        anchor_gwl: float | None = None,
                        max_h: int = DEFAULT_STEPS) -> tuple[np.ndarray, list[dict]]:
    """Build the (steps x len(fnames)) future-mode feature matrix + src records.

    ``hist`` is the station slice with columns [time, gwl, ...drivers] sorted by
    time and restricted to rows <= anchor_t; the last row is the anchor.
    Returns (X float32, src_records) with column order == fnames.
    """
    if mode != "future":
        raise ValueError(f"build_step_features only supports mode='future', got {mode!r}")
    base_cols = [c for c in fnames if c not in H_COLS]
    h_cols = [c for c in fnames if c in H_COLS]

    _lc = hist.sort_values("time")
    river_last = _last_obs(_lc, "river_level")
    canal_last = _last_obs(_lc, "canal_level")

    fw, srcs = future_weather_matrix(station=station, district=district,
                                     anchor_t=anchor_t, steps=steps,
                                     om_frame=om_frame, climatology=climatology,
                                     rain_clim=rain_clim, river_forecast=river_forecast,
                                     river_last=river_last, canal_last=canal_last,
                                     forecast_days=forecast_days)

    hs = np.arange(1, steps + 1, dtype=np.int64)
    ts = anchor_t + pd.to_timedelta(hs * STEP_H, unit="h")

    tail_rain = _sixh_tail(hist, "rain", anchor_t)
    tail_temp = _sixh_tail(hist, "temp", anchor_t)
    tail_hum = _sixh_tail(hist, "humidity", anchor_t)
    i0 = len(tail_rain) - 1   # anchor position inside the tail array
    # observed tail INCLUDING the anchor row, then the future forcing values
    rain_ser = np.concatenate([tail_rain[: i0 + 1], fw["rain"].to_numpy()])
    temp_ser = np.concatenate([tail_temp[: i0 + 1], fw["temp"].to_numpy()])
    hum_ser = np.concatenate([tail_hum[: i0 + 1], fw["humidity"].to_numpy()])

    # GWL memory pinned at the anchor row (hist may hold NaNs before fill).
    g = hist["gwl"].to_numpy(dtype=np.float64)
    anchor_gwl = float(g[-1]) if anchor_gwl is None else float(anchor_gwl)
    memory: dict = {"gwl": anchor_gwl}
    for c, lag in GWL_LAGS:
        memory[c] = float(g[-1 - lag]) if len(g) > lag else np.nan
    for c, win in GWL_MEAN_ROLLS:
        memory[c] = float(pd.Series(g).rolling(win, min_periods=1).mean().iloc[-1])
    for c, win in GWL_STD_ROLLS:
        memory[c] = float(pd.Series(g).rolling(win, min_periods=1).std().iloc[-1])

    rows = []
    for k in range(steps):
        t = ts[k]
        p = i0 + (k + 1)   # index inside concatenated series for step k+1
        row = dict(memory)
        # raw point drivers at step k (river/canal from forecast/clim/persistence)
        for c in ("temp", "humidity", "solar", "wind_speed", "pressure"):
            row[c] = float(fw[c].iloc[k])
        row["river_level"] = fw["river_level"].iloc[k]
        row["canal_level"] = fw["canal_level"].iloc[k]
        # rain window sums over [p-win+1 .. p]
        for out, win in RAIN_ROLL_PAIRS:
            lo = max(0, p - win + 1)
            row[out] = float(np.nansum(rain_ser[lo:p + 1])) if p >= lo else np.nan
        for out, src, win in ROLL_MEAN_PAIRS:
            ser = temp_ser if src == "temp" else hum_ser
            lo = max(0, p - win + 1)
            w = ser[lo:p + 1]
            row[out] = float(np.nanmean(w)) if np.isfinite(w).sum() > 0 else np.nan
        row.update(_doy_hour_features(t))
        row.update(_horizon_feats(hs[k], max_h=max_h))
        rows.append(row)

    df = pd.DataFrame(rows)
    X = df[base_cols + h_cols].to_numpy(dtype=np.float32)
    return X, srcs