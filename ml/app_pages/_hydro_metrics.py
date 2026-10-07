"""_hydro_metrics — hydrologist-facing computations over the 6-hourly record.

Deliberately independent of the ML feature pipeline: everything here is a
descriptive statistic a hydrogeologist would compute by hand from a hydrograph.
No model, no training, no artifact writes. Where a quantity carries an
assumption (e.g. recession filtering) the assumption is a named constant so it can
be inspected and changed.

Units: GWL in metres on the station's own datum, rain in mm, time in 6-hour steps
(`STEP_H` hours). "Observed" means `gwl_obs == 1` when the aligned table carries
provenance; otherwise every present reading is treated as observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STEP_H = 6
STEPS_PER_DAY = 24 // STEP_H          # 4

# --- recession identification ------------------------------------------------
RAIN_EVENT_MM = 5.0        # a 6 h total above this starts a recharge episode
RECESSION_EXCLUDE_D = 10   # ignore this many days after a rain event
MIN_RECESSION_DAYS = 60    # below this, no recession constant is reported
TAU_MIN_D, TAU_MAX_D = 3.0, 400.0

# --- deficit / anomaly -------------------------------------------------------
DEFICIT_WINDOW_D = 30
CLIM_TAIL_DAYS = 365       # climatology is built from everything BEFORE this tail
MIN_CLIM_DAYS = 120

# --- quality -----------------------------------------------------------------
STALE_DAYS = 7
SPARSE_COVERAGE = 0.80


def _daily(df: pd.DataFrame, col: str = "gwl", how: str = "median") -> pd.Series:
    """Calendar-daily series (NaN where the day has no reading)."""
    d = pd.DataFrame({"date": pd.to_datetime(df["time"]).dt.normalize(),
                      "v": pd.to_numeric(df[col], errors="coerce")})
    g = d.groupby("date")["v"]
    return g.median() if how == "median" else g.sum()


def observed_mask(df: pd.DataFrame) -> pd.Series:
    """True where the GWL reading is real (not carried forward)."""
    present = pd.to_numeric(df["gwl"], errors="coerce").notna()
    if "gwl_obs" not in df.columns:
        return present
    return present & (pd.to_numeric(df["gwl_obs"], errors="coerce") == 1.0)


def station_summary(df: pd.DataFrame) -> dict:
    """Headline descriptive block for one station."""
    if df is None or df.empty:
        return {}
    t = pd.to_datetime(df["time"])
    gwl = pd.to_numeric(df["gwl"], errors="coerce")
    obs = observed_mask(df)
    g_obs = gwl[obs]
    latest_t = t[gwl.notna()].max()
    now = t.max()
    recency_d = float((now - latest_t).total_seconds()) / 86400.0 if pd.notna(latest_t) else np.nan
    span_d = float((t.max() - t.min()).total_seconds()) / 86400.0
    cov90 = float(obs[t >= t.max() - pd.Timedelta(days=90)].mean()) if len(t) else np.nan
    return {
        "station": str(df["Station"].iloc[0]) if "Station" in df.columns else "",
        "district": str(df["District"].iloc[0]) if "District" in df.columns else "",
        "first": t.min(), "last": t.max(),
        "span_years": round(span_d / 365.25, 2),
        "n_slots": int(len(df)),
        "n_observed": int(obs.sum()),
        "pct_observed": round(100 * float(obs.mean()), 1) if len(df) else np.nan,
        "latest_gwl": round(float(g_obs.iloc[-1]), 3) if len(g_obs) else np.nan,
        "latest_t": latest_t,
        "recency_days": round(recency_d, 1) if np.isfinite(recency_d) else np.nan,
        "coverage_90d": round(cov90, 3) if np.isfinite(cov90) else np.nan,
        "min_gwl": round(float(g_obs.min()), 3) if len(g_obs) else np.nan,
        "max_gwl": round(float(g_obs.max()), 3) if len(g_obs) else np.nan,
        "range_m": round(float(g_obs.max() - g_obs.min()), 3) if len(g_obs) else np.nan,
    }


# ---------------------------------------------------------------------------
# recession / aquifer response
# ---------------------------------------------------------------------------
def recession_analysis(df: pd.DataFrame, rain_col: str = "rain") -> dict:
    """Fit the linear-reservoir recession constant, hydrogeologist style.

    Method: take the daily MEDIAN observed level, restrict to falling days that
    are at least `RECESSION_EXCLUDE_D` days clear of a rain event above
    `RAIN_EVENT_MM`, then regress log(-dGWL/dt) on GWL. The slope is -1/tau, so
    `tau` is the time constant of a linear reservoir -- the single number that
    says how fast this well drains when nothing is recharging it.

    Returns {} when the record cannot support a fit, rather than a fabricated
    value.
    """
    out: dict = {}
    if df is None or df.empty:
        return out
    obs = observed_mask(df)
    d = df.loc[obs, ["time", "gwl"]].copy()
    if len(d) < 30:
        return out
    daily = _daily(d, "gwl", "median").dropna()
    if len(daily) < MIN_RECESSION_DAYS:
        return out
    out["recession_days_available"] = int(len(daily))
    out["daily_span_days"] = int((daily.index.max() - daily.index.min()).days)

    dd = pd.DataFrame({"gwl": daily})
    dd["dt"] = daily.index.to_series().diff().dt.days
    dd["dgwl"] = daily.diff()
    dd["rate"] = dd["dgwl"] / dd["dt"]          # m/day, negative = falling

    # exclude days near a rain event
    if rain_col in df.columns:
        rain_d = _daily(df, rain_col, "sum").reindex(daily.index)
        dd["rain"] = rain_d
        recent = (dd["rain"].fillna(0) > RAIN_EVENT_MM).rolling(
            RECESSION_EXCLUDE_D, min_periods=1).max().astype(bool)
        # also exclude the window AFTER an event
        after = recent.shift(1, fill_value=False).rolling(
            RECESSION_EXCLUDE_D, min_periods=1).max().astype(bool)
        dd["wet"] = recent | after
    else:
        dd["wet"] = False

    mask = (dd["rate"] < 0) & dd["rate"].notna() & np.isfinite(dd["gwl"]) & (~dd["wet"])
    n = int(mask.sum())
    out["recession_days_used"] = n
    if n < MIN_RECESSION_DAYS:
        out["tau_days"] = np.nan
        out["note"] = (f"only {n} clean recession days (need {MIN_RECESSION_DAYS})")
        return out

    x = dd.loc[mask, "gwl"].to_numpy(float)
    y = np.log(-dd.loc[mask, "rate"].to_numpy(float))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < MIN_RECESSION_DAYS or np.ptp(x) <= 1e-9:
        out["tau_days"] = np.nan
        out["note"] = "recession levels give no usable spread"
        return out
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or np.nan
    tau = -1.0 / b if b < 0 else np.nan
    out.update({
        "tau_days": round(float(tau), 2) if np.isfinite(tau) and TAU_MIN_D <= tau <= TAU_MAX_D else np.nan,
        "tau_fit_r2": round(1 - ss_res / ss_tot, 3) if np.isfinite(ss_tot) else np.nan,
        "recession_rate_median_m_per_d": round(float(dd.loc[mask, "rate"].median()), 4),
    })
    if not (np.isfinite(out["tau_days"])):
        out["note"] = (f"slope {b:.4g} gave tau outside [{TAU_MIN_D}, {TAU_MAX_D}] d")
    return out


def recession_episodes(df: pd.DataFrame, rain_col: str = "rain",
                       max_days: int = 365) -> pd.DataFrame:
    """Table of recent falling limbs: start, duration, drop, mean rate."""
    obs = observed_mask(df)
    d = df.loc[obs, ["time", "gwl"]].copy()
    if d.empty:
        return pd.DataFrame()
    daily = _daily(d, "gwl", "median").dropna()
    if len(daily) < 3:
        return pd.DataFrame()
    daily = daily[daily.index >= daily.index.max() - pd.Timedelta(days=max_days)]
    falling = daily.diff() < 0
    eps, cur = [], None
    for ts, is_fall in falling.items():
        if is_fall and cur is None:
            cur = {"start": ts, "start_gwl": float(daily.loc[:ts].iloc[-2])}
        elif not is_fall and cur is not None:
            cur.update(end=ts, end_gwl=float(daily.loc[:ts].iloc[-1]))
            eps.append(cur)
            cur = None
    if cur is not None:
        cur.update(end=daily.index[-1], end_gwl=float(daily.iloc[-1]))
        eps.append(cur)
    rows = []
    for e in eps:
        days = (e["end"] - e["start"]).days
        if days < 2:
            continue
        drop = e["start_gwl"] - e["end_gwl"]
        rows.append({"start": e["start"].date(), "end": e["end"].date(),
                     "days": days, "drop_m": round(drop, 3),
                     "rate_m_per_d": round(drop / days, 4)})
    return pd.DataFrame(rows).sort_values("drop_m", ascending=False) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# rainfall / recharge condition
# ---------------------------------------------------------------------------
def rain_block(df: pd.DataFrame, rain_col: str = "rain") -> dict:
    """Recent rainfall totals, antecedent accumulation and a deficit percentile."""
    out: dict = {}
    if df is None or df.empty or rain_col not in df.columns:
        return out
    r = pd.to_numeric(df[rain_col], errors="coerce")
    t = pd.to_datetime(df["time"])
    n = len(df)

    def _sum(steps: int) -> float:
        s = r.iloc[max(0, n - steps):]
        return float(s.sum(skipna=True)) if s.notna().any() else np.nan

    def _obs_frac(steps: int) -> float:
        s = r.iloc[max(0, n - steps):]
        return float(s.notna().mean()) if len(s) else np.nan

    for label, steps in (("24h", STEPS_PER_DAY), ("7d", 7 * STEPS_PER_DAY),
                         ("30d", 30 * STEPS_PER_DAY), ("90d", 90 * STEPS_PER_DAY)):
        out[f"rain_{label}_mm"] = round(_sum(steps), 1)
        out[f"rain_{label}_obs_frac"] = round(_obs_frac(steps), 3)

    # API with the corpus-measured tau and a couple of neighbours
    for tau in (7, 22, 60):
        lam = float(np.exp(-(STEP_H / 24.0) / tau))
        x = np.where(np.isfinite(r.to_numpy(float)), r.to_numpy(float), 0.0)
        y = np.zeros(len(x))
        acc = 0.0
        for i, v in enumerate(x):
            acc = lam * acc + v
            y[i] = acc
        out[f"api_{tau}d_mm"] = round(float(y[-1]), 1)

    # deficit: last 30 daily sums vs the same 30-day window in previous years
    daily = _daily(df, rain_col, "sum")
    if len(daily) >= MIN_CLIM_DAYS + DEFICIT_WINDOW_D:
        cutoff = daily.index.max() - pd.Timedelta(days=CLIM_TAIL_DAYS)
        clim = daily[daily.index < cutoff]
        recent = daily[daily.index > cutoff]
        if len(clim) >= MIN_CLIM_DAYS and len(recent) >= DEFICIT_WINDOW_D:
            roll_recent = recent.rolling(DEFICIT_WINDOW_D).sum().iloc[-1]
            roll_clim = clim.rolling(DEFICIT_WINDOW_D).sum().dropna()
            if np.isfinite(roll_recent) and len(roll_clim) > 30:
                out["rain_30d_deficit_pct"] = round(
                    100.0 * float((roll_clim < roll_recent).mean()), 1)
                out["rain_30d_median_mm"] = round(float(roll_clim.median()), 1)
                out["rain_30d_vs_median"] = round(
                    float(roll_recent) - float(roll_clim.median()), 1)
    return out


def gwl_anomaly(df: pd.DataFrame) -> dict:
    """Where the current level sits against this station's own history.

    Uses the observed daily median. The percentile is against the full record,
    which is a statement about this well only -- levels are station-datum
    referenced and are NOT comparable between wells.
    """
    obs = observed_mask(df)
    daily = _daily(df.loc[obs, ["time", "gwl"]], "gwl", "median").dropna()
    if len(daily) < 60:
        return {}
    idx = daily.index
    cutoff = idx.max() - pd.Timedelta(days=CLIM_TAIL_DAYS)
    hist = daily[idx < cutoff]
    ref = hist if len(hist) >= MIN_CLIM_DAYS else daily
    latest = float(daily.iloc[-1])
    pct = 100.0 * float((ref < latest).mean())
    doy = int(idx.max().dayofyear)
    same_season = ref[np.abs(ref.index.dayofyear - doy) <= 15]
    return {
        "latest_gwl": round(latest, 3),
        "pct_of_history": round(pct, 1),
        "seasonal_median": round(float(same_season.median()), 3) if len(same_season) >= 10 else np.nan,
        "seasonal_vs_now": round(latest - float(same_season.median()), 3)
        if len(same_season) >= 10 else np.nan,
        "history_days": int(len(ref)),
    }


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------
def quality_block(df: pd.DataFrame) -> dict:
    """Completeness, staleness and provenance for one station."""
    if df is None or df.empty:
        return {}
    obs = observed_mask(df)
    t = pd.to_datetime(df["time"])
    q = {
        "slots": int(len(df)),
        "observed_frac": round(float(obs.mean()), 3),
        "carried_frac": round(float(pd.to_numeric(df["gwl"], errors="coerce").notna().mean()
                                   - obs.mean()), 3) if "gwl_obs" in df.columns else 0.0,
    }
    last_obs = t[obs].max()
    q["last_observed"] = last_obs
    q["stale_days"] = round(float((t.max() - last_obs).total_seconds()) / 86400.0, 1) \
        if pd.notna(last_obs) else np.nan
    q["has_provenance"] = "gwl_obs" in df.columns
    # monthly completeness, for a coverage strip
    m = pd.DataFrame({"m": t.dt.to_period("M"), "obs": obs})
    q["_monthly"] = m.groupby("m")["obs"].mean()
    q["verdict"] = (
        "stale" if q.get("stale_days", 0) > STALE_DAYS
        else "sparse" if q["observed_frac"] < SPARSE_COVERAGE
        else "ok"
    )
    return q


def score_station(df: pd.DataFrame) -> dict:
    """One-row fleet summary used by the monitoring screen."""
    s = station_summary(df)
    q = quality_block(df)
    g = gwl_anomaly(df)
    r = rain_block(df)
    dry = None
    if "rain_30d_obs_frac" in r and pd.notna(r.get("rain_30d_mm")):
        dry = r["rain_30d_mm"]
    return {
        "station": s.get("station"), "district": s.get("district"),
        "latest_gwl_m": s.get("latest_gwl"),
        "recency_days": s.get("recency_days"),
        "coverage": q.get("observed_frac"),
        "verdict": q.get("verdict"),
        "pct_of_history": g.get("pct_of_history"),
        "seasonal_vs_now_m": g.get("seasonal_vs_now"),
        "rain_30d_mm": dry,
        "rain_30d_deficit_pct": r.get("rain_30d_deficit_pct"),
        "api_22d_mm": r.get("api_22d_mm"),
    }
