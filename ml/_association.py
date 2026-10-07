"""_association — how every external parameter is connected to a well.

Replaces the ad-hoc joins in `04_align` / `04b_align_6h` with one declarative
spec per parameter, shared by the batch alignment and (to be adopted by) the
serving path so training and inference cannot drift.

Why this exists
---------------
The previous design had three faults:

1. **A hard radius cliff.** `04_align.nearest()` did
   ``near = d <= radius_km; if not near.any(): continue`` — a well 151 km from
   the only gauge got *nothing*, indistinguishable from a well with no driver
   data at all. Absence was invisible.

2. **Only rainfall was blended.** Weather used the single nearest gauge at full
   weight, so a well 140 km away was treated the same as one 5 km away, and
   distance was never surfaced to the model.

3. **River and canal were joined in the wrong space.** `04_align` took the
   nearest gauge's *absolute level*. Those gauges publish on their own local
   datum, so transplanting an absolute level between them is physically
   meaningless. `refresh/future_drivers.py` already does the right thing for the
   forecast path (borrow the *delta*, anchor to the station's own baseline); the
   batch path did not.

Design
------
* **k-nearest, weight-decayed — no cliff.** Always take the k nearest gauges and
  weight by ``exp(-(d/L)^2)``. A distant-only well gets a tiny weight instead of
  no value, and the weight sum is emitted as a coverage column so the model (and
  a hydrologist) can discount it.
* **Method per parameter, not per code path.** `idw` for areal fields (rainfall,
  weather, ET0); `weighted_delta` for datum-relative series (river, canal, and
  optionally neighbouring wells).
* **Coverage always.** `<param>_n`, `<param>_dmin_km`, `<param>_wsum` accompany
  every association.

All functions here are pure — no filesystem, no config imports, no pandas state —
so they can be unit-tested without the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0
D_MIN_KM = 0.05          # floor on distance, matching the old IDW guard


# ---------------------------------------------------------------------------
# per-parameter specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriverSpec:
    """How one external parameter attaches to a well.

    radius_km   outer search radius. Still used as a *search* bound, but a well
                with no gauge inside it now falls back to its k nearest rather
                than to nothing.
    k           number of gauges to blend (k=1 reproduces nearest-gauge).
    length_km   distance-decay scale L in exp(-(d/L)^2). Should reflect how
                spatially coherent the variable is: larger for smooth fields
                (pressure, temperature), smaller for patchy ones (rainfall).
    method      "idw"           -> weighted mean of the gauge values
                "weighted_delta" -> weighted mean of each gauge's *anomaly*
                                    against its own trailing baseline
    value_scale rough magnitude, used only for documentation/plots.
    """

    key: str
    radius_km: float
    k: int
    length_km: float
    method: str = "idw"
    value_scale: str = "auto"
    note: str = ""

    def weight(self, dist_km: np.ndarray) -> np.ndarray:
        """Gaussian distance decay, floored at D_MIN_KM so a co-located gauge
        can't produce an infinite weight."""
        d = np.maximum(np.asarray(dist_km, dtype=float), D_MIN_KM)
        return np.exp(-((d / float(self.length_km)) ** 2))


# Length scales are physical judgement, not fitted. They set how fast a gauge's
# information decays with distance; a value too large makes the association
# over-confident at long range, too small discards usable data.
SPECS: dict[str, DriverSpec] = {
    # --- areal fields: smooth, blend several gauges -------------------------
    "rainfall":     DriverSpec("rainfall", 150.0, 3, 50.0, "idw", "mm",
                               "Areal flux; recharge integrates over area, so blend."),
    "temperature":  DriverSpec("temperature", 200.0, 3, 120.0, "idw", "degC",
                               "Smooth field; a single distant gauge is a poor estimate."),
    "humidity":     DriverSpec("humidity", 200.0, 3, 120.0, "idw", "%", "Smooth field."),
    "solar":        DriverSpec("solar", 200.0, 3, 150.0, "idw", "W/m2", "Very smooth."),
    "wind_speed":   DriverSpec("wind_speed", 200.0, 3, 120.0, "idw", "km/h", "Moderately smooth."),
    "pressure":     DriverSpec("pressure", 250.0, 3, 200.0, "idw", "hPa",
                               "Synoptic scale — the smoothest driver available."),
    "eto":          DriverSpec("eto", 200.0, 3, 120.0, "idw", "mm/d",
                               "Reference ET is a smooth areal field; same logic as rainfall."),
    # --- datum-relative series: NEVER transplant absolute levels ------------
    "river_level":  DriverSpec("river_level", 150.0, 3, 100.0, "weighted_delta", "m",
                               "Gauges publish on their own datum — only the delta transfers."),
    "canal_level":  DriverSpec("canal_level", 150.0, 3, 100.0, "weighted_delta", "m",
                               "Regulated and datum-relative; delta only."),
    # --- listed for completeness; excluded from the numeric model -----------
    "wind_direction": DriverSpec("wind_direction", 200.0, 3, 120.0, "idw", "deg",
                                 "CIRCULAR — a plain mean is wrong. Excluded from the model; "
                                 "needs von Mises statistics before use."),
}

# Parameters that must not enter the numeric feature set as-is.
EXCLUDED_FROM_MODEL = {"wind_direction"}

# Parameters whose values are datum-relative by construction.
DATUM_RELATIVE = {k for k, s in SPECS.items() if s.method == "weighted_delta"}


def spec(key: str) -> DriverSpec:
    if key not in SPECS:
        raise KeyError(f"no association spec for {key!r}; known: {sorted(SPECS)}")
    return SPECS[key]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Vectorised over the *second* point."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = np.asarray(lat2) - lat1
    dlon = np.asarray(lon2) - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def k_nearest(anchor: pd.DataFrame, gauges: pd.DataFrame, key: str
              ) -> pd.DataFrame:
    """Gauge links for every anchor, k-nearest with NO radius cliff.

    Returns ``anchor_station, src_station, dist_km, weight, w_raw, rank,
    n_in_radius``. `weight` is the per-anchor normalised weight used for
    combining; `w_raw` is the kernel value ``exp(-(d/L)^2)`` before
    normalisation — that is the quantity whose sum measures how much
    information an anchor actually receives (the sum of `weight` is always 1
    by construction and carries no coverage signal).

    A well whose k nearest all sit beyond `radius_km` still gets links — with
    small weights — plus ``n_in_radius = 0`` so the shortfall is explicit rather
    than silent.
    """
    s = spec(key)
    rows: list[dict] = []
    g_lat = gauges["lat"].to_numpy(float)
    g_lon = gauges["lon"].to_numpy(float)
    g_sta = gauges["Station"].to_numpy()
    for _, a in anchor.iterrows():
        alat, alon = a.get("lat"), a.get("lon")
        if not (np.isfinite(alat) and np.isfinite(alon)):
            continue                                    # unmappable anchor
        d = haversine_km(alat, alon, g_lat, g_lon)
        finite = np.isfinite(d)
        if not finite.any():
            continue
        order = np.argsort(np.where(finite, d, np.inf))[: s.k]
        order = order[np.isfinite(d[order])]
        if order.size == 0:
            continue
        dd = d[order]
        w_raw = s.weight(dd)                # kernel value: the effective information
        wsum = float(w_raw.sum())
        w = w_raw / wsum if wsum > 0 else np.full(w_raw.shape, 1.0 / w_raw.shape[0])
        n_in = int((dd <= s.radius_km).sum())
        for rank, (ix, dist, wt, wr) in enumerate(
                zip(order, dd, w, w_raw), start=1):
            rows.append({
                "anchor_station": a["Station"],
                "src_station": str(g_sta[ix]),
                "dist_km": float(dist),
                "weight": float(wt),
                "w_raw": float(wr),
                "rank": rank,
                "in_radius": bool(dist <= s.radius_km),
                "n_in_radius": n_in,
                "k": int(s.k),
            })
    cols = ["anchor_station", "src_station", "dist_km", "weight", "w_raw",
            "rank", "in_radius", "n_in_radius", "k"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# combination
# ---------------------------------------------------------------------------
def combine_idw(links: pd.DataFrame, values: pd.DataFrame, value_col: str = "value"
                ) -> pd.DataFrame:
    """Weighted areal mean per (anchor, time).

    `values` is long: Station, time, value.
    """
    if links.empty or values.empty:
        return _empty_combined()
    m = links.merge(values, left_on="src_station", right_on="Station", how="inner")
    m = m.dropna(subset=["time", "value"])
    if m.empty:
        return _empty_combined()
    if "w_raw" not in m.columns:
        m = m.assign(w_raw=m["weight"])     # caller supplied pre-normalised links
    w = m["weight"].to_numpy(float)
    v = m[value_col].to_numpy(float)
    m = m.assign(_wv=w * v)
    g = m.groupby(["anchor_station", "time"], as_index=False).agg(
        value=("_wv", "sum"),
        w_avail=("weight", "sum"),
        wsum=("w_raw", "sum"),              # effective information, NOT ~1.0
        n=("src_station", "nunique"),
        dmin_km=("dist_km", "min"),
    )
    # renormalise where a slot is missing for some gauges
    g["value"] = g["value"] / g["w_avail"].replace(0.0, np.nan)
    return g.rename(columns={"anchor_station": "Station", "wsum": "wsum"})[
        ["Station", "time", "value", "n", "dmin_km", "wsum"]]


def gauge_anomaly(values: pd.DataFrame, window_steps: int) -> pd.DataFrame:
    """Each gauge's value minus its own **trailing** rolling mean.

    This is what makes a datum-relative series transferable: the *variation* is
    comparable between gauges even when the levels are not.

    Causality: the baseline is shifted one step, so at time *t* it covers
    ``[t-window, t-1]`` and never includes ``value[t]``. Without the shift the
    anomaly is partly self-referential — the same defect the correlation path had
    with its full-sample deseasonalisation. The first step of each gauge is
    therefore NaN, which is correct: no history, no anomaly.
    """
    if values.empty:
        return values.assign(value=np.nan)
    v = values.sort_values(["Station", "time"]).copy()
    base = (v.groupby("Station", sort=False)["value"]
              .transform(lambda s: s.rolling(window_steps, min_periods=1).mean()
                                   .shift(1)))
    v["value"] = v["value"] - base
    return v


def combine_weighted_delta(links: pd.DataFrame, values: pd.DataFrame,
                           window_steps: int = 120, value_col: str = "value"
                           ) -> pd.DataFrame:
    """Weighted mean of neighbour *anomalies* — never an absolute level.

    `window_steps` is the trailing baseline window in 6 h steps (120 = 30 d).
    The returned `value` is an anomaly in the parameter's own units; the caller
    must NOT add a level back, because the well's own datum already supplies the
    anchor.
    """
    if links.empty or values.empty:
        return _empty_combined()
    anom = gauge_anomaly(values.rename(columns={value_col: "value"}), window_steps)
    return combine_idw(links, anom, value_col="value")


def _empty_combined() -> pd.DataFrame:
    return pd.DataFrame(columns=["Station", "time", "value", "n", "dmin_km", "wsum"])


def combine(links: pd.DataFrame, values: pd.DataFrame, key: str,
            window_steps: int = 120) -> pd.DataFrame:
    """Dispatch on the parameter's declared method."""
    s = spec(key)
    if s.method == "weighted_delta":
        return combine_weighted_delta(links, values, window_steps=window_steps)
    return combine_idw(links, values)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def asinh_scale(x):
    """Compress a zero-inflated heavy tail while keeping small-value resolution.

    Used for rainfall: mostly 0, occasionally 100+ mm. `asinh` is ~x for small
    values and ~log(2x) for large, so dry-season structure survives — unlike
    log1p, which crushes exactly the range where recharge decisions live.
    """
    return np.arcsinh(np.asarray(x, dtype=float))


def zscore(x, mean: float, std: float):
    if not np.isfinite(std) or std <= 0:
        return np.asarray(x, dtype=float) - mean
    return (np.asarray(x, dtype=float) - mean) / std


def training_stats(series: pd.Series, cutoff: pd.Timestamp) -> dict:
    """Mean/std/prevalence from TRAINING-ERA data only.

    Every normalisation statistic in the pipeline must come from this function.
    Fitting a scaler on the full series is the same class of leak as the split
    target overlap, and far easier to do by accident.
    """
    s = series[series.index < cutoff] if isinstance(series.index, pd.DatetimeIndex) \
        else series
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "zero_frac": float("nan")}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "n": int(s.size),
        "zero_frac": float((s == 0).mean()),
    }


def normalise(series: pd.Series, stats: dict, method: str = "zscore") -> pd.Series:
    """Apply a training-era statistic. NaN stays NaN — XGBoost handles it natively."""
    if method == "asinh":
        return pd.Series(asinh_scale(series.to_numpy()), index=series.index)
    if method == "zscore":
        return pd.Series(zscore(series.to_numpy(), stats.get("mean", 0.0),
                                stats.get("std", 1.0)), index=series.index)
    raise ValueError(f"unknown normalisation {method!r}")


# Per-parameter normalisation, applied AFTER association.
NORMALISATION: dict[str, str] = {
    "rainfall": "asinh",
    "temperature": "zscore",
    "humidity": "zscore",
    "solar": "zscore",
    "wind_speed": "zscore",
    "pressure": "zscore",
    "eto": "asinh",
    # already anomalies by construction; scale only
    "river_level": "zscore",
    "canal_level": "zscore",
}


def normalisation_for(key: str) -> str:
    return NORMALISATION.get(key, "zscore")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def coverage_summary(links: pd.DataFrame, n_anchors: int, key: str) -> dict:
    """How well this parameter is actually connected — for logs and QC.

    `median_wsum` is the median over anchors of the **raw** kernel sum
    ``Σ exp(-(d/L)^2)`` — the effective information an anchor receives. It is
    deliberately NOT the sum of the normalised `weight` column, which is 1.0
    for every anchor by construction and would report "full coverage" for a
    well whose nearest gauge is 300 km away.
    """
    s = spec(key)
    if links.empty:
        return {"parameter": key, "anchors": n_anchors, "linked": 0,
                "linked_pct": 0.0, "median_wsum": 0.0, "median_dmin_km": None,
                "anchors_with_gauge_in_radius": 0, "method": s.method}
    per = links[links["rank"] == 1]
    raw = links.assign(_raw=s.weight(links["dist_km"].to_numpy()))
    wsum_per_anchor = raw.groupby("anchor_station")["_raw"].sum()
    return {
        "parameter": key,
        "anchors": int(n_anchors),
        "linked": int(per["anchor_station"].nunique()),
        "linked_pct": round(100.0 * per["anchor_station"].nunique()
                            / max(1, n_anchors), 1),
        "median_wsum": round(float(wsum_per_anchor.median()), 4),
        "median_dmin_km": round(float(per["dist_km"].median()), 1),
        "anchors_with_gauge_in_radius": int(
            (per["n_in_radius"] > 0).sum()),
        "method": s.method,
    }
