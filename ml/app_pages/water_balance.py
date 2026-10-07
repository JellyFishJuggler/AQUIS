"""Water balance — recharge condition for a station.

Shows how much rain the well has received, how much is still held in the soil
column (via the antecedent precipitation index), and how the current 30-day window
compares with that well's own history.

What this deliberately does NOT show, because the data does not exist in this
repository: evapotranspiration, so there is no true rainfall-minus-ET0 balance;
and groundwater extraction, so there is no pumpage term. Those gaps are stated on
the page rather than papered over — they are the two things that would most
improve a real water-balance view here.

Page contract: module-level script (run directly by `st.navigation`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _charts as C
import _data as D
import _hydro_metrics as M


@st.cache_data(ttl="10m", show_spinner="Reading this station's record…")
def _frame(station: str) -> pd.DataFrame | None:
    return D.station_frame(station)


@st.cache_data(ttl="10m", show_spinner=False)
def _stations() -> pd.DataFrame:
    return D.station_universe()


@st.cache_data(ttl="10m", show_spinner=False)
def _rain_block(station: str) -> dict:
    df = _frame(station)
    return M.rain_block(df) if df is not None else {}


@st.cache_data(ttl="10m", show_spinner=False)
def _api_curves(station: str, taus: tuple[int, ...]) -> pd.DataFrame:
    df = _frame(station)
    if df is None or "rain" not in df.columns:
        return pd.DataFrame()
    r = pd.to_numeric(df["rain"], errors="coerce")
    x = np.where(np.isfinite(r.to_numpy(float)), r.to_numpy(float), 0.0)
    out = {}
    for tau in taus:
        lam = float(np.exp(-(M.STEP_H / 24.0) / tau))
        y = np.empty_like(x)
        acc = 0.0
        for i, v in enumerate(x):
            acc = lam * acc + v
            y[i] = acc
        out[f"API {tau} d"] = y
    idx = pd.to_datetime(df["time"])
    frame = pd.DataFrame(out, index=idx)
    return frame[frame.index >= idx.max() - pd.Timedelta(days=730)]


def _tile(v, unit="", fmt="{:.1f}") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return fmt.format(v) + unit


st.title("Water balance")
st.caption("Recharge condition for one well, from the rainfall record.")

uni = _stations()
if uni.empty:
    st.error("**Data unavailable.** `data/aligned/table_6h.parquet` not found.")
    st.stop()

names = sorted(uni["Station"].astype(str))
default = 0
if st.session_state.get("focus_station") in names:
    default = names.index(st.session_state["focus_station"])
station = st.selectbox("Station", names, index=default)
st.session_state["focus_station"] = station

df = _frame(station)
rain_present = (df is not None and "rain" in df.columns
                and pd.to_numeric(df["rain"], errors="coerce").notna().any())
if not rain_present:
    st.warning(
        f"**No rainfall record for {station}.** Rainfall reaches a well through the gauge "
        "association built in `04_align`; a well with no gauge inside the search radius "
        "carries no rain column at all rather than a filled one."
    )
    st.stop()

rb = _rain_block(station)
if not rb:
    st.warning("Rainfall column present but too thin to summarise.")
    st.stop()

# ---- accumulation ----------------------------------------------------------
st.subheader("Rainfall accumulation")
k = st.columns(5)
k[0].metric("Last 24 h", _tile(rb.get("rain_24h_mm"), " mm"))
k[1].metric("Last 7 d", _tile(rb.get("rain_7d_mm"), " mm"))
k[2].metric("Last 30 d", _tile(rb.get("rain_30d_mm"), " mm"))
k[3].metric("Last 90 d", _tile(rb.get("rain_90d_mm"), " mm"))
k[4].metric("30 d vs own median", _tile(rb.get("rain_30d_vs_median"), " mm", "{:+.1f}"))

obs_cols = [c for c in rb if c.endswith("_obs_frac")]
if obs_cols:
    st.caption(
        "Observed-rain coverage per window: "
        + " · ".join(f"{c.replace('rain_', '').replace('_obs_frac', '')} "
                     f"{100 * rb[c]:.0f}%" for c in obs_cols)
        + ". A window below ~80% is partly unobserved — read the total as a floor."
    )

if "rain_30d_deficit_pct" in rb:
    pct = rb["rain_30d_deficit_pct"]
    label = ("very dry" if pct < 10 else "dry" if pct < 25
             else "below normal" if pct < 45 else "normal" if pct <= 55
             else "above normal" if pct <= 75 else "wet")
    st.metric("30-day rainfall percentile vs this station's history",
              f"{pct:.0f}th — {label}",
              help="Median 30-day total for this window historically: "
                   f"{rb.get('rain_30d_median_mm', float('nan')):.0f} mm")

# ---- antecedent precipitation index ----------------------------------------
st.divider()
st.subheader("Antecedent precipitation index")
curves = _api_curves(station, (7, 22, 60))
if not curves.empty:
    st.plotly_chart(
        C.series(curves, height=340, y_title="API (mm)",
                 title="Antecedent precipitation index, 2 years"),
        width="stretch", config={"displayModeBar": False},
    )
    a = st.columns(3)
    a[0].metric("API τ=7 d", _tile(rb.get("api_7d_mm"), " mm"))
    a[1].metric("API τ=22 d", _tile(rb.get("api_22d_mm"), " mm"))
    a[2].metric("API τ=60 d", _tile(rb.get("api_60d_mm"), " mm"))
    st.caption(
        "API is an exponentially decayed rainfall memory: each 6 h slot retains "
        "λ = exp(−(6/24)/τ) of the previous accumulation. τ = 22 d is the recharge lag "
        "measured in this corpus (`outputs/lag_curves.csv`). It proxies water still "
        "available to percolate — it is not a soil-moisture measurement."
    )

# ---- rainfall against level ------------------------------------------------
st.divider()
st.subheader("Rainfall against level")
obs = M.observed_mask(df)
daily_rain = (df.assign(date=df["time"].dt.normalize())
                .groupby("date")["rain"].sum())
daily_gwl = M._daily(df.loc[obs, ["time", "gwl"]], "gwl", "median")
both = pd.DataFrame({"rain_30d_mm": daily_rain.rolling(30).sum(),
                     "gwl": daily_gwl}).dropna()
if len(both) > 60:
    both = both[both.index >= both.index.max() - pd.Timedelta(days=365 * 5)]
    st.plotly_chart(
        C.rain_and_level(both["rain_30d_mm"], both["gwl"], height=380),
        width="stretch", config={"displayModeBar": False},
    )
    st.caption(
        "30-day rainfall (mm, bars) and observed level (m, line, right axis). The "
        "recharge response is lagged and damped — expect the level to turn weeks after "
        "the rain, not with it. The level axis is scaled to the data, not to zero."
    )

# ---- honest gaps -----------------------------------------------------------
st.divider()
st.warning(
    "**This is a rainfall-balance view, not a true water balance.** Two terms are "
    "missing from the data in this repository, and both matter:\n\n"
    "* **Evapotranspiration (ET₀)** — never computed anywhere in the ML pipeline, so "
    "rainfall-minus-ET₀ (effective recharge) cannot be shown. Open-Meteo already "
    "retrieves the radiation, temperature, humidity and wind that FAO-56 needs, but the "
    "fetch aggregates hourly to daily and keeps only a 365-day window.\n"
    "* **Groundwater extraction** — no pumpage dataset exists "
    "(`data/soil/cgwb_extraction_district.csv` is absent). Pumping is the largest "
    "unobserved term in any well's balance.\n\n"
    "Until those exist, treat this page as *recharge condition*, not a closed balance."
)
