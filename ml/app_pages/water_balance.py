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


@st.cache_data(ttl="10m", show_spinner=False)
def _district_climate(district: str) -> pd.DataFrame:
    """Daily Open-Meteo district-centroid rain and reference ET0, if fetched."""
    path = D.DATA / "cfs" / "openmeteo_weather_daily.parquet"
    weather = D.read_parquet_safe(path)
    required = {"District", "date", "rain_mm", "et0_mm"}
    if weather is None or not required.issubset(weather.columns):
        return pd.DataFrame()
    weather = weather[weather["District"].astype(str).str.casefold()
                      == str(district).casefold()].copy()
    if weather.empty:
        return pd.DataFrame()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
    for col in ("rain_mm", "et0_mm"):
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather = weather.dropna(subset=["date", "rain_mm", "et0_mm"])
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    weather = weather[weather["date"] < today_utc]
    weather = weather.sort_values("date").drop_duplicates("date", keep="last")
    weather["p_minus_et0_mm"] = weather["rain_mm"] - weather["et0_mm"]
    return weather.reset_index(drop=True)


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
        "association built in `pipeline_data`; a well with no gauge inside the search radius "
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

# ---- district climate demand proxy -----------------------------------------
st.divider()
st.subheader("District climate water-demand proxy")
district = str(uni.loc[uni["Station"].astype(str) == station, "District"].iloc[0])
climate = _district_climate(district)
if climate.empty:
    st.info(
        "Reference ET₀ has been added to the Open-Meteo fetch. The saved weather "
        "artifact does not contain `et0_mm` yet; run the ML weather refresh to populate it."
    )
else:
    latest = climate["date"].max()
    window = climate[climate["date"] >= latest - pd.Timedelta(days=29)]
    if len(window) < 24:
        st.info("Not enough complete district weather days to summarize a 30-day window.")
    else:
        rain30 = float(window["rain_mm"].sum())
        et030 = float(window["et0_mm"].sum())
        cols = st.columns(3)
        cols[0].metric("District rain · 30 d", f"{rain30:.1f} mm")
        cols[1].metric("Reference ET₀ · 30 d", f"{et030:.1f} mm")
        cols[2].metric("P − ET₀ · 30 d", f"{rain30 - et030:+.1f} mm")
        recent = climate[climate["date"] >= latest - pd.Timedelta(days=364)]
        st.line_chart(
            recent.set_index("date")[["rain_mm", "et0_mm", "p_minus_et0_mm"]],
            x_label="Date", y_label="Daily total (mm)",
            alt="District-centroid daily precipitation, reference evapotranspiration, and their difference",
        )
        st.caption(
            f"Open-Meteo modeled weather at the {district} district centroid; latest "
            f"complete day {latest:%Y-%m-%d}. P − ET₀ is a climatic surplus/deficit "
            "indicator, not effective rainfall, aquifer recharge, or a closed water balance."
        )

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
    "**This page is still not a closed groundwater water balance.** It can show "
    "rainfall condition and, when the weather refresh has populated it, district-level "
    "Open-Meteo reference ET₀ and P − ET₀. These are climate indicators only: ET₀ is "
    "not actual evapotranspiration, and P − ET₀ is not effective rainfall or aquifer "
    "recharge because runoff, infiltration, soil storage, and other fluxes are not "
    "modeled here.\n\n"
    "* **Groundwater extraction** — no pumpage dataset exists "
    "(`data/soil/cgwb_extraction_district.csv` is absent). Pumping is the largest "
    "unobserved term in any well's balance.\n\n"
    "Treat this page as *recharge condition and atmospheric demand context*, not a "
    "closed water balance."
)
