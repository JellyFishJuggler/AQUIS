"""Station — the hydrograph workspace. One well, read the way a hydrogeologist does.

Sections: where the level sits now, the observed hydrograph, this well's recession
behaviour, and optionally the frozen model's outlook overlaid. Descriptive first,
model second.

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


@st.cache_data(ttl="10m", show_spinner="Fitting the recession…")
def _recession(station: str) -> dict:
    df = _frame(station)
    return M.recession_analysis(df) if df is not None else {}


@st.cache_data(ttl="10m", show_spinner=False)
def _episodes(station: str, days: int) -> pd.DataFrame:
    df = _frame(station)
    return M.recession_episodes(df, max_days=days) if df is not None else pd.DataFrame()


@st.cache_data(ttl="5m", show_spinner="Scoring the frozen model (read-only)…")
def _forecast(station: str) -> dict:
    return D.trajectory_forecast_safe(station)


def _hydrograph(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Continuous level series over the trailing window, plus a carried overlay.

    Deliberately ONE continuous column. An earlier version split the series into
    mutually-exclusive `observed` and `carried` columns, which meant every row held
    a NaN in one of them — so each line was drawn with ~half its points missing and
    the chart looked empty. The carried overlay keeps the provenance visible
    without breaking the main trace.
    """
    x = df[df["time"] >= df["time"].max() - pd.Timedelta(days=days)].copy()
    x["gwl"] = pd.to_numeric(x["gwl"], errors="coerce")
    obs = M.observed_mask(x)
    out = pd.DataFrame({
        "level_m": x["gwl"].to_numpy(),
        "carried_forward_m": x["gwl"].where(~obs & x["gwl"].notna()).to_numpy(),
    }, index=pd.to_datetime(x["time"]))
    return out.sort_index()


def _hydrograph_with_forecast(df: pd.DataFrame, fc: dict) -> pd.DataFrame:
    """Short observed tail + the model's quantile paths on one shared axis."""
    pts = pd.DataFrame(fc["trajectory"])
    pts["time"] = pd.to_datetime(pts["time"])
    fut = pts.set_index("time")[["q05", "q50", "q95"]].rename(
        columns={"q05": "forecast q05", "q50": "forecast median", "q95": "forecast q95"})
    tail = _hydrograph(df, 90)
    tail = tail[tail.index <= fut.index.max()]
    out = tail.join(fut, how="outer").sort_index()
    return out.rename(columns={"level_m": "level (m)",
                               "carried_forward_m": "carried forward"})


st.title("Station")

uni = _stations()
if uni.empty:
    st.error("**Data unavailable.** `data/aligned/table_6h.parquet` not found.")
    st.stop()

# ---- station picker: district first so the list stays navigable ------------
districts = ["(all)"] + sorted(uni["District"].dropna().astype(str).unique())
c1, c2, c3, c4 = st.columns([1.2, 2, 1, 1.4])
with c1:
    dist = st.selectbox("District", districts, index=0)
with c2:
    pool = uni if dist == "(all)" else uni[uni["District"].astype(str) == dist]
    names = sorted(pool["Station"].astype(str))
    default = 0
    if st.session_state.get("focus_station") in names:
        default = names.index(st.session_state["focus_station"])
    station = st.selectbox("Station", names, index=default)
with c3:
    years = st.selectbox("Window", [1, 2, 3, 5], index=1)
with c4:
    show_carried = st.toggle("Mark carried-forward slots", value=False,
                             help="Overlay the slots filled by forward-fill across a "
                                  "telemetry gap. They are not measurements.")

st.session_state["focus_station"] = station

df = _frame(station)
if df is None or df.empty:
    st.warning(f"**No record for {station}.**")
    st.stop()

days = int(years * 365)
s = M.station_summary(df)

k = st.columns(6)
k[0].metric("Latest level", f"{s['latest_gwl']:.2f} m",
            help="On this station's own datum — not comparable between wells.")
k[1].metric("Last reading", f"{s['recency_days']:.0f} d ago")
k[2].metric("Data completeness", f"{100 * s['pct_observed']:.0f}%",
            help="Share of 6-hourly slots backed by a real reading.")
k[3].metric("Record span", f"{s['span_years']:.1f} y")
k[4].metric("Observed range", f"{s['range_m']:.2f} m")
k[5].metric("District", s["district"] or "—")

g = M.gwl_anomaly(df)
if g:
    a = st.columns(3)
    a[0].metric("Percentile of own history", f"{g['pct_of_history']:.0f}th")
    if np.isfinite(g.get("seasonal_median", np.nan)):
        a[1].metric("Seasonal median", f"{g['seasonal_median']:.2f} m")
        a[2].metric("vs seasonal median", f"{g['seasonal_vs_now']:+.2f} m")
    st.caption(
        f"Percentile is against this well's own record only ({g['history_days']:,} days). "
        "Levels are datum-referenced per station, so cross-well level comparisons are "
        "not meaningful."
    )

st.divider()

# ---- hydrograph ------------------------------------------------------------
st.subheader("Hydrograph")
hg = _hydrograph(df, days)
if hg.empty or hg["level_m"].notna().sum() == 0:
    st.info("No readings in this window.")
else:
    st.plotly_chart(
        C.hydrograph(hg, days=days, height=440, show_carried=show_carried),
        width="stretch", config={"displayModeBar": False},
    )
    n_obs = int(hg["level_m"].notna().sum())
    n_car = int(hg["carried_forward_m"].notna().sum())
    st.caption(
        f"{n_obs:,} readings over {days} days, y-axis scaled to the data"
        + (f" · {n_car:,} carried-forward slots marked in amber (telemetry gaps, not "
           "measurements)" if n_car and show_carried else "")
        + ". Use the 1m/6m/1y/all buttons to zoom the time axis."
    )
    if "gwl_obs" not in df.columns:
        st.caption("This aligned table carries no provenance column, so carried-forward "
                   "readings cannot be distinguished from measurements.")

# ---- optional model overlay ------------------------------------------------
ok, why = D.inference_available()
with st.expander("Model outlook (frozen trajectory model)", expanded=False):
    if not ok:
        st.info(f"Inference unavailable — {why}. Nothing is retrained to work around this.")
    elif st.toggle("Score the frozen model for this station", value=False):
        fc = _forecast(station)
        if fc.get("error"):
            st.warning(f"{fc.get('reason', 'error')}: {fc['error']}")
        else:
            st.plotly_chart(
                C.hydrograph_with_forecast(_hydrograph_with_forecast(df, fc), height=460),
                width="stretch", config={"displayModeBar": False},
            )
            t30 = fc.get("trajectory_30d", {}) or {}
            oc = fc.get("overall_confidence", {}) or {}
            m = st.columns(4)
            m[0].metric("Anchor", f"{fc['anchor_gwl']:.3f} m")
            m[1].metric("+30 d (median)", f"{t30.get('level', np.nan):.3f} m")
            m[2].metric("+30 d change", f"{t30.get('change', np.nan):+.3f} m")
            m[3].metric("Confidence", str(oc.get("level", "—")))
            st.caption(f"Reliability: {oc.get('reason', '')}")
            st.caption(
                "The band is the calibrated 90% interval. See **Model trust** for the "
                "measured coverage — it currently over-covers, so the band is conservative."
            )

# ---- recession behaviour ---------------------------------------------------
st.divider()
st.subheader("Recession behaviour")
rec = _recession(station)
if not rec or not np.isfinite(rec.get("tau_days", np.nan)):
    st.info("No recession constant could be fitted for this station."
            + (f" ({rec.get('note')})" if rec.get("note") else ""))
else:
    r = st.columns(4)
    r[0].metric("Recession constant τ", f"{rec['tau_days']:.0f} d",
                help="Time for the level to fall to 1/e of its excess, from clean falling "
                     "limbs clear of rain. Fitted on this station's own record only.")
    r[1].metric("Fit R²", f"{rec['tau_fit_r2']:.2f}")
    r[2].metric("Typical fall rate", f"{rec['recession_rate_median_m_per_d']:+.3f} m/d")
    r[3].metric("Recession days used", f"{rec['recession_days_used']:,}")
    st.caption(
        "Method: daily median observed level → falling days at least "
        f"{M.RECESSION_EXCLUDE_D} d clear of a rain event above {M.RAIN_EVENT_MM} mm → "
        "regression of log(−dGWL/dt) on GWL; τ = −1/slope. A large τ means a slow, "
        "storage-rich response; a small τ means the well drains quickly."
    )

eps = _episodes(station, days)
if not eps.empty:
    st.markdown("**Largest falling limbs in this window**")
    st.dataframe(eps.head(10), width="stretch", hide_index=True)
    st.caption("Candidate recession episodes, ranked by total drop.")
