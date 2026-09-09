"""Forecast page — 2026 backtest curve + deterministic 30-day forward forecast.

Forward forecasts rebuild the feature frame from the 6-hourly aligned table for
the selected station (same code path as training via 06_features.build_full) and
predict the 30-day change GWL(t+120) − GWL(t); level = anchor + change. The
anchor (latest observed GWL) is shown as a KPI, never as a predicted value.
"""

import altair as alt
import pandas as pd
import streamlit as st

from _model import (load_predictions, load_models, load_feature_config,
                    load_model_metrics, load_quantile_calibration, forward_forecast)
from _utils import load_table_6h, station_recency

MODELS = {"xgboost": "XGBoost", "ridge": "Linear (Ridge)"}
PRED_COL = {"xgboost": "xgb", "ridge": "ridge"}
HORIZON = 30
Z = 1.96

st.set_page_config(page_title="AQUIS — forecast", page_icon=":material/troubleshoot:", layout="wide")
st.title("Forecast — 30-day groundwater outlook")

preds = load_predictions()
table6 = load_table_6h()
metrics = load_model_metrics()
qcal = load_quantile_calibration()
has_qband = "q05_lvl" in preds.columns

recency = station_recency()
stations = [s for s in recency.sort_values(ascending=False).index.astype(str)
            if s in set(preds["Station"].astype(str))]
stations.extend(sorted(set(preds["Station"].astype(str)) - set(stations)))
col1, col2 = st.columns(2)
with col1:
    station = st.selectbox("Station", stations, key="fc_station")
with col2:
    model_name = st.selectbox("Model", list(MODELS), format_func=lambda k: MODELS[k], key="fc_model")

full = load_models()
xgb = full["xgboost"]()
ridge = full["ridge"]()
cfg = load_feature_config()

rs = metrics[(metrics["model"] == model_name) & (metrics["horizon"] == HORIZON)]["resid_std"].iloc[0]
pred_col = PRED_COL[model_name]

p = preds[(preds["Station"] == station)].sort_values("date")
obs = p[p["gwl"].notna()]
act = p[p["target"].notna()]

g1, g2, g3, g4 = st.columns(4)
g1.metric("Latest observed GWL", f"{obs['gwl'].iloc[-1]:.2f} m" if not obs.empty else "—",
          help="Today's reading — the anchor for the 30-day outlook. Never modeled/predicted.")
g2.metric("2026 anchored reads", f"{p['target'].notna().sum():,}")
g3.metric("Model RMSE (2026 test)", f"{metrics[(metrics['model']==model_name)&(metrics['horizon']==HORIZON)]['rmse'].iloc[0]:.3f} m")
if model_name == "xgboost" and qcal:
    g4.metric("Outlook 90% interval ±", f"~{qcal.get('half_width_median_m', 0):.2f} m",
              help=f"calibrated median half-width (coverage {qcal.get('coverage_stride_raw_q', ''):.0%} "
                   f"on non-overlap windows); ±p90 {qcal.get('half_width_p90_m', '')} m")
else:
    g4.metric("Outlook 95% band ±", f"{Z*rs:.2f} m")

st.subheader(f"2026 backtest — {MODELS[model_name]}, 30-day outlook")
wide = p.melt(
    id_vars=["date", "gwl"],
    value_vars=["target", pred_col],
    var_name="series", value_name="gwl_m",
)
if has_qband and model_name == "xgboost":
    b = p[p["target"].notna()]
    band = pd.DataFrame({"date": b["date"], "lo": b["q05_lvl"], "hi": b["q95_lvl"]})
    band_help = "Calibrated q05/q95 interval (non-overlap coverage ≥80%)."
else:
    band = pd.DataFrame({
        "date": p[p["target"].notna()]["date"],
        "lo": p[p["target"].notna()][pred_col] - Z * rs,
        "hi": p[p["target"].notna()][pred_col] + Z * rs,
    })
    band_help = f"±{Z}× test residual std ({Z*rs:.2f} m, uncalibrated)."
lines = alt.Chart(wide).mark_line().encode(
    x=alt.X("date:T", title=None),
    y=alt.Y("gwl_m:Q", title="GWL (m)"),
    color=alt.Color("series:N", scale=alt.Scale(
        domain=["target", pred_col],
        range=["#9b9b9b", "#4ecca3"])),
    strokeDash=alt.condition(alt.datum.series == "target", alt.value([1]), alt.value([0])),
    tooltip=["date", "series", "gwl_m"],
)
band_chart = alt.Chart(band).mark_errorband(extent="ci", color="#4ecca3", opacity=0.08).encode(
    x="date:T", y="lo:Q", y2="hi:Q",
    tooltip=[alt.Tooltip("lo:Q", title=band_help)],
)
st.altair_chart(band_chart + lines, use_container_width=True, height=380)
st.caption(f"Shaded band = {band_help}")

st.subheader("Forward outlook (from latest data)")
fc = forward_forecast(station)
if not fc:
    st.warning("No feature frame for this station.")
else:
    anchor, date_from = fc["anchor"], fc["date_from"]
    pred_xgb, pred_ridge = fc["pred_xgb"], fc["pred_ridge"]
    bh = fc["band_half"]
    fwd_rows = [
        ["Anchor date", date_from.strftime("%Y-%m-%d %H:%M")],
        ["Outlook date (+30 d)", (date_from + pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M")],
        ["Anchor GWL (observed)", f"{anchor:.2f} m"],
        ["XGBoost outlook", f"{anchor + pred_xgb:.2f} m"],
        ["Ridge outlook", f"{anchor + pred_ridge:.2f} m"],
        ["Persistence", f"{anchor:.2f} m"],
        ["XGBoost 90% interval (q05–q95)",
         f"{fc['q05_level']:.2f} … {fc['q95_level']:.2f} m" if fc.get("q05_level") is not None else "—"],
    ]
    if bh is not None:
        fwd_rows.append(["90% interval ± (median)", f"{bh:.2f} m"])
    fwd = pd.DataFrame(fwd_rows, columns=["metric", "value"])
    st.dataframe(fwd, use_container_width=True, hide_index=True)
    sm = fc.get("station_stride_rmse")
    if sm is not None:
        ref = float(qcal.get("half_width_median_m", 1.0)) if qcal else 1.0
        st.caption(f"Station honest RMSE (non-overlap windows): **{sm:.2f} m** — its "
                   f"90% interval half-width is {fc['band_half'] or 0:.2f} m "
                   f"({sm / ref:.1f}× the fleet-median band) when Q-calibrated.")

    seg = pd.DataFrame({
        "date": [date_from, date_from + pd.Timedelta(days=30)],
        "XGBoost": [anchor, anchor + pred_xgb],
        "Ridge": [anchor, anchor + pred_ridge],
        "Persistence": [anchor, anchor],
    }).melt("date", var_name="model", value_name="gwl_m")
    seg_chart = alt.Chart(seg).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("gwl_m:Q", title="GWL (m)"),
        color=alt.Color("model:N"),
        tooltip=["model", "date", "gwl_m"],
    ).properties(height=260)
    st.altair_chart(seg_chart, use_container_width=True)
    st.caption("Outlook = latest observed GWL (anchor) carried forward + predicted 30-day change. "
               "Persistence assumes the reading holds unchanged for 30 days.")

with st.expander("Station static context (soil · LULC)", icon=":material/api:"):
    from _soil import load_soil
    from _lulc import load_lulc
    sp = load_soil()
    sr = sp[sp["Station"] == station]
    if sr.empty:
        st.caption("No soil profile mapped for this station.")
    else:
        r = sr.iloc[0]
        st.markdown(
            f"**Soil (ISRIC SoilGrids v2)** — sand {r['sand_0_30']:.0f}% / silt "
            f"{r['silt_0_30']:.0f}% / clay {r['clay_0_30']:.0f}% at 0–30 cm; "
            f"sand {r['sand_60_100']:.0f}% / clay {r['clay_60_100']:.0f}% at 60–100 cm; "
            f"BDOD {r['bdod_0_30']:.2f} g/cm³."
        )
    dist = None
    if "District" in p.columns:
        dist = p["District"].iloc[0] if "District" in p.columns else None
    if dist is None:
        dsel = table6[table6["Station"] == station]
        dist = dsel["District"].iloc[0] if "District" in dsel.columns and len(dsel) else None
    if dist:
        lu = load_lulc()
        lr = lu[lu["District"] == dist]
        if not lr.empty:
            r2 = lr.iloc[0]
            top = sorted(
                [(c[:-4].replace("_", " "), r2[c]) for c in lu.columns if c.endswith("_pct")],
                key=lambda x: x[1], reverse=True)
            st.markdown(f"**LULC ({dist})** — " + "; ".join(
                f"{k} {v:.0f}%" for k, v in top[:4]))
        else:
            st.markdown(f"**LULC ({dist})** — no district data fetched.")
    st.caption("Static layers are displayed for context only; the forecast uses the 30-day "
               "change model (soil/LULC gated).")