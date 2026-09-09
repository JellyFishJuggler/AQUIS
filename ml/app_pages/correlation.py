from pathlib import Path

import pandas as pd
import streamlit as st

from _utils import load_report, load_lag_curves, nice

st.title("Correlation report")
st.caption(
    "Per-station Spearman/Pearson aggregated as the median across stations. "
    "Three modes expose different artefacts: raw, deseasoned (DoY-removed) and "
    "first-differenced (day-to-day change)."
)

rep = load_report()

mode = st.segmented_control("Correlation mode", ["raw", "deseason", "diff"], default="raw")
if not mode:
    st.stop()

metric = st.segmented_control("Metric", ["spearman", "pearson"], default="spearman")
if not metric:
    st.stop()

sub = rep[(rep["mode"] == mode) & (rep["metric"] == metric)].copy()
sub = sub.dropna(subset=["corr"])
sub = sub.sort_values("corr", ascending=False)

col1, col2 = st.columns([2, 3])
with col1:
    with st.container(border=True):
        st.markdown(f"**{metric} · {mode}**")
        st.dataframe(
            sub[["driver", "corr", "stations_ok"]]
            .rename(columns={"driver": "Driver", "corr": "Correlation", "stations_ok": "Stations"}),
            hide_index=True,
        )
with col2:
    chart = sub.copy()
    chart["label"] = chart["driver"].map(nice)
    st.bar_chart(chart, x="label", y="corr", color="#4ecca3")

st.header("Recharge lag — GWL vs rainfall")
lag = load_lag_curves()
st.line_chart(lag, x="lag", y="pearson_med")
best = lag.iloc[lag["pearson_med"].abs().idxmax()]
st.success(f"Best rainfall recharge lag: **{int(best['lag'])} days** (median Pearson **{best['pearson_med']:+.3f}**)")

with st.expander("Full report table", icon=":material/table_chart:"):
    clean = rep.dropna(subset=["corr"]).sort_values(["driver", "metric", "mode"])
    st.dataframe(
        clean.rename(columns={
            "driver": "Driver", "metric": "Metric", "mode": "Mode",
            "corr": "Correlation", "stations_ok": "Stations"}),
        hide_index=True,
        height=420,
    )

st.header("Static features vs station groundwater")

from _soil import load_soil
from _utils import load_table

tbl = load_table()
if tbl.empty:
    st.caption("No aligned table — cannot summarise per-station GWL.")
else:
    summ = tbl.groupby("Station")["gwl"].agg(["mean", "std", "median"]).reset_index()
    soil = load_soil()
    if soil.empty:
        st.caption("Soil profiles not yet fetched.")
    else:
        m = summ.merge(soil[["Station", "District", "sand_0_30", "silt_0_30", "clay_0_30",
                             "bdod_0_30", "sand_60_100", "silt_60_100", "clay_60_100"]],
                       on="Station", how="inner")
        rows = []
        for col in ["sand_0_30", "silt_0_30", "clay_0_30", "bdod_0_30",
                    "sand_60_100", "silt_60_100", "clay_60_100"]:
            r = m[[col, "mean", "std"]].dropna()
            if len(r) >= 10:
                rows.append({
                    "feature": col, "n": len(r),
                    "corr_vs_mean_gwl": r[col].corr(r["mean"], method="spearman"),
                    "corr_vs_spread_gwl": r[col].corr(r["std"], method="spearman"),
                })
        soil_corr = pd.DataFrame(rows).sort_values("corr_vs_mean_gwl", key=abs, ascending=False)
        st.markdown("**Soil texture vs station-level GWL** (mean depth & variability, Spearman, n stations)")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(soil_corr.rename(columns={
                "feature": "Soil feature", "n": "Stations",
                "corr_vs_mean_gwl": "ρ vs mean GWL", "corr_vs_spread_gwl": "ρ vs GWL spread"}),
                hide_index=True, use_container_width=True)
        with c2:
            if not soil_corr.empty:
                top = soil_corr.reindex(soil_corr["corr_vs_mean_gwl"].abs().sort_values(
                    ascending=False).index).head(8)
                import altair as alt
                chart = top.copy()
                chart["label"] = chart["feature"]
                st.altair_chart(alt.Chart(chart).mark_bar(color="#4ecca3").encode(
                    x=alt.X("corr_vs_mean_gwl:Q", title="ρ vs mean GWL"),
                    y=alt.Y("label:N", title=None, sort="-x"),
                    tooltip=["feature", "corr_vs_mean_gwl", "corr_vs_spread_gwl", "n"],
                ).properties(height=240), use_container_width=True)
        st.caption("Static texture is mapped for " +
                   f"**{m.Station.nunique():,} stations**. Same 'exploratory' caveat as LULC — "
                   "it is not fed to the model.")

        lulc = pd.read_csv(Path(__file__).resolve().parent.parent / "outputs" / "correlation_lulc.csv") \
            if Path(__file__).resolve().parent.parent.joinpath("outputs", "correlation_lulc.csv").exists() \
            else pd.DataFrame()
        if not lulc.empty:
            lc = lulc.reindex(lulc["pearson_r"].abs().sort_values(ascending=False).index)
            st.markdown("**LULC class share vs district GWL** (Pearson, exploratory)")
            lc1, lc2 = st.columns([1, 2])
            with lc1:
                st.dataframe(lc.rename(columns={
                    "lulc_feature": "LULC class", "pearson_r": "r vs district GWL",
                    "districts": "Districts", "gwl_metric": "GWL metric"}),
                    hide_index=True, use_container_width=True)
            with lc2:
                topc = lc.head(8)
                st.altair_chart(alt.Chart(topc).mark_bar(color="#e67676").encode(
                    x=alt.X("pearson_r:Q", title="r vs district GWL"),
                    y=alt.Y("lulc_feature:N", title=None, sort="-x"),
                    tooltip=["lulc_feature", "pearson_r", "districts"],
                ).properties(height=240), use_container_width=True)
