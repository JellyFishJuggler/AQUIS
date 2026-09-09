"""Fleet page — whole-fleet 30-day forecast scan (decline / recovery).

Reads the snapshot built by `11_fleet.py` (outputs/fleet_forecast.csv).
"""

import altair as alt
import pandas as pd
import streamlit as st

from _model import load_fleet_table, load_fleet_district

DECLINE_M = 0.3

st.set_page_config(page_title="AQUIS — fleet", page_icon=":material/query_stats:", layout="wide")
st.title("Fleet — 30-day outlook across all monitored stations")
st.caption("Every data-rich station (≥2 y history, ≥2000 readings) is scored with the same "
           "pooled model used on the Forecast page. The headline change is the **quantile "
           "median (q50)** estimate, so a degenerate pooled-regression output on stations "
           "with missing recent drivers can't fake a decline.")

df = load_fleet_table()
dtab = load_fleet_district()
if df.empty:
    st.warning("No fleet snapshot found — run `ml/11_fleet.py` first.")
    st.stop()

good = df[df["category"] != "unreliable"].copy()
good["significant"] = good["band_half_m"].notna() & (good["change_30d_m"].abs() > good["band_half_m"])

a, b, c, d, e, f = st.columns(6)
a.metric("Stations scored", f"{len(good):,}")
b.metric("Districts", f"{good['district'].nunique()}")
c.metric("Median 30-d change", f"{good['change_30d_m'].median():+.3f} m")
d.metric("Declining (≤−0.3 m)", f"{int((good['change_30d_m'] <= -DECLINE_M).sum())}")
e.metric("Recovering (≥+0.3 m)", f"{int((good['change_30d_m'] >= DECLINE_M).sum())}")
f.metric("Quality-gated out", f"{int((df['category'] == 'unreliable').sum())}")
st.caption("Unreliable = missing/stale anchor GWL or forecast outside the observed range. "
           "Nearly all 30-day changes fall inside their calibrated 90% interval "
           "(band half-width), so bucket counts above are point-estimate flags, not "
           "statistically significant moves.")

st.subheader("Significant movers (|Δ| larger than the 90% band)")
movers = good[good["significant"]]
if movers.empty:
    st.info("No station moved further than its calibrated 90% interval half-width on "
            "this snapshot — every forecast lies inside interval noise. Refresh the scan "
            "when new data lands (`python 11_fleet.py --force`).")
else:
    st.warning(f"**{len(movers)} station(s)** with a 30-day change beyond their 90% band:")
    st.dataframe(movers.sort_values("change_30d_m").rename(columns={
        "station": "Station", "district": "District", "change_30d_m": "30-d Δ (m)",
        "band_half_m": "90% band ±", "category": "Category"}),
        use_container_width=True, hide_index=True)

st.subheader("Distribution of forecast 30-day change (q50 median)")
hist = alt.Chart(good).mark_bar(color="#4ecca3").encode(
    alt.X("change_30d_m:Q", bin=alt.Bin(maxbins=40), title="Forecast 30-d change (m)"),
    alt.Y("count():Q", title="stations"),
    tooltip=["count()"],
).properties(height=220)
rule = alt.Chart(pd.DataFrame({"x": [-DECLINE_M, DECLINE_M]})).mark_rule(color="#e2555e", strokeDash=[4, 4]).encode(
    x="x:Q")
st.altair_chart(hist + rule, use_container_width=True)
st.caption("Dashed lines: decline/recovery thresholds at ±0.3 m.")

st.subheader("District rollup")
if not dtab.empty:
    dtab["decline_share"] = (dtab["n_decline"] / dtab["n"]).round(3)
    st.dataframe(dtab.rename(columns={
        "district": "District", "n": "Stations", "n_decline": "Declining",
        "n_recover": "Recovering", "median_change_m": "Median Δ (m)",
        "decline_share": "Decline share",
    })[["District", "Stations", "Declining", "Recovering", "Median Δ (m)", "Decline share"]],
        use_container_width=True, hide_index=True)

st.subheader("Station-level scan")
c1, c2, c3 = st.columns(3)
cats = c1.multiselect(
    "Category", sorted(df["category"].dropna().unique()),
    default=[c for c in ["decline (high)", "decline", "recovering", "stable"]
             if c in sorted(df["category"].dropna().unique())],
    key="fleet_cat")
dist = c2.selectbox("District", ["All"] + sorted(good["district"].dropna().unique()), key="fleet_dist")
movers_only = c3.checkbox("Movers only (|Δ| ≥ 0.3 m)", key="fleet_movers")

tbl = df[df["category"].isin(cats)].copy()
if dist != "All":
    tbl = tbl[tbl["district"] == dist]
if movers_only:
    tbl = tbl[tbl["change_30d_m"].abs() >= DECLINE_M]
tbl = tbl.sort_values("change_30d_m")

show = tbl.rename(columns={
    "station": "Station", "district": "District", "anchor": "Anchor GWL (m)",
    "change_30d_m": "30-d Δ (m)", "change_pooled_m": "Δ pooled (m)",
    "band_half_m": "90% band ±", "category": "Category", "plausible": "Plausible",
    "age_days": "Age (d)",
})
extra = ["Δ pooled (m)"] if "change_pooled_m" in tbl else []
show["Δ > band"] = show["90% band ±"].notna() & (show["30-d Δ (m)"].abs() > show["90% band ±"])
sel_cols = ["Station", "District", "Anchor GWL (m)", "30-d Δ (m)", *extra,
            "90% band ±", "Δ > band", "Category", "Plausible", "Age (d)"]
st.dataframe(show[sel_cols], use_container_width=True, hide_index=True)
st.caption("`Δ > band` = |30-d Δ| larger than the calibrated 90% interval half-width "
           "(a genuine move rather than interval noise). `Δ pooled` = pooled regression "
           "delta (noisier).")

with st.expander("Method & caveats"):
    st.markdown(
        "- **Path:** `ml/11_fleet.py` — per station, rebuild the feature frame "
        "(`06_features.build_full`) from the aligned 6-hourly table, then score with the "
        "pooled XGBoost + calibrated quantile q05/q50/q95 model (same forward path as the "
        "Forecast page).\n"
        "- **Headline change** = q50 level − anchor; the pooled delta is kept as secondary "
        "because it saturates to identical extremes on stations whose recent lag/driver "
        "columns are empty (seen as identical Δ values across distant stations) — the q50 "
        "median stays flat/stable there.\n"
        "- **Quality gate:** stations with a NaN anchor (latest reading spike-flagged), a "
        "stale last record (>45 d), or a forecast outside the observed range ±25 m are "
        "marked `unreliable` and excluded from counts.\n"
        "- **Signature of a real move:** |30-d Δ| comfortably larger than the 90% band "
        "half-width. Almost no station clears that today; remind the user these are near-"
        "flat, interval-noise-level changes.\n"
        f"- Thresholds: decline ≤−0.3 m, recovery ≥+0.3 m per 30 days (high decline ≤−0.6 m).\n"
        "- Refresh: rerun `python 11_fleet.py --force` (≈4 min).")