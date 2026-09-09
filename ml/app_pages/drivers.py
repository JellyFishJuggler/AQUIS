import altair as alt
import pandas as pd
import streamlit as st

from _utils import load_table, nice

st.title("Drivers vs groundwater level")
st.caption("Pick a station to see the aligned daily drivers against its groundwater level.")

table = load_table()

recency = table.groupby("Station")["date"].max()
stations = list(recency.sort_values(ascending=False).index.astype(str))
stn = st.selectbox("Select GWL station", stations)
if not stn:
    st.stop()

g = table[table["Station"] == stn].copy()
st.metric("Station", stn, f"{g['District'].iloc[0]}" if len(g) else "")

from _soil import load_soil
sr = load_soil()
sr = sr[sr["Station"] == stn]
if not sr.empty:
    r = sr.iloc[0]
    st.caption(f"Soil: sand {r['sand_0_30']:.0f}% / silt {r['silt_0_30']:.0f}% / clay "
               f"{r['clay_0_30']:.0f}% at 0–30 cm · {r['sand_60_100']:.0f}% / "
               f"{r['clay_60_100']:.0f}% sand/clay at 60–100 cm (ISRIC, static).")

meta_cols = {"Station", "date", "District", "Tehsil"}
driver_cols = [c for c in g.columns
               if c not in meta_cols
               and not c.endswith("_n") and not c.endswith("_dist_km")]

drivers = st.multiselect(
    "Drivers to overlay",
    driver_cols,
    default=[c for c in ["rain_7d", "rain_30d", "temp", "river_level", "pressure"]
             if c in g.columns],
)

if drivers:
    for d in drivers:
        label = nice(d)
        sub = g[["date", d, "gwl"]].dropna()
        if sub.empty:
            st.caption(f"No overlapping data for `{d}` at this station.")
            continue
        # normalise both series for visual comparison (z-score)
        z = sub.copy()
        for col in (d, "gwl"):
            s = z[col]
            z[col] = (s - s.mean()) / s.std(ddof=0)
        z = z.melt(id_vars="date", var_name="series", value_name="normalised")
        z["series"] = z["series"].replace({"gwl": "GWL (z)", d: f"{label} (z)"})
        chart = (
            alt.Chart(z)
            .mark_line(point=False)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("normalised:Q", title="Normalised (z-score)"),
                color=alt.Color("series:N", scale=alt.Scale(
                    domain=["GWL (z)", f"{label} (z)"],
                    range=["#e0e0e0", "#4ecca3"])),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("normalised:Q"), "series:N"],
            )
            .properties(height=220)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
else:
    st.caption("Choose at least one driver to compare.")

with st.expander("Raw today layout", icon=":material/table_chart:"):
    st.dataframe(
        g[["date", "gwl"] + [d for d in drivers if d in g.columns]].tail(200),
        hide_index=True,
        height=360,
    )
