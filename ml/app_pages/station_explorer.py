import altair as alt
import pandas as pd
import streamlit as st

from _utils import load_table, nice

st.title("Station explorer")
st.caption("Browse per-station groundwater trends and driver coverage.")

table = load_table()

meta_cols = {"Station", "date", "District", "Tehsil"}
driver_cols = [c for c in table.columns
               if c not in meta_cols
               and not c.endswith("_n") and not c.endswith("_dist_km")]

with st.sidebar:
    district = st.selectbox("District", ["All"] + sorted(table["District"].dropna().unique()))
    driver = st.selectbox("Driver to inspect", driver_cols, index=driver_cols.index("rain_7d") if "rain_7d" in driver_cols else 0)

sub = table
if district != "All":
    sub = sub[sub["District"] == district]

# coverage: how many days per station for the chosen driver
rec = sub[sub[driver].notna()]
day_counts = rec.groupby("Station").size().sort_values(ascending=False)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Stations (district)", f"{sub['Station'].nunique()}", border=True)
with c2:
    st.metric("Rows", f"{len(sub):,}", border=True)
with c3:
    st.metric(f"Stations with {nice(driver)}", f"{rec['Station'].nunique()}", border=True)

if day_counts.empty:
    st.warning(f"No station in this selection has `{driver}` data.")
    st.stop()

st.subheader(f"Top stations by {nice(driver)} day coverage")
st.bar_chart(day_counts.head(15))

pick = day_counts.head(1).index[0]
st.subheader(f"Timeseries — {pick}")
g = table[table["Station"] == pick].dropna(subset=[driver]).copy()
g = g[g[driver].notna()]
if g.empty:
    st.caption("No driver data for top station")
    st.stop()

long = g.melt(id_vars="date", value_vars=["gwl", driver], var_name="series", value_name="value")
chart = (
    alt.Chart(long)
    .mark_line(point=False)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["gwl", driver], range=["#e0e0e0", "#4ecca3"])),
    )
    .properties(height=240)
    .interactive()
)
st.altair_chart(chart, width="stretch")

st.subheader("GWL trend (median surface)")
med = sub.groupby("date")["gwl"].median().dropna().reset_index()
st.line_chart(med.set_index("date")["gwl"])

st.subheader(f"Soil texture — {pick} (static, ISRIC)")

from _soil import load_soil
soil = load_soil()
sp = soil[soil["Station"] == pick]
if sp.empty:
    st.caption("No soil profile mapped for this station yet.")
else:
    r = sp.iloc[0]
    with st.container(horizontal=True):
        st.metric("Sand 0–30 cm", f"{r['sand_0_30']:.0f}%", border=True)
        st.metric("Silt 0–30 cm", f"{r['silt_0_30']:.0f}%", border=True)
        st.metric("Clay 0–30 cm", f"{r['clay_0_30']:.0f}%", border=True)
        st.metric("Bulk density 0–30 cm", f"{r['bdod_0_30']:.2f} g/cm³", border=True)
    tex = pd.DataFrame({
        "depth": ["0–30 cm", "0–30 cm", "0–30 cm", "60–100 cm", "60–100 cm", "60–100 cm"],
        "fraction": ["sand", "silt", "clay", "sand", "silt", "clay"],
        "pct": [r["sand_0_30"], r["silt_0_30"], r["clay_0_30"],
                r["sand_60_100"], r["silt_60_100"], r["clay_60_100"]],
    })
    tex_chart = alt.Chart(tex).mark_bar().encode(
        x=alt.X("fraction:N", title=None),
        y=alt.Y("pct:Q", title="%"),
        column=alt.Column("depth:N", title=None),
        color=alt.Color("fraction:N", legend=None),
        tooltip=["depth", "fraction", "pct"],
    ).properties(width=80)
    st.altair_chart(tex_chart)
    st.caption("SoilGrids v2 (ISRIC) — point query, 250 m resolution. Static feature, "
               "shown for context; gated out of the pooled model (redundant with station ordinals).")
