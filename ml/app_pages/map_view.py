"""Map — Uttar Pradesh station network, with a 2D plan and a 3D view.

Click a pin to select the station: `session_state["focus_station"]` is written,
and every other page reads it as its default, so the workflow is *click here →
open Station / Water balance / Quality*.

Three things this deliberately does:
  * frames Uttar Pradesh ONLY (viewport locked to the state's bounding box, dark
    basemap so neighbouring states read as background rather than content);
  * draws the state outline, from `_up_boundary.py`;
  * offers a 3D view where **height is real data** — the depth to water — so the
    extrusion is a spatial terrain of the water table, not decoration.

Uses `st.pydeck_chart` (pydeck ships with Streamlit) with `on_select="rerun"` and
`selection_mode="single-object"`. folium / streamlit-folium are NOT installed in
this venv and are not required.

Page contract: module-level script (run directly by `st.navigation`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _data as D
import _up_boundary as B

COND_COLOR = {
    "falling fast": [200, 40, 40, 200],
    "falling": [225, 130, 40, 200],
    "stable": [70, 130, 180, 185],
    "rising": [60, 160, 110, 200],
    "rising fast": [30, 130, 80, 210],
    "sparse": [140, 100, 170, 200],
    "no recent data": [120, 120, 120, 170],
    "cannot assess": [150, 150, 150, 150],
}
LAYER_ID = "stations"


@st.cache_data(ttl="10m", show_spinner=False)
def _coords() -> pd.DataFrame:
    return D.station_coords()


@st.cache_data(ttl="10m", show_spinner="Computing station indicators…")
def _indicators(days: int) -> pd.DataFrame | None:
    df = D.recent_window(days=days)
    if df is None or df.empty:
        return None
    gwl = pd.to_numeric(df["gwl"], errors="coerce")
    obs = gwl.notna()
    if "gwl_obs" in df.columns:
        obs = obs & (pd.to_numeric(df["gwl_obs"], errors="coerce") == 1.0)
    df = df.assign(_obs=obs, _gwl=gwl)
    last = df["time"].max()
    g = df.groupby("Station")
    rows = pd.DataFrame({
        "last_observed": g.apply(lambda x: x.loc[x["_obs"], "time"].max(),
                                 include_groups=False),
        "n_slots": g.size(),
        "n_observed": g["_obs"].sum(),
        "latest_gwl": g["_gwl"].last(),
    })
    rows["coverage"] = (rows["n_observed"] / rows["n_slots"]).round(3)
    rows["recency_days"] = ((last - rows["last_observed"]).dt.total_seconds()
                            / 86400.0).round(1)

    dd = (df[df["_obs"]].assign(date=df["time"].dt.normalize())
          .groupby(["Station", "date"])["_gwl"].median().reset_index())
    cut = last - pd.Timedelta(days=30)

    def _rec(gp: pd.DataFrame) -> pd.Series:
        cur = gp[gp["date"] >= cut]["_gwl"]
        old = gp[gp["date"] < cut]["_gwl"]
        return pd.Series({"gwl_now": float(cur.iloc[-1]) if len(cur) else np.nan,
                          "gwl_old": float(old.iloc[-1]) if len(old) else np.nan})

    rows = rows.join(dd.groupby("Station").apply(_rec, include_groups=False))
    rows["change_30d_m"] = (rows["gwl_now"] - rows["gwl_old"]).round(3)
    return rows.reset_index()


def _condition(r: pd.Series) -> str:
    if pd.isna(r.get("recency_days")) or r["recency_days"] > 14:
        return "no recent data"
    if pd.notna(r.get("coverage")) and r["coverage"] < 0.5:
        return "sparse"
    ch = r.get("change_30d_m")
    if pd.isna(ch):
        return "cannot assess"
    if ch <= -0.5:
        return "falling fast"
    if ch <= -0.2:
        return "falling"
    if ch >= 0.5:
        return "rising fast"
    if ch >= 0.2:
        return "rising"
    return "stable"


st.title("Map")
st.caption(
    "Uttar Pradesh groundwater telemetry network. Click a station to select it — "
    "every other page then opens on that station."
)

coords = _coords()
if coords.empty:
    st.error(
        "**No coordinates available.** `data/meta/selected_gwl_stations.csv` is missing "
        "or has no Latitude/Longitude columns (written by `01_select.py`)."
    )
    st.stop()

# ---- controls --------------------------------------------------------------
c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
with c1:
    view_mode = st.radio("View", ["2D plan", "3D terrain"], horizontal=True)
with c2:
    colour_by = st.selectbox("Colour pins by", ["Condition", "Coverage", "None"],
                             index=0)
with c3:
    window = st.selectbox("Indicator window", [60, 120, 240], index=1,
                          format_func=lambda d: f"last {d} days")
with c4:
    search = st.text_input("Highlight matching stations",
                           placeholder="station or district text")

pitch, bearing, height = 0.0, 0.0, 620
if view_mode == "3D terrain":
    d1, d2, d3 = st.columns([2, 2, 2])
    with d1:
        pitch = st.slider("Tilt (°)", 15, 70, 50, step=5,
                          help="0° is straight down. A tilt reveals the vertical "
                               "structure of the water table.")
    with d2:
        bearing = st.slider("Rotate (°)", 0, 355, 0, step=5)
    with d3:
        height = st.slider("Map height", 480, 980, 760, step=20)
    st.caption(
        "Column height is **depth to water** (deeper = taller), so the vertical axis "
        "is data, not decoration. Stations missing a recent level are drawn flat."
    )
else:
    height = st.slider("Map height", 420, 900, 620, step=20)

ind = _indicators(int(window))
if ind is None:
    st.warning("**Station indicators unavailable** — the aligned table could not be read. "
               "Pins are shown from the station metadata alone.")
    layer_df = coords.copy()
    layer_df["condition"] = "cannot assess"
    layer_df["coverage"] = np.nan
    layer_df["change_30d_m"] = np.nan
    layer_df["recency_days"] = np.nan
    layer_df["latest_gwl"] = np.nan
else:
    layer_df = coords.merge(ind, on="Station", how="left")
    layer_df["condition"] = layer_df.apply(_condition, axis=1)
    merged = int(layer_df["coverage"].notna().sum())
    st.caption(
        f"{len(coords):,} stations mapped · {merged:,} with a record in the last "
        f"{window} days · {len(coords) - merged:,} without (shown greyed, so the gap "
        "is visible rather than hidden)."
    )

# ---- colour ---------------------------------------------------------------
layer_df["radius"] = 900.0
if colour_by == "Condition":
    layer_df["color"] = layer_df["condition"].map(COND_COLOR).apply(
        lambda v: v if isinstance(v, list) else [150, 150, 150, 140])
elif colour_by == "Coverage":
    def _cov_colour(v):
        if not np.isfinite(v):
            return [150, 150, 150, 140]
        t = float(np.clip(v, 0.0, 1.0))
        return [int(210 * (1 - t) + 30 * t), int(60 * (1 - t) + 160 * t), 70, 190]
    layer_df["color"] = pd.to_numeric(layer_df["coverage"], errors="coerce").apply(_cov_colour)
else:
    layer_df["color"] = [[70, 130, 180, 170]] * len(layer_df)

n_match = 0
if search and search.strip():
    q = search.strip().lower()
    dist = (layer_df["District"].fillna("").astype(str) if "District" in layer_df.columns
            else pd.Series("", index=layer_df.index))
    hit = (layer_df["Station"].astype(str).str.lower() + " " + dist.str.lower()).str.contains(
        q, regex=False, na=False)
    n_match = int(hit.sum())
    if n_match:
        layer_df.loc[hit, "color"] = [[255, 210, 60, 235]] * n_match
        layer_df.loc[hit, "radius"] = 2200.0
layer_df["radius"] = pd.to_numeric(layer_df["radius"], errors="coerce").fillna(900.0)

if search and search.strip():
    st.caption(f"**{n_match} station(s) highlighted** in amber." if n_match
               else "No station name or district matches that text.")

# legend
if colour_by in ("Condition", "Coverage"):
    if colour_by == "Condition":
        counts = layer_df["condition"].value_counts()
        swatches = []
        for k, v in counts.items():
            rgb = COND_COLOR.get(k, [150, 150, 150, 255])
            swatches.append(
                f"<span style='display:inline-block;margin:0 12px 0 0'>"
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:50%;background:rgb({rgb[0]},{rgb[1]},{rgb[2]})'></span>"
                f" <small>{k} ({int(v)})</small></span>")
        st.markdown(" ".join(swatches), unsafe_allow_html=True)
    else:
        st.markdown("<small>Coverage: <span style='color:#d23c46'>poor</span> → "
                    "<span style='color:#1ea050'>good</span> · grey = no recent "
                    "record</small>", unsafe_allow_html=True)

# ---- deck ------------------------------------------------------------------
try:
    import pydeck as pdk
except Exception as e:  # noqa: BLE001
    st.error(f"pydeck unavailable ({type(e).__name__}: {e}). It normally ships with "
             "Streamlit; check the venv.")
    st.stop()

outline = B.load_outline()
boundary_layer = pdk.Layer(
    "PolygonLayer",
    data=[{"polygon": ring} for ring in outline],
    id="up-boundary",
    get_polygon="polygon",
    stroked=True,
    filled=True,
    get_line_color=[120, 190, 255, 220],
    get_fill_color=[40, 80, 130, 45],
    line_width_min_pixels=2,
    pickable=False,
)

if view_mode == "3D terrain":
    lvl = pd.to_numeric(layer_df["latest_gwl"], errors="coerce")
    # Depth to water, metres below the station datum. Values above datum are
    # clamped so every column keeps a visible base.
    depth = (-lvl).clip(lower=0)
    # Elevation is METRES of real vertical extent, exaggerated 12x so a 10-30 m
    # depth range is legible at state scale.
    layer_df = layer_df.assign(
        elevation=(depth.fillna(0.0) * 12.0).to_numpy(),
        depth_m=depth.round(2),
    )
    station_layer = pdk.Layer(
        "ColumnLayer",
        data=layer_df,
        id=LAYER_ID,
        get_position=["longitude", "latitude"],
        get_elevation="elevation",
        radius=1400,
        elevation_scale=1,
        extruded=True,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        disk_resolution=10,
    )
    tooltip = {
        "html": "<b>{Station}</b><br/>{District}<br/>"
                "level {latest_gwl} m · depth {depth_m} m<br/>"
                "30 d {change_30d_m} m · coverage {coverage}<br/>{condition}"
                "<br/><i>column height = depth to water, 12× exaggerated</i>",
        "style": {"backgroundColor": "#22303f", "color": "white", "fontSize": "12px"},
    }
else:
    station_layer = pdk.Layer(
        "ScatterplotLayer",
        data=layer_df,
        id=LAYER_ID,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=4,
        radius_max_pixels=14,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": "<b>{Station}</b><br/>{District}<br/>"
                "level {latest_gwl} m · 30 d {change_30d_m} m<br/>"
                "coverage {coverage} · {condition}",
        "style": {"backgroundColor": "#22303f", "color": "white", "fontSize": "12px"},
    }

(vmin_lon, vmin_lat), (vmax_lon, vmax_lat) = B.UP_BBOX
view = pdk.ViewState(
    longitude=B.UP_CENTER[0], latitude=B.UP_CENTER[1],
    zoom=6.0 if view_mode == "2D plan" else 5.5,
    pitch=pitch, bearing=bearing,
    min_zoom=5.0, max_zoom=11.0,
)

chart = pdk.Deck(
    layers=[boundary_layer, station_layer],
    initial_view_state=view,
    tooltip=tooltip,
    map_style="dark",
)

event = st.pydeck_chart(
    chart, height=int(height), on_select="rerun",
    selection_mode="single-object", key=f"station_map_{view_mode}",
)

# ---- selection -------------------------------------------------------------
picked = None
sel = getattr(event, "selection", None) or {}
objects = sel.get("objects", {}) if isinstance(sel, dict) else {}
hits = objects.get(LAYER_ID) or []
if hits:
    picked = str(hits[0].get("Station", "")) or None

s1, s2 = st.columns([3, 2])
with s1:
    if picked:
        row = layer_df[layer_df["Station"] == picked].iloc[0]
        lvl, chg = row.get("latest_gwl"), row.get("change_30d_m")
        bits = [f"**{picked}**", str(row.get("District", "—"))]
        bits.append(f"level {lvl:.2f} m" if pd.notna(lvl) else "no recent level")
        if pd.notna(chg):
            bits.append(f"30 d {chg:+.2f} m")
        bits.append(str(row.get("condition", "—")))
        st.success(" · ".join(bits))
        if st.session_state.get("focus_station") != picked:
            st.session_state["focus_station"] = picked
        st.caption("Selected. Open **Station**, **Water balance** or **Quality** — each "
                   "will open on this well.")
    else:
        st.info("Click a station to select it.")
with s2:
    manual = st.selectbox("…or pick from the list",
                          ["(none)"] + sorted(layer_df["Station"].astype(str)), index=0)
    if manual != "(none)":
        st.session_state["focus_station"] = manual
        st.caption(f"Focus set to **{manual}**.")

st.caption(
    f"State outline: {B.outline_source()}. Station coordinates come from "
    "`data/meta/selected_gwl_stations.csv`; the aligned 6-hourly table carries none, "
    "so stations without a coordinate are absent here. The viewport is locked to the "
    "state's bounding box, but the basemap itself is global — zooming out will show "
    "neighbouring states."
)
