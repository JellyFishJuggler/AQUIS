"""Data and limitations: source coverage and station freshness metadata."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _data as D


@st.cache_data(ttl="10m", max_entries=4, show_spinner=False)
def _manifest() -> list | None:
    return D.manifest()


@st.cache_data(ttl="10m", max_entries=4, show_spinner=False)
def _selected_stations() -> pd.DataFrame | None:
    return D.read_csv(D.META / "selected_gwl_stations.csv")


@st.cache_data(ttl="10m", max_entries=4, show_spinner=False)
def _stale_stations() -> pd.DataFrame | None:
    return D.read_csv(D.META / "stale_watchlist.csv")


@st.cache_data(ttl="10m", max_entries=4, show_spinner=False)
def _openmeteo_meta() -> dict | None:
    return D.read_json(D.META / "openmeteo_meta.json")


st.title("Data and limitations")
st.caption(
    "Coverage and freshness are shown from existing ML metadata. This page reads "
    "local files only; it does not fetch or refresh data."
)

manifest = _manifest()
if not manifest:
    st.info("**Data/artifact unavailable.** `data/meta/manifest.json` is missing or unreadable.")
else:
    sources = pd.DataFrame(manifest)
    if "source" in sources.columns:
        sources = sources.set_index("source", drop=False)
    gwl = sources.loc["gwl"] if "gwl" in sources.index else None
    rain = sources.loc["rainfall"] if "rainfall" in sources.index else None
    weather = sources.loc["temperature"] if "temperature" in sources.index else None

    with st.container(horizontal=True):
        if gwl is not None:
            st.metric("GWL stations in manifest", f"{int(gwl.get('stations', 0)):,}")
            st.metric("GWL districts", f"{int(gwl.get('districts', 0)):,}")
        if rain is not None:
            st.metric("Rain gauge stations", f"{int(rain.get('stations', 0)):,}")
            st.metric("Rainfall districts", f"{int(rain.get('districts', 0)):,}")

    st.subheader("Source coverage snapshot")
    visible = [c for c in ("source", "rows", "stations", "districts", "span")
               if c in sources.columns]
    st.dataframe(sources[visible].reset_index(drop=True), width="stretch", hide_index=True)
    if "stations" in sources.columns and "source" in sources.columns:
        chart = sources[sources["stations"].notna()].set_index("source")["stations"]
        st.bar_chart(chart, width="stretch", height=360)

    st.subheader("Weather feed freshness")
    om = _openmeteo_meta()
    if om:
        stamp = pd.to_datetime(om.get("fetched_at"), errors="coerce", utc=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Weather districts", str(om.get("districts", "—")))
        c2.metric("Weather rows", f"{int(om.get('rows', 0)):,}")
        c3.metric("Last recorded fetch", stamp.strftime("%Y-%m-%d %H:%M UTC")
                  if pd.notna(stamp) else "Unavailable")
        failures = om.get("district_failures") or {}
        if failures:
            st.warning(f"The metadata records fetch failures for {len(failures)} districts.")
            st.json(failures, expanded=False)
    else:
        st.info("**Data/artifact unavailable.** `data/meta/openmeteo_meta.json` is missing.")

st.subheader("Station freshness and selection")
selected = _selected_stations()
stale = _stale_stations()
if selected is None:
    st.info("**Data/artifact unavailable.** `selected_gwl_stations.csv` is missing.")
else:
    total = len(selected)
    full = (selected["full26"].astype(str).str.lower().eq("true").sum()
            if "full26" in selected.columns else 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Stations in selected list", f"{total:,}")
    c2.metric("Marked full-2026", f"{int(full):,}")
    c3.metric("Not marked full-2026", f"{total - int(full):,}")

if stale is not None and not stale.empty:
    st.caption(f"Stale/sparse watchlist entries: {len(stale):,}")
    reason_col = "reason" if "reason" in stale.columns else None
    if reason_col:
        reasons = stale[reason_col].fillna("unspecified").value_counts().rename_axis("reason")
        st.bar_chart(reasons, width="stretch", height=300)
    with st.expander("Inspect watchlist", expanded=False):
        st.dataframe(stale.head(100), width="stretch", hide_index=True)
else:
    st.info("No stale watchlist artifact is available on this checkout.")

st.subheader("What limits the current ML results")
st.markdown(
    """
- Rainfall and point-weather coverage is uneven across stations; a driver being
  listed in the model schema does not mean it is observed near every well.
- River and canal values can be sparse or proxy-based. Check source and date
  coverage before interpreting a feature as a local hydrological cause.
- `HYDRO_COLS` are experimental and untrained. In particular, `rain_api*` has a
  training/serving mismatch until both paths use the same point-in-time definition.
- Feature importance and correlations are associations; they do not establish
  that rainfall, pumping, or another driver caused a specific groundwater change.
- Groundwater level alone cannot establish water quality or identify pumping.
"""
)
