import streamlit as st

from _utils import load_manifest, source_nice, BASE, META

SOURCE_URLS = {
    "gwl": "https://nwdp.nwic.gov.in/dataset/e6a8307d-de95-4773-8e9f-a84d7a177d53",
    "rainfall": "https://nwdp.nwic.gov.in/dataset/9c7f3d81-e9f8-4d7e-a0d5-468b9941c503",
    "temperature": "https://nwdp.nwic.gov.in/dataset/799e6b02-4822-4f6b-875d-a2145105a16d",
    "river_level": "https://nwdp.nwic.gov.in/dataset/92242bf5-b976-49e9-b036-aea753be68b9",
    "humidity": "https://nwdp.nwic.gov.in/dataset/bdee1690-2932-41ae-8b1c-3abece099210",
    "solar": "https://nwdp.nwic.gov.in/dataset/b473cb05-f583-4373-a039-414254a94428",
    "wind_speed": "https://nwdp.nwic.gov.in/dataset/c418c625-baad-40b7-bf92-19a625ca7138",
    "wind_direction": "https://nwdp.nwic.gov.in/dataset/fe3e5c9a-8f10-4d41-8c7c-c07beeb03c94",
    "pressure": "https://nwdp.nwic.gov.in/dataset/f1bc9009-d25b-4df9-95b5-57664de7fb38",
    "canal_level": "https://nwdp.nwic.gov.in/dataset/fc0bd3a7-b10b-42d5-a7a6-66b580a156c2",
    "canal_discharge": "https://nwdp.nwic.gov.in/dataset/11c25d35-d510-457e-844b-7be258610093",
}

st.title("Data sources")
st.caption("Per-source quality summary for the selected 29-district / 600-station set.")

man = load_manifest()
cols = st.columns(2)
cards = list(man)

good = [m for m in cards if m["rows"] > 0]
empty = [m for m in cards if m["rows"] == 0]

if good:
    st.subheader("Sources with data in selected districts")
    for i, m in enumerate(good):
        col = cols[i % 2]
        with col:
            with st.container(border=True):
                url = SOURCE_URLS.get(m["source"])
                name = source_nice(m["source"])
                st.markdown(f"**{name}**"
                            + (f" — [NWDP dataset ↗]({url})" if url else ""))
                st.metric("Rows", f"{m['rows']:,}", border=True)
                st.caption(
                    f"{m['stations']} stations · {m['districts']} districts · "
                    f"span {m['span']} · cap-dropped {m['dropped_by_caps']:,}"
                )
                if m.get("discharge_gate1_only"):
                    st.caption(":orange-badge[discharge] Gate-1 series only (approximation)")

if empty:
    st.subheader("Sources with no data in selected districts")
    st.caption(
        "; ".join(
            f"[{source_nice(m['source'])} ↗]({SOURCE_URLS[m['source']]})"
            if m["source"] in SOURCE_URLS else source_nice(m["source"])
            for m in empty
        )
        + " — canal discharge and reservoir/barrage gauges lie outside the selected "
        "29 districts, so they carry no features here."
    )

st.header("Static hydrogeology")

soil_col, xcol = st.columns(2)
with soil_col:
    with st.container(border=True):
        st.markdown("**Soil texture (static)**")
        from _soil import load_soil, load_soil_meta
        import pandas as pd
        soil = load_soil()
        smeta = load_soil_meta()
        if not soil.empty:
            st.metric("Stations mapped", f"{soil['Station'].nunique():,}", border=True)
            st.caption(
                f"{smeta.get('source', 'ISRIC SoilGrids v2')} · 8 features: "
                "sand/silt/clay/bdod at 0–30 cm (thickness-weighted) and 60–100 cm"
            )
            st.caption("[ISRIC SoilGrids v2 ↗](https://rest.isric.org/docs/home) · "
                       "[SoilGrids portal ↗](https://soilgrids.org)")
        else:
            st.caption("SoilGrids fetch incomplete — no per-station soil features yet.")

with xcol:
    with st.container(border=True):
        st.markdown("**Groundwater extraction (district × year)**")
        from _soil import load_extraction
        import pandas as _pd
        ext = load_extraction()
        if not ext.empty:
            st.metric("Assessments", f"{len(ext):,}", border=True)
            st.caption("CGWB dynamic assessment: net annual draft + stage of development. "
                       "[CGWB ↗](https://cgwb.gov.in/dynamic-ground-water-resources.html)")
        else:
            st.caption(
                ":orange-badge[not loaded] no `data/soil/cgwb_extraction_district.csv`. "
                "Add district × year → `net_annual_gw_draft_mcm`, `stage_of_development_pct`."
            )

st.header("Land use / land cover (static, ISRO Bhuvan)")

import altair as alt
import pandas as pd
from _lulc import load_lulc, load_lulc_meta

lulc = load_lulc()
lmeta = load_lulc_meta()
if lulc.empty:
    st.caption(":orange-badge[not fetched] run `ml/08_lulc.py` to pull the 29-district LULC stats.")
else:
    l1, l2, l3 = st.columns([1, 2, 2])
    l1.metric("Districts", f"{len(lulc):,}", border=True)
    l1.caption(f"{lmeta.get('cycles', '2005-06 + 2011-12')} cycles · "
               f"features from the 2011-12 50K cycle")
    st.caption("[ISRO Bhuvan LULC statistics ↗](https://bhuvan-app1.nrsc.gov.in/api/)")
    long = pd.read_csv(META / "lulc_long.csv")
    l2.caption("Area share per LULC class (2011-12), by district")
    stacked = alt.Chart(long[long["year"] == "2011-12"]).mark_bar().encode(
        x=alt.X("district:N", title=None, sort="-y"),
        y=alt.Y("pct:Q", title="% of district area"),
        color=alt.Color("class:N", legend=alt.Legend(columns=2, symbolLimit=30)),
        tooltip=["district", "class", "area_sqkm", "pct"],
    ).properties(height=300)
    l2.altair_chart(stacked, use_container_width=True)
    l3.caption("LULC share vs district GWL summary (n=28 districts)")
    corr_l = pd.read_csv(BASE / "outputs" / "correlation_lulc.csv")
    l3.dataframe(
        corr_l.dropna(subset=["pearson_r"]).sort_values(
            "pearson_r", key=lambda s: s.abs(), ascending=False).head(10),
        hide_index=True, width="stretch",
    )
    with st.expander("LULC ↔ GWL correlation detail", icon=":material/analytics:"):
        st.dataframe(corr_l, hide_index=True, width="stretch")
        st.caption("Static class %. Correlations are over 28 districts — exploratory only "
                   "(fallen land ↔ deeper GWL r≈+0.42; built-up ↔ shallower GWL r≈−0.42).")
        st.caption("Not fed to the forecast model: district/station ordinals already absorb "
                   "this constant per-district signal.")

st.header("Association method")
st.write(
    "Rainfall is blended via **IDW (1/d²)** across the 3 nearest gauges within 150 km. "
    "Weather/river/canal join on the **nearest gauge** within a radius; river gauges publish "
    "no coordinates, so they fall back to a **district-level proxy** (flagged in the README)."
)

with st.expander("Association link counts", icon=":material/link:"):
    import pandas as pd
    from pathlib import Path
    from _utils import META

    links = {}
    for p in sorted(Path(META).glob("assoc_*.csv")):
        links[p.stem.removeprefix("assoc_")] = len(pd.read_csv(p))
    st.dataframe(
        pd.DataFrame(list(links.items()), columns=["Driver", "Well-gauge links"]),
        hide_index=True,
    )

st.caption("All values from `ml/data/`. Nothing here is committed or production.")
