"""Sources page (trimmed) — every data source the dashboard reads, as label + link."""

import streamlit as st

from _utils import source_nice

NWDP_URLS = {
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

FORECAST_URLS = {
    "Open-Meteo (archive + forecast weather)": "https://open-meteo.com",
    "CWC / FMISC flood dashboard (7-day river forecast)": "http://117.250.2.226:84/FMISCFloodDashboard/up_flood_dashboard.html",
}

STATIC_URLS = {
    "ISRIC SoilGrids v2": "https://soilgrids.org",
    "CGWB dynamic groundwater resources": "https://cgwb.gov.in/dynamic-ground-water-resources.html",
    "ISRO Bhuvan LULC": "https://bhuvan-app1.nrsc.gov.in/api/",
    "Ollama (local LLM server)": "https://ollama.com",
    "Llama 3.2 (3B) via Ollama": "https://ollama.com/library/llama3.2",
}

st.title("Sources")
st.caption("Every data source this dashboard reads, with its dataset link.")

st.markdown("### NWDP / NWIC datasets")
st.markdown(
    "\n".join(
        f"- **{source_nice(src)}** — [dataset ↗]({url})"
        for src, url in NWDP_URLS.items()
    )
)

st.markdown("### Forecast driver feeds")
st.markdown(
    "\n".join(f"- **{name}** — [link ↗]({url})" for name, url in FORECAST_URLS.items())
)

st.markdown("### Static & supporting")
st.markdown(
    "\n".join(f"- **{name}** — [link ↗]({url})" for name, url in STATIC_URLS.items())
)