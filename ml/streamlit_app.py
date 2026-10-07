"""AQUIS ML showcase — read-only Streamlit entry point.

Run from this directory with ``python -m streamlit run streamlit_app.py``.
The app presents existing ML data and artifacts; it does not train models,
fetch data, or write generated outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ML_ROOT = Path(__file__).resolve().parent
APP_PAGES = ML_ROOT / "app_pages"

# `st.navigation` runs each page with `exec(code, module.__dict__)`, which does NOT
# put the page's own directory on `sys.path` the way running a script directly
# would. The pages import their shared helpers by flat name (`_data`,
# `_hydro_metrics`), so both the ml root (for `06_features`, `_trajectory`, ...)
# and app_pages (for those helpers) must be importable before `page.run()`.
# Order matters: app_pages first, so `_data` can never be shadowed by `ml/_data`.
for _p in (APP_PAGES, ML_ROOT):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

st.set_page_config(
    page_title="AQUIS — groundwater platform",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("AQUIS")
st.sidebar.caption("Groundwater monitoring & forecasting")
st.sidebar.info(
    "Read-only: no training, fetching, or artifact generation. "
    "Hydrology views are descriptive; model views state their protocol."
)


def _p(name: str, title: str, icon: str):
    return st.Page(str(ML_ROOT / "app_pages" / name), title=title, icon=icon)


page = st.navigation(
    {
        "Monitor": [
            _p("map_view.py", "Map", ":material/map:"),
            _p("monitoring.py", "Monitoring", ":material/dashboard:"),
            _p("station.py", "Station", ":material/water_drop:"),
            _p("water_balance.py", "Water balance", ":material/rainy:"),
            _p("quality.py", "Quality", ":material/fact_check:"),
        ],
        "Forecast": [
            _p("forecast_explorer.py", "Forecast explorer", ":material/query_stats:"),
        ],
        "Model": [
            _p("overview.py", "Model overview", ":material/home:"),
            _p("evaluation.py", "Model trust", ":material/assessment:"),
            _p("feature_explorer.py", "Features", ":material/tune:"),
        ],
        "Reference": [
            _p("data_quality.py", "Sources & limits", ":material/database:"),
        ],
    },
    position="sidebar",
)
page.run()

