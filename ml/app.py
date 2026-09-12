import streamlit as st

st.set_page_config(
    page_title="AQUIS — groundwater explorer",
    page_icon=":material/analytics:",
    layout="wide",
)

from _assistant import get_df
from _utils import sidebar_station_picker

# --- Global sidebar: District + Station pickers, visible on EVERY page ---
# Rendered here (before page.run()) so the selection exists regardless of
# which page is open. Pages must NOT call sidebar_station_picker themselves
# (duplicate widget keys) — they read sb_district / sb_station instead.
_df = get_df()
_recency = _df.groupby("Station")["time"].max().sort_values(ascending=False)
_all_stations = [str(s) for s in _recency.index]
sidebar_station_picker(_df, _all_stations)

page = st.navigation(
    [
        st.Page("app_pages/assistant.py", title="Assistant", icon=":material/smart_toy:", default=True),
        st.Page("app_pages/correlation.py", title="Correlation", icon=":material/query_stats:"),
        st.Page("app_pages/forecast.py", title="Forecast", icon=":material/troubleshoot:"),
        st.Page("app_pages/data_sources.py", title="Sources", icon=":material/database:"),
    ]
)

page.run()