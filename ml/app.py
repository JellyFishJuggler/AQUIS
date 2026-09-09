import streamlit as st

st.set_page_config(
    page_title="AQUIS — correlation explorer",
    page_icon=":material/analytics:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/home:"),
        st.Page("app_pages/correlation.py", title="Correlation", icon=":material/query_stats:"),
        st.Page("app_pages/drivers.py", title="Drivers", icon=":material/science:"),
        st.Page("app_pages/station_explorer.py", title="Stations", icon=":material/monitoring:"),
        st.Page("app_pages/model.py", title="Model", icon=":material/model_training:"),
        st.Page("app_pages/forecast.py", title="Forecast", icon=":material/troubleshoot:"),
        st.Page("app_pages/fleet.py", title="Fleet", icon=":material/query_stats:"),
        st.Page("app_pages/assistant.py", title="Assistant", icon=":material/smart_toy:"),
        st.Page("app_pages/data_sources.py", title="Sources", icon=":material/database:"),
    ]
)

page.run()