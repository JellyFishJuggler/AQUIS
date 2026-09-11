import streamlit as st

st.set_page_config(
    page_title="AQUIS — groundwater explorer",
    page_icon=":material/analytics:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page("app_pages/assistant.py", title="Assistant", icon=":material/smart_toy:", default=True),
        st.Page("app_pages/correlation.py", title="Correlation", icon=":material/query_stats:"),
        st.Page("app_pages/forecast.py", title="Forecast", icon=":material/troubleshoot:"),
        st.Page("app_pages/data_sources.py", title="Sources", icon=":material/database:"),
    ]
)

page.run()