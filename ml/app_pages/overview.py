import streamlit as st

from _utils import load_selected_gwl, load_report

st.title("Overview")
st.caption("Total monitored stations, and the strongest / weakest driver vs groundwater.")

gwl = load_selected_gwl()
st.metric(
    "Total stations",
    f"{gwl['Station'].nunique():,}",
    f"{gwl['District'].nunique()} districts",
    border=True,
)

rep = load_report().dropna(subset=["corr"])
raw = rep[(rep["mode"] == "raw") & (rep["metric"] == "spearman")].copy()
best = raw.loc[raw["corr"].abs().idxmax()]
weak = raw.loc[raw["corr"].abs().idxmin()]

with st.container(horizontal=True):
    st.metric(
        "Strongest driver",
        best["driver"],
        f"Spearman {best['corr']:+.3f} · {int(best['stations_ok'])} stations",
        border=True,
    )
    st.metric(
        "Weakest driver",
        weak["driver"],
        f"Spearman {weak['corr']:+.3f}",
        border=True,
    )