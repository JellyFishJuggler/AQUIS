import pandas as pd
import streamlit as st

from _utils import load_manifest, load_selected_gwl, load_report

st.title("AQUIS — groundwater correlation explorer")
st.caption(
    "Correlation-first validation of which NWIC environmental drivers affect "
    "groundwater level. Sandbox (`ml/`), not production."
)

man = load_manifest()
gwl = load_selected_gwl()

gwl_rows = next(m["rows"] for m in man if m["source"] == "gwl")
driver_rows = sum(m["rows"] for m in man) - gwl_rows
cov_by_source = {m["source"]: m["stations"] for m in man}

with st.container(horizontal=True):
    st.metric("Selected GWL stations", f"{gwl['Station'].nunique()}", f"{gwl['District'].nunique()} districts", border=True)
    st.metric("GWL records", f"{gwl_rows:,}", "2021-01-01 → 2026-09-05", border=True)
    st.metric("Driver records", f"{driver_rows:,}", f"{len(cov_by_source) - 1} drivers", border=True)
    st.metric("Soil profiles mapped", "580 stations", "ISRIC SoilGrids v2", border=True)

st.header("Driver signal (median correlation vs groundwater)")

report = load_report().dropna(subset=["corr"])
raw = report[(report["mode"] == "raw") & (report["metric"] == "spearman")].copy()

best_row = raw.loc[raw["corr"].abs().idxmax()]
dry_row = raw.loc[raw["corr"].abs().idxmin()]

with st.container(horizontal=True):
    st.metric(
        "Strongest driver",
        f"{best_row['driver']}",
        f"Spearman {best_row['corr']:+.3f} · {int(best_row['stations_ok'])} stations",
        border=True,
    )
    st.metric(
        "Weakest driver",
        f"{dry_row['driver']}",
        f"Spearman {dry_row['corr']:+.3f}",
        border=True,
    )

st.subheader("Drivers ranked by raw |Spearman|")
ranked_bar = raw.copy()
ranked_bar["label"] = ranked_bar["driver"]
st.bar_chart(ranked_bar.sort_values("corr", key=abs), x="label", y="corr", color="#4ecca3")

st.subheader("Driver station coverage")
st.bar_chart(
    pd.DataFrame([{"source": k, "stations": v} for k, v in cov_by_source.items()
                  if k != "gwl"]).sort_values("stations", ascending=False),
    x="source",
    y="stations",
)

st.caption("Data read from local `ml/outputs/` and `ml/data/` — see README for the decision log.")