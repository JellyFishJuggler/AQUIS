"""Correlation page — correlation matrix + feature importance (same feature set)."""

import pandas as pd
import streamlit as st

from _model import load_importance
from _utils import load_report, nice

st.title("Correlation report")
st.caption(
    "Two complementary views of the same feature set. The matrix correlates every "
    "feature the forecast model uses against current groundwater level; the chart "
    "shows how much each feature contributes inside the XGBoost model."
)

rep = load_report().dropna(subset=["corr"])
rep["series"] = rep["mode"] + " · " + rep["metric"]

raw_sp = rep[rep["series"] == "raw · spearman"].set_index("driver")
order = raw_sp["corr"].abs().sort_values(ascending=False).index

st.subheader("Feature correlation with groundwater level")
st.caption(
    "Median Spearman/Pearson across stations — raw drivers plus the engineered "
    "features the model trains on (GWL lags and rolling stats, calendar cycles). "
    "Ordinal station/district codes are kept out here; they only appear in the "
    "model chart below."
)
raw = rep[rep["mode"] == "raw"].pivot_table(
    index="driver", columns="metric", values="corr"
).reindex(order)
raw.index = raw.index.map(nice)
st.dataframe(
    raw.round(3),
    hide_index=True,
    column_config={
        "spearman": st.column_config.NumberColumn("Spearman r", format="%.3f"),
        "pearson": st.column_config.NumberColumn("Pearson r", format="%.3f"),
    },
    width="stretch",
)

with st.expander("Detrended views — level drivers only"):
    st.caption(
        "First-difference and de-seasonalised correlations isolate short-term "
        "driver effects from the slow seasonal GWL cycle."
    )
    det = rep[rep["mode"].isin(["diff", "deseason"])].pivot_table(
        index="driver", columns=["mode", "metric"], values="corr"
    )
    det.index = det.index.map(nice)
    st.dataframe(det.round(3), hide_index=True, width="stretch")

st.subheader("Model feature importance")
st.caption(
    "XGBoost gain on the 30-day forecast target — top 15 of the same features "
    "listed above (station / district are ordinal codes)."
)
imp = load_importance().sort_values("gain", ascending=False).head(15)
imp["label"] = imp["feature"].map(lambda f: nice(f) if pd.notna(f) else "(overall)")
st.bar_chart(imp, x="label", y="gain", color="#4ecca3")