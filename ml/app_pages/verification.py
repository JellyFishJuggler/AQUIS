import json
from pathlib import Path

import pandas as pd
import streamlit as st

from refresh import config, verification

REFRESH_DIR = config.REFRESH_DIR

st.title("Forecast verification")
st.caption(
    "Realised groundwater readings vs the forecast that was live at the time. "
    "Drift trips the scheduler into an out-of-window retrain when the 30-day "
    "error climbs 10% above the promoted backtest baseline, or coverage "
    "collapses, for two consecutive summary cycles."
)

summary_path = Path(verification.VERIF_SUMMARY)
if not summary_path.exists():
    st.info("No verification summary yet — it appears after forecast cycles "
            "start scoring realised readings.")
    st.stop()

summary = json.loads(summary_path.read_text())
base = summary.get("baseline_30d_rmse")
st.metric("Ledger rows", f"{summary.get('ledger_rows', 0):,}", border=True)

with st.container(horizontal=True):
    st.metric(
        "24h MAE",
        f"{summary['windows'].get('24h', {}).get('mae'):.3f} m"
        if summary["windows"].get("24h", {}).get("mae") is not None else "n/a",
        border=True,
    )
    st.metric(
        "7d MAE",
        f"{summary['windows'].get('7d', {}).get('mae'):.3f} m"
        if summary["windows"].get("7d", {}).get("mae") is not None else "n/a",
        border=True,
    )
    st.metric(
        "30d MAE",
        f"{summary['windows'].get('30d', {}).get('mae'):.3f} m"
        if summary["windows"].get("30d", {}).get("mae") is not None else "n/a",
        border=True,
    )
    st.metric(
        "30d coverage",
        f"{summary['windows'].get('30d', {}).get('coverage_95')*100:.0f}%"
        if summary["windows"].get("30d", {}).get("coverage_95") is not None else "n/a",
        border=True,
    )

drift = summary.get("drift", {})
with st.container(horizontal=True):
    st.metric(
        "Drift",
        "TRIPPED" if drift.get("tripped") else "ok",
        drift.get("reason") if drift.get("reason") else "within tolerance",
        border=True,
    )

st.subheader("Windows (rolling)")
rows = []
for w, d in summary.get("windows", {}).items():
    if not d:
        continue
    rows.append({"window": w, "n": d["n"], "mae": d["mae"], "rmse": d["rmse"],
                 "bias": d["bias"], "coverage_95": d["coverage_95"]})
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("No scored horizons yet — realisations accumulate as NWIC "
               "publishes newer readings past forecast anchors.")

ledger_path = Path(verification.VERIF_LEDGER)
if ledger_path.exists():
    st.subheader("Ledger sample")
    led = pd.read_parquet(ledger_path)
    show = ["station", "anchor_time", "horizon_h", "target_time",
            "q05", "q50", "q95", "actual", "abs_err", "coverage_hit",
            "model_version"]
    cols = [c for c in show if c in led.columns]
    st.dataframe(led[cols].sort_values("target_time", ascending=False).head(200),
                 use_container_width=True, hide_index=True)