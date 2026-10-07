"""Page 1 — Overview: what the pipeline does, what is deployed, and the flow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _data as D


def _kv(df: pd.DataFrame, col: str) -> dict:
    if df is None or col not in df.columns:
        return {}
    return dict(zip(df["artifact"], df[col]))


st.title("AQUIS ML — pipeline overview")
st.caption(
    "Groundwater-level forecasting for Uttar Pradesh telemetry wells. "
    "Read-only showcase: nothing here trains, retrains, or regenerates an artifact."
)

# ---- what the pipeline does -------------------------------------------
st.subheader("What the model does")
left, right = st.columns([3, 2])
with left:
    st.markdown(
        """
**Task.** Forecast the **30-day change in groundwater level** for a well, on a
**6-hourly** grid (120 steps of 6 h), then report the level as
`today's observed reading + predicted change`.

**Why a *change* and not a level.** Well datums are station-local — roughly
three-quarters of readings are negative, meaning "metres below the reference
datum". Predicting an absolute level across pooled wells would mostly teach the
model *which datum this is*. Modelling the change makes persistence
(`no change`) the natural benchmark and removes the datum from the learning
problem. The anchor reading is therefore always an **input, never a prediction**.

**Two engines ship today.** A pooled single-horizon XGBoost supplies the
+30-day benchmark used by the fleet scan and the assistant; a shared
direct multi-horizon XGBoost ("trajectory v2") supplies the genuine 120-step
curve with quantile bands that the forecast surface shows.
"""
    )
with right:
    st.markdown("**Evaluation flow**")
    st.code(
        "telemetry (NWIC 6-hourly GWL)\n"
        "        +\n"
        "drivers (rain IDW, weather, river,\n"
        "         canal, calendar)\n"
        "        |\n"
        "        v\n"
        "04/04b  align to 6 h grid\n"
        "        |\n"
        "        v\n"
        "06/14   causal features + target\n"
        "        |\n"
        "        v\n"
        "06/31   train (train < 2025-10)\n"
        "        |\n"
        "        v\n"
        "07/32   evaluate (val, 2026 test,\n"
        "         non-overlap backtest)\n"
        "        |\n"
        "        v\n"
        "33      confidence + calibration",
        language="text",
    )

# ---- deployed models ---------------------------------------------------
st.subheader("Deployed models")
cfg = D.feature_config()
tcfg = D.traj_config()
cal = D.traj_calibration()
rel = D.traj_reliability()

if cfg is None and tcfg is None:
    st.warning("No model configuration artifacts found under `ml/models/`.")
else:
    rows = []
    if tcfg:
        rows.append({
            "engine": "trajectory v2 (Forecast surface)",
            "artifact": "models/traj_xgb_q{05,50,95}.json",
            "target": tcfg.get("target", "—"),
            "horizon": f"direct 1..{tcfg.get('max_h', '—')} steps "
                       f"({int(tcfg.get('max_h', 0)) * 6 // 24} d), no recursion",
            "features": f"{len(tcfg.get('feature_cols', []))}",
            "trained_at": tcfg.get("trained_at", "—"),
        })
    if cfg:
        rows.append({
            "engine": "pooled 30-day (benchmark / fleet / assistant)",
            "artifact": "models/xgb_multihorizon.joblib",
            "target": cfg.get("target", "—"),
            "horizon": "single, 120 steps (30 d)",
            "features": f"{len(cfg.get('num_cols', []))} numeric + 2 ordinals",
            "trained_at": cfg.get("trained_at", "—"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    if cfg:
        c1.metric("Stations (pooled)", f"{len(cfg.get('stations', [])):,}")
        c2.metric("Districts (pooled)", f"{len(cfg.get('districts', [])):,}")
    if rel:
        thr = rel.get("thresholds", {})
        c3.metric("Confidence levels", " / ".join(rel.get("levels", [])))
        c4.metric("Coverage floor", thr.get("coverage_min", "—"))

# ---- serving state -----------------------------------------------------
st.subheader("Serving state")
rt = D.runtime_json()
meta = D.refresh_meta()
s1, s2, s3 = st.columns(3)
if rt:
    s1.metric("feature_mode", str(rt.get("feature_mode", "—")))
    s2.metric("model_version", str(rt.get("model_version") or "unset"))
else:
    s1.info("`data/refresh/runtime.json` absent")
if meta:
    s3.metric("data_status", str(meta.get("data_status", "—")))
else:
    s3.info("`data/refresh/meta.json` absent")
if rt and rt.get("feature_mode") == "flat":
    st.caption(
        "`feature_mode = flat` means the forecast step replicates the anchor row "
        "across all horizons; the horizon-aware `future` mode is gated behind a "
        "backtest and is not active."
    )

# ---- calibration at a glance ------------------------------------------
if cal:
    st.subheader("Reported interval calibration")
    dfc = pd.DataFrame({
        "horizon (h)": cal.get("horizons_h", []),
        "raw coverage": cal.get("coverage_raw", []),
        "calibrated coverage": cal.get("coverage_calibrated", []),
        "widening s": cal.get("widening_s", []),
        "n anchors": cal.get("n", []),
    })
    st.dataframe(dfc, width="stretch", hide_index=True)
    st.warning(
        "**Protocol caveat.** In the current artifact the widening `s` is fitted "
        "on the same anchor set whose coverage it reports, so "
        "`calibrated coverage = 0.90` is a description of the fit, not an "
        "independent validation. Treat it as in-sample until the calibration "
        "and reporting anchors are separated."
    )

# ---- artifact inventory ------------------------------------------------
st.subheader("Artifact inventory")
rep = D.artifact_report()
n_present = int(rep["present"].sum())
st.caption(f"{n_present} / {len(rep)} known artifacts present on this checkout.")
show_missing_only = st.checkbox("Show missing only", value=(n_present < len(rep)))
view = rep[~rep["present"]] if show_missing_only else rep
view = view.copy()
view["bytes"] = view["bytes"].map(D.human_bytes)
st.dataframe(view, width="stretch", hide_index=True)
