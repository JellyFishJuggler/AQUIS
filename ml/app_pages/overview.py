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

**Two quantile GRU forecast surfaces are selected in the local runtime.** The
pooled daily GRU predicts one +30-day endpoint; the trajectory GRU predicts
120 direct six-hourly horizons with q05/q50/q95 bands. XGBoost remains the
benchmark, and the trajectory dispatcher uses it only when runtime explicitly
selects XGBoost. Backend selection describes the local runtime configuration;
it is not a model-promotion claim.
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
        "34/36   train trajectory / pooled GRU\n"
        "        v\n"
        "32/37   backtest + compare",
        language="text",
    )

# ---- deployed models ---------------------------------------------------
st.subheader("Deployed models")
cfg = D.feature_config()
tcfg = D.traj_dl_config()
pcfg = D.pooled_dl_config()
cal = D.traj_dl_calibration()
rel = D.traj_dl_reliability()

if cfg is None and tcfg is None and pcfg is None:
    st.warning("No model configuration artifacts found under `ml/models/`.")
else:
    rows = []
    if tcfg:
        rows.append({
            "engine": "Trajectory quantile GRU (active forecast surface)",
            "artifact": "models/traj_dl.pt",
            "target": tcfg.get("target", "—"),
            "horizon": f"direct 1..{tcfg.get('max_h', '—')} steps "
                       f"({int(tcfg.get('max_h', 0)) * 6 // 24} d), no recursion",
            "features": f"{len(tcfg.get('feature_cols', []))} sequence features · "
                        f"hidden {tcfg.get('hidden', '—')}",
            "trained_at": tcfg.get("trained_at", "—"),
        })
    if pcfg:
        rows.append({
            "engine": "Pooled daily quantile GRU (active forward surface)",
            "artifact": "models/pooled_dl.pt",
            "target": pcfg.get("target", "—"),
            "horizon": "single endpoint (+30 d)",
            "features": f"{len(pcfg.get('feature_cols', []))} features · "
                        f"hidden {pcfg.get('hidden', '—')} · "
                        f"{len(pcfg.get('seeds', []))}-seed ensemble",
            "trained_at": pcfg.get("trained_at", "—"),
        })
    if cfg:
        rows.append({
            "engine": "Pooled XGBoost (benchmark / fallback)",
            "artifact": "models/xgb_multihorizon.joblib",
            "target": cfg.get("target", "—"),
            "horizon": "single endpoint (+30 d)",
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
backend = D.trajectory_backend_status()
s1, s2, s3, s4 = st.columns(4)
if rt:
    s1.metric("Trajectory requested", str(rt.get("traj_backend", "dl")).upper())
    s2.metric("Trajectory load", "degraded" if backend.get("degraded") else "ready")
    s3.metric("Pooled forward requested", str(rt.get("forward_backend", "dl")).upper())
else:
    s1.metric("Trajectory requested", "DL (default)")
    s2.metric("Trajectory load", "degraded" if backend.get("degraded") else "ready")
    s3.metric("Pooled forward requested", "DL (default)")
if meta:
    s4.metric("Data status", str(meta.get("data_status", "—")))
else:
    s4.info("`data/refresh/meta.json` absent")
if backend.get("degraded"):
    st.error(f"Configured trajectory backend is unavailable: {backend.get('error', 'unknown error')}")
else:
    st.caption(f"Trajectory dispatcher resolved **{str(backend.get('backend', 'unknown')).upper()}** for this app process.")
if rt and rt.get("feature_mode") == "flat":
    st.caption(
        "`feature_mode = flat` applies to the XGBoost trajectory feature builder. "
        "The active GRU uses its observed 120-step history and direct, "
        "horizon-conditioned output heads."
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
    st.caption(
        "For the trajectory GRU snapshot, widening is fitted on alternating "
        "anchors and coverage is reported on the disjoint complement. This is "
        "a within-period calibration check, not a guarantee for future seasons."
    )

# ---- GRU evaluation snapshot ------------------------------------------
st.subheader("DL-GRU evaluation snapshot")
pool_eval = D.pooled_dl_eval()
traj_eval = D.traj_dl_backtest_summary()
pool_col, traj_col = st.columns(2)
with pool_col:
    st.markdown("**Pooled daily GRU · one 30-day endpoint**")
    if pool_eval and pool_eval.get("DL"):
        dl = pool_eval["DL"]
        persist = pool_eval.get("persistence", {})
        paired = dl.get("paired_vs_persist", {})
        ci = paired.get("ci", [])
        with st.container(border=True):
            st.metric("RMSE", f"{dl.get('rmse', float('nan')):.3f} m",
                      delta=f"vs persistence {persist.get('rmse', float('nan')):.3f} m")
            st.caption(
                f"MAE {dl.get('mae', float('nan')):.3f} m · "
                f"{dl.get('n', 0):,} stride rows · {dl.get('n_stations', 0):,} stations · "
                f"direction accuracy {dl.get('dir_acc', float('nan')):.1%} · "
                f"90% interval coverage {pool_eval.get('DL_coverage_90', float('nan')):.1%}"
            )
            if len(ci) == 2:
                st.caption(
                    f"Paired station-bootstrap DL/persistence RMSE ratio "
                    f"{paired.get('ratio', float('nan')):.3f} "
                    f"(95% CI {ci[0]:.3f}–{ci[1]:.3f})."
                )
            st.caption("Saved 2026 stride evaluation snapshot; aggregate JSON only, with no row-level pooled-GRU predictions.")
    else:
        st.info("`outputs/pooled_dl_eval.json` is unavailable.")

with traj_col:
    st.markdown("**Trajectory GRU · 120 six-hourly horizons**")
    if traj_eval:
        rmse = traj_eval.get("30d_rmse_traj")
        persist = traj_eval.get("30d_rmse_persistence")
        with st.container(border=True):
            if rmse is not None and persist is not None:
                st.metric("30-day RMSE", f"{rmse:.3f} m",
                          delta=f"vs persistence {persist:.3f} m")
            st.caption(
                f"{traj_eval.get('anchors_scored', 0):,} anchors · "
                f"{traj_eval.get('stations_sampled', 0):,} sampled stations · "
                f"direction accuracy {traj_eval.get('30d_dir_acc', float('nan')):.1%} · "
                f"calibrated 90% coverage {traj_eval.get('30d_coverage_calibrated', float('nan')):.1%}"
            )
            if not traj_eval.get("promote_trajectory", True):
                st.warning("Trajectory promotion gate is not passed; short-horizon checks remain unmet.")
            st.caption("Saved backtest snapshot; do not compare its RMSE directly with the pooled daily GRU protocol.")
    else:
        st.info("`outputs/traj_backtest_summary_dl.json` is unavailable.")

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
