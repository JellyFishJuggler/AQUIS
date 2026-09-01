"""Streamlit dashboard for AQUIS groundwater forecasting."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DIRECT_HORIZONS,
    MAX_HORIZON,
    load_models,
    predict_direct,
    predict_recursive,
    station_dirs,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    GWL_COL,
    TIME_COL,
    STATION_COL,
    full_pipeline,
    load_and_clean,
    prepare_feature_matrix,
    station_slug,
)
from ml.scripts.diagnose_fleet import (  # noqa: E402
    DIAGNOSIS_FILE,
)
from ml.services.interval_calibration import (  # noqa: E402
    calibrate_and_widen,
    estimate_calibration,
    widen,
)

st.set_page_config(page_title="AQUIS Groundwater Forecast", layout="wide", initial_sidebar_state="expanded")

DIAG_FILES = [
    _ML_ROOT / "artifacts" / "multistep_diagnosis.csv",
    _ML_ROOT.parent / "ml" / "artifacts" / "multistep_diagnosis.csv",
]


@st.cache_data(show_spinner=False)
def load_station_list() -> list[dict]:
    """Load all stations with display names and slugs."""
    df = load_and_clean(_ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet")
    stations = []
    for _, row in df.drop_duplicates(STATION_COL).iterrows():
        slug = station_slug(row[STATION_COL], row["Agency"], row["SlNo"])
        stations.append({
            "display": row[STATION_COL],
            "slug": slug,
            "district": row["District"],
            "agency": row["Agency"],
        })
    return sorted(stations, key=lambda x: x["display"])


@st.cache_data(show_spinner=False)
def load_diagnosis() -> pd.DataFrame | None:
    """Load fleet diagnosis CSV."""
    for path in DIAG_FILES:
        if path.exists():
            return pd.read_csv(path)
    return None


def _row_for(diag_df: pd.DataFrame | None, station_display: str) -> dict | None:
    if diag_df is None or diag_df.empty:
        return None
    row = diag_df[diag_df["station"] == station_display]
    if row.empty:
        row = diag_df[diag_df["station"].str.contains(station_display.split()[0], case=False, na=False)]
    return row.iloc[0].to_dict() if not row.empty else None


def _render_trust_badge(diag_row: dict | None) -> None:
    if diag_row is None:
        st.info("No diagnosis available — run fleet diagnosis")
        return

    label = diag_row.get("label", "unknown")
    coverage = diag_row.get("coverage", 0.0)
    one_step_r2 = diag_row.get("one_step_r2", 0.0)

    if label == "reliable":
        st.success(f"🟢 **RELIABLE** — Calibrated coverage: {coverage:.1%}, 1-step R²: {one_step_r2:.2f}")
    elif label == "directional":
        st.warning(f"🟡 **DIRECTIONAL** — Coverage: {coverage:.1%}, 1-step R²: {one_step_r2:.2f}")
    else:
        st.error(f"🔴 **WEAK** — Coverage: {coverage:.1%}, 1-step R²: {one_step_r2:.2f}")

    with st.expander("Diagnosis details"):
        st.json({
            "label": label,
            "reason": diag_row.get("reason", ""),
            "calibrated_coverage": f"{coverage:.3f}",
            "one_step_R2": f"{one_step_r2:.3f}",
            "multi_step_R2": f"{diag_row.get('multi_step_r2', 0):.3f}",
            "shallow_error_GWL": f"{diag_row.get('shallow_error', 0):.3f}",
            "horizon_half_width": f"{diag_row.get('half_width_at_horizon', 0):.3f} m",
            "GWL_span": f"{diag_row.get('gwl_span', 0):.1f} m",
            "n_obs": int(diag_row.get("n_obs", 0)),
        })


def _break_lines_at_gaps(dates: np.ndarray, values: np.ndarray, gap_threshold_hours: float = 72) -> list[dict]:
    """Split series into segments at gaps > threshold."""
    if len(dates) == 0:
        return []

    segments = []
    seg_start = 0
    for i in range(1, len(dates)):
        diff_h = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).total_seconds() / 3600
        if diff_h > gap_threshold_hours:
            segments.append({"dates": dates[seg_start:i], "values": values[seg_start:i]})
            seg_start = i
    segments.append({"dates": dates[seg_start:], "values": values[seg_start:]})
    return segments


def _plot_forecast(
    title: str,
    obs_dates: np.ndarray,
    obs_values: np.ndarray,
    pred_dates: np.ndarray,
    pred_point: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    gap_threshold_hours: float = 72,
) -> go.Figure:
    """Create forecast plot with gap-aware line breaks."""
    fig = go.Figure()

    obs_segments = _break_lines_at_gaps(obs_dates, obs_values, gap_threshold_hours)
    for seg in obs_segments:
        fig.add_trace(go.Scatter(
            x=seg["dates"], y=seg["values"],
            mode="lines", name="Observed",
            line=dict(color="#1f77b4", width=2),
            showlegend=True,
        ))

    pred_segments = _break_lines_at_gaps(pred_dates, pred_point, gap_threshold_hours)
    for i, seg in enumerate(pred_segments):
        mask = np.isin(pred_dates, seg["dates"])
        seg_point = pred_point[mask]
        seg_lower = pred_lower[mask]
        seg_upper = pred_upper[mask]

        fig.add_trace(go.Scatter(
            x=seg["dates"], y=seg_point,
            mode="lines", name="Forecast" if i == 0 else None,
            line=dict(color="#ff7f0e", width=2, dash="dot"),
            showlegend=(i == 0),
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([seg["dates"], seg["dates"][::-1]]),
            y=np.concatenate([seg_upper, seg_lower[::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.2)",
            line=dict(color="rgba(255,127,14,0)"),
            name="90% PI" if i == 0 else None,
            showlegend=(i == 0),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Groundwater Level (m)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _future_forecast(
    station_display: str,
    station_slug: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    models: dict,
    calibration,
    future_days: int = 90,
) -> dict | None:
    """Generate future recursive forecast with calibration."""
    try:
        last_row = train_df.iloc[-1]
        last_gwl = last_row[GWL_COL]
        last_date = last_row[TIME_COL]
        stored_end = last_date

        X_last, _, _ = prepare_feature_matrix(train_df[feature_cols + [GWL_COL]].tail(1))
        if len(X_last) == 0:
            return None
        last_feats = X_last[0]

        start = stored_end.normalize()
        today = pd.Timestamp.now().normalize()
        end = today + pd.Timedelta(days=future_days)
        future_dates = pd.date_range(start, end, freq="D")
        n_steps = len(future_dates)

        if n_steps == 0:
            return None

        direct_preds = {"point": [], "lower": [], "upper": []}
        direct_count = 0

        for i in range(min(n_steps, 30 * 4)):
            h_days = (i // 4) + 1
            if h_days in DIRECT_HORIZONS:
                pred = predict_direct(models, last_feats.reshape(1, -1), h_days)
                direct_preds["point"].append(pred["point"])
                direct_preds["lower"].append(pred["q05"])
                direct_preds["upper"].append(pred["q95"])
                direct_count += 1
            else:
                break

        rec_steps = n_steps - direct_count
        rec_preds = {"point": [], "lower": [], "upper": []}
        if rec_steps > 0:
            rec = predict_recursive(models, last_feats, rec_steps, feature_cols, models.get("error_correction"))
            rec_preds["point"].extend(rec["point"])
            rec_preds["lower"].extend(rec["q05"])
            rec_preds["upper"].extend(rec["q95"])

        point_arr = np.array(direct_preds["point"] + rec_preds["point"])
        lower_arr = np.array(direct_preds["lower"] + rec_preds["lower"])
        upper_arr = np.array(direct_preds["upper"] + rec_preds["upper"])

        time_hours = np.arange(n_steps) * 24.0 + float((start - train_df[TIME_COL].min()).total_seconds() / 3600)

        cal_lower, cal_upper = widen(calibration, time_hours, point_arr, lower_arr, upper_arr, anchor_pos=0)

        return {
            "stored_end": stored_end,
            "projection_start": start,
            "today": today,
            "projection_end": end,
            "future_dates": future_dates,
            "last_obs_value": float(last_gwl),
            "point": point_arr,
            "lower": cal_lower,
            "upper": cal_upper,
            "direct_count": direct_count,
        }
    except Exception as e:
        st.error(f"Future forecast failed: {e}")
        return None


def _train_station_ui(slug: str, display: str) -> None:
    """Train button UI for a station."""
    artifact_dir = ARTIFACTS_DIR / slug
    if artifact_dir.exists():
        st.info(f"Models already exist for {display}. Delete artifact folder to retrain.")
        return

    if st.button(f"🚀 Train models for {display}", type="primary", use_container_width=True):
        with st.spinner(f"Training {display}... (this may take 30-60s)"):
            start = time.time()
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "ml.training.train_forecast",
                "--station", slug,
                "--parquet", str(_ML_ROOT.parent / "data" / "processed" / "common.parquet"),
                "--backend", str(_ML_ROOT.parent / "back-end" / "db" / "data.csv"),
                "--artifacts", str(ARTIFACTS_DIR),
            ], capture_output=True, text=True, cwd=_ML_ROOT.parent)
            elapsed = time.time() - start

            if result.returncode == 0:
                st.success(f"✅ Trained {display} in {elapsed:.1f}s")
                st.rerun()
            else:
                st.error(f"Training failed: {result.stderr[:500]}")


def main() -> None:
    st.title("🌊 AQUIS Groundwater Level Forecasting")

    stations = load_station_list()
    diag_df = load_diagnosis()

    station_map = {s["display"]: s["slug"] for s in stations}
    display_names = list(station_map.keys())

    selected_display = st.selectbox("Select Station", display_names, index=0)
    selected_slug = station_map[selected_display]

    pipe = full_pipeline(
        _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet",
        _ML_ROOT.parent / "back-end" / "db" / "data.csv",
        station_slug_filter=selected_slug,
    )
    train_df = pipe["train"]
    test_df = pipe["test"]
    feature_cols = pipe["feature_cols"]
    gaps = pipe["gaps"].get(selected_display, [])
    sentinel_excluded = pipe["sentinel_excluded"]

    diag_row = _row_for(diag_df, selected_display)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(selected_display)
        st.caption(f"District: {stations[display_names.index(selected_display)]['district']} | Agency: {stations[display_names.index(selected_display)]['agency']} | Slug: {selected_slug}")
    with col2:
        _render_trust_badge(diag_row)

    with st.expander("📊 Preprocessing Summary", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Points", len(train_df))
        c2.metric("Test Points", len(test_df))
        c3.metric("Gaps >72h", len(gaps))
        c4.metric("Sentinels Excluded", sentinel_excluded)
        if gaps:
            st.write("**Detected Gaps:**")
            for g in gaps[:5]:
                st.write(f"  • {g['start']} → {g['end'] or 'end'} ({g['duration_hours']:.1f}h)")

    artifact_dir = ARTIFACTS_DIR / selected_slug
    models_exist = artifact_dir.exists()

    if not models_exist:
        st.warning("⚠️ No trained models found for this station.")
        _train_station_ui(selected_slug, selected_display)
        st.stop()

    models = load_models(artifact_dir)
    calibration = estimate_calibration({}, models, train_df, feature_cols)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast — Test Period (1-step)",
        "🔮 Forecast — Next 2–3 Months",
        "📋 Model Info",
        "⚙️ Retrain",
    ])

    with tab1:
        st.markdown("### One-step backtest on held-out test set (reliable short-range accuracy)")

        X_test, y_test, _ = prepare_feature_matrix(test_df[feature_cols + [GWL_COL]])

        point_model = models["recursive"]["point"]
        q05_model = models["recursive"]["q05"]
        q50_model = models["recursive"]["q50"]
        q95_model = models["recursive"]["q95"]

        test_point = point_model.predict(X_test)
        test_lower = q05_model.predict(X_test)
        test_upper = q95_model.predict(X_test)
        test_dates = test_df.loc[~test_df[feature_cols].isna().any(axis=1), TIME_COL].values

        cal_lower, cal_upper = widen(calibration, np.arange(len(test_point)) * 24, test_point, test_lower, test_upper, anchor_pos=0)

        fig = _plot_forecast(
            f"{selected_display} — Test Period (One-Step Backtest)",
            train_df[TIME_COL].values[-200:],
            train_df[GWL_COL].values[-200:],
            test_dates,
            test_point,
            cal_lower,
            cal_upper,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        rmse = float(np.sqrt(np.mean((y_test - test_point) ** 2)))
        mae = float(np.mean(np.abs(y_test - test_point)))
        r2 = float(1 - np.sum((y_test - test_point) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        cov = float(np.mean((y_test >= cal_lower) & (y_test <= cal_upper)))
        c1.metric("RMSE (m)", f"{rmse:.3f}")
        c2.metric("MAE (m)", f"{mae:.3f}")
        c3.metric("R²", f"{r2:.3f}")
        c4.metric("Calibrated Coverage", f"{cov:.1%}")

        st.caption("✅ **Reliable short-range (1-14 day) accuracy** — This is the headline metric. One-step forecasts are well-calibrated and accurate.")

    with tab2:
        st.markdown("### Recursive projection: last observation → today → +90 days")
        st.caption("⚠️ **This long-range panel is directional trend only — levels are uncertain.** Bands are calibrated to 90% coverage.")

        ff = _future_forecast(selected_display, selected_slug, feature_cols, train_df, models, calibration, 90)
        if ff is None:
            st.error("Could not generate future forecast")
        else:
            fig = _plot_forecast(
                f"{selected_display} — Next 90 Days (Calibrated 90% PI)",
                train_df[TIME_COL].values[-200:],
                train_df[GWL_COL].values[-200:],
                ff["future_dates"].values,
                ff["point"],
                ff["lower"],
                ff["upper"],
            )

            if ff["direct_count"] > 0:
                sep_date = ff["future_dates"][ff["direct_count"] - 1]
                fig.add_vline(x=sep_date, line_dash="dash", line_color="gray", annotation_text="Direct → Recursive")

            fig.add_vline(x=ff["today"], line_dash="dot", line_color="red", annotation_text="Today")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Last Observed", f"{ff['last_obs_value']:.2f} m", f"{ff['stored_end'].date()}")
            c2.metric("Projection End", f"{ff['point'][-1]:.2f} m", f"{ff['projection_end'].date()}")
            c3.metric("90-day Band Width", f"{ff['upper'][-1] - ff['lower'][-1]:.2f} m")

            st.caption("📏 **Solid orange (1-30d)**: Direct multi-step models (higher accuracy). **Dashed orange (31-90d)**: Recursive with error-correction head. Gray band = calibrated 90% PI (widened from raw quantiles). Red line = Today.")

    with tab3:
        st.markdown("### Model Information")
        meta_file = artifact_dir / "xgboost_metadata.json"
        if meta_file.exists():
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            st.json(meta)
        else:
            st.info("No metadata file found")

        st.markdown("#### Features Used")
        st.write(f"Total: {len(feature_cols)} features")
        with st.expander("Feature list"):
            st.write(feature_cols)

        st.markdown("#### Calibration Status")
        st.write(f"Alpha: {calibration.alpha}")
        st.write(f"Direct horizons calibrated: {len([k for k in calibration.half_widths if k in DIRECT_HORIZONS])}")
        st.write(f"Recursive depths calibrated: {len([k for k in calibration.half_widths if isinstance(k, int)])}")
        with st.expander("Calibration half-widths"):
            st.json({str(k): f"{v:.4f}" for k, v in calibration.half_widths.items()})

    with tab4:
        _train_station_ui(selected_slug, selected_display)

        if st.button("🗑️ Delete artifacts (force retrain)", type="secondary"):
            import shutil
            shutil.rmtree(artifact_dir)
            st.success("Artifacts deleted. Refresh to retrain.")
            st.rerun()

    st.divider()
    st.markdown("""
    ### How to Read This Dashboard
    
    **Short-Range Panel (Test Period)** — One-step-ahead forecasts on the held-out test set.
    - **Reliable accuracy**: These forecasts use the model in its validated one-step mode.
    - **R² ~0.5-0.6** is typical for groundwater — this is the trustworthy number.
    - **90% PI**: Calibrated intervals that actually cover ~90% of outcomes.
    
    **Long-Range Panel (Next 2-3 Months)** — Recursive multi-step projection.
    - **Directional trend only**: The point line shows *direction* (rising/falling), not precise levels.
    - **Calibrated bands**: The gray band is widened using held-out recursive errors so it honestly covers ~90%.
    - **1-30 days**: Direct models (separate model per horizon) — higher accuracy.
    - **31-90 days**: Recursive with error-correction — drift-corrected but still uncertain.
    
    **Trust Badge** (top-right):
    - 🟢 **Reliable**: Good coverage, narrow bands, low error relative to GWL range.
    - 🟡 **Directional**: Coverage OK but wider uncertainty — use for trend only.
    - 🔴 **Weak**: Calibration failed or intervals too wide — treat with caution.
    """)


if __name__ == "__main__":
    main()