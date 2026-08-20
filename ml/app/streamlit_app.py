"""
Streamlit visualization app for GPR preprocessing output + prediction overlay.

Run:
    streamlit run ml/app/streamlit_app.py
"""

import json
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

_PARQUET_PATH = _ML_ROOT / "data" / "processed" / "common.parquet"
_ARTIFACTS = _ML_ROOT / "artifacts"

st.set_page_config(page_title="GPR Inspector", layout="wide")
st.title("GPR Preprocessing Inspector")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from ml.utils import format_duration  # noqa: E402
from ml.preprocessing.gpr import (  # noqa: E402
    full_pipeline,
    station_slug,
    GWL_COL,
    TIME_COL,
)
from ml.models.gaussian_process import build_gpr  # noqa: E402


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_common():
    return pd.read_parquet(_PARQUET_PATH)


def load_metadata(station: str) -> dict | None:
    meta_path = _ARTIFACTS / station_slug(station) / "metadata.json"
    if meta_path.is_file():
        with open(meta_path) as f:
            return json.load(f)
    return None


df = load_common()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "training_running" not in st.session_state:
    st.session_state.training_running = False
if "training_result" not in st.session_state:
    st.session_state.training_result = None


# ---------------------------------------------------------------------------
# Station selection
# ---------------------------------------------------------------------------
stations = sorted(df["Station"].unique())
selected_station = st.selectbox("Select Station", stations)

# Filter raw data
station_df = (
    df[df["Station"] == selected_station]
    .sort_values(TIME_COL)
    .reset_index(drop=True)
)

meta = load_metadata(selected_station)


# ---------------------------------------------------------------------------
# Basic stats
# ---------------------------------------------------------------------------
st.subheader("Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Points", f"{len(station_df):,}")
c2.metric(
    "Date Range (days)",
    f"{(station_df[TIME_COL].max() - station_df[TIME_COL].min()).days}",
)

# Gap detection
time_diffs = station_df[TIME_COL].diff().dropna()
GAP_THRESHOLD_HOURS = 9.0
gap_mask = time_diffs.dt.total_seconds() / 3600 > GAP_THRESHOLD_HOURS
n_gaps = int(gap_mask.sum())
c3.metric("Gaps Detected", n_gaps)

if meta:
    c4.metric(
        "Train / Test",
        f"{meta['train_size']:,} / {meta['test_size']:,}",
    )
else:
    c4.metric("Train / Test", "N/A (not trained)")


# ---------------------------------------------------------------------------
# Raw time series plot
# ---------------------------------------------------------------------------
st.subheader(f"Raw GWL Time Series — {selected_station}")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=station_df[TIME_COL],
        y=station_df[GWL_COL],
        mode="lines",
        line=dict(width=0.8),
        name="GWL",
    )
)

# Add train/test split line if we have pipeline data from training
if meta:
    slug = station_slug(selected_station)
    model_dir = _ARTIFACTS / slug
    if (model_dir / "gpr_model.joblib").is_file():
        data = full_pipeline(_PARQUET_PATH, station=selected_station)
        station_df_proc = data["station_df"]
        split_time = station_df_proc[TIME_COL].values[len(data["X_train"])]
        fig.add_vline(
            x=split_time,
            line_dash="dash",
            line_color="red",
            annotation_text="Train | Test boundary",
        )

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Water Level (m)",
    height=400,
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Train Model — live status panel
# ---------------------------------------------------------------------------
st.subheader("Train Model")


def _run_training(result_holder: dict) -> None:
    """Background thread: fit GPR, evaluate, save model, update holder."""
    data = result_holder["data"]
    gpr = build_gpr(data["x_scaler"], n_restarts=5)

    try:
        gpr.fit(data["X_train_scaled"], data["y_train_scaled"])
    except Exception as e:
        result_holder["error"] = str(e)
        result_holder["done"] = True
        return

    # Predict + evaluate
    y_scaler = data["y_scaler"]
    y_train_pred_sc, _ = gpr.predict(data["X_train_scaled"], return_std=True)
    y_test_pred_sc, std_test_sc = gpr.predict(
        data["X_test_scaled"], return_std=True
    )

    y_train_pred = y_scaler.inverse_transform(
        y_train_pred_sc.reshape(-1, 1)
    ).ravel()
    y_test_pred = y_scaler.inverse_transform(
        y_test_pred_sc.reshape(-1, 1)
    ).ravel()

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    train_rmse = float(np.sqrt(mean_squared_error(data["y_train"], y_train_pred)))
    train_mae = float(mean_absolute_error(data["y_train"], y_train_pred))
    train_r2 = float(r2_score(data["y_train"], y_train_pred))
    test_rmse = float(np.sqrt(mean_squared_error(data["y_test"], y_test_pred)))
    test_mae = float(mean_absolute_error(data["y_test"], y_test_pred))
    test_r2 = float(r2_score(data["y_test"], y_test_pred))

    # Save per-station artifacts
    slug = station_slug(data["station"])
    out_dir = _ARTIFACTS / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gpr, out_dir / "gpr_model.joblib")
    joblib.dump(data["x_scaler"], out_dir / "x_scaler.pkl")
    joblib.dump(data["y_scaler"], out_dir / "y_scaler.pkl")

    from datetime import datetime, timezone

    metadata = {
        "station": data["station"],
        "slug": slug,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(data["X_train_scaled"]),
        "test_size": len(data["X_test_scaled"]),
        "total_points": len(data["X_train_scaled"]) + len(data["X_test_scaled"]),
        "train_metrics": {
            "rmse": train_rmse, "mae": train_mae, "r2": train_r2
        },
        "test_metrics": {
            "rmse": test_rmse, "mae": test_mae, "r2": test_r2
        },
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_),
        "kernel": str(gpr.kernel_),
        "fit_duration": format_duration(time.monotonic() - result_holder["start"]),
        "n_restarts": 5,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    result_holder["done"] = True
    result_holder["metrics"] = {
        "Train RMSE": train_rmse,
        "Train MAE": train_mae,
        "Train R2": train_r2,
        "Test RMSE": test_rmse,
        "Test MAE": test_mae,
        "Test R2": test_r2,
    }
    result_holder["duration"] = format_duration(
        time.monotonic() - result_holder["start"]
    )
    result_holder["log_marginal_likelihood"] = float(
        gpr.log_marginal_likelihood_value_
    )


if st.session_state.training_result is not None:
    res = st.session_state.training_result
    if "error" in res:
        st.error(f"Training failed: {res['error']}")
    else:
        st.success(f"Training complete — {res['duration']}")
        st.caption(
            f"Log-marginal-likelihood: {res['log_marginal_likelihood']:.2f}"
        )
        st.json(res["metrics"])

    if st.button("Retrain"):
        st.session_state.training_result = None
        st.rerun()

elif st.session_state.training_running:
    result_holder = st.session_state._fit_result

    if result_holder["done"]:
        st.session_state.training_running = False
        st.session_state.training_result = result_holder
        st.rerun()
    else:
        elapsed = time.monotonic() - result_holder["start"]
        status_area = st.empty()
        with status_area.container():
            st.write("**Training Gaussian Process...**")
            st.write(f"Station: {result_holder['station']}")
            st.write(f"Elapsed: {format_duration(elapsed)}")
        time.sleep(1)
        st.rerun()

else:
    if st.button("Train GPR Model"):
        data = full_pipeline(_PARQUET_PATH, station=selected_station)

        result_holder = {
            "done": False,
            "start": time.monotonic(),
            "station": data["station"],
            "data": data,
            "metrics": None,
            "duration": None,
            "error": None,
        }
        st.session_state._fit_result = result_holder
        st.session_state.training_running = True

        thread = threading.Thread(
            target=_run_training, args=(result_holder,), daemon=True
        )
        thread.start()
        st.rerun()


# ---------------------------------------------------------------------------
# GPR Prediction overlay (test period, from saved model)
# ---------------------------------------------------------------------------
slug = station_slug(selected_station)
model_path = _ARTIFACTS / slug / "gpr_model.joblib"

if model_path.exists() and meta is not None:
    st.subheader("GPR Prediction — Test Period")

    @st.cache_resource
    def load_gpr_model(s):
        d = _ARTIFACTS / s
        gpr = joblib.load(d / "gpr_model.joblib")
        x_sc = joblib.load(d / "x_scaler.pkl")
        y_sc = joblib.load(d / "y_scaler.pkl")
        return gpr, x_sc, y_sc

    gpr_model, x_scaler, y_scaler = load_gpr_model(slug)

    # Re-run pipeline to get test data with scaled features
    data = full_pipeline(_PARQUET_PATH, station=selected_station)
    station_df_proc = data["station_df"]
    n_train = len(data["X_train"])
    test_df = station_df_proc.iloc[n_train:].copy()

    X_test_sc = data["X_test_scaled"]
    y_test_actual = data["y_test"]

    y_pred_sc, std_sc = gpr_model.predict(X_test_sc, return_std=True)
    y_pred = y_scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    y_unc = std_sc * y_scaler.scale_[0]

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=test_df[TIME_COL].values,
            y=y_test_actual,
            mode="lines",
            name="Actual",
            line=dict(color="steelblue", width=1),
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=test_df[TIME_COL].values,
            y=y_pred,
            mode="lines",
            name="GPR Prediction",
            line=dict(color="red", width=1),
        )
    )
    upper = y_pred + 1.96 * y_unc
    lower = y_pred - 1.96 * y_unc
    fig3.add_trace(
        go.Scatter(
            x=pd.concat([test_df[TIME_COL], test_df[TIME_COL].iloc[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(width=0),
            name="95% CI",
        )
    )
    fig3.update_layout(
        xaxis_title="Date",
        yaxis_title="GWL (m)",
        height=400,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Interactive Prediction Panel
# ---------------------------------------------------------------------------
st.subheader("Interactive Prediction")

if model_path.exists():

    @st.cache_resource
    def load_gpr_model_for_pred(s):
        d = _ARTIFACTS / s
        gpr = joblib.load(d / "gpr_model.joblib")
        x_sc = joblib.load(d / "x_scaler.pkl")
        y_sc = joblib.load(d / "y_scaler.pkl")
        return gpr, x_sc, y_sc

    gpr_pred, x_scaler_pred, y_scaler_pred = load_gpr_model_for_pred(slug)

    time_input = st.text_input(
        "Time values (hours since first reading, comma-separated)",
        placeholder="e.g. 1000, 2000, 3000",
    )

    if st.button("Predict"):
        try:
            times = [
                float(t.strip()) for t in time_input.split(",") if t.strip()
            ]
        except ValueError:
            st.error("Invalid input. Enter numeric values separated by commas.")
            times = []

        if times:
            X_in = np.asarray(times, dtype=float).reshape(-1, 1)
            X_in_sc = x_scaler_pred.transform(X_in)
            y_pred_sc, std_sc = gpr_pred.predict(X_in_sc, return_std=True)
            y_pred = y_scaler_pred.inverse_transform(
                y_pred_sc.reshape(-1, 1)
            ).ravel()
            y_std = std_sc * y_scaler_pred.scale_[0]

            cols = st.columns(len(times))
            for col, t, p, s in zip(cols, times, y_pred, y_std):
                col.metric(
                    label=f"t={t:.0f}h",
                    value=f"{p:.4f} m",
                    delta=f"+/- {1.96*s:.4f} m (95% CI)",
                )

            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=times,
                    y=y_pred,
                    mode="markers+text",
                    text=[f"{p:.4f}" for p in y_pred],
                    textposition="top center",
                    error_y=dict(
                        type="data",
                        array=1.96 * y_std,
                        visible=True,
                    ),
                    name="Prediction",
                    marker=dict(size=10, color="red"),
                )
            )
            fig_pred.update_layout(
                xaxis_title="Time (hours since first reading)",
                yaxis_title="Predicted GWL (m)",
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("Train the model first to enable predictions.")


# ---------------------------------------------------------------------------
# Gap details
# ---------------------------------------------------------------------------
if n_gaps > 0:
    with st.expander(f"Gap Details ({n_gaps} gaps)"):
        gap_indices = station_df.index[1:][gap_mask.values]
        gap_rows = station_df.loc[gap_indices, [TIME_COL]].copy()
        gap_rows["gap_hours"] = time_diffs[gap_mask].dt.total_seconds() / 3600
        st.dataframe(gap_rows.reset_index(drop=True), use_container_width=True)
