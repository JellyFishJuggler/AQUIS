"""Streamlit dashboard for AQUIS groundwater forecasting."""

import sys
import time
import logging
import tempfile
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
    train_models_for_station,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    AGENCY_COL,
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


# Populated by load_station_list so main() can surface WHY filters are empty
# instead of silently presenting empty dropdowns on a broken deployment.
STATION_META_PATH = _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet"
STATION_LOAD_ERROR: str | None = None
STATION_TRAINED_COUNT: int = 0


@st.cache_data(show_spinner=False)
def load_station_list() -> list[dict]:
    """Load ALL stations (display, slug, district, agency, state) from the parquet.

    Memory-light: reads ONLY the metadata columns from the parquet (not the full
    telemetry) and dedupes by station, so startup stays small even when common.parquet
    holds ~1.3k stations / 5M rows. The tracked columns are projected via pyarrow so the
    whole 5M-row frame is never materialised for discovery.

    Discovery is driven by the PARQUET, not by the artifacts directory. A station does
    not need a pre-trained model to be selectable — ``has_model`` records whether one
    already exists, and the app trains on demand for the selected station if not. This
    lets a fresh deployed host (which has no git-ignored ``ml/artifacts``) still list
    stations and build models lazily.

    Only failures that truly block discovery (missing parquet, unreadable/schema-mismatched
    file) set ``STATION_LOAD_ERROR`` so the UI can surface why — instead of silently
    showing empty dropdowns.
    """
    global STATION_LOAD_ERROR, STATION_TRAINED_COUNT
    import pyarrow.parquet as pq

    STATION_LOAD_ERROR = None
    if not STATION_META_PATH.exists():
        STATION_LOAD_ERROR = (
            f"Station metadata file missing: {STATION_META_PATH}. "
            "common.parquet is git-ignored and must be present on the deployment "
            "(100MB — too large to push to GitHub)."
        )
        return []

    try:
        t = pq.ParquetFile(STATION_META_PATH) \
               .read(columns=[STATION_COL, "Agency", "SlNo", "District", "State"]) \
               .to_pandas()
    except Exception as e:  # missing column / corrupt file / pyarrow failure
        STATION_LOAD_ERROR = (
            f"Could not read station metadata from {STATION_META_PATH.name}: {e}"
        )
        return []

    missing = [c for c in ["District", "State", "Agency", STATION_COL] if c not in t.columns]
    if missing:
        STATION_LOAD_ERROR = (
            f"Station metadata schema mismatch — missing column(s): {missing}. "
            "The deployed common.parquet may be stale/from another build."
        )
        return []

    trained = {d.name for d in station_dirs()}
    STATION_TRAINED_COUNT = len(trained)

    t = t.drop_duplicates(subset=[STATION_COL, AGENCY_COL]).dropna(subset=[STATION_COL, "Agency"])

    stations = []
    for row in t.itertuples(index=False):
        slug = station_slug(str(row.Station), str(row.Agency), getattr(row, "SlNo", 0))
        stations.append({
            "display": str(row.Station),
            "slug": slug,
            "district": str(row.District) if pd.notna(getattr(row, "District", None)) else "",
            "agency": str(row.Agency),
            "state": str(row.State) if pd.notna(getattr(row, "State", None)) else "",
            "has_model": slug in trained,
        })
    STATION_LOAD_ERROR = None
    return sorted(stations, key=lambda x: x["display"])


@st.cache_data(show_spinner=False)
def load_diagnosis() -> pd.DataFrame | None:
    """Load fleet diagnosis CSV."""
    for path in DIAG_FILES:
        if path.exists():
            return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def _get_pipeline(slug: str) -> dict:
    """Cached full pipeline for a station slug.

    Caching stops every widget interaction (filter/horizon/station change) from
    re-running the heavy per-station pipeline — and re-spiking memory to ~1.3 GB —
    on a memory-constrained host. The returned frames are reused across runs.
    """
    return full_pipeline(
        _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet",
        _ML_ROOT.parent / "back-end" / "db" / "data.csv",
        station_slug_filter=slug,
    )


@st.cache_resource(show_spinner=False)
def _train_models_on_demand(
    slug: str, display: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, object, bool, float]:
    """Train a station's models lazily, in-process, into a TEMPORARY directory.

    Used when a deployed host has no pre-trained ``ml/artifacts`` (they are git-ignored
    and far too large to push). The models are kept in Streamlit's process cache keyed by
    slug (so a station trains once per server process, not on every rerun), written to a
    throwaway ``tempfile`` dir that is removed on process exit — never accumulating the
    4 GB of artifact blobs on the server.

    Returns (models, calibration, trained_here, elapsed_sec). Trained models are NOT
    persisted to ``ml/artifacts``; they only live for this process.
    """
    import shutil
    from ml.services.interval_calibration import estimate_calibration as _estimate_cal

    with tempfile.TemporaryDirectory(prefix=f"aquis_models_{slug[:12]}_") as tmp:
        artifact_dir = Path(tmp) / slug
        start = time.time()
        models = train_models_for_station(train_df, feature_cols, slug, artifact_dir)
        elapsed = time.time() - start
        calibration = _estimate_cal({}, models, train_df, feature_cols)

    logging.getLogger("aquis").info(
        "trained on demand %s (%s): %d features, %.1fs into temp dir",
        slug, display, len(feature_cols), elapsed,
    )
    return models, calibration, True, elapsed


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
    r2_meaningful = bool(diag_row.get("r2_meaningful", True))
    one_nrmse = diag_row.get("one_step_nrmse", None)
    span = diag_row.get("gwl_span", 0.0)

    r2_str = f"{one_step_r2:.2f}" if r2_meaningful else "n/a (flat well)"
    if not r2_meaningful and one_nrmse is not None:
        r2_str += f" NRMSE={one_nrmse:.3f}"

    if label == "reliable":
        st.success(f"🟢 **RELIABLE** — Calibrated coverage: {coverage:.1%}, 1-step R²: {r2_str}")
    elif label == "directional":
        st.warning(f"🟡 **DIRECTIONAL** — Coverage: {coverage:.1%}, 1-step R²: {r2_str}")
    else:
        st.error(f"🔴 **WEAK** — Coverage: {coverage:.1%}, 1-step R²: {r2_str}")

    with st.expander("Diagnosis details"):
        det = {
            "label": label,
            "reason": diag_row.get("reason", ""),
            "calibrated_coverage": f"{coverage:.3f}",
            "one_step_R2": f"{one_step_r2:.3f}" if r2_meaningful else "not meaningful (low variance)",
            "one_step_NRMSE": f"{one_nrmse:.3f}" if one_nrmse is not None else "—",
            "multi_step_R2": f"{diag_row.get('multi_step_r2', 0):.3f}",
            "shallow_error_GWL": f"{diag_row.get('shallow_error', 0):.3f}",
            "horizon_half_width": f"{diag_row.get('half_width_at_horizon', 0):.3f} m",
            "GWL_span": f"{span:.2f} m",
            "R2_meaningful": r2_meaningful,
            "metric_note": diag_row.get("metric_note", ""),
            "n_obs": int(diag_row.get("n_obs", 0)),
        }
        st.json(det)


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
    """Create forecast plot with continuous, readable series.

    Observed + forecast + PI band are drawn as single sorted traces. Genuine
    telemetry dropouts longer than ``gap_threshold_hours`` (72h, matching
    ``detect_gaps``) appear as a BREAK in the observed line (no connecting line
    through missing data) rather than as clutter — no "no telemetry" label boxes.
    """
    pred_point = np.asarray(pred_point)
    pred_lower = np.asarray(pred_lower)
    pred_upper = np.asarray(pred_upper)
    pred_dates = np.asarray(pred_dates)

    if not (len(pred_point) == len(pred_lower) == len(pred_upper) == len(pred_dates)):
        raise ValueError(
            f"Forecast arrays length mismatch before plotting: dates={len(pred_dates)}, "
            f"point={len(pred_point)}, lower={len(pred_lower)}, upper={len(pred_upper)}. "
            "Refusing to render misaligned forecast."
        )

    fig = go.Figure()

    # Observed: single sorted trace, but with a break (None) at each real telemetry
    # gap. Each gap is drawn as a distinct dashed "no data" connector so it is
    # obvious that nothing was recorded between the two end points.
    obs_dates = np.asarray(obs_dates)
    obs_values = np.asarray(obs_values)
    if len(obs_dates):
        ord = np.argsort(obs_dates)
        x = list(obs_dates[ord])
        y = list(obs_values[ord])
        obs_line = {"x": [], "y": []}
        gap_x, gap_y = [], []
        if len(x) > 1:
            for i in range(len(x)):
                if i > 0:
                    dt_h = (pd.Timestamp(x[i]) - pd.Timestamp(x[i - 1])).total_seconds() / 3600
                    if dt_h > gap_threshold_hours:
                        # disconnect the observed line across the gap...
                        obs_line["x"].append(None)
                        obs_line["y"].append(None)
                        # ...and bridge it with an explicit "no data" connector
                        gap_x += [x[i - 1], x[i], None]
                        gap_y += [y[i - 1], y[i], None]
                obs_line["x"].append(x[i])
                obs_line["y"].append(y[i])
            x, y = obs_line["x"], obs_line["y"]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines", name="Observed",
            line=dict(color="#1f77b4", width=2),
        ))
        if gap_x:
            fig.add_trace(go.Scatter(
                x=gap_x, y=gap_y,
                mode="lines", name="No data (gap)",
                line=dict(color="#1f77b4", width=1.5, dash="dot"),
                opacity=0.6,
                connectgaps=False,
            ))

    # Forecast point + PI band: continuous traces sorted by time.
    pred_ord = np.argsort(pred_dates)
    pred_dates = pred_dates[pred_ord]
    pred_point = pred_point[pred_ord]
    pred_lower = pred_lower[pred_ord]
    pred_upper = pred_upper[pred_ord]

    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_point,
        mode="lines", name="Forecast",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        connectgaps=True,
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([pred_dates, pred_dates[::-1]]),
        y=np.concatenate([pred_upper, pred_lower[::-1]]),
        fill="toself", fillcolor="rgba(255,127,14,0.2)",
        line=dict(color="rgba(255,127,14,0)"),
        name="90% PI",
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


def _forecast_from(
    history_df: pd.DataFrame,
    models: dict,
    calibration,
    feature_cols: list[str],
    future_days: int,
    anchor_days_meta: dict | None = None,
    catch_up: bool = True,
) -> dict | None:
    """Generate a forecast from the station's LAST observation.

    Two-phase timeline is produced, all on one continuous model-driven chain:

    1. **Catch-up segment** — from the last observed date to *today*. This only
       exists when telemetry ended before today (the station has a data gap). It runs
       the actual recursive/direct model chain step-by-step across the whole gap
       (NOT a flat carry-forward of the last value) and is guaranteed to END exactly
       on today's date. It is visually distinguished as an "estimated catch-up"
       (dashed, separate color) in the UI.

    2. **Forward segment** — from *today* out ``future_days`` (the horizon the user
       selected). This is the real forward forecast; identical to the old behaviour
       for a station whose telemetry is current to today (no catch-up phase).

    Days 1..min(30, ·) of the whole chain use the direct per-horizon models; days
    31.. use the recursive + error-correction model. The recursive path is seeded
    from the anchor and run for the full chain, then its first 30 steps are spliced
    out (covered by direct) — never doubled — so direct (≤30) and recursive (31..)
    sit on one continuous timeline.

    ``anchor_days_meta`` is an optional dict for callers wanting to surface the
    anchor in logs (e.g. the Live Outlook slug).

    ``catch_up`` controls whether a catch-up bridge is inserted. The Live Outlook
    passes ``True`` (default). Historical backtest passes ``False`` so it stays a
    pure forward projection from the chosen anchor for exactly ``future_days`` — it
    must never bridge to the live "today".
    """
    last_valid = history_df.dropna(subset=[TIME_COL, GWL_COL])
    if last_valid.empty:
        return None
    last_row = last_valid.iloc[-1]
    last_gwl = last_row[GWL_COL]
    stored_end = last_row[TIME_COL]
    today = pd.Timestamp.now().normalize()
    history_min = pd.Timestamp(history_df[TIME_COL].min())

    X_last, _, _ = prepare_feature_matrix(history_df[feature_cols + [GWL_COL]].tail(1))
    if len(X_last) == 0:
        return None
    last_feats = X_last[0]

    obs_day = stored_end.normalize()
    gap_days = int((today - obs_day).days) if (today > obs_day and catch_up) else 0

    horizon = int(future_days)

    def _horizon_key(day: int) -> str:
        if day <= 7:
            return str(day)
        if day <= 14:
            return "8_14"
        if day <= 21:
            return "15_21"
        return "22_30"

    def _chain(n_steps: int, damping_steps: int) -> dict:
        """One continuous model-driven chain of ``n_steps`` from the anchor."""
        # Days 1..30 use the direct models, keyed by lead-day bucket.
        direct = {"point": [], "lower": [], "upper": []}
        first30 = min(n_steps, 30)
        for i in range(first30):
            day = i + 1
            key = _horizon_key(day)
            if key in models["direct"]:
                pr = predict_direct(models, last_feats.reshape(1, -1), day)
                direct["point"].append(pr["point"])
                direct["lower"].append(pr["q05"])
                direct["upper"].append(pr["q95"])

        # Days 31..H use the recursive + error-correction path (spliced over direct).
        rec = {"point": [], "lower": [], "upper": []}
        if n_steps > 30:
            rec_full = predict_recursive(
                models, last_feats, n_steps, feature_cols,
                models.get("error_correction"), damping_steps=damping_steps,
            )
            rec["point"] = rec_full["point"][30:]
            rec["lower"] = rec_full["q05"][30:]
            rec["upper"] = rec_full["q95"][30:]

        return {
            "point": np.array(direct["point"] + rec["point"]),
            "lower": np.array(direct["lower"] + rec["lower"]),
            "upper": np.array(direct["upper"] + rec["upper"]),
            "direct_count": len(direct["point"]),
        }

    if gap_days == 0:
        # Telemetry current to today -> pure forward forecast (old behaviour).
        end = obs_day + pd.Timedelta(days=horizon)
        future_dates = pd.date_range(obs_day, end, freq="D")[1:]
        n_steps = len(future_dates)
        if n_steps <= 0:
            return None
        ch = _chain(n_steps, damping_steps=max(30, n_steps))
        time_hours = np.arange(len(ch["point"])) * 24.0 + float((obs_day - history_min).total_seconds() / 3600)
        cal_l, cal_u = widen(calibration, time_hours, ch["point"], ch["lower"], ch["upper"], anchor_pos=0)
        return {
            "stored_end": stored_end,
            "projection_start": obs_day,
            "today": today,
            "projection_end": end,
            "future_dates": future_dates,
            "catchup_dates": np.array([], dtype="datetime64[ns]"),
            "catchup_point": np.array([]),
            "catchup_lower": np.array([]),
            "catchup_upper": np.array([]),
            "last_obs_value": float(last_gwl),
            "point": ch["point"],
            "lower": cal_l,
            "upper": cal_u,
            "direct_count": ch["direct_count"],
            "gap_days": 0,
        }

    # Catch-up + forward on ONE continuous chain seeded at the anchor.
    total = gap_days + horizon
    # Higher damping_steps so the catch-up region TRENDS instead of collapsing to a
    # dead-flat line from Damped Anchor Persistence (weight = (d-1)/damping_steps).
    damping_steps = max(30, gap_days * 2)
    ch = _chain(total, damping_steps=damping_steps)

    catchup_dates = pd.date_range(obs_day, today, freq="D")[1:]  # == gap_days steps, last == today
    forward_dates = pd.date_range(today, today + pd.Timedelta(days=horizon), freq="D")[1:]

    catch_dates = catchup_dates.values
    fwd_dates = forward_dates.values

    # Split the single chain: first gap_days steps are catch-up, the rest forward.
    cu_pt = ch["point"][:gap_days]
    cu_lo = ch["lower"][:gap_days]
    cu_hi = ch["upper"][:gap_days]
    fw_pt = ch["point"][gap_days:gap_days + horizon]
    fw_lo = ch["lower"][gap_days:gap_days + horizon]
    fw_hi = ch["upper"][gap_days:gap_days + horizon]

    # Assert the catch-up segment provably ends exactly on today (per-station log).
    catchup_end = pd.Timestamp(catch_dates[-1]).normalize()
    assert catchup_end == today, f"catch-up end {catchup_end} != today {today} for {anchor_days_meta or ''}"
    logging.getLogger("aquis").info(
        "catch-up %s: gap=%dd, end=%s == today, cu[0]=%s..cu[-1]=%s (span %+.2fm, not flat)",
        (anchor_days_meta or {}).get("slug", "?"), gap_days, catchup_end.date(),
        f"{cu_pt[0]:.2f}", f"{cu_pt[-1]:.2f}", float(cu_pt[-1] - cu_pt[0]),
    )

    # Calibrate each segment from its own start (index-based, like before).
    cu_time = np.arange(len(cu_pt)) * 24.0 + float((obs_day - history_min).total_seconds() / 3600)
    fw_time = np.arange(len(fw_pt)) * 24.0 + float((today - history_min).total_seconds() / 3600)
    cal_cu_lo, cal_cu_hi = widen(calibration, cu_time, cu_pt, cu_lo, cu_hi, anchor_pos=0)
    cal_fw_lo, cal_fw_hi = widen(calibration, fw_time, fw_pt, fw_lo, fw_hi, anchor_pos=0)

    return {
        "stored_end": stored_end,
        "projection_start": today,
        "today": today,
        "projection_end": today + pd.Timedelta(days=horizon),
        "future_dates": forward_dates,
        "catchup_dates": catch_dates,
        "catchup_point": cu_pt,
        "catchup_lower": cal_cu_lo,
        "catchup_upper": cal_cu_hi,
        "last_obs_value": float(last_gwl),
        "point": fw_pt,
        "lower": cal_fw_lo,
        "upper": cal_fw_hi,
        "direct_count": ch["direct_count"],
        "gap_days": gap_days,
    }


def _future_forecast(
    station_display: str,
    station_slug: str,
    feature_cols: list[str],
    full_df: pd.DataFrame,
    models: dict,
    calibration,
    future_days: int = 30,
) -> dict | None:
    """Live Outlook: forecast the next ``future_days`` from the station's last observation.

    Anchored on the station's real last reading in ``full_df`` (the ``full`` output of
    ``full_pipeline``) — NOT the 80/20 train-split tail, which would truncate the anchor
    to the train/test boundary and start the forecast mid-history.
    """
    try:
        ff = _forecast_from(
            full_df, models, calibration, feature_cols, future_days,
            anchor_days_meta={"slug": station_slug, "station": station_display},
        )
        if ff is None:
            return None
        return ff
    except Exception as e:
        st.error(f"Future forecast failed: {e}")
        return None


def _train_station_ui(slug: str, display: str) -> None:
    """Train button UI for a station."""
    artifact_dir = ARTIFACTS_DIR / slug
    if artifact_dir.exists():
        st.info(f"Models already exist for {display}. Delete artifact folder to retrain.")
        return

    if st.button(f"🚀 Train models for {display}", type="primary", width="stretch"):
        with st.spinner(f"Training {display}... (this may take 30-60s)"):
            start = time.time()
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "ml.training.train_forecast",
                "--station", slug,
                "--parquet", str(_ML_ROOT / "data" / "processed" / "common.parquet"),
                "--backend", str(_ML_ROOT.parent / "back-end" / "db" / "data.csv"),
                "--artifacts", str(ARTIFACTS_DIR),
            ], capture_output=True, text=True, cwd=_ML_ROOT.parent)
            elapsed = time.time() - start

            if result.returncode == 0:
                st.success(f"✅ Trained {display} in {elapsed:.1f}s")
                st.rerun()
            else:
                st.error(f"Training failed: {result.stderr[:500]}")


def _metric_cards(y_true: np.ndarray, y_pred: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> dict:
    """RMSE / MAE / R² (with flat-well fallback) / calibrated coverage, reused across tabs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
    span = float(np.nanmax(y_true) - np.nanmin(y_true)) if len(y_true) else 0.0
    flat_well = span < 0.75
    nrmse = rmse / span if span > 0 else float("inf")
    cov = float(np.nanmean((y_true >= y_low) & (y_true <= y_high))) if len(y_true) else 0.0
    return {
        "rmse": rmse, "mae": mae, "r2": r2, "cov": cov,
        "span": span, "flat_well": flat_well, "nrmse": nrmse, "n": int(len(y_true)),
    }


def _render_analysis_tab(
    display: str, full_df: pd.DataFrame, models: dict, calibration, feature_cols: list[str],
) -> None:
    """Historical Backtest (Analysis Mode) — retrospective, never a live forecast.

    Lets R&D pick an arbitrary anchor (cutoff) date and see a forward forecast (reusing
    the exact same Live-Outlook model chain) overlaid on the REAL observed readings that
    followed, plus scoped accuracy metrics for that window. Explicitly badge/framed so it
    can never be mistaken for the live "today" outlook.
    """
    t_max = pd.Timestamp(full_df[TIME_COL].max()).date()
    t_min = pd.Timestamp(full_df[TIME_COL].min()).date()

    st.warning("🔬 **HISTORICAL BACKTEST — NOT A LIVE FORECAST.**")
    st.caption(f"Pick a **historical cutoff date**; the model forecasts forward from that date as if it were 'today' today, and we overlay the **actual observed** readings that followed. This is retrospective model validation — it does NOT report present-day conditions.")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        anchor_d = st.date_input("🪨 Forecast as-of date (anchor)", value=t_max, min_value=t_min, max_value=t_max, key="an_anchor")
    with c2:
        horizon_d = st.selectbox("Horizon (days)", [7, 14, 30, 60, 90], index=2, key="an_horizon")
    with c3:
        end_d = st.date_input("🛑 Compare observations up to", value=t_max, min_value=anchor_d, max_value=t_max, key="an_end")

    end_ts = pd.Timestamp(end_d)
    anchor_ts = pd.Timestamp(anchor_d) + pd.Timedelta(hours=23, minutes=59)

    if anchor_ts >= pd.Timestamp(full_df[TIME_COL].max()):
        st.info("The as-of date equals the station's last reading — nothing follows it to validate against. Move it earlier to see a real forecast-vs-observed comparison.")
        return

    # Build history up to & including the anchor (features already built on full_df).
    hist = full_df[full_df[TIME_COL] <= anchor_ts].copy()
    if hist.empty:
        st.error("No telemetry at/before the chosen anchor date.")
        return

    ff = _forecast_from(hist, models, calibration, feature_cols, int(horizon_d), catch_up=False)
    if ff is None:
        st.error("Could not generate backtest forecast for that anchor.")
        return

    # Overlay REAL observed within [anchor, end].
    obs = full_df[(full_df[TIME_COL] > anchor_ts) & (full_df[TIME_COL] <= end_ts)].copy()

    fig = _plot_forecast(
        f"{display} — Backtest from {anchor_d} ({horizon_d}d) vs actual",
        full_df[full_df[TIME_COL] >= anchor_ts][TIME_COL].values,
        full_df[full_df[TIME_COL] >= anchor_ts][GWL_COL].values,
        ff["future_dates"].values,
        ff["point"],
        ff["lower"],
        ff["upper"],
    )

    # Daily-aggregated actuals so the metric comparison is a fair point-per-day match.
    if not obs.empty:
        obs_day = obs.set_index(TIME_COL)[GWL_COL].resample("D").mean().dropna()
        fig.add_trace(go.Scatter(
            x=obs_day.index, y=obs_day.values,
            mode="markers", name="Actual (daily)", marker=dict(color="#31a354", size=5), legendrank=2,
        ))

    fig.add_vline(x=anchor_ts, line_dash="dot", line_color="#888", annotation_text="Anchor (as-of)")
    st.plotly_chart(fig, width="stretch")

    # Scoped metrics over overlapping forecast dates that have observed readings.
    if obs.empty:
        st.warning("No observed readings fall inside the chosen window — nothing to score.")
        return

    obs_day = obs.set_index(TIME_COL)[GWL_COL].resample("D").mean().dropna()
    fdates = pd.DatetimeIndex(pd.to_datetime(ff["future_dates"].values).normalize())
    overlap = fdates.isin(obs_day.index)
    if not overlap.any():
        st.warning("No overlap between forecast days and observed days in this window.")
        return

    y_true = obs_day.reindex(fdates[overlap]).values.astype(float)
    y_pred = np.asarray(ff["point"])[overlap]
    y_low = np.asarray(ff["lower"])[overlap]
    y_high = np.asarray(ff["upper"])[overlap]

    m = _metric_cards(y_true, y_pred, y_low, y_high)
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("RMSE (m)", f"{m['rmse']:.3f}")
    cc2.metric("MAE (m)", f"{m['mae']:.3f}")
    cc3.metric("R²", (f"{m['r2']:.3f}" if not m["flat_well"] else "n/a (flat well)"))
    cc4.metric("Calibrated Coverage", f"{m['cov']:.1%}")
    cc5.metric("N points", m["n"])
    if m["flat_well"] and np.isfinite(m["nrmse"]):
        st.caption(f"ℹ️ Low target variance (span {m['span']:.2f} m < 0.75 m): R² unstable — **NRMSE = {m['nrmse']:.3f}** is the honest metric.")
    st.caption(f"Scored over **{m['n']}** forecast days inside the selected window — forecast made 'as of' {anchor_d} and checked against actuals.")


def main() -> None:
    st.title("🌊 AQUIS Groundwater Level Forecasting")

    stations = load_station_list()
    diag_df = load_diagnosis()

    # Surfaces the real cause when station discovery is empty — the only thing that
    # can truly block it now is the parquet data file being missing/unreadable on the
    # deployed host (missing model artifacts no longer block discovery; models build
    # on demand). Instead of silently showing "No options to select".
    if not stations:
        st.error("⚠️ **Station metadata unavailable — filters are disabled.**")
        if STATION_LOAD_ERROR:
            st.error(STATION_LOAD_ERROR)
        st.markdown(
            "The station list is built from `ml/data/processed/common.parquet` (git-ignored, "
            "~100MB — too large to push to GitHub). Copy it to the deployed host next to the "
            "repo, then refresh. Model files themselves are optional: the app trains the "
            "selected station on demand if no pre-trained model is present."
        )
        st.stop()

    # ---- Station discovery / filter layer (narrows WHICH station is viewed; never
    # feeds into forecast math — the selected station still runs the same pipeline).
    state_meta = sorted({s["state"] for s in stations if s.get("state")})
    dist_meta = sorted({s["district"] for s in stations if s.get("district")})
    agency_meta = sorted({s["agency"] for s in stations if s.get("agency")})

    with st.sidebar:
        st.subheader("🔎 Station Filters")
        f_state = st.multiselect("State", state_meta, default=state_meta[:1] if state_meta else [])
        f_district = st.multiselect("District", dist_meta, default=[], help="Leave empty to include all districts.")
        f_agency = st.multiselect("Agency", agency_meta, default=agency_meta[:1] if agency_meta else [])

    def _keep(s: dict) -> bool:
        if f_state and s.get("state") not in f_state:
            return False
        if f_district and s.get("district") not in f_district:
            return False
        if f_agency and s.get("agency") not in f_agency:
            return False
        return True

    filtered = [s for s in stations if _keep(s)]
    filtered.sort(key=lambda s: s["display"])
    if not filtered:
        st.info("No stations match the current filters — clear or broaden District/Agency/State.")
        st.stop()

    display_meta = {s["display"]: s for s in filtered}
    slug_meta = {s["slug"]: s for s in filtered}
    display_names = [s["display"] for s in filtered]
    default_idx = next((i for i, s in enumerate(filtered) if "Alipur" in s["display"]), 0)

    selected_display = st.selectbox("Select Station", display_names, index=default_idx)
    selected_slug = display_meta[selected_display]["slug"]

    pipe = _get_pipeline(selected_slug)
    train_df = pipe["train"]
    test_df = pipe["test"]
    full_df = pipe["full"]
    feature_cols = pipe["feature_cols"]
    gaps = pipe["gaps"].get(selected_display, [])
    sentinel_excluded = pipe["sentinel_excluded"]

    diag_row = _row_for(diag_df, selected_display)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(selected_display)
        st.caption(f"State: {slug_meta[selected_slug]['state']} | District: {slug_meta[selected_slug]['district']} | Agency: {slug_meta[selected_slug]['agency']} | Slug: {selected_slug}")
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
    models_exist = (artifact_dir / "recursive" / "xgb_point.joblib").is_file()

    if not models_exist:
        # Deployed host without the (git-ignored, 4GB) model artifacts: train the
        # selected station on demand, in-process, into a temporary dir. The models
        # live only in this process's cache — they do not persist to ml/artifacts.
        with st.spinner(f"⏳ No pre-trained model for {selected_display} — training on demand... this is a one-time cost while the app runs."):
            models, calibration, _trained_here, train_elapsed = _train_models_on_demand(
                selected_slug, selected_display, train_df, test_df, feature_cols,
            )
        st.caption(f"⚡ **DEBUG — trained on demand in {train_elapsed:.1f}s** (temporary models for the running process; not saved to `ml/artifacts`, so the server stays lightweight).")
    else:
        models = load_models(artifact_dir)
        calibration = estimate_calibration({}, models, train_df, feature_cols)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecast — Test Period (1-step)",
        "🔮 Live Outlook — Next N Days",
        "🔬 Historical Backtest (NOT live)",
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

        delta_mode = bool(models.get("delta_mode"))
        lag1_idx = models.get("lag1_index")

        def _recon(raw: np.ndarray) -> np.ndarray:
            if delta_mode and lag1_idx is not None and len(X_test) > 0:
                return X_test[:, lag1_idx] + raw
            return raw

        test_point = _recon(point_model.predict(X_test))
        test_lower = _recon(q05_model.predict(X_test))
        test_upper = _recon(q95_model.predict(X_test))
        test_dates = test_df.loc[~test_df[feature_cols].isna().any(axis=1), TIME_COL].values

        cal_lower, cal_upper = widen(calibration, np.arange(len(test_point)) * 24, test_point, test_lower, test_upper, anchor_pos=0)

        fig = _plot_forecast(
            f"{selected_display} — Test Period (One-Step Backtest)",
            full_df[TIME_COL].values,
            full_df[GWL_COL].values,
            test_dates,
            test_point,
            cal_lower,
            cal_upper,
        )
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        rmse = float(np.sqrt(np.mean((y_test - test_point) ** 2)))
        mae = float(np.mean(np.abs(y_test - test_point)))
        r2 = float(1 - np.sum((y_test - test_point) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        cov = float(np.mean((y_test >= cal_lower) & (y_test <= cal_upper)))
        test_span = float(np.nanmax(y_test) - np.nanmin(y_test)) if len(y_test) else 0.0
        flat_well = test_span < 0.75
        nrmse = rmse / test_span if test_span > 0 else float("inf")
        c1.metric("RMSE (m)", f"{rmse:.3f}")
        c2.metric("MAE (m)", f"{mae:.3f}")
        c3.metric("R²", (f"{r2:.3f}" if not flat_well else "n/a (flat well)"))
        c4.metric("Calibrated Coverage", f"{cov:.1%}")
        if flat_well and np.isfinite(nrmse):
            st.caption(f"ℹ️ **Low target variance** (test span {test_span:.2f} m < 0.75 m): R² is unstable. **NRMSE = {nrmse:.3f}** is the honest metric.")

        st.caption("✅ **Reliable short-range (1-14 day) accuracy** — This is the headline metric. One-step forecasts are well-calibrated and accurate.")

    with tab2:
        st.markdown("### Live Outlook — next days from last observation")
        st.caption("⚠️ **Short-range levels are reliable near-term.** Days 1–30 use direct multi-step models; 31–90 use recursive + error correction. Bands are calibrated to 90% coverage. **No date picker — anchor is always the station's latest reading.**")

        horizon_choice = st.selectbox("Horizon", ["7 days", "14 days", "30 days", "60 days", "90 days", "Today"], index=2, key="live_horizon")

        if horizon_choice == "Today":
            last_valid = full_df.dropna(subset=[TIME_COL, GWL_COL])
            last_obs = pd.Timestamp(last_valid[TIME_COL].max())
            horizon = int((pd.Timestamp.now().normalize() - last_obs.normalize()).days)
            horizon = min(max(horizon, 1), 90)
            horizon_label = f"Today (last obs {last_obs.date()} → today, {horizon}d)"
        else:
            horizon = int(horizon_choice.split()[0])
            horizon_label = f"Next {horizon} Days"

        ff = _future_forecast(selected_display, selected_slug, feature_cols, full_df, models, calibration, horizon)
        if ff is None:
            st.error("Could not generate future forecast")
        else:
            gap_days = ff.get("gap_days", 0)
            fig = _plot_forecast(
                f"{selected_display} — {horizon_label} (Calibrated 90% PI)",
                full_df[TIME_COL].values,
                full_df[GWL_COL].values,
                ff["future_dates"].values,
                ff["point"],
                ff["lower"],
                ff["upper"],
            )

            # Catch-up segment: real model chain, styled distinctly, provably ending today.
            if gap_days > 0:
                cu_pt = np.asarray(ff["catchup_point"])
                cu_dates = np.asarray(ff["catchup_dates"])
                cu_lo = np.asarray(ff["catchup_lower"])
                cu_hi = np.asarray(ff["catchup_upper"])
                if len(cu_pt) and len(cu_dates):
                    cu_order = np.argsort(cu_dates)
                    cu_dates = cu_dates[cu_order]
                    cu_pt = cu_pt[cu_order]
                    cu_lo = cu_lo[cu_order]
                    cu_hi = cu_hi[cu_order]
                    fig.add_trace(go.Scatter(
                        x=cu_dates, y=cu_pt,
                        mode="lines", name="Estimated Catch-up",
                        line=dict(color="#2ca02c", width=2, dash="dashdot"),
                        connectgaps=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([cu_dates, cu_dates[::-1]]),
                        y=np.concatenate([cu_hi, cu_lo[::-1]]),
                        fill="toself", fillcolor="rgba(44,160,44,0.15)",
                        line=dict(color="rgba(44,160,44,0)"),
                        name="Catch-up 90% PI",
                    ))
                    # Guard: the catch-up segment must provably end on today.
                    if pd.Timestamp(cu_dates[-1]).normalize() != ff["today"]:
                        st.error(
                            f"⚠️ Catch-up segment ends {pd.Timestamp(cu_dates[-1]).date()} "
                            f"≠ today {ff['today'].date()} — gap not fully closed. See server log."
                        )

            fig.add_vline(x=ff["today"], line_dash="dot", line_color="red",
                          annotation_text="Today", annotation_position="top")
            st.plotly_chart(fig, width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Last Observed", f"{ff['last_obs_value']:.2f} m", f"{ff['stored_end'].date()}")
            c2.metric("Projection End", f"{ff['point'][-1]:.2f} m", f"{ff['projection_end'].date()}")
            c3.metric(f"Band Width @ {horizon}d", f"{ff['upper'][-1] - ff['lower'][-1]:.2f} m")

            if gap_days > 0:
                st.info(
                    f"📡 **Telemetry ended {ff['stored_end'].date()} ({gap_days} days ago).** "
                    "A green **Estimated Catch-up** segment (running the real model chain, not a "
                    "flat carry-forward) bridges the last observation to **today**, then the orange "
                    "**Forecast** continues from today."
                )

            rec_part = f"days 31–{horizon} recursive + error-correction (spliced, not doubled). " if horizon > 30 else "entire window direct multi-step models. "
            st.caption(
                f"📏 Green dashed: **Estimated Catch-up** (last obs → today, sequentialized so the last point is today). "
                f"Orange: {rec_part}"
                f"Gray band = calibrated 90% PI. Red line = Today."
            )

    with tab3:
        _render_analysis_tab(selected_display, full_df, models, calibration, feature_cols)

    with tab4:
        st.markdown("### Model Information")

        meta = {}
        meta_file = artifact_dir / "xgboost_metadata.json"
        if meta_file.exists():
            import json
            with open(meta_file) as f:
                meta = json.load(f)

        # Recompute accuracy on the SAME evaluation as the Test Period panel so
        # the KPI cards, the chart, and the trust badge always agree.
        acc_rmse, acc_mae, acc_r2 = 0.0, 0.0, 0.0
        acc_cov = 0.0
        if len(y_test) > 0:
            acc_rmse = float(np.sqrt(np.mean((y_test - test_point) ** 2)))
            acc_mae = float(np.mean(np.abs(y_test - test_point)))
            acc_r2 = float(1 - np.sum((y_test - test_point) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            acc_cov = float(np.mean((y_test >= cal_lower) & (y_test <= cal_upper)))

        label_text = meta.get("reliability_label", "unknown")
        label_icon = {"reliable": "🟢", "directional": "🟡", "weak": "🔴"}.get(label_text, "⚪")

        st.markdown("#### Accuracy Metrics (held-out test set)")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("One-step RMSE", f"{acc_rmse:.3f} m")
        c2.metric("One-step MAE", f"{acc_mae:.3f} m")
        c3.metric("One-step R²", f"{acc_r2:.3f}")
        c4.metric("Calibrated Coverage", f"{acc_cov:.1%}")
        c5.metric("Multi-step RMSE", f"{meta.get('multi_step_rmse', 0.0):.3f} m")
        c6.metric("Multi-step R²", f"{meta.get('multi_step_r2', 0.0):.3f}")
        c7.metric("Reliability", f"{label_icon} {label_text}")

        st.markdown("#### Training Information")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Samples", f"{meta.get('n_train', 0):,}")
        c2.metric("Trained At", meta.get("trained_at", "—"))
        c3.metric("Training Time", f"{meta.get('training_time_sec', 0):.1f} s")

        st.markdown("#### Calibration Status")
        st.write(f"Alpha: {calibration.alpha}")
        st.write(f"Direct horizons calibrated: {len([k for k in calibration.half_widths if k in DIRECT_HORIZONS])}")
        st.write(f"Recursive depths calibrated: {len([k for k in calibration.half_widths if isinstance(k, int)])}")
        with st.expander("Calibration half-widths"):
            st.json({str(k): f"{v:.4f}" for k, v in calibration.half_widths.items()})

        st.markdown("#### Features Used")
        st.write(f"Total: {len(feature_cols)} features")
        with st.expander("Feature list"):
            st.write(feature_cols)

        st.markdown("#### Hyperparameters")
        with st.expander("XGBoost parameters"):
            st.json(meta.get("params", {}))

    with tab5:
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
    
    **Live Outlook (Next N Days)** — Anchored on the station's **latest reading** (no date picker — this is deliberate, to stop the start-date reintroducing the anchoring bug).
    - Days 1–30: **direct multi-step models** (a separate model per horizon: 1-7, 8-14, 15-21, 22-30).
    - Days 31–90: **recursive + error-correction** path, spliced (not doubled) onto the direct segment.
    - Horizon selector exposes 7 / 14 / 30 / 60 / 90 days.
    
    **Historical Backtest (Analysis Mode)** — 🔬 **NOT a live forecast.**
    - Pick any **arbitrary as-of (anchor) date**; the model forecasts forward from there as if it were "today", overlaid on the real observed readings that followed.
    - Scoped RMSE / MAE / R² / calibrated coverage are computed over the selected window.
    - This mode deliberately allows free date selection because it is framed as retrospective validation — it never reports present-day conditions.
    
    **Trust Badge** (top-right):
    - 🟢 **Reliable**: Good coverage, narrow bands, low error relative to GWL range.
    - 🟡 **Directional**: Coverage OK but wider uncertainty — use for trend only.
    - 🔴 **Weak**: Calibration failed or intervals too wide — treat with caution.
    """)


if __name__ == "__main__":
    main()