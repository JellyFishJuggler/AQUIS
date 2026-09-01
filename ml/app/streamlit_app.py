"""
Streamlit dashboard for the XGBoost + quantile-regression forecast pipeline.

Monitors groundwater level (GWL) telemetry and produces point/quantile
forecasts with a 90% prediction interval per station.

Layers (top to bottom):
  1. Header + global station selector (single source of truth)
  2. Selected-station overview KPIs (level, forecast, trend, status)
  3. Station health (condition, thresholds, forecast, interpretation)
  4. Real-time groundwater level time series
  5. Forecast (test-period validation + actual future forecast + interactive)
  6. Model / prediction information (train + metrics + technical details)
  7. Diagnostics & validation (fleet validation summary + collapsible details)
  8. Observations (recent readings + telemetry gap details)
  9. Notes / interpretation (explanatory text)
  10. All stations / data explorer (collapsed by default)

Run:
    streamlit run ml/app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

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
_SNAPSHOT_CSV = _ARTIFACTS / "dashboard_forecasts.csv"

st.set_page_config(
    page_title="GWL Forecast Inspector",
    page_icon=":material/water_drop:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from ml.models.xgboost_quantile import (  # noqa: E402
    POINT_MODEL_FILE,
    get_test_predictions,
    train_xgb_quantile_for_station,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    GWL_COL,
    TIME_COL,
    resolve_slug_dir,
)
from ml.services.forecast_snapshots import (  # noqa: E402
    future_preds_from_df,
    test_preds_from_df,
)
from ml.services.interval_calibration import calibrate_and_widen  # noqa: E402


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_common():
    return pd.read_parquet(_PARQUET_PATH)


@st.cache_data(show_spinner=False)
def _snapshot_df():
    """Precomputed dashboard forecast snapshot (model curves as fallback)."""
    if not _SNAPSHOT_CSV.is_file():
        return None
    return pd.read_csv(
        _SNAPSHOT_CSV, dtype={"found_2026": "Int64", "last_obs_value": "Float64"}
    )


def _weights_present(station: str) -> bool:
    station_dir = resolve_slug_dir(_ARTIFACTS, station)
    return station_dir is not None and (station_dir / POINT_MODEL_FILE).is_file()


@st.cache_data(show_spinner=False)
def load_xgb_metadata(station: str) -> dict | None:
    station_dir = resolve_slug_dir(_ARTIFACTS, station)
    meta_path = station_dir / "xgboost_metadata.json" if station_dir else None
    if meta_path and meta_path.is_file():
        with open(meta_path) as f:
            return json.load(f)
    return None


def _model_version(station: str) -> float:
    station_dir = resolve_slug_dir(_ARTIFACTS, station)
    model_path = station_dir / POINT_MODEL_FILE if station_dir else None
    return model_path.stat().st_mtime if model_path and model_path.is_file() else 0.0


@st.cache_data(show_spinner=False)
def load_test_predictions(station: str, _version: float) -> dict | None:
    try:
        return get_test_predictions(station)
    except FileNotFoundError:
        # Cloud deploys may lack the LFS-shipped weights -> snapshot fallback.
        return test_preds_from_df(_snapshot_df(), station)


@st.cache_data(show_spinner=False)
def load_series(station: str):
    df = load_common()
    return (
        df[df["Station"] == station]
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_presence(_version: float):
    path = _ARTIFACTS / "2026_station_presence.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_stepwise(_version: float):
    path = _ARTIFACTS / "stepwise_comparison.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_diagnosis(_version: float):
    path = _ARTIFACTS / "multistep_diagnosis.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_xgb_summary(_version: float):
    path = _ARTIFACTS / "xgboost_summary.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_decision_support(_version: float):
    path = _ARTIFACTS / "decision_support.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _row_for(df: pd.DataFrame | None, station: str) -> pd.Series | None:
    if df is None or not len(df):
        return None
    match = df[df["station"] == station]
    if not len(match):
        return None
    return match.iloc[0]


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _fmt_num(v, spec: str = ".4f", fallback: str = "—") -> str:
    v = _num(v)
    return fallback if v is None else f"{v:{spec}}"


def _fmt_pct(v, fallback: str = "—") -> str:
    v = _num(v)
    return fallback if v is None else f"{v:.1%}"


# ---------------------------------------------------------------------------
# Status style maps (color + icon + text — never color alone)
# ---------------------------------------------------------------------------
_PRIORITY_STYLE = {
    "PRIORITY": ("#f85149", ":material/error:", "Priority"),
    "MONITOR": ("#d29922", ":material/visibility:", "Monitor"),
    "OK": ("#3fb950", ":material/check_circle:", "OK"),
    "INSUFFICIENT": ("#8b949e", ":material/help:", "Insufficient"),
}
_TREND_ICON = {
    "declining": ":material/trending_down:",
    "rising": ":material/trending_up:",
    "stable": ":material/trending_flat:",
    "indeterminate": ":material/swap_vert:",
}


def _priority_badge(value: str) -> str:
    if value is None or str(value).upper() not in _PRIORITY_STYLE:
        value = "INSUFFICIENT"
    key = _PRIORITY_STYLE[str(value).upper()]
    return f":{key[0]}-badge[{key[1]} {key[2]}]"


# ---------------------------------------------------------------------------
# Decision-support card helpers
# ---------------------------------------------------------------------------
def _station_decision(decision: pd.DataFrame | None, station: str) -> pd.Series | None:
    if decision is None or not len(decision):
        return None
    rows = decision[decision["station"] == station]
    if not len(rows):
        return None
    return rows.iloc[0]


def _level_gauge_html(level_now, critical, caution, proj90, proj180) -> str:
    """Horizontal gauge: deep (critical, red) -> shallow (green).

    Current level and +90d/+180d projections are drawn as dots on the scale,
    with dashed pins for the critical / caution thresholds.
    """
    span = abs(critical - caution) or 1.0
    lo = critical - 0.15 * span
    hi = caution + 0.6 * span

    def pct(v):
        if v <= lo:
            return 2.0
        if v >= hi:
            return 98.0
        return 2.0 + (v - lo) / (hi - lo) * 96.0

    crit_p = pct(critical)
    caut_p = pct(caution)
    grad = (
        f"linear-gradient(90deg,#f85149 0%,#d29922 {crit_p:.1f}%,"
        f"#3fb950 100%)"
    )

    def dot(v, size, color, tooltip, z=4):
        return (
            f'<div style="position:absolute;left:{pct(v):.1f}%;top:6px;'
            f'transform:translateX(-50%);width:{size}px;height:{size}px;'
            f'background:{color};border:2px solid #0d1117;border-radius:50%;'
            f'box-shadow:0 1px 4px rgba(0,0,0,.6);z-index:{z};" '
            f'title="{tooltip}"></div>'
        )

    dots = dot(level_now, 18, "#ffffff", f"current {level_now:.2f} m", z=5)
    if not pd.isna(proj90):
        dots += dot(proj90, 10, "#58a6ff", f"+90d {proj90:.2f} m")
    if not pd.isna(proj180):
        dots += dot(proj180, 10, "#a371f7", f"+180d {proj180:.2f} m")

    bar = (
        f'<div style="height:30px;border-radius:15px;background:{grad};'
        f'position:relative;">'
        f'<div style="position:absolute;left:{crit_p:.1f}%;top:0;bottom:0;'
        f'border-left:2px dashed rgba(255,255,255,.9);z-index:2;"></div>'
        f'<div style="position:absolute;left:{caut_p:.1f}%;top:0;bottom:0;'
        f'border-left:2px dashed rgba(255,255,255,.9);z-index:2;"></div>'
        f"{dots}</div>"
    )
    labels = (
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:12px;color:#9aa4b2;margin-top:3px;">'
        f'<span style="margin-left:{max(crit_p - 2, 0):.1f}%;">'
        f'critical {critical:.2f} m</span>'
        f'<span style="margin-left:{max(caut_p - crit_p - 2, 1):.1f}%;">'
        f'caution {caution:.2f} m</span>'
        f'<span>shallow →</span></div>'
    )
    return f'<div style="margin:8px 0 2px;">{bar}{labels}</div>'


# ---------------------------------------------------------------------------
# Reusable layout components
# ---------------------------------------------------------------------------
def _inject_theme_css() -> None:
    """Small typography polish on top of the GitHub-dark theme.

    Keeps primary values large and secondary text readable without making
    anything smaller — no layout overrides.
    """
    st.markdown(
        """
<style>
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] p {
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 1.2;
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.85rem;
    color: #9aa4b2;
}
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] p {
    font-size: 0.8rem;
}
[data-testid="stCaptionContainer"] p {
    font-size: 0.85rem;
    color: #9aa4b2;
    line-height: 1.4;
}
[data-testid="stMarkdown"] p {
    line-height: 1.5;
}
</style>
""",
        unsafe_allow_html=True,
    )


def header_section(total_stations, data_min, data_max) -> None:
    """Layer 1 — dashboard header + station controls."""
    _inject_theme_css()
    st.title("Groundwater Level Forecast Inspector")
    st.caption(
        "Monitors groundwater level (GWL) telemetry and produces "
        "XGBoost quantile-regression forecasts with a 90% prediction interval "
        "for the selected station."
    )
    meta = st.caption(
        f"{total_stations} stations  ·  NWIC telemetry  ·  "
        f"data window {data_min:%Y-%m} → {data_max:%Y-%m}"
    )


def station_control_bar() -> str:
    """Global filter — the selected station drives every station-specific view.

    Single, searchable station selector (the one source of truth). Status
    pills below summarise the selected station only.
    """
    with st.container(border=True):
        station = st.selectbox(
            "Station — every section below represents this station",
            stations,
            key="station_selector",
        )
        with st.container(horizontal=True):
            dec = _station_decision(decision, station)
            if dec is not None:
                st.markdown(
                    _priority_badge(dec["priority"]), unsafe_allow_html=False
                )
            pres = _row_for(presence, station)
            if pres is not None:
                if bool(pres["found_2026"]):
                    st.markdown(
                        ":green-badge[:material/cloud_done: In 2026 feed]"
                    )
                else:
                    st.markdown(
                        ":orange-badge[:material/cloud_off: Not in 2026 feed]"
                    )
            trained = load_xgb_metadata(station) is not None
            if trained:
                st.markdown(":blue-badge[:material/model_training: Model trained]")
            else:
                st.markdown(
                    ":gray-badge[:material/model_training: Not trained]"
                )
    return station


def _status_tile(dec, level, caution, critical) -> None:
    """Status / priority tile — semantic badge + one-line context."""
    if dec is None:
        with st.container(border=True):
            st.markdown("**Status / priority**")
            st.markdown(":gray-badge[:material/help: No decision row]")
            st.caption("Run decision_support to generate.")
        return
    priority = str(dec["priority"]).upper()
    context = "priority grid not available"
    if level is not None and critical is not None:
        margin = level - critical
        if margin <= 0:
            context = f"{abs(margin):.2f} m below critical"
        elif caution is not None and level <= caution:
            context = f"{caution - level:.2f} m below caution"
        else:
            context = f"{margin:.2f} m above critical"
    with st.container(border=True):
        st.markdown("**Status / priority**")
        st.markdown(_priority_badge(priority))
        st.caption(context)


def _primary_kpis(station_df, dec, sum_row=None) -> None:
    """Overview KPI row — primary numbers for the selected station.

    Answers in order: what is the current level, what is the forecast, is
    the trend improving/worsening, how accurate is the model, and what is
    the status / priority.
    """
    last_gwl = float(station_df[GWL_COL].iloc[-1]) if len(station_df) else None
    if dec is not None and not np.isnan(dec["level_now"]):
        level = _num(dec["level_now"])
    else:
        level = last_gwl
    critical = _num(dec["critical"]) if dec is not None else None
    caution = _num(dec["caution"]) if dec is not None else None

    level_value = _fmt_num(level, ".2f", "—") + " m"
    level_delta = (
        f"critical {critical:.2f} m"
        if critical is not None and level is not None
        else "no thresholds"
    )

    p90 = _num(dec["proj_90d"]) if dec is not None else None
    p180 = _num(dec["proj_180d"]) if dec is not None else None

    direction = None if dec is None else str(dec["trend_direction"]).lower()
    if direction in ("nan", "none", ""):
        direction = None
    if direction is not None and pd.isna(dec["trend_m_yr"]):
        trend_value, trend_delta = direction.title(), "no measurable slope"
    elif direction is not None:
        slope = abs(float(dec["trend_m_yr"]))
        trend_value = f"{slope:.2f} m/yr"
        trend_delta = f"{direction} · last 2 yr"
    else:
        trend_value, trend_delta = "—", "no decision row"

    with st.container(horizontal=True):
        st.metric("Current level", level_value, level_delta, border=True)
        st.metric(
            "90-day forecast",
            _fmt_num(p90, ".2f") + " m",
            f"+180d {_fmt_num(p180, '.2f')} m",
            border=True,
        )
        st.metric("Trend", trend_value, trend_delta, border=True)
        if sum_row is not None and _num(sum_row.get("r2")) is not None:
            _r2 = _num(sum_row["r2"])
            _rmse = _num(sum_row.get("rmse"))
            st.metric(
                "XGB accuracy (R²)",
                f"{_r2:.3f}",
                (
                    f"RMSE {_rmse:.3f} m "
                    if _rmse is not None
                    else None
                ),
                border=True,
                help=(
                    "How accurately the XGBoost model reproduces real readings "
                    "(one step ahead — short range, ~1–14 days — on held-out "
                    "data). R² = 1.0 is perfect, 0.0 = no better than predicting "
                    "the average. RMSE is the typical error in metres. This is "
                    "the reliable headline number; months-ahead forecasts are "
                    "directional only."
                ),
            )
        _status_tile(dec, level, caution, critical)


def _secondary_stats(station_df, meta, sw_row, sum_row, pres_row) -> None:
    """Layer 2b — station data + per-station validation strip."""
    n = len(station_df)
    date_span = (
        (station_df[TIME_COL].max() - station_df[TIME_COL].min()).days if n else 0
    )
    time_diffs = station_df[TIME_COL].diff().dropna()
    gap_mask = time_diffs.dt.total_seconds() / 3600 > GAP_THRESHOLD_HOURS
    n_gaps = int(gap_mask.sum())

    one_r2 = _num(sum_row["r2"]) if sum_row is not None else None
    multi_r2 = _num(sw_row["multi_step_r2"]) if sw_row is not None else None
    multi_cov = (
        _num(sw_row["multi_coverage_90"]) if sw_row is not None else None
    )
    n_records = _num(pres_row["n_records"]) if pres_row is not None else None

    with st.container(border=True):
        st.markdown("**Station data & validation**")
        with st.container(horizontal=True):
            st.metric("Total points", f"{n:,}", border=True)
            st.metric("Date range (days)", f"{date_span:,}", border=True)
            st.metric("Gaps detected", f"{n_gaps:,}", border=True)
            st.metric(
                "One-step R²",
                _fmt_num(one_r2),
                border=True,
                help=(
                    "How well the model predicts 1 step ahead on unseen data. "
                    "1.0 = perfect, 0.0 = no better than the average. "
                    ">0.9 = excellent, <0.5 = weak."
                ),
            )
            st.metric(
                "Multi-step R²",
                _fmt_num(multi_r2),
                border=True,
                help=(
                    "How well the model predicts when it feeds its own "
                    "predictions back in (recursive / long-range). Usually much "
                    "lower than one-step — this is the honest number for "
                    "months-ahead forecasts."
                ),
            )
            st.metric(
                "Multi-step 90% cov.",
                _fmt_pct(multi_cov),
                border=True,
                help=(
                    "Share of real readings that fall inside the model's 90% "
                    "prediction interval over a long recursive forecast. "
                    "Target is 90%; well below that means the interval is "
                    "too narrow to trust."
                ),
            )
            st.metric(
                "2026 records",
                f"{int(n_records):,}" if n_records is not None else "—",
                border=True,
                help=(
                    "Live 2026 readings found via the NWIC API. Data is fetched "
                    "in batches from separate 2021-2025 and 2026-2030 NWIC "
                    "resources — not streamed in real time."
                ),
            )
    st.session_state._n_gaps = n_gaps
    st.session_state._gap_mask = gap_mask
    st.session_state._time_diffs = time_diffs


def status_health_card(dec) -> None:
    """Station health — grouped summary for the currently selected station.

    Selected-station story: CURRENT CONDITION (level, trend, thresholds),
    FORECAST (+90d/+180d with 90% prediction intervals, coverage) and
    INTERPRETATION (narrative). All values come from decision_support.csv.
    """
    st.subheader("Station health")
    if dec is None:
        with st.container(border=True):
            st.info(
                "No decision-support row for this station. Run "
                "`python -m ml.scripts.decision_support`."
            )
        return

    fg, _, _ = _PRIORITY_STYLE.get(
        str(dec["priority"]).upper(), _PRIORITY_STYLE["INSUFFICIENT"]
    )
    direction = str(dec["trend_direction"]).lower()
    if direction in ("nan", "none", ""):
        direction = "indeterminate"
    trend_icon = _TREND_ICON.get(direction, ":material/swap_vert:")
    if pd.isna(dec["trend_m_yr"]):
        slope_txt = "—"
        trend_txt = f"{trend_icon} Trend {direction}"
    else:
        slope = abs(float(dec["trend_m_yr"]))
        slope_txt = f"{slope:.2f} m/yr"
        trend_txt = (
            f"{trend_icon} Trend {direction} · "
            f"{slope_txt} (last 2 yr)"
        )

    with st.container(border=True):
        st.markdown(f"#### {dec['station']}")
        st.markdown(f"**{_priority_badge(dec['priority'])}**  ·  {trend_txt}")

        margin = float(dec["level_now"]) - float(dec["critical"])

        st.markdown("**Current condition**")
        st.markdown(
            _level_gauge_html(
                float(dec["level_now"]),
                float(dec["critical"]),
                float(dec["caution"]),
                _num(dec["proj_90d"]),
                _num(dec["proj_180d"]),
            ),
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Current level",
            f"{float(dec['level_now']):.2f} m",
            f"{margin:+.2f} m vs critical",
        )
        c2.metric("Trend", slope_txt, direction.title())
        c3.metric("Caution ≤", f"{float(dec['caution']):.2f} m")
        c4.metric("Critical ≤", f"{float(dec['critical']):.2f} m")

        st.markdown("**Forecast**")
        p90, lo90, hi90 = _num(dec["proj_90d"]), _num(dec["lo_90d"]), _num(dec["hi_90d"])
        p180, lo180, hi180 = (
            _num(dec["proj_180d"]),
            _num(dec["lo_180d"]),
            _num(dec["hi_180d"]),
        )
        mc = _num(dec["multi_cov_90"])
        if p90 is None and p180 is None:
            st.caption("Model projections not available for this station.")
        else:
            f1, f2, f3 = st.columns(3)
            f1.metric(
                "+90d forecast",
                _fmt_num(p90, ".2f") + " m",
                (
                    f"90% PI {lo90:.2f}–{hi90:.2f} m"
                    if lo90 is not None and hi90 is not None
                    else None
                ),
            )
            f2.metric(
                "+180d forecast",
                _fmt_num(p180, ".2f") + " m",
                (
                    f"90% PI {lo180:.2f}–{hi180:.2f} m"
                    if lo180 is not None and hi180 is not None
                    else None
                ),
            )
            f3.metric(
                "90% multi-step coverage",
                _fmt_pct(mc),
                help=(
                    "Share of real readings the model's 90% interval actually "
                    "covered over a long recursive forecast. Target 90%; below "
                    "60% the projection is only directional."
                ),
            )

        st.markdown("**Interpretation**")
        if mc is not None and mc < 0.60:
            st.caption(
                f"⚠ Recursive long-horizon forecast is weak — measured "
                f"90% box coverage only {mc:.0%} (< 60%), so treat the "
                f"projections as directional."
            )
        st.markdown(
            f'<div style="background:#161b22;border-left:4px solid {fg};'
            f'padding:10px 14px;border-radius:6px;color:#e6edf3;">'
            f"{dec['narrative']}</div>",
            unsafe_allow_html=True,
        )


def time_series_card(station, station_df, test_preds, dec) -> None:
    """Layer 4 — real-time groundwater level series for the selected station."""
    st.subheader(f"Real-time groundwater level — {station}")
    if not len(station_df):
        st.warning("No telemetry available for this station.")
        return

    show_thr = st.checkbox(
        "Show caution / critical thresholds", value=True, key="show_thr"
    )
    show_gaps = st.checkbox(
        "Highlight telemetry gaps", value=False, key="show_gaps"
    )

    ts = station_df[TIME_COL]
    vals = station_df[GWL_COL]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=vals,
            mode="lines",
            line=dict(width=1.2, color="#58a6ff"),
            name="GWL",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>GWL %{y:.2f} m<extra></extra>",
        )
    )

    if test_preds is not None and len(test_preds["time"]):
        split_time = pd.Timestamp(test_preds["time"][0])
        fig.add_vline(x=split_time, line_dash="dash", line_color="#8b949e")
        fig.add_annotation(
            x=split_time,
            y=1.0,
            yref="paper",
            text="Train | Test boundary",
            showarrow=False,
            yshift=18,
            font=dict(color="#8b949e"),
        )

    if show_thr and dec is not None:
        xmin, xmax = ts.min(), ts.max()
        for y, label, color in [
            (float(dec["caution"]), "Caution", "#d29922"),
            (float(dec["critical"]), "Critical", "#f85149"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[xmin, xmax],
                    y=[y, y],
                    mode="lines",
                    line=dict(color=color, dash="dash", width=1),
                    name=label,
                    hovertemplate=f"{label} {y:.2f} m<extra></extra>",
                )
            )

    if show_gaps:
        time_diffs = station_df[TIME_COL].diff().dropna()
        gap_mask = time_diffs.dt.total_seconds() / 3600 > GAP_THRESHOLD_HOURS
        if gap_mask.any():
            gap_idx = station_df.index[1:][gap_mask.values]
            gap_hours = time_diffs[gap_mask].dt.total_seconds().values
            fig.add_trace(
                go.Scatter(
                    x=station_df.loc[gap_idx, TIME_COL],
                    y=station_df.loc[gap_idx, GWL_COL],
                    mode="markers",
                    customdata=np.stack(
                        [gap_hours], axis=-1
                    ),
                    marker=dict(symbol="x", size=8, color="#f85149"),
                    name=f"Gap ({int(gap_mask.sum())})",
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>gap %{customdata[0]:.0f} h"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water level (m)",
        height=460,
        margin=dict(l=56, r=20, t=52, b=48),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0,
        ),
        hovermode="x",
    )
    st.plotly_chart(fig, width="stretch")

    with st.container(horizontal=True):
        st.metric(
            "Observed min",
            f"{float(vals.min()):.2f} m",
            border=True,
        )
        st.metric(
            "Observed max",
            f"{float(vals.max()):.2f} m",
            border=True,
        )
        st.metric(
            "Latest reading",
            f"{float(vals.iloc[-1]):.2f} m",
            border=True,
        )
        st.metric(
            "Last observed",
            f"{station_df[TIME_COL].iloc[-1]:%Y-%m-%d %H:%M}",
            border=True,
        )


def _test_forecast_figure(test_preds) -> go.Figure:
    """Test-period figure — actual vs model in the held-out window."""
    t = test_preds["time"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=test_preds["actual"],
            mode="lines",
            name="Observed",
            line=dict(color="#58a6ff", width=1.4),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>observed %{y:.2f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=test_preds["point"],
            mode="lines",
            name="Forecast",
            line=dict(color="#f0883e", width=1.6),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>forecast %{y:.2f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([pd.Series(t), pd.Series(t).iloc[::-1]]),
            y=np.concatenate([test_preds["upper"], test_preds["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(240,136,62,0.14)",
            line=dict(width=0),
            name="90% interval",
            hovertemplate="%{x|%Y-%m-%d}<br>interval %{y:.2f} m<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="GWL (m)",
        height=420,
        margin=dict(l=56, r=20, t=36, b=48),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def _presence_facts(pres_row) -> tuple[bool, pd.Timestamp | None]:
    """2026-feed facts for a station: (found_in_2026, last_2026_ts | None)."""
    if pres_row is not None and pd.notna(pres_row.get("found_2026")) and bool(pres_row["found_2026"]):
        ts = pd.to_datetime(pres_row.get("max_ts"), errors="coerce")
        return True, (ts if pd.notna(ts) else None)
    return False, None


@st.cache_data(show_spinner=False)
def _future_forecast(station, mver, pver, today, future_days=90, tail_days=90) -> dict | None:
    """Actual future forecast via the existing recursive pipeline.

    The projection runs continuously from the station's LAST STORED reading
    (zero gap) through ``today`` and forward ~``future_days`` — everything
    past the stored-data archive is model-generated, never observed.  The
    recursive interval is widened (calibrated) so it honestly covers ~90% of
    multi-step outcomes; the point projection is unchanged.
    Returns None when the model is untrained or the station has no data.
    """
    station_df = load_series(station)
    if not len(station_df):
        return None
    pres_row = _row_for(load_presence(pver), station)
    found_2026, feed_ts = _presence_facts(pres_row)

    ts = station_df[TIME_COL]
    first = ts.min()
    stored_end = pd.Timestamp(ts.max())
    start = stored_end.normalize()
    end = pd.Timestamp(today).normalize() + pd.Timedelta(days=future_days)
    future_dates = pd.date_range(start, end, freq="D")
    if not len(future_dates):
        return None

    base_hour = float((start - first).total_seconds() / 3600)
    future_hours = base_hour + np.arange(len(future_dates), dtype=float) * 24.0
    tail = station_df[ts >= stored_end - pd.Timedelta(days=tail_days)]
    try:
        res = calibrate_and_widen(station, future_hours)
    except FileNotFoundError:
        # Cloud deploys may lack the LFS-shipped weights -> snapshot fallback.
        snap = future_preds_from_df(_snapshot_df(), station)
        if snap is None:
            return None
        tail = station_df[ts >= stored_end - pd.Timedelta(days=tail_days)]
        snap["tail_time"] = tail[TIME_COL].reset_index(drop=True)
        snap["tail_gwl"] = tail[GWL_COL].reset_index(drop=True)
        return snap
    except ValueError:
        return None
    return {
        "tail_time": tail[TIME_COL].reset_index(drop=True),
        "tail_gwl": tail[GWL_COL].reset_index(drop=True),
        "stored_end": stored_end,
        "projection_start": start,
        "today": pd.Timestamp(today).normalize(),
        "projection_end": end,
        "future_dates": future_dates,
        "last_obs_value": float(station_df[GWL_COL].iloc[-1]),
        "last_obs_time": stored_end,
        "found_2026": bool(found_2026),
        "feed_ts": feed_ts,
        "point": np.asarray(res["point"]),
        "lower": np.asarray(res["lower"]),
        "upper": np.asarray(res["upper"]),
    }


def _future_forecast_figure(ff) -> go.Figure:
    """Future forecast figure — observed tail + continuous projection.

    The projection starts at the last stored reading (zero gap) and runs
    through "Today" to ~+90d.  Two markers split the story:
    "Stored data ends · Projection starts" and "Today".
    """
    future = ff["future_dates"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ff["tail_time"],
            y=ff["tail_gwl"],
            mode="lines",
            name="Observed (stored)",
            line=dict(color="#58a6ff", width=1.4),
            hovertemplate="%{x|%b %d, %Y}<br>observed %{y:.2f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future,
            y=ff["point"],
            mode="lines",
            name="Projection",
            line=dict(color="#f0883e", width=2.2),
            hovertemplate="%{x|%b %d, %Y}<br>projection %{y:.2f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([pd.Series(future), pd.Series(future).iloc[::-1]]),
            y=np.concatenate([ff["upper"], ff["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(240,136,62,0.14)",
            line=dict(width=0),
            name="90% interval",
            hovertemplate="%{x|%b %d, %Y}<br>interval %{y:.2f} m<extra></extra>",
        )
    )
    span_days = float(
        (future[-1] - ff["tail_time"].iloc[0]).total_seconds() / 86400
    )
    if span_days < 200:
        xaxis = dict(tickformat="%b %d", dtick="M15", showgrid=True)
    else:
        xaxis = dict(tickformat="%b '%y", dtick="M2", showgrid=True)

    fig.add_vline(x=ff["projection_start"], line_dash="dash", line_color="#8b949e")
    fig.add_annotation(
        x=ff["projection_start"],
        y=1.0,
        yref="paper",
        text="Stored data ends \u00b7 Projection starts",
        showarrow=False,
        xanchor="left",
        yshift=22,
        font=dict(color="#8b949e", size=11),
    )
    fig.add_vline(x=ff["today"], line_dash="dash", line_color="#f0883e")
    fig.add_annotation(
        x=ff["today"],
        y=0.99,
        yref="paper",
        text="Today",
        showarrow=False,
        xanchor="right",
        yshift=6,
        font=dict(color="#f0883e", size=11),
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="GWL (m)",
        height=420,
        margin=dict(l=56, r=20, t=44, b=48),
        xaxis=xaxis,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def _render_trust_badge(diag_row) -> None:
    """Per-station honest-forecast badge (reliable / directional / weak).

    Reads the fleet diagnosis (held-out calibrated multi-step coverage) and
    renders a clear, color-coded trust statement for this station's long-range
    forecast.  Quietly does nothing when the row is missing.
    """
    label = str(diag_row.get("label", "")).strip().lower()
    reason = str(diag_row.get("reason", "")).strip()
    cov = diag_row.get("coverage")
    cov_txt = f" {float(cov):.0%}" if cov is not None and not (
        isinstance(cov, str) and cov == ""
    ) else ""
    if label == "reliable":
        st.success(
            f":material/verified: **Reliable forecast**{cov_txt} — calibrated "
            f"multi-step coverage is at target. Short-range is trustworthy; "
            f"long-range bands honestly reflect uncertainty."
        )
    elif label == "directional":
        st.warning(
            f":material/trending_up: **Directional only**{cov_txt} — use for "
            f"trend, not exact levels. {reason.capitalize() if reason else 'Long-range bands are wide.'}"
        )
    elif label == "weak":
        st.error(
            f":material/error: **Weak / data-quality issue** — {reason or 'forecast not reliable for this station.'}"
        )
    else:
        return


def forecast_card(station, station_df, meta, test_preds, dec, artifacts_version, diag_row=None) -> None:
    """Layer 5 — forecasting module (two coordinated panels).

    LEFT:  "Forecast — Test Period"       -> historical backtest (unchanged).
    RIGHT: "Forecast — Next 2–3 Months"   -> continuous model projection from
           the last stored reading through "Today" and ~+90d ahead; the 2026
           stretch is model-generated (no stored 2026 readings locally) and
           clearly explained.
    """
    st.subheader(f"Forecast — {station}")

    if not _weights_present(station) and _SNAPSHOT_CSV.is_file():
        st.caption(
            ":material/cloud_off: Live model weights are not available on this "
            "deployment — showing precomputed snapshot forecasts."
        )

    if diag_row is not None and pd.notna(diag_row.get("label")):
        _render_trust_badge(diag_row)

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**Forecast — Test Period**")
            st.caption("Historical model evaluation")
            if test_preds is None:
                if meta is None:
                    st.info("Train the model first to enable predictions.")
                else:
                    st.info("Test-period predictions not available for this station.")
            else:
                st.plotly_chart(_test_forecast_figure(test_preds), width="stretch")
                if meta:
                    pm = meta["point_metrics"]
                    qm = meta["quantile_metrics"]
                    with st.container(horizontal=True):
                        st.metric(
                            "Test RMSE",
                            f"{pm['rmse']:.4f} m",
                            border=True,
                            help="Typical size of the model's error on unseen "
                            "data, in metres. Lower is better (0.03 m ≈ 3 cm).",
                        )
                        st.metric(
                            "Test MAE",
                            f"{pm['mae']:.4f} m",
                            border=True,
                            help="Average absolute error on unseen data, in "
                            "metres. Lower is better.",
                        )
                        st.metric(
                            "Test R²",
                            f"{pm['r2']:.4f}",
                            border=True,
                            help="Accuracy score: 1.0 = perfect, 0.0 = no "
                            "better than predicting the average.",
                        )
                        st.metric(
                            "90% PI coverage",
                            f"{qm['coverage_90']:.1%}",
                            border=True,
                            help="Share of real readings that actually fell "
                            "inside the model's 90% interval. Target ≈ 90%.",
                        )
                    st.caption(
                        "These test metrics are one-step (each row keeps its "
                        "real lagged data). The long-range projection panel on "
                        "the right is much harder — see the note below."
                    )

    with right:
        with st.container(border=True):
            st.markdown("**Forecast — Next 2–3 Months**")
            st.caption("Actual future forecast")
            ff = _future_forecast(
                station,
                _model_version(station),
                artifacts_version,
                pd.Timestamp.today().normalize(),
            )
            if ff is None:
                if meta is None:
                    st.info("Train the model first to enable predictions.")
                else:
                    st.info("Future forecast unavailable for this station.")
            else:
                st.plotly_chart(_future_forecast_figure(ff), width="stretch")
                last_obs = float(ff["last_obs_value"])
                end_pt = float(ff["point"][-1])
                win_days = int((ff["projection_end"] - ff["projection_start"]).days)
                with st.container(horizontal=True):
                    st.metric("Latest observed", f"{last_obs:.2f} m", border=True)
                    st.metric("Projection days", f"{win_days} days", border=True)
                    st.metric(
                        "Forecast at +90d",
                        f"{end_pt:.2f} m",
                        border=True,
                    )
                    st.metric(
                        "Expected change",
                        f"{end_pt - last_obs:+.2f} m",
                        border=True,
                    )
                    st.metric(
                        "90% PI (horizon)",
                        f"{float(ff['lower'][-1]):.2f}–{float(ff['upper'][-1]):.2f} m",
                        border=True,
                        help=(
                            "Honest 90% band at the far end of this projection, "
                            "widened to reflect the uncertainty that accrues the "
                            "further out the model predicts. The wider it is, the "
                            "more uncertain that +90d level truly is."
                        ),
                    )
                st.caption(
                    ":material/trending_up: **This long-range panel is directional "
                    "trend only — levels are uncertain.** The point line is the "
                    "model's best-guess projection, but individual values far out "
                    "are not reliable; treat them as a trend, not a precise "
                    "prediction. The shaded band is the honest ~90% range."
                )
                st.caption(
                    f"Observed = stored **2021–2025** telemetry; readings end "
                    f"{ff['stored_end']:%Y-%m-%d}. Everything after the dashed "
                    f"line is a **model projection** through {ff['today']:%Y-%m-%d} "
                    f"(+90d). Data is fetched in batches from NWIC's separate "
                    f"2021-2025 and 2026-2030 resources — not streamed in real "
                    "time."
                )
                st.caption(
                    ":material/timeline: **Why the projection can diverge from "
                    "the last reading:** for long-range forecasts the model "
                    "re-uses its own previous predictions as inputs (recursive "
                    "forecasting), so small errors build up step by step. A jump "
                    "right at the dashed line is expected, not a bug."
                )
                if not ff["found_2026"]:
                    st.caption(
                        f"Not found in the 2026 NWIC feed — the projection extends "
                        "from its stored telemetry only."
                    )
                elif ff["feed_ts"] is not None and ff["feed_ts"] < ff["today"]:
                    st.caption(
                        f"The station's 2026 telemetry was last observed "
                        f"{ff['feed_ts']:%Y-%m-%d}; the projection continues to "
                        f"{ff['today']:%Y-%m-%d} by assumption."
                    )


def model_card(station, meta) -> None:
    """Layer 6 — structured model / prediction information."""
    st.subheader("Model & prediction information")
    if meta is None:
        with st.container(border=True):
            st.warning("No trained model exists for this station yet.")
            if st.button("Train Model"):
                with st.spinner(
                    "Training XGBoost + quantile models (a few seconds)..."
                ):
                    result = train_xgb_quantile_for_station(station, verbose=False)
                st.session_state.last_train = {
                    "station": station,
                    "duration": result["duration"],
                    "point_metrics": result["point_metrics"],
                    "quantile_metrics": result["quantile_metrics"],
                }
                st.rerun()
        return

    with st.container(border=True):
        st.success(
            f"XGBoost quantile model trained in {meta['duration']} "
            f"({meta['trained_at']})"
        )
        left, right = st.columns([2, 1])
        with left:
            with st.container(horizontal=True):
                st.metric("Train / test", f"{meta['train_size']:,} / {meta['test_size']:,}", border=True)
                st.metric("Trained at", f"{meta['trained_at']}", border=True)
                st.metric("Duration", f"{meta['duration']}", border=True)
        with right:
            if st.button("Retrain Model"):
                with st.spinner("Training XGBoost quantile models..."):
                    result = train_xgb_quantile_for_station(station, verbose=False)
                st.session_state.last_train = {
                    "station": station,
                    "duration": result["duration"],
                    "point_metrics": result["point_metrics"],
                    "quantile_metrics": result["quantile_metrics"],
                }
                st.rerun()

        pm = meta["point_metrics"]
        qm = meta["quantile_metrics"]
        with st.container(border=True):
            st.markdown("**Test performance (held-out)**")
            st.caption(
                "This is how accurately the model re-creates real readings it "
                "was NOT shown during training. Hover any metric for meaning."
            )
            with st.container(horizontal=True):
                st.metric(
                    "RMSE",
                    f"{pm['rmse']:.4f} m",
                    border=True,
                    help="Root mean squared error — typical size of the model's "
                    "mistake, in metres. Lower is better (0.03 m ≈ 3 cm).",
                )
                st.metric(
                    "MAE",
                    f"{pm['mae']:.4f} m",
                    border=True,
                    help="Mean absolute error — average absolute mistake in "
                    "metres. More intuitive than RMSE. Lower is better.",
                )
                st.metric(
                    "R²",
                    f"{pm['r2']:.4f}",
                    border=True,
                    help="Variance explained: 1.0 = perfect, 0.0 = no better "
                    "than predicting the average. This station's accuracy score.",
                )
                st.metric(
                    "90% PI mean width",
                    f"{qm['mean_interval_width']:.4f} m",
                    border=True,
                    help="Average width of the 90% prediction interval. A real "
                    "reading has 90% chance to fall inside this band.",
                )

        with st.expander("Model details / technical info", icon=":material/tune:"):
            st.markdown("**Features used by the model**")
            st.markdown(
                "`" + "`, `".join(meta["features"]) + "`"
            )
            st.markdown(
                "Forecast = point prediction (XGBoost regressor) plus three "
                "quantile regressors (5th / 50th / 95th percentiles) giving a "
                "**90% prediction interval**. Features are the existing lags / "
                "rolling means already engineered for the pipeline (features "
                "listed above)."
            )
            st.markdown(
                f"**Training data** — {meta['train_size']:,} readings "
                f"(first {meta['train_size']:,} points of the station's "
                f"series), test {meta['test_size']:,}.")
        # ^ Training/test counts come from xgboost_metadata.json.

    if (
        "last_train" in st.session_state
        and st.session_state.last_train.get("station") == station
    ):
        lt = st.session_state.last_train
        st.success(f"Just finished training in {lt['duration']}.")
        with st.expander("Most recent retrain — metrics", icon=":material/query_stats:"):
            st.json(
                {
                    "point_metrics": lt["point_metrics"],
                    "quantile_metrics": lt["quantile_metrics"],
                }
            )


def diagnostics_section(stations_total, xgb_summary, stepwise, presence, diagnosis) -> None:
    """Diagnostics & validation — fleet validation summary + 2026 coverage.

    Summaries are visible by default; the 2026 coverage detail sits behind an
    expander.
    """
    st.subheader("Diagnostics & validation")
    st.caption(
        f"Fleet-wide XGBoost accuracy across all {stations_total} stations — "
        "and which stations still report in the 2026 NWIC feed."
    )

    @st.cache_data(show_spinner=False)
    def classify_r2(r2_values):
        s = pd.Series(r2_values)
        return {
            ">0.9": int((s > 0.9).sum()),
            "0.5–0.9": int(((s > 0.5) & (s <= 0.9)).sum()),
            "0–0.5": int(((s > 0) & (s <= 0.5)).sum()),
            "≤0": int((s <= 0).sum()),
        }

    with st.container(border=True):
        st.markdown("**Validation summary**")
        if xgb_summary is not None and len(xgb_summary):
            buckets = classify_r2(xgb_summary["r2"])
            with st.container(horizontal=True):
                for label, value in buckets.items():
                    st.metric(
                        f"One-step R² {label}",
                        value,
                        border=True,
                        help=(
                            "Number of stations whose one-step accuracy (R²) "
                            "falls in this range. R² near 1 = model matches "
                            "real readings almost perfectly."
                        ),
                    )
            st.caption(
                f"Median one-step accuracy (R²): **{xgb_summary['r2'].median():.4f}**  "
                f"|  Median error (RMSE): **{xgb_summary['rmse'].median():.4f} m**  "
                f"({len(xgb_summary)} stations)"
            )
        else:
            st.info("xgboost_summary.csv not found. Run a batch training first.")

        if stepwise is not None and len(stepwise):
            sw = stepwise.dropna(subset=["one_step_r2", "multi_step_r2"])
            if len(sw):
                with st.container(horizontal=True):
                    st.metric(
                        "Multi-step median R²",
                        f"{sw['multi_step_r2'].median():.4f}",
                        border=True,
                        help=(
                            "Median accuracy for months-ahead forecasts, where "
                            "the model feeds its own predictions back. Expected "
                            "to be much lower than one-step — the honest "
                            "long-range number."
                        ),
                    )
                    st.metric(
                        "Multi-step median 90% coverage",
                        f"{sw['multi_coverage_90'].median():.1%}",
                        border=True,
                        help=(
                            "Share of real readings inside the model's raw 90% "
                            "interval over a long recursive forecast, before "
                            "calibration. This is why calibration is needed; "
                            "see the calibrated coverage below."
                        ),
                    )
                if diagnosis is not None and len(diagnosis):
                    dc = diagnosis["coverage"].dropna()
                    dw = int((diagnosis["label"] == "weak").sum())
                    dd = int((diagnosis["label"] == "directional").sum())
                    dr = int((diagnosis["label"] == "reliable").sum())
                    with st.container(horizontal=True):
                        st.metric(
                            "Calibrated median 90% coverage",
                            f"{dc.median():.1%}",
                            border=True,
                            help=(
                                "Median held-out multi-step coverage AFTER honest "
                                "interval calibration. Target ~90%."
                            ),
                        )
                        st.metric("Reliable stations", f"{dr}", border=True)
                        st.metric("Directional-only", f"{dd}", border=True)
                        st.metric("Weak / data issues", f"{dw}", border=True)
                st.caption(
                    "Long-range (multi-step) forecasts are **directional trend "
                    "only**. The raw 90% band is too narrow for months out; the "
                    "dashboard **calibrates (widens)** it, so it honestly covers "
                    "~90% of real outcomes. Wide bands = genuine uncertainty, "
                    "not a bug. One-step accuracy (short range) is the reliable "
                    "headline number."
                )

    with st.expander("2026 NWIC live-data coverage", icon=":material/cloud:"):
        if presence is not None and len(presence):
            found = presence[presence["found_2026"] == True]  # noqa: E712
            missing = presence[presence["found_2026"] == False]  # noqa: E712
            m1, m2, m3 = st.columns(3)
            m1.metric("Stations found in 2026", len(found))
            m2.metric("Stations missing in 2026", len(missing))
            m3.metric(
                "Total 2026 records (found)",
                f"{int(found['n_records'].sum()):,}",
            )
            if len(missing):
                st.write(
                    "Stations not present in the 2026 NWIC resource "
                    "(telemetry gaps / renames):"
                )
                st.dataframe(
                    missing[["station", "nearest"]].reset_index(drop=True),
                    width="stretch",
                    hide_index=True,
                )
            st.write("Coverage detail (found stations):")
            st.dataframe(
                found[
                    ["station", "n_records", "min_ts", "max_ts", "api_calls"]
                ]
                .sort_values("n_records", ascending=False)
                .reset_index(drop=True),
                width="stretch",
                height=420,
                hide_index=True,
                column_config={
                    "station": st.column_config.TextColumn("Station"),
                    "n_records": st.column_config.NumberColumn("2026 records", format="%d"),
                    "api_calls": st.column_config.NumberColumn("API calls", format="%d"),
                },
            )
        else:
            st.info("2026_station_presence.csv not found. Run "
                    "`python -m ml.scripts.validate_against_2026 --presence` "
                    "first.")


def observations_section(station, station_df, gap_mask, time_diffs, n_gaps) -> None:
    """Layer 8 — recent observations + telemetry gap details."""
    st.subheader(f"Observations — {station}")
    if len(station_df):
        recent = station_df.tail(200)[[TIME_COL, GWL_COL]].rename(
            columns={TIME_COL: "Time", GWL_COL: "GWL (m)"}
        )
        st.markdown(
            f"Showing the most recent **{len(recent):,}** of "
            f"**{len(station_df):,}** readings."
        )
        with st.container(height=300):
            st.dataframe(
                recent.reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "Time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                    "GWL (m)": st.column_config.NumberColumn("GWL (m)", format="%.2f"),
                },
            )
    if n_gaps > 0:
        with st.expander(f"Gap details ({n_gaps} gaps)", icon=":material/event_busy:"):
            gap_indices = station_df.index[1:][gap_mask.values]
            gap_rows = station_df.loc[gap_indices, [TIME_COL]].copy()
            gap_rows["gap_hours"] = time_diffs[gap_mask].dt.total_seconds() / 3600
            st.dataframe(
                gap_rows.rename(columns={TIME_COL: "Time"}).reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "Time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                    "gap_hours": st.column_config.NumberColumn("Gap (hours)", format="%.1f"),
                },
            )
    else:
        st.caption("No telemetry gaps detected (threshold > 9 h).")


def notes_section() -> None:
    """Layer 9 — grouped explanatory text (all existing).

    Reorganizes the dashboard's explanatory material under readable headings
    instead of a single wall of text.
    """
    st.subheader("Notes & interpretation")
    with st.expander("How to read the forecast & validation", icon=":material/science:"):
        st.markdown(
            """
**Short range (~1–14 days) is the reliable headline.** Day-to-day and the
next 1–2 weeks the model is genuinely accurate; that is the number shown in
the overview.

**Long range is directional trend only — levels are uncertain.** Months-ahead,
the model re-uses its own predictions (recursive forecasting) so error
accumulates. The shaded **90% band is widened (calibrated) to be honest**: it
covers ~90% of real outcomes at that horizon, which means it can look wide —
that wide band *is* the uncertainty, not a bug.

The forecast card overlays observed values, the model's point forecast, and
that honest 90% prediction interval. Where a station's calibrated multi-step
coverage is weak, its card is marked **directional only** or **weak**.
"""
        )
    with st.expander("How the priority grid works", icon=":material/rule:"):
        st.markdown(
            """
**critical** = the deepest 10% of the station's own past readings;
**caution** = the deepest 30%. Declining water table into the caution zone →
MONITOR; already below critical (or projected there in 180 days) → PRIORITY.
Trends use a robust Theil–Sen slope over the last 2 years.
"""
        )
    with st.expander("About the metrics", icon=":material/info:"):
        st.markdown(
            """
- **One-step R²** — held-out prediction quality one sample ahead (short
  range, ~1–14 days; each prediction still sees the real readings right
  before it). This is the **reliable** "accuracy" number shown in the KPIs.
- **Multi-step / recursive R² & coverage** — the honest number for long
  forecasts: the model feeds its own predictions back, error accumulates.
  The 90% interval is **calibrated** so it truly covers ~90% of real
  multi-step outcomes (widened accordingly). A station is **reliable**,
  **directional only**, or **weak** based on this held-out coverage and its
  interval width relative to its observed range.
- **RMSE / MAE** — root mean squared / mean absolute error on the held-out
  test set (m). RMSE ≈ typical mistake in metres.
- **90% PI coverage** — share of true held-out points that fall inside the
  model's 90% prediction interval (target ~90%, met after calibration).

**The jump at the forecast start:** in the *Next 2–3 Months* panel the model
must extend from the last stored reading. Its very first projected steps can
diverge from the observed values because beyond the last real point it predicts
from its own outputs (recursive), not from real telemetry. This is normal and
expected — verify against the dashed line marking where data ends.
"""
        )
    with st.expander("Data source & coverage", icon=":material/cloud:"):
        st.markdown(
            """
Telemetry is fetched from the **NWIC (National Water Information Centre)**
API in **batches** — it is not streamed in real time. A one-shot paginated
fetch (~336K records, 93 stations, 6-hourly cadence) is stored locally in
`common.parquet`, and the models are trained on that archive.

NWIC publishes data in **separate time-bucketed resources**: the 
**2021–2025** set used for training, and a distinct **2026–2030** resource.
The 2026 resource is live-checked here only for coverage validation
(which stations still report) — it is deliberately not merged into the
training data, which is why stored readings end in 2025 and everything
beyond is a projection.

The **2026 NWIC coverage** expander shows which stations still appear in the
2026 resource — stations missing there are typically telemetry gaps or renames.
"""
        )


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
df = load_common()

_ARTIFACTS.mkdir(parents=True, exist_ok=True)
_diag_files = [
    _ARTIFACTS / "2026_station_presence.csv",
    _ARTIFACTS / "stepwise_comparison.csv",
    _ARTIFACTS / "xgboost_summary.csv",
    _ARTIFACTS / "decision_support.csv",
    _ARTIFACTS / "multistep_diagnosis.csv",
]
_present_files = [p for p in _diag_files if p.is_file()]
_artifacts_version = (
    max(p.stat().st_mtime for p in _present_files) if _present_files else 0.0
)

presence = load_presence(_artifacts_version)
stepwise = load_stepwise(_artifacts_version)
xgb_summary = load_xgb_summary(_artifacts_version)
decision = load_decision_support(_artifacts_version)
diagnosis = load_diagnosis(_artifacts_version)

stations = sorted(df["Station"].unique())
GAP_THRESHOLD_HOURS = 9.0

# --- Layer 1: header + controls ------------------------------------------
header_section(
    len(stations), df[TIME_COL].min(), df[TIME_COL].max()
)
selected_station = station_control_bar()

station_df = load_series(selected_station)
meta = load_xgb_metadata(selected_station)
test_preds = load_test_predictions(selected_station, _model_version(selected_station))
dec = _station_decision(decision, selected_station)
sw_row = _row_for(stepwise, selected_station)
sum_row = _row_for(xgb_summary, selected_station)
pres_row = _row_for(presence, selected_station)
diag_row = _row_for(diagnosis, selected_station)

# --- Layer 2: overview KPIs ----------------------------------------------
st.subheader(f"Selected station overview — {selected_station}")
_primary_kpis(station_df, dec, sum_row)
_secondary_stats(station_df, meta, sw_row, sum_row, pres_row)

# --- Layer 3: station health (selected station) --------------------------
status_health_card(dec)

# --- Layer 4: real-time time series --------------------------------------
time_series_card(selected_station, station_df, test_preds, dec)

# --- Layer 5: forecast ---------------------------------------------------
forecast_card(selected_station, station_df, meta, test_preds, dec, _artifacts_version, diag_row)

# --- Layer 6: model / prediction information -----------------------------
model_card(selected_station, meta)

# --- Layer 7: diagnostics & validation (fleet) ---------------------------
diagnostics_section(len(stations), xgb_summary, stepwise, presence, diagnosis)

# --- Layer 8: observations -----------------------------------------------
n_gaps = int(st.session_state.get("_n_gaps", 0))
gap_mask = st.session_state.get("_gap_mask")
time_diffs = st.session_state.get("_time_diffs")
if gap_mask is None or time_diffs is None:
    _td = station_df[TIME_COL].diff().dropna()
    gap_mask = _td.dt.total_seconds() / 3600 > GAP_THRESHOLD_HOURS
    n_gaps = int(gap_mask.sum())
    time_diffs = _td
observations_section(selected_station, station_df, gap_mask, time_diffs, n_gaps)

# --- Layer 9: notes -------------------------------------------------------
notes_section()