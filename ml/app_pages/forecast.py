"""Forecast page — the 6-hourly groundwater trajectory forecast card.

Presents the 120 genuine 6-hour forecast points produced by the multi-horizon
trajectory engine (_trajectory.py) as a single, clean dashboard card on the
app's dark theme. Observed history stops at the anchor; the forecast begins
6 h later and never turns the anchor value into a predicted path. Only the
backend-produced q05/q50/q95 values are drawn — nothing is smoothed, bridged
or interpolated. The trajectory v2 forecast is the only forecast shown on
this page; no other forecast model appears here.

Confidence (HIGH / DIRECTIONAL / LOW) carries an evidence-based reason from
the backend and is surfaced prominently; the per-point confidence levels and
their driver sources live in the tooltip, the detail table and the metrics.

Caching: the trajectory run is cached per (station, data version); the version
fingerprint invalidates exactly when data or weights change (or daily, so the
recency metric stays honest). The exported Snapshot PNG is cached on the same
key.
"""

import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from _model import load_predictions
from _utils import load_table_6h, station_recency

from app_charts import (COL_OBSERVED, COL_TRAJECTORY, CONF_COLORS, CONF_ORDER,
                        trajectory_chart)
from snapshot import trajectory_snapshot_png

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# confidence level -> (label, accent colour, one-line guidance)
CONF_STYLE = {
    "HIGH": ("High confidence", "#34d399",
             "Strengths dominate — interval, direction and drivers all agree."),
    "DIRECTIONAL": ("Directional", "#fbbf24",
                    "Direction is supported; the exact numerical level has higher uncertainty."),
    "LOW": ("Indicative only", "#f472b6",
            "Forecast is indicative only — inspect the evidence before relying on exact values."),
}
DIR_GLYPH = {
    "rising": ("↑", "Rising", "#34d399"),
    "declining": ("↓", "Falling", "#f472b6"),
    "stable": ("→", "Stable", "#fbbf24"),
}
_OBS_TAIL_DAYS = 15


def _traj_version() -> str:
    """Fingerprint of everything the trajectory depends on (models, calibration,
    reliability tables, weather, aligned 6h table) + the calendar day, so the
    cache refreshes exactly when data or weights change (or daily, to keep the
    recency metric honest)."""
    paths = [
        _DATA_DIR / "aligned" / "table_6h.parquet",
        _MODELS_DIR / "traj_config.json",
        _MODELS_DIR / "traj_calibration.json",
        _MODELS_DIR / "traj_reliability.json",
        _MODELS_DIR / "xgb_q50.joblib",
        _MODELS_DIR / "xgb_multihorizon.joblib",
        _MODELS_DIR / "xgb_q05.joblib",
        _MODELS_DIR / "xgb_q95.joblib",
        _DATA_DIR / "cfs" / "openmeteo_weather_daily.parquet",
    ]
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p).encode())
        if p.exists():
            h.update(str(p.stat().st_mtime_ns).encode())
    h.update(pd.Timestamp.now().date().isoformat().encode())
    return h.hexdigest()[:12]


@st.cache_data(show_spinner="6-hourly trajectory run…", max_entries=32)
def _trajectory_cached(station: str, version: str) -> dict:
    from _trajectory import trajectory_forecast
    return trajectory_forecast(station)


@st.cache_data(ttl="1h", max_entries=2)
def _river_forecast_districts() -> tuple[str, ...]:
    """Districts covered by the live CWC river forecast snapshot (if any)."""
    try:
        from refresh.future_drivers import _load_river_forecast

        rfc = _load_river_forecast()
        if rfc is None or rfc.empty:
            return ()
        return tuple(sorted({str(d).upper() for d in rfc["district"]}))
    except Exception:  # noqa: BLE001 - availability display is decorative only
        return ()


def _observed_tail(station, up_to, table6, days: int = _OBS_TAIL_DAYS) -> pd.DataFrame:
    """Daily-mean observed GWL for the last `days` days up to `up_to` (Timestamp)
    — never anything later than the anchor."""
    obs_t = (table6[table6["Station"] == station]
             .dropna(subset=["gwl"]).sort_values("time"))
    window = obs_t[(obs_t["time"] <= up_to)
                   & (obs_t["time"] >= up_to - pd.Timedelta(days=days))].copy()
    tail = window.groupby(window["time"].dt.normalize()).agg(
        date=("time", "last"), gwl=("gwl", "mean"))
    tail = tail[tail["gwl"].notna()]
    return tail.reset_index(drop=True)


@st.cache_data(ttl="1h", max_entries=16)
def _snapshot_png(station: str, version: str) -> bytes:
    """Render the forecast-card PNG bytes via matplotlib (dark theme)."""
    traj = _trajectory_cached(station, version)
    if not traj or "error" in traj:
        return b""
    tail = _observed_tail(station, pd.Timestamp(traj["anchor_time"]), table6)
    return trajectory_snapshot_png(traj=traj, tail=tail, station=station)


st.set_page_config(page_title="AQUIS — forecast", page_icon=":material/troubleshoot:", layout="wide")
st.title("Forecast")
st.caption("30-day groundwater outlook · **120 genuine 6-hour forecast points** · "
           "multi-horizon trajectory model anchored on the latest observed reading")

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #262b3d !important;
        border-radius: 12px;
    }
    .aquis-legend {
        font-family: monospace; font-size: 0.8rem; color: #9aa0a6;
        display: flex; flex-wrap: wrap; gap: 0.9rem; align-items: center;
        margin: 0.4rem 0 0.2rem 0;
    }
    .aquis-legend .chip { display: inline-flex; align-items: center; gap: 0.35rem; }
    .aquis-legend .sw { width: 16px; height: 3px; border-radius: 2px; display: inline-block; }
    .aquis-legend .sw-band { width: 16px; height: 10px; border-radius: 2px; display: inline-block; }
    .aquis-legend .sw-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .aquis-dir {
        font-family: monospace; font-weight: 700; font-size: 1.05rem;
        padding: 0.05rem 0 0.35rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

table6 = load_table_6h()
preds = load_predictions()

recency = station_recency()
stations = [s for s in recency.sort_values(ascending=False).index.astype(str)
            if s in set(preds["Station"].astype(str))]
stations.extend(sorted(set(preds["Station"].astype(str)) - set(stations)))
station = st.selectbox("Station", stations, key="fc_station")
last_t = recency.get(station)
if last_t is not None:
    st.caption(f"Latest reading: **{pd.Timestamp(last_t):%Y-%m-%d %H:%M}** — the forecast anchor")

try:
    from refresh.publish import freshness_for

    _fr = freshness_for(station)
    if _fr.get("published") and _fr.get("forecast_generated"):
        _dot = {"fresh": "🟢", "stale": "🟠", "unknown": "⚪"}.get(_fr.get("data_status"), "⚪")
        st.caption(f"{_dot} Refresh pipeline · data **{_fr['data_status']}** · "
                   f"forecast generated {pd.Timestamp(_fr['forecast_generated']):%Y-%m-%d %H:%M} · "
                   f"model {_fr.get('model_version') or '—'} · "
                   f"anchor {_fr.get('latest_observed') or '—'}"
                   + (f" · {_fr['stale_reason']}" if _fr.get("stale_reason") else ""))
except Exception:  # noqa: BLE001 - freshness is decorative, never break the page
    pass


def _render_trajectory(station, table6):
    try:
        traj = _trajectory_cached(station, _traj_version())
    except Exception as e:  # noqa: BLE001
        st.warning(f"Trajectory engine unavailable: {e}")
        return
    if not traj or "error" in traj:
        st.warning(traj.get("error", "No trajectory available for this station."))
        return

    pts = pd.DataFrame(traj["trajectory"])
    pts["time"] = pd.to_datetime(pts["time"])
    anchor_t = pd.Timestamp(traj["anchor_time"])
    anchor_gwl = float(traj["anchor_gwl"])
    t30 = traj["trajectory_30d"]
    dir_ = traj["direction"]
    oc = traj["overall_confidence"]

    with st.container(border=True):
        # --- header: title + metadata | Snapshot (right-aligned) ------------
        head, snap = st.columns([3, 1], vertical_alignment="center")
        with head:
            st.markdown("#### 30-day groundwater outlook")
            st.caption("120 genuine 6-hour forecast points · **Anchor** "
                       f"{anchor_t:%d %b %Y · %H:%M} · **Forecast** "
                       f"{pts['time'].iloc[0]:%d %b %H:%M} → {pts['time'].iloc[-1]:%d %b %Y}")
        with snap:
            png = _snapshot_png(station, _traj_version())
            if png:
                fname = f"AQUIS_{station.replace(' ', '_')}_forecast_30d.png"
                st.download_button("Snapshot", data=png, file_name=fname,
                                   mime="image/png", width="stretch",
                                   help="Export this forecast card as a PNG image (dark theme).")

        # --- observed history ----------------------------------------------
        tail = _observed_tail(station, anchor_t, table6)

        # --- manual compact legend (no Altair legend) ------------------------
        st.markdown(
            f"""
            <div class="aquis-legend">
              <span class="chip"><span class="sw" style="background:{COL_OBSERVED};"></span>Observed</span>
              <span class="chip"><span class="sw" style="background:{COL_TRAJECTORY};"></span>Forecast (q50)</span>
              <span class="chip"><span class="sw-band" style="background:{COL_TRAJECTORY};opacity:0.3;"></span>90% uncertainty</span>
              <span class="chip">
                <span class="sw-dot" style="background:{CONF_COLORS['HIGH']};opacity:0.55;"></span>HIGH
                <span class="sw-dot" style="background:{CONF_COLORS['DIRECTIONAL']};opacity:0.55;"></span>DIRECTIONAL
                <span class="sw-dot" style="background:{CONF_COLORS['LOW']};opacity:0.55;"></span>LOW
                <span style="color:#666;">· confidence per horizon</span>
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- the chart: observed tail | anchor | genuine 120-point trajectory --
        chart = trajectory_chart(pts=pts, tail=tail, anchor_t=anchor_t, height=480)
        st.altair_chart(chart, width="stretch")

        # --- direction banner ------------------------------------------------
        glyph, dlabel, dcolor = DIR_GLYPH.get(dir_["label"], ("→", dir_["label"], "#fbbf24"))
        sa = dir_.get("sign_accuracy_30d")
        sa_txt = f" · sign-accuracy {sa:.0%}" if sa is not None else ""
        st.markdown(
            f'<div class="aquis-dir" style="color:{dcolor};">'
            f"{glyph} {dlabel} · {dir_['change_q50_30d']:+.2f} m over 30 days{sa_txt}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # --- compact metric cards (6) ----------------------------------------
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Anchor GWL (observed)", f"{anchor_gwl:.2f} m",
                  help="Latest observed reading — the anchor. Never predicted.")
        c2.metric("+24 h", f"{pts['q50'].iloc[4]:.2f} m",
                  help=f"6-hourly median at +24 h · confidence {pts['confidence_level'].iloc[4]}")
        c3.metric("+7 d", f"{pts['q50'].iloc[27]:.2f} m",
                  help=f"6-hourly median at day 7 · confidence {pts['confidence_level'].iloc[27]}")
        c4.metric("+30 d", f"{t30['level']:.2f} m",
                  delta=f"{t30['change']:+.2f} m",
                  help="The genuine 120th 6-hourly forecast point (30 days).")
        c5.metric("30 d change", f"{t30['change']:+.2f} m",
                  delta=f"sign-accuracy {sa:.0%}" if sa is not None else None,
                  help="q50 level at +30 d minus the anchor.")
        lvl = oc["level"]
        _label, _color, _guide = CONF_STYLE.get(lvl, (lvl, "#34d399", ""))
        c6.metric("Confidence", _label,
                  help=(oc.get("reason") or "—") + ((" · " + _guide) if _guide else ""))

        # --- future drivers ---------------------------------------------------
        src = pts["driver_source"].astype(str)
        has_om = any(s.strip().lower().startswith("open-meteo") for s in src)
        has_clim = any(s.strip().lower().startswith("climatology") for s in src)
        any_unavailable = any(s.strip().lower().startswith("driver source unavailable")
                              for s in src)
        dist = None
        dd = table6[table6["Station"] == station]
        if len(dd) and "District" in dd.columns:
            dist = str(dd["District"].iloc[0])
        has_cwc = bool(dist) and dist.upper() in _river_forecast_districts()

        drivers = []
        if has_om:
            drivers.append("**Open-Meteo** forecast · days 1–16")
        if has_cwc:
            drivers.append("**CWC** river forecast · where available (this district)")
        if has_clim:
            drivers.append("**Climatology** · beyond the 16-day forecast limit")
        drivers.append("**Persistence** · final fallback")
        with st.expander("Future drivers"):
            st.markdown("· ".join(drivers))
            if any_unavailable:
                st.caption("A driver source is unavailable for part of this horizon — "
                           "the backend downgrades confidence at those points.")
            else:
                st.caption("Drivers used ahead of the anchor are legitimate forecasts, "
                           "not future observations — the anchor is the latest observed reading.")

        # --- detail table -----------------------------------------------------
        with st.expander("120-point forecast table"):
            detail = pts[["time", "q05", "q50", "q95", "confidence_level", "driver_source"]].copy()
            detail["time"] = detail["time"].dt.strftime("%Y-%m-%d %H:%M")
            detail = detail.rename(columns={
                "time": "timestamp", "q05": "q05 (m)", "q50": "q50 (m)",
                "q95": "q95 (m)", "confidence_level": "confidence", "driver_source": "driver"})
            st.dataframe(detail, width="stretch", hide_index=True)

        # --- confidence & reliability details ---------------------------------
        with st.expander("Confidence & reliability details"):
            counts = {c: 0 for c in CONF_ORDER}
            for c in pts["confidence_level"]:
                if c in counts:
                    counts[c] += 1
            st.caption("Per-point confidence across the 120 horizons: "
                       + " · ".join(f"**{counts[c]} {c}**" for c in CONF_ORDER if counts[c]))
            st.caption(f"**{lvl}** — confident direction: {dir_['label']} "
                       f"({dir_['change_q50_30d']:+.2f} m). {oc.get('reason') or '—'}")
            ev = traj["evidence"]
            st.json({
                "anchor": anchor_t.isoformat(),
                "trajectory_30d": t30,
                "direction": dir_,
                "overall_confidence": oc,
                "station_integrity": ev["station_integrity"],
                "anchor_out_of_range": ev["anchor_ood"],
                "oscillation": ev["stability_oscillation"],
                "recency_days": ev["recency_days"],
                "recent90_coverage": ev["recent90_coverage"],
            })


_render_trajectory(station, table6)