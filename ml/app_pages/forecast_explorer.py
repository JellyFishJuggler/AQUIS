"""Page 2 — Forecast explorer: observed history + the frozen model's output.

Read-only. The optional live path calls the EXISTING inference engine
(``_trajectory.trajectory_forecast``), which builds features from the aligned
table and scores the frozen quantile models. It never trains and writes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import _charts as C
import _data as D


@st.cache_data(ttl="10m", show_spinner=False)
def _stations() -> list[str]:
    return D.station_names()


@st.cache_data(ttl="10m", show_spinner="Reading observed history…")
def _history(station: str, tail_days: int) -> pd.DataFrame | None:
    return D.station_history(station, tail_days=tail_days)


@st.cache_data(ttl="10m", show_spinner="Reading 2026 station observations…")
def _full_history(station: str) -> pd.DataFrame | None:
    return D.station_history(station, tail_days=0)


@st.cache_data(ttl="10m", show_spinner=False)
def _archived(station: str) -> dict | None:
    return D.archived_forecast(station)


@st.cache_data(ttl="5m", show_spinner="Scoring the frozen model (read-only)…")
def _live(station: str) -> dict:
    return D.trajectory_forecast_safe(station)


@st.cache_data(ttl="10m", show_spinner="Reading realized forecast matches…")
def _verification_ledger() -> pd.DataFrame | None:
    return D.verification_ledger()


def _traj_frame(payload: dict) -> pd.DataFrame | None:
    pts = (payload or {}).get("trajectory")
    if not pts:
        return None
    df = pd.DataFrame(pts)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)


@st.cache_data(ttl="10m", show_spinner="Matching archived forecasts to realised readings…")
def _archive_validation(station: str, horizon_h: int = 720) -> pd.DataFrame | None:
    """Issued-vs-realised pairs from the forecast archive.

    The verification ledger only records horizons whose target had already been
    observed when a forecast was first scored, so recent anchors never accumulate
    a 30-day row. The archive is a genuine point-in-time record of what was
    issued, so it can be matched against the reading actually observed then.
    """
    df = D.archive_validation(station, horizon_h=horizon_h)
    return df if df is not None and not df.empty else None


def _render_validation(matched: pd.DataFrame, station: str, daily_median: bool,
                       actual_2026: pd.DataFrame | None = None,
                       label: str = "", model_version: bool = False) -> None:
    """Metrics + chart + detail table for an issued-vs-realised frame.

    Expected columns: `target_time`, `actual`, `q50`, optional `q05`/`q95`,
    optional `anchor_time`/`generated_ts`/`model_version`.
    """
    matched = matched.copy()
    matched["error"] = matched["q50"] - matched["actual"]
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Matched readings", f"{len(matched):,}")
    v2.metric("30-day RMSE", f"{np.sqrt(np.mean(matched['error'] ** 2)):.3f} m")
    v3.metric("30-day MAE", f"{np.mean(np.abs(matched['error'])):.3f} m")
    v4.metric("Mean forecast error", f"{matched['error'].mean():+.3f} m")
    if label:
        st.caption(f"Source: **{label}**. These are 30-day (720 h) targets only.")

    vband = (np.isfinite(matched["q05"]) & np.isfinite(matched["q95"]) &
             np.isfinite(matched["actual"]))
    if vband.any():
        coverage = ((matched.loc[vband, "actual"] >= matched.loc[vband, "q05"]) &
                    (matched.loc[vband, "actual"] <= matched.loc[vband, "q95"])).mean()
        mean_width = (matched.loc[vband, "q95"] - matched.loc[vband, "q05"]).mean()
        b1, b2 = st.columns(2)
        b1.metric("q05–q95 coverage", f"{coverage:.1%}")
        b2.metric("Mean interval width", f"{mean_width:.3f} m")

    if actual_2026 is not None and not actual_2026.empty:
        actual_ser = actual_2026.set_index("time")["gwl"]
        actual_end = actual_2026["time"].max()
        actual_label = "all observed 2026 readings"
    else:
        actual_ser = matched.set_index("target_time")["actual"]
        actual_end = matched["target_time"].max()
        actual_label = "matched observed readings"
    m_idx = matched.set_index("target_time")
    m_idx = m_idx.assign(error=m_idx["q50"] - m_idx["actual"])
    st.plotly_chart(
        C.paired_actual_forecast(
            m_idx, height=460, daily_median=daily_median,
            forecast_col="q50",
            title=f"{station} — issued 30-day forecast vs realised",
        ),
        width="stretch", config={"displayModeBar": False},
    )
    st.caption(
        f"{station} · actual line: {actual_label} through {actual_end:%Y-%m-%d}; "
        f"forecast markers: exact matched 30-day targets from "
        f"{matched['target_time'].min():%Y-%m-%d} to "
        f"{matched['target_time'].max():%Y-%m-%d}. Missing forecast dates are left "
        "blank; no interpolation is used."
    )

    detail_cols = [c for c in ("anchor_time", "target_time", "actual", "q50",
                               "q05", "q95", "error", "model_version",
                               "generated_ts")
                   if c in matched.columns]
    with st.expander("Inspect matched forecast values", expanded=False):
        st.dataframe(matched[detail_cols], width="stretch", hide_index=True)


def _plot(hist: pd.DataFrame | None, traj: pd.DataFrame | None,
          anchor_t=None, title: str = "", daily_median: bool = True):
    """Observed tail + forecast median + calibrated band, on a shared axis.

    Uses the shared Plotly builders in `_charts`, whose y-axis is scaled to the
    data — `st.line_chart` anchors at zero, which flattens a level series that
    sits tens of metres below datum.
    """
    parts = []
    if hist is not None and not hist.empty:
        parts.append(pd.DataFrame({
            "time": hist["time"], "observed": hist["gwl"],
            "q05": None, "q50": None, "q95": None}))
    if traj is not None and not traj.empty:
        parts.append(pd.DataFrame({
            "time": traj["time"], "observed": None,
            "q05": traj["q05"], "q50": traj["q50"], "q95": traj["q95"]}))
    if not parts:
        st.info("Nothing to plot.")
        return
    frame = pd.concat(parts, ignore_index=True).sort_values("time")
    frame = frame.dropna(subset=["time"]).set_index("time")
    for c in ("observed", "q05", "q50", "q95"):
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    if title:
        st.caption(title)

    fig = go.Figure()
    if "observed" in frame.columns and frame["observed"].notna().any():
        o = frame["observed"]
        if daily_median:
            o = o.groupby(pd.to_datetime(o.index).normalize()).median()
        fig.add_trace(go.Scatter(x=o.index, y=o, mode="lines", name="observed",
                                 line=dict(color=C.LEVEL, width=1.6),
                                 connectgaps=False,
                                 hovertemplate="%{x|%d %b %Y %H:%M}<br>observed "
                                               "%{y:.3f} m<extra></extra>"))
    if frame.get("q95") is not None and frame["q95"].notna().any():
        # Upper edge carries no legend entry and the lower edge carries the band's
        # name: a trace named "" renders in the legend as the literal "undefined".
        fig.add_trace(go.Scatter(x=frame.index, y=frame["q95"], mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=frame.index, y=frame["q05"], mode="lines",
                                 name="calibrated 90% interval", line=dict(width=0),
                                 fill="tonexty", fillcolor=C.BAND,
                                 hovertemplate="q05 %{y:.3f} m<extra></extra>"))
        fig.add_trace(go.Scatter(x=frame.index, y=frame["q50"], mode="lines",
                                 name="forecast median",
                                 line=dict(color=C.MEDIAN, width=2.2),
                                 hovertemplate="%{x|%d %b %Y %H:%M}<br>median "
                                               "%{y:.3f} m<extra></extra>"))
    stack = [frame[c] for c in ("observed", "q05", "q95") if c in frame.columns
             and frame[c].notna().any()]
    if stack:
        fig.update_yaxes(range=C._padded_range(pd.concat(stack, ignore_index=True)),
                         title_text="level (m)")
    fig = C._figure(fig, 420)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _band_table(traj: pd.DataFrame) -> pd.DataFrame:
    out = traj.copy()
    out["width"] = out["q95"] - out["q05"]
    keep = [c for c in ("time", "q05", "q50", "q95", "width",
                        "confidence_level", "driver_source", "reliability_reason")
            if c in out.columns]
    return out[keep]


st.title("Forecast explorer")
st.caption(
    "Observed history and the frozen model's output for one station. "
    "No training, no regeneration — artifacts are read as-is."
)

stations = _stations()
if not stations:
    st.error(
        "**Data/artifact unavailable.** No station list could be built: "
        "`ml/models/feature_config.json` and `ml/data/aligned/table_6h.parquet` "
        "are both missing, so there is nothing to select."
    )
    st.stop()

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    station = st.selectbox("Station", stations, index=0)
with c2:
    tail_days = st.selectbox("History window", [90, 180, 400, 1200], index=2)
with c3:
    daily_median = st.toggle("Daily medians", value=True,
                             help="Collapse the 6-hourly series to a calendar-daily "
                                  "median. Raw readings oscillate through the day, "
                                  "which hides the multi-day signal in a long view.")

hist = _history(station, int(tail_days))
if hist is None or hist.empty:
    st.warning(
        f"**Data/artifact unavailable.** `data/aligned/table_6h.parquet` has no "
        f"rows for **{station}**, so the observed tail cannot be shown."
    )
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Observations shown", f"{hist['gwl'].notna().sum():,}")
    m2.metric("Latest reading", f"{hist['time'].max():%Y-%m-%d %H:%M}")
    last = hist["gwl"].dropna()
    m3.metric("Latest GWL", f"{last.iloc[-1]:.2f} m" if len(last) else "—")

# ---- actual-vs-issued forecast validation -----------------------------
st.subheader("2026 actual vs issued 30-day forecast")
actual_2026 = _full_history(station)
if actual_2026 is not None and not actual_2026.empty:
    actual_2026["time"] = pd.to_datetime(actual_2026["time"], errors="coerce")
    actual_2026["gwl"] = pd.to_numeric(actual_2026["gwl"], errors="coerce")
    actual_2026 = actual_2026[
        actual_2026["time"].dt.year.eq(2026) & actual_2026["gwl"].notna()
    ].sort_values("time")
ledger = _verification_ledger()
if ledger is None:
    st.info(
        "**Data/artifact unavailable.** `data/refresh/verification.parquet` "
        "does not contain saved forecasts matched to realized readings."
    )
    if actual_2026 is not None and not actual_2026.empty:
        obs = actual_2026.set_index("time")["gwl"].to_frame("actual GWL")
        st.plotly_chart(
            C.series(obs, height=400, daily_median=True, y_title="level (m)",
                     title="Observed 2026 level (daily medians)"),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption(
            f"{station} · observed 2026 values through "
            f"{actual_2026['time'].max():%Y-%m-%d}; forecast artifact unavailable."
        )
else:
    ledger["station"] = ledger["station"].astype(str)
    ledger["target_time"] = pd.to_datetime(ledger["target_time"], errors="coerce")
    ledger["anchor_time"] = pd.to_datetime(ledger["anchor_time"], errors="coerce")
    ledger["generated_ts"] = pd.to_datetime(ledger["generated_ts"], errors="coerce")
    ledger["horizon_h"] = pd.to_numeric(ledger["horizon_h"], errors="coerce")
    for col in ("actual", "q05", "q50", "q95"):
        ledger[col] = pd.to_numeric(ledger[col], errors="coerce")

    matched = ledger[
        (ledger["station"] == str(station)) &
        (ledger["horizon_h"] == 720) &
        ledger["target_time"].dt.year.eq(2026)
    ].dropna(subset=["target_time", "actual", "q50"]).copy()

    if matched.empty:
        # The ledger only records horizons whose target had ALREADY been observed
        # when the forecast was first scored, so recent anchors never get a 30-day
        # row. The forecast archive is a genuine point-in-time record of what was
        # issued, so validate against that instead.
        arch = _archive_validation(station)
        if arch is not None and not arch.empty:
            st.info(
                "The verification ledger holds no 30-day rows for this station — it "
                "only records horizons realised by the time a forecast was first "
                "scored. Falling back to **archived issued forecasts**, which is the "
                "honest point-in-time record."
            )
            _render_validation(arch, station, daily_median,
                               label="archived issued forecasts")
        else:
            st.info(
                f"No 30-day validation available for **{station}**: the verification "
                "ledger has no matching horizon, and the forecast archive holds no "
                "generation whose 30-day target coincides with an observed reading. "
                "No values are interpolated or filled."
            )
            if actual_2026 is not None and not actual_2026.empty:
                obs = actual_2026.set_index("time")["gwl"].to_frame("actual GWL")
                st.plotly_chart(
                    C.series(obs, height=400, daily_median=True, y_title="level (m)",
                             title="Observed 2026 level (daily medians)"),
                    width="stretch", config={"displayModeBar": False},
                )
    else:
        # A target may have been forecast in more than one published generation.
        # Keep the latest issued value per target timestamp for a readable series.
        matched = (matched.sort_values("generated_ts")
                   .drop_duplicates(subset=["target_time"], keep="last")
                   .sort_values("target_time"))
        _render_validation(matched, station, daily_median,
                           actual_2026=actual_2026, label="verification ledger",
                           model_version=True)

# ---- frozen-model scoring (optional, explicit) ------------------------
st.subheader("Frozen model output")
ok, why = D.inference_available()
if not ok:
    st.info(
        f"**Inference unavailable — {why}.** Showing archived/observed data only. "
        "Nothing is retrained to work around this."
    )
    traj = None
else:
    run_live = st.toggle(
        "Score the frozen trajectory model for this station (read-only)",
        value=False,
        help="Calls _trajectory.trajectory_forecast on the existing aligned "
             "table. No training, no writes.",
    )
    if run_live:
        payload = _live(station)
        if payload.get("error"):
            st.warning(
                f"**Cannot score this station.** `{payload.get('reason', 'error')}`: "
                f"{payload['error']}"
            )
            traj = None
        else:
            traj = _traj_frame(payload)
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Anchor", f"{payload.get('anchor_gwl', float('nan')):.3f} m")
            a2.metric("Anchor time", str(payload.get("anchor_time", "—"))[:16])
            t30 = payload.get("trajectory_30d", {}) or {}
            a3.metric("+30 d (q50)", f"{t30.get('level', float('nan')):.3f} m")
            a4.metric("+30 d change", f"{t30.get('change', float('nan')):+.3f} m")
            oc = payload.get("overall_confidence", {}) or {}
            st.caption(
                f"Overall confidence **{oc.get('level', '—')}** — "
                f"{oc.get('reason', '')}"
            )
            if payload.get("direction"):
                d = payload["direction"]
                st.caption(
                    f"Direction **{d.get('label', '—')}** · "
                    f"sign accuracy reported at 30 d: {d.get('sign_accuracy_30d')}"
                )
    else:
        traj = None
        st.caption("Toggle on to score the frozen model for this station.")

_plot(hist, traj, title=f"{station} — observed tail and model trajectory",
      daily_median=daily_median)

if traj is not None and not traj.empty:
    st.subheader("Per-horizon detail")
    st.dataframe(_band_table(traj), width="stretch", hide_index=True)
    st.caption(
        "Confidence levels come from the evidence-based framework in "
        "`models/traj_reliability.json`, applied at inference. A `LOW` at short "
        "horizons is expected: the model's margin over persistence there is "
        "within the observation noise floor."
    )

# ---- archived generation ----------------------------------------------
st.subheader("Archived forecast generation")
arch = _archived(station)
if arch is None:
    st.info(
        "**Data/artifact unavailable.** No archived generation found for this "
        "station under `data/refresh/forecast_archive/`. The archive keeps the "
        "*previous* published forecast each time a new one is published, so it "
        "only exists for stations the refresh daemon has served at least twice."
    )
else:
    st.caption(
        f"Anchor **{arch.get('anchor_time', '—')}** · model "
        f"**{arch.get('model', '—')}** · forecast_generated "
        f"**{arch.get('forecast_generated', arch.get('forecast_generation_ts', '—'))}**"
    )
    atraj = _traj_frame(arch)
    if atraj is not None:
        st.dataframe(_band_table(atraj).head(24), width="stretch", hide_index=True)
        st.caption("First 24 of 120 steps shown.")
