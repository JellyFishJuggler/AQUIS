"""Page 4 — Model evaluation, with the evaluation protocol labelled per result.

The whole point of this page is that **numbers from different protocols are not
comparable**. Every table states its basis, and the page warns explicitly where a
comparison would be misleading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _charts as C
import _data as D


@st.cache_data(ttl="10m", show_spinner=False)
def _mm() -> pd.DataFrame | None:
    return D.model_metrics()


@st.cache_data(ttl="10m", show_spinner=False)
def _hm() -> pd.DataFrame | None:
    return D.honest_metrics()


@st.cache_data(ttl="10m", show_spinner=False)
def _tb() -> pd.DataFrame | None:
    return D.traj_backtest_metrics()


@st.cache_data(ttl="10m", show_spinner=False)
def _tbs() -> dict | None:
    return D.traj_backtest_summary()


@st.cache_data(ttl="10m", show_spinner=False)
def _scv() -> pd.DataFrame | None:
    return D.spatial_cv_metrics()


@st.cache_data(ttl="10m", show_spinner=False)
def _abl() -> pd.DataFrame | None:
    return D.ablation()


@st.cache_data(ttl="10m", show_spinner=False)
def _arima() -> pd.DataFrame | None:
    return D.arima_vs_xgb()


@st.cache_data(ttl="10m", show_spinner=False)
def _ss() -> pd.DataFrame | None:
    return D.station_summary()


@st.cache_data(ttl="10m", show_spinner="Reading saved 2026 predictions…")
def _predictions_2026() -> pd.DataFrame | None:
    return D.predictions_2026()


@st.cache_data(ttl="10m", show_spinner="Reading published forecast verification…")
def _verification_ledger() -> pd.DataFrame | None:
    return D.verification_ledger()


def _protocol_box(basis: str, what: str, caveat: str | None = None):
    st.caption(f"**Basis:** {basis}")
    st.markdown(what)
    if caveat:
        st.warning(caveat)


st.title("Model evaluation")
st.caption(
    "Every result is labelled with its evaluation protocol. Comparisons across "
    "protocols are not valid and are called out explicitly."
)

with st.expander("Protocol cheat-sheet — read this first", expanded=True):
    st.markdown(
        """
| Protocol | Unit of evidence | Dependence | Comparable with |
|---|---|---|---|
| **Raw 6-hourly rows** | every 6 h anchor, scored at the single GWL reading 30 d later | serially correlated | same raw-row protocol |
| **Stride subset** | one point target per station every 120 six-hour steps (30 d) | still serially correlated | same stride protocol |
| **Trajectory backtest** | anchors spaced 14 days | yes | other anchor-based results |
| **Spatial CV folds** | held-out geographic block | yes for geography, **not** for station identity | other spatial results |
| **Daily-mean origins** | 10 stations × 5 origins × 30 d | yes | only other daily-mean results |

Raw rows and stride rows score different temporal samples. The stride subset
reduces repeated adjacent anchors, but its point errors are still temporally
correlated; neither row count should be read as a count of independent studies.
"""
    )

# =====================================================================
# paired 2026 actuals vs forecasts
# =====================================================================
st.header("2026 actual values vs forecasts — paired targets")
preds_2026 = _predictions_2026()
if preds_2026 is None:
    st.info(
        "**Data/artifact unavailable.** `outputs/predictions_2026.parquet` is "
        "missing or does not contain the required paired actual/forecast columns."
    )
else:
    p30 = preds_2026[pd.to_numeric(preds_2026["horizon"], errors="coerce") == 30].copy()
    model_cols = {
        "XGBoost": "xgb",
        "Ridge": "ridge",
        "Persistence": "persist",
        "Climatology": "clim",
    }
    c1, c2 = st.columns(2)
    with c1:
        model_label = st.selectbox("Forecast model", list(model_cols))
    with c2:
        stations = sorted(p30["Station"].dropna().astype(str).unique())
        selected_station = st.selectbox(
            "Station", stations or ["No station rows"], key="validation_station")

    pred_col = model_cols[model_label]
    p30["actual"] = pd.to_numeric(p30["target"], errors="coerce")
    p30["forecast"] = pd.to_numeric(p30[pred_col], errors="coerce")
    p30["anchor_time"] = pd.to_datetime(p30["time"], errors="coerce")
    # The saved artifact is built from a regular 6-hour grid; its 30-day label
    # comes from the row exactly 120 steps after the forecast anchor.
    p30["target_time"] = p30["anchor_time"] + pd.to_timedelta(
        pd.to_numeric(p30["horizon"], errors="coerce"), unit="D")
    p30 = p30[p30["target_time"].dt.year.eq(2026)]
    paired = p30.dropna(subset=["actual", "forecast", "target_time"]).copy()
    shown = paired[paired["Station"].astype(str) == selected_station].copy()
    shown["error"] = shown["forecast"] - shown["actual"]

    st.caption(
        "Basis: saved pooled-model 30-day test predictions. Actual target and "
        "forecast are taken from the same artifact row; only pairs where both are "
        "finite are scored. The target date is anchor time + 30 days on the aligned "
        "6-hour grid."
    )
    st.warning(
        "These saved 2026 predictions predate the target-time purge now present in "
        "source and have not been regenerated. Treat this as a paired inspection "
        "of the existing artifact, not a validation score for the corrected pipeline."
    )

    if shown.empty:
        st.info("No finite actual/forecast pairs are available for this selection.")
    else:
        rmse = float(np.sqrt(np.mean(shown["error"].to_numpy() ** 2)))
        mae = float(np.mean(np.abs(shown["error"].to_numpy())))
        bias = float(shown["error"].mean())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched target readings", f"{len(shown):,}")
        m2.metric("RMSE", f"{rmse:.3f} m")
        m3.metric("MAE", f"{mae:.3f} m")
        m4.metric("Mean forecast error", f"{bias:+.3f} m")

        interval_cols = {"q05_lvl", "q95_lvl"}
        if model_label == "XGBoost" and interval_cols.issubset(shown.columns):
            lo = pd.to_numeric(shown["q05_lvl"], errors="coerce")
            hi = pd.to_numeric(shown["q95_lvl"], errors="coerce")
            interval_ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(shown["actual"])
            if interval_ok.any():
                picp = float(((shown.loc[interval_ok, "actual"] >= lo[interval_ok]) &
                              (shown.loc[interval_ok, "actual"] <= hi[interval_ok])).mean())
                width = float((hi[interval_ok] - lo[interval_ok]).mean())
                i1, i2 = st.columns(2)
                i1.metric("Saved q05–q95 coverage", f"{picp:.1%}")
                i2.metric("Mean interval width", f"{width:.3f} m")
                st.caption(
                    f"Interval metrics use {int(interval_ok.sum()):,} matched pairs; "
                    "coverage is reported on this saved evaluation set, not an "
                    "independent calibration holdout."
                )

        chart_frame = shown.sort_values("target_time").set_index("target_time")
        chart_frame["error"] = (pd.to_numeric(chart_frame["forecast"], errors="coerce")
                                - pd.to_numeric(chart_frame["actual"], errors="coerce"))
        st.subheader(f"{selected_station} — actual and forecast level")
        st.plotly_chart(
            C.paired_actual_forecast(chart_frame, height=440,
                                     forecast_name="30-day forecast — honest backtest (stride grid)",
                                     daily_median=True),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption(
            "Left: observed level and the honest-backtest 30-day forecasts "
            "(one point per stride anchor; the Feb–Oct span on this panel is the "
            "saved stride grid, not the live issued archive, which starts "
            "~Sep 2026). Right: their difference — bars above zero mean the "
            "forecast was too high. Both panels use daily medians, because the raw "
            "6-hourly oscillation hides the multi-day signal. The right-hand axis "
            "is centred on zero, with the mean bias as a dotted line, so systematic "
            "over- or under-prediction is visible rather than buried under the "
            "level range."
        )
        st.caption(
            f"Matched target dates: {chart_frame.index.min():%Y-%m-%d} to "
            f"{chart_frame.index.max():%Y-%m-%d}; one series point per paired "
            "30-day forecast target."
        )
        st.dataframe(
            chart_frame.reset_index()[["target_time", "actual", "forecast", "error"]],
            width="stretch", hide_index=True,
        )

st.subheader("Published trajectory forecasts matched to 2026 observations")
ledger = _verification_ledger()
if ledger is None:
    st.info(
        "**Data/artifact unavailable.** `data/refresh/verification.parquet` "
        "does not contain the published-forecast verification ledger."
    )
else:
    ledger["anchor_time"] = pd.to_datetime(ledger["anchor_time"], errors="coerce")
    ledger["target_time"] = pd.to_datetime(ledger["target_time"], errors="coerce")
    ledger["generated_ts"] = pd.to_datetime(ledger["generated_ts"], errors="coerce")
    ledger["horizon_h"] = pd.to_numeric(ledger["horizon_h"], errors="coerce")
    ledger["actual"] = pd.to_numeric(ledger["actual"], errors="coerce")
    ledger["q50"] = pd.to_numeric(ledger["q50"], errors="coerce")
    for bound in ("q05", "q95"):
        ledger[bound] = pd.to_numeric(ledger[bound], errors="coerce")
    ledger = ledger[ledger["target_time"].dt.year.eq(2026)].copy()
    ledger = ledger.dropna(subset=["target_time", "horizon_h", "actual", "q50"])

    if ledger.empty:
        st.info("The ledger has no matched forecast/actual rows with target dates in 2026.")
    else:
        versions = sorted(ledger["model_version"].dropna().astype(str).unique())
        if versions:
            version_dates = ledger.groupby(ledger["model_version"].astype(str))["generated_ts"].max()
            newest_version = str(version_dates.idxmax()) if version_dates.notna().any() else versions[-1]
            version = st.selectbox(
                "Forecast model version", versions,
                index=versions.index(newest_version), key="verification_model_version")
            ledger = ledger[ledger["model_version"].astype(str) == version]

        horizons_h = sorted(ledger["horizon_h"].dropna().astype(int).unique())
        if horizons_h:
            default_h = 720 if 720 in horizons_h else horizons_h[-1]
            horizon_h = st.selectbox(
                "Forecast horizon",
                horizons_h,
                index=horizons_h.index(default_h),
                format_func=lambda h: f"{h} h ({h / 24:g} d)",
                key="verification_horizon",
            )
            stations = sorted(ledger["station"].dropna().astype(str).unique())
            station = st.selectbox(
                "Verification station", stations or ["No station rows"],
                key="verification_station")
            matched = ledger[ledger["horizon_h"] == horizon_h].copy()
            matched = matched[matched["station"].astype(str) == station]

            st.caption(
                "These are forecasts actually published by the refresh pipeline, " 
                "matched by station and exact target timestamp to later observed "
                "GWL. Each row is one forecast origin/horizon pair; different "
                "origins can forecast the same station on different dates."
            )
            if matched.empty:
                st.info("No matched forecast/actual pairs for this selection.")
            else:
                matched["error"] = matched["q50"] - matched["actual"]
                mm1, mm2, mm3, mm4 = st.columns(4)
                mm1.metric("Matched forecast points", f"{len(matched):,}")
                mm2.metric("RMSE", f"{np.sqrt(np.mean(matched['error'] ** 2)):.3f} m")
                mm3.metric("MAE", f"{np.mean(np.abs(matched['error'])):.3f} m")
                mm4.metric("Mean forecast error", f"{matched['error'].mean():+.3f} m")

                interval_ok = (np.isfinite(matched["q05"]) &
                               np.isfinite(matched["q95"]) &
                               np.isfinite(matched["actual"]))
                if interval_ok.any():
                    coverage = ((matched.loc[interval_ok, "actual"] >=
                                 matched.loc[interval_ok, "q05"]) &
                                (matched.loc[interval_ok, "actual"] <=
                                 matched.loc[interval_ok, "q95"])).mean()
                    width = (matched.loc[interval_ok, "q95"] -
                             matched.loc[interval_ok, "q05"]).mean()
                    ii1, ii2 = st.columns(2)
                    ii1.metric("Published interval coverage", f"{coverage:.1%}")
                    ii2.metric("Mean interval width", f"{width:.3f} m")
                    st.caption(
                        f"Interval metrics use {int(interval_ok.sum()):,} points with "
                        "both bounds present."
                    )

                series = (matched.sort_values("target_time")
                          .set_index("target_time")[["actual", "q50"]]
                          .rename(columns={"q50": "forecast median"}))
                series["error"] = series["forecast median"] - series["actual"]
                st.plotly_chart(
                    C.paired_actual_forecast(series, height=440, daily_median=True,
                                             forecast_col="forecast median",
                                             forecast_name="30-day forecast — live issued (archive)",
                                             title="published trajectory vs realised"),
                    width="stretch", config={"displayModeBar": False},
                )
                detail_cols = ["anchor_time", "target_time", "horizon_h",
                               "actual", "q50", "q05", "q95", "error"]
                st.dataframe(matched[detail_cols].sort_values("target_time"),
                             width="stretch", hide_index=True)

# =====================================================================
# 1. pooled 30-day
# =====================================================================
st.header("1 · Pooled 30-day model — all anchors and a stride sample")
mm = _mm()
hm = _hm()

if mm is None and hm is None:
    st.info("**Artifact unavailable.** `outputs/model_metrics.csv` / "
            "`honest_metrics.csv` not found.")
else:
    if mm is not None:
        _protocol_box(
            "every 6-hourly anchor in the 2026 test split; each predicts one endpoint 30 d later",
            "Four-way comparison on the level basis, single 30-day horizon. "
            "`persistence` carries today's reading forward unchanged; "
            "`climatology` is a per-station day-of-year mean from training years.",
            "The dense series of endpoint errors is temporally correlated and "
            "weights stations with more test observations more heavily. The row "
            "count is not an independent sample size; use the stride result as a "
            "sparser view, while still accounting for station/time dependence. "
            "These saved metrics predate the target-time purge now in source and "
            "have not been regenerated; treat them as historical artifact results.",
        )
        cols = [c for c in ("model", "horizon", "rows", "rmse", "mae", "r2",
                            "within_025", "within_05", "within_10")
                if c in mm.columns]
        st.dataframe(mm[cols], width="stretch", hide_index=True)

    if hm is not None:
        _protocol_box(
            "stride subset: one 30-day endpoint forecast per station every 120 six-hour steps",
            "This samples endpoint predictions about 30 days apart; the target is "
            "one observed GWL point, not a 30-day trajectory window.",
            "Errors can still be serially correlated across adjacent origins and "
            "clustered by station, so `n_windows` is not a count of independent "
            "statistical replicates. This saved artifact also predates the "
            "target-time purge and has not been regenerated.",
        )
        cols = [c for c in ("model", "horizon", "raw_rows", "stride_rows", "n_windows",
                            "basis_raw_rmse", "basis_stride_rmse",
                            "window_rmse_mean", "window_rmse_median",
                            "window_rmse_p90") if c in hm.columns]
        st.dataframe(hm[cols], width="stretch", hide_index=True)
        if "basis_stride_rmse" in hm.columns and "model" in hm.columns:
            base = hm.set_index("model")["basis_stride_rmse"]
            if "xgb" in base.index and "persist" in base.index:
                gain = 100 * (base["persist"] - base["xgb"]) / base["persist"]
                st.metric("XGBoost vs persistence (30-day stride)",
                          f"{gain:+.2f}%",
                          help="The entire driver feature set contributes a "
                               "fraction of this margin.")

    ss = _ss()
    if ss is not None and "xgb_stride_rmse" in ss.columns:
        st.subheader("Per-station honesty check")
        beat = ss["xgb_stride_rmse"] < ss["persist_stride_rmse"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Stations with a window", f"{int(beat.notna().sum()):,}")
        c2.metric("Model beats persistence", f"{int(beat.sum()):,}")
        c3.metric("Model loses to persistence", f"{int((~beat & beat.notna()).sum()):,}")
        st.caption(
            "A pooled model can win on average while losing on individual wells. "
            "This is the distribution the aggregate RMSE hides."
        )
        worst = ss.dropna(subset=["xgb_stride_rmse", "persist_stride_rmse"]).copy()
        worst["delta"] = worst["xgb_stride_rmse"] - worst["persist_stride_rmse"]
        st.dataframe(
            worst.nlargest(15, "delta")[
                [c for c in ("station", "district", "n_windows",
                             "xgb_stride_rmse", "persist_stride_rmse", "delta")
                 if c in worst.columns]],
            width="stretch", hide_index=True,
        )
        st.caption("15 stations where the model is furthest behind persistence.")

# =====================================================================
# 2. trajectory backtest
# =====================================================================
st.header("2 · Trajectory v2 — per-horizon backtest")
tb = _tb()
tbs = _tbs()
if tb is None:
    st.info("**Artifact unavailable.** `outputs/traj_backtest_metrics.csv` not found.")
else:
    _protocol_box(
        "anchors spaced 14 days on observed readings, 2025-10-01 → 2026-08-20, "
        "scored at exact observed times",
        "The direct multi-horizon model scored once per anchor for every horizon, "
        "so each point is a genuine model output rather than a recursion step.",
        "The source now purges training and validation labels by target timestamp, "
        "but the saved backtest artifacts predate that code change and have not "
        "been regenerated. Treat their margin against persistence as provisional.",
    )
    st.dataframe(tb, width="stretch", hide_index=True)

    if "horizon_h" in tb.columns:
        t1, t2, t3 = st.tabs(["RMSE vs persistence", "Interval coverage",
                              "Interval half-width"])
        # The horizon axis is categorical in hours, not a datetime, so these use the
        # shared bar builder rather than the time-series one.
        with t1:
            d = tb.set_index("horizon_h")[[c for c in ("rmse", "rmse_persist", "rmse_clim")
                                           if c in tb.columns]]
            for c in d.columns:
                st.plotly_chart(
                    C.bars(d.index, d[c], name=str(c), height=260,
                           title=str(c), y_title="RMSE (m)"),
                    width="stretch", config={"displayModeBar": False},
                )
            st.caption("A margin inside ~0.05 m is within the noise band the "
                       "project itself declares.")
        with t2:
            if "coverage_raw" in tb.columns:
                st.plotly_chart(
                    C.bars(tb["horizon_h"], tb["coverage_raw"], height=320,
                           name="raw coverage", color=C.RAIN,
                           title="raw (uncalibrated) coverage", y_title="coverage"),
                    width="stretch", config={"displayModeBar": False},
                )
                st.caption("Raw (uncalibrated) coverage. The calibrated figure is "
                           "fitted on these same anchors, so it is in-sample.")
        with t3:
            if "halfwidth_median" in tb.columns:
                st.plotly_chart(
                    C.bars(tb["horizon_h"], tb["halfwidth_median"], height=320,
                           name="half-width", color=C.CARRIED,
                           title="median interval half-width", y_title="m"),
                    width="stretch", config={"displayModeBar": False},
                )

if tbs:
    st.subheader("Backtest summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anchors scored", f"{tbs.get('anchors_scored', 0):,}")
    c2.metric("Stations", f"{tbs.get('stations_sampled', 0):,}")
    traj = tbs.get("30d_rmse_traj")
    pers = tbs.get("30d_rmse_persistence")
    if traj and pers:
        c3.metric("Trajectory vs persistence", f"{100 * (pers - traj) / pers:+.2f}%")
    st.json(tbs, expanded=False)
    if tbs.get("30d_rmse_prev_recursive") is not None:
        st.error(
            f"**Internal inconsistency in this artifact.** The summary records a "
            f"*removed* recursive model at **{tbs['30d_rmse_prev_recursive']} m**, "
            f"which is better at +30 d than the promoted trajectory "
            f"(**{traj} m**). The promotion decision in `gate_check.py` never "
            f"compares against that number. Worth resolving before the trajectory "
            f"margin is treated as settled."
        )

# =====================================================================
# 3. spatial CV
# =====================================================================
st.header("3 · Spatial cross-validation")
scv = _scv()
if scv is None:
    st.info("**Artifact unavailable.** `outputs/spatial_cv_metrics.csv` not found.")
else:
    _protocol_box(
        "leave-one-spatial-block-out (k-means on station coordinates), 5 folds, "
        "2026 test rows of the held-out block",
        "Intended to answer: does the model generalise to an unseen region, or "
        "does it memorise station/district identity?",
        "Read with care — held-out stations are absent from the fold's training "
        "rows, so their global numeric station IDs are unseen values, not "
        "memorised station examples. Ordinals are not physical spatial features, "
        "however, and their arbitrary ordering can affect predictions for unseen "
        "IDs. Treat this as provisional spatial evidence.",
    )
    st.dataframe(scv, width="stretch", hide_index=True)
    if "rmse" in scv.columns:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean fold RMSE", f"{scv['rmse'].mean():.3f} m")
        c2.metric("Median fold RMSE", f"{scv['rmse'].median():.3f} m")
        c3.metric("Fold spread", f"{scv['rmse'].max() - scv['rmse'].min():.3f} m")
        st.plotly_chart(
            C.bars(scv["block"], scv["rmse"], height=320, name="fold RMSE",
                   title="RMSE by held-out spatial block", y_title="RMSE (m)"),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption("A 5× spread across geographic folds is the real story here: "
                   "the pooled number is not representative of every region.")

# =====================================================================
# 4. ablation
# =====================================================================
st.header("4 · Feature-group ablation for the pooled model")
abl = _abl()
if abl is None:
    st.info("**Artifact unavailable.** `outputs/ablation.csv` not found.")
else:
    _protocol_box(
        "retrain with one feature group removed, score the 2026 test split "
        "(raw and stride bases)",
        "Measures the marginal value of each driver family.",
        "Every delta here is inside or near the ±0.05 m noise band, so the "
        "ranking is weak evidence. Note also that dropping the GWL-lag group "
        "left the model with *more* trees, which suggests those columns were "
        "adding noise rather than signal.",
    )
    st.dataframe(abl, width="stretch", hide_index=True)
    if "config" in abl.columns and "stride_rmse" in abl.columns:
        base = abl.loc[abl["config"] == "baseline", "stride_rmse"]
        if len(base):
            d = abl.copy()
            d["delta_vs_baseline"] = d["stride_rmse"] - float(base.iloc[0])
            st.dataframe(d[["config", "num_feats", "n_trees", "stride_rmse",
                            "delta_vs_baseline"]],
                         width="stretch", hide_index=True)
            st.caption("Positive = worse than baseline when the group is removed.")

# =====================================================================
# 5. ARIMA
# =====================================================================
st.header("5 · SARIMA baseline (different protocol — not comparable)")
ar = _arima()
if ar is None:
    st.info("**Artifact unavailable.** `outputs/arima_vs_xgb.csv` not found.")
else:
    _protocol_box(
        "daily means, 10 stations, 5 rolling 2026 origins, 30-day horizon",
        "Univariate SARIMAX(1,1,1)×(1,0,1,7) refit per origin, matched against the "
        "frozen trajectory model on the same origins.",
        "This protocol is **not** the 6-hourly level-RMSE table above: different "
        "resolution, different stations, different sample. Do not mix the numbers. "
        "Also note the XGB arm here reads paired drivers and forward weather "
        "forecasts while SARIMA is univariate, and the XGB intervals are raw "
        "(uncalibrated) — so the coverage comparison favours neither model "
        "cleanly.",
    )
    st.dataframe(ar, width="stretch", hide_index=True)
    overall = ar[(ar.get("station") == "OVERALL")] if "station" in ar.columns else ar
    if "model" in overall.columns and "rmse" in overall.columns:
        st.dataframe(overall[["model", "n", "mae", "rmse", "picp90",
                              "mean_interval_width"]],
                     width="stretch", hide_index=True)
