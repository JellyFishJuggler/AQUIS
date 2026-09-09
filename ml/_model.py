"""Model + evaluation output loaders for the ml app.

Reads ml/models/*.joblib + feature_config.json and outputs/*.csv from
06_train / 07_evaluate. All paths are resolved relative to the app root; the
pipeline scripts in this same directory (06/07) regenerate them.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
OUT = BASE / "outputs"


@st.cache_resource(show_spinner=False)
def load_xgb_model() -> xgb.Booster:
    return joblib.load(MODELS / "xgb_multihorizon.joblib")


@st.cache_resource(show_spinner=False)
def load_linear_model():
    return joblib.load(MODELS / "linear_multihorizon.joblib")


@st.cache_resource(show_spinner=False)
def load_quantile_models() -> dict:
    out = {}
    for q in ("q05", "q50", "q95"):
        f = MODELS / f"xgb_{q}.joblib"
        out[q] = joblib.load(f) if f.exists() else None
    return out


@st.cache_data(ttl="10m", max_entries=4)
def load_quantile_calibration() -> dict:
    f = MODELS / "quantile_calibration.json"
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    return {"xgboost": load_xgb_model, "ridge": load_linear_model}


@st.cache_data(ttl="10m", max_entries=2)
def load_feature_config() -> dict:
    return json.loads((MODELS / "feature_config.json").read_text())


@st.cache_data(ttl="10m", max_entries=2)
def load_model_metrics() -> pd.DataFrame:
    return pd.read_csv(OUT / "model_metrics.csv")


@st.cache_data(ttl="24h", max_entries=2)
def load_predictions() -> pd.DataFrame:
    cols = ["Station", "District", "date", "time", "target", "gwl",
            "xgb", "ridge", "q05_lvl", "q95_lvl"]
    return pd.read_parquet(OUT / "predictions_2026.parquet", columns=cols)


@st.cache_data(ttl="10m", max_entries=2)
def load_importance() -> pd.DataFrame:
    return pd.read_csv(OUT / "feature_importance.csv")


@st.cache_data(ttl="10m", max_entries=2)
def load_coefficients() -> pd.DataFrame:
    return pd.read_csv(OUT / "feature_coefficients.csv")


@st.cache_data(ttl="10m", max_entries=4)
def load_honest_metrics() -> pd.DataFrame:
    return pd.read_csv(OUT / "honest_metrics.csv")


@st.cache_data(ttl="10m", max_entries=4)
def load_overlap_summary() -> dict:
    return json.loads((OUT / "overlap.json").read_text())


@st.cache_data(ttl="10m", max_entries=4)
def load_residual_acf() -> dict:
    return json.loads((OUT / "residual_acf.json").read_text())


@st.cache_data(ttl="10m", max_entries=4)
def load_residual_spatial() -> dict:
    return json.loads((OUT / "residual_spatial.json").read_text())


@st.cache_data(ttl="10m", max_entries=4)
def load_spatial_cv_metrics() -> pd.DataFrame:
    f = OUT / "spatial_cv_metrics.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl="10m", max_entries=4)
def load_ablation() -> pd.DataFrame:
    return pd.read_csv(OUT / "ablation.csv")


@st.cache_data(ttl="15m", max_entries=4)
def load_fleet_table() -> pd.DataFrame:
    f = OUT / "fleet_forecast.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl="15m", max_entries=4)
def load_fleet_district() -> pd.DataFrame:
    f = OUT / "fleet_district.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl="15m", max_entries=4)
def load_diagnostics() -> dict:
    f = OUT / "diagnostics.json"
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data(ttl="15m", max_entries=4)
def load_diagnostics_permutation() -> pd.DataFrame:
    f = OUT / "diagnostics_permutation.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl="15m", max_entries=4)
def load_station_summary() -> pd.DataFrame:
    f = OUT / "station_summary.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl="15m", max_entries=48)
def forward_forecast(station: str) -> dict:
    """Deterministic 30-day forward outlook for one station (same path as training).

    Rebuilds the feature frame from the 6-hourly aligned table via
    ``06_features.build_full`` for the station slice, then scores it with the pooled
    XGBoost + Ridge models. Anchor = latest observed GWL (never predicted).
    Returns an empty dict if the station has no usable feature frame.
    """
    import importlib

    from _utils import load_table_6h

    feats_mod = importlib.import_module("06_features")
    cfg = load_feature_config()
    t = load_table_6h()
    t = t[t["Station"] == station].copy()
    t = feats_mod.drop_gwl_spikes(t)
    t["date"] = t["time"].dt.normalize()
    feats, _ = feats_mod.build_full(t, keep_na=True)
    last = feats.sort_values("time").tail(1)
    if last.empty:
        return {}
    xgb_b = load_xgb_model()
    linear_b = load_linear_model()
    qmods = load_quantile_models()
    fnames = list(getattr(xgb_b, "feature_names_in_", cfg["num_cols"]))
    pred_xgb = float(xgb_b.predict(last[fnames])[0])
    pred_ridge = float(
        linear_b.predict(last[list(cfg["num_cols"]) + ["Station", "District"]])[0])
    anchor = float(last["gwl"].iloc[0])
    date_from = pd.to_datetime(last["time"]).iloc[0]

    # calibrated quantile interval (delta target -> level via anchor)
    qlev: dict[str, float] = {}
    k = float(load_quantile_calibration().get("widen_factor_k", 1.0))
    if qmods.get("q05") and qmods.get("q95") and qmods.get("q50"):
        pred_q = {q: float(m.predict(last[fnames])[0]) for q, m in qmods.items() if m is not None}
        med = anchor + pred_q["q50"]
        lo = anchor + pred_q["q05"]
        hi = anchor + pred_q["q95"]
        qlev = {"q05": med - k * (med - lo), "q50": med,
                "q95": med + k * (hi - med)}
    band_half = float((qlev["q95"] - qlev["q05"]) / 2) if qlev else None

    # per-station uncertainty context from the honest re-score (test stations only)
    sumry = load_station_summary()
    row = sumry[sumry["station"] == station]
    station_stride_rmse = float(row["xgb_stride_rmse"].iloc[0]) if len(row) else None

    return {
        "station": station,
        "date_from": date_from,
        "date_to": date_from + pd.Timedelta(days=30),
        "anchor": anchor,
        "pred_xgb": pred_xgb,
        "pred_ridge": pred_ridge,
        "xgb_level": anchor + pred_xgb,
        "ridge_level": anchor + pred_ridge,
        "q05_level": qlev.get("q05"),
        "q50_level": qlev.get("q50"),
        "q95_level": qlev.get("q95"),
        "widen_k": k,
        "band_half": band_half,
        "station_stride_rmse": station_stride_rmse,
    }