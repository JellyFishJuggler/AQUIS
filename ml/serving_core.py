"""Serving core: artifact loaders, cached data readers and runtime dispatch.

Merges the former ``_model`` / ``_utils`` / ``_traj_dispatch`` modules. Streamlit
is optional: without it the cache decorators become no-ops, so the dispatcher
stays importable in the torch-only test environment.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

try:
    import streamlit as st
except ImportError:  # torch-only env: dispatcher still needs to import
    st = None

if st is not None:
    cache_resource, cache_data = st.cache_resource, st.cache_data
else:
    def _no_cache(**_):
        return lambda fn: fn
    cache_resource = cache_data = _no_cache

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
META = DATA / "meta"
ALIGNED = DATA / "aligned"
MODELS = BASE / "models"
OUT = BASE / "outputs"
RUNTIME = DATA / "refresh" / "runtime.json"
ALIGNED_6H = ALIGNED / "table_6h.parquet"

TRAJ_DL_CFG = MODELS / "traj_dl_config.json"
TRAJ_DL_MODEL = MODELS / "traj_dl.pt"
TRAJ_DL_CALIB = MODELS / "traj_dl_calibration.json"
TRAJ_DL_RELI = MODELS / "traj_dl_reliability.json"

DRIVER_LABELS = {
    "rain_1d": "Rain, 1 day", "rain_7d": "Rain, 7 days",
    "rain_30d": "Rain, 30 days", "rain": "Rain (daily)",
    "temp": "Air temperature", "humidity": "Relative humidity",
    "solar": "Solar radiation", "wind_speed": "Wind speed",
    "wind_dir": "Wind direction", "pressure": "Atmospheric pressure",
    "river_level": "River water level", "canal_level": "Canal water level",
    "gwl": "GWL (current)", "lag1": "GWL lag 6h", "lag4": "GWL lag 1d",
    "lag8": "GWL lag 2d", "lag28": "GWL lag 7d", "lag120": "GWL lag 30d",
    "gwl_roll7_mean": "GWL 7d mean", "gwl_roll7_std": "GWL 7d spread",
    "gwl_roll30_mean": "GWL 30d mean", "gwl_roll30_std": "GWL 30d spread",
    "temp_7d": "Temp 7d mean", "humidity_7d": "Humidity 7d mean",
    "year": "Year", "month_sin": "Month (sin)", "month_cos": "Month (cos)",
    "doy_sin": "Day-of-year (sin)", "doy_cos": "Day-of-year (cos)",
    "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)",
    "monsoon": "Monsoon season", "st_id": "Station (ordinal)",
    "dist_id": "District (ordinal)",
}
SOURCE_LABELS = {
    "gwl": "Groundwater level", "rainfall": "Rainfall",
    "temperature": "Temperature", "river_level": "River water level",
    "humidity": "Humidity", "solar": "Solar radiation",
    "wind_speed": "Wind speed", "wind_direction": "Wind direction",
    "pressure": "Pressure", "canal_level": "Canal water level",
    "canal_discharge": "Canal discharge",
}


# --------------------------------------------------------------------------- #
# cached artifact loaders
# --------------------------------------------------------------------------- #
@cache_resource(show_spinner=False)
def load_xgb_model():
    return joblib.load(MODELS / "xgb_multihorizon.joblib")


@cache_resource(show_spinner=False)
def load_linear_model():
    return joblib.load(MODELS / "linear_multihorizon.joblib")


@cache_resource(show_spinner=False)
def load_quantile_models() -> dict:
    return {q: joblib.load(MODELS / f"xgb_{q}.joblib")
            if (MODELS / f"xgb_{q}.joblib").exists() else None
            for q in ("q05", "q50", "q95")}


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_quantile_calibration() -> dict:
    f = MODELS / "quantile_calibration.json"
    return json.loads(f.read_text()) if f.exists() else {}


@cache_resource(show_spinner=False)
def load_models() -> dict:
    return {"xgboost": load_xgb_model, "ridge": load_linear_model}


@cache_data(ttl="10m", max_entries=2, show_spinner=False)
def load_feature_config() -> dict:
    return json.loads((MODELS / "feature_config.json").read_text())


@cache_data(ttl="10m", max_entries=2, show_spinner=False)
def load_model_metrics() -> pd.DataFrame:
    return pd.read_csv(OUT / "model_metrics.csv")


@cache_data(ttl="24h", max_entries=2, show_spinner=False)
def load_predictions() -> pd.DataFrame:
    cols = ["Station", "District", "date", "time", "target", "gwl",
            "xgb", "ridge", "q05_lvl", "q95_lvl"]
    return pd.read_parquet(OUT / "predictions_2026.parquet", columns=cols)


@cache_data(ttl="10m", max_entries=2, show_spinner=False)
def load_importance() -> pd.DataFrame:
    return pd.read_csv(OUT / "feature_importance.csv")


@cache_data(ttl="10m", max_entries=2, show_spinner=False)
def load_coefficients() -> pd.DataFrame:
    return pd.read_csv(OUT / "feature_coefficients.csv")


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_honest_metrics() -> pd.DataFrame:
    return pd.read_csv(OUT / "honest_metrics.csv")


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_overlap_summary() -> dict:
    return json.loads((OUT / "overlap.json").read_text())


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_residual_acf() -> dict:
    return json.loads((OUT / "residual_acf.json").read_text())


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_residual_spatial() -> dict:
    return json.loads((OUT / "residual_spatial.json").read_text())


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_spatial_cv_metrics() -> pd.DataFrame:
    f = OUT / "spatial_cv_metrics.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_ablation() -> pd.DataFrame:
    return pd.read_csv(OUT / "ablation.csv")


@cache_data(ttl="15m", max_entries=4, show_spinner=False)
def _load_optional_csv(name: str) -> pd.DataFrame:
    f = OUT / name
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def load_fleet_table() -> pd.DataFrame:
    return _load_optional_csv("fleet_forecast.csv")


def load_fleet_district() -> pd.DataFrame:
    return _load_optional_csv("fleet_district.csv")


@cache_data(ttl="15m", max_entries=4, show_spinner=False)
def load_diagnostics() -> dict:
    f = OUT / "diagnostics.json"
    return json.loads(f.read_text()) if f.exists() else {}


@cache_data(ttl="15m", max_entries=4, show_spinner=False)
def load_diagnostics_permutation() -> pd.DataFrame:
    f = OUT / "diagnostics_permutation.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def load_station_summary() -> pd.DataFrame:
    return _load_optional_csv("station_summary.csv")


# --------------------------------------------------------------------------- #
# cached shared tables
# --------------------------------------------------------------------------- #
@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_manifest() -> dict:
    return json.loads((META / "manifest.json").read_text())


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_selected_gwl() -> pd.DataFrame:
    return pd.read_csv(META / "selected_gwl_stations.csv")


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_district_set() -> list[str]:
    return sorted(set(json.loads((META / "selected_districts.json").read_text())))


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_report() -> pd.DataFrame:
    return pd.read_csv(OUT / "correlation_report.csv")


@cache_data(ttl="10m", max_entries=4, show_spinner=False)
def load_lag_curves() -> pd.DataFrame:
    return pd.read_csv(OUT / "lag_curves.csv")


@cache_data(ttl="10m", max_entries=2, show_spinner=False)
def load_table() -> pd.DataFrame:
    df = pd.read_parquet(ALIGNED / "table.parquet")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(["Station", "date"]).reset_index(drop=True)


@cache_data(ttl="10m", max_entries=2, show_spinner=False)
def load_table_6h() -> pd.DataFrame:
    df = pd.read_parquet(ALIGNED_6H)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.sort_values(["Station", "time"]).reset_index(drop=True)


@cache_data(ttl="10m", max_entries=2, show_spinner=False)
def station_recency() -> pd.Series:
    """Most-recent reading per station, cached on top of ``load_table_6h``."""
    return load_table_6h().groupby("Station")["time"].max()


def nice(name: str) -> str:
    return DRIVER_LABELS.get(name, name.replace("_", " ").capitalize())


def source_nice(name: str) -> str:
    return SOURCE_LABELS.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------- #
# pooled direct-30d forward path
# --------------------------------------------------------------------------- #
@cache_data(ttl="15m", max_entries=48, show_spinner=False)
def _forward_forecast_xgb(station: str) -> dict:
    """Deterministic 30-day forward outlook for one station (XGB benchmark).

    Anchor = latest observed GWL (never predicted); the model predicts the
    CHANGE, levels are reconstructed from the observed anchor.
    """
    import importlib

    feats_mod = importlib.import_module("pipeline_model")
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

    qlev: dict[str, float] = {}
    k = float(load_quantile_calibration().get("widen_factor_k", 1.0))
    if qmods.get("q05") and qmods.get("q95") and qmods.get("q50"):
        pred_q = {q: float(m.predict(last[fnames])[0]) for q, m in qmods.items() if m is not None}
        med = anchor + pred_q["q50"]
        lo = anchor + min(pred_q["q05"], pred_q["q50"])
        hi = anchor + max(pred_q["q95"], pred_q["q50"])
        qlev = {"q05": med - k * (med - lo), "q50": med,
                "q95": med + k * (hi - med)}
    band_half = float((qlev["q95"] - qlev["q05"]) / 2) if qlev else None

    sumry = load_station_summary()
    row = sumry[sumry["station"] == station]
    station_stride_rmse = float(row["xgb_stride_rmse"].iloc[0]) if len(row) else None

    return {
        "station": station, "date_from": date_from,
        "date_to": date_from + pd.Timedelta(days=30), "anchor": anchor,
        "pred_xgb": pred_xgb, "pred_ridge": pred_ridge,
        "xgb_level": anchor + pred_xgb, "ridge_level": anchor + pred_ridge,
        "q05_level": qlev.get("q05"), "q50_level": qlev.get("q50"),
        "q95_level": qlev.get("q95"), "widen_k": k, "band_half": band_half,
        "station_stride_rmse": station_stride_rmse,
    }


def forward_backend() -> str:
    """Runtime-selected forward backend: ``dl`` (default) or ``xgb``."""
    try:
        return str(json.loads(RUNTIME.read_text()).get("forward_backend", "dl") or "dl").lower()
    except (OSError, ValueError):
        return "dl"


@cache_data(ttl="15m", max_entries=48, show_spinner=False)
def forward_forecast(station: str) -> dict:
    """Dispatch to the runtime-selected backend; XGB is the fallback path."""
    if forward_backend() == "dl":
        try:
            from serving_engines import dl_forward_forecast

            out = dl_forward_forecast(station)
            if out:
                return out
        except Exception as e:  # pragma: no cover - defensive fallback
            print(f"forward DL backend failed, using xgb: {e}", file=sys.stderr)
    return _forward_forecast_xgb(station)


# --------------------------------------------------------------------------- #
# trajectory runtime dispatch
# --------------------------------------------------------------------------- #
VALID_BACKENDS = ("dl", "xgb")
_PROBE_CACHE: dict = {}
_PROBE_SIG = None


def traj_backend() -> str:
    """Requested trajectory backend from runtime.json (default ``dl``)."""
    try:
        if RUNTIME.exists():
            return str(json.loads(RUNTIME.read_text()).get("traj_backend", "dl")).lower()
    except Exception:  # noqa: BLE001 - corrupt runtime file must not break the app
        pass
    return "dl"


def _probe_signature():
    parts = []
    for p in (TRAJ_DL_CFG, TRAJ_DL_MODEL, TRAJ_DL_CALIB, TRAJ_DL_RELI, ALIGNED_6H):
        try:
            parts.append((p.name, p.stat().st_mtime_ns))
        except OSError:
            parts.append((p.name, None))
    return tuple(parts)


def dl_probe(*, use_cache: bool = True) -> dict:
    """Cheap, artifact-mtime-cached check that the GRU engine can be loaded."""
    global _PROBE_SIG
    sig = _probe_signature()
    if use_cache and _PROBE_CACHE and sig == _PROBE_SIG:
        return _PROBE_CACHE

    err = None
    try:
        import importlib

        import torch  # noqa: F401 - the failure mode we are guarding against

        missing = [str(p) for p in (TRAJ_DL_CFG, TRAJ_DL_MODEL,
                                    TRAJ_DL_CALIB, ALIGNED_6H) if not p.exists()]
        if missing:
            raise FileNotFoundError("missing: " + ", ".join(missing))
        cfg = json.loads(TRAJ_DL_CFG.read_text())
        if TRAJ_DL_CALIB.exists():
            json.loads(TRAJ_DL_CALIB.read_text())
        if TRAJ_DL_RELI.exists():
            json.loads(TRAJ_DL_RELI.read_text())
        dlmod = importlib.import_module("pipeline_traj_dl")
        net = dlmod.TrajGRU(len(cfg["feature_cols"]), hidden=int(cfg["hidden"]))
        net.load_state_dict(torch.load(TRAJ_DL_MODEL, map_location="cpu",
                                       weights_only=True))
        net.eval()
    except Exception as exc:  # noqa: BLE001 - probe turns any failure into status
        err = f"{type(exc).__name__}: {exc}"

    result = {"ok": err is None, "error": err,
              "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _PROBE_CACHE.clear()
    _PROBE_CACHE.update(result)
    _PROBE_SIG = sig
    return result


def backend_status(*, probe: bool = True) -> dict:
    """Resolved backend for /health and /fleet/scan."""
    requested = traj_backend()
    if requested == "dl":
        probe_out = dl_probe() if probe else {"ok": True, "error": None}
        return {"requested": requested, "backend": "dl",
                "degraded": not probe_out["ok"], "error": probe_out.get("error")}
    if requested == "xgb":
        return {"requested": requested, "backend": "xgb",
                "degraded": False, "error": None}
    return {"requested": requested, "backend": "xgb", "degraded": False,
            "error": None,
            "fallback_reason": f"unknown traj_backend {requested!r} -> xgb"}


def active_trajectory_forecast(station: str, *, allow_fallback: bool = False):
    """Dispatch a trajectory forecast to the selected engine.

    With ``traj_backend=dl`` a load/scoring failure is returned as a degraded
    error payload; it never silently switches to XGBoost unless the caller opts
    in with ``allow_fallback=True``.
    """
    requested = traj_backend()
    if requested == "dl":
        try:
            from serving_engines import dl_trajectory_forecast

            return dl_trajectory_forecast(station)
        except Exception as e:  # noqa: BLE001 - surface, never hide
            detail = f"{type(e).__name__}: {e}"
            print(f"[traj_backend=dl] GRU trajectory engine unavailable: {detail}",
                  file=sys.stderr)
            if not allow_fallback:
                return {
                    "error": (f"traj_backend=dl but the GRU trajectory engine "
                              f"failed to load: {detail}"),
                    "reason": "backend_unavailable", "backend": "dl",
                    "degraded": True,
                }
            print("[traj_backend=dl] explicit allow_fallback -> xgb", file=sys.stderr)
    from serving_engines import xgb_trajectory_forecast

    return xgb_trajectory_forecast(station)
