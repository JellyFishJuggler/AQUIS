"""_trajectory.py - runtime engine for the genuine 6-hourly trajectory (v2).

Scores the shared direct multi-horizon models (q05/q50/q95) at the latest
observed anchor row with h = 1..120 -> 120 genuine 6-hourly forecast deltas;
level = observed anchor + delta (the anchor is NEVER a prediction). Applies the
per-horizon calibration from the honest backtest and the evidence-based
confidence framework (models/traj_reliability.json) which also folds in, at
inference time: station integrity (recency / recent coverage / spikes), future
driver source (Open-Meteo <= 16d forecast / climatology beyond / unavailable),
anchor out-of-training-range OOD, and trajectory stability. The numerical +30d
endpoint and its uncertainty continue to come from the production direct-30d
model (benchmark); the trajectory still reports its own genuine +30d point.

Used by the Forecast page; unit-tested for the causal contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import importlib

F = importlib.import_module("06_features")

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
TRAJ_CFG = ROOT / "models" / "traj_config.json"
CALIB = ROOT / "models" / "traj_calibration.json"
RELI = ROOT / "models" / "traj_reliability.json"
OM = ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet"
OM_META = ROOT / "data" / "meta" / "driver_climatology_meta.json"
CLIM = ROOT / "data" / "meta" / "driver_climatology.parquet"
RAIN_CLIM = ROOT / "data" / "meta" / "district_rain_climatology.parquet"
RFC = ROOT / "data" / "cfs" / "river_forecast_cwc.parquet"
RUNTIME = ROOT / "data" / "refresh" / "runtime.json"

MAX_H = 120
FORECAST_STEPS = 64          # Open-Meteo 16 forecast days on the 6h grid
STEP_H = 6

_CACHE: dict = {}


def _load():
    if _CACHE:
        return _CACHE
    import xgboost as xgb
    from _model import forward_forecast
    cfg = json.loads(TRAJ_CFG.read_text())
    fnames = list(cfg["feature_cols"])
    models = {n: xgb.Booster(model_file=str(ROOT / "models" / f"traj_xgb_{n}.json"))
              for n in ("q05", "q50", "q95")}
    cal = json.loads(CALIB.read_text()) if CALIB.exists() else None
    rel = json.loads(RELI.read_text()) if RELI.exists() else None
    districts_om: set[str] = set()
    if OM.exists():
        districts_om = set(pd.read_parquet(OM, columns=["District"])["District"].astype(str))
    clim_stations: set[str] = set()
    if OM_META.exists():
        clim_stations = set(json.loads(OM_META.read_text()).get("stations", []))
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl = F.drop_gwl_spikes(tbl)

    # feature_mode: "flat" everywhere by default (bit-identical to production);
    # the refresh pipeline opts into "future" only after its backtest passes.
    feature_mode = "flat"
    try:
        if RUNTIME.exists():
            feature_mode = json.loads(RUNTIME.read_text()).get("feature_mode", "flat")
    except Exception:  # noqa: BLE001 - corrupt runtime file must not break the app
        feature_mode = "flat"

    om_frame = None
    clim_frame = None
    rain_clim_frame = None
    if feature_mode == "future":
        om_frame = pd.read_parquet(OM) if OM.exists() else None
        clim_frame = pd.read_parquet(CLIM) if CLIM.exists() else None
        rain_clim_frame = pd.read_parquet(RAIN_CLIM) if RAIN_CLIM.exists() else None

    _CACHE.update(models=models, fnames=fnames, cal=cal, rel=rel,
                  districts_om=districts_om, clim_stations=clim_stations,
                  feature_mode=feature_mode, om_frame=om_frame,
                  clim_frame=clim_frame, rain_clim_frame=rain_clim_frame,
                  tbl=tbl, forward_forecast=forward_forecast)
    return _CACHE


def _bucket(h: int, rel: dict) -> int:
    bh = [int(b) for b in rel["bucket_rules"]]
    b = np.searchsorted(bh, h, side="right") - 1
    return bh[max(0, b)]


def _widen_s(h: int, cal: dict | None) -> float:
    if not cal:
        return 1.0
    hs = list(cal["horizons_h"])
    ss = list(cal["widening_s"])
    return float(np.interp(h * STEP_H, hs, ss))


def _driver_score(h: int, district: str, om_districts: set) -> tuple[float, str]:
    if h <= FORECAST_STEPS and district in om_districts:
        return 1.0, "Open-Meteo forecast (<=16d)"
    if district in om_districts:
        return 0.7, "climatology (beyond Open-Meteo forecast)"
    return 0.35, "driver source unavailable (no district forecast/climatology)"


def _conf_level(h, row, station_row, bucket_rule, rel):
    w = rel["weights"]
    score = (w["interval_quality"] * row["interval"]
             + w["perf_vs_persist"] * row["perf"]
             + w["direction"] * row["direction"]
             + w["width_scaled"] * row["width"]
             + w["station_integrity"] * station_row["integrity"]
             + w["driver_availability"] * row["driver"]
             + w["anchor_ood"] * station_row["ood"]
             + w["trajectory_stability"] * station_row["stability"])
    thr = rel["thresholds"]
    reasons = []
    if bucket_rule["interval_quality"] != "ok":
        reasons.append("interval coverage below threshold")
    if not bucket_rule["beats_persistence"]:
        reasons.append("horizon error >= persistence")
    if not (bucket_rule["direction_supported"] or bucket_rule.get("direction_marginal")):
        reasons.append("direction weak at this horizon")
    if bucket_rule.get("width_scaled_ok") is False:
        reasons.append("interval width large vs error budget")
    if station_row["integrity"] < 0.6:
        reasons.append(station_row["integrity_reason"])
    if row["driver"] < 0.7:
        reasons.append(row["driver_reason"].lower())
    if station_row["ood"] < 1.0:
        reasons.append("anchor outside typical training range")
    if station_row["stability"] < 1.0:
        reasons.append("trajectory oscillates")

    if (score >= thr["horizon_row_score_high"] and bucket_rule["interval_quality"] == "ok"
            and station_row["integrity"] >= 0.6 and row["driver"] >= 0.7
            and bucket_rule["direction_supported"]):
        level = "HIGH"
    elif (score >= thr["horizon_row_score_directional"] and station_row["integrity"] >= 0.4
          and (bucket_rule["direction_supported"] or bucket_rule.get("direction_marginal"))):
        level = "DIRECTIONAL"
    else:
        level = "LOW"
    reason = "; ".join(reasons) if reasons else "forecast conditions support reliability"
    return level, reason


def trajectory_forecast(station: str) -> dict:
    c = _load()
    models, fnames, cal, rel = c["models"], c["fnames"], c["cal"], c["rel"]
    ffwd = c["forward_forecast"]
    if rel is None:
        return {"error": "trajectory reliability tables not built (run 33)"}
    if cal is None:
        return {"error": "trajectory calibration not built (run 32)"}

    tbl = c["tbl"]
    sta = tbl[tbl["Station"].astype(str) == station].sort_values("time")
    if sta.empty:
        return {"error": f"no data for station {station}"}
    hist = sta.dropna(subset=["gwl"])
    if hist.empty:
        return {"error": "no observed GWL"}
    anchor_t = hist["time"].iloc[-1]
    hist_d = sta[sta["time"] <= anchor_t].copy()
    hist_d["date"] = hist_d["time"].dt.normalize()
    feats, _ = F.build_full(hist_d, keep_na=True, horizon_steps=120, horizon_label=30)
    feats = feats.sort_values("time")
    if feats.empty:
        return {"error": "no feature frame"}
    last = feats.iloc[-1:].copy()
    anc = float(last["gwl"].iloc[0])
    if not np.isfinite(anc):
        return {"error": "anchor GWL invalid"}

    base_cols = [x for x in fnames if x not in ("h", "h_sin", "h_cos")]
    hs = np.arange(1, MAX_H + 1, dtype=np.float32)
    if c["feature_mode"] == "future":
        from refresh.future_drivers import build_step_features, _load_river_forecast
        district = str(sta["District"].iloc[0])
        rfc = _load_river_forecast() if RFC.exists() else None
        X, _ = build_step_features(
            station=station, district=district, anchor_t=anchor_t,
            hist=hist_d, fnames=fnames, om_frame=c["om_frame"],
            climatology=c["clim_frame"], rain_clim=c["rain_clim_frame"],
            river_forecast=rfc,
            forecast_days=int(FORECAST_STEPS / 4), steps=MAX_H,
            mode="future", anchor_gwl=anc)
    else:
        base = np.asarray(last[base_cols].values[0], dtype=np.float32)
        X = np.hstack([np.repeat(base[None, :], MAX_H, axis=0),
                       hs[:, None],
                       np.sin(2 * np.pi * hs / MAX_H)[:, None],
                       np.cos(2 * np.pi * hs / MAX_H)[:, None]])
    import xgboost as xgb
    dm = xgb.DMatrix(X, feature_names=fnames)
    q05 = models["q05"].predict(dm) + anc
    q50 = models["q50"].predict(dm) + anc
    q95 = models["q95"].predict(dm) + anc
    lo = np.minimum(q05, q50 - 1e-6)
    hi = np.maximum(q95, q50 + 1e-6)
    widen = np.array([_widen_s(h, cal) for h in range(1, MAX_H + 1)])
    lo = q50 - widen * (q50 - lo)
    hi = q50 + widen * (hi - q50)

    district = str(sta["District"].iloc[0]).upper()
    om_districts = c["districts_om"]

    now = pd.Timestamp.now()
    recency_d = float((now - anchor_t).total_seconds()) / 86400.0
    last90 = sta[sta["time"] >= anchor_t - pd.Timedelta(days=90)]
    cover90 = float(last90["gwl"].notna().mean()) if len(last90) else 0.0
    if recency_d <= 7 and cover90 >= 0.6:
        integrity, ireason = 1.0, "station data fresh"
    elif recency_d <= 30 and cover90 >= 0.3:
        integrity, ireason = 0.6, f"stale reading ({recency_d:.0f} days)"
    else:
        integrity, ireason = 0.4, (f"low recent coverage ({cover90:.0%}, "
                                   f"recency {recency_d:.0f}d)")
    g = sta["gwl"].dropna()
    p01, p99 = float(np.percentile(g, 1)), float(np.percentile(g, 99))
    ood = 0.6 if (anc < p01 or anc > p99) else 1.0
    flips = int(np.sum(np.diff(np.sign(np.diff(q50))) != 0)) if MAX_H > 2 else 0
    stability = 0.7 if flips > int(0.25 * (MAX_H - 2)) else 1.0
    station_row = {"integrity": integrity, "integrity_reason": ireason,
                   "ood": ood, "stability": stability}

    points = []
    for h in range(1, MAX_H + 1):
        b = _bucket(h, rel)
        br = rel["bucket_rules"][str(b)]
        dscr, dsrc = _driver_score(h, district, om_districts)
        row = {
            "interval": (1.0 if (br.get("calibrated_coverage") is not None
                                 and float(br["calibrated_coverage"]) >= rel["thresholds"]["coverage_min"])
                         else 0.4),
            "perf": (1.0 if br["beats_persistence"]
                     else (0.5 if (br.get("perf_vs_persist_ratio") or 0) >= 0.9 else 0.25)),
            "direction": (1.0 if br["direction_supported"]
                          else (0.65 if br.get("direction_marginal") else 0.35)),
            "width": 1.0 if br.get("width_scaled_ok") else 0.6,
            "driver": dscr, "driver_reason": dsrc,
        }
        lvl, reason = _conf_level(h, row, station_row, br, rel)
        points.append({
            "time": pd.Timestamp(anchor_t + pd.Timedelta(hours=h * STEP_H)).isoformat(),
            "gwl": float(q50[h - 1]), "q05": float(lo[h - 1]), "q50": float(q50[h - 1]),
            "q95": float(hi[h - 1]),
            "confidence_level": lvl, "reliability_reason": reason,
            "driver_source": dsrc,
        })

    ff = ffwd(station)
    endpoint = None
    if ff:
        endpoint = {"level": ff.get("q50_level"), "q05": ff.get("q05_level"),
                    "q95": ff.get("q95_level"), "anchor": ff.get("anchor"),
                    "band_half": ff.get("band_half")}

    end_change = float(q50[MAX_H - 1] - anc)
    direction = "stable"
    if abs(end_change) >= 0.15:
        direction = "declining" if end_change < 0 else "rising"
    br30 = rel["bucket_rules"][str(_bucket(MAX_H, rel))]
    ep_agreement = None
    if endpoint and endpoint["level"] is not None:
        ep_change = float(endpoint["level"] - anc)
        ep_agreement = bool(np.sign(ep_change) == np.sign(end_change) or abs(end_change) < 0.15)
    lvl_30, reason_30 = _conf_level(MAX_H, row, station_row, br30, rel)

    return {
        "station": station,
        "anchor_time": pd.Timestamp(anchor_t).isoformat(),
        "anchor_gwl": anc,
        "model": "direct multi-horizon shared XGBoost (q05/q50/q95)",
        "trajectory": points,
        "trajectory_30d": {
            "times": [p["time"] for p in points],
            "level": float(q50[MAX_H - 1]),
            "q05": float(lo[MAX_H - 1]), "q95": float(hi[MAX_H - 1]),
            "change": float(end_change),
        },
        "endpoint_production": endpoint,
        "direction": {
            "label": direction,
            "change_q50_30d": float(end_change),
            "agreement_with_production": ep_agreement,
            "sign_accuracy_30d": br30.get("sign_acc"),
        },
        "overall_confidence": {
            "level": lvl_30,
            "reason": reason_30,
            "endpoint_match": ep_agreement,
        },
        "evidence": {
            "station_integrity": station_row,
            "anchor_ood": bool(ood < 1.0),
            "stability_oscillation": flips,
            "recency_days": round(recency_d, 1),
            "recent90_coverage": round(cover90, 3),
        },
    }


def preview(station: str) -> None:
    out = trajectory_forecast(station)
    if "error" in out:
        print(out["error"])
        return
    counts = {c: 0 for c in ("HIGH", "DIRECTIONAL", "LOW")}
    srcs = {}
    for p in out["trajectory"]:
        counts[p["confidence_level"]] += 1
        srcs[p["driver_source"].split()[0]] = srcs.get(p["driver_source"].split()[0], 0) + 1
    print(f"anchor {out['anchor_gwl']:.3f} m @ {out['anchor_time']}")
    print(f"conf: {counts}")
    print(f"driver: {srcs}")
    print(f"traj30d: {out['trajectory_30d']}")
    print(f"direction: {out['direction']}")
    print(f"overall: {out['overall_confidence']}")
    print(f"endpoint: {out['endpoint_production']}")


if __name__ == "__main__":
    import sys
    preview(sys.argv[1] if len(sys.argv) > 1 else "ASHADHA PRATHMIK VIDYALAYA")