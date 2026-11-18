"""Serving engines: XGBoost and quantile-GRU inference for both surfaces.

Merges the former ``_trajectory`` (direct multi-horizon XGBoost) and ``_dl``
(sequence/pooled GRU) modules. Shared causal helpers, station context and
response shaping live here once so both backends keep identical schemas.

Both engines forecast the CHANGE; the level is always observed anchor + delta.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

F = importlib.import_module("pipeline_model")

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
OM = ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet"
OM_META = ROOT / "data" / "meta" / "driver_climatology_meta.json"
CLIM = ROOT / "data" / "meta" / "driver_climatology.parquet"
RAIN_CLIM = ROOT / "data" / "meta" / "district_rain_climatology.parquet"
RFC = ROOT / "data" / "cfs" / "river_forecast_cwc.parquet"
RUNTIME = ROOT / "data" / "refresh" / "runtime.json"

TRAJ_CFG = ROOT / "models" / "traj_config.json"
CALIB = ROOT / "models" / "traj_calibration.json"
RELI = ROOT / "models" / "traj_reliability.json"
TRAJ_XGB = {n: ROOT / "models" / f"traj_xgb_{n}.json"
            for n in ("q05", "q50", "q95")}

TRAJ_DL_CFG = ROOT / "models" / "traj_dl_config.json"
TRAJ_MODEL = ROOT / "models" / "traj_dl.pt"
TRAJ_CALIB = ROOT / "models" / "traj_dl_calibration.json"
TRAJ_RELI = ROOT / "models" / "traj_dl_reliability.json"
FWD_CFG = ROOT / "models" / "pooled_dl_config.json"
FWD_MODEL = ROOT / "models" / "pooled_dl.pt"
FWD_EVAL = ROOT / "outputs" / "pooled_dl_eval.json"

MAX_H = 120
FORECAST_STEPS = 64          # Open-Meteo 16 forecast days on the 6h grid
STEP_H = 6

_XGB_CACHE: dict = {}
_XGB_SIG = None
_TRAJ_CACHE: dict = {}
_TRAJ_SIG = None
_FWD_CACHE: dict = {}
_FWD_SIG = None


def clear_caches() -> None:
    """Invalidate all three engine caches after a retrain or table rebuild."""
    global _XGB_SIG, _TRAJ_SIG, _FWD_SIG
    for cache in (_XGB_CACHE, _TRAJ_CACHE, _FWD_CACHE):
        cache.clear()
    _XGB_SIG = _TRAJ_SIG = _FWD_SIG = None


# --------------------------------------------------------------------------- #
# shared causal helpers (identical for both backends)
# --------------------------------------------------------------------------- #
def _signature(paths) -> tuple:
    parts = []
    for p in paths:
        try:
            parts.append((p.name, p.stat().st_mtime_ns))
        except OSError:
            parts.append((p.name, None))
    return tuple(parts)


def _clean_table() -> pd.DataFrame:
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    return F.drop_gwl_spikes(tbl)


def _bucket(h: int, rel: dict) -> int:
    bh = [int(b) for b in rel["bucket_rules"]]
    b = np.searchsorted(bh, h, side="right") - 1
    return bh[max(0, b)]


def _widen_s(h: int, cal: dict | None) -> float:
    if not cal:
        return 1.0
    return float(np.interp(h * STEP_H, list(cal["horizons_h"]), list(cal["widening_s"])))


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


def _station_context(sta: pd.DataFrame, anchor_t, anc: float):
    """Evidence shared by both engines: recency, coverage, OOD, stability."""
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
    station_row = {"integrity": integrity, "integrity_reason": ireason,
                   "ood": ood, "stability": 1.0}
    return station_row, recency_d, cover90


def _anchor_frame(tbl: pd.DataFrame, station: str):
    """Latest observed anchor + causal feature frame shared by both engines."""
    sta = tbl[tbl["Station"].astype(str) == station].sort_values("time")
    if sta.empty:
        return None, {"error": f"no data for station {station}", "reason": "no_data"}
    hist = sta.dropna(subset=["gwl"])
    if hist.empty:
        return None, {"error": "no observed GWL", "reason": "no_observed_gwl"}
    anchor_t = hist["time"].iloc[-1]
    hist_d = sta[sta["time"] <= anchor_t].copy()
    hist_d["date"] = hist_d["time"].dt.normalize()
    feats, _ = F.build_full(hist_d, keep_na=True, horizon_steps=MAX_H, horizon_label=30)
    feats = feats.sort_values("time")
    if feats.empty:
        return None, {"error": "no feature frame", "reason": "no_feature_frame"}
    last = feats.iloc[-1:].copy()
    anc = float(last["gwl"].iloc[0])
    if not np.isfinite(anc):
        return None, {"error": "anchor GWL invalid", "reason": "anchor_invalid"}
    return (sta, anchor_t, hist_d, anc, feats, last), None


def _endpoint(ffwd, station: str) -> dict | None:
    ff = ffwd(station) if ffwd else None
    if not ff:
        return None
    return {"level": ff.get("q50_level"), "q05": ff.get("q05_level"),
            "q95": ff.get("q95_level"), "anchor": ff.get("anchor"),
            "band_half": ff.get("band_half")}


def _response(station, anchor_t, anc, model, q50, lo, hi, rel, district,
              om_districts, station_row, recency_d, cover90, flips, endpoint):
    """Shape the 120-point response identically for XGB and GRU."""
    points = []
    for h in range(1, MAX_H + 1):
        br = rel["bucket_rules"][str(_bucket(h, rel))]
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
            "gwl": float(q50[h - 1]), "q05": float(lo[h - 1]),
            "q50": float(q50[h - 1]), "q95": float(hi[h - 1]),
            "confidence_level": lvl, "reliability_reason": reason,
            "driver_source": dsrc,
        })

    end_change = float(q50[MAX_H - 1] - anc)
    direction = "stable"
    if abs(end_change) >= 0.15:
        direction = "declining" if end_change < 0 else "rising"
    br30 = rel["bucket_rules"][str(_bucket(MAX_H, rel))]
    ep_agreement = None
    if endpoint and endpoint["level"] is not None:
        ep_change = float(endpoint["level"] - anc)
        ep_agreement = bool(np.sign(ep_change) == np.sign(end_change)
                            or abs(end_change) < 0.15)
    lvl_30, reason_30 = _conf_level(MAX_H, row, station_row, br30, rel)

    return {
        "station": station,
        "anchor_time": pd.Timestamp(anchor_t).isoformat(),
        "anchor_gwl": anc,
        "model": model,
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
            "level": lvl_30, "reason": reason_30,
            "endpoint_match": ep_agreement,
        },
        "evidence": {
            "station_integrity": station_row,
            "anchor_ood": bool(station_row["ood"] < 1.0),
            "stability_oscillation": flips,
            "recency_days": round(recency_d, 1),
            "recent90_coverage": round(cover90, 3),
        },
    }


# --------------------------------------------------------------------------- #
# surface (a): direct multi-horizon XGBoost
# --------------------------------------------------------------------------- #
def _xgb_signature():
    paths = [TRAJ_CFG, CALIB, RELI, ALIGNED, OM, OM_META, RUNTIME]
    paths += list(TRAJ_XGB.values())
    return _signature(paths)


def _load_xgb():
    global _XGB_SIG
    sig = _xgb_signature()
    if _XGB_CACHE and sig == _XGB_SIG:
        return _XGB_CACHE
    import xgboost as xgb
    try:
        from serving_core import forward_forecast
    except Exception:  # noqa: BLE001 - serving_core needs streamlit; optional here
        forward_forecast = None
    cfg = json.loads(TRAJ_CFG.read_text())
    fnames = list(cfg["feature_cols"])
    models = {n: xgb.Booster(model_file=str(p)) for n, p in TRAJ_XGB.items()}
    cal = json.loads(CALIB.read_text()) if CALIB.exists() else None
    rel = json.loads(RELI.read_text()) if RELI.exists() else None
    districts_om = set()
    if OM.exists():
        districts_om = set(pd.read_parquet(OM, columns=["District"])["District"].astype(str))
    clim_stations = set()
    if OM_META.exists():
        clim_stations = set(json.loads(OM_META.read_text()).get("stations", []))
    feature_mode = "flat"
    try:
        if RUNTIME.exists():
            feature_mode = json.loads(RUNTIME.read_text()).get("feature_mode", "flat")
    except Exception:  # noqa: BLE001 - corrupt runtime file must not break the app
        feature_mode = "flat"
    om_frame = clim_frame = rain_clim_frame = None
    if feature_mode == "future":
        om_frame = pd.read_parquet(OM) if OM.exists() else None
        clim_frame = pd.read_parquet(CLIM) if CLIM.exists() else None
        rain_clim_frame = pd.read_parquet(RAIN_CLIM) if RAIN_CLIM.exists() else None

    _XGB_CACHE.clear()
    _XGB_CACHE.update(models=models, fnames=fnames, cal=cal, rel=rel,
                      districts_om=districts_om, clim_stations=clim_stations,
                      feature_mode=feature_mode, om_frame=om_frame,
                      clim_frame=clim_frame, rain_clim_frame=rain_clim_frame,
                      tbl=_clean_table(), forward_forecast=forward_forecast)
    _XGB_SIG = sig
    return _XGB_CACHE


def xgb_trajectory_forecast(station: str) -> dict:
    c = _load_xgb()
    models, fnames, cal, rel = c["models"], c["fnames"], c["cal"], c["rel"]
    if rel is None:
        return {"error": "trajectory reliability tables not built (run 33)",
                "reason": "artifacts_missing"}
    if cal is None:
        return {"error": "trajectory calibration not built (run 32)",
                "reason": "artifacts_missing"}
    frame, err = _anchor_frame(c["tbl"], station)
    if err:
        return err
    sta, anchor_t, hist_d, anc, feats, last = frame

    hs = np.arange(1, MAX_H + 1, dtype=np.float32)
    if c["feature_mode"] == "future":
        from refresh.refresh_data import _load_river_forecast, build_step_features
        district = str(sta["District"].iloc[0])
        rfc = _load_river_forecast() if RFC.exists() else None
        X, _ = build_step_features(
            station=station, district=district, anchor_t=anchor_t, hist=hist_d,
            fnames=fnames, om_frame=c["om_frame"], climatology=c["clim_frame"],
            rain_clim=c["rain_clim_frame"], river_forecast=rfc,
            forecast_days=int(FORECAST_STEPS / 4), steps=MAX_H,
            mode="future", anchor_gwl=anc)
    else:
        base_cols = [x for x in fnames if x not in ("h", "h_sin", "h_cos")]
        base = np.asarray(last[base_cols].values[0], dtype=np.float32)
        X = np.hstack([np.repeat(base[None, :], MAX_H, axis=0), hs[:, None],
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

    station_row, recency_d, cover90 = _station_context(sta, anchor_t, anc)
    q50_arr = np.asarray(q50, dtype=float)
    flips = int(np.sum(np.diff(np.sign(np.diff(q50_arr))) != 0)) if MAX_H > 2 else 0
    station_row["stability"] = 0.7 if flips > int(0.25 * (MAX_H - 2)) else 1.0
    return _response(station, anchor_t, anc,
                     "direct multi-horizon shared XGBoost (q05/q50/q95)",
                     q50_arr, lo, hi, rel, str(sta["District"].iloc[0]).upper(),
                     c["districts_om"], station_row, recency_d, cover90, flips,
                     _endpoint(c["forward_forecast"], station))


# --------------------------------------------------------------------------- #
# surface (a): sequence GRU
# --------------------------------------------------------------------------- #
def _traj_signature():
    return _signature((ALIGNED, TRAJ_DL_CFG, TRAJ_MODEL, TRAJ_CALIB,
                       TRAJ_RELI, RUNTIME))


def _load_traj():
    global _TRAJ_SIG
    sig = _traj_signature()
    if _TRAJ_CACHE and sig == _TRAJ_SIG:
        return _TRAJ_CACHE
    import torch
    torch.set_num_threads(1)
    cfg = json.loads(TRAJ_DL_CFG.read_text())
    dlmod = importlib.import_module("pipeline_traj_dl")
    net = dlmod.TrajGRU(len(cfg["feature_cols"]), hidden=int(cfg["hidden"]))
    net.load_state_dict(torch.load(TRAJ_MODEL, map_location="cpu", weights_only=True))
    net.eval()
    cal = json.loads(TRAJ_CALIB.read_text()) if TRAJ_CALIB.exists() else None
    rel = json.loads(TRAJ_RELI.read_text()) if TRAJ_RELI.exists() else None
    districts_om = set()
    if OM.exists():
        districts_om = set(pd.read_parquet(OM, columns=["District"])["District"].astype(str))

    _TRAJ_CACHE.clear()
    _TRAJ_CACHE.update(torch=torch, net=net, feat_cols=list(cfg["feature_cols"]),
                       window_w=int(cfg["window_w"]),
                       max_h=int(cfg.get("max_h", MAX_H)),
                       cal=cal, rel=rel, districts_om=districts_om,
                       tbl=_clean_table())
    _TRAJ_SIG = sig
    return _TRAJ_CACHE


def _predict_deltas(c, feats):
    """Sequence GRU forward pass -> (MAX_H, 3) sorted delta quantiles."""
    torch = c["torch"]
    feat_cols = c["feat_cols"]
    W = int(c["window_w"])
    Xseq = feats[feat_cols].to_numpy(np.float32)
    Xseq = np.where(np.isfinite(Xseq), Xseq, 0.0).astype(np.float32)
    n = len(Xseq)
    w = min(W, n)
    seq = np.zeros((1, W, len(feat_cols)), np.float32)
    seqm = np.zeros((1, W, 1), np.float32)
    seq[0, W - w:] = Xseq[-w:]
    seqm[0, W - w:, 0] = 1.0
    static = Xseq[-1]
    hs = np.arange(1, MAX_H + 1, dtype=np.float32)
    hf = np.stack([hs / MAX_H, np.sin(2 * np.pi * hs / MAX_H),
                   np.cos(2 * np.pi * hs / MAX_H)], axis=-1)[None].astype(np.float32)
    with torch.no_grad():
        pred = c["net"](torch.from_numpy(seq), torch.from_numpy(seqm),
                        torch.from_numpy(static[None, :]),
                        torch.from_numpy(hf))[0].numpy()
    return np.sort(pred, axis=1)


def dl_trajectory_forecast(station: str) -> dict:
    c = _load_traj()
    cal, rel = c["cal"], c["rel"]
    if rel is None:
        return {"error": "DL trajectory reliability tables not built (run 33 with TRAJ_MODEL=dl)",
                "reason": "artifacts_missing"}
    if cal is None:
        return {"error": "DL trajectory calibration not built (run 32 with TRAJ_MODEL=dl)",
                "reason": "artifacts_missing"}
    frame, err = _anchor_frame(c["tbl"], station)
    if err:
        return err
    sta, anchor_t, _hist_d, anc, feats, _last = frame

    pred = _predict_deltas(c, feats)
    q05 = pred[:, 0] + anc
    q50 = np.asarray(pred[:, 1] + anc, dtype=float)
    q95 = pred[:, 2] + anc
    lo = np.minimum(q05, q50 - 1e-6)
    hi = np.maximum(q95, q50 + 1e-6)
    widen = np.array([_widen_s(h, cal) for h in range(1, MAX_H + 1)])
    lo = q50 - widen * (q50 - lo)
    hi = q50 + widen * (hi - q50)

    station_row, recency_d, cover90 = _station_context(sta, anchor_t, anc)
    flips = int(np.sum(np.diff(np.sign(np.diff(q50))) != 0)) if MAX_H > 2 else 0
    station_row["stability"] = 0.7 if flips > int(0.25 * (MAX_H - 2)) else 1.0

    try:
        from serving_core import forward_forecast
        ff = forward_forecast(station)
    except Exception:  # noqa: BLE001 - serving_core imports streamlit; optional here
        ff = None
    return _response(station, anchor_t, anc,
                     "sequence GRU + horizon-conditioned quantile heads (q05/q50/q95)",
                     q50, lo, hi, rel, str(sta["District"].iloc[0]).upper(),
                     c["districts_om"], station_row, recency_d, cover90, flips,
                     _endpoint(lambda s: ff, station))


# --------------------------------------------------------------------------- #
# surface (b): pooled daily GRU
# --------------------------------------------------------------------------- #
def _fwd_signature():
    return _signature((ALIGNED, FWD_CFG, FWD_MODEL))


def _load_fwd():
    global _FWD_SIG
    sig = _fwd_signature()
    if _FWD_CACHE and sig == _FWD_SIG:
        return _FWD_CACHE
    import torch
    torch.set_num_threads(1)
    cfg = json.loads(FWD_CFG.read_text())
    states = torch.load(FWD_MODEL, map_location="cpu", weights_only=True)
    mod = importlib.import_module("pipeline_traj_dl")
    nets = []
    for st in states:
        net = mod.GRUNetQ(2 * len(cfg["feature_cols"]), nq=len(cfg["quantiles"]),
                          hidden=int(cfg["hidden"]))
        net.load_state_dict(st)
        net.eval()
        nets.append(net)
    _FWD_CACHE.clear()
    _FWD_CACHE.update(torch=torch, nets=nets, cfg=cfg, tbl=_clean_table(),
                      quantiles=list(cfg["quantiles"]),
                      window_w=int(cfg["window_w"]),
                      cols=list(cfg["feature_cols"]),
                      widen_k=float(cfg.get("widen_k", 1.0)), norm=cfg["norm"],
                      mod=mod)
    _FWD_SIG = sig
    return _FWD_CACHE


def _daily_last(feats: pd.DataFrame) -> pd.DataFrame:
    f = feats.sort_values("time").copy()
    f["_d"] = f["time"].dt.normalize()
    return f.groupby("_d", sort=True).tail(1).sort_values("_d")


def _predict_delta(c, last: pd.DataFrame):
    """Return the (q05,q50,q95) 30-day DELTA from the ensemble."""
    torch = c["torch"]
    cols, W = c["cols"], c["window_w"]
    norm = c["norm"]
    X = last[cols].to_numpy(np.float32)
    M = np.isfinite(X).astype(np.float32)
    FTR = np.concatenate([np.where(M, X, 0.0).astype(np.float32), M], axis=1)
    gwl = last["gwl"].to_numpy(np.float64)
    dw = np.empty_like(gwl)
    dw[0] = 0.0
    dw[1:] = gwl[1:] - gwl[:-1]
    dw[np.isnan(gwl)] = np.nan
    G = np.stack([gwl, dw], axis=1).astype(np.float32)

    mu = np.asarray(norm["f_mu"], np.float32)
    sd = np.asarray(norm["f_sd"], np.float32)
    Fn = ((FTR - mu) / sd).astype(np.float32)
    gmu = np.asarray(norm["g_mu"], np.float32)
    gsd = np.asarray(norm["g_sd"], np.float32)
    Gn = ((G - gmu) / gsd).astype(np.float32)
    ym, ys = float(norm["y_mu"]), float(norm["y_sd"])

    w = min(W, len(Gn))
    seq = np.zeros((1, W, 2), np.float32)
    seq[0, W - w:] = Gn[-w:]
    mask = np.zeros((1, W, 1), np.float32)
    mask[0, W - w:, 0] = 1.0
    feat = Fn[-1:].astype(np.float32)
    t_seq = torch.from_numpy(seq)
    t_mask = torch.from_numpy(mask)
    t_feat = torch.from_numpy(feat)
    preds = []
    with torch.no_grad():
        for net in c["nets"]:
            preds.append(net(t_seq, t_mask, t_feat)[0].numpy())
    d = np.sort(np.mean(preds, axis=0) * ys + ym)
    return float(d[0]), float(d[1]), float(d[2])


def dl_forward_forecast(station: str) -> dict:
    c = _load_fwd()
    tbl = c["tbl"]
    t = tbl[tbl["Station"].astype(str) == station].copy()
    if t.empty:
        return {}
    t = t.dropna(subset=["time"]).sort_values("time")
    t["date"] = t["time"].dt.normalize()
    feats, _ = F.build_full(t, keep_na=True)
    if feats.empty or "gwl" not in feats.columns:
        return {}
    last_rows = _daily_last(feats)
    if last_rows.empty:
        return {}
    anchor = float(last_rows["gwl"].iloc[-1])
    if not np.isfinite(anchor):
        return {}
    d05, d50, d95 = _predict_delta(c, last_rows)

    k = c["widen_k"]
    med = anchor + d50
    lo = min(med - k * (med - (anchor + d05)), med)
    hi = max(med + k * ((anchor + d95) - med), med)
    date_from = pd.to_datetime(last_rows["time"].iloc[-1])
    stride = None
    if FWD_EVAL.exists():
        try:
            stride = float(json.loads(FWD_EVAL.read_text())["DL"]["rmse"])
        except Exception:  # noqa: BLE001
            stride = None
    return {
        "station": station, "date_from": date_from,
        "date_to": date_from + pd.Timedelta(days=30), "anchor": anchor,
        "pred_xgb": d50, "pred_ridge": None,
        "xgb_level": anchor + d50, "ridge_level": None,
        "q05_level": lo, "q50_level": med, "q95_level": hi,
        "widen_k": k, "band_half": float((hi - lo) / 2),
        "station_stride_rmse": stride,
    }


# Backward-compatible names for callers that used the merged GRU entry points.
trajectory_forecast = dl_trajectory_forecast
forward_forecast = dl_forward_forecast


def preview(station: str, *, engine: str = "dl") -> None:
    fc = ((dl_trajectory_forecast if engine == "dl" else xgb_trajectory_forecast)(station)
          if "--forward" not in sys.argv else dl_forward_forecast(station))
    if not fc or "error" in fc:
        print((fc or {}).get("error", "empty forecast"))
        return
    counts = {c: 0 for c in ("HIGH", "DIRECTIONAL", "LOW")}
    for p in fc.get("trajectory", []):
        counts[p["confidence_level"]] += 1
    print(f"anchor {fc['anchor']:.3f} m @ {fc.get('anchor_time', '-')}")
    print(f"conf: {counts}")
    if "trajectory_30d" in fc:
        print(f"traj30d: {fc['trajectory_30d']}")
        print(f"direction: {fc['direction']}")
        print(f"endpoint: {fc['endpoint_production']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--forward", "--xgb")]
    engine = "xgb" if "--xgb" in sys.argv else "dl"
    if "--forward" in sys.argv:
        print(json.dumps(dl_forward_forecast(args[0] if args else
                                             "ASHADHA PRATHMIK VIDYALAYA"),
                         indent=2, default=str))
    else:
        preview(args[0] if args else "ASHADHA PRATHMIK VIDYALAYA", engine=engine)
