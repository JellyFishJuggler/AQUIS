"""_dl.py - DL (quantile GRU) serving for both production surfaces.

Two streamlit-free engines that are drop-in siblings of the incumbent XGBoost
paths, selected at runtime via ``data/refresh/runtime.json``:

* ``trajectory_forecast`` - 6-hourly trajectory (surface a). Same output schema
  and per-horizon calibration/confidence framework as
  ``_trajectory.trajectory_forecast``; only the q05/q50/q95 deltas come from the
  sequence GRU (``models/traj_dl.pt``, built by ``34_train_traj_dl.py``). Flag:
  ``{{"traj_backend": "dl"}}``.
* ``forward_forecast`` - pooled 30-day delta (surface b). Same output schema as
  ``_model.forward_forecast``; the delta + q05/q50/q95 come from the pooled
  daily-GRU quantile ensemble (``models/pooled_dl.pt``, built by
  ``36_train_pooled_dl.py``). Flag: ``{{"forward_backend": "dl"}}``.

No streamlit dependency: reads ``data/aligned/table_6h.parquet`` directly and
imports torch lazily, so it is importable/testable in the torch-only env
(venv312). Unit-tested for schema parity against the incumbent backends.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

F = importlib.import_module("06_features")

from _trajectory import (  # noqa: E402 - shared pure helpers keep parity
    _bucket, _widen_s, _conf_level, _driver_score,
    MAX_H, FORECAST_STEPS, STEP_H, RUNTIME,
)

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"

TRAJ_CFG = ROOT / "models" / "traj_dl_config.json"
TRAJ_MODEL = ROOT / "models" / "traj_dl.pt"
TRAJ_CALIB = ROOT / "models" / "traj_dl_calibration.json"
TRAJ_RELI = ROOT / "models" / "traj_dl_reliability.json"
OM = ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet"

FWD_CFG = ROOT / "models" / "pooled_dl_config.json"
FWD_MODEL = ROOT / "models" / "pooled_dl.pt"
FWD_EVAL = ROOT / "outputs" / "pooled_dl_eval.json"

_TRAJ_CACHE: dict = {}
_TRAJ_SIG = None
_FWD_CACHE: dict = {}
_FWD_SIG = None


def clear_caches() -> None:
    """Invalidate both engines' caches (a long-running scheduler/API calls this
    after a retrain or table rebuild instead of serving the process-lifetime
    copy)."""
    _TRAJ_CACHE.clear()
    _FWD_CACHE.clear()


def _traj_signature():
    """Mtime signature of every artifact the trajectory engine caches (a
    reloaded model / rebuilt calibration / refreshed table invalidates the cache
    instead of serving the process-lifetime copy)."""
    parts = []
    for p in (ALIGNED, TRAJ_CFG, TRAJ_MODEL, TRAJ_CALIB, TRAJ_RELI, RUNTIME):
        try:
            parts.append((p.name, p.stat().st_mtime_ns))
        except OSError:
            parts.append((p.name, None))
    return tuple(parts)


def _fwd_signature():
    parts = []
    for p in (ALIGNED, FWD_CFG, FWD_MODEL):
        try:
            parts.append((p.name, p.stat().st_mtime_ns))
        except OSError:
            parts.append((p.name, None))
    return tuple(parts)


# --------------------------------------------------------------------------- #
# surface (a): 6-hourly trajectory - sequence GRU + horizon-conditioned heads
# --------------------------------------------------------------------------- #
def _load_traj():
    global _TRAJ_SIG
    sig = _traj_signature()
    if _TRAJ_CACHE and sig == _TRAJ_SIG:
        return _TRAJ_CACHE
    import torch
    torch.set_num_threads(1)
    cfg = json.loads(TRAJ_CFG.read_text())
    dlmod = importlib.import_module("34_train_traj_dl")
    net = dlmod.TrajGRU(len(cfg["feature_cols"]), hidden=int(cfg["hidden"]))
    net.load_state_dict(torch.load(TRAJ_MODEL, map_location="cpu", weights_only=True))
    net.eval()
    cal = json.loads(TRAJ_CALIB.read_text()) if TRAJ_CALIB.exists() else None
    rel = json.loads(TRAJ_RELI.read_text()) if TRAJ_RELI.exists() else None
    districts_om: set[str] = set()
    if OM.exists():
        districts_om = set(pd.read_parquet(OM, columns=["District"])["District"].astype(str))
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl = F.drop_gwl_spikes(tbl)

    _TRAJ_CACHE.clear()
    _TRAJ_CACHE.update(torch=torch, net=net, feat_cols=list(cfg["feature_cols"]),
                  window_w=int(cfg["window_w"]), max_h=int(cfg.get("max_h", MAX_H)),
                  cal=cal, rel=rel, districts_om=districts_om, tbl=tbl)
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


def trajectory_forecast(station: str) -> dict:
    c = _load_traj()
    cal, rel = c["cal"], c["rel"]
    if rel is None:
        return {"error": "DL trajectory reliability tables not built (run 33 with TRAJ_MODEL=dl)",
                "reason": "artifacts_missing"}
    if cal is None:
        return {"error": "DL trajectory calibration not built (run 32 with TRAJ_MODEL=dl)",
                "reason": "artifacts_missing"}

    tbl = c["tbl"]
    sta = tbl[tbl["Station"].astype(str) == station].sort_values("time")
    if sta.empty:
        return {"error": f"no data for station {station}", "reason": "no_data"}
    hist = sta.dropna(subset=["gwl"])
    if hist.empty:
        return {"error": "no observed GWL", "reason": "no_observed_gwl"}
    anchor_t = hist["time"].iloc[-1]
    hist_d = sta[sta["time"] <= anchor_t].copy()
    hist_d["date"] = hist_d["time"].dt.normalize()
    feats, _ = F.build_full(hist_d, keep_na=True, horizon_steps=120, horizon_label=30)
    feats = feats.sort_values("time")
    if feats.empty:
        return {"error": "no feature frame", "reason": "no_feature_frame"}
    last = feats.iloc[-1:].copy()
    anc = float(last["gwl"].iloc[0])
    if not np.isfinite(anc):
        return {"error": "anchor GWL invalid", "reason": "anchor_invalid"}

    pred = _predict_deltas(c, feats)
    q05 = pred[:, 0] + anc
    q50 = pred[:, 1] + anc
    q95 = pred[:, 2] + anc
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

    try:
        from _model import forward_forecast
        ff = forward_forecast(station)
    except Exception:  # noqa: BLE001 - _model imports streamlit; optional here
        ff = None
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
        "model": "sequence GRU + horizon-conditioned quantile heads (q05/q50/q95)",
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
    for p in out["trajectory"]:
        counts[p["confidence_level"]] += 1
    print(f"anchor {out['anchor_gwl']:.3f} m @ {out['anchor_time']}")
    print(f"conf: {counts}")
    print(f"traj30d: {out['trajectory_30d']}")
    print(f"direction: {out['direction']}")
    print(f"endpoint: {out['endpoint_production']}")


# --------------------------------------------------------------------------- #
# surface (b): pooled 30-day delta - daily-GRU quantile ensemble
# --------------------------------------------------------------------------- #
def _load_fwd():
    global _FWD_SIG
    sig = _fwd_signature()
    if _FWD_CACHE and sig == _FWD_SIG:
        return _FWD_CACHE
    import torch
    torch.set_num_threads(1)
    cfg = json.loads(FWD_CFG.read_text())
    states = torch.load(FWD_MODEL, map_location="cpu", weights_only=True)
    mod = importlib.import_module("36_train_pooled_dl")
    nets = []
    for st in states:
        net = mod.GRUNetQ(2 * len(cfg["feature_cols"]), nq=len(cfg["quantiles"]),
                          hidden=int(cfg["hidden"]))
        net.load_state_dict(st); net.eval()
        nets.append(net)
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl = F.drop_gwl_spikes(tbl)

    _FWD_CACHE.clear()
    _FWD_CACHE.update(torch=torch, nets=nets, cfg=cfg, tbl=tbl,
                  quantiles=list(cfg["quantiles"]), window_w=int(cfg["window_w"]),
                  cols=list(cfg["feature_cols"]), widen_k=float(cfg.get("widen_k", 1.0)),
                  norm=cfg["norm"], mod=mod)
    _FWD_SIG = sig
    return _FWD_CACHE


def _daily_last(feats: pd.DataFrame) -> pd.DataFrame:
    f = feats.sort_values("time").copy()
    f["_d"] = f["time"].dt.normalize()
    return f.groupby("_d", sort=True).tail(1).sort_values("_d")


def _predict_delta(c, last: pd.DataFrame):
    """Return the (q05,q50,q95) 30-day DELTA from the ensemble for the last day."""
    torch = c["torch"]
    cols, W = c["cols"], c["window_w"]
    norm = c["norm"]
    X = last[cols].to_numpy(np.float32)
    M = np.isfinite(X).astype(np.float32)
    FTR = np.concatenate([np.where(M, X, 0.0).astype(np.float32), M], axis=1)
    gwl = last["gwl"].to_numpy(np.float64)
    dw = np.empty_like(gwl); dw[0] = 0.0; dw[1:] = gwl[1:] - gwl[:-1]
    dw[np.isnan(gwl)] = np.nan
    G = np.stack([gwl, dw], axis=1).astype(np.float32)

    mu = np.asarray(norm["f_mu"], np.float32); sd = np.asarray(norm["f_sd"], np.float32)
    Fn = ((FTR - mu) / sd).astype(np.float32)
    gmu = np.asarray(norm["g_mu"], np.float32); gsd = np.asarray(norm["g_sd"], np.float32)
    Gn = ((G - gmu) / gsd).astype(np.float32)
    ym, ys = float(norm["y_mu"]), float(norm["y_sd"])

    w = min(W, len(Gn))
    seq = np.zeros((1, W, 2), np.float32); seq[0, W - w:] = Gn[-w:]
    mask = np.zeros((1, W, 1), np.float32); mask[0, W - w:, 0] = 1.0
    feat = Fn[-1:].astype(np.float32)
    t_seq = torch.from_numpy(seq); t_mask = torch.from_numpy(mask); t_feat = torch.from_numpy(feat)
    preds = []
    with torch.no_grad():
        for net in c["nets"]:
            preds.append(net(t_seq, t_mask, t_feat)[0].numpy())
    d = np.mean(preds, axis=0) * ys + ym
    d = np.sort(d)
    return float(d[0]), float(d[1]), float(d[2])


def forward_forecast(station: str) -> dict:
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
    lo = med - k * (med - (anchor + d05))
    hi = med + k * ((anchor + d95) - med)
    lo = min(lo, med); hi = max(hi, med)

    date_from = pd.to_datetime(last_rows["time"].iloc[-1])
    stride = None
    if FWD_EVAL.exists():
        try:
            stride = float(json.loads(FWD_EVAL.read_text())["DL"]["rmse"])
        except Exception:  # noqa: BLE001
            stride = None

    return {
        "station": station,
        "date_from": date_from,
        "date_to": date_from + pd.Timedelta(days=30),
        "anchor": anchor,
        "pred_xgb": d50,
        "pred_ridge": None,
        "xgb_level": anchor + d50,
        "ridge_level": None,
        "q05_level": lo,
        "q50_level": med,
        "q95_level": hi,
        "widen_k": k,
        "band_half": float((hi - lo) / 2),
        "station_stride_rmse": stride,
    }


# --------------------------------------------------------------------------- #
# surface (b): pooled 30-day delta - daily-GRU quantile ensemble
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if "--forward" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--forward"]
        print(json.dumps(forward_forecast(args[0] if args else
                                          "ASHADHA PRATHMIK VIDYALAYA"),
                         indent=2, default=str))
    else:
        preview(sys.argv[1] if len(sys.argv) > 1 else "ASHADHA PRATHMIK VIDYALAYA")
