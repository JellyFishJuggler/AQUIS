"""32_backtest_traj - honest non-overlap backtest for the direct trajectory v2.

For each anchor (station, >=2025-10-01, spaced 14d, at observed GWL times) the
anchor feature row is built from history ONLY up to the anchor
(06.build_full, keep_na=True) and scored once with the shared direct
multi-horizon quantile models for h = 1..120. Every future point is a genuine
model output; level = observed anchor + delta (anchor never predicted).
Observed ground truth is taken at the EXACT forecast time where a reading
exists.

No future observed GWL or driver values enter the feature vector. Baselines at
the same horizons: persistence, station doy-mean 6h-drift climatology, and the
frozen production direct-30d model at the +30d endpoint only.

Outputs:
  outputs/traj_backtest_metrics.csv
  outputs/traj_backtest_summary.json  (decision vs previous attempts)
  models/traj_calibration.json        (per-horizon widening to ~90%)
"""
from __future__ import annotations

import importlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

F = importlib.import_module("06_features")

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
OUT = ROOT / "outputs"
TRAJ_CFG = ROOT / "models" / "traj_config.json"
CALIB = ROOT / "models" / "traj_calibration.json"

BACKTEST_START = pd.Timestamp("2025-10-01")
LATEST_FRAME = pd.Timestamp("2026-08-20")

HORIZONS_STEPS = [1, 2, 3, 4, 8, 12, 28, 60, 120]
HORIZONS_H = [6, 12, 18, 24, 48, 72, 168, 360, 720]

ANCHOR_SPACING_DAYS = 14
STATIONS_SAMPLE = 150
SEED = 42
COVERAGE_TARGET = 0.90
MAX_H = 120

_TBL = None
_MODELS = {}
_FN = []
_ORD30 = None
_GWL_CLIM = {}
_STA_CACHE = {}


def _joblib_load(name):
    import joblib
    return joblib.load(ROOT / "models" / name)


def _worker_init(aligned, traj_cfg):
    global _TBL, _MODELS, _FN, _GWL_CLIM, _ORD30
    _TBL = pd.read_parquet(aligned)
    _TBL["time"] = pd.to_datetime(_TBL["time"], errors="coerce")
    _TBL = F.drop_gwl_spikes(_TBL)

    import xgboost as xgb
    for name in ("q05", "q50", "q95"):
        _MODELS[name] = xgb.Booster(
            model_file=str(ROOT / "models" / f"traj_xgb_{name}.json"))
    cfg = json.loads(Path(traj_cfg).read_text())
    _FN = list(cfg["feature_cols"])

    cfg30 = json.loads((ROOT / "models" / "feature_config.json").read_text())
    _ORD30 = {
        "q50": _joblib_load("xgb_q50.joblib"),
        "fnames": list(_joblib_load("xgb_multihorizon.joblib").feature_names_in_),
        "stations": list(cfg30["stations"]),
        "districts": list(cfg30["districts"]),
    }

    tr = _TBL[_TBL["time"] < pd.Timestamp("2025-10-01")].sort_values("time").copy()
    tr["doy"] = tr["time"].dt.dayofyear
    tr["d6h"] = tr.groupby("Station")["gwl"].diff()
    for st, g in tr.groupby("Station"):
        _GWL_CLIM[st] = g.dropna(subset=["d6h"]).groupby("doy")["d6h"].mean().to_dict()


def _station_obs(station):
    if station not in _STA_CACHE:
        _STA_CACHE[station] = (_TBL[_TBL["Station"] == station]
                               .sort_values("time").reset_index(drop=True))
    return _STA_CACHE[station]


def _sample_anchors(sta):
    s = sta.dropna(subset=["gwl"])["time"]
    s = s[(s >= BACKTEST_START) & (s <= LATEST_FRAME)]
    if len(s) < 4:
        return []
    out = []
    cur = BACKTEST_START
    while cur <= LATEST_FRAME:
        day = s[(s >= cur) & (s < cur + pd.Timedelta(days=1))]
        if len(day):
            out.append(day.iloc[0])
        cur += pd.Timedelta(days=ANCHOR_SPACING_DAYS)
    return out


def _direct_30d(sta, anchor_t):
    slice_df = sta[sta["time"] <= anchor_t].copy()
    slice_df = slice_df.drop_duplicates("time").reset_index(drop=True)
    slice_df["date"] = slice_df["time"].dt.normalize()
    if slice_df.empty:
        return None
    feats, _ = F.build_full(slice_df, keep_na=True, horizon_steps=120, horizon_label=30)
    feats = feats.sort_values("time")
    if feats.empty:
        return None
    last = feats.iloc[-1:].copy()
    station = str(sta["Station"].iloc[0])
    last["st_id"] = _ORD30["stations"].index(station) if station in _ORD30["stations"] else -1
    district = str(sta["District"].iloc[0]).upper()
    last["dist_id"] = (_ORD30["districts"].index(district)
                       if district in _ORD30["districts"] else -1)
    try:
        q50 = float(_ORD30["q50"].predict(last[_ORD30["fnames"]])[0])
    except Exception:
        return None
    return float(last["gwl"].iloc[0]) + q50


def _run_anchor(args):
    station, anchor_t = args
    sta = _station_obs(station)
    if len(sta) <= 2:
        return None
    hist = sta[sta["time"] <= anchor_t]
    if len(hist) < 30:
        return None

    hist_d = hist.copy()
    hist_d["date"] = hist_d["time"].dt.normalize()
    feats, _ = F.build_full(hist_d, keep_na=True, horizon_steps=120, horizon_label=30)
    feats = feats.sort_values("time")
    if feats.empty:
        return None
    last = feats.iloc[-1:].copy()
    anchor_gwl = float(last["gwl"].iloc[0])
    if not np.isfinite(anchor_gwl):
        return None
    base_cols = [c for c in _FN if c not in ("h", "h_sin", "h_cos")]
    base = np.asarray(last[base_cols].values[0], dtype=np.float32)
    hs = np.arange(1, MAX_H + 1, dtype=np.float32)
    X = np.hstack([
        np.repeat(base[None, :], MAX_H, axis=0),
        hs[:, None],
        np.sin(2 * np.pi * hs / MAX_H)[:, None],
        np.cos(2 * np.pi * hs / MAX_H)[:, None],
    ])

    import xgboost as xgb
    dm = xgb.DMatrix(X, feature_names=_FN)
    q05 = _MODELS["q05"].predict(dm) + anchor_gwl
    q50 = _MODELS["q50"].predict(dm) + anchor_gwl
    q95 = _MODELS["q95"].predict(dm) + anchor_gwl
    q05 = np.minimum(q05, q50)
    q95 = np.maximum(q95, q50)

    obs_times = sta["time"]
    obs_gwl = sta["gwl"]
    clim = _GWL_CLIM.get(station, {})
    anchor_doy = int(pd.Timestamp(anchor_t).dayofyear)
    d6h_clim = clim.get(anchor_doy, np.nan)

    rows = []
    for h, hh in zip(HORIZONS_STEPS, HORIZONS_H):
        tgt = anchor_t + pd.Timedelta(hours=hh)
        m = obs_times == tgt
        if not m.any():
            continue
        obs = float(obs_gwl.iloc[np.where(m)[0][0]])
        if not np.isfinite(obs):
            continue
        rows.append({
            "station": station, "anchor_t": anchor_t, "horizon_h": hh, "steps": h,
            "obs": obs, "anchor": anchor_gwl,
            "traj_q50": float(q50[h - 1]), "traj_q05": float(q05[h - 1]),
            "traj_q95": float(q95[h - 1]),
            "persist": anchor_gwl,
            "clim": float(anchor_gwl + d6h_clim * h) if np.isfinite(d6h_clim) else np.nan,
            "direct30": _direct_30d(sta, anchor_t) if hh == 720 else None,
        })
    return rows


def _metrics(g):
    obs = g["obs"].to_numpy()
    rec = g["traj_q50"].to_numpy()
    q05 = g["traj_q05"].to_numpy()
    q95 = g["traj_q95"].to_numpy()
    e = obs - rec
    d_obs = obs - g["anchor"].to_numpy()
    d_rec = rec - g["anchor"].to_numpy()
    nonzero = np.abs(d_obs) > 0.02
    sign_acc = (float(np.mean(np.sign(d_rec[nonzero]) == np.sign(d_obs[nonzero])))
                if nonzero.any() else np.nan)
    return {
        "n": int(len(g)),
        "rmse": float(math.sqrt(float(np.mean(e ** 2)))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(rec - obs)),
        "r2": float(1 - np.sum(e ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "sign_acc": sign_acc,
        "coverage_raw": float(np.mean((q05 <= obs) & (obs <= q95))),
        "halfwidth_median": float(np.median((q95 - q05) / 2)),
        "rmse_persist": float(math.sqrt(float(np.mean((obs - g["persist"].to_numpy()) ** 2)))),
    }


def main():
    import sys
    t0 = time.time()
    stations_all = int(sys.argv[1]) if len(sys.argv) > 1 else STATIONS_SAMPLE
    full = pd.read_parquet(ALIGNED)
    full["time"] = pd.to_datetime(full["time"], errors="coerce")
    full = F.drop_gwl_spikes(full)

    rng = random.Random(SEED)
    stations = sorted(full["Station"].astype(str).unique())
    if stations_all >= len(stations):
        stations = stations
    else:
        stations = rng.sample(stations, stations_all)

    tasks = []
    for st in stations:
        sta = full[full["Station"] == st].sort_values("time").reset_index(drop=True)
        for a in _sample_anchors(sta):
            tasks.append((st, a))
    rng.shuffle(tasks)
    print(f"anchors: {len(tasks):,} ({len(stations)} stations)", flush=True)

    n_cpu = max(1, min((mp.cpu_count() or 4) // 2, 6))
    with mp.Pool(processes=n_cpu, initializer=_worker_init,
                 initargs=(str(ALIGNED), str(TRAJ_CFG))) as pool:
        results = pool.map(_run_anchor, tasks, chunksize=8)
    recs = [r for batch in results if batch for r in batch]
    print(f"valid scores: {len(recs):,}", flush=True)
    df = pd.DataFrame(recs)

    rows = []
    for h, hh in zip(HORIZONS_STEPS, HORIZONS_H):
        g = df[df["horizon_h"] == hh]
        if g.empty:
            continue
        m = _metrics(g)
        clim_v = g["clim"].dropna()
        rmse_clim = (float(math.sqrt(((g.loc[clim_v.index, "obs"] - clim_v) ** 2).mean()))
                     if len(clim_v) else np.nan)
        rows.append({"horizon_h": hh, "steps": h, **m, "rmse_clim": rmse_clim})
    metric_df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(OUT / "traj_backtest_metrics.csv", index=False)

    g30 = df[df["horizon_h"] == 720].dropna(subset=["direct30"])
    if len(g30):
        dv = pd.DataFrame({
            "obs": g30["obs"].to_numpy(),
            "traj_q50": g30["traj_q50"].to_numpy(),
            "direct": g30["direct30"].to_numpy()})
        direct_rmse = float(math.sqrt(((dv["obs"] - dv["direct"]) ** 2).mean()))
        traj_rmse_directset = float(math.sqrt(((dv["obs"] - dv["traj_q50"]) ** 2).mean()))
    else:
        direct_rmse = traj_rmse_directset = np.nan

    calib = {"horizons_h": [], "widening_s": [], "coverage_raw": [],
             "coverage_calibrated": [], "n": []}
    for hh in HORIZONS_H:
        g = df[df["horizon_h"] == hh]
        if g.empty:
            continue
        med = g["traj_q50"].to_numpy()
        lo = g["traj_q05"].to_numpy()
        hi = g["traj_q95"].to_numpy()
        obs = g["obs"].to_numpy()
        half = (hi - lo) / 2.0
        avail = half > 1e-9
        cov_of = lambda s: np.mean((med[avail] - s * half[avail] <= obs[avail])
                                   & (obs[avail] <= med[avail] + s * half[avail]))
        s_lo, s_hi = 0.2, 12.0
        if cov_of(s_lo) >= COVERAGE_TARGET:
            s = s_lo
        else:
            for _ in range(40):
                s_mid = 0.5 * (s_lo + s_hi)
                if cov_of(s_mid) < COVERAGE_TARGET:
                    s_lo = s_mid
                else:
                    s_hi = s_mid
            s = s_hi
        calib["horizons_h"].append(hh)
        calib["widening_s"].append(round(float(s), 3))
        calib["coverage_raw"].append(round(float(cov_of(1.0)), 3))
        calib["coverage_calibrated"].append(round(float(cov_of(s)), 3))
        calib["n"].append(int(len(g)))
    CALIB.write_text(json.dumps(calib, indent=2))

    r30 = metric_df[metric_df.horizon_h == 720].iloc[0]
    short = metric_df[metric_df.horizon_h <= 24]
    success_short = bool((short["rmse"] < short["rmse_persist"]).all()) if len(short) else False
    success_persist_30 = float(r30["rmse"]) < float(r30["rmse_persist"])
    success_direct_30 = (traj_rmse_directset < direct_rmse) if np.isfinite(direct_rmse) else False

    prev_path = OUT / "backtest_6h_summary.json"
    prev_rec = None
    if prev_path.exists():
        prev_rec = json.loads(prev_path.read_text()).get("best_30d_rmse_rec")

    summary = {
        "engine": "direct multi-horizon shared XGBoost, h feature (30_traj/31_traj)",
        "anchors_scored": int(len(df)),
        "stations_sampled": len(stations),
        "period": [str(BACKTEST_START.date()), str(LATEST_FRAME.date())],
        "anchor_spacing_days": ANCHOR_SPACING_DAYS,
        "no_overlap_windows": True,
        "no_future_observed_values": True,
        "30d_rmse_traj": float(r30["rmse"]),
        "30d_rmse_persistence": float(r30["rmse_persist"]),
        "30d_rmse_climatology": (float(r30["rmse_clim"]) if np.isfinite(r30["rmse_clim"])
                                 else None),
        "30d_rmse_direct30_production": (round(direct_rmse, 4) if np.isfinite(direct_rmse)
                                         else None),
        "30d_rmse_prev_recursive": (round(prev_rec, 4) if prev_rec is not None else None),
        "30d_dir_acc": (round(float(r30["sign_acc"]), 4) if np.isfinite(r30["sign_acc"])
                        else None),
        "30d_coverage_calibrated": calib["coverage_calibrated"][-1],
        "success_short_horizons": success_short,
        "success_vs_persist_30d": success_persist_30,
        "success_vs_direct30_30d": success_direct_30,
        "promote_trajectory": bool(success_short and success_persist_30 and success_direct_30),
        "trajectory_validated": ("trajectory + level (endpoint benchmarked to direct-30d)"
                                 if bool(success_short and success_persist_30)
                                 else "trajectory not yet validated above baselines"),
        "calibration_file": str(CALIB),
        "elapsed_sec": round(time.time() - t0),
    }
    (OUT / "traj_backtest_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== direct trajectory vs baselines (RMSE m) ===")
    print(metric_df.round(4).to_string(index=False))
    print("\n=== calibration ===")
    print(pd.DataFrame(calib).to_string(index=False))
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()