"""18_backtest_6h — honest backtest + per-horizon calibration for the recursive
6-hourly trajectory engine (17_recursive).

Method
------
* Test/backtest period: observations at/after 2025-10-01 (the pipeline train cut),
  ending BEFORE the latest observed frame used by the live page. Anchors are
  spaced > 30 days apart so no two evaluation windows overlap on the observed GWL —
  every score is a genuine out-of-sample multi-step forecast.
* For each anchor the +6h recursive engine (17_recursive.recursive_trajectory) is
  run on ONLY the observation history <= anchor: no future observed drivers, no
  future observed GWL. q50 median path = headline.
* Horzion grid: 6h | 12h | 1d | 3d | 7d | 15d | 30d (1|2|4|12|28|60|120 steps).
* Baselines at the SAME horizons: persistence (anchor carried forward),
  station-day-of-year GWL climatology (training-period mean), and the frozen
  direct 30-day endpoint model at 30d.
* Calibration: per-horizon widening factor s for the raw q05/q95 band so the
  band reaches ~90% empirical coverage (bisection over all pooled anchors).

Outputs
-------
* models/backtest_6h_calibration.json   -> per-horizon s + coverage (drives app)
* outputs/backtest_6h_metrics.csv       -> RMSE/MAE/R2/bias per horizon per model
* outputs/backtest_6h_summary.json      -> decision + success criteria
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import importlib

R = importlib.import_module("17_recursive")
F = importlib.import_module("06_features")

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
OUT = ROOT / "outputs"

TRAIN_CUT = pd.Timestamp("2025-10-01")
BACKTEST_START = pd.Timestamp("2025-10-01")   # first allowed anchor
LATEST_FRAME = pd.Timestamp("2026-08-20")     # anchors end 30d before data end

HORIZONS_STEPS = [1, 2, 4, 12, 28, 60, 120]   # 6h..30d
HORIZONS_H = [6, 12, 24, 72, 168, 360, 720]

ANCHOR_SPACING_DAYS = 14
STATIONS_SAMPLE = 150
SEED = 42
COVERAGE_TARGET = 0.90

_TBL: pd.DataFrame | None = None
_MODELS: dict | None = None
_ORD: tuple[list[str], list[str]] | None = None
_NUMCOLS: list[str] = []
_GWL_CLIM: dict[str, dict[int, float]] = {}
_DIRECT30: dict | None = None
_STA_CACHE: dict[str, pd.DataFrame] = {}


def _worker_init(aligned: str, cfg_path: str, cfg30: str) -> None:
    global _TBL, _MODELS, _ORD, _NUMCOLS, _GWL_CLIM, _DIRECT30
    _TBL = pd.read_parquet(aligned)
    _TBL["time"] = pd.to_datetime(_TBL["time"], errors="coerce")
    _TBL = F.drop_gwl_spikes(_TBL)
    _MODELS = R.load_models()
    _ORD = R.load_trainset_ordinals()
    cfg = json.loads(Path(cfg_path).read_text())
    _NUMCOLS = list(cfg["num_cols"])

    tr = _TBL[_TBL["time"] < TRAIN_CUT].copy()
    tr["doy"] = tr["time"].dt.dayofyear
    for st, g in tr.groupby("Station"):
        _GWL_CLIM[st] = g.groupby("doy")["gwl"].mean().to_dict()

    point30 = _joblib_load("xgb_multihorizon.joblib")
    _DIRECT30 = {
        "point": point30,
        "q50": _joblib_load("xgb_q50.joblib"),
        "fnames": list(point30.feature_names_in_),
        "stations": list(json.loads(Path(cfg30).read_text())["stations"]),
        "districts": list(json.loads(Path(cfg30).read_text())["districts"]),
    }


def _joblib_load(name: str):
    import joblib
    return joblib.load(ROOT / "models" / name)


def _init_direct30_raw():
    global _DIRECT30
    cfg30 = json.loads((ROOT / "models" / "feature_config.json").read_text())
    point = _joblib_load("xgb_multihorizon.joblib")
    _DIRECT30 = {
        "point": point,
        "q50": _joblib_load("xgb_q50.joblib"),
        "fnames": list(point.feature_names_in_),
        "stations": list(cfg30["stations"]),
        "districts": list(cfg30["districts"]),
    }


def _station_obs(station: str) -> pd.DataFrame:
    if station not in _STA_CACHE:
        _STA_CACHE[station] = (
            _TBL[_TBL["Station"] == station].sort_values("time").reset_index(drop=True))
    return _STA_CACHE[station]


def _sample_anchors(sta: pd.DataFrame) -> list[pd.Timestamp]:
    obs = sta.dropna(subset=["gwl"])["time"]
    if len(obs) < 4:
        return []
    lo, hi = BACKTEST_START, LATEST_FRAME
    obs = obs[(obs >= lo) & (obs <= hi)]
    if len(obs) < 4:
        return []
    out: list[pd.Timestamp] = []
    cur = lo
    while cur <= hi:
        day = obs[(obs >= cur) & (obs < cur + pd.Timedelta(days=1))]
        if len(day):
            out.append(day.iloc[0])
        cur = cur + pd.Timedelta(days=ANCHOR_SPACING_DAYS)
    return out


def _direct_30d(station: str, anchor_t: pd.Timestamp):
    """Frozen production direct-30d model at the anchor (t <= anchor frame)."""
    sta = _station_obs(station)
    slice_df = sta[sta["time"] <= anchor_t].copy()
    slice_df = slice_df.drop_duplicates("time").reset_index(drop=True)
    slice_df["date"] = slice_df["time"].dt.normalize()
    feats, _ = F.build_full(slice_df, keep_na=True, horizon_steps=120, horizon_label=30)
    feats = feats.sort_values("time")
    if feats.empty:
        return None
    last = feats.iloc[-1:].copy()
    last["st_id"] = _DIRECT30["stations"].index(station) if station in _DIRECT30["stations"] else -1
    district = str(sta["District"].iloc[0]).upper()
    last["dist_id"] = _DIRECT30["districts"].index(district) if district in _DIRECT30["districts"] else -1
    try:
        q50 = float(_DIRECT30["q50"].predict(last[_DIRECT30["fnames"]])[0])
    except Exception:
        return None
    return float(last["gwl"].iloc[0]) + q50


def _run_anchor(args) -> dict | None:
    station, anchor_t = args
    sta = _station_obs(station)
    if len(sta) <= 2:
        return None
    hist = sta[sta["time"] <= anchor_t]
    if len(hist) < R.OBS_WINDOW:
        return None
    district = str(sta["District"].iloc[0])
    district = district.upper() if district else ""
    st_names, dist_names = _ORD
    if R.resolve_station(station, st_names) is None:
        return None
    try:
        out = R.recursive_trajectory(
            station, scenario="climatology", steps=120, obs_tbl=hist,
            models=_MODELS, ordinals=(st_names, dist_names))
    except Exception:
        return None
    if "error" in out or not out.get("reliable", False):
        return None

    t_idx = {pd.Timestamp(x): i for i, x in enumerate(out["times"])}
    rec = {"q05": out["gwl_q05"], "q50": out["gwl_q50"], "q95": out["gwl_q95"]}
    clim = _GWL_CLIM.get(station, {})
    rows: list[dict] = []
    for h, h_steps in zip(HORIZONS_H, HORIZONS_STEPS):
        tgt = anchor_t + pd.Timedelta(hours=h)
        if tgt not in t_idx:
            continue
        idx = t_idx[tgt]
        obs_mask = sta["time"] == tgt
        if not obs_mask.any():
            continue
        obs = float(sta.loc[obs_mask, "gwl"].iloc[0])
        if not np.isfinite(obs):
            continue
        anchor_gwl = float(hist["gwl"].dropna().iloc[-1])
        clim_v = clim.get(int(tgt.dayofyear))
        rows.append({
            "station": station, "anchor_t": anchor_t, "horizon_h": h, "steps": h_steps,
            "obs": obs, "anchor": anchor_gwl,
            "rec_q50": rec["q50"][idx], "rec_q05": rec["q05"][idx], "rec_q95": rec["q95"][idx],
            "persist": anchor_gwl,
            "clim": clim_v if clim_v is not None else np.nan,
        })
    return rows


def main() -> None:
    t0 = time.time()
    global _TBL
    full = pd.read_parquet(ALIGNED)
    full["time"] = pd.to_datetime(full["time"], errors="coerce")
    full = F.drop_gwl_spikes(full)
    _TBL = full
    _init_direct30_raw()

    rng = random.Random(SEED)
    stations = sorted(full["Station"].astype(str).unique())
    stations = rng.sample(stations, min(STATIONS_SAMPLE, len(stations)))

    anchors_by_station: list[tuple] = []
    for st in stations:
        sta = full[full["Station"] == st].sort_values("time").reset_index(drop=True)
        for a in _sample_anchors(sta):
            anchors_by_station.append((st, a))
    rng.shuffle(anchors_by_station)
    print(f"anchors to score: {len(anchors_by_station):,} "
          f"({len(stations)} stations)", flush=True)

    tasks = anchors_by_station
    n_cpu = max(1, min((mp.cpu_count() or 4) // 2, 6))
    with mp.Pool(processes=n_cpu,
                 initializer=_worker_init,
                 initargs=(str(ALIGNED), str(ROOT / "models" / "6h_feature_config.json"),
                           str(ROOT / "models" / "feature_config.json"))) as pool:
        results = pool.map(_run_anchor, tasks, chunksize=8)

    recs = [r for batch in results if batch for r in batch]
    print(f"valid scores: {len(recs):,}", flush=True)
    df = pd.DataFrame(recs)

    rows = []
    for h in HORIZONS_H:
        g = df[df["horizon_h"] == h]
        if g.empty:
            continue
        rec = g["rec_q50"]; obs = g["obs"]
        rmse = math.sqrt(((obs - rec) ** 2).mean())
        mae = (obs - rec).abs().mean()
        bias = (rec - obs).mean()
        r2 = 1 - ((obs - rec) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
        cov = ((g["rec_q05"] <= obs) & (obs <= g["rec_q95"])).mean()
        width = (g["rec_q95"] - g["rec_q05"]).median()
        persist_rmse = math.sqrt(((obs - g["persist"]) ** 2).mean())
        clim_rmse = math.sqrt(((obs - g["clim"]) ** 2).mean())
        rows.append({
            "horizon_h": h, "steps": int(g["steps"].iloc[0]), "n": len(g),
            "rmse_rec": round(rmse, 4), "mae_rec": round(mae, 4),
            "bias_rec": round(bias, 4), "r2_rec": round(r2, 4),
            "coverage_raw": round(cov, 4), "halfwidth_median": round(width / 2, 4),
            "rmse_persist": round(persist_rmse, 4), "rmse_clim": round(clim_rmse, 4),
        })

    metric_df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(OUT / "backtest_6h_metrics.csv", index=False)

    g30 = df[df["horizon_h"] == 720]
    direct_vals = []
    for row in (df[df["horizon_h"] == 720].to_dict("records")):
        d = _direct_30d(row["station"], pd.Timestamp(row["anchor_t"]))
        if d is not None:
            direct_vals.append({"obs": row["obs"], "rec_q50": row["rec_q50"], "direct": d})
    dv = pd.DataFrame(direct_vals)
    direct_rmse = None
    if not dv.empty:
        direct_rmse = math.sqrt(((dv["obs"] - dv["direct"]) ** 2).mean())
        rec30_rmse = math.sqrt(((dv["obs"] - dv["rec_q50"]) ** 2).mean())
        metric_df.loc[metric_df.horizon_h == 720, "rmse_direct30"] = round(direct_rmse, 4)
        metric_df.loc[metric_df.horizon_h == 720, "rmse_rec_directset"] = round(rec30_rmse, 4)
    else:
        rec30_rmse = float(metric_df.loc[metric_df.horizon_h == 720, "rmse_rec"].iloc[0])
        direct_rmse = None

    # ---- per-horizon widening calibration (bisection on band half-width) ----
    calib = {"horizons_h": [], "widening_s": [], "coverage_raw": [],
             "coverage_calibrated": [], "n": []}
    for h in HORIZONS_H:
        g = df[df["horizon_h"] == h]
        if g.empty:
            continue
        med = g["rec_q50"].to_numpy()
        lo = g["rec_q05"].to_numpy()
        hi = g["rec_q95"].to_numpy()
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
                cov_m = cov_of(s_mid)
                if cov_m < COVERAGE_TARGET:
                    s_lo = s_mid
                else:
                    s_hi = s_mid
            s = s_hi
        cov_end = cov_of(s)
        calib["horizons_h"].append(h)
        calib["widening_s"].append(round(float(s), 3))
        calib["coverage_raw"].append(round(float(cov_of(1.0)), 3))
        calib["coverage_calibrated"].append(round(float(cov_end), 3))
        calib["n"].append(int(len(g)))

    calib_path = ROOT / "models" / "backtest_6h_calibration.json"
    calib_path.write_text(json.dumps(calib, indent=2))

    g30rmse = float(metric_df.loc[metric_df.horizon_h == 720, "rmse_rec"].iloc[0])
    direct_final = direct_rmse
    baselines = metric_df.loc[metric_df.horizon_h == 720, ["rmse_persist", "rmse_clim"]].iloc[0]
    summary = {
        "engine": "recursive 6h x120 (17_recursive)",
        "anchors_scored": int(len(df)),
        "stations_sampled": len(stations),
        "period": [str(BACKTEST_START.date()), str(LATEST_FRAME.date())],
        "anchor_spacing_days": ANCHOR_SPACING_DAYS,
        "no_overlap_windows": True,
        "no_future_observed_values": True,
        "best_30d_rmse_rec": g30rmse,
        "best_30d_rmse_direct_baseline": direct_final,
        "best_30d_rmse_persistence": float(baselines["rmse_persist"]),
        "best_30d_rmse_climatology": float(baselines["rmse_clim"]),
        "success_vs_direct": direct_final is not None and g30rmse < direct_final,
        "calibration_file": str(calib_path),
        "elapsed_sec": round(time.time() - t0),
    }
    (OUT / "backtest_6h_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== recursion vs baselines (RMSE m) ===")
    print(metric_df.to_string(index=False))
    print("\n=== per-horizon calibration ===")
    print(pd.DataFrame(calib).to_string(index=False))
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()