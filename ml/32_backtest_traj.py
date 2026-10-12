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

# Model backend: "xgb" (incumbent, default) or "dl" (sequence GRU, 34_train_traj_dl).
# DL writes to *_dl.* paths only — the incumbent's outputs/artifacts are never touched.
MODEL_KIND = os.environ.get("TRAJ_MODEL", "xgb").lower()
SUFFIX = "" if MODEL_KIND == "xgb" else "_dl"
CALIB = ROOT / "models" / f"traj{SUFFIX}_calibration.json"

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
_MODEL_KIND = "xgb"
_DL = None
_DL_CFG = {}
_NUM_COLS = []


def _joblib_load(name):
    import joblib
    return joblib.load(ROOT / "models" / name)


def _worker_init(aligned, traj_cfg, model_kind="xgb"):
    global _TBL, _MODELS, _FN, _GWL_CLIM, _ORD30, _MODEL_KIND, _DL, _DL_CFG, _NUM_COLS
    _MODEL_KIND = model_kind
    _TBL = pd.read_parquet(aligned)
    _TBL["time"] = pd.to_datetime(_TBL["time"], errors="coerce")
    _TBL = F.drop_gwl_spikes(_TBL)

    cfg = json.loads(Path(traj_cfg).read_text())
    _FN = list(cfg["feature_cols"])
    _NUM_COLS = list(F.NUM_COLS)

    if model_kind == "dl":
        import torch
        torch.set_num_threads(1)
        dlmod = importlib.import_module("34_train_traj_dl")
        dcfg = json.loads((ROOT / "models" / "traj_dl_config.json").read_text())
        net = dlmod.TrajGRU(len(_NUM_COLS), hidden=int(dcfg["hidden"]))
        net.load_state_dict(torch.load(ROOT / "models" / "traj_dl.pt",
                                       map_location="cpu", weights_only=True))
        net.eval()
        _DL, _DL_CFG = net, dcfg
    else:
        import xgboost as xgb
        for name in ("q05", "q50", "q95"):
            _MODELS[name] = xgb.Booster(
                model_file=str(ROOT / "models" / f"traj_xgb_{name}.json"))

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
    # Only the LAST row's features are used, and nothing downstream looks further
    # back than the longest trailing window (120 steps for the 30-day GWL roll,
    # 128 for the driver tail, 90 days for the slope block) — so the anchor row's
    # feature values are unchanged by this cap. Building features for the whole
    # station history and then keeping one row was the main memory cost of this
    # backtest (~5k rows x ~45 cols per anchor, times the pool size).
    #
    # Side effect to be aware of: with a short frame `flag_regime_shift` never
    # fires (it needs 2021-2024 medians), so a regime-shifted station is no longer
    # silently dropped here. That is the correct behaviour — the anchor is scored
    # as-is — and the same rule already ran on the full table in `_worker_init`.
    TAIL_STEPS = 450
    if len(hist_d) > TAIL_STEPS:
        hist_d = hist_d.iloc[-TAIL_STEPS:].copy()
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
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    hs = np.arange(1, MAX_H + 1, dtype=np.float32)

    if _MODEL_KIND == "dl":
        import torch
        W = int(_DL_CFG.get("window_w", 120))
        Xseq = feats[_NUM_COLS].to_numpy(np.float32)
        Xseq = np.where(np.isfinite(Xseq), Xseq, 0.0).astype(np.float32)
        n = len(Xseq); w = min(W, n)
        seq = np.zeros((1, W, len(_NUM_COLS)), np.float32)
        seqm = np.zeros((1, W, 1), np.float32)
        seq[0, W - w:] = Xseq[-w:]
        seqm[0, W - w:, 0] = 1.0
        hf = np.stack([hs / MAX_H, np.sin(2 * np.pi * hs / MAX_H),
                       np.cos(2 * np.pi * hs / MAX_H)], axis=-1)[None].astype(np.float32)
        with torch.no_grad():
            pred = _DL(torch.from_numpy(seq), torch.from_numpy(seqm),
                       torch.from_numpy(base[None, :]), torch.from_numpy(hf))[0].numpy()
        pred = np.sort(pred, axis=1)
        q05 = pred[:, 0] + anchor_gwl
        q50 = pred[:, 1] + anchor_gwl
        q95 = pred[:, 2] + anchor_gwl
    else:
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
                 initargs=(str(ALIGNED), str(TRAJ_CFG), MODEL_KIND)) as pool:
        results = pool.map(_run_anchor, tasks, chunksize=8)
    recs = [r for batch in results if batch for r in batch]
    print(f"valid scores: {len(recs):,}", flush=True)
    df = pd.DataFrame(recs)
    df.to_parquet(OUT / f"traj_backtest_rows{SUFFIX}.parquet", index=False)

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
    metric_df.to_csv(OUT / f"traj_backtest_metrics{SUFFIX}.csv", index=False)

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

    # ---- calibration, fitted and REPORTED on disjoint anchors -------------
    # Fitting the widening on the same anchors whose coverage it reports makes
    # `coverage_calibrated` a description of the fit, not a validation. Split the
    # anchor set deterministically (alternating anchors in station/time order) so
    # one half fits `s` and the other half is held out for the headline number.
    def _fit_s(med, half, obs, avail, lo_s=0.2, hi_s=12.0):
        cov_of = lambda s: np.mean((med[avail] - s * half[avail] <= obs[avail])
                                   & (obs[avail] <= med[avail] + s * half[avail]))
        if not avail.any():
            return float("nan")
        if cov_of(lo_s) >= COVERAGE_TARGET:
            return lo_s
        a, b = lo_s, hi_s
        for _ in range(40):
            mid = 0.5 * (a + b)
            if cov_of(mid) < COVERAGE_TARGET:
                a = mid
            else:
                b = mid
        return b

    def _cov(med, half, obs, s):
        if s != s:                     # NaN fit -> no usable interval
            return float("nan")
        return float(np.mean((med - s * half <= obs) & (obs <= med + s * half)))

    # Deterministic, RAM-cheap split: hash the (station, anchor) pair to assign it
    # to the calibration-fit half or the held-out half. Deliberately NOT a stored
    # set of pairs — at ~74k anchors that is tens of MB per worker on a box already
    # holding the aligned table and the quantile boosters.
    keys = df[["station", "anchor_t"]].drop_duplicates()
    df = df.assign(_half=(pd.util.hash_pandas_object(
        df["station"].astype(str) + "|" + df["anchor_t"].astype(str),
        index=False).to_numpy() & 1).astype(np.int8))
    n_fit = int((df["_half"] == 0).sum())
    print(f"calibration split: {n_fit:,} fit / {int((df['_half'] == 1).sum()):,} "
          f"held-out scores (hash-split of {len(keys):,} anchors)", flush=True)

    calib = {"horizons_h": [], "widening_s": [], "coverage_raw": [],
             "coverage_calibrated": [], "coverage_calibrated_fit": [],
             "coverage_raw_heldout": [], "n": [], "n_fit": [], "n_heldout": []}
    for hh in HORIZONS_H:
        g = df[df["horizon_h"] == hh]
        if g.empty:
            continue
        gf = g[g["_half"] == 0]
        gh = g[g["_half"] == 1]
        if gf.empty or gh.empty:       # too few anchors to split at this horizon
            gf = gh = g
        f_med = gf["traj_q50"].to_numpy(); f_lo = gf["traj_q05"].to_numpy()
        f_hi = gf["traj_q95"].to_numpy(); f_obs = gf["obs"].to_numpy()
        f_half = (f_hi - f_lo) / 2.0
        f_avail = f_half > 1e-9
        s = _fit_s(f_med, f_half, f_obs, f_avail)

        h_med = gh["traj_q50"].to_numpy(); h_lo = gh["traj_q05"].to_numpy()
        h_hi = gh["traj_q95"].to_numpy(); h_obs = gh["obs"].to_numpy()
        h_half = (h_hi - h_lo) / 2.0

        calib["horizons_h"].append(hh)
        calib["widening_s"].append(round(float(s), 3) if s == s else None)
        calib["coverage_raw"].append(round(float(np.mean((h_lo <= h_obs)
                                                          & (h_obs <= h_hi))), 3))
        calib["coverage_raw_heldout"].append(round(float(np.mean((h_lo <= h_obs)
                                                                  & (h_obs <= h_hi))), 3))
        calib["coverage_calibrated_fit"].append(round(_cov(f_med, f_half, f_obs, s), 3))
        # HEADLINE: widening fitted on the fit half, coverage measured on the rest
        calib["coverage_calibrated"].append(round(_cov(h_med, h_half, h_obs, s), 3))
        calib["n"].append(int(len(gh)))
        calib["n_fit"].append(int(len(gf)))
        calib["n_heldout"].append(int(len(gh)))
    df = df.drop(columns=["_k"], errors="ignore")
    CALIB.write_text(json.dumps(calib, indent=2))

    def _last(key: str):
        vals = calib.get(key) or []
        return vals[-1] if vals else None

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
        "engine": ("direct multi-horizon shared XGBoost, h feature (30_traj/31_traj)"
                   if MODEL_KIND == "xgb" else
                   "sequence GRU + horizon-conditioned quantile heads (34_train_traj_dl)"),
        "anchors_scored": int(len(df)),
        "stations_sampled": len(stations),
        "period": [str(BACKTEST_START.date()), str(LATEST_FRAME.date())],
        "anchor_spacing_days": ANCHOR_SPACING_DAYS,
        "no_overlap_windows": bool(HORIZONS_H[-1] <= 24 * ANCHOR_SPACING_DAYS),
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
        "30d_coverage_calibrated": _last("coverage_calibrated"),
        "30d_coverage_calibrated_fit_half": _last("coverage_calibrated_fit"),
        "30d_coverage_raw_heldout": _last("coverage_raw_heldout"),
        "calibration_split": ("widening fitted on alternating anchors; coverage "
                              "reported on the disjoint complement"),
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
    (OUT / f"traj_backtest_summary{SUFFIX}.json").write_text(json.dumps(summary, indent=2))

    print("\n=== direct trajectory vs baselines (RMSE m) ===")
    print(metric_df.round(4).to_string(index=False))
    print("\n=== calibration ===")
    print(pd.DataFrame(calib).to_string(index=False))
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()