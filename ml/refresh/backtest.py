"""refresh/backtest.py — honest non-overlap candidate vs incumbent backtest.

Both models (candidate from data/refresh/candidate, incumbent from ml/models)
are scored on the SAME recent anchor set over the most recent
``candidate_backtest_window_days``-day window, so promotion is apples-to-apples:
candidate must beat the incumbent's own 30-day RMSE on identical anchors, beat
persistence, and meet the calibrated coverage target. Anchors are spaced on
observed GWL times exactly like 32_backtest_traj. No future observed GWL or
driver value enters any feature vector.
"""

from __future__ import annotations

import importlib
import json
import math
import random

import numpy as np
import pandas as pd

F = importlib.import_module("06_features")

HORIZONS_STEPS = [1, 2, 3, 4, 8, 12, 28, 60, 120]
HORIZONS_H = [6, 12, 18, 24, 48, 72, 168, 360, 720]
COVERAGE_TARGET = 0.90
MAX_H = 120
SEED = 42


def sample_anchors(tbl: pd.DataFrame, window_start: pd.Timestamp,
                   window_end: pd.Timestamp, spacing_days: int = 14,
                   n_stations: int = 60, seed: int = SEED,
                   stations: list[str] | None = None) -> list[tuple[str, pd.Timestamp]]:
    """Deterministic list of (station, anchor_t) inside [window_start, window_end]."""
    rng = random.Random(seed)
    all_st = sorted(tbl["Station"].astype(str).unique())
    if stations is None:
        if n_stations and n_stations < len(all_st):
            stations = rng.sample(all_st, n_stations)
        else:
            stations = all_st
    tasks: list[tuple[str, pd.Timestamp]] = []
    for st in stations:
        sta = tbl[tbl["Station"].astype(str) == st].sort_values("time")
        s = sta.dropna(subset=["gwl"])["time"]
        s = s[(s >= window_start) & (s <= window_end)]
        if len(s) < 4:
            continue
        cur = window_start
        while cur <= window_end:
            day = s[(s >= cur) & (s < cur + pd.Timedelta(days=1))]
            if len(day):
                tasks.append((st, day.iloc[0]))
            cur += pd.Timedelta(days=spacing_days)
    rng.shuffle(tasks)
    return tasks


def load_models(model_dir, feature_names: list[str]) -> dict:
    import xgboost as xgb
    return {n: xgb.Booster(model_file=str(model_dir / f"traj_xgb_{n}.json"))
            for n in ("q05", "q50", "q95")}


def _anchors_feature(sta, anchor_t, mode: str, fnames: list[str],
                     future_data=None, forecast_days: int = 16) -> tuple[np.ndarray, float] | None:
    hist = sta[sta["time"] <= anchor_t]
    if len(hist) < 30:
        return None
    hist_d = hist.copy()
    hist_d["date"] = hist_d["time"].dt.normalize()
    if mode == "future":
        from refresh.future_drivers import build_step_features
        om, clim, rcclim = future_data or (None, None, None)
        anc = float(hist_d["gwl"].dropna().iloc[-1])
        X, _ = build_step_features(
            station=str(sta["Station"].iloc[0]),
            district=str(sta["District"].iloc[0]), anchor_t=pd.Timestamp(anchor_t),
            hist=hist_d, fnames=fnames, om_frame=om, climatology=clim,
            rain_clim=rcclim, forecast_days=forecast_days, steps=MAX_H,
            mode="future", anchor_gwl=anc)
        return X, anc
    feats, _ = F.build_full(hist_d, keep_na=True, horizon_steps=MAX_H, horizon_label=30)
    feats = feats.sort_values("time")
    if feats.empty:
        return None
    last = feats.iloc[-1:].copy()
    anc = float(last["gwl"].iloc[0])
    if not np.isfinite(anc):
        return None
    base_cols = [c for c in fnames if c not in ("h", "h_sin", "h_cos")]
    base = np.asarray(last[base_cols].values[0], dtype=np.float32)
    hs = np.arange(1, MAX_H + 1, dtype=np.float32)
    X = np.hstack([np.repeat(base[None, :], MAX_H, axis=0),
                   hs[:, None],
                   np.sin(2 * np.pi * hs / MAX_H)[:, None],
                   np.cos(2 * np.pi * hs / MAX_H)[:, None]])
    return X, anc


def _direct30_level(sta, anchor_t, ord30: dict) -> float | None:
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
    last["st_id"] = (ord30["stations"].index(station)
                     if station in ord30["stations"] else -1)
    district = str(sta["District"].iloc[0]).upper()
    last["dist_id"] = (ord30["districts"].index(district)
                       if district in ord30["districts"] else -1)
    try:
        q50 = float(ord30["q50"].predict(last[ord30["fnames"]])[0])
    except Exception:  # noqa: BLE001
        return None
    return float(last["gwl"].iloc[0]) + q50


def _load_ord30(root) -> dict:
    import joblib
    cfg30 = json.loads((root / "models" / "feature_config.json").read_text())
    return {
        "q50": joblib.load(root / "models" / "xgb_q50.joblib"),
        "fnames": list(joblib.load(root / "models" / "xgb_multihorizon.joblib").feature_names_in_),
        "stations": list(cfg30["stations"]),
        "districts": list(cfg30["districts"]),
    }


def run_backtest_experiment(*, tbl, tasks, fnames: list[str], providers: dict,
                            mode: str = "flat", future_data=None,
                            forecast_days: int = 16, root=None,
                            clim_by_station: dict | None = None) -> dict:
    """Score every provider (label -> model dir) on the same task set.

    Returns {"rows": DataFrame, "per_h": {label: DataFrame metrics}, "calib": dict}.
    ``root`` = ml/ directory (needed for the direct-30d benchmark).
    """
    if root is None:
        root = F_PATH_ROOT
    ord30 = _load_ord30(root) if (root / "models" / "xgb_q50.joblib").exists() else None
    loaded = {label: load_models(mdir, fnames) for label, mdir in providers.items()}

    import xgboost as xgb
    recs = []
    for (station, anchor_t) in tasks:
        sta = tbl[tbl["Station"].astype(str) == station].sort_values("time").reset_index(drop=True)
        feat = _anchors_feature(sta, anchor_t, mode, fnames, future_data, forecast_days)
        if feat is None:
            continue
        X, anc = feat
        dm = xgb.DMatrix(X, feature_names=fnames)
        observ = {label: (m["q05"].predict(dm) + anc, m["q50"].predict(dm) + anc,
                          m["q95"].predict(dm) + anc) for label, m in loaded.items()}
        obs_times = sta["time"].to_numpy()
        obs_gwl = sta["gwl"].to_numpy()
        clim = (clim_by_station or {}).get(station, {})
        d6h_clim = clim.get(int(pd.Timestamp(anchor_t).dayofyear), np.nan)
        direct30 = _direct30_level(sta, anchor_t, ord30) if ord30 is not None else None

        for (label, _) in providers.items():
            q05, q50, q95 = observ[label]
            q05 = np.minimum(q05, q50)
            q95 = np.maximum(q95, q50)
            for h, hh in zip(HORIZONS_STEPS, HORIZONS_H):
                tgt = pd.Timestamp(anchor_t) + pd.Timedelta(hours=hh)
                idx = int(np.searchsorted(obs_times, tgt))
                if idx >= len(obs_times) or pd.Timestamp(obs_times[idx]) != tgt:
                    continue
                obs = float(obs_gwl[idx])
                if not np.isfinite(obs):
                    continue
                recs.append({
                    "label": label, "station": station, "anchor_t": pd.Timestamp(anchor_t),
                    "horizon_h": hh, "steps": h, "obs": obs, "anchor": float(anc),
                    "traj_q50": float(q50[h - 1]), "traj_q05": float(q05[h - 1]),
                    "traj_q95": float(q95[h - 1]),
                    "persist": float(anc),
                    "clim": float(anc + d6h_clim * h) if np.isfinite(d6h_clim) else np.nan,
                    "direct30": direct30 if hh == 720 else None,
                })
    df = pd.DataFrame(recs)

    per_h = {}
    for label in providers:
        rows = []
        for h, hh in zip(HORIZONS_STEPS, HORIZONS_H):
            g = df[(df["label"] == label) & (df["horizon_h"] == hh)]
            if g.empty:
                continue
            rows.append({**{"horizon_h": hh, "steps": h}, **_metrics(g)})
        per_h[label] = pd.DataFrame(rows)

    calib = _calibrate(df)   # candidate calibration (same anchors, candidate model)
    return {"rows": df, "per_h": per_h, "calib": calib}


def _metrics(g: pd.DataFrame) -> dict:
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
        "sign_acc": sign_acc,
        "coverage_raw": float(np.mean((q05 <= obs) & (obs <= q95))),
        "halfwidth_median": float(np.median((q95 - q05) / 2)),
        "rmse_persist": float(math.sqrt(float(np.mean((obs - g["persist"].to_numpy()) ** 2)))),
    }


def _calibrate(df: pd.DataFrame) -> dict:
    """Per-horizon widening, FITTED and REPORTED on disjoint anchors.

    Fitting the widening on the same anchors whose coverage it reports would make
    `coverage_calibrated` a description of the fit rather than a validation. The
    anchors are split deterministically (alternating in station/time order): the
    fit half chooses `s`, and the held-out half produces the headline coverage.
    """
    out = {"horizons_h": [], "widening_s": [], "coverage_raw": [],
           "coverage_calibrated": [], "coverage_calibrated_fit": [],
           "coverage_raw_heldout": [], "n": [], "n_fit": [], "n_heldout": []}
    if df.empty:
        return out

    # Deterministic, RAM-cheap split: hash the (station, anchor) pair to assign it
    # to the fit half or the held-out half. Not a stored set of pairs, which would
    # grow with the anchor count on an already memory-tight box.
    df = df.assign(_half=(pd.util.hash_pandas_object(
        df["station"].astype(str) + "|" + df["anchor_t"].astype(str),
        index=False).to_numpy() & 1).astype(np.int8))

    target = COVERAGE_TARGET + 0.02

    def _fit_s(med, half, obs, avail, lo_s=0.2, hi_s=12.0):
        if not avail.any():
            return float("nan")
        cov = lambda s: float(np.mean((med[avail] - s * half[avail] <= obs[avail])
                                      & (obs[avail] <= med[avail] + s * half[avail])))
        if cov(lo_s) >= target:
            return lo_s
        a, b = lo_s, hi_s
        for _ in range(40):
            m = 0.5 * (a + b)
            if cov(m) < target:
                a = m
            else:
                b = m
        return b

    def _cov(med, half, obs, s):
        if s != s:                     # NaN fit -> no usable interval
            return float("nan")
        return float(np.mean((med - s * half <= obs) & (obs <= med + s * half)))

    for hh in HORIZONS_H:
        g = df[df["horizon_h"] == hh]
        if g.empty:
            continue
        gf, gh = g[g["_half"] == 0], g[g["_half"] == 1]
        if gf.empty or gh.empty:       # too few anchors to split at this horizon
            gf = gh = g
        f_med, f_lo, f_hi, f_obs = (gf["traj_q50"].to_numpy(), gf["traj_q05"].to_numpy(),
                                    gf["traj_q95"].to_numpy(), gf["obs"].to_numpy())
        f_half = (f_hi - f_lo) / 2.0
        s = _fit_s(f_med, f_half, f_obs, f_half > 1e-9)

        h_med, h_lo, h_hi, h_obs = (gh["traj_q50"].to_numpy(), gh["traj_q05"].to_numpy(),
                                    gh["traj_q95"].to_numpy(), gh["obs"].to_numpy())
        h_half = (h_hi - h_lo) / 2.0

        out["horizons_h"].append(float(hh))
        out["widening_s"].append(round(float(s), 3) if s == s else None)
        out["coverage_raw"].append(round(float(np.mean((h_lo <= h_obs)
                                                        & (h_obs <= h_hi))), 3))
        out["coverage_raw_heldout"].append(round(float(np.mean((h_lo <= h_obs)
                                                               & (h_obs <= h_hi))), 3))
        out["coverage_calibrated_fit"].append(round(_cov(f_med, f_half, f_obs, s), 3))
        # HEADLINE (held out): widening fitted on the other half
        out["coverage_calibrated"].append(round(_cov(h_med, h_half, h_obs, s), 3))
        out["n"].append(int(len(gh)))
        out["n_fit"].append(int(len(gf)))
        out["n_heldout"].append(int(len(gh)))
    return out


from pathlib import Path  # noqa: E402

F_PATH_ROOT = Path(__file__).resolve().parent.parent