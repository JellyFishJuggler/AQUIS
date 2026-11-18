"""Trajectory v2 chain: datasets, XGBoost train/backtest, reliability, paired CI.

Merges the old 30/31/32/33/35 numbered scripts. The GRU variants live in
``pipelines.traj_dl``; the batch CLI below dispatches both modules.

Run a stage with:  python -m pipelines.traj <stage>. The ``pipeline_traj``
root shim forwards to this module for backward compatibility.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
F = importlib.import_module("pipelines.model")


# ====================== multi-horizon datasets (from 30_traj_datasets.py)
ROOT = Path(__file__).resolve().parents[1]
FE6 = ROOT / "data" / "features_6h"
FEATRAJ_OUT = ROOT / "data" / "features_traj"
FEATRAJ_OUT.mkdir(parents=True, exist_ok=True)

MAX_H = 120                 # 30 days on the 6h grid
TRAIN_STRIDE = 8            # row subsampling for the train long-form
HORIZONS_TRAIN = list(range(1, 21)) + [24, 28, 32, 36, 40, 48, 56, 64, 80, 96, 120]
HORIZONS_VAL = list(range(1, MAX_H + 1))

NUM_COLS = list(F.NUM_COLS)
META = ["Station", "District", "time", "st_id", "dist_id"]
HCOL = ["h", "h_sin", "h_cos"]


def _horizon_feats(h: int) -> dict:
    return {"h": h, "h_sin": float(np.sin(2 * np.pi * h / MAX_H)),
            "h_cos": float(np.cos(2 * np.pi * h / MAX_H))}


def _axis_and_targets(parts: list[pd.DataFrame]):
    """Per-station observed time axis (unique, sorted) + cleaned gwl values."""
    t = pd.concat([p[["time", "gwl"]] for p in parts], ignore_index=True)
    t = t.sort_values("time").drop_duplicates("time").dropna(subset=["gwl"])
    return t["time"].to_numpy(), t["gwl"].to_numpy(dtype=np.float64)


def _expand(rows: pd.DataFrame, times_st: np.ndarray, gwl_st: np.ndarray,
            horizons: list[int]) -> pd.DataFrame:
    """For each candidate row, emit (h, target_delta, horizon feats, row feats).

    target_delta = gwl(t+h*6h) - gwl(t) when gwl(t+h*6h) is observed at the
    exact time; rows/horizons without an exact observation are skipped.
    ``target_time`` records the ACTUAL observed target timestamp (the matched
    grid time), so callers can purge every label that reaches a later split.
    """
    t0s = rows["time"].to_numpy(dtype="datetime64[ns]")
    g0 = rows["gwl"].to_numpy(dtype=np.float64)
    out: list[pd.DataFrame] = []
    for h in horizons:
        tt = t0s + np.timedelta64(h * 6, "h")
        q = np.searchsorted(times_st, tt)
        qq = np.clip(q, 0, len(times_st) - 1)
        ok = q < len(times_st)
        ok &= times_st[qq] == tt
        delta = np.full(len(rows), np.nan)
        delta[ok] = gwl_st[qq[ok]] - g0[ok]
        keep = ok & np.isfinite(delta)
        if not keep.any():
            continue
        idx = np.where(keep)[0]
        hf = _horizon_feats(h)
        chunk = rows.iloc[idx].assign(
            h=h, h_sin=hf["h_sin"], h_cos=hf["h_cos"],
            target_delta=np.asarray(delta[keep], dtype=np.float32),
            target_time=tt[idx])
        out.append(chunk)
    if not out:
        return pd.DataFrame(columns=list(rows.columns) + HCOL + ["target_delta",
                                                                 "target_time"])
    return pd.concat(out, ignore_index=True)


def datasets_main() -> None:
    t0 = time.time()
    tr = pd.read_parquet(FE6 / "train.parquet", columns=META + NUM_COLS)
    va = pd.read_parquet(FE6 / "val.parquet", columns=META + NUM_COLS)
    te = pd.read_parquet(FE6 / "test.parquet", columns=META + NUM_COLS)

    for df in (tr, va, te):
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # all feature rows define the station's observed axis (cleaned gwl)
    axis = pd.concat([tr, va, te], ignore_index=True)
    stations = sorted(axis["Station"].astype(str).unique())
    print(f"stations: {len(stations):,}", flush=True)

    train_rows, val_rows = [], []
    for i, st in enumerate(stations, 1):
        sub = axis[axis["Station"].astype(str) == st]
        times_st, gwl_st = _axis_and_targets([sub])

        rows_tr = tr[tr["Station"].astype(str) == st].iloc[::TRAIN_STRIDE]
        rows_va = va[va["Station"].astype(str) == st]
        if len(rows_tr):
            train_rows.append(_expand(rows_tr, times_st, gwl_st, HORIZONS_TRAIN))
        if len(rows_va):
            val_rows.append(_expand(rows_va, times_st, gwl_st, HORIZONS_VAL))
        if i % 100 == 0:
            print(f"  {i}/{len(stations)} stations "
                  f"(train so far {sum(len(x) for x in train_rows):,})", flush=True)

    trb = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    vab = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()

    # --- target-time purge (A): no label may be observed inside a later split
    #     — per anchor AND horizon, using the exact observed target timestamp. ---
    if not trb.empty:
        n_pre = len(trb)
        trb = trb[trb["target_time"] < F.VAL_START]
        n_purge = n_pre - len(trb)
        if n_purge:
            print(f"purge train: {n_purge:,} rows with target >= {F.VAL_START.date()}", flush=True)
    if not vab.empty:
        n_pre = len(vab)
        vab = vab[vab["target_time"] < F.TEST_START]
        n_purge = n_pre - len(vab)
        if n_purge:
            print(f"purge val:   {n_purge:,} rows with target >= {F.TEST_START.date()}", flush=True)

    for df in (trb, vab):
        if not df.empty:
            df["h"] = df["h"].astype(np.int16)
            df["st_id"] = df["st_id"].astype(np.int32)
            df["dist_id"] = df["dist_id"].astype(np.int32)
            for c in NUM_COLS:
                df[c] = df[c].astype(np.float32)
            df["target_delta"] = df["target_delta"].astype(np.float32)

    trb.to_parquet(FEATRAJ_OUT / "train.parquet", index=False)
    vab.to_parquet(FEATRAJ_OUT / "val.parquet", index=False)

    prep = {
        "paradigm": "direct multi-horizon shared model (no recursion)",
        "target": "GWL(t+h*6h) - GWL(t), exact observed time match (no stale shift)",
        "train_stride": TRAIN_STRIDE,
        "h_train_list": HORIZONS_TRAIN,
        "h_val_all": MAX_H,
        "num_cols": NUM_COLS,
        "horizon_cols": HCOL,
        "train_rows": int(len(trb)),
        "val_rows": int(len(vab)),
        "stations": int(len(stations)),
        "split": "train < 2025-10-01 (row-subsampled x horizon-subsampled), "
                 "val = all, purged by target_time",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (FEATRAJ_OUT / "prep_traj.json").write_text(json.dumps(prep, indent=2))
    print(f"\ntrain rows: {len(trb):,} | val rows: {len(vab):,}  ({(time.time() - t0):.0f}s)")
    if not trb.empty:
        per_h = trb.groupby("h").size()
        print(per_h.head(12).to_string())


# ====================== shared direct XGBoost (from 31_train_traj.py)
ROOT = Path(__file__).resolve().parents[1]
FTRAJ = ROOT / "data" / "features_traj"
OUTM = ROOT / "models"
OUTM.mkdir(parents=True, exist_ok=True)

ALPHAS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
MAX_H = 120
TX_SEED = 42
EVAL_FRAC = 0.10      # time-ordered tail used for early stopping
PARAMS = dict(
    max_depth=7, learning_rate=0.05, min_child_weight=60,
    subsample=0.9, colsample_bytree=0.8,
    n_estimators=900, tree_method="hist", max_bin=128, n_jobs=6, random_state=TX_SEED,
)
EARLY = 25


def _load(sub: str) -> pd.DataFrame:
    return pd.read_parquet(FTRAJ / f"{sub}.parquet")


def _fnames() -> list[str]:
    return list(F.NUM_COLS) + ["h", "h_sin", "h_cos"]


def _per_h(va: pd.DataFrame, q5: np.ndarray, q50: np.ndarray,
           q9: np.ndarray, steps: list[int]) -> list[dict]:
    y = va["target_delta"].to_numpy()
    hh = va["h"].to_numpy()
    rows = []
    for h in steps:
        m = hh == h
        if not m.any():
            continue
        yh = y[m]; p05, p50, p95 = q5[m], q50[m], q9[m]
        e = yh - p50
        rows.append({
            "h": int(h), "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(p50 - yh)),
            "coverage": float(np.mean((p05 <= yh) & (yh <= p95))),
            "halfwidth_median": float(np.median((p95 - p05) / 2)),
        })
    return rows


def train_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny subset, quick")
    args = ap.parse_args()

    t0 = time.time()
    tr = _load("train")
    va = _load("val")

    fn = _fnames()
    order = [c for c in fn if c in tr.columns]

    if args.smoke:
        rng = np.random.default_rng(TX_SEED)
        keep = rng.choice(len(tr), size=min(400_000, len(tr)), replace=False)
        tr = tr.iloc[np.sort(keep)].copy()
        va = va[va["h"] <= 30].copy()

    tr = tr.sort_values("time").reset_index(drop=True)
    n_eval = max(1, int(len(tr) * EVAL_FRAC))
    dtrain = xgb.DMatrix(tr.iloc[:-n_eval][order], label=tr.iloc[:-n_eval]["target_delta"])
    deval = xgb.DMatrix(tr.iloc[-n_eval:][order], label=tr.iloc[-n_eval:]["target_delta"])
    dval = xgb.DMatrix(va[order], label=va["target_delta"])
    print(f"fit {dtrain.num_row():,} | eval {deval.num_row():,} | val {dval.num_row():,} "
          f"| features {len(order)}", flush=True)

    preds: dict[str, np.ndarray] = {}
    for name, alpha in ALPHAS.items():
        p = dict(PARAMS, objective="reg:quantileerror", quantile_alpha=alpha)
        bst = xgb.train(p, dtrain, num_boost_round=PARAMS["n_estimators"],
                        evals=[(deval, "eval")], early_stopping_rounds=EARLY,
                        verbose_eval=False)
        bst.save_model(OUTM / f"traj_xgb_{name}.json")
        preds[name] = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        tail = np.sqrt(np.mean((va["target_delta"].to_numpy() - preds[name]) ** 2))
        print(f"  {name}: best_iter={bst.best_iteration} val_overall_rmse={tail:.4f}",
              flush=True)

    q5 = preds["q05"]; q50 = preds["q50"]; q9 = preds["q95"]
    q5 = np.minimum(q5, q50); q9 = np.maximum(q9, q50)   # enforce order at all h
    steps = [1, 2, 3, 4, 8, 12, 28, 60, 120] if not args.smoke else [1, 2, 8, 30]
    rows = _per_h(va, q5, q50, q9, steps)
    val_metrics = pd.DataFrame(rows)

    cfg = {
        "paradigm": "direct multi-horizon shared model (no recursion)",
        "target": "GWL(t+h*6h) - GWL(t); level = anchor + delta (anchor observed only)",
        "feature_cols": order,
        "quantiles": ALPHAS,
        "max_h": MAX_H,
        "early_stopping_tail_frac": EVAL_FRAC,
        "fit_rows": int(dtrain.num_row()),
        "eval_rows": int(deval.num_row()),
        "val_rows_scored": int(dval.num_row()),
        "smoke": args.smoke,
        "val_per_h_metrics": val_metrics.round(4).to_dict("records"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUTM / "traj_config.json").write_text(json.dumps(cfg, indent=2))

    print("\nper-horizon validation (delta RMSE m / coverage / half-width):")
    print(val_metrics.round(4).to_string(index=False))
    print(f"\nelapsed {(time.time() - t0):.0f}s -> models/traj_xgb_*.json + traj_config.json")


# ====================== honest 2026 backtest (from 32_backtest_traj.py)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


ROOT = Path(__file__).resolve().parents[1]
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
        dlmod = importlib.import_module('pipelines.traj_dl')
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
    # station history and then keeping one row was the backtest_main memory cost of this
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


def backtest_main():
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


# ====================== reliability tables (from 33_traj_reliability.py)
ROOT = Path(__file__).resolve().parents[1]
MODEL_KIND = os.environ.get("TRAJ_MODEL", "xgb").lower()
SUFFIX = "" if MODEL_KIND == "xgb" else "_dl"
METRICS = ROOT / "outputs" / f"traj_backtest_metrics{SUFFIX}.csv"
CALIB = ROOT / "models" / f"traj{SUFFIX}_calibration.json"
RE_OUT = ROOT / "models" / f"traj{SUFFIX}_reliability.json"

COVERAGE_MIN = 0.80
DIR_GOOD = 0.62
DIR_MARGINAL = 0.55
PERF_TOL = 1.0


def reliability_main() -> None:
    cv = pd.read_csv(METRICS)
    cal = json.loads(CALIB.read_text())
    buckets = [int(x) for x in cal["horizons_h"]]
    cal_map = {b: c for b, c in zip(cal["horizons_h"], cal["coverage_calibrated"])}
    s_map = {b: s for b, s in zip(cal["horizons_h"], cal["widening_s"])}

    rules = {}
    for _, r in cv.iterrows():
        b = int(r["horizon_h"])
        perf_ratio = float(r["rmse_persist"]) / float(r["rmse"]) if r["rmse"] > 0 else np.nan
        cover = cal_map.get(b, np.nan)
        half = float(r["halfwidth_median"])
        width_ok = None
        try:
            err = float(r["rmse"])
            width_ok = bool(half <= 2.0 * err)
        except Exception:
            pass
        sa = float(r["sign_acc"]) if pd.notna(r["sign_acc"]) else np.nan
        rules[str(b)] = {
            "horizon_h": b,
            "n": int(r["n"]),
            "calibrated_coverage": round(cover, 4) if pd.notna(cover) else None,
            "widening_s": s_map.get(b),
            "rmse": round(float(r["rmse"]), 4),
            "rmse_persist": round(float(r["rmse_persist"]), 4),
            "perf_vs_persist_ratio": round(perf_ratio, 4) if np.isfinite(perf_ratio) else None,
            "sign_acc": round(sa, 4) if np.isfinite(sa) else None,
            "interval_quality": "ok" if (pd.notna(cover) and cover >= COVERAGE_MIN) else "poor",
            "width_scaled_ok": width_ok,
            "beats_persistence": bool(np.isfinite(perf_ratio) and perf_ratio >= PERF_TOL),
            "direction_supported": (
                bool(np.isfinite(sa) and sa >= DIR_GOOD) if pd.notna(sa) else False),
            "direction_marginal": (
                bool(np.isfinite(sa) and DIR_MARGINAL <= sa < DIR_GOOD)
                if pd.notna(sa) else False),
        }

    out = {
        "framework": "evidence-based confidence (not interval width alone)",
        "levels": ["HIGH", "DIRECTIONAL", "LOW"],
        "thresholds": {
            "coverage_min": COVERAGE_MIN,
            "direction_good": DIR_GOOD,
            "direction_marginal": DIR_MARGINAL,
            "perf_vs_persist_min": PERF_TOL,
            "horizon_row_score_high": 0.70,
            "horizon_row_score_directional": 0.45,
        },
        "bucket_rules": rules,
        "weights": {
            "interval_quality": 0.15, "perf_vs_persist": 0.20, "direction": 0.20,
            "width_scaled": 0.10, "station_integrity": 0.15, "driver_availability": 0.10,
            "anchor_ood": 0.05, "trajectory_stability": 0.05,
        },
        "runtime_note": ("station integrity, driver source (Open-Meteo<=16d forecast / "
                         "climatology beyond / unavailable), anchor OOD and trajectory "
                         "stability are evaluated at inference; a missing driver or "
                         "uncovered requirement must downgrade the horizon to "
                         "DIRECTIONAL/LOW."),
    }
    RE_OUT.write_text(json.dumps(out, indent=2))

    print(f"wrote {RE_OUT}")
    for b in buckets:
        r = rules[str(b)]
        beats = "Y" if r["beats_persistence"] else "N"
        print(f"  {b:>4}h  cov={r['calibrated_coverage']}  "
              f"ratio={r['perf_vs_persist_ratio']}  sign={r['sign_acc']}  "
              f"int={r['interval_quality']}  beats={beats}")


# ====================== paired DL-vs-XGB CI (from 35_traj_paired_ci.py)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KEYS = ["station", "anchor_t", "horizon_h"]
N_BOOT = 2000
CI_SEED = 20261011


def _rmse(y, p):
    return float(math.sqrt(np.mean((y - p) ** 2)))


def _load_pairs(suffix):
    return pd.read_parquet(OUT / f"traj_backtest_rows{suffix}.parquet")


def _pair(horizon):
    dl = _load_pairs("_dl")
    xgb = _load_pairs("")
    dl = dl[dl["horizon_h"] == horizon][KEYS + ["obs", "traj_q50"]]
    xgb = xgb[xgb["horizon_h"] == horizon][KEYS + ["obs", "traj_q50"]]
    m = dl.merge(xgb, on=KEYS, suffixes=("_dl", "_xgb"))
    assert np.allclose(m["obs_dl"], m["obs_xgb"]), "obs mismatch across models"
    return m


def _bootstrap(m, rng):
    stations = m["station"].to_numpy()
    uniq = np.unique(stations)
    idx_by_station = {s: np.where(stations == s)[0] for s in uniq}
    y = m["obs_dl"].to_numpy()
    p_dl = m["traj_q50_dl"].to_numpy()
    p_xgb = m["traj_q50_xgb"].to_numpy()
    deltas = np.empty(N_BOOT)
    ratios = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by_station[s] for s in pick])
        r_dl = _rmse(y[ix], p_dl[ix])
        r_xgb = _rmse(y[ix], p_xgb[ix])
        deltas[b] = r_dl - r_xgb
        ratios[b] = r_dl / r_xgb
    return deltas, ratios


def paired_ci_main():
    rng = np.random.default_rng(CI_SEED)
    horizons = sorted(pd.unique(_load_pairs("_dl")["horizon_h"]).tolist())
    rows = []
    print(f"{'h':>5} {'n':>6} {'rmse_dl':>9} {'rmse_xgb':>9} "
          f"{'delta':>9} {'ratio':>7} {'ratio_lo':>9} {'ratio_hi':>9} beat")
    for h in horizons:
        m = _pair(h)
        y = m["obs_dl"].to_numpy()
        r_dl = _rmse(y, m["traj_q50_dl"].to_numpy())
        r_xgb = _rmse(y, m["traj_q50_xgb"].to_numpy())
        deltas, ratios = _bootstrap(m, rng)
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        beat = bool(hi < 1.0)
        print(f"{h:>5} {len(m):>6} {r_dl:>9.4f} {r_xgb:>9.4f} "
              f"{r_dl - r_xgb:>9.4f} {r_dl / r_xgb:>7.4f} "
              f"{lo:>9.4f} {hi:>9.4f} {beat}")
        rows.append({"horizon_h": int(h), "n": int(len(m)),
                     "rmse_dl": r_dl, "rmse_xgb": r_xgb,
                     "delta": r_dl - r_xgb, "ratio": r_dl / r_xgb,
                     "ratio_lo": float(lo), "ratio_hi": float(hi),
                     "dl_beats_xgb": beat,
                     "delta_lo": float(np.percentile(deltas, 2.5)),
                     "delta_hi": float(np.percentile(deltas, 97.5))})

    res = {"n_boot": N_BOOT, "seed": CI_SEED, "per_horizon": rows,
           "dl_beats_xgb_all_horizons": all(r["dl_beats_xgb"] for r in rows),
           "dl_beats_xgb_30d": next(r["dl_beats_xgb"] for r in rows
                                    if r["horizon_h"] == 720)}
    (OUT / "traj_paired_ci_dl_vs_xgb.json").write_text(json.dumps(res, indent=2))
    print("\n30d (h=720) DL beats XGB:", res["dl_beats_xgb_30d"],
          "| all horizons:", res["dl_beats_xgb_all_horizons"])
    print("wrote", OUT / "traj_paired_ci_dl_vs_xgb.json")


# ================================================================ batch CLI
STAGES = {
    "datasets": ("pipelines.traj", "datasets_main"),
    "train": ("pipelines.traj", "train_main"),
    "backtest": ("pipelines.traj", "backtest_main"),
    "reliability": ("pipelines.traj", "reliability_main"),
    "paired_ci": ("pipelines.traj", "paired_ci_main"),
    "train_dl": ("pipelines.traj_dl", "train_dl_main"),
    "train_pooled_dl": ("pipelines.traj_dl", "train_pooled_dl_main"),
    "backtest_pooled_dl": ("pipelines.traj_dl", "backtest_pooled_dl_main"),
}
ALIASES = {
    "30_traj_datasets": "datasets", "31_train_traj": "train",
    "32_backtest_traj": "backtest", "33_traj_reliability": "reliability",
    "34_train_traj_dl": "train_dl", "35_traj_paired_ci": "paired_ci",
    "36_train_pooled_dl": "train_pooled_dl",
    "37_backtest_pooled_dl": "backtest_pooled_dl",
}


def run_stage(stage: str) -> None:
    from importlib import import_module

    key = ALIASES.get(stage, stage)
    if key not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from: {', '.join(STAGES)}")
    module, func = STAGES[key]
    saved_argv = sys.argv
    if len(saved_argv) > 1 and saved_argv[1] == stage:
        sys.argv = [saved_argv[0]] + saved_argv[2:]
    try:
        getattr(import_module(module), func)()
    finally:
        sys.argv = saved_argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "--list"):
        print("stages: " + ", ".join(STAGES))
        print("aliases: " + ", ".join(ALIASES))
        return 0
    run_stage(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
