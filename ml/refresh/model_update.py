"""refresh/model_update.py — scheduled full refit + gated promotion of the
shared pooled trajectory model.

The XGBoost quantile models here are pooled across all stations and horizons;
incremental boosting is NOT a safe update strategy for the current
reg:quantileerror/hist setup (later rounds do not rebalance earlier quantile
crossings), so an update is a FULL refit on the moving window:

    train   = observed data <  now - (holdout + backtest) days
    val     = [now - (holdout + backtest), now - backtest)   (early stopping)
    backtest= [now - backtest_days, now)                     (honest anchors)

The CANDIDATE is trained to data/refresh/candidate, backtested against the
INCUMBENT (current production models) on the SAME anchors (identical feature
matrices), and promoted ONLY if it passes the quality gates:

    1. RMSE(candidate, 30d) < RMSE(incumbent, 30d) * (1 + promote_tolerance)
    2. RMSE(candidate, 30d) < RMSE(persistence, 30d)
    3. candidate calibrated 30d coverage >= coverage_target
    4. short-horizon (<= 24 h) candidate beats persistence

On any failure the production model, calibration, reliability and metric files
are left untouched (previous model remains live). On success the swap is atomic
(backup -> install -> verify), the version is bumped, and the app cache picks
the new weights up automatically via the mtime fingerprint.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import importlib

F = importlib.import_module("06_features")
T = importlib.import_module("30_traj_datasets")

ROOT = Path(__file__).resolve().parent.parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
PROD_MODELS = ROOT / "models"
CAND_DIR = ROOT / "data" / "refresh" / "candidate"
METRICS_OUT = ROOT / "outputs" / "traj_backtest_metrics.csv"
RELI_SCRIPT = "33_traj_reliability.py"

ALPHAS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
MAX_H = 120
SEED = 42
EVAL_FRAC = 0.10
# Sampling stride in the candidate long-frame. 16 = half the rows of the old 8
# (the box has limited RAM and the shared scheduler runs alongside a live app);
# the much larger historical window still dominates the fit while keeping the
# candidate's peak memory under the machine's available headroom.
TRAIN_STRIDE = 16
HORIZONS_TRAIN = list(range(1, 21)) + [24, 28, 32, 36, 40, 48, 56, 64, 80, 96, 120]
HORIZONS_VAL = list(range(1, MAX_H + 1))
PARAMS = dict(max_depth=7, learning_rate=0.05, min_child_weight=60,
              subsample=0.9, colsample_bytree=0.8, n_estimators=900,
              tree_method="hist", max_bin=128, n_jobs=6, random_state=SEED)
EARLY = 25


class RetrainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# candidate long-frame (moving cutoffs)
# ---------------------------------------------------------------------------
def _cutoffs(cfg, now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    now = now or pd.Timestamp.now()
    h = pd.Timedelta(days=cfg.candidate_holdout_days)
    w = pd.Timedelta(days=cfg.candidate_backtest_window_days)
    train_cut = now - h - w
    val_end = now - w
    return train_cut, val_end, now


def build_candidate_longframe(cfg, *, tbl: pd.DataFrame | None = None,
                              out_dir: Path = CAND_DIR, smoke: bool = False,
                              smoke_stations: int = 3) -> dict:
    train_cut, val_end, now = _cutoffs(cfg)
    if tbl is None:
        tbl = pd.read_parquet(ALIGNED)
        tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")

    feats, _ = F.build_full(tbl, keep_na=True, horizon_steps=MAX_H, horizon_label=30)
    feats["time"] = pd.to_datetime(feats["time"], errors="coerce")
    for c in list(F.NUM_COLS) + ["h", "h_sin", "h_cos"]:
        if c in feats.columns:
            feats[c] = feats[c].astype(np.float32)
    if smoke:
        stations = sorted(feats["Station"].astype(str).unique())[:smoke_stations]
        feats = feats[feats["Station"].astype(str).isin(stations)]

    out_dir.mkdir(parents=True, exist_ok=True)
    tr_path, va_path = out_dir / "train.parquet", out_dir / "val.parquet"
    tr_path.unlink(missing_ok=True)
    va_path.unlink(missing_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Per-station row budgets so the total stays inside candidate_*_row_cap.
    # We stride-sample each station's *expanded* rows uniformly (deterministic
    # order = station then horizon) instead of dropping whole stations, which
    # would bias the fleet. This bounds both disk and the DMatrix in RAM.
    all_st = feats["Station"].astype(str).unique()
    tr_budget = max(8, cfg.candidate_train_row_cap // max(1, len(all_st)))
    va_budget = max(8, cfg.candidate_val_row_cap // max(1, len(all_st)))

    # Write per-station expanded chunks straight to parquet (streaming) instead
    # of accumulating every station's rows in RAM. Peak memory is now bounded
    # by a single station's feature blocks rather than the whole candidate set,
    # which is what was letting the shared scheduler retrain OOM the box.
    writers: dict[Path, pq.ParquetWriter] = {}
    n_tr = n_va = 0
    for st in all_st:
        sub = feats[feats["Station"].astype(str) == st].sort_values("time")
        times_st, gwl_st = T._axis_and_targets([sub[["time", "gwl"]]])
        rows_va = sub[(sub["time"] >= train_cut) & (sub["time"] < val_end)]
        rows_va = rows_va[rows_va["gwl"].notna()]
        rows_tr = sub[(sub["time"] < train_cut) & sub["gwl"].notna()].iloc[::TRAIN_STRIDE]
        if not rows_tr.empty:
            chunk = T._expand(rows_tr, times_st, gwl_st, HORIZONS_TRAIN)
            if not chunk.empty:
                if len(chunk) > tr_budget:
                    k = int(np.ceil(len(chunk) / tr_budget))
                    chunk = chunk.iloc[::k]
                for c in list(F.NUM_COLS) + ["h", "h_sin", "h_cos"]:
                    if c in chunk.columns:
                        chunk[c] = chunk[c].astype(np.float32)
                chunk["h"] = chunk["h"].astype(np.int16)
                chunk["target_delta"] = chunk["target_delta"].astype(np.float32)
                tbl_arrow = pa.Table.from_pandas(chunk, preserve_index=False)
                if tr_path not in writers:
                    writers[tr_path] = pq.ParquetWriter(tr_path, tbl_arrow.schema)
                writers[tr_path].write_table(tbl_arrow)
                n_tr += len(chunk)
                del chunk, tbl_arrow
        if not rows_va.empty:
            chunk = T._expand(rows_va, times_st, gwl_st, HORIZONS_VAL)
            if not chunk.empty:
                if len(chunk) > va_budget:
                    k = int(np.ceil(len(chunk) / va_budget))
                    chunk = chunk.iloc[::k]
                for c in list(F.NUM_COLS) + ["h", "h_sin", "h_cos"]:
                    if c in chunk.columns:
                        chunk[c] = chunk[c].astype(np.float32)
                chunk["h"] = chunk["h"].astype(np.int16)
                chunk["target_delta"] = chunk["target_delta"].astype(np.float32)
                tbl_arrow = pa.Table.from_pandas(chunk, preserve_index=False)
                if va_path not in writers:
                    writers[va_path] = pq.ParquetWriter(va_path, tbl_arrow.schema)
                writers[va_path].write_table(tbl_arrow)
                n_va += len(chunk)
                del chunk, tbl_arrow
        del sub
    for w in writers.values():
        w.close()

    trb = pd.read_parquet(tr_path) if n_tr else pd.DataFrame()
    vab = pd.read_parquet(va_path) if n_va else pd.DataFrame()
    if not trb.empty:
        trb["h"] = trb["h"].astype(np.int16)
    if not vab.empty:
        vab["h"] = vab["h"].astype(np.int16)
    n_stations = int(feats["Station"].astype(str).nunique())
    del feats
    prep = {
        "paradigm": "direct multi-horizon shared model (no recursion)",
        "train_period": [str(train_cut.date()), str(val_end.date())],
        "backtest_window": [str(val_end.date()), str(now.date())],
        "train_rows": int(len(trb)), "val_rows": int(len(vab)),
        "row_budgets": {"train_cap": cfg.candidate_train_row_cap,
                         "val_cap": cfg.candidate_val_row_cap,
                         "per_station_train": tr_budget,
                         "per_station_val": va_budget},
        "stations": n_stations,
        "smoke": bool(smoke),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "prep_candidate.json").write_text(json.dumps(prep, indent=2))
    return prep


def train_candidate(cfg, *, out_dir: Path = CAND_DIR, smoke: bool = False) -> dict:
    import xgboost as xgb
    tr = pd.read_parquet(out_dir / "train.parquet")
    va = pd.read_parquet(out_dir / "val.parquet")
    fn = list(F.NUM_COLS) + ["h", "h_sin", "h_cos"]
    order = [c for c in fn if c in tr.columns]
    if tr.empty or va.empty:
        raise RetrainError("candidate long-frame empty — nothing to train on")
    if smoke:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(tr), size=min(200_000, len(tr)), replace=False)
        tr = tr.iloc[np.sort(keep)].copy()
        va = va[va["h"] <= 30].copy()

    tr = tr.sort_values("time").reset_index(drop=True)
    n_eval = max(1, int(len(tr) * EVAL_FRAC))
    dtrain = xgb.DMatrix(tr.iloc[:-n_eval][order], label=tr.iloc[:-n_eval]["target_delta"])
    deval = xgb.DMatrix(tr.iloc[-n_eval:][order], label=tr.iloc[-n_eval:]["target_delta"])
    dval = xgb.DMatrix(va[order], label=va["target_delta"])

    preds = {}
    for name, alpha in ALPHAS.items():
        p = dict(PARAMS, objective="reg:quantileerror", quantile_alpha=alpha)
        bst = xgb.train(p, dtrain, num_boost_round=PARAMS["n_estimators"],
                        evals=[(deval, "eval")], early_stopping_rounds=EARLY,
                        verbose_eval=False)
        bst.save_model(out_dir / f"traj_xgb_{name}.json")
        preds[name] = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))

    q5, q50, q9 = preds["q05"], preds["q50"], preds["q95"]
    q5 = np.minimum(q5, q50)
    q9 = np.maximum(q9, q50)
    steps = [1, 2, 3, 4, 8, 12, 28, 60, 120] if not smoke else [1, 2, 8, 30]
    per_h = _per_h(va, q5, q50, q9, steps)

    cfgout = {
        "paradigm": "direct multi-horizon shared model (no recursion)",
        "target": "GWL(t+h*6h) - GWL(t); level = anchor + delta (anchor observed only)",
        "feature_cols": order,
        "quantiles": ALPHAS,
        "max_h": MAX_H,
        "early_stopping_tail_frac": EVAL_FRAC,
        "fit_rows": int(dtrain.num_row()),
        "eval_rows": int(deval.num_row()),
        "val_rows_scored": int(dval.num_row()),
        "smoke": smoke,
        "val_per_h_metrics": per_h,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "traj_config.json").write_text(json.dumps(cfgout, indent=2))
    return cfgout


def _per_h(va, q5, q50, q9, steps) -> list[dict]:
    y = va["target_delta"].to_numpy()
    hh = va["h"].to_numpy()
    rows = []
    for h in steps:
        m = hh == h
        if not m.any():
            continue
        yh, p05, p50, p95 = y[m], q5[m], q50[m], q9[m]
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


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------
def run_gate_evaluation(cfg, *, cand_dir: Path = CAND_DIR, tbl=None,
                        mode: str = "flat", future_data=None,
                        forecast_days: int = 16, stations=None) -> dict:
    from refresh import backtest
    train_cut, val_end, now = _cutoffs(cfg)
    if tbl is None:
        tbl = pd.read_parquet(ALIGNED)
        tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
        tbl = F.drop_gwl_spikes(tbl)

    fnames = list(F.NUM_COLS) + ["h", "h_sin", "h_cos"]
    inc_cfg = json.loads((PROD_MODELS / "traj_config.json").read_text())
    if list(inc_cfg["feature_cols"]) != fnames:
        raise RetrainError("production feature schema changed; refusing to retrain")
    if not (cand_dir / "traj_xgb_q50.json").exists():
        raise RetrainError("no candidate model staged")

    tasks = backtest.sample_anchors(
        tbl, window_start=val_end, window_end=now,
        spacing_days=cfg.anchor_spacing_days, n_stations=cfg.backtest_stations,
        stations=stations)
    if len(tasks) < 10:
        raise RetrainError(f"too few backtest anchors ({len(tasks)} < 10) in recent window")

    result = backtest.run_backtest_experiment(
        tbl=tbl, tasks=tasks, fnames=fnames,
        providers={"candidate": cand_dir, "incumbent": PROD_MODELS},
        mode=mode, future_data=future_data, forecast_days=forecast_days, root=ROOT)
    return _gate(cfg, result)


def _gate(cfg, result) -> dict:
    ph = result["per_h"]
    if "candidate" not in ph or "incumbent" not in ph:
        raise RetrainError("backtest produced no metrics for candidate/incumbent")
    r30c = _row(ph["candidate"], 720)
    r30i = _row(ph["incumbent"], 720)
    cal30 = next((c for h, c in zip(result["calib"]["horizons_h"],
                                   result["calib"]["coverage_calibrated"]) if int(h) == 720), None)
    short = ph["candidate"]
    short = short[short["horizon_h"] <= 24] if not short.empty else short
    cand_rmse = float(r30c["rmse"]) if r30c is not None else float("nan")
    inc_rmse = float(r30i["rmse"]) if r30i is not None else float("nan")
    persist = float(r30c["rmse_persist"]) if r30c is not None else float("nan")
    n = int(r30c["n"]) if r30c is not None else 0

    beats_inc = bool(np.isfinite(cand_rmse) and np.isfinite(inc_rmse)
                     and cand_rmse < inc_rmse * (1.0 + cfg.promote_tolerance))
    beats_persist = bool(np.isfinite(cand_rmse) and np.isfinite(persist)
                         and cand_rmse < persist)
    coverage_ok = bool(cal30 is not None and cal30 >= cfg.coverage_target)
    short_ok = bool((not short.empty)
                    and (short["rmse"] < short["rmse_persist"]).all())
    min_n = n >= 30

    gate = {
        "decision": "promote" if (beats_inc and beats_persist and coverage_ok
                                  and short_ok and min_n) else "keep",
        "reasons": {
            "beats_incumbent": beats_inc,
            "beats_persistence": beats_persist,
            "coverage_ok": coverage_ok,
            "short_horizons_ok": short_ok,
            "min_scores": min_n,
        },
        "candidate_30d_rmse": round(cand_rmse, 4),
        "incumbent_30d_rmse": round(inc_rmse, 4),   # same anchors, apples-to-apples
        "persistence_30d_rmse": round(persist, 4),
        "candidate_30d_coverage_raw": round(float(r30c["coverage_raw"]), 4) if r30c is not None else None,
        "candidate_30d_coverage_calibrated": cal30,
        "anchors_scores": n,
        "promote_tolerance": cfg.promote_tolerance,
        "coverage_target": cfg.coverage_target,
        "backtest": {"candidate": ph["candidate"].round(4).to_dict("records"),
                     "incumbent": ph["incumbent"].round(4).to_dict("records")},
        "calibration": result["calib"],
    }
    return gate


def _row(df: pd.DataFrame, hh: int) -> pd.Series | None:
    sub = df[df["horizon_h"] == hh]
    return sub.iloc[0] if len(sub) else None


# ---------------------------------------------------------------------------
# promote (atomic) / reject
# ---------------------------------------------------------------------------
def _install_models(cand_dir: Path) -> None:
    backup_dir = PROD_MODELS
    preserved = {}
    for f in ("traj_xgb_q05.json", "traj_xgb_q50.json", "traj_xgb_q95.json",
              "traj_config.json", "traj_calibration.json", "traj_reliability.json"):
        src = backup_dir / f
        if src.exists():
            preserved[f] = src.with_suffix(src.suffix + ".bak.promote")
            shutil.copy2(src, preserved[f])
    try:
        for f in ("traj_xgb_q05.json", "traj_xgb_q50.json", "traj_xgb_q95.json"):
            shutil.copy2(cand_dir / f, backup_dir / f)
    except Exception:
        for src, dst in preserved.items():
            shutil.copy2(dst, backup_dir / src)
        raise


def rebuild_reliability(per_h_candidate: pd.DataFrame) -> None:
    """Run 33 on the candidate's per-horizon metrics + fresh calibration."""
    if METRICS_OUT.exists():
        shutil.copy2(METRICS_OUT, Path(str(METRICS_OUT) + ".bak.promote"))
    per_h_candidate.to_csv(METRICS_OUT, index=False)
    try:
        proc = subprocess.run([sys.executable, "-u", str(ROOT / RELI_SCRIPT)],
                              cwd=ROOT, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RetrainError(f"33_traj_reliability failed: {proc.stderr[-800:]}")
    finally:
        Path(str(METRICS_OUT) + ".bak.promote").unlink(missing_ok=True)


def promote(cfg, state, gate: dict, cand_dir: Path = CAND_DIR) -> dict:
    """Atomically promote the candidate; returns the promotion record."""
    cand_cfg = json.loads((cand_dir / "traj_config.json").read_text())
    cand_cfg["version"] = cand_cfg.get("trained_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    cand_cfg["promoted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cand_cfg["gate"] = gate["reasons"]

    _install_models(cand_dir)
    (PROD_MODELS / "traj_config.json").write_text(json.dumps(cand_cfg, indent=2))
    (PROD_MODELS / "traj_calibration.json").write_text(json.dumps(gate["calibration"], indent=2))
    rebuild_reliability(pd.DataFrame(gate["backtest"]["candidate"]))

    state.update(
        model_version=cand_cfg["version"],
        model_trained_at=cand_cfg["trained_at"],
        last_retrain_ts=cand_cfg["promoted_at"],
        last_retrain_ok=True,
        last_retrain_error=None,
    )
    return {"promoted": True, "version": cand_cfg["version"],
            "candidate_30d_rmse": gate["candidate_30d_rmse"],
            "incumbent_30d_rmse": gate["incumbent_30d_rmse"]}


def reject(state, gate: dict, error: str = "") -> dict:
    state.update(last_retrain_ts=pd.Timestamp.now().isoformat(),
                 last_retrain_ok=False,
                 last_retrain_error=error or f"gate failed: {gate.get('reasons')}")
    return {"promoted": False, "gate": gate, "error": error}