"""refresh_engines — anchored inference, verification ledger, backtest and gated refit.

Merged from inference/verification/backtest/model_update. Both backtest and refit
score candidate vs incumbent on the SAME recent anchors; promotion writes models
only after all quality gates pass. Forecasts always predict a CHANGE and the level
is observed anchor + delta — the anchor itself is never predicted.
"""
from __future__ import annotations

import importlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import serving_engines as engines
from serving_core import active_trajectory_forecast, traj_backend
from refresh import refresh_core as pubmod
from refresh.refresh_core import (REFRESH_DIR, RefreshState, atomic_write_json,
                                  models_frozen, refresh_lock)

F = importlib.import_module("pipeline_model")
T = importlib.import_module("pipeline_traj")
ROOT = Path(__file__).resolve().parent.parent
F_PATH_ROOT = ROOT   # former backtest alias

# --- inference ---------------------------------------------------------------
LAST_ANCHORS = REFRESH_DIR / "last_anchors.parquet"
DL_CFG = ROOT / "models" / "traj_dl_config.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_engine():
    # the scheduler is long-running: never reuse stale tables/models
    engines.clear_caches()
    return engines._load_xgb()


def current_anchors() -> pd.Series:
    """Per-station latest OBSERVED gwl time from the spike-dropped aligned table."""
    c = _load_engine()
    s = c["tbl"].dropna(subset=["gwl"])
    g = s.groupby(s["Station"].astype(str))["time"].max()
    return g.rename_axis("Station")


def previous_anchors() -> pd.Series:
    if not LAST_ANCHORS.exists():
        return pd.Series(dtype="datetime64[ns]")
    df = pd.read_parquet(LAST_ANCHORS)
    if df.empty or "Station" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return df.set_index(df["Station"].astype(str))["anchor_time"].astype("datetime64[ns]")


def refresh_forecasts(*, cfg, state: RefreshState, stations=None, max_stations: int = 0,
                      skip_unchanged: bool = True, publish: bool = True,
                      dry_run: bool = False, source_report: dict | None = None,
                      allow_empty: bool = False,
                      engine_extra: dict | None = None) -> dict:
    """Compute + publish per-station anchored trajectory forecasts for one cycle.

    ``allow_empty`` is set only when the GWL fetch succeeded with zero new rows:
    an empty result is then a benign ``no_new_source`` no-op. Any other short
    result (a real fetch/inference failure) still aborts the publish.
    """
    c = _load_engine()
    fnames_model = c["fnames"]
    tbl = c["tbl"]
    all_stations = sorted(tbl["Station"].astype(str).unique())
    if stations is None:
        stations = all_stations
    else:
        stations = [s for s in stations if s in set(all_stations)]
    if max_stations:
        stations = stations[:max_stations]
    cur = current_anchors()
    prev = previous_anchors()
    forecasts, errors, unchanged = [], {}, 0
    backend = traj_backend()
    if backend == "dl" and DL_CFG.exists():
        try:
            fnames_model = json.loads(DL_CFG.read_text()).get("feature_cols", fnames_model)
        except Exception:  # noqa: BLE001
            pass
    for station in stations:
        anchor = cur.get(station)
        if skip_unchanged and publish and not dry_run:
            if station in prev.index and prev.loc[station] == anchor:
                unchanged += 1
                continue
        out = active_trajectory_forecast(station)
        if "error" in out:
            errors[station] = out["error"]
        else:
            out["model_features"] = fnames_model
            forecasts.append(out)
    no_new_source = False
    if not dry_run and publish:
        if len(forecasts) < cfg.min_fresh_stations:
            if allow_empty and not forecasts:
                no_new_source = True
            else:
                raise RuntimeError(
                    f"publish aborted: only {len(forecasts)} fresh forecasts "
                    f"(< min_fresh_stations={cfg.min_fresh_stations})")
        if not no_new_source:
            if backend == "dl":
                cfgj = json.loads(DL_CFG.read_text()) if DL_CFG.exists() else {}
                engine = "trajectory-v2-dl"
            else:
                tcfg = engines.TRAJ_CFG
                cfgj = json.loads(tcfg.read_text()) if tcfg.exists() else {}
                engine = "trajectory-v2"
            refreshed_at = _utcnow_iso()
            model_version = cfgj.get("version") or str(cfgj.get("trained_at", ""))
            meta = pubmod.publish_forecasts(
                forecasts, engine=engine, model_version=model_version,
                model_trained_at=cfgj.get("trained_at"), source_report=source_report,
                keep_staging=True)
            # stamp THIS cycle's time (the previous-state value made fresh data
            # report "last refresh 556h ago" right after a successful run)
            pubmod.update_freshness(meta, cfg.stale_after_hours, refreshed_at)
            anchors_df = pd.DataFrame({"Station": list(cur.index),
                                       "anchor_time": list(cur.values)})
            pubmod.atomic_write_parquet(LAST_ANCHORS, anchors_df)
            state.update(last_forecast_refresh_ts=refreshed_at)
    return {
        "dry_run": dry_run,
        "published": bool(publish and not dry_run and not no_new_source),
        "no_new_source": bool(not dry_run and publish and no_new_source),
        "stations": len(stations),
        "forecasted": len(forecasts),
        "unchanged_skipped": unchanged,
        "table_latest": (str(cur.max()) if len(cur) else None),
        "failed": errors,
        "model_features": fnames_model,
        "engine_extra": engine_extra,
    }


def status() -> dict:
    """Convenience summary for the CLI/UI without running anything."""
    meta = pubmod.read_meta()
    anchors = current_anchors()
    return {
        "forecast_generation_ts": meta.get("forecast_generation_ts"),
        "model_version": meta.get("model_version"),
        "model_trained_at": meta.get("model_trained_at"),
        "data_status": meta.get("data_status"),
        "stale_reason": meta.get("stale_reason"),
        "stations_forecasted": meta.get("stations_forecasted"),
        "stations_in_table": int(len(anchors)),
        "latest_observed_ts": (str(anchors.max()) if len(anchors) else None),
    }


# --- verification ------------------------------------------------------------
VERIF_LEDGER = REFRESH_DIR / "verification.parquet"
VERIF_SUMMARY = REFRESH_DIR / "verification_summary.json"
_LEDGER_KEY_COLS = ("station", "anchor_time", "horizon_h", "target_time")
KNOWN_HORIZONS_H = [6, 12, 24, 36, 48, 72, 120, 240, 360, 480, 720]
DRIFT_RMSE_TOL = 1.10
DRIFT_CONSECUTIVE_CYCLES = 2
DRIFT_COVERAGE_MIN = 0.70


def _read_aligned_actuals() -> pd.DataFrame | None:
    try:
        tbl = _load_engine()["tbl"][["Station", "time", "gwl"]].dropna(subset=["gwl"]).copy()
        tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
        return tbl
    except Exception:  # noqa: BLE001
        return None


def _iter_published_forecasts(only_newer_than: pd.Timestamp | None = None):
    """Yield (station, pod, anchor_ts) for archived then current forecast generations."""
    def _yield_generations(d):
        if not d.exists():
            return
        for p in sorted(d.glob("*.json")):
            try:
                pod = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(pod, dict) or "trajectory" not in pod:
                continue
            station, anchor = pod.get("station"), pod.get("anchor_time")
            if not station or not anchor:
                continue
            try:
                anchor_ts = pd.Timestamp(anchor)
            except Exception:  # noqa: BLE001
                continue
            if only_newer_than is not None and anchor_ts <= only_newer_than:
                continue
            yield station, pod, anchor_ts

    if not pubmod.FORECASTS_DIR.exists():
        return
    seen: set[tuple[str, str]] = set()
    for station, pod, anchor_ts in _yield_generations(pubmod.FORECAST_ARCHIVE):
        key = (station, pod.get("forecast_generated") or str(anchor_ts))
        if key in seen:
            continue
        seen.add(key)
        yield station, pod, anchor_ts
    for station, pod, anchor_ts in _yield_generations(pubmod.FORECASTS_DIR):
        yield station, pod, anchor_ts


def _index_actuals(actuals: pd.DataFrame | None) -> dict:
    """Group actuals by station once per verify run (old per-pod scan was O(pods*rows))."""
    if actuals is None or actuals.empty:
        return {}
    if "Station" not in actuals.columns or "time" not in actuals.columns:
        return {}
    return {str(k): g for k, g in actuals.groupby(actuals["Station"].astype(str), sort=False)}


def _score_pod(pod: dict, by_station: dict) -> list[dict]:
    """Score one published forecast against realised values (gaps allowed)."""
    station = pod.get("station")
    anchor_ts = pd.Timestamp(pod.get("anchor_time"))
    model_version = str(pod.get("model") or pod.get("model_version") or "")
    generated_ts = pod.get("forecast_generated") or pod.get("generated_ts")
    steps = pod.get("trajectory")
    if not isinstance(steps, list) or len(steps) < 2:
        return []
    sub = (by_station or {}).get(str(station))
    if sub is None or sub.empty:
        return []
    sub = sub[sub["time"] > anchor_ts]
    if sub.empty:
        return []
    by_time: dict[pd.Timestamp, float] = {}
    for t, v in zip(sub["time"], sub["gwl"]):
        key = pd.Timestamp(t)
        if key not in by_time:
            by_time[key] = v
    rows = []
    for k, step in enumerate(steps, start=1):
        try:
            t_target = pd.Timestamp(step["time"])
        except Exception:  # noqa: BLE001
            continue
        horizon_h = int(k * 6)
        if t_target not in by_time:
            continue
        actual = float(by_time[t_target])
        q50 = float(step.get("q50") or step.get("gwl") or np.nan)
        q05 = float(step.get("q05") or np.nan)
        q95 = float(step.get("q95") or np.nan)
        if not np.isfinite(q50) or not np.isfinite(actual):
            continue
        rows.append({
            "station": station,
            "anchor_time": anchor_ts,
            "horizon_h": horizon_h,
            "target_time": t_target,
            "q05": q05, "q50": q50, "q95": q95,
            "actual": actual,
            "abs_err": abs(actual - q50),
            "sq_err": (actual - q50) ** 2,
            "coverage_hit": bool(not (np.isfinite(q05) and np.isfinite(q95)) or (q05 <= actual <= q95)),
            "model_version": model_version,
            "generated_ts": generated_ts,
        })
    return rows


def _pod_generated_epoch(pod: dict) -> float | None:
    for key in ("forecast_generated", "generated_ts"):
        v = pod.get(key)
        if not v:
            continue
        try:
            return pd.Timestamp(v).timestamp()
        except Exception:  # noqa: BLE001
            pass
    return None


def _load_ledger() -> pd.DataFrame:
    if not VERIF_LEDGER.exists():
        return pd.DataFrame(columns=[
            "station", "anchor_time", "horizon_h", "target_time",
            "q05", "q50", "q95", "actual", "abs_err", "sq_err",
            "coverage_hit", "model_version", "generated_ts"])
    return pd.read_parquet(VERIF_LEDGER)


def _ledger_key_set(ledger: pd.DataFrame) -> set:
    if ledger is None or ledger.empty:
        return set()
    if any(c not in ledger.columns for c in _LEDGER_KEY_COLS):
        return set()
    return set(map(tuple, ledger[list(_LEDGER_KEY_COLS)]
                   .itertuples(index=False, name=None)))


def _row_key(row: dict) -> tuple:
    return tuple(row[c] for c in _LEDGER_KEY_COLS)


def write_ledger(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset=list(_LEDGER_KEY_COLS), keep="last")
    df = df.sort_values(["anchor_time", "station", "horizon_h"])
    VERIF_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = VERIF_LEDGER.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(VERIF_LEDGER)


def _load_backtest_baseline() -> float | None:
    """Reference 30d RMSE for the live model (gate_history -> curated metrics -> None)."""
    try:
        gaterep = REFRESH_DIR / "gate_history.json"
        if gaterep.exists():
            hist = json.loads(gaterep.read_text())
            if isinstance(hist, list) and hist:
                last = hist[-1]
                v = last.get("candidate_30d_rmse") or last.get("incumbent_30d_rmse")
                if v:
                    return float(v)
            elif isinstance(hist, dict) and hist.get("candidate_30d_rmse"):
                return float(hist["candidate_30d_rmse"])
    except Exception:  # noqa: BLE001
        pass
    return None


def build_summary(ledger: pd.DataFrame) -> dict:
    """Rolling 24h/7d/30d MAE, RMSE, bias, coverage + drift verdict."""
    base = {
        "updated_ts": _utcnow_iso(),
        "ledger_rows": int(len(ledger)),
        "windows": {},
        "drift": {"tripped": False, "reason": None, "windows": 0},
        "baseline_30d_rmse": None,
    }
    if ledger.empty:
        return base
    anchor = pd.to_datetime(ledger["anchor_time"], errors="coerce").max()
    baseline = _load_backtest_baseline()
    base["baseline_30d_rmse"] = baseline
    led = ledger.copy()
    led["abs_err"] = pd.to_numeric(led["abs_err"], errors="coerce")
    led["sq_err"] = pd.to_numeric(led["sq_err"], errors="coerce")
    led["bias"] = pd.to_numeric(led["q50"], errors="coerce") - pd.to_numeric(led["actual"], errors="coerce")
    led["coverage_hit"] = led["coverage_hit"].fillna(True).astype(bool)
    trip_reason = None
    for label, days in (("24h", 1), ("7d", 7), ("30d", 30)):
        sub = led[led["anchor_time"] >= anchor - pd.Timedelta(days=days)]
        win = {
            "n": int(len(sub)),
            "mae": float(sub["abs_err"].mean()) if len(sub) else None,
            "rmse": float(np.sqrt(sub["sq_err"].mean())) if len(sub) else None,
            "bias": float(sub["bias"].mean()) if len(sub) else None,
            "coverage_95": float(sub["coverage_hit"].mean()) if len(sub) else None,
        }
        base["windows"][label] = win
        if label == "30d" and len(sub) >= 10:
            rmse, cov = win["rmse"], win["coverage_95"]
            if baseline is not None and rmse is not None and rmse > baseline * DRIFT_RMSE_TOL:
                trip_reason = f"30d RMSE {rmse:.3f} > baseline {baseline:.3f} * {DRIFT_RMSE_TOL}"
            elif cov is not None and cov < DRIFT_COVERAGE_MIN:
                trip_reason = f"30d coverage {cov:.3f} < {DRIFT_COVERAGE_MIN}"
    # single-cycle trip; scheduler turns this into a persistent consecutive streak
    base["drift"] = {
        "tripped": trip_reason is not None,
        "reason": trip_reason,
        "windows": 0 if trip_reason is None else DRIFT_CONSECUTIVE_CYCLES,
        "required": DRIFT_CONSECUTIVE_CYCLES,
    }
    return base


def verify_forecasts(*, min_generation_age_s: int = 300, dry_run: bool = False) -> dict:
    """Score old-enough published forecasts, de-dup into the ledger, rebuild summary."""
    if dry_run:
        return {"scored": 0, "appended": 0, "updated": False}
    now = time.time()
    generated_cutoff = now - min_generation_age_s
    ledger = _load_ledger()
    existing_keys = _ledger_key_set(ledger)
    new_rows: list[dict] = []
    by_station = _index_actuals(_read_aligned_actuals())
    for station, pod, anchor_ts in _iter_published_forecasts():
        gen = _pod_generated_epoch(pod)
        if gen is not None and gen > now:          # paranoid: future timestamps
            continue
        if gen is not None and gen >= generated_cutoff:
            continue
        for r in _score_pod(pod, by_station):
            if _row_key(r) in existing_keys:
                continue
            new_rows.append(r)
    appended = 0
    if new_rows:
        fresh = pd.DataFrame(new_rows)
        if not ledger.empty:
            fresh = pd.concat([ledger, fresh], ignore_index=True)
        write_ledger(fresh)
        ledger = fresh
        appended = len(new_rows)
    summary = build_summary(ledger)
    summary["drift"].setdefault("updated_ts", summary.get("updated_ts"))
    VERIF_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(VERIF_SUMMARY, summary)
    return {"scored": len(new_rows), "appended": appended,
            "ledger_rows": int(len(ledger)), "updated": True, "summary": summary}


def drift_status(state: dict | None) -> dict:
    """Direct scheduler-facing drift status (no rebuild). Reads the summary."""
    if not VERIF_SUMMARY.exists():
        return {"tripped": False, "reason": None, "updated_ts": None}
    try:
        return json.loads(VERIF_SUMMARY.read_text()).get("drift", {}) or \
            {"tripped": False, "reason": None, "updated_ts": None}
    except Exception:  # noqa: BLE001
        return {"tripped": False, "reason": None, "updated_ts": None}


# --- backtest ----------------------------------------------------------------
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
                     future_data=None, forecast_days: int = 16
                     ) -> tuple[np.ndarray, float] | None:
    hist = sta[sta["time"] <= anchor_t]
    if len(hist) < 30:
        return None
    hist_d = hist.copy()
    hist_d["date"] = hist_d["time"].dt.normalize()
    if mode == "future":
        from refresh.refresh_data import build_step_features
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
    X = np.hstack([np.repeat(base[None, :], MAX_H, axis=0), hs[:, None],
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
    last["st_id"] = (ord30["stations"].index(station) if station in ord30["stations"] else -1)
    district = str(sta["District"].iloc[0]).upper()
    last["dist_id"] = (ord30["districts"].index(district) if district in ord30["districts"] else -1)
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


def _metrics(g: pd.DataFrame) -> dict:
    obs, rec = g["obs"].to_numpy(), g["traj_q50"].to_numpy()
    q05, q95 = g["traj_q05"].to_numpy(), g["traj_q95"].to_numpy()
    e = obs - rec
    d_obs, d_rec = obs - g["anchor"].to_numpy(), rec - g["anchor"].to_numpy()
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
    """Per-horizon widening fitted on one half of anchors, reported on the other."""
    out = {"horizons_h": [], "widening_s": [], "coverage_raw": [],
           "coverage_calibrated": [], "coverage_calibrated_fit": [],
           "coverage_raw_heldout": [], "n": [], "n_fit": [], "n_heldout": []}
    if df.empty:
        return out
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
        if gf.empty or gh.empty:
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
        out["coverage_raw"].append(round(float(np.mean((h_lo <= h_obs) & (h_obs <= h_hi))), 3))
        out["coverage_raw_heldout"].append(round(float(np.mean((h_lo <= h_obs) & (h_obs <= h_hi))), 3))
        out["coverage_calibrated_fit"].append(round(_cov(f_med, f_half, f_obs, s), 3))
        out["coverage_calibrated"].append(round(_cov(h_med, h_half, h_obs, s), 3))
        out["n"].append(int(len(gh)))
        out["n_fit"].append(int(len(gf)))
        out["n_heldout"].append(int(len(gh)))
    return out


def run_backtest_experiment(*, tbl, tasks, fnames: list[str], providers: dict,
                            mode: str = "flat", future_data=None,
                            forecast_days: int = 16, root=None,
                            clim_by_station: dict | None = None) -> dict:
    """Score every provider (label -> model dir) on the same anchor set."""
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
        obs_times, obs_gwl = sta["time"].to_numpy(), sta["gwl"].to_numpy()
        clim = (clim_by_station or {}).get(station, {})
        d6h_clim = clim.get(int(pd.Timestamp(anchor_t).dayofyear), np.nan)
        direct30 = _direct30_level(sta, anchor_t, ord30) if ord30 is not None else None
        for label, mdir in providers.items():
            q05, q50, q95 = observ[label]
            q05, q95 = np.minimum(q05, q50), np.maximum(q95, q50)
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
                    "traj_q95": float(q95[h - 1]), "persist": float(anc),
                    "clim": float(anc + d6h_clim * h) if np.isfinite(d6h_clim) else np.nan,
                    "direct30": direct30 if hh == 720 else None,
                })
    df = pd.DataFrame(recs)
    per_h = {}
    for label in providers:
        rows = []
        for h, hh in zip(HORIZONS_STEPS, HORIZONS_H):
            g = df[(df["label"] == label) & (df["horizon_h"] == hh)]
            if not g.empty:
                rows.append({**{"horizon_h": hh, "steps": h}, **_metrics(g)})
        per_h[label] = pd.DataFrame(rows)
    return {"rows": df, "per_h": per_h, "calib": _calibrate(df)}


# --- scheduled full refit + gated promotion -----------------------------------
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
PROD_MODELS = ROOT / "models"
CAND_DIR = ROOT / "data" / "refresh" / "candidate"
METRICS_OUT = ROOT / "outputs" / "traj_backtest_metrics.csv"
RELI_STAGE = "reliability"
ALPHAS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
EVAL_FRAC = 0.10
TRAIN_STRIDE = 16
HORIZONS_TRAIN = list(range(1, 21)) + [24, 28, 32, 36, 40, 48, 56, 64, 80, 96, 120]
HORIZONS_VAL = list(range(1, MAX_H + 1))
PARAMS = dict(max_depth=7, learning_rate=0.05, min_child_weight=60,
              subsample=0.9, colsample_bytree=0.8, n_estimators=900,
              tree_method="hist", max_bin=128, n_jobs=6, random_state=SEED)
EARLY = 25


class RetrainError(RuntimeError):
    pass


def _cutoffs(cfg, now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    now = now or pd.Timestamp.now()
    h, w = pd.Timedelta(days=cfg.candidate_holdout_days), pd.Timedelta(days=cfg.candidate_backtest_window_days)
    return now - h - w, now - w, now


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
    all_st = feats["Station"].astype(str).unique()
    tr_budget = max(8, cfg.candidate_train_row_cap // max(1, len(all_st)))
    va_budget = max(8, cfg.candidate_val_row_cap // max(1, len(all_st)))
    # Stream per-station chunks straight to parquet so peak RAM stays bounded.
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
            chunk = chunk[chunk["target_time"] < train_cut]  # no training label inside val window
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
            chunk = chunk[chunk["target_time"] < val_end]  # no val label inside backtest window
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
        "stations": n_stations, "smoke": bool(smoke),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "prep_candidate.json").write_text(json.dumps(prep, indent=2))
    return prep


def _per_h(va, q5, q50, q9, steps) -> list[dict]:
    y, hh = va["target_delta"].to_numpy(), va["h"].to_numpy()
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


def train_candidate(cfg, *, out_dir: Path = CAND_DIR, smoke: bool = False) -> dict:
    import xgboost as xgb
    tr, va = pd.read_parquet(out_dir / "train.parquet"), pd.read_parquet(out_dir / "val.parquet")
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
    q5, q9 = np.minimum(q5, q50), np.maximum(q9, q50)
    steps = [1, 2, 3, 4, 8, 12, 28, 60, 120] if not smoke else [1, 2, 8, 30]
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
        "val_per_h_metrics": _per_h(va, q5, q50, q9, steps),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "traj_config.json").write_text(json.dumps(cfgout, indent=2))
    return cfgout


def _row(df: pd.DataFrame, hh: int) -> pd.Series | None:
    sub = df[df["horizon_h"] == hh]
    return sub.iloc[0] if len(sub) else None


def _gate(cfg, result) -> dict:
    ph = result["per_h"]
    if "candidate" not in ph or "incumbent" not in ph:
        raise RetrainError("backtest produced no metrics for candidate/incumbent")
    r30c, r30i = _row(ph["candidate"], 720), _row(ph["incumbent"], 720)
    cal = result["calib"]
    hs = list(cal["horizons_h"])

    def _at(key: str, hh: float = 720.0):
        vals = list(cal.get(key) or [])
        for i, h in enumerate(hs):
            if int(h) == int(hh) and i < len(vals):
                return vals[i]
        return None

    cal30, cal30_fit, cal30_raw = (_at("coverage_calibrated"),
                                   _at("coverage_calibrated_fit"),
                                   _at("coverage_raw_heldout"))
    short = ph["candidate"]
    short = short[short["horizon_h"] <= 24] if not short.empty else short
    cand_rmse = float(r30c["rmse"]) if r30c is not None else float("nan")
    inc_rmse = float(r30i["rmse"]) if r30i is not None else float("nan")
    persist = float(r30c["rmse_persist"]) if r30c is not None else float("nan")
    n = int(r30c["n"]) if r30c is not None else 0
    beats_inc = bool(np.isfinite(cand_rmse) and np.isfinite(inc_rmse)
                     and cand_rmse < inc_rmse * (1.0 + cfg.promote_tolerance))
    beats_persist = bool(np.isfinite(cand_rmse) and np.isfinite(persist) and cand_rmse < persist)
    coverage_ok = bool(cal30 is not None and cal30 >= cfg.coverage_target)
    short_ok = bool((not short.empty) and (short["rmse"] < short["rmse_persist"]).all())
    min_n = n >= 30
    return {
        "decision": "promote" if (beats_inc and beats_persist and coverage_ok
                                  and short_ok and min_n) else "keep",
        "reasons": {"beats_incumbent": beats_inc, "beats_persistence": beats_persist,
                    "coverage_ok": coverage_ok, "short_horizons_ok": short_ok,
                    "min_scores": min_n},
        "candidate_30d_rmse": round(cand_rmse, 4),
        "incumbent_30d_rmse": round(inc_rmse, 4),   # same anchors, apples-to-apples
        "persistence_30d_rmse": round(persist, 4),
        "candidate_30d_coverage_raw": round(float(r30c["coverage_raw"]), 4) if r30c is not None else None,
        "candidate_30d_coverage_calibrated": cal30,
        "candidate_30d_coverage_calibrated_fit_half": cal30_fit,
        "candidate_30d_coverage_raw_heldout": cal30_raw,
        "coverage_basis": ("widening fitted on the fit half of the anchors; "
                           "coverage_ok reads the DISJOINT held-out half"),
        "anchors_scores": n,
        "promote_tolerance": cfg.promote_tolerance,
        "coverage_target": cfg.coverage_target,
        "backtest": {"candidate": ph["candidate"].round(4).to_dict("records"),
                     "incumbent": ph["incumbent"].round(4).to_dict("records")},
        "calibration": result["calib"],
    }


def run_gate_evaluation(cfg, *, cand_dir: Path = CAND_DIR, tbl=None,
                        mode: str = "flat", future_data=None,
                        forecast_days: int = 16, stations=None) -> dict:
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
    tasks = sample_anchors(tbl, window_start=val_end, window_end=now,
                           spacing_days=cfg.anchor_spacing_days,
                           n_stations=cfg.backtest_stations, stations=stations)
    if len(tasks) < 10:
        raise RetrainError(f"too few backtest anchors ({len(tasks)} < 10) in recent window")
    result = run_backtest_experiment(
        tbl=tbl, tasks=tasks, fnames=fnames,
        providers={"candidate": cand_dir, "incumbent": PROD_MODELS},
        mode=mode, future_data=future_data, forecast_days=forecast_days, root=ROOT)
    return _gate(cfg, result)


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
    """Run the reliability stage on candidate metrics + fresh calibration."""
    if METRICS_OUT.exists():
        shutil.copy2(METRICS_OUT, Path(str(METRICS_OUT) + ".bak.promote"))
    per_h_candidate.to_csv(METRICS_OUT, index=False)
    try:
        proc = subprocess.run([sys.executable, "-u", "-m", "pipeline_traj", RELI_STAGE],
                              cwd=ROOT, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RetrainError(f"reliability stage failed: {proc.stderr[-800:]}")
    finally:
        Path(str(METRICS_OUT) + ".bak.promote").unlink(missing_ok=True)


def promote(cfg, state, gate: dict, cand_dir: Path = CAND_DIR) -> dict:
    """Atomically promote the candidate; returns the promotion record."""
    if models_frozen(cfg):
        raise RetrainError(
            f"models frozen until {cfg.models_frozen_until}: promotion blocked")
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


# Backwards-compatible module aliases (old refresh.inference/verification/... share this module).
inference = verification = backtest = model_update = sys.modules[__name__]
