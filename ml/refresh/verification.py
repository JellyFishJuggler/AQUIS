"""refresh/verification.py — realized-vs-forecast ledger and drift detection.

After every forecast *publish*, walk the freshly published per-station
forecasts, join them against the realised GWL readings that have since landed
in the aligned table, and append a compact vertical ledger row per
(station, anchor_time, horizon). Rolling summaries (24h/7d/30d MAE, RMSE, bias,
credible coverage) are rebuilt on demand, and a cross-cycle drift rule drives
the scheduler's forced retrain.

Outputs:
  * data/refresh/verification.parquet         append-only ledger (vertical)
  * data/refresh/verification_summary.json    rolling metrics + drift status
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from refresh import publish as pubmod
from refresh.config import REFRESH_DIR
from refresh.publish import atomic_write_json

VERIF_LEDGER = REFRESH_DIR / "verification.parquet"
VERIF_SUMMARY = REFRESH_DIR / "verification_summary.json"

# drift rule (config defaults are overridable via refresh_config.json later)
KNOWN_HORIZONS_H = [6, 12, 24, 36, 48, 72, 120, 240, 360, 480, 720]
DRIFT_RMSE_TOL = 1.10          # realised 30d RMSE may exceed backtest baseline by 10%
DRIFT_CONSECUTIVE_CYCLES = 2   # must trip for this many cycles before forcing retrain
DRIFT_COVERAGE_MIN = 0.70      # realised 95% coverage may not collapse below this


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_aligned_actuals() -> pd.DataFrame | None:
    """Recent realised GWL readings (spike-dropped), for matching against anchors."""
    try:
        from refresh import inference
        eng = inference._load_engine()
        tbl = eng["tbl"][["Station", "time", "gwl"]].dropna(subset=["gwl"])
        tbl = tbl.copy()
        tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
        return tbl
    except Exception:  # noqa: BLE001
        return None


def _iter_published_forecasts(only_newer_than: pd.Timestamp | None = None):
    """Yield (station, pod) for every forecast generation worth scoring.

    Yields archived generations first, then the current per-station files, so
    realisations that landed between publish cycles can be scored against the
    forecast that was live at the time. Fails-open (a corrupt/unknown file is
    skipped, never fatal).
    """
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
            station = pod.get("station")
            anchor = pod.get("anchor_time")
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


def _score_pod(pod: dict, seen: pd.DataFrame) -> list[dict]:
    """Score one published forecast against realised values.

    Returns ledger rows (station, anchor_time, horizon_h, target_time,
    q05, q50, q95, actual, abs_err, sq_err, coverage_hit, model_version,
    generated_ts). Every horizon that has both a published step and a realised
    reading is scored (gaps allowed — actuals may arrive sporadically).
    """
    station = pod.get("station")
    anchor_ts = pd.Timestamp(pod.get("anchor_time"))
    model_version = str(pod.get("model") or pod.get("model_version") or "")
    generated_ts = pod.get("forecast_generated") or pod.get("generated_ts")

    steps = pod.get("trajectory")
    if not isinstance(steps, list) or len(steps) < 2:
        return []

    # realised readings strictly after the anchor (the anchor itself is
    # historical, already known when the forecast was made)
    if seen is None or seen.empty:
        return []
    sub = seen[seen["Station"].astype(str) == station]
    sub = sub[sub["time"] > anchor_ts]
    if sub.empty:
        return []

    rows = []
    for k, step in enumerate(steps, start=1):
        try:
            t_target = pd.Timestamp(step["time"])
        except Exception:  # noqa: BLE001
            continue
        horizon_h = int(k * 6)
        hit = sub[sub["time"] == t_target]
        if hit.empty:
            continue
        actual = float(hit["gwl"].iloc[0])
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
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "actual": actual,
            "abs_err": abs(actual - q50),
            "sq_err": (actual - q50) ** 2,
            "coverage_hit": bool(not (np.isfinite(q05) and np.isfinite(q95)) or (q05 <= actual <= q95)),
            "model_version": model_version,
            "generated_ts": generated_ts,
        })
    return rows


def verify_forecasts(*, min_generation_age_s: int = 300, dry_run: bool = False) -> dict:
    """Score all published forecasts whose generation is old enough to have
    collected realisations, append credible rows to the ledger, and rebuild
    the rolling summary.

    Returns the fresh summary dict (or a short report on dry-run / skips).
    """
    if dry_run:
        return {"scored": 0, "appended": 0, "updated": False}

    # only score forecasts at least N seconds old, so the very first cycle
    # (publish then immediate verify) doesn't compare against nothing
    now = time.time()
    generated_cutoff = now - min_generation_age_s

    ledger = _load_ledger()
    new_rows: list[dict] = []

    # load realised readings once per run (fresh aligned table, not a stale cache)
    actuals = _read_aligned_actuals()

    for station, pod, anchor_ts in _iter_published_forecasts():
        gen = _pod_generated_epoch(pod)
        if gen is not None and gen > now:          # paranoid: future timestamps
            continue
        if gen is not None and gen >= generated_cutoff:
            continue
        rows = _score_pod(pod, actuals)
        for r in rows:
            # de-dup: skip if an identical (station, anchor, horizon, target)
            # row already exists in the ledger from a prior cycle
            if _dup_exists(ledger, r):
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

    return {
        "scored": len(new_rows),
        "appended": appended,
        "ledger_rows": int(len(ledger)),
        "updated": True,
        "summary": summary,
    }


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


def _dup_exists(ledger: pd.DataFrame, row: dict) -> bool:
    if ledger.empty:
        return False
    d = ledger[(ledger["station"] == row["station"])
               & (ledger["anchor_time"] == row["anchor_time"])
               & (ledger["horizon_h"] == row["horizon_h"])
               & (ledger["target_time"] == row["target_time"])]
    return not d.empty


def write_ledger(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(
        subset=["station", "anchor_time", "horizon_h", "target_time"], keep="last")
    df = df.sort_values(["anchor_time", "station", "horizon_h"])
    VERIF_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = VERIF_LEDGER.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(VERIF_LEDGER)


def build_summary(ledger: pd.DataFrame) -> dict:
    """Rolling realised-error metrics + drift status.

    24h/7d/30d windows (relative to ledger's latest anchor), each reporting
    MAE, RMSE, bias (mean signed err), n and 95% cred coverage. Drift trips
    when 30d RMSE > backtest baseline * DRIFT_RMSE_TOL *or* realised coverage
    < DRIFT_COVERAGE_MIN for DRIFT_CONSECUTIVE_CYCLES in a row.
    """
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

    # realise q50 values as float64
    led = ledger.copy()
    led["abs_err"] = pd.to_numeric(led["abs_err"], errors="coerce")
    led["sq_err"] = pd.to_numeric(led["sq_err"], errors="coerce")
    led["bias"] = pd.to_numeric(led["q50"], errors="coerce") - pd.to_numeric(led["actual"], errors="coerce")
    led["coverage_hit"] = led["coverage_hit"].fillna(True).astype(bool)

    drift_windows = 0
    trip_reason = None
    for label, days in (("24h", 1), ("7d", 7), ("30d", 30)):
        lo = anchor - pd.Timedelta(days=days)
        sub = led[led["anchor_time"] >= lo]
        win = {
            "n": int(len(sub)),
            "mae": float(sub["abs_err"].mean()) if len(sub) else None,
            "rmse": float(np.sqrt(sub["sq_err"].mean())) if len(sub) else None,
            "bias": float(sub["bias"].mean()) if len(sub) else None,
            "coverage_95": float(sub["coverage_hit"].mean()) if len(sub) else None,
        }
        base["windows"][label] = win
        if label == "30d" and len(sub) >= 10:
            rmse = win["rmse"]
            cov = win["coverage_95"]
            if baseline is not None and rmse is not None and rmse > baseline * DRIFT_RMSE_TOL:
                trip_reason = f"30d RMSE {rmse:.3f} > baseline {baseline:.3f} * {DRIFT_RMSE_TOL}"
            elif cov is not None and cov < DRIFT_COVERAGE_MIN:
                trip_reason = f"30d coverage {cov:.3f} < {DRIFT_COVERAGE_MIN}"

    # single-cycle rule trip. The scheduler turns this into a persistent
    # consecutive-cycle streak (state.json "drift_streak"), because a fresh
    # summary is rebuilt every verify and cannot know about prior cycles by
    # itself. tripped==True here means "the live 30d window violates the rule".
    base["drift"] = {
        "tripped": trip_reason is not None,
        "reason": trip_reason,
        "windows": 0 if trip_reason is None else DRIFT_CONSECUTIVE_CYCLES,
        "required": DRIFT_CONSECUTIVE_CYCLES,
    }
    return base


def _load_backtest_baseline() -> float | None:
    """Reference 30d RMSE for the current prod model (for drift comparison).

    Reads the last promoted model's recorded 30d RMSE from meta/state when
    available; falls back to the curated backtest metrics file, else None
    (meaning drift is judged on coverage only).
    """
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


def drift_status(state: dict | None) -> dict:
    """Direct scheduler-facing drift status (no rebuild). Reads the summary."""
    if not VERIF_SUMMARY.exists():
        return {"tripped": False, "reason": None, "updated_ts": None}
    try:
        return json.loads(VERIF_SUMMARY.read_text()).get("drift", {}) or \
            {"tripped": False, "reason": None, "updated_ts": None}
    except Exception:  # noqa: BLE001
        return {"tripped": False, "reason": None, "updated_ts": None}