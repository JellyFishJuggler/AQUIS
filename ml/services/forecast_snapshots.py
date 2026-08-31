"""Precomputed dashboard forecast snapshot (CSV) + fallback readers.

The Streamlit dashboard renders two model-driven forecast panels:

    "Forecast — Test Period"          actuals vs predictions on the held-out
                                      test window (``get_test_predictions``)
    "Forecast — Next 2–3 Months"      continuous recursive projection from
                                      the last stored reading -> today -> +90d

On Streamlit Community Cloud the per-station XGBoost weights are frequently
unavailable (the ``.joblib`` files travel through Git LFS and Cloud's smudge
step is historically unreliable), which silently degrades both panels to
empty states.  This module exports every station's model-generated curves
into ONE small committed CSV (``artifacts/dashboard_forecasts.csv``, ~5-10 MB
at 4-decimal rounding) and lets the dashboard rebuild the exact panel dicts
from it as a fallback.

Local runs (weights present) keep the live prediction path untouched; the
snapshot is only used when loading the models fails.  The observed series /
tail stay parquet-sourced (the parquet is committed), so only the model-
generated curves live in the snapshot.

Exports:
    write_dashboard_forecasts()   train-model curves -> CSV
    verify_snapshot()             CSV-vs-live equality check
    load_snapshot_df()            read the CSV
    test_preds_from_df()          rebuild the LEFT-panel dict
    future_preds_from_df()        rebuild the RIGHT-panel dict
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_PARQUET,
    POINT_MODEL_FILE,
    _load_config,
    get_station_series,
    get_test_predictions,
    predict_xgb_quantile,
)
from ml.preprocessing.timeseries import GWL_COL, TIME_COL  # noqa: E402

SNAPSHOT_FILE = "dashboard_forecasts.csv"
_ROUND = 4
_TOL = 5e-4

SNAPSHOT_COLUMNS = [
    "station",
    "slug",
    "split",
    "date",
    "actual",
    "point",
    "lower",
    "upper",
    "stored_end",
    "projection_start",
    "today",
    "projection_end",
    "last_obs_value",
    "found_2026",
    "feed_ts",
]


def _round4(value) -> float:
    return round(float(value), _ROUND)


def station_dirs(artifacts_root: Path | None = None) -> list[Path]:
    """Trained station directories (have a point model + metadata)."""
    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    return sorted(
        p
        for p in root.iterdir()
        if (p / POINT_MODEL_FILE).is_file() and (p / "xgboost_metadata.json").is_file()
    )


def _presence_facts(pres_df: pd.DataFrame | None, station: str) -> tuple[bool, pd.Timestamp | None]:
    """2026-feed facts for a station, mirroring the dashboard's helper."""
    if pres_df is None or not len(pres_df):
        return False, None
    rows = pres_df[pres_df["station"] == station]
    if not len(rows):
        return False, None
    r = rows.iloc[0]
    if pd.notna(r.get("found_2026")) and bool(r["found_2026"]):
        ts = pd.to_datetime(r.get("max_ts"), errors="coerce")
        return True, (ts if pd.notna(ts) else None)
    return False, None


def _future_from_models(
    cfg: dict,
    slug: str,
    parquet_path: Path | str = DEFAULT_PARQUET,
    today: pd.Timestamp | None = None,
    future_days: int = 90,
) -> dict | None:
    """Recursive projection dict — mirror of the dashboard's live path."""
    df = get_station_series(parquet_path, cfg.get("station", slug))
    if not len(df):
        return None
    ts = df[TIME_COL]
    first = ts.min()
    stored_end = pd.Timestamp(ts.max())
    start = stored_end.normalize()
    today = train_today = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    end = train_today.normalize() + pd.Timedelta(days=future_days)
    future_dates = pd.date_range(start, end, freq="D")
    if not len(future_dates):
        return None

    base_hour = float((start - first).total_seconds() / 3600)
    future_hours = base_hour + np.arange(len(future_dates), dtype=float) * 24.0
    res = predict_xgb_quantile(future_hours, slug)
    return {
        "stored_end": stored_end,
        "projection_start": start,
        "today": train_today.normalize(),
        "projection_end": end,
        "future_dates": future_dates,
        "last_obs_value": float(df[GWL_COL].iloc[-1]),
        "point": np.asarray(res["point"]),
        "lower": np.asarray(res["lower"]),
        "upper": np.asarray(res["upper"]),
    }


def write_dashboard_forecasts(
    csv_path: Path | str,
    artifacts_root: Path | str | None = None,
    stations: list[str] | None = None,
    future_days: int = 90,
) -> Path:
    """Export trained-model curves for every station into the snapshot CSV.

    ``stations`` filters by station-directory slug; ``None`` = all.  Returns
    the CSV path.
    """
    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    csv_path = Path(csv_path)
    pres_path = root / "2026_station_presence.csv"
    pres = pd.read_csv(pres_path) if pres_path.is_file() else None
    today = pd.Timestamp.now()

    records: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for out_dir in station_dirs(root):
        slug = out_dir.name
        if stations is not None and slug not in stations:
            continue
        cfg = _load_config(out_dir)
        name = cfg.get("station", slug)
        found, feed_ts = _presence_facts(pres, name)

        try:
            tp = get_test_predictions(slug)
        except Exception as exc:  # noqa: BLE001
            skipped.append((slug, f"test: {exc!r}"))
            continue
        times = pd.to_datetime(np.asarray(tp["time"]))
        for i in range(len(times)):
            records.append(
                {
                    "station": name,
                    "slug": slug,
                    "split": "test",
                    "date": times[i],
                    "actual": _round4(tp["actual"][i]),
                    "point": _round4(tp["point"][i]),
                    "lower": _round4(tp["lower"][i]),
                    "upper": _round4(tp["upper"][i]),
                    "stored_end": "",
                    "projection_start": "",
                    "today": "",
                    "projection_end": "",
                    "last_obs_value": "",
                    "found_2026": "",
                    "feed_ts": "",
                }
            )

        ff = _future_from_models(cfg, slug, DEFAULT_PARQUET, today, future_days=future_days)
        if ff is None:
            skipped.append((slug, "future: no data"))
            continue
        for i, dt in enumerate(ff["future_dates"]):
            records.append(
                {
                    "station": name,
                    "slug": slug,
                    "split": "future",
                    "date": dt,
                    "actual": "",
                    "point": _round4(ff["point"][i]),
                    "lower": _round4(ff["lower"][i]),
                    "upper": _round4(ff["upper"][i]),
                    "stored_end": ff["stored_end"].isoformat(),
                    "projection_start": ff["projection_start"].isoformat(),
                    "today": ff["today"].isoformat(),
                    "projection_end": ff["projection_end"].isoformat(),
                    "last_obs_value": _round4(ff["last_obs_value"]),
                    "found_2026": int(found),
                    "feed_ts": feed_ts.isoformat() if feed_ts is not None else "",
                }
            )

    if not records:
        raise SystemExit(f"No XGBoost models found under {root}.")

    df = pd.DataFrame(records, columns=SNAPSHOT_COLUMNS)
    df.to_csv(csv_path, index=False)
    size_mb = csv_path.stat().st_size / 1024 / 1024
    stations_done = df["slug"].nunique()
    print(
        f"Exported {stations_done} stations -> {csv_path} ({size_mb:.1f} MB, "
        f"{len(df):,} rows)"
    )
    if skipped:
        print("Skipped stations:")
        for slug, reason in skipped:
            print(f"  {slug}: {reason}")
    return csv_path


def load_snapshot_df(csv_path: Path | str) -> pd.DataFrame | None:
    """Read the snapshot CSV, or None when it does not exist."""
    path = Path(csv_path)
    if not path.is_file():
        return None
    return pd.read_csv(path, dtype={"found_2026": "Int64", "last_obs_value": "Float64"})


def test_preds_from_df(df: pd.DataFrame | None, station: str) -> dict | None:
    """Rebuild the LEFT-panel dict (``get_test_predictions`` shape)."""
    if df is None or not len(df):
        return None
    d = df[(df["station"] == station) & (df["split"] == "test")]
    if not len(d):
        return None
    d = d.sort_values("date")
    return {
        "time": pd.Series(pd.to_datetime(d["date"].to_numpy())),
        "actual": d["actual"].astype(float).to_numpy(),
        "point": d["point"].astype(float).to_numpy(),
        "lower": d["lower"].astype(float).to_numpy(),
        "upper": d["upper"].astype(float).to_numpy(),
    }


def future_preds_from_df(df: pd.DataFrame | None, station: str) -> dict | None:
    """Rebuild the RIGHT-panel dict (``_future_forecast`` shape, minus tail)."""
    if df is None or not len(df):
        return None
    d = df[(df["station"] == station) & (df["split"] == "future")]
    if not len(d):
        return None
    d = d.sort_values("date")
    r0 = d.iloc[0]
    feed = pd.to_datetime(r0["feed_ts"], errors="coerce")
    if pd.isna(feed):
        feed = None
    return {
        "stored_end": pd.Timestamp(r0["stored_end"]),
        "projection_start": pd.Timestamp(r0["projection_start"]),
        "today": pd.Timestamp(r0["today"]),
        "projection_end": pd.Timestamp(r0["projection_end"]),
        "future_dates": pd.DatetimeIndex(pd.to_datetime(d["date"].to_numpy())),
        "last_obs_value": float(r0["last_obs_value"]),
        "found_2026": bool(pd.notna(r0["found_2026"]) and r0["found_2026"] == 1),
        "feed_ts": feed,
        "point": d["point"].astype(float).to_numpy(),
        "lower": d["lower"].astype(float).to_numpy(),
        "upper": d["upper"].astype(float).to_numpy(),
    }


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if len(a) == len(b) else float("inf")


def verify_snapshot(
    csv_path: Path | str,
    artifacts_root: Path | str | None = None,
    stations: list[str] | None = None,
) -> list[str]:
    """CSV-vs-live equality within rounding tolerance; returns failure messages.

    The live future projection is recomputed with the CSV's frozen ``today``
    (the snapshot is a point-in-time export), so day-boundary drift cannot
    produce false mismatches.
    """
    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    df = load_snapshot_df(csv_path)
    failures: list[str] = []
    checked = 0

    for out_dir in station_dirs(root):
        slug = out_dir.name
        if stations is not None and slug not in stations:
            continue
        cfg = _load_config(out_dir)
        name = cfg.get("station", slug)

        ff_csv = future_preds_from_df(df, name)
        tp_csv = test_preds_from_df(df, name)
        if tp_csv is None or ff_csv is None:
            failures.append(f"{slug}: snapshot rows missing")
            continue
        snap_today = ff_csv["today"]

        try:
            tp = get_test_predictions(slug)
            ff = _future_from_models(cfg, slug, DEFAULT_PARQUET, snap_today)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{slug}: live compute failed: {exc!r}")
            continue

        for key in ("point", "lower", "upper"):
            for label, live, snap in (
                ("test actual", tp["actual"], tp_csv["actual"]),
                (f"test {key}", tp[key], tp_csv[key]),
                (f"future {key}", ff[key], ff_csv[key]),
            ):
                diff = _max_abs_diff(live, snap)
                if diff > _TOL:
                    failures.append(f"{slug}: {label} mismatch {diff:.3g}")
        if not len(ff["future_dates"]) == len(ff_csv["future_dates"]):
            failures.append(f"{slug}: future date count differs")
        if abs(ff["last_obs_value"] - ff_csv["last_obs_value"]) > _TOL:
            failures.append(f"{slug}: last_obs_value differs")
        checked += 1

    print(f"Verified {checked} stations against {csv_path}; {len(failures)} mismatches")
    for msg in failures:
        print(f"  FAIL {msg}")
    return failures