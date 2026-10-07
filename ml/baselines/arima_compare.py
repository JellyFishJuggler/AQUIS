"""Compare a daily SARIMA baseline with the frozen pooled trajectory XGBoost.

Uses only 2026 test-period rolling origins. No model artifacts are written.
Requires statsmodels (SARIMAX); no dependency files are changed here.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

_FEAT_MOD = None


def _features_mod():
    """Load 06_features.py by path (leading-digit filename is not importable)."""
    global _FEAT_MOD
    if _FEAT_MOD is None:
        spec = importlib.util.spec_from_file_location(
            "arima_baseline_features", ROOT / "06_features.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # cache only after success: a failed exec
        _FEAT_MOD = mod              # must not leave a partial module behind
    return _FEAT_MOD

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
# 06_features.py does "from _soil import ..." / "from _lulc import ...": ensure
# the ml/ root is importable no matter where this script is launched from.
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "aligned" / "table_6h.parquet"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2027-01-01")
HORIZONS = (1, 7, 15, 30)
N_ORIGINS = 5
MAX_SHORT_GAP = 3
# Minimum clean history (no gap > MAX_SHORT_GAP days) immediately before an
# origin. 180 daily obs is ample for SARIMA(1,1,1)x(1,0,1,7) (6 params); the
# archive never yields 365 clean days before a 2026 origin (best ≈ 280).
MIN_HISTORY = 180


def select_stations(daily: pd.DataFrame, valid: dict[str, list]) -> list[str]:
    """Pick 10 stations with >= N_ORIGINS valid origins, spread low→high variance.

    ``valid`` maps station -> list of qualifying 2026 origins (same filters as
    the evaluation loop). Among qualifiers, stations are ranked by pre-test
    daily variance and 10 are taken at evenly spaced ranks so the comparison
    covers calm and volatile regimes instead of only the sparse extremes.
    """
    pre = daily[daily.date < TEST_START]
    var = pre.groupby("Station").gwl.var().dropna()
    qual = sorted([s for s, orgs in valid.items() if len(orgs) >= N_ORIGINS and s in var.index],
                  key=lambda s: var[s])
    print(f"Selection logic: stations with >={N_ORIGINS} valid origins: {len(qual)}; "
          f"10 taken evenly spaced by pre-test daily variance (low→high). "
          f"Validity already excludes origins the frozen XGB pipeline refuses "
          f"(telemetry regime shift, same rule as 06_features.flag_regime_shift).")
    if len(qual) < 10:
        raise SystemExit(f"Only {len(qual)} stations have >={N_ORIGINS} valid origins; need 10.")
    idx = np.linspace(0, len(qual) - 1, 10, dtype=int)
    picked = [qual[i] for i in idx]
    for s in picked:
        print(f"  {s}: var={var[s]:.4f}, valid_origins={len(valid[s])}")
    return picked


def yearly_median_map(raw_all: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Per-station {year: median gwl} over 6h readings (mirrors flag_regime_shift)."""
    df = raw_all.dropna(subset=["gwl"]).copy()
    df["year"] = df["time"].dt.year
    med = df.groupby(["Station", "year"])["gwl"].median()
    out: dict[str, dict[int, float]] = {}
    for (st, yr), v in med.items():
        out.setdefault(str(st), {})[int(yr)] = float(v)
    return out


def xgb_feasible(ymaps: dict[str, dict[int, float]], station: str,
                 origin: pd.Timestamp, max_delta: float = 15.0) -> bool:
    """True when build_full(history <= origin) would keep the station, i.e. no
    yearly median (2021..origin-year, 2026 partial to the origin) deviates more
    than max_delta from the station's 2021-2024 baseline. Same rule as
    06_features.flag_regime_shift; telemetry datum jumps make XGB refuse the
    station in production too, so such origins cannot be matched.
    """
    ym = ymaps.get(station, {})
    base_yrs = [y for y in range(2021, 2025) if y in ym]
    if not base_yrs:
        return True
    base = float(np.median([ym[y] for y in base_yrs]))
    oy = pd.Timestamp(origin).year
    for y in range(2021, oy + 1):
        if y in ym and abs(ym[y] - base) > max_delta:
            return False
    return True


def clean_history(hist: pd.Series) -> pd.Series | None:
    """History usable for fitting: walk back from the origin, allowing gaps up
    to MAX_SHORT_GAP days (interpolated); stop at the first longer outage.

    Returns the interpolated segment, or None when it holds fewer than
    MIN_HISTORY observations.
    """
    vals: list = []
    gap = 0
    for v in hist.iloc[::-1]:
        if pd.notna(v):
            vals.append(v)
            gap = 0
        else:
            gap += 1
            if gap > MAX_SHORT_GAP:
                break
            vals.append(np.nan)
    # The walk stops inside the long outage that bounds the segment; drop its
    # partial gap days so the segment starts at a real observation (leading
    # NaNs would otherwise survive limit_area="inside" and void the segment).
    while vals and pd.isna(vals[-1]):
        vals.pop()
    seg = pd.Series(vals[::-1]).interpolate(method="linear", limit=MAX_SHORT_GAP,
                                            limit_area="inside")
    if seg.isna().any() or seg.notna().sum() < MIN_HISTORY:
        return None
    return seg


def sarima_forecast(hist: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    filled = clean_history(hist)
    # None means a long outage sits inside the trailing history, or the clean
    # run immediately before the origin is shorter than MIN_HISTORY days.
    if filled is None:
        return None
    # Weekly seasonality is tractable on daily data and avoids a 365-state model.
    fit = SARIMAX(filled, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                  trend="n", enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False, maxiter=100)
    fc = fit.get_forecast(steps=30)
    ci = np.asarray(fc.conf_int(alpha=0.10), dtype=float)
    return np.asarray(fc.predicted_mean, dtype=float), ci[:, 0], ci[:, 1]


def load_xgb():
    import xgboost as xgb
    cfg = json.loads((MODELS / "traj_config.json").read_text())
    models = {q: xgb.Booster(model_file=str(MODELS / f"traj_xgb_{q}.json"))
              for q in ("q05", "q50", "q95")}
    meta = json.loads((MODELS / "feature_config.json").read_text())
    return cfg["feature_cols"], models, sorted(meta["stations"]), sorted(meta["districts"])


def xgb_forecast(history: pd.DataFrame, station: str, cols: list[str], models: dict,
                 station_order: list[str], district_order: list[str]):
    F = _features_mod()
    feat, _ = F.build_full(history.copy(), keep_na=True, horizon_steps=120, horizon_label=30)
    if feat.empty:
        return None
    last = feat.sort_values("time").iloc[-1:].copy()
    district = str(history.District.iloc[0]).upper()
    last["st_id"] = station_order.index(station) if station in station_order else -1
    last["dist_id"] = district_order.index(district) if district in district_order else -1
    base_cols = [c for c in cols if c not in ("h", "h_sin", "h_cos")]
    base = np.asarray(last[base_cols], dtype=np.float32)[0]
    hs = np.arange(1, 121, dtype=np.float32)
    X = np.column_stack((np.repeat(base[None, :], 120, axis=0), hs,
                         np.sin(2 * np.pi * hs / 120), np.cos(2 * np.pi * hs / 120)))
    dm = xgb.DMatrix(X, feature_names=cols)
    anchor = float(last.gwl.iloc[0])
    q05, q50, q95 = [models[q].predict(dm) + anchor for q in ("q05", "q50", "q95")]
    # Convert the four 6-hour model steps per date to the same daily-mean basis
    # used for observations and SARIMAX. Average each quantile path over 4 steps.
    q05 = q05.reshape(30, 4).mean(axis=1)
    q50 = q50.reshape(30, 4).mean(axis=1)
    q95 = q95.reshape(30, 4).mean(axis=1)
    return np.minimum(q05, q50), q50, np.maximum(q95, q50)


def metrics(y, pred, lo, hi):
    e = np.asarray(pred) - np.asarray(y)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "picp90": float(np.mean((np.asarray(y) >= lo) & (np.asarray(y) <= hi))),
            "mean_interval_width": float(np.mean(np.asarray(hi) - np.asarray(lo)))}


def main():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
    except ImportError as exc:
        raise SystemExit("statsmodels is required; it is not installed. Install it in the active environment; requirements were not changed.") from exc
    raw_all = pd.read_parquet(DATA)
    raw_all["time"] = pd.to_datetime(raw_all.time)
    raw_all = raw_all.sort_values(["Station", "time"])
    # Keep the full 6-hour axis for XGBoost feature generation; use observed GWL
    # readings only for daily targets and station statistics.
    raw = raw_all.dropna(subset=["gwl"])[["Station", "District", "time", "gwl"]]
    # Daily mean aggregation; exact daily index exposes missing days as gaps.
    daily_parts = []
    for station, g in raw.groupby("Station", sort=False):
        d = g.set_index("time").gwl.resample("D").mean().to_frame("gwl")
        d["Station"] = station
        d["date"] = d.index
        daily_parts.append(d.reset_index(drop=True))
    daily = pd.concat(daily_parts, ignore_index=True)
    daily_by_station = {s: g.set_index("date").gwl.sort_index()
                        for s, g in daily.groupby("Station", sort=False)}
    # Candidate 2026 origins with 30 complete future days in the archive.
    available_end = raw.time.max().normalize() - pd.Timedelta(days=30)
    all_days = pd.date_range(TEST_START, min(TEST_END - pd.Timedelta(days=31), available_end), freq="D")
    if len(all_days) < N_ORIGINS:
        raise SystemExit("Not enough complete 30-day targets inside the 2026 test period.")
    # One validity pass per station; selection and evaluation share it so the
    # >=N_ORIGINS guarantee holds for every reported station. Origins where the
    # frozen XGB pipeline would drop the station (telemetry regime shift) are
    # excluded up front — they can never yield a matched pair.
    ymaps = yearly_median_map(raw_all)
    valid: dict[str, list] = {}
    for station, sd in daily_by_station.items():
        orgs = []
        for origin in all_days:
            target_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=30, freq="D")
            if sd.reindex(target_dates).isna().any():
                continue
            if origin not in sd.index or pd.isna(sd.get(origin)):
                continue
            if clean_history(sd.loc[:origin]) is None:
                continue
            if not xgb_feasible(ymaps, station, origin):
                continue
            orgs.append(origin)
        valid[station] = orgs
    stations = select_stations(daily, valid)
    cols, models, station_order, district_order = load_xgb()
    records = []
    skipped: dict[str, int] = {}
    for station in stations:
        t0 = time.perf_counter()
        sd = daily_by_station[station]
        valid_origins = valid[station]
        assert len(valid_origins) >= N_ORIGINS, station
        station_origins = np.asarray(valid_origins)[np.linspace(0, len(valid_origins) - 1,
                                                                N_ORIGINS, dtype=int)]
        for origin in station_origins:
            target_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=30, freq="D")
            actual = sd.reindex(target_dates)
            hist_daily = sd.loc[:origin]
            try:
                afc = sarima_forecast(hist_daily)
                if afc is None:
                    skipped[station] = skipped.get(station, 0) + 1
                    print(f"  origin {origin.date()} skipped: clean history < {MIN_HISTORY}d")
                    continue
                raw_st = raw_all[(raw_all.Station == station) &
                                 (raw_all.time <= origin + pd.Timedelta(hours=23, minutes=59))]
                if len(raw_st) < 120:
                    skipped[station] = skipped.get(station, 0) + 1
                    print(f"  origin {origin.date()} skipped: <120 6h obs for XGB features")
                    continue
                xf = xgb_forecast(raw_st, station, cols, models, station_order, district_order)
                if xf is None:
                    skipped[station] = skipped.get(station, 0) + 1
                    print(f"  origin {origin.date()} skipped: empty XGB feature frame")
                    continue
                a50, a05, a95 = afc
                x05, x50, x95 = xf
                y = actual.to_numpy(dtype=float)
                for i in range(30):
                    records.append({"station": station, "origin": origin, "day": i + 1,
                                    "actual": y[i], "arima": a50[i], "arima_lo90": a05[i],
                                    "arima_hi90": a95[i], "xgb": x50[i],
                                    "xgb_lo90": x05[i], "xgb_hi90": x95[i]})
            except Exception as exc:  # preserve per-station runtime and continue
                skipped[station] = skipped.get(station, 0) + 1
                print(f"  origin {origin.date()} skipped: {type(exc).__name__}: {exc}")
        print(f"station runtime {station}: {time.perf_counter() - t0:.1f}s")
    if skipped:
        print("Skipped origins per station (gaps/errors): " +
              ", ".join(f"{s}: {n}" for s, n in sorted(skipped.items())))
    else:
        print("Skipped origins per station: none")
    pred = pd.DataFrame(records)
    if pred.empty:
        raise SystemExit("No comparable forecast windows survived gap/coverage checks.")
    rows = []
    groups = [(s, g) for s, g in pred.groupby("station")] + [("OVERALL", pred)]
    for station, g in groups:
        for model in ("arima", "xgb"):
            m = metrics(g.actual, g[model], g[f"{model}_lo90"], g[f"{model}_hi90"])
            rows.append({"station": station, "scope": "all_days", "model": model,
                         "n": len(g), **m})
            for day in HORIZONS:
                gd = g[g.day == day]
                if len(gd):
                    rows.append({"station": station, "scope": f"day_{day}", "model": model,
                                 "n": len(gd), **metrics(gd.actual, gd[model],
                                    gd[f"{model}_lo90"], gd[f"{model}_hi90"])})
    result = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / "arima_vs_xgb.csv", index=False)
    # Persist the per-(station, origin, day) predictions so a COMBINATION can be
    # fitted later without re-running SARIMAX. `arima_xgb_combine.py` consumes this;
    # from the aggregate CSV alone the ensemble weights cannot be derived.
    pred_path = OUT / "arima_vs_xgb_predictions.csv"
    pred.to_csv(pred_path, index=False)
    print(f"Wrote per-origin predictions -> {pred_path} ({len(pred):,} rows)")
    overall = result[(result.station == "OVERALL") & (result.scope == "all_days")].set_index("model")
    origin_counts = pred.groupby("station").origin.nunique()
    with (OUT / "arima_vs_xgb.md").open("w") as f:
        f.write("# Daily SARIMA vs pooled trajectory XGBoost\n\n")
        f.write(f"2026 rolling origins: {origin_counts.min()}–{origin_counts.max()} per station; stations: {pred.station.nunique()}.\n\n")
        f.write("| Model | MAE | RMSE | PICP (90%) | Mean interval width |\n|---|---:|---:|---:|---:|\n")
        for model in ("arima", "xgb"):
            m = overall.loc[model]
            f.write(f"| {model.upper()} | {m.mae:.3f} | {m.rmse:.3f} | {m.picp90:.3f} | {m.mean_interval_width:.3f} |\n")
        f.write("\nARIMA choice: SARIMAX(1,1,1)×(1,0,1,7), because weekly seasonality is practical on daily data.\n")
        f.write("Short gaps up to 3 days were interpolated; origins with longer history gaps or missing future labels were skipped.\n")
        f.write("Stations the frozen XGB pipeline refuses (yearly-median regime shift > 15 m, "
                "telemetry datum error) were excluded from selection — no matched pair can exist there.\n")
        f.write(f"ARIMA history rule: >= {MIN_HISTORY} clean days (no gap > 3d) immediately before each origin; "
                "the 365-day rule was unusable because no station has 365 clean days before a 2026 origin.\n")
        f.write("\n")
        f.write(f"- MAE winner: {overall.mae.idxmin()} ({overall.mae.min():.3f} m).\n")
        f.write(f"- RMSE winner: {overall.rmse.idxmin()} ({overall.rmse.min():.3f} m).\n")
        f.write(f"- Coverage/width: ARIMA {overall.loc['arima','picp90']:.1%}/{overall.loc['arima','mean_interval_width']:.3f} m; XGB {overall.loc['xgb','picp90']:.1%}/{overall.loc['xgb','mean_interval_width']:.3f} m; likely reflects pooled driver/trajectory information versus a single-station weekly seasonal model.\n")
    print(f"Wrote {OUT / 'arima_vs_xgb.csv'} and {OUT / 'arima_vs_xgb.md'}")
    print(result[result.station == "OVERALL"].to_string(index=False))


if __name__ == "__main__":
    main()
