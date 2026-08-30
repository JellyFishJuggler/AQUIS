"""Decision-support layer: turn trained XGBoost forecasts into actionable priorities.

For every station with trained artifacts:

- **Trend** — linear fit on the filtered series (last ~2 years) -> m/yr, plus a
  direction label.  Sign convention used throughout: *more-negative GWL means
  deeper*, so a negative slope = declining (falling) water table.
- **Thresholds** — derived purely from the station's own history:
  ``critical`` = level reached by only the deepest 10% of past readings,
  ``caution`` = deepest 30%.  No external/manual configuration needed.
- **Projection** — recursive (multi-step) XGBoost forecast at +90 and +180 days
  with the 90% PI (q05..q95) — i.e. the honest long-horizon mode.
- **Priority** — a 2x2 decision grid (~pitch: decision-support, not a badge):
    declining + normal zone        -> MONITOR
    declining + caution/critical   -> PRIORITY
    stable/rising + caution zone   -> MONITOR
    stable/rising + normal zone    -> OK
    (escalation: projected into critical zone in 180d -> PRIORITY regardless)
- **Narrative** — plain-language strategy line with the numbers, plus an
  honesty note when the station's recursive 90% coverage is weak (read from
  ``stepwise_comparison.csv`` if present).

Writes ``artifacts/decision_support.csv``.

Usage (from repo root):
    python -m ml.scripts.decision_support
    python -m ml.scripts.decision_support --station "Asafpur"
    python -m ml.scripts.decision_support --limit 5
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_PARQUET,
    predict_xgb_quantile,
)
from ml.preprocessing.timeseries import GWL_COL, TIME_COL, get_station_series  # noqa: E402

OUT_CSV = _ML_ROOT / "artifacts" / "decision_support.csv"
STEPWISE_CSV = _ML_ROOT / "artifacts" / "stepwise_comparison.csv"

DEEPEST_FRAC_CRITICAL = 0.10  # level reached by only this fraction of history
DEEPEST_FRAC_CAUTION = 0.30
TREND_WINDOW_YEARS = 2.0  # recent window used for the direction estimate
HORIZON_DAYS = [90, 180]
MAX_MONTH_STEP_M = 4.0  # month-to-month median jump above this => sensor block
TREND_DECLINE_MYR = -0.05  # m/yr slower than this => "declining"
WEAK_COVERAGE = 0.60  # recursive 90% coverage below this => honesty note


def _deep_level(series: np.ndarray, frac: float) -> float:
    """Value below which only ``frac`` of history lies (the deep end)."""
    if len(series) < 20:
        return float("nan")
    return float(np.quantile(series, frac))


def compute_trend(sdf: pd.DataFrame) -> tuple[float, str]:
    """Robust slope (m/yr) over the recent window; negative => deeper/declining.

    Theil-Sen (Sen's slope) is used rather than OLS: it is outlier-robust,
    which matters because some stations still carry sensor spikes that survive
    the magnitude filter.  Negative => declining water table.  A guard is
    applied: if month-to-month medians jump by more than ``MAX_MONTH_STEP_M``
    (a stuck/block sensor), the trend is reported as indeterminate.
    """
    t = pd.to_datetime(sdf[TIME_COL].values)
    y = sdf[GWL_COL].values.astype(float)
    window = TREND_WINDOW_YEARS * 365.25 * 24
    t_hours = (t - t.min()).total_seconds() / 3600
    recent = t_hours >= (t_hours.max() - window)
    xr = t_hours[recent]
    yr = y[recent]
    if len(xr) < 60:
        return float("nan"), "insufficient"

    months = pd.Series(t[recent]).dt.to_period("M").astype(str).to_numpy()
    med_by_month = pd.Series(yr).groupby(months).median().sort_index()
    if len(med_by_month) >= 2:
        steps = np.abs(np.diff(med_by_month.to_numpy()))
        if steps.max() > MAX_MONTH_STEP_M:
            return float("nan"), "indeterminate"

    slope, _, _, _ = theilslopes(yr, xr)
    slope_m_yr = float(slope) * (365.25 * 24)  # per hour -> per year
    if slope_m_yr < TREND_DECLINE_MYR:
        direction = "declining"
    elif slope_m_yr > -TREND_DECLINE_MYR:
        direction = "rising"
    else:
        direction = "stable"
    return slope_m_yr, direction


def classify_priority(
    level_now: float,
    proj_180d: float | None,
    critical: float,
    caution: float,
    trend_m_yr: float,
    trend_indeterminate: bool = False,
) -> str:
    if np.isnan(level_now) or np.isnan(critical):
        return "INSUFFICIENT"
    in_critical_now = level_now <= critical
    in_caution_now = level_now <= caution
    proj_critical = (proj_180d is not None) and (proj_180d <= critical)
    declining = (not trend_indeterminate) and trend_m_yr < TREND_DECLINE_MYR
    rising = (not trend_indeterminate) and trend_m_yr > -TREND_DECLINE_MYR

    if in_critical_now or proj_critical:
        return "PRIORITY"
    if in_caution_now:
        return "PRIORITY" if declining else "MONITOR"
    # above caution zone
    return "MONITOR" if declining else ("OK" if rising else "OK")


def build_narrative(
    row: dict, multi_cov: float | None, w_critical: float, w_caution: float
) -> str:
    p90 = row["proj_90d"]
    p180 = row["proj_180d"]
    trend_txt = (
        f"{row['trend_direction']} {abs(row['trend_m_yr']):.2f} m/yr"
        if row.get("trend_m_yr") is not None
        else "trend indeterminate (sensor-blocks)"
    )
    parts = [f"{row['station']}: level {row['level_now']:.2f} m, {trend_txt}."]
    if row["trend_direction"] == "indeterminate":
        parts.append(
            "Recent record shows a sensor step-block; direction not trustworthy."
        )
    if not (p90 is None and p180 is None):
        a = [f"+90d {p90:.2f} m" if p90 is not None else None,
             f"+180d {p180:.2f} m" if p180 is not None else None]
        a = [x for x in a if x]
        parts.append("projected " + ", ".join(a) + ".")
    parts.append(f"caution {w_caution:.2f} m, critical {w_critical:.2f} m.")
    action = {
        "PRIORITY": "-> PRIORITY: investigate recharge status / step up monitoring.",
        "MONITOR": "-> MONITOR: watch trend, reassess quarterly.",
        "OK": "-> OK: no immediate action.",
        "INSUFFICIENT": "-> insufficient data for a recommendation.",
    }[row["priority"]]
    parts.append(action)
    if multi_cov is not None and multi_cov < WEAK_COVERAGE:
        parts.append(
            f"(long-horizon forecast weak: recursive 90% coverage "
            f"{multi_cov:.0%} < 60%)"
        )
    return " ".join(parts)


def decision_for_station(
    station: str,
    stepwise_cov: dict[str, float] | None,
) -> dict:
    sdf = get_station_series(DEFAULT_PARQUET, station)
    if len(sdf) < 60:
        return {
            "station": station, "priority": "INSUFFICIENT",
            "narrative": f"{station}: insufficient history. "
            "-> insufficient data for a recommendation.",
        }

    y = sdf[GWL_COL].values.astype(float)
    level_now = float(y[-1])
    critical = _deep_level(y, DEEPEST_FRAC_CRITICAL)
    caution = _deep_level(y, DEEPEST_FRAC_CAUTION)
    trend_m_yr, direction = compute_trend(sdf)

    base_hours = float(sdf["time_hours"].iloc[-1])
    horizons = [h * 24 for h in HORIZON_DAYS]
    try:
        res = predict_xgb_quantile(base_hours + np.array(horizons, dtype=float), station)
    except FileNotFoundError:
        return {
            "station": station, "priority": "INSUFFICIENT",
            "narrative": f"{station}: no trained model artifact in {ARTIFACTS_DIR}.",
        }

    proj = {}
    for days, point, lo, hi in zip(
        HORIZON_DAYS, res["point"], res["lower"], res["upper"]
    ):
        proj[f"proj_{days}d"] = float(point)
        proj[f"lo_{days}d"] = float(lo)
        proj[f"hi_{days}d"] = float(hi)

    trend_indeterminate = direction == "indeterminate"
    priority = classify_priority(
        level_now, proj.get("proj_180d"), critical, caution, trend_m_yr,
        trend_indeterminate,
    )
    multi_cov = stepwise_cov.get(station) if stepwise_cov else None

    row = {
        "station": station,
        "trend_direction": direction,
        "trend_m_yr": round(trend_m_yr, 3) if not np.isnan(trend_m_yr) else None,
        "level_now": round(level_now, 3),
        **{k: round(v, 3) for k, v in proj.items()},
        "caution": round(caution, 3) if not np.isnan(caution) else None,
        "critical": round(critical, 3) if not np.isnan(critical) else None,
        "multi_cov_90": round(multi_cov, 3) if multi_cov is not None else None,
        "priority": priority,
    }
    row["narrative"] = build_narrative(row, multi_cov, critical, caution)
    return row


def load_stepwise_coverage() -> dict[str, float] | None:
    if not STEPWISE_CSV.is_file():
        return None
    df = pd.read_csv(STEPWISE_CSV)
    if "multi_coverage_90" not in df or "station" not in df:
        return None
    out = {}
    for _, r in df.iterrows():
        if not pd.isna(r.get("multi_coverage_90")):
            out[r["station"]] = float(r["multi_coverage_90"])
    return out or None


CSV_FIELDS = [
    "station", "trend_direction", "trend_m_yr", "level_now",
    "proj_90d", "lo_90d", "hi_90d",
    "proj_180d", "lo_180d", "hi_180d",
    "caution", "critical", "multi_cov_90", "priority", "narrative",
]


def write_csv(rows: list[dict], out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if k not in r or r[k] is None else r[k]) for k in CSV_FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args()

    df = pd.read_parquet(DEFAULT_PARQUET)
    stations = sorted(df["Station"].unique().tolist())
    if args.station:
        stations = [s for s in stations if args.station.lower() in s.lower()]
    if args.limit:
        stations = stations[: args.limit]
    print(f"decision support for {len(stations)} stations...")

    cov = load_stepwise_coverage()
    rows = []
    for i, s in enumerate(stations, 1):
        r = decision_for_station(s, cov)
        row = {k: str(r.get(k, "") if r.get(k) is not None else "") for k in CSV_FIELDS}
        rows.append(row)
        print(f"  [{i}/{len(stations)}] {s:.<55} {row['priority']}")
        if i % 10 == 0:
            write_csv(rows, args.out)

    write_csv(rows, args.out)
    print(f"\nwrote {args.out}")

    by_p = pd.Series([r["priority"] for r in rows]).value_counts()
    if len(by_p):
        print("priority summary:\n" + by_p.to_string())


if __name__ == "__main__":
    main()