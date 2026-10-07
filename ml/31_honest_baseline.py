"""31_honest_baseline — out-of-sample baseline read from EXISTING artifacts.

Answers two questions that the current `outputs/` cannot answer as written:

1. **Does the model's margin over persistence survive a fair split?**
   `predictions_2026.parquet` carries every test row, so the RMSE can be
   recomputed on a held-out half of the stations. A margin that exists only when
   the same stations inform both the fit and the score is not a deployment margin.

2. **Is the pooled interval actually calibrated?**
   `quantile_calibration.json` reports `coverage_stride`, but the widening `k` is
   chosen on the same stride rows whose coverage it reports. This recomputes
   coverage with `k` fitted on one half of the stations and measured on the other.

This script **reads and recomputes only**. It writes nothing to `models/`, changes
no model, and does not regenerate any feature frame, so it is safe to run at any
time. It is a *diagnostic*, not a replacement for rerunning the pipeline: the
feature frames still carry the pre-purge split, so the numbers here remain
contaminated by whatever target leakage the purge is meant to remove. Treat them
as an upper bound until `06_features -> ... -> 32_backtest_traj` is rerun.

Outputs (new files only):
    outputs/honest_baseline_oos.json
    outputs/honest_baseline_oos.csv

Run from the ml directory:
    python 31_honest_baseline.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
PREDS = OUT / "predictions_2026.parquet"
Q_CAL = MODELS / "quantile_calibration.json"

COVERAGE_TARGET = 0.80          # mirrors 11_quantile.COVERAGE_TARGET
PERSIST_COL = "persist"
MODEL_COLS = ("xgb", "ridge", "persist", "clim")


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(math.sqrt(float(np.mean((p[ok] - y[ok]) ** 2))))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    return float(np.mean((lo[ok] <= y[ok]) & (y[ok] <= hi[ok])))


def fit_widen(med, half, obs, target=COVERAGE_TARGET, lo_s=0.2, hi_s=12.0) -> float:
    """Smallest widening s reaching the target coverage on the FIT rows."""
    avail = half > 1e-9
    if not avail.any():
        return float("nan")
    def cov(s):
        return float(np.mean((med[avail] - s * half[avail] <= obs[avail])
                             & (obs[avail] <= med[avail] + s * half[avail])))
    if cov(lo_s) >= target:
        return lo_s
    a, b = lo_s, hi_s
    for _ in range(60):
        m = 0.5 * (a + b)
        if cov(m) < target:
            a = m
        else:
            b = m
    return b


def main() -> None:
    if not PREDS.exists():
        raise SystemExit(f"missing {PREDS.relative_to(ROOT)} — run 07_evaluate.py first")

    df = pd.read_parquet(PREDS)
    df = df[df["horizon"] == 30].copy() if "horizon" in df.columns else df
    if "target" not in df.columns or "gwl" not in df.columns:
        raise SystemExit("predictions lack `target`/`gwl`; cannot score level RMSE")

    y = df["target"].to_numpy(dtype=float)
    anchor = df["gwl"].to_numpy(dtype=float)
    stride = df["stride"].to_numpy(dtype=bool) if "stride" in df.columns \
        else np.zeros(len(df), dtype=bool)

    # ---- split by STATION parity: no station informs both halves -----------
    # Uses a station-level hash rather than a sorted-alternating slice so the two
    # halves are stable under row reordering and the assignment costs no extra
    # memory at 378k rows.
    stations = sorted(df["Station"].astype(str).unique())
    st_arr = df["Station"].astype(str).to_numpy()
    st_hash = pd.util.hash_pandas_object(pd.Series(st_arr), index=False).to_numpy()
    fit_st = set(pd.Series(st_arr[st_hash % 2 == 0]).unique())
    fit = np.isin(st_arr, list(fit_st))
    rep = ~fit

    print(f"rows={len(df):,} stations={len(stations)} "
          f"stride_rows={int(stride.sum()):,}")
    print(f"station split: {len(fit_st)} fit / {len(stations) - len(fit_st)} held out")
    print(f"  fit rows={int(fit.sum()):,}  held-out rows={int(rep.sum()):,}  "
          f"held-out stride rows={int((rep & stride).sum()):,}")

    rows = []
    for label, mask in (("all_rows", np.ones(len(df), bool)),
                        ("stride_rows", stride),
                        ("oos_fit_stations", fit),
                        ("oos_heldout_stations", rep),
                        ("oos_heldout_stride", rep & stride)):
        if not mask.any():
            continue
        rec = {"basis": label, "rows": int(mask.sum()),
               "stations": int(df.loc[mask, "Station"].nunique())}
        for name in MODEL_COLS:
            if name in df.columns:
                rec[f"{name}_rmse"] = round(rmse(y[mask], df[name].to_numpy()[mask]), 4)
        base = rec.get("persist_rmse")
        if base and rec.get("xgb_rmse"):
            rec["xgb_vs_persist_pct"] = round(100 * (base - rec["xgb_rmse"]) / base, 2)
        rows.append(rec)
    table = pd.DataFrame(rows)

    # ---- pooled interval: fit k on one station half, report on the other ---
    quant = {}
    if {"q05_lvl", "q95_lvl"} <= set(df.columns):
        med = anchor + (df["xgb"].to_numpy(dtype=float) - anchor)   # q50 level = xgb level
        lo_raw = df["q05_lvl"].to_numpy(dtype=float)
        hi_raw = df["q95_lvl"].to_numpy(dtype=float)
        k_shipped = float(json.loads(Q_CAL.read_text()).get("widen_factor_k", 1.0)) \
            if Q_CAL.exists() else 1.0

        for label, fmask, rmask in (
            ("fit_half", fit & stride, rep & stride),
            ("all_stride", stride, stride),
        ):
            if not (fmask.any() and rmask.any()):
                continue
            # raw band edges around the median, then fit s on the fit rows only
            f_half = np.maximum((hi_raw[fmask] - lo_raw[fmask]) / 2.0, 0.0)
            f_med = med[fmask]
            f_obs = y[fmask]
            s = fit_widen(f_med, f_half, f_obs)
            r_half = np.maximum((hi_raw[rmask] - lo_raw[rmask]) / 2.0, 0.0)
            r_med = med[rmask]
            r_obs = y[rmask]
            quant[label] = {
                "s_fitted_on_fit_half": round(float(s), 3) if s == s else None,
                "coverage_fit_half_at_s": round(
                    coverage(f_obs, f_med - s * f_half, f_med + s * f_half), 4) if s == s else None,
                "coverage_heldout_at_s": round(
                    coverage(r_obs, r_med - s * r_half, r_med + s * r_half), 4) if s == s else None,
                "coverage_heldout_raw_band": round(
                    coverage(r_obs, r_med - r_half, r_med + r_half), 4),
                "n_fit": int(fmask.sum()), "n_heldout": int(rmask.sum()),
            }
        quant["shipped_widen_factor_k"] = k_shipped
        quant["note"] = ("`coverage_heldout_at_s` is the honest number: s is fitted on "
                         "the disjoint station half. The shipped `coverage_stride` in "
                         "quantile_calibration.json is fitted and reported on the same "
                         "rows, so it is in-sample.")
    else:
        quant["note"] = "predictions lack q05_lvl/q95_lvl — interval check skipped"

    payload = {
        "source": "outputs/predictions_2026.parquet",
        "caveat": ("recomputed from PRE-PURGE feature frames; target leakage that the "
                   "split purge is meant to remove is still present, so these are an "
                   "upper bound on honest performance"),
        "split": {"by": "Station parity (alternating in sorted order)",
                  "n_fit_stations": len(fit_st),
                  "n_heldout_stations": len(stations) - len(fit_st)},
        "bases": table.to_dict("records"),
        "quantile": quant,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "honest_baseline_oos.json").write_text(json.dumps(payload, indent=2))
    table.to_csv(OUT / "honest_baseline_oos.csv", index=False)

    print()
    print(table.round(4).to_string(index=False))
    if quant.get("fit_half"):
        q = quant["fit_half"]
        print(f"\ninterval: s_fit={q['s_fitted_on_fit_half']} "
              f"fit-half cov={q['coverage_fit_half_at_s']} "
              f"HELD-OUT cov={q['coverage_heldout_at_s']} "
              f"(raw band {q['coverage_heldout_raw_band']})")
    print(f"\nwrote {OUT / 'honest_baseline_oos.json'}")


if __name__ == "__main__":
    main()
