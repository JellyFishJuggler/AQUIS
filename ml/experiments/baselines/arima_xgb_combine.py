"""arima_xgb_combine — quantify the ARIMA + pooled-XGB *combination*.

What already exists in this repo is a **matched-pair comparison**, not a
combination: `baselines/arima_compare.py` scores SARIMA and the frozen trajectory
XGBoost on the same (station, origin) pairs across a 30-day horizon. Nothing
blends them. This module examines what a blend would buy, and how much of it is
real rather than fitted.

INPUT (produced by `arima_compare.py`):
    ml/outputs/arima_vs_xgb_predictions.csv
    columns: station, origin, day, actual, arima, arima_lo90, arima_hi90,
             xgb, xgb_lo90, xgb_hi90

If that file is absent, run `arima_compare.py` once — it now persists the
per-origin predictions. Everything here is offline analysis: no model is fitted
by this script beyond the blend weights themselves.

OUTPUT:
    ml/outputs/arima_xgb_combine.csv             per-method metrics
    ml/outputs/arima_xgb_combine_by_horizon.csv  per-method metrics at day 1/7/15/30
    ml/outputs/arima_xgb_combine.json            weights + protocol + verdict
    ml/outputs/arima_xgb_combine.md              narrative summary

PROTOCOL WARNING. This analysis inherits the ARIMA comparison's protocol:
daily means, ~10 stations, 5 rolling 2026 origins, 30-day horizon. It is **not**
the 6-hourly level-RMSE protocol used elsewhere in the pipeline, and its numbers
must not be quoted alongside those. Read the leave-one-origin-out section before
believing any blend weight.

Run from the ml directory:
    python baselines/arima_xgb_combine.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PRED = OUT / "arima_vs_xgb_predictions.csv"

MODELS = ("arima", "xgb")
HORIZONS = (1, 7, 15, 30)
RIDGE_LAMBDA = 1e-6          # tiny ridge keeps the 1-D solve stable
W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    e = np.asarray(pred, float) - np.asarray(y, float)
    ok = np.isfinite(e)
    e = e[ok]
    if e.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    return {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e))}


def interval_metrics(y, lo, hi) -> dict:
    """Coverage and width. Reported because RMSE alone ignores the band."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return {"picp90": np.nan, "mean_width": np.nan, "interval_score": np.nan}
    y, lo, hi = y[ok], lo[ok], hi[ok]
    alpha = 0.10
    width = hi - lo
    # Winkler / interval score: penalises BOTH miscoverage and excessive width,
    # which is what stops "just widen the band" from looking like an improvement.
    score = (width
             + (2 / alpha) * np.maximum(lo - y, 0.0)
             + (2 / alpha) * np.maximum(y - hi, 0.0))
    return {"picp90": float(np.mean((lo <= y) & (y <= hi))),
            "mean_width": float(np.mean(width)),
            "interval_score": float(np.mean(score))}


# ---------------------------------------------------------------------------
# weighting
# ---------------------------------------------------------------------------
def fit_weight_ls(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares weight w minimising ||w*a + (1-w)*x - y||, clipped to [0,1].

    Closed form restricted to the 1-D family (rather than a free 2-coefficient
    fit) so the result stays an interpretable blend instead of a rescaled model,
    and is far less prone to overfitting a handful of origins.
    """
    d = np.asarray(a, float) - np.asarray(x, float)
    r = np.asarray(y, float) - np.asarray(x, float)
    ok = np.isfinite(d) & np.isfinite(r)
    if not ok.any():
        return 0.5
    d, r = d[ok], r[ok]
    denom = float(np.dot(d, d)) + RIDGE_LAMBDA
    if denom <= 0:
        return 0.5
    w = float(np.dot(d, r) / denom)
    return float(min(1.0, max(0.0, w)))


def weight_inverse_rmse(rmse_a: float, rmse_x: float) -> float:
    """w on ARIMA by inverse-RMSE. A defensible default when fitting is unsafe."""
    if not (np.isfinite(rmse_a) and np.isfinite(rmse_x)) or (rmse_a + rmse_x) <= 0:
        return 0.5
    return float((1.0 / rmse_a) / ((1.0 / rmse_a) + (1.0 / rmse_x)))


def blend(a, x, w: float) -> np.ndarray:
    return w * np.asarray(a, float) + (1.0 - w) * np.asarray(x, float)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    if not PRED.exists():
        raise SystemExit(
            f"Missing {PRED.relative_to(ROOT)}.\n"
            "It is written by baselines/arima_compare.py. Run that once "
            "(it refits SARIMAX per origin), then re-run this script."
        )

    df = pd.read_csv(PRED)
    need = {"station", "origin", "day", "actual", "arima", "xgb"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"predictions file lacks columns: {sorted(missing)}")
    df["origin"] = pd.to_datetime(df["origin"], errors="coerce")
    df = df.dropna(subset=["actual", "arima", "xgb"]).reset_index(drop=True)

    n_st = df["station"].nunique()
    n_org = df.groupby("station")["origin"].nunique()
    y = df["actual"].to_numpy()
    a = df["arima"].to_numpy()
    x = df["xgb"].to_numpy()

    m_a, m_x = metrics(y, a), metrics(y, x)
    w_ls = fit_weight_ls(a, x, y)
    w_ir = weight_inverse_rmse(m_a["rmse"], m_x["rmse"])

    # ---- is a blend better than the better single model? -------------------
    rows = []
    for name, pred, extra in (
        ("arima", a, interval_metrics(y, df["arima_lo90"], df["arima_hi90"])
         if {"arima_lo90", "arima_hi90"} <= set(df.columns) else {}),
        ("xgb", x, interval_metrics(y, df["xgb_lo90"], df["xgb_hi90"])
         if {"xgb_lo90", "xgb_hi90"} <= set(df.columns) else {}),
        ("mean_50_50", blend(a, x, 0.5), {}),
        (f"ls_fit_w{w_ls:.2f}", blend(a, x, w_ls), {}),
        (f"inv_rmse_w{w_ir:.2f}", blend(a, x, w_ir), {}),
    ):
        rows.append({"method": name, **metrics(y, pred), **extra})
    overall = pd.DataFrame(rows)

    # full weight sweep, so the shape of the trade-off is visible
    sweep = pd.DataFrame([
        {"w_arima": w, "w_xgb": round(1 - w, 2), **metrics(y, blend(a, x, float(w)))}
        for w in W_GRID
    ])

    # ---- per-horizon ------------------------------------------------------
    hrows = []
    for day in HORIZONS:
        g = df[df["day"] == day]
        if g.empty:
            continue
        gy, ga, gx = (g["actual"].to_numpy(), g["arima"].to_numpy(),
                      g["xgb"].to_numpy())
        gw = fit_weight_ls(ga, gx, gy)
        for name, pred in (("arima", ga), ("xgb", gx),
                           ("mean_50_50", blend(ga, gx, 0.5)),
                           (f"ls_fit_w{gw:.2f}", blend(ga, gx, gw))):
            hrows.append({"day": day, "method": name, **metrics(gy, pred)})
    by_h = pd.DataFrame(hrows)

    # ---- leave-one-origin-out: the only honest test of a fitted weight ----
    origins = sorted(df["origin"].dropna().unique())
    cv_rows = []
    for o in origins:
        tr, te = df[df["origin"] != o], df[df["origin"] == o]
        if tr.empty or te.empty:
            continue
        w = fit_weight_ls(tr["arima"].to_numpy(), tr["xgb"].to_numpy(),
                          tr["actual"].to_numpy())
        p = blend(te["arima"].to_numpy(), te["xgb"].to_numpy(), w)
        cv_rows.append({
            "held_out_origin": str(pd.Timestamp(o).date()), "w_arima": round(w, 3),
            "blend_rmse": metrics(te["actual"].to_numpy(), p)["rmse"],
            "arima_rmse": metrics(te["actual"].to_numpy(), te["arima"].to_numpy())["rmse"],
            "xgb_rmse": metrics(te["actual"].to_numpy(), te["xgb"].to_numpy())["rmse"],
        })
    cv = pd.DataFrame(cv_rows)

    # ---- verdict ----------------------------------------------------------
    best_single = min(m_a["rmse"], m_x["rmse"])
    best_single_name = "arima" if m_a["rmse"] <= m_x["rmse"] else "xgb"
    best_blend = min(overall[overall["method"].str.startswith(("ls_fit", "inv_rmse",
                                                               "mean_"))]["rmse"])
    in_sample_gain = 100 * (best_single - best_blend) / best_single if best_single else np.nan

    oos_verdict = None
    if not cv.empty:
        cv_blend = float(cv["blend_rmse"].mean())
        cv_single = float(np.minimum(cv["arima_rmse"], cv["xgb_rmse"]).mean())
        oos_gain = 100 * (cv_single - cv_blend) / cv_single if cv_single else np.nan
        wins = int((cv["blend_rmse"] < np.minimum(cv["arima_rmse"],
                                                  cv["xgb_rmse"])).sum())
        oos_verdict = {
            "folds": int(len(cv)), "blend_rmse_mean": round(cv_blend, 4),
            "best_single_rmse_mean": round(cv_single, 4),
            "gain_pct": round(float(oos_gain), 2),
            "folds_beating_best_single": wins,
            "w_arima_range": [float(cv["w_arima"].min()), float(cv["w_arima"].max())],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT / "arima_xgb_combine.csv", index=False)
    by_h.to_csv(OUT / "arima_xgb_combine_by_horizon.csv", index=False)
    sweep.to_csv(OUT / "arima_xgb_weight_sweep.csv", index=False)
    if not cv.empty:
        cv.to_csv(OUT / "arima_xgb_combine_cv.csv", index=False)

    summary = {
        "what_exists_today": (
            "a matched-pair COMPARISON on shared (station, origin) pairs — "
            "no blend, no ensemble, no stacking anywhere in the pipeline"
        ),
        "protocol": {
            "basis": "daily means, rolling 2026 origins, 30-day horizon",
            "stations": int(n_st),
            "origins_per_station": [int(n_org.min()), int(n_org.max())],
            "rows": int(len(df)),
            "comparable_with": "nothing else in the pipeline — not the 6-hourly "
                               "level-RMSE tables",
        },
        "single_model_rmse": {"arima": round(m_a["rmse"], 4),
                              "xgb": round(m_x["rmse"], 4)},
        "weights": {"ls_fit_w_arima": round(w_ls, 3),
                    "inverse_rmse_w_arima": round(w_ir, 3)},
        "in_sample_best_blend_rmse": round(float(best_blend), 4),
        "in_sample_gain_vs_best_single_pct": round(float(in_sample_gain), 2),
        "leave_one_origin_out": oos_verdict,
        "verdict": None,
    }
    if oos_verdict:
        if oos_verdict["gain_pct"] > 1.0 and oos_verdict["folds_beating_best_single"] >= 3:
            summary["verdict"] = ("a blend helps on held-out origins — but the sample is "
                                  f"only {oos_verdict['folds']} origins, so treat it as a "
                                  "hypothesis until more origins accumulate")
        else:
            summary["verdict"] = ("no reliable blend gain out of sample; keep the better "
                                  "single model rather than fitting weights")
    (OUT / "arima_xgb_combine.json").write_text(json.dumps(summary, indent=2))

    # ---- markdown ---------------------------------------------------------
    with (OUT / "arima_xgb_combine.md").open("w") as f:
        f.write("# ARIMA + pooled-XGB combination analysis\n\n")
        f.write("## What exists today\n\n")
        f.write("`baselines/arima_compare.py` performs a **matched-pair comparison** on "
                "shared (station, origin) pairs. There is no ensemble, blend, or "
                "stacking anywhere in the pipeline; this document quantifies what a "
                "blend would add.\n\n")
        f.write(f"Stations: {n_st} · origins per station: {int(n_org.min())}–"
                f"{int(n_org.max())} · scored rows: {len(df):,}\n\n")
        f.write("> **Protocol:** daily means, rolling 2026 origins, 30-day horizon. "
                "**Not comparable** with the 6-hourly level-RMSE tables.\n\n")
        f.write("## All days\n\n")
        f.write("| Method | n | RMSE | MAE | bias | PICP90 | mean width | interval score |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in overall.iterrows():
            has_iv = pd.notna(r.get("picp90", np.nan))
            iv = (f"{r['picp90']:.3f} | {r['mean_width']:.3f} | "
                  f"{r['interval_score']:.3f}" if has_iv else "— | — | —")
            f.write(f"| {r['method']} | {int(r['n'])} | {r['rmse']:.3f} | "
                    f"{r['mae']:.3f} | {r['bias']:+.3f} | {iv} |\n")
        f.write("\nInterval columns are shown only for the single models; a blended "
                "band would need its own calibration and is deliberately not invented "
                "here.\n\n")
        f.write(f"Least-squares blend weight on ARIMA: **{w_ls:.3f}** · "
                f"inverse-RMSE weight: **{w_ir:.3f}**\n\n")
        if oos_verdict:
            f.write("## Leave-one-origin-out (the honest test)\n\n")
            f.write("| Held-out origin | w_arima | blend RMSE | ARIMA RMSE | XGB RMSE |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for _, r in cv.iterrows():
                f.write(f"| {r['held_out_origin']} | {r['w_arima']:.2f} | "
                        f"{r['blend_rmse']:.3f} | {r['arima_rmse']:.3f} | "
                        f"{r['xgb_rmse']:.3f} |\n")
            f.write(f"\nMean blend RMSE {oos_verdict['blend_rmse_mean']:.3f} vs best "
                    f"single {oos_verdict['best_single_rmse_mean']:.3f} "
                    f"(**{oos_verdict['gain_pct']:+.2f}%**); the blend beat the better "
                    f"single model in **{oos_verdict['folds_beating_best_single']}** of "
                    f"{oos_verdict['folds']} folds.\n\n")
            f.write(f"**Verdict:** {summary['verdict']}.\n")
    print(overall.round(4).to_string(index=False))
    if not cv.empty:
        print("\nleave-one-origin-out:")
        print(cv.round(4).to_string(index=False))
    print(f"\nwrote {OUT / 'arima_xgb_combine.md'}")


if __name__ == "__main__":
    main()
