"""Diagnostic 4b: fleet-wide best-lag test of the issued 30-day forecast.

Read-only w.r.t. production data; writes only into this directory.
"""
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PREDS = os.path.join(ROOT, "outputs", "predictions_2026.parquet")
ALIGNED = os.path.join(ROOT, "data", "aligned", "table_6h.parquet")

LAGS = np.arange(0, 46)  # 0..45 days
MIN_ROWS_PER_STATION = 3
MIN_MATCHED_LAGS = 10
DASHBOARD_PREFIXES = ["Azamgarh", "Deoria", "Banda", "Meerut", "Varanasi", "ASHADHA"]


def main():
    # ---- 1. stride rows -------------------------------------------------
    p = pd.read_parquet(PREDS, columns=["Station", "time", "xgb", "target", "persist", "stride"])
    p = p[p["stride"] == True].copy()  # noqa: E712
    p["time"] = pd.to_datetime(p["time"])
    p = p[np.isfinite(p["xgb"]) & np.isfinite(p["target"])]
    counts = p.groupby("Station").size()
    keep = counts[counts >= MIN_ROWS_PER_STATION].index
    p = p[p["Station"].isin(keep)].reset_index(drop=True)
    p["row_id"] = np.arange(len(p))

    # ---- 2. aligned actual levels --------------------------------------
    a = pd.read_parquet(ALIGNED, columns=["Station", "time", "gwl"])
    a["time"] = pd.to_datetime(a["time"])
    a = a.dropna(subset=["gwl"]).sort_values(["Station", "time"])

    # ---- 3. best-lag test (vectorized mesh of anchor x lag) -------------
    n, nl = len(p), len(LAGS)
    mesh = pd.DataFrame({
        "row_id": np.repeat(p["row_id"].values, nl),
        "Station": np.repeat(p["Station"].values, nl),
        "lag": np.tile(LAGS, n),
        "time": (p["time"].values.astype("datetime64[s]")[:, None]
                 + np.array(LAGS, dtype="timedelta64[D]").astype("timedelta64[s]")
                 .reshape(1, nl)).ravel(),
        "issued": np.repeat(p["xgb"].values, nl),
    })
    m = mesh.merge(a.rename(columns={"time": "time", "gwl": "actual"}),
                   on=["Station", "time"], how="inner")
    m["err2"] = (m["issued"] - m["actual"]) ** 2

    g = (m.groupby(["Station", "lag"], sort=True)
           .agg(n_match=("err2", "size"), sse=("err2", "sum"))
           .reset_index())
    g["rmse"] = np.sqrt(g["sse"] / g["n_match"])

    wide_rmse = g.pivot(index="Station", columns="lag", values="rmse").reindex(columns=LAGS)
    wide_n = g.pivot(index="Station", columns="lag", values="n_match").reindex(columns=LAGS).fillna(0)

    n_valid_lags = (wide_n > 0).sum(axis=1)
    eligible = n_valid_lags[n_valid_lags >= MIN_MATCHED_LAGS].index

    # best lag: min RMSE among lags with >=1 match; ties -> smallest lag
    masked = wide_rmse.where(wide_n > 0)
    best_lag = masked.idxmin(axis=1)  # first occurrence = smallest lag on ties
    rmse_at_best = masked.min(axis=1)

    # ---- 4a/4b. per-station table --------------------------------------
    rows = []
    for st in eligible:
        sub = p[p["Station"] == st]
        e_x = sub["xgb"] - sub["target"]
        e_p = sub["persist"] - sub["target"]
        rows.append({
            "station": st,
            "n_stride_rows": int(len(sub)),
            "best_lag_days": int(best_lag[st]),
            "rmse_at_best_lag": float(rmse_at_best[st]),
            "rmse_at_lag30": float(wide_rmse.loc[st, 30]) if wide_n.loc[st, 30] > 0 else None,
            "rmse_xgb_30d": float(np.sqrt(np.mean(e_x ** 2))),
            "rmse_persist_30d": float(np.sqrt(np.mean(e_p ** 2))),
        })
    out = pd.DataFrame(rows)
    out["rmse_persist_shifted"] = out["rmse_persist_30d"]
    out = out.sort_values("station").reset_index(drop=True)
    out.to_csv(os.path.join(HERE, "best_lag.csv"), index=False)

    # ---- 4c. fleet summary ---------------------------------------------
    bl = out["best_lag_days"].to_numpy()
    hist = {int(k): int(v) for k, v in
            zip(*np.unique(bl, return_counts=True)) if k in set(LAGS.tolist())}
    full_hist = {int(l): int((bl == l).sum()) for l in LAGS}
    within = int(((bl >= 25) & (bl <= 35)).sum())
    pct_within = 100.0 * within / len(out) if len(out) else 0.0

    dash = {}
    missing_prefixes = []
    for pf in DASHBOARD_PREFIXES:
        sub = out[out["station"].str.lower().str.startswith(pf.lower())]
        if len(sub) == 0:
            missing_prefixes.append(pf)
            dash[pf] = None
            continue
        entries = []
        for _, r in sub.iterrows():
            entries.append({
                "station": r["station"],
                "best_lag_days": int(r["best_lag_days"]),
                "rmse_xgb_30d": round(float(r["rmse_xgb_30d"]), 4),
                "rmse_persist_30d": round(float(r["rmse_persist_30d"]), 4),
                "xgb_beats_persist": bool(r["rmse_xgb_30d"] < r["rmse_persist_30d"]),
            })
        dash[pf] = {
            "n_stations": int(len(sub)),
            "stations": entries,
            "mean_best_lag_days": round(float(sub["best_lag_days"].mean()), 2),
            "all_xgb_beats_persist": bool((sub["rmse_xgb_30d"] < sub["rmse_persist_30d"]).all()),
            "n_xgb_beats_persist": int((sub["rmse_xgb_30d"] < sub["rmse_persist_30d"]).sum()),
        }

    summary = {
        "total_stations_tested": int(len(out)),
        "stride_rows_used": int(out["n_stride_rows"].sum()),
        "lag_distribution": {
            "mean_best_lag_days": round(float(bl.mean()), 3),
            "median_best_lag_days": float(np.median(bl)),
            "min_best_lag_days": int(bl.min()),
            "max_best_lag_days": int(bl.max()),
            "histogram_0_to_45": {str(k): v for k, v in full_hist.items() if v > 0},
        },
        "pct_stations_best_lag_25_to_35_days": round(pct_within, 3),
        "n_stations_best_lag_25_to_35_days": within,
        "dashboard_stations": dash,
        "dashboard_prefixes_with_no_station": missing_prefixes,
        "xgb_beats_persist_pct_fleet": round(
            100.0 * float((out["rmse_xgb_30d"] < out["rmse_persist_30d"]).mean()), 3),
        "png_written": False,
        "notes": [],
    }

    # ---- 4d. lag distribution chart -------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(full_hist.keys(), full_hist.values(), color="#1f77b4")
        ax.axvspan(24.5, 35.5, color="green", alpha=0.12, label="25-35 d window")
        ax.set_xlabel("Best lag (days)")
        ax.set_ylabel("Number of stations")
        ax.set_title("Fleet best-lag distribution (issued 30-d q50 vs actual)")
        ax.set_xticks(list(LAGS))
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "lag_distribution.png"), dpi=120)
        plt.close(fig)
        summary["png_written"] = True
    except Exception as exc:  # matplotlib missing / headless failure
        summary["notes"].append(f"lag_distribution.png skipped: {exc!r}")

    with open(os.path.join(HERE, "fleet_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- 5. console summary ---------------------------------------------
    print(f"stations tested: {len(out)}  stride rows: {int(out['n_stride_rows'].sum())}")
    print(f"best lag mean/median: {bl.mean():.2f} / {np.median(bl):.1f} d")
    print(f"within 25-35 d: {within}/{len(out)} = {pct_within:.1f}%")
    print(f"histogram: {summary['lag_distribution']['histogram_0_to_45']}")
    print(f"xgb beats 30-d persistence: {summary['xgb_beats_persist_pct_fleet']}% of stations")
    print("--- dashboard prefixes ---")
    for pf in DASHBOARD_PREFIXES:
        d = dash[pf]
        if d is None:
            print(f"{pf:10s}: NO STATION FOUND (prefix match)")
            continue
        print(f"{pf:10s}: n={d['n_stations']} mean_best_lag={d['mean_best_lag_days']} "
              f"xgb_beats_persist={d['n_xgb_beats_persist']}/{d['n_stations']}")
        for e in d["stations"]:
            print(f"    {e['station'][:40]:40s} lag={e['best_lag_days']:2d} "
                  f"xgb_rmse={e['rmse_xgb_30d']:8.3f} pers_rmse={e['rmse_persist_30d']:8.3f} "
                  f"beats={e['xgb_beats_persist']}")
    if missing_prefixes:
        print("prefixes with no aligned/stride station:", missing_prefixes)


if __name__ == "__main__":
    main()
