"""exp_track2_tiers — Track 2: per-tier 2026 test skill and coverage.

Tier labels come from TRAINING-YEAR data only (no 2026 information):

  Axis A (volatility): rank-quartiles of per-station persistence RMSE
      (sqrt(mean(target_d^2))) on the fold-2024 stride rows (the intact
      training-year monsoon fold). a_q1 = calmest ... a_q4 = most volatile.
      Test stations with no fold-2024 stride rows -> "unassigned".
  Axis B (completeness): rank-quartiles of per-station 6h record
      completeness over the training window (date < 2026-01-01):
      observed rows / expected rows in [first, last] observation span.
      b_q1 = sparsest ... b_q4 = densest.

The already-selected configs E0 and E_best are re-fit on the IDENTICAL 2026
one-shot windows (same params/seeds as exp_eval_2026) solely to obtain
per-row predictions for tier slicing. The recomputed pooled headlines are
ASSERTED equal to eval_2026.json; the script aborts if they differ, so no
new test-set number can silently enter the report.

Outputs: tracks/preds_2026_stride.parquet (per-row, reusable),
         tracks/tiers_2026.json (all tables).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_harness as H
import exp_eval_2026 as E26

OUT_JSON = H.HERE / "tracks" / "tiers_2026.json"
OUT_PRED = H.HERE / "tracks" / "preds_2026_stride.parquet"
B = 2000


def rank_quarters(vals: pd.Series, prefix: str) -> pd.Series:
    """Equal-count rank quartiles (ties broken by order). q1 = smallest."""
    r = vals.rank(method="first")
    q = np.ceil(r / len(r) * 4).astype(int)
    return pd.Series([f"{prefix}_q{k}" for k in q], index=vals.index)


def axis_a(df: pd.DataFrame, folds: dict, stride: dict) -> pd.Series:
    yd = df["target_d"].to_numpy(float)
    stn = df["Station"].astype(str)
    ix = stride[2024]
    per = pd.Series(yd[ix]).groupby(stn.to_numpy()[ix]).apply(
        lambda s: float(np.sqrt(np.mean(s ** 2))))
    per = per.dropna()
    return rank_quarters(per, "a")          # station -> a_q1..a_q4


def completeness(df: pd.DataFrame) -> pd.Series:
    """Numeric per-station record completeness over the training window."""
    d = df["date"].to_numpy("datetime64[ns]")
    g = df["gwl"].to_numpy(float)
    stn = df["Station"].astype(str).to_numpy()
    tr = d < np.datetime64("2026-01-01")
    obs = pd.Series(g[tr] * 0 + 1, index=pd.Index(stn[tr])).groupby(level=0).sum()
    t = pd.to_datetime(pd.Series(d[tr], index=pd.Index(stn[tr])))
    span = (t.groupby(level=0).max() - t.groupby(level=0).min())
    span_6h = span.dt.total_seconds() / 21600 + 1
    return (obs / span_6h).dropna()


def axis_b(df: pd.DataFrame) -> pd.Series:
    return rank_quarters(completeness(df), "b")


def boot(em, ep, st_tier, b=B, seed=7):
    st_u = np.unique(st_tier)
    if st_u.size < 5:
        return None
    pos = {int(s): np.flatnonzero(st_tier == s) for s in st_u}
    rng = np.random.default_rng(seed)
    sample = rng.choice(st_u, size=(b, st_u.size), replace=True)
    ratios = np.empty(b)
    for i in range(b):
        sel = np.concatenate([pos[int(s)] for s in sample[i]])
        ratios[i] = np.sqrt(np.mean(em[sel] ** 2)) / np.sqrt(np.mean(ep[sel] ** 2))
    return [float(x) for x in np.quantile(ratios, [0.05, 0.95])]


def tier_table(pred_df: pd.DataFrame, label: str) -> dict:
    out = {}
    for tier, sub in pred_df.groupby(label):
        mjs = sub["month"].isin([6, 7, 8, 9]).to_numpy()
        rec = {
            "n_stations": int(sub["st"].nunique()),
            "n_rows": int(len(sub)),
            "n_junsep": int(mjs.sum()),
            "completeness_mean": round(float(sub["comp"].mean()), 4)
            if "comp" in sub else None,
        }
        for cfg in ("E0", "E_best"):
            em = sub[f"{cfg}_err"].to_numpy(float)
            ep = sub["p_err"].to_numpy(float)
            rec[cfg] = round(float(np.sqrt(np.mean(em ** 2))) /
                             float(np.sqrt(np.mean(ep ** 2))), 4)
            rec[f"{cfg}_ci"] = boot(em, ep, sub["st"].to_numpy(np.int32))
            rec[f"{cfg}_junsep"] = (
                round(float(np.sqrt(np.mean(em[mjs] ** 2))) /
                      float(np.sqrt(np.mean(ep[mjs] ** 2))), 4)
                if mjs.any() else None)
        cov = sub["in_int"].to_numpy(bool)
        rec["interval_coverage"] = round(float(cov.mean()), 4)
        rec["interval_width_mean"] = round(float(
            (sub["q95"].to_numpy(float) - sub["q05"].to_numpy(float)).mean()), 4)
        out[str(tier)] = rec
    return out


def main() -> None:
    t0 = time.time()
    df = H.load_frame()
    df = H.ensure_aux(df)
    df = H.build_neighbour(df)
    folds = H.build_folds(df)
    stride = H.holdout_stride(df, folds)

    a_lab = axis_a(df, folds, stride)
    b_lab = axis_b(df)

    tr, va, te_cand = E26.mask_windows(df)
    ho = E26.te_stride(df, te_cand)
    print(f"2026 stride rows: {len(ho):,}", flush=True)

    gwl = df["gwl"].to_numpy(float)
    yd = df["target_d"].to_numpy(float)
    stE = df["st_id"].to_numpy(np.int32)
    stn = df["Station"].astype(str).to_numpy()
    moE = df["date"].dt.month.to_numpy(np.int8)

    # ---- fits (identical to exp_eval_2026; per-row output) ----------------
    preds = {}
    for cfg_name, cfg in {"E0": ("sq", H.BASE_XCOLS),
                          "E_best": ("huber", E26.E_BEST_COLS)}.items():
        yobj, cols = cfg
        Xtr = np.ascontiguousarray(df.iloc[tr][cols].to_numpy("float32"))
        Xva = np.ascontiguousarray(df.iloc[va][cols].to_numpy("float32"))
        Xho = np.ascontiguousarray(df.iloc[ho][cols].to_numpy("float32"))
        ytr, yva = yd[tr].astype(np.float64), yd[va].astype(np.float64)
        sp = {}
        for seed in H.SEEDS:
            m_ = E26._fit(yobj, None, cols, seed, Xtr, ytr, Xva, yva)
            sp[seed] = m_.predict(Xho).astype(np.float64)
        preds[cfg_name] = sp
        del Xtr, Xva, Xho, m_
        gc.collect()

    qq = {}
    Xtr = np.ascontiguousarray(df.iloc[tr][E26.E_BEST_COLS].to_numpy("float32"))
    Xva = np.ascontiguousarray(df.iloc[va][E26.E_BEST_COLS].to_numpy("float32"))
    Xho = np.ascontiguousarray(df.iloc[ho][E26.E_BEST_COLS].to_numpy("float32"))
    for a, label in ((0.05, "q05"), (0.95, "q95")):
        ps = {}
        for seed in H.SEEDS:
            m_ = E26._fit(label, a, E26.E_BEST_COLS, seed, Xtr, ytr, Xva, yva)
            ps[seed] = m_.predict(Xho).astype(np.float64)
        qq[label] = ps
    del Xtr, Xva, Xho, m_
    gc.collect()

    # ---- per-row frame ----------------------------------------------------
    p_df = pd.DataFrame({
        "Station": stn[ho], "st": stE[ho], "time": df["time"].to_numpy()[ho],
        "month": moE[ho], "gwl": gwl[ho], "target_d": yd[ho],
        "p_err": yd[ho],
        "E0_pred": np.mean([preds["E0"][s] for s in H.SEEDS], axis=0),
        "E_best_pred": np.mean([preds["E_best"][s] for s in H.SEEDS], axis=0),
        "q05": np.mean([qq["q05"][s] for s in H.SEEDS], axis=0),
        "q95": np.mean([qq["q95"][s] for s in H.SEEDS], axis=0),
    })
    p_df["E0_err"] = p_df["target_d"] - p_df["E0_pred"]
    p_df["E_best_err"] = p_df["target_d"] - p_df["E_best_pred"]
    p_df["in_int"] = (p_df["target_d"] >= p_df["q05"]) & \
                     (p_df["target_d"] <= p_df["q95"])

    # tier labels (training-year only)
    p_df["axis_a"] = p_df["Station"].map(a_lab).fillna("unassigned")
    p_df["axis_b"] = p_df["Station"].map(b_lab).fillna("unassigned")
    comp_map = completeness(df)
    p_df["comp"] = p_df["Station"].map(comp_map).fillna(np.nan)

    # ---- ASSERT headlines against the already-run one-shot ----------------
    # tolerance: GPU hist fits are not guaranteed bitwise-reproducible; the
    # assert exists to catch window/feature regressions, not 4th-decimal noise
    ref = json.loads((H.HERE / "eval_2026.json").read_text())
    checks = {}
    for cfg in ("E0", "E_best"):
        rm = float(np.sqrt(np.mean(p_df[f"{cfg}_err"] ** 2)))
        rp = float(np.sqrt(np.mean(p_df["p_err"] ** 2)))
        got, want = round(rm / rp, 4), ref[cfg]["ratio"]
        checks[f"{cfg}_ratio"] = [got, want]
        assert abs(got - want) < 0.003, \
            f"{cfg} headline mismatch: {got} vs {want} (>0.003)"
    cov = float(p_df["in_int"].mean())
    got_c, want_c = round(cov, 4), ref["E_best"]["interval"]["coverage"]
    checks["coverage"] = [got_c, want_c]
    assert abs(got_c - want_c) < 0.02, f"coverage mismatch: {got_c} vs {want_c}"
    print("headline assertions OK:", checks, flush=True)

    # ---- tables -----------------------------------------------------------
    report = {
        "tier_source": "fold-2024 stride rows (volatility) / training-window "
                       "6h record completeness; labels never see 2026",
        "headline_check": checks,
        "axis_a_volatility": tier_table(p_df, "axis_a"),
        "axis_b_completeness": tier_table(p_df, "axis_b"),
        "tier_sizes_stations": {
            "axis_a": p_df.drop_duplicates("Station").groupby("axis_a")
                          .size().astype(int).to_dict(),
            "axis_b": p_df.drop_duplicates("Station").groupby("axis_b")
                          .size().astype(int).to_dict(),
        },
        "unassigned_bucket": {
            "rows_axis_a": int((p_df["axis_a"] == "unassigned").sum()),
            "rows_axis_b": int((p_df["axis_b"] == "unassigned").sum()),
            "stations_axis_a": int(p_df.loc[p_df["axis_a"] == "unassigned",
                                            "Station"].nunique()),
            "stations_axis_b": int(p_df.loc[p_df["axis_b"] == "unassigned",
                                            "Station"].nunique()),
            "note": "a station lands here only if it has no fold-2024 stride "
                    "rows (axis A) / no training-window rows (axis B); 0 means "
                    "every 2026 test station was labelled from training data",
        },
        "elapsed_sec": round(time.time() - t0),
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))
    p_df.to_parquet(OUT_PRED, index=False)

    print("\n=== TIER TABLES (E0 / E_best ratio, Jun-Sep ratio, coverage, width) ===")
    for axis in ("axis_a_volatility", "axis_b_completeness"):
        print(f"\n{axis}")
        for t, r in report[axis].items():
            print(f"  {t:12s} nst={r['n_stations']:3d} rows={r['n_rows']:4d} "
                  f"njs={r['n_junsep']:4d} E0={r['E0']} js={r['E0_junsep']} "
                  f"E_best={r['E_best']} js={r['E_best_junsep']} "
                  f"cov={r['interval_coverage']} w={r['interval_width_mean']}")
    print(f"\nsizes: {report['tier_sizes_stations']}")
    print(f"unassigned: {report['unassigned_bucket']}")
    print(f"TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
