"""scratch_assoc_ablate — 3-arm association ablation on the PRODUCTION honest
stride protocol. Isolated: writes ONLY under data/scratch_assoc/. Production
data/aligned, data/features and models/ are never touched.

Arms (same split VAL_START=2025-10-01 / TEST_START=2026-01-01 via 06_features.split):
  A : old links as-is (table_6h.parquet)
  B : rainfall ONLY replaced by _association.k_nearest + Gaussian combine
  C : B + per-station wsum_* / <k>_dmin_km static cols for weather+canal drivers

Each arm trains pooled XGBoost (reg:squarederror, delta 30d, early stop on val)
on 3 seeds; score = 30 d stride rows (step_idx % 120 == 0) on test. Reported:
per-arm stride RMSE mean+spread, model/persistence ratio with station-bootstrap
CI overall and split Jun-Sep vs other months, and a two-fold spatial CV
(station-held-out) for C vs B. A winner is declared only if the CI excludes 1
AND the gain strictly exceeds the seed spread of both competing arms.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

F = importlib.import_module("06_features")
A = importlib.import_module("_association")
import config  # noqa: E402   (ROOT/data paths)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "scratch_assoc"
OUT.mkdir(parents=True, exist_ok=True)

TRACK_A = ROOT / "data" / "aligned" / "table_6h.parquet"
SEEDS = (42, 840, 2056)
XGB_PARAMS = dict(
    n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
    objective="reg:squarederror", eval_metric="rmse", n_jobs=6, verbose=False,
)
WEATHER_EXTRAS = ["temperature", "humidity", "solar", "wind_speed", "pressure",
                  "canal_level"]


# ---------------------------------------------------------------------------
# association / aligned-track construction (scratch only)
# ---------------------------------------------------------------------------
def load_norm(name: str) -> pd.DataFrame:
    p = Path(config.RAW) / f"{name}_norm.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def new_values(key: str, anchors: pd.DataFrame) -> pd.DataFrame:
    gauges = load_norm(key).dropna(subset=["lat", "lon"])
    if gauges.empty:
        return pd.DataFrame()
    g = gauges.copy()
    g["bin"] = g["time"].dt.floor(pd.Timedelta("6h"))
    agg = "sum" if key == "rainfall" else "mean"
    values = (g.groupby(["Station", "bin"], as_index=False)["value"]
              .agg(agg).rename(columns={"bin": "time"}))
    links = A.k_nearest(anchors, gauges, key)
    if links.empty or values.empty:
        return pd.DataFrame()
    return A.combine(links, values, key)


def station_meta(key: str, anchors: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    gauges = load_norm(key).dropna(subset=["lat", "lon"])
    links = A.k_nearest(anchors, gauges, key)
    if links.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = links.groupby("anchor_station").agg(wsum=("w_raw", "sum"),
                                            dmin_km=("dist_km", "min"))
    return g["wsum"], g["dmin_km"]


def build_arm_a() -> pd.DataFrame:
    track = pd.read_parquet(TRACK_A)
    track["time"] = pd.to_datetime(track["time"], errors="coerce")
    return track


def build_arm_b(base: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    rain = new_values("rainfall", anchors)
    if rain.empty:
        raise RuntimeError("arm B: no new rainfall values")
    out = base.copy().drop(columns=["rain"], errors="ignore")
    out = out.merge(rain[["Station", "time", "value"]]
                    .rename(columns={"value": "rain"}), on=["Station", "time"],
                    how="left")
    return out


def build_arm_c(base_b: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    out = base_b.copy()
    for key in WEATHER_EXTRAS:
        wsum, dmin = station_meta(key, anchors)
        out[f"{key}_wsum"] = out["Station"].map(wsum)
        out[f"{key}_dmin_km"] = out["Station"].map(dmin)
    return out


def prepare_track(track: pd.DataFrame) -> pd.DataFrame:
    return F.drop_gwl_spikes(track)


# ---------------------------------------------------------------------------
# honest stride protocol (same-split feature build per arm)
# ---------------------------------------------------------------------------
def arm_prep(track: pd.DataFrame, xcols: list[str]):
    feats, extra = F.build_full(prepare_track(track))
    xcols = F.NUM_COLS + [c for c in xcols if c not in F.NUM_COLS]
    parts = F.split(feats)
    del feats
    gc.collect()
    train, val = parts["train"], parts["val"]
    test = parts["test"].sort_values(["Station", "time"]).reset_index(drop=True)
    test["step_idx"] = test.groupby("Station").cumcount()
    stride = (test["step_idx"] % 120 == 0).to_numpy()
    Xte = test[xcols].copy()
    return {
        "xcols": xcols,
        "Xtr": train, "ytr_d": train["target_d"],
        "Xva": val, "yva_d": val["target_d"],
        "train_station": train["Station"].astype(str).to_numpy(),
        "val_station": val["Station"].astype(str).to_numpy(),
        "Xte": Xte,
        "te_idx": np.flatnonzero(stride),
        "te_station": test["Station"].astype(str).to_numpy()[stride],
        "te_month": pd.to_datetime(test["time"]).dt.month.to_numpy()[stride],
        "true_l": test["target"].to_numpy(dtype=np.float64)[stride],
        "anchor": test["gwl"].to_numpy(dtype=np.float64)[stride],
    }


def run_arm(prep: dict, label: str, fit_stations=None, holdout_stations=None,
            seeds=SEEDS, tags=(True, True, True)) -> tuple[dict, dict]:
    """Fit per seed (optionally on a station subset), score on stride rows
    (optionally restricted to holdout stations). Returns (report, fits)."""
    tr_mask = np.ones(prep["Xtr"].shape[0], dtype=bool)
    va_mask = np.ones(prep["Xva"].shape[0], dtype=bool)
    if fit_stations is not None:
        fs = set(fit_stations)
        tr_mask &= np.isin(prep["train_station"], list(fs))
        va_mask &= np.isin(prep["val_station"], list(fs))
    assert tr_mask.sum() > 0 and va_mask.sum() > 0, f"{label}: empty fit slice"

    stride_mask = np.ones(prep["te_station"].size, dtype=bool)
    if holdout_stations is not None:
        hs = set(holdout_stations)
        stride_mask &= np.isin(prep["te_station"], list(hs))
    assert stride_mask.sum() > 0, f"{label}: empty holdout stride slice"

    xc = prep["xcols"]
    Xtr = prep["Xtr"].loc[tr_mask, xc]
    ytr = prep["ytr_d"].loc[tr_mask].to_numpy(float)
    Xva = prep["Xva"].loc[va_mask, xc]
    yva = prep["yva_d"].loc[va_mask].to_numpy(float)
    Xte = prep["Xte"]

    deltas = []
    per_seed = {}
    fits = {}
    for seed in seeds:
        m = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
        try:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=60,
                  verbose=False)
        except TypeError:
            m.set_params(early_stopping_rounds=60)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        d = m.predict(Xte)[prep["te_idx"]][stride_mask].astype(np.float64)
        deltas.append(d)
        e = prep["true_l"][stride_mask] - (prep["anchor"][stride_mask] + d)
        per_seed[seed] = float(math.sqrt(float(np.mean(e ** 2))))
        fits[(label, seed)] = m

    dmean = np.mean(np.stack(deltas), axis=0)
    err = prep["true_l"][stride_mask] - (prep["anchor"][stride_mask] + dmean)
    perr = prep["true_l"][stride_mask] - prep["anchor"][stride_mask]
    station = prep["te_station"][stride_mask]
    month = prep["te_month"][stride_mask]

    def ratio_ci(e1, e2, st, b=2000, sd=2026):
        rng = np.random.default_rng(sd)
        stl = np.unique(st)
        r = [math.sqrt(np.mean(e1[np.isin(st, rng.choice(stl, stl.size, True))] ** 2))
             / math.sqrt(np.mean(e2[np.isin(st, rng.choice(stl, stl.size, True))] ** 2))
             for _ in range(b)]
        r = np.array(r)
        return float(np.mean(r)), (float(np.percentile(r, 2.5)),
                                   float(np.percentile(r, 97.5)))
    # NOTE: resample stations once per draw (shared membership), not twice
    def ratio_ci2(e1, e2, st, b=2000, sd=2026):
        rng = np.random.default_rng(sd)
        stl = np.unique(st)
        vals = []
        for _ in range(b):
            keep = np.isin(st, rng.choice(stl, stl.size, replace=True))
            a = math.sqrt(np.mean(e1[keep] ** 2))
            c = math.sqrt(np.mean(e2[keep] ** 2))
            vals.append(a / c)
        r = np.asarray(vals)
        return float(np.mean(r)), (float(np.percentile(r, 2.5)),
                                   float(np.percentile(r, 97.5)))

    rmse_m = float(math.sqrt(np.mean(err ** 2)))
    rmse_p = float(math.sqrt(np.mean(perr ** 2)))
    mono = np.isin(month, (6, 7, 8, 9))
    report = {
        "label": label,
        "n_stride": int(err.size),
        "stations": int(np.unique(station).size),
        "rmse_model": rmse_m,
        "rmse_persist": rmse_p,
        "per_seed_rmse": {str(k): round(v, 4) for k, v in per_seed.items()},
        "seed_spread_rmse": round(max(per_seed.values()) - min(per_seed.values()), 4),
        "ratio_all": dict(zip(("mean", "ci"), [ratio_ci2(err, perr, station)[0],
                                               list(ratio_ci2(err, perr, station)[1])])),
        "ratio_junsep": dict(zip(("mean", "ci"), [ratio_ci2(err[mono], perr[mono], station[mono])[0],
                                                  list(ratio_ci2(err[mono], perr[mono], station[mono])[1])])),
        "n_junsep": int(mono.sum()),
        "ratio_other": dict(zip(("mean", "ci"), [ratio_ci2(err[~mono], perr[~mono], station[~mono])[0],
                                                 list(ratio_ci2(err[~mono], perr[~mono], station[~mono])[1])])),
        "n_other": int((~mono).sum()),
    }
    return report, {"station": station, "month": month, "err": err,
                    "perr": perr}


def declare_winner(rep_b, rep_c, cv):
    """Rule: C beats B only if spatial-CV CI(C/B) upper < 1 AND relative gain
    exceeds max(seed_spread) of the two arms' pooled RMSE (as a fraction)."""
    ci = cv["ratio_C_over_B"]["ci"]
    ci_excl = ci[1] < 1.0
    spread_rel = max(rep_b["seed_spread_rmse"] / rep_b["rmse_model"],
                     rep_c["seed_spread_rmse"] / rep_c["rmse_model"])
    gain_rel = 1.0 - cv["ratio_C_over_B"]["mean"]
    return {
        "ci_excludes_1": bool(ci_excl),
        "seed_spread_rel": round(float(spread_rel), 4),
        "gain_rel": round(float(gain_rel), 4),
        "c_beats_b_on_holdout": bool(ci_excl and gain_rel > spread_rel),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    anchors = (load_norm("gwl").drop_duplicates("Station")
               [["Station", "District", "Tehsil", "lat", "lon"]]
               .reset_index(drop=True))
    print(f"anchors: {len(anchors):,}", flush=True)

    ta = build_arm_a()
    print(f"arm A track: {len(ta):,} rows / {ta.Station.nunique():,} st", flush=True)
    tb = build_arm_b(ta, anchors)
    print(f"arm B track: {len(tb):,} rows / {tb.Station.nunique():,} st", flush=True)
    tc = build_arm_c(tb, anchors)
    print(f"arm C track: {len(tc):,} rows / {tc.Station.nunique():,} st", flush=True)

    xa = F.NUM_COLS + ["st_id", "dist_id"]
    xb = xa
    xc = F.NUM_COLS + [f"{k}_{s}" for k in WEATHER_EXTRAS for s in ("wsum", "dmin_km")] \
        + ["st_id", "dist_id"]
    print("xcols A: %d  B: %d  C: %d" % (len(xa), len(xb), len(xc)), flush=True)
    del ta

    prep = {}
    for name, track, xc2 in (("A", tb, xa), ("B", tb, xb), ("C", tc, xc)):
        print(f"prep {name} ...", flush=True)
        prep[name] = arm_prep(track, xc2)
        print(f"  train={len(prep[name]['Xtr']):,} val={len(prep[name]['Xva']):,} "
              f"te stride={prep[name]['te_station'].size:,} / "
              f"{np.unique(prep[name]['te_station']).size} st", flush=True)
        if name == "B":
            del tb
        elif name == "C":
            del tc

    reports = {}
    for name in ("A", "B", "C"):
        print(f"fit arm {name} (3 seeds) ...", flush=True)
        r, _ = run_arm(prep[name], name)
        reports[name] = r
        print(f"  {name}: rmse={r['rmse_model']:.4f} persist={r['rmse_persist']:.4f} "
              f"seeds={r['per_seed_rmse']} spread={r['seed_spread_rmse']:.4f}", flush=True)

    # two-fold spatial CV on the SAME station folds for B and C: fit on one
    # half of STATIONS, score on the other half's stride rows, then swap.
    print("spatial CV (2-fold, B vs C on held-out stations) ...", flush=True)
    stations = np.unique(np.concatenate([prep["B"]["te_station"], prep["C"]["te_station"]]))
    h = (pd.util.hash_pandas_object(pd.Series(stations)).to_numpy() & 1).astype(int)
    fold_held = {0: stations[h == 1].tolist(), 1: stations[h == 0].tolist()}
    eB, eC = [], []
    for fold, held in fold_held.items():
        fit_st = [s for s in stations if s not in set(held)]
        rB, aB = run_arm(prep["B"], "cvB%d" % fold, fit_stations=fit_st,
                         holdout_stations=held)
        rC, aC = run_arm(prep["C"], "cvC%d" % fold, fit_stations=fit_st,
                         holdout_stations=held)
        print(f"  fold {fold}: held-out stride B={rB['rmse_model']:.4f} "
              f"C={rC['rmse_model']:.4f} (n={rB['n_stride']})", flush=True)
        eB.append(aB); eC.append(aC)

    # pooled held-out ratio C/B (both folds, station bootstrap)
    st_all = np.concatenate([x["station"] for x in eB])
    erB = np.concatenate([x["err"] for x in eB])
    erC = np.concatenate([x["err"] for x in eC])
    rng = np.random.default_rng(99)
    stl = np.unique(st_all)
    vals = []
    for _ in range(2000):
        keep = np.isin(st_all, rng.choice(stl, stl.size, replace=True))
        vals.append(math.sqrt(np.mean(erC[keep] ** 2)) / math.sqrt(
            np.mean(erB[keep] ** 2)))
    vals = np.array(vals)
    cv = {
        "rmse_b_heldout": float(math.sqrt(np.mean(erB ** 2))),
        "rmse_c_heldout": float(math.sqrt(np.mean(erC ** 2))),
        "n_heldout_stride": int(st_all.size),
        "ratio_C_over_B": {"mean": float(np.mean(vals)),
                           "ci": (float(np.percentile(vals, 2.5)),
                                  float(np.percentile(vals, 97.5)))},
        "p_C_better": float(np.mean(vals < 1)),
    }
    decision = declare_winner(reports["B"], reports["C"], cv)

    report = {
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                 for k, v in reports.items()},
        "spatial_cv": cv,
        "decision": decision,
        "seeds": list(SEEDS),
        "note": "stride b) 30-day honest protocol; Jun-Sep = months 6..9 of the anchor "
                "date. ratio = RMSE(model)/RMSE(persistence), station bootstrap B=2000.",
        "elapsed_sec": round(time.time() - t0),
    }
    (OUT / "ablation_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== ABLATION REPORT ===")
    print(json.dumps({k: v for k, v in report.items() if k != "note"},
                     indent=2))
    for name in ("A", "B", "C"):
        r = reports[name]
        print(f"\n{name}: model={r['rmse_model']:.4f} persist={r['rmse_persist']:.4f} "
              f"ratio={r['ratio_all']['mean']:.4f} CI{tuple(round(x, 4) for x in r['ratio_all']['ci'])})")
        print(f"   Jun-Sep n={r['n_junsep']} ratio={r['ratio_junsep']['mean']:.4f} "
              f"CI{tuple(round(x, 4) for x in r['ratio_junsep']['ci'])}")
        print(f"   other   n={r['n_other']} ratio={r['ratio_other']['mean']:.4f} "
              f"CI{tuple(round(x, 4) for x in r['ratio_other']['ci'])}")


if __name__ == "__main__":
    sys.exit(main())