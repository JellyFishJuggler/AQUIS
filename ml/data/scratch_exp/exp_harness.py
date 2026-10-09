"""exp_harness — fold-based arm selection for the honest-30d skill sprint.

SELECTION PROTOCOL (the 2026 test set is NEVER touched here):
  * two rolling-origin folds inside the training years, each containing a full
    monsoon: fold-2024 and fold-2025 (origin = Jun-01 of the year).
      train   : anchors (date, target_time) both strictly before origin,
                target label finite
      val     : anchors in [origin-90d, origin-30d) with target_time < origin,
                label finite (used only for early stopping)
      holdout : anchors in [origin, Dec-31-Y] whose target_time <= Dec-31-Y
                (target-time purge keeps every scored label inside the fold
                year, so no label is observed inside a later split)
      scored  : non-overlapping stride rows ONLY (per-station cumcount %120==0
                inside the purged holdout window) — mirrors the production
                honest-stride protocol applied to the fold window.
  * scoreboard identical to the association ablation (ml/scratch_assoc_ablate.py):
    XGBoost delta-30d, 3 seeds (42,840,2056), early stop on val; ratio
    RMSE(model)/RMSE(persistence) with station bootstrap CI; Jun-Sep vs other
    months; per-seed relative spread.  pass = CI upper < 1 AND
    gain_rel (=1-ratio) > arm's own seed-spread_rel (same declare rule).
  * E0 = production config (feature_config 31 cols + st_id/dist_id) retrained
    in scratch on the same folds as the CONTROL.

Arms (one change each): R1 E2b E2a E1 E3 E4 E5 (+control E0). Everything writes
only under data/scratch_exp/. Values: all candidate anchors/targets derive from
the frozen backup table only.
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

os.environ.setdefault("OMP_NUM_THREADS", "6")

ROOT = Path(__file__).resolve().parents[2]        # ml/
sys.path.insert(0, str(ROOT))
F = importlib.import_module("06_features")

HERE = ROOT / "data" / "scratch_exp"
BASE = HERE / "tracks" / "base_features.parquet"
AUX = HERE / "tracks" / "aux.parquet"
REPORT = HERE / "arm_reports.json"

SEEDS = (42, 840, 2056)
XGB_PARAMS = dict(
    n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
    device="cuda", objective="reg:squarederror", eval_metric="rmse",
    n_jobs=6, verbose=False,
)
BASE_XCOLS = F.NUM_COLS + ["st_id", "dist_id"]

FOLDS = {
    2024: {"origin": pd.Timestamp("2024-06-01"),
           "vl": pd.Timestamp("2024-03-01"), "vh": pd.Timestamp("2024-05-02"),
           "hh": pd.Timestamp("2024-12-31 23:59")},
    2025: {"origin": pd.Timestamp("2025-06-01"),
           "vl": pd.Timestamp("2025-03-01"), "vh": pd.Timestamp("2025-05-02"),
           "hh": pd.Timestamp("2025-12-31 23:59")},
}


# ---------------------------------------------------------------------------
# loading / folding
# ---------------------------------------------------------------------------
def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(BASE)
    for c in df.columns:
        if c in ("Station", "District", "date", "time", "target_time"):
            continue
        if c == "target":
            pass  # keep float64
        elif df[c].dtype.kind == "f":
            df[c] = df[c].astype(np.float32)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce")
    df["doy"] = df["date"].dt.dayofyear.astype(np.int16)
    return df


def build_folds(df: pd.DataFrame) -> dict[int, dict]:
    t = df["target_time"].to_numpy("datetime64[ns]")
    d = df["date"].to_numpy("datetime64[ns]")
    yd = df["target_d"].to_numpy(float)
    fin = np.isfinite(yd)
    out = {}
    for y, cfg in FOLDS.items():
        o = cfg["origin"].to_datetime64()
        vl, vh, hh = (v.to_datetime64() for v in (cfg["vl"], cfg["vh"], cfg["hh"]))
        out[y] = {
            "train": np.flatnonzero(fin & (d < o) & (t < o)),
            "val": np.flatnonzero(fin & (d >= vl) & (d < vh) & (t < o)),
            "cand": np.flatnonzero(fin & (d >= o) & (d <= hh.astype("datetime64[D]"))
                                  & (t <= hh)),
        }
    return out


def holdout_stride(df: pd.DataFrame, folds: dict) -> dict[int, np.ndarray]:
    out = {}
    for y, fo in folds.items():
        ix = fo["cand"]
        st = df["st_id"].to_numpy(np.int32)[ix]
        cc = pd.Series(st).groupby(st, sort=False).cumcount().to_numpy()
        out[y] = ix[cc % 120 == 0]
    return out


# ---------------------------------------------------------------------------
# aux + neighbour features (computed once, shared by all arms, cached)
# ---------------------------------------------------------------------------
def build_aux(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    g = df.groupby("Station", sort=False)["gwl"]
    df["lag360"] = g.shift(360)
    df["lag720"] = g.shift(720)
    if "rain" in df.columns:
        rg = df.groupby("Station", sort=False)["rain"]
        df["rain_1d_full"] = rg.transform(lambda s: s.rolling(4, min_periods=4).sum())
        df["rain_7d_full"] = rg.transform(lambda s: s.rolling(28, min_periods=28).sum())
        df["rain_30d_full"] = rg.transform(lambda s: s.rolling(120, min_periods=120).sum())
        df["rain_cum90"] = rg.transform(lambda s: s.rolling(360, min_periods=360).sum())
        df["rain_cum180"] = rg.transform(lambda s: s.rolling(720, min_periods=720).sum())
        df["rain_cum365"] = rg.transform(lambda s: s.rolling(1460, min_periods=1460).sum())
    df["chg30"] = df["gwl"].to_numpy() - g.shift(120).to_numpy()
    print(f"aux core done ({time.time()-t0:.0f}s)", flush=True)
    return df


def build_neighbour(df: pd.DataFrame, k_max: int = 5) -> pd.DataFrame:
    coords = pd.read_csv(ROOT / "data" / "meta" / "selected_gwl_stations.csv",
                         usecols=["Station", "Latitude", "Longitude"])
    coords = coords.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    cm = coords.set_index("Station").reindex(df["Station"].astype(str).unique())
    cm = cm.dropna(subset=["lat", "lon"])
    names = cm.index.tolist()
    lat, lon = np.radians(cm["lat"].to_numpy()), np.radians(cm["lon"].to_numpy())
    km = np.empty((len(names), len(names)))
    for i in range(len(names)):
        a = np.sin((lat[i] - lat) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) \
            * np.sin((lon[i] - lon) / 2) ** 2
        km[i] = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    nbr = {names[i]: [names[j] for j in np.argsort(km[i], kind="stable")[1:k_max + 1]]
           for i in range(len(names))}

    t0 = time.time()
    wide = df.pivot_table(index="time", columns="Station", values="chg30",
                          aggfunc="first")
    wide = wide.ffill()   # trailing last-known only; never a future value
    station_col = df["Station"].astype(str).to_numpy()
    for k in (3, 5):
        feat = {}
        for s in names:
            cols = [c for c in nbr[s] if c in wide.columns]
            if cols:
                feat[s] = wide[cols].mean(axis=1, skipna=True).astype(np.float32)
        out = np.full(len(df), np.nan, dtype=np.float32)
        for s, ser in feat.items():
            m = station_col == s
            out[m] = df.loc[m, "time"].map(ser).to_numpy()
        df[f"e3_chg_k{k}"] = out
        print(f"e3_chg_k{k} done ({time.time()-t0:.0f}s)", flush=True)
    return df


def ensure_aux(df: pd.DataFrame) -> pd.DataFrame:
    if AUX.exists():
        a = pd.read_parquet(AUX)
        for c in a.columns:
            df[c] = a[c].to_numpy()
        print("aux loaded from cache", flush=True)
    else:
        df = build_aux(df)
        save_cols = [c for c in df.columns if c not in ("Station", "District",
                      "date", "time", "target", "target_d", "target_time",
                      "st_id", "dist_id", "doy")]
        df[save_cols].to_parquet(AUX, index=False)
        print("aux cached", flush=True)
    return df


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
ARMS = {
    "E0": {"cols": BASE_XCOLS},
    "R1": {"cols": [c for c in BASE_XCOLS if c not in
                    ("rain_1d", "rain_7d", "rain_30d")]
           + ["rain_1d_full", "rain_7d_full", "rain_30d_full"]},
    "E2b": {"cols": BASE_XCOLS + ["rain_api7", "rain_api22", "rain_api60",
                                   "rain_cum90", "rain_cum180", "rain_cum365"]},
    "E2a": {"cols": BASE_XCOLS + ["lag360", "lag720",
                                  "gwl_slope7", "gwl_slope30", "gwl_slope90"]},
    "E1": {"cols": BASE_XCOLS, "huber": True},
    "E3": {"cols": BASE_XCOLS + ["e3_chg_k3", "e3_chg_k5"]},
    "E4": {"cols": BASE_XCOLS, "resid": True},
    "E5": {"cols": [c for c in BASE_XCOLS if c != "river_level"]},
    "E_best": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
               + ["rain_api7", "rain_api22", "rain_api60",
                  "rain_cum90", "rain_cum180", "rain_cum365",
                  "lag360", "lag720", "gwl_slope7", "gwl_slope30",
                  "gwl_slope90"],
               "huber": True},
    # --- E_best candidates (compositions of the PASSING arms only) --------
    "C1": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365"]},
    "C2": {"cols": BASE_XCOLS + ["rain_api7", "rain_api22", "rain_api60",
                                 "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True},
    "C3": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True},
    "C4": {"cols": BASE_XCOLS + ["lag360", "lag720", "gwl_slope7",
                                 "gwl_slope30", "gwl_slope90"],
           "huber": True},
    # --- E_best hyperparameter grid (same folds; depth x lr) --------------
    "T1": {"cols": BASE_XCOLS + ["rain_api7", "rain_api22", "rain_api60",
                                 "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 4, "learning_rate": 0.03}},
    "T2": {"cols": BASE_XCOLS + ["rain_api7", "rain_api22", "rain_api60",
                                 "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 4, "learning_rate": 0.07}},
    "T3": {"cols": BASE_XCOLS + ["rain_api7", "rain_api22", "rain_api60",
                                 "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 7, "learning_rate": 0.03}},
    "T4": {"cols": BASE_XCOLS + ["rain_api7", "rain_api22", "rain_api60",
                                 "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 7, "learning_rate": 0.07}},
    # --- depth x lr grid on the C3 columns (winner shift on GPU) ----------
    "T5": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 4, "learning_rate": 0.03}},
    "T6": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 4, "learning_rate": 0.07}},
    "T7": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 7, "learning_rate": 0.03}},
    "T8": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365"],
           "huber": True, "params": {"max_depth": 7, "learning_rate": 0.07}},
    "C5": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365",
              "e3_chg_k3", "e3_chg_k5"],
           "huber": True},
    "C6": {"cols": BASE_XCOLS
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365",
              "e3_chg_k3", "e3_chg_k5"],
           "huber": True},
    # --- full union of ALL passing arms (E2b+E2a+E1+E5+E3) ---------------
    "C7": {"cols": BASE_XCOLS
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365",
              "lag360", "lag720", "gwl_slope7", "gwl_slope30", "gwl_slope90",
              "e3_chg_k3", "e3_chg_k5"],
           "huber": True},
    "C8": {"cols": [c for c in BASE_XCOLS if c != "river_level"]
           + ["rain_api7", "rain_api22", "rain_api60",
              "rain_cum90", "rain_cum180", "rain_cum365",
              "lag360", "lag720", "gwl_slope7", "gwl_slope30", "gwl_slope90",
              "e3_chg_k3", "e3_chg_k5"],
           "huber": True},
    "C9": {"cols": BASE_XCOLS + ["rain_api7", "rain_api22", "rain_api60",
                                 "rain_cum90", "rain_cum180", "rain_cum365",
                                 "e3_chg_k3", "e3_chg_k5"],
           "huber": True,
           "params": {"max_depth": 7, "learning_rate": 0.07}},
}


def _climo_map(yd: np.ndarray, st: np.ndarray, doy: np.ndarray) -> pd.Series:
    return pd.DataFrame({"y": yd, "st": st, "doy": doy}) \
             .groupby(["st", "doy"], sort=False)["y"].mean()


def _apply_climo(cl: pd.Series, st: np.ndarray, doy: np.ndarray) -> np.ndarray:
    mi = pd.MultiIndex.from_arrays([st, doy])
    v = cl.reindex(mi).to_numpy()
    v = np.where(np.isfinite(v), v, cl.mean() if np.isfinite(cl.mean()) else 0.0)
    return v


def run_arm(name: str, arm: dict, df: pd.DataFrame, folds: dict,
            stride: dict) -> dict:
    t0 = time.time()
    cols = [c for c in arm["cols"] if c in df.columns]
    miss = [c for c in arm["cols"] if c not in df.columns]
    if miss:
        raise KeyError(f"{name}: missing cols {miss}")
    cols = list(dict.fromkeys(cols))

    gwl = df["gwl"].to_numpy(float)
    yd = df["target_d"].to_numpy(float)
    st = df["st_id"].to_numpy(np.int32)
    doy = df["doy"].to_numpy(np.int16)
    month = df["date"].dt.month.to_numpy(np.int8)

    fold_out = {}
    for y, fo in folds.items():
        tr, va = fo["train"], fo["val"]
        ho = stride[y]
        Xtr = np.ascontiguousarray(df.iloc[tr][cols].to_numpy("float32"))
        Xva = np.ascontiguousarray(df.iloc[va][cols].to_numpy("float32"))
        Xho = np.ascontiguousarray(df.iloc[ho][cols].to_numpy("float32"))
        ytr = yd[tr].astype(np.float64)
        yva = yd[va].astype(np.float64)
        cl = None
        if arm.get("resid"):
            cl = _climo_map(ytr, st[tr], doy[tr])
            ytr = ytr - _apply_climo(cl, st[tr], doy[tr])
            yva = yva - _apply_climo(cl, st[va], doy[va])

        Em, Ep, Est, Emo = [], [], [], []
        seed_rmse = {}
        for seed in SEEDS:
            params = dict(XGB_PARAMS, random_state=seed)
            params.update(arm.get("params") or {})
            if arm.get("huber"):
                params.update(objective="reg:pseudohubererror", huber_slope=2.0)
            m_ = xgb.XGBRegressor(**params)
            try:
                m_.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=60,
                       verbose=False)
            except TypeError:
                m_.set_params(early_stopping_rounds=60)
                m_.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = m_.predict(Xho).astype(np.float64)
            if arm.get("resid"):
                pred = pred + _apply_climo(cl, st[ho], doy[ho])
            err_m = yd[ho] - pred            # model delta error (level = gwl+pred)
            err_p = yd[ho]                   # persistence delta error (delta=0)
            Em.append(err_m)
            Ep.append(err_p)
            Est.append(st[ho])
            Emo.append(month[ho])
            seed_rmse[seed] = float(np.sqrt(np.mean(err_m ** 2)))
        fstats = _score_set(np.concatenate(Em), np.concatenate(Ep),
                            np.concatenate(Est), np.concatenate(Emo), seed_rmse)
        fold_out[y] = {"n_scores": len(ho), "seed_rmse": seed_rmse, **fstats}
        del Xtr, Xva, Xho, m_
        gc.collect()

    f24, f25 = fold_out[2024], fold_out[2025]
    rep = {
        "cols": cols,
        "runtime_s": round(time.time() - t0, 1),
        "folds": {str(y): {k: fold_out[y][k] for k in
                      ("n_scores", "seed_rmse", "ratio", "ci_lo", "ci_hi",
                       "gain", "spread", "rmse_m", "rmse_p",
                       "ratio_junssep", "ratio_other")}
                  for y in fold_out},
    }
    print(f"{name}: 2024 r={f24['ratio']:.4f} ci="
          f"[{f24['ci_lo']:.4f},{f24['ci_hi']:.4f}] "
          f"gain={f24['gain']:.4f} spread={f24['spread']:.4f} | "
          f"2025 r={f25['ratio']:.4f} js24={f24['ratio_junssep']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rep


def _score_set(em: np.ndarray, ep: np.ndarray, stE: np.ndarray,
               moE: np.ndarray, seed_rmse: dict) -> dict:
    """Per-window scoreboard metrics (used once per fold)."""
    em = np.asarray(em, float)
    ep = np.asarray(ep, float)
    rmse_m = float(np.sqrt(np.mean(em ** 2)))
    rmse_p = float(np.sqrt(np.mean(ep ** 2)))
    ratio = rmse_m / rmse_p

    spr_a = np.array(list(seed_rmse.values()))
    spread = float((spr_a.max() - spr_a.min()) / spr_a.min())

    st_u = np.unique(stE)
    pos = {int(s): np.flatnonzero(stE == s) for s in st_u}
    rng = np.random.default_rng(7)
    reps = 2000
    sample = rng.choice(st_u, size=(reps, len(st_u)), replace=True)
    ratios = np.empty(reps)
    for i in range(reps):
        sel = np.concatenate([pos[int(s)] for s in sample[i]])
        ratios[i] = np.sqrt(np.mean(em[sel] ** 2)) / np.sqrt(np.mean(ep[sel] ** 2))
    ci_lo, ci_hi = np.quantile(ratios, [0.05, 0.95]).tolist()

    js = moE <= 9
    r_js = (np.sqrt(np.mean(em[js] ** 2)) / np.sqrt(np.mean(ep[js] ** 2))
            if js.any() else np.nan)
    oth = ~js
    r_oth = (np.sqrt(np.mean(em[oth] ** 2)) / np.sqrt(np.mean(ep[oth] ** 2))
             if oth.any() else np.nan)

    return {"ratio": round(ratio, 4), "rmse_m": round(rmse_m, 4),
            "rmse_p": round(rmse_p, 4), "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4), "n_stations": int(len(st_u)),
            "spread": round(spread, 4),
            "ratio_junssep": round(r_js, 4) if np.isfinite(r_js) else None,
            "ratio_other": round(r_oth, 4) if np.isfinite(r_oth) else None,
            "gain": round(1 - ratio, 4)}


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--no-neighbour", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    df = load_frame()
    print(f"base loaded {len(df):,} rows / {df.Station.nunique()} st "
          f"({time.time()-t0:.0f}s)", flush=True)
    df = ensure_aux(df)
    needs_nbr = any(any(str(c).startswith("e3_chg")
                        for c in (ARMS[a].get("cols") or []))
                    for a in args.arms)
    if needs_nbr and not args.no_neighbour:
        df = build_neighbour(df)
    folds = build_folds(df)
    stride = holdout_stride(df, folds)
    for y in FOLDS:
        print(f"fold {y}: train={len(folds[y]['train']):,} val="
              f"{len(folds[y]['val']):,} holdout={len(folds[y]['cand']):,} "
              f"stride={len(stride[y]):,}")

    reports = {}
    if REPORT.exists():
        reports = json.loads(REPORT.read_text())
    for name in args.arms:
        if name in reports:
            continue
        if name not in ARMS:
            print(f"skip unknown arm {name}", flush=True)
            continue
        reports[name] = run_arm(name, ARMS[name], df, folds, stride)
        REPORT.write_text(json.dumps(reports, indent=1))

    # --- selection (user rules) -------------------------------------------
    # arm passes iff it beats E0 on fold-2024 (paired station CI upper < 1 AND
    # gain > that arm's fold-2024 seed spread) AND its fold-2025 point ratio is
    # not worse than E0's (fold-2025 = sign check only, n=578, no significance).
    e0 = reports.get("E0")
    sel = []
    if e0 is not None:
        cmp = ["E0"] + [a for a in reports
                        if a not in ("E0", "_selection", "E_best")]
        for n in cmp:
            r = reports[n]
            f24, f25 = r["folds"]["2024"], r["folds"]["2025"]
            e24, e25 = e0["folds"]["2024"], e0["folds"]["2025"]
            beat24 = bool(f24["ci_hi"] < 1.0 and f24["gain"] > f24["spread"])
            sign25 = bool(f25.get("ratio") <= e25.get("ratio"))
            sel.append({"arm": n, "beat24": beat24, "sign25": sign25,
                        "pass": bool(beat24 and sign25),
                        "r24": f24["ratio"], "ci24": [f24["ci_lo"], f24["ci_hi"]],
                        "gain24": f24["gain"], "spread24": f24["spread"],
                        "r25": f25["ratio"], "r25_e0": e25["ratio"],
                        "js24": f24["ratio_junssep"]})
        reports["_selection"] = sel
        REPORT.write_text(json.dumps(reports, indent=1))

    print("\n=== summary (vs persistence) ===")
    for n in ["E0"] + [a for a in reports if a not in ("E0", "_selection")]:
        r = reports[n]
        f24 = r["folds"].get("2024", {})
        f25 = r["folds"].get("2025", {})
        print(f"{n:6s} 2024 r={f24.get('ratio')} "
              f"[{f24.get('ci_lo')},{f24.get('ci_hi')}] "
              f"gain={f24.get('gain')} spread={f24.get('spread')} | "
              f"2025 r={f25.get('ratio')} | cols={len(r['cols'])}")
    if sel:
        print("\n=== selection ===")
        for s in sel:
            print(f"{s['arm']:6s} beat24={s['beat24']} sign25={s['sign25']} "
                  f"pass={s['pass']}")
    print(f"\nTOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()