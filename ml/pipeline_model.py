"""Feature engineering, training, evaluation, experiments and fleet scan.

The feature contract (columns, delta target, purged split) is shared by batch
training and every serving path, so callers import this module directly.
Data fetching, alignment and the batch CLI live in ``pipeline_data``.

Run a stage with:  python -m pipeline_data <stage>
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _hydrology import (API_TAUS, DEV_WINS, gwl_imputed_flag, gwl_observed,
                        gwl_trend_block, rain_memory_block)
from static_loaders import (SOIL_COLS, EXTRACTION_COLS, LULC_COLS, load_soil,
                            load_extraction, load_lulc, extraction_year_grid)
from _thresholds import DECLINE_M, category

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
FEATS = ROOT / "data" / "features"
FEATS_6H = ROOT / "data" / "features_6h"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

XGB_PARAMS = dict(
    n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
    objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=6,
)
Q_XGB_PARAMS = dict(XGB_PARAMS, objective="reg:quantileerror")

HORIZONS = (30,)                       # days
STEPS_PER_DAY = 4                      # 6-hourly grid
HORIZON_STEPS = 30 * STEPS_PER_DAY
VAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-01-01")

NUM_COLS = [
    "gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
    "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
    "rain_1d", "rain_7d", "rain_30d",
    "temp", "temp_7d", "humidity", "humidity_7d", "solar", "wind_speed",
    "pressure", "river_level", "canal_level",
    "year", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "hour_sin", "hour_cos", "monsoon",
]
# Causal hydrology block: computed/carried in every frame but NOT in the frozen
# trained schema (enabling it requires a deliberate retrain + gate pass).
HYDRO_COLS = [
    "gwl_dev7", "gwl_dev30", "gwl_slope7", "gwl_slope30", "gwl_slope90",
    "gwl_obs7", "gwl_obs30", "gwl_obs90", "gwl_imputed",
    "rain_api7", "rain_api22", "rain_api60",
    "rain_obs7", "rain_obs22", "rain_obs60",
]
STATIC_SOIL_COLS = SOIL_COLS
ENABLE_SOIL_FEATURES = False
STATIC_EXTRACTION_COLS = EXTRACTION_COLS
STATIC_LULC_COLS = LULC_COLS
ENABLE_RAIN_EXP = False
RAIN_EXP_COL = "rain_exp_30d"
KEEP = ["Station", "District", "time", "date", "horizon", "target", "target_d",
        "target_time", "st_id", "dist_id"] + NUM_COLS + HYDRO_COLS

# ---------------------------------------------------------------- correlate-only
TARGET = "gwl"
RAIN_WINDOWS = (1, 7, 30)
ENG_COLS = [
    "gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
    "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
    "temp_7d", "humidity_7d", "year", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "hour_sin", "hour_cos", "monsoon",
]
LEVEL_COLS = ["temp", "humidity", "solar", "wind_speed", "pressure", "river_level",
              "canal_level", "canal_discharge", "res_okhla", "res_okhla_agra_canal",
              "res_matatila", "res_ganga_rmc", "res_ganga_1"]
DRIVER_COLS = ["rain_1d", "rain_7d", "rain_30d", "temp", "humidity", "solar",
               "wind_speed", "pressure", "river_level", "canal_level"]
TOLERANCES = {"within_025": 0.25, "within_05": 0.5, "within_10": 1.0}


# ============================================================ stage: features
def drop_gwl_spikes(tbl: pd.DataFrame, z: float = 30.0) -> pd.DataFrame:
    """Per-station robust spike mask (median + 30 * 1.4826 * MAD) -> NaN."""
    def mask(g: pd.Series) -> np.ndarray:
        med = g.median()
        mad = (g - med).abs().median() * 1.4826
        return np.abs(g - med) > z * max(mad, 1e-6)
    out = tbl.copy()
    out["_spike"] = out.groupby("Station")["gwl"].transform(mask)
    bad = int(out["_spike"].sum())
    out.loc[out["_spike"], "gwl"] = np.nan
    if bad:
        print(f"gwl spikes -> NaN: {bad:,} rows", flush=True)
    return out.drop(columns=["_spike"])


def flag_regime_shift(tbl: pd.DataFrame, max_delta: float = 15.0) -> list[str]:
    """Stations whose per-year median jumps >max_delta m from the 2021-2024
    baseline: persistent datum/telemetry error, so drop the well wholesale."""
    df = tbl[["Station", "date", "gwl"]].copy()
    df["year"] = df["date"].dt.year
    med = df.dropna(subset=["gwl"]).groupby(["Station", "year"])["gwl"].median().unstack()
    base_yrs = [y for y in range(2021, 2025) if y in med.columns]
    if not base_yrs:
        return []
    base = med[base_yrs].median(axis=1)
    dev = med.sub(base, axis=0).abs().dropna(axis=1, how="all")
    return sorted(str(s) for s in dev[dev.max(axis=1) > max_delta].index)


def build(tbl: pd.DataFrame, keep_na: bool = False,
          horizon_steps: int = HORIZON_STEPS, horizon_label: int = 30) -> pd.DataFrame:
    """One feature row per (station, 6h slot). Predicts the CHANGE, never the
    level: target_d = GWL(t+h) - GWL(t), anchor gwl stays a feature."""
    df = tbl.sort_values(["Station", "time"]).copy()
    df["date"] = df["time"].dt.normalize()
    g = df.groupby("Station", sort=False)["gwl"]
    for k, steps in (("lag1", 1), ("lag4", 4), ("lag8", 8),
                     ("lag28", 28), ("lag120", 120)):
        df[k] = g.shift(steps)
    for name, win in (("gwl_roll7", 28), ("gwl_roll30", 120)):
        df[f"{name}_mean"] = g.transform(lambda s: s.rolling(win, min_periods=1).mean())
        df[f"{name}_std"] = g.transform(lambda s: s.rolling(win, min_periods=1).std())
    if "rain" in df.columns:
        rg = df.groupby("Station", sort=False)["rain"]
        for name, win in (("rain_1d", 4), ("rain_7d", 28), ("rain_30d", 120)):
            df[name] = rg.transform(lambda s: s.rolling(win, min_periods=1).sum())
    for col, win in (("temp", 28), ("humidity", 28)):
        if col in df.columns:
            df[f"{col}_7d"] = df.groupby("Station", sort=False)[col].transform(
                lambda s: s.rolling(win, min_periods=1).mean())

    # Hydrology block shared verbatim with refresh/future_drivers via _hydrology.
    obs_flags = df["gwl_obs"] if "gwl_obs" in df.columns else None
    observed = gwl_observed(df["gwl"], obs_flags)
    df["gwl_imputed"] = gwl_imputed_flag(df["gwl"], obs_flags)
    for w in DEV_WINS:
        df[f"gwl_dev{w}"] = df["gwl"] - df[f"gwl_roll{w}_mean"]
    for c, values in gwl_trend_block(df["time"], df["gwl"], observed, df["Station"]).items():
        df[c] = values
    rain_cols = ([f"rain_api{int(t)}" for t in API_TAUS]
                 + [f"rain_obs{int(t)}" for t in API_TAUS])
    if "rain" in df.columns:
        # No rain imputation upstream: a finite value is a real observation.
        for _, idx in df.groupby("Station", sort=False).groups.items():
            vals = df.loc[idx, "rain"].to_numpy(dtype=np.float64)
            for c, a in rain_memory_block(vals, np.isfinite(vals)).items():
                df.loc[idx, c] = a
    else:
        for c in rain_cols:
            df[c] = np.nan

    t = df["time"]
    mo, doy = t.dt.month, t.dt.dayofyear
    hr = t.dt.hour + t.dt.minute / 60.0
    df["year"] = t.dt.year
    df["month_sin"] = np.sin(2 * np.pi * mo / 12)
    df["month_cos"] = np.cos(2 * np.pi * mo / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["monsoon"] = doy - 152

    df["target"] = g.shift(-horizon_steps)
    # Actual timestamp supplying the label, so split() purges by observation time.
    df["target_time"] = df.groupby("Station", sort=False)["time"].shift(-horizon_steps)
    df["horizon"] = horizon_label
    df["target_d"] = df["target"] - df["gwl"]
    if not keep_na:
        df = df.dropna(subset=["target", "gwl"])

    stations = sorted(df["Station"].astype(str).unique())
    districts = sorted(df["District"].astype(str).unique())
    df["st_id"] = df["Station"].astype(str).map({s: i for i, s in enumerate(stations)}).astype(np.int32)
    df["dist_id"] = df["District"].astype(str).map({d: i for i, d in enumerate(districts)}).astype(np.int32)
    for c in NUM_COLS + HYDRO_COLS:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    # Keep KEEP columns first and any computed extras after them (selecting
    # df[KEEP] would silently drop rain_api*/rain_obs*/gwl_imputed).
    ordered = [c for c in KEEP if c in df.columns]
    extras = [c for c in df.columns if c not in set(ordered)]
    return df[ordered + extras]


def rain_exp_30d(tbl: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Trailing-climate expected rain over the next 30 days per (District, date)."""
    t = tbl[["District", "time", "rain"]].copy()
    t["date"] = t["time"].dt.normalize()
    daily = t.groupby(["District", "date"])["rain"].sum().reset_index()
    daily["year"], daily["doy"] = daily["date"].dt.year, daily["date"].dt.dayofyear
    out = []
    for district, dd in daily.groupby("District"):
        piv = dd.pivot_table(index="doy", columns="year", values="rain", aggfunc="mean")
        years = sorted(int(c) for c in piv.columns)
        prior = {y: piv[[c for c in years if c < y]].mean(axis=1) for y in years}
        for _, row in dd.iterrows():
            pr = prior.get(int(row["year"]))
            if pr is None or pr.notna().sum() == 0:
                out.append((district, row["date"], np.nan))
                continue
            prof = pr.reindex(range(1, 367))
            total, ok = 0.0, True
            for k in range(1, 31):
                v = prof.get((int(row["doy"]) + k - 1) % 366 + 1, np.nan)
                if pd.isna(v):
                    ok = False
                    break
                total += v
            out.append((district, row["date"], total if ok else np.nan))
    exp = pd.DataFrame(out, columns=["District", "date", RAIN_EXP_COL])
    return feats.merge(exp, on=["District", "date"], how="left")


def build_full(tbl: pd.DataFrame, keep_na: bool = False,
               horizon_steps: int = HORIZON_STEPS,
               horizon_label: int = 30) -> tuple[pd.DataFrame, list[str]]:
    """build() + static soil/extraction; returns (features, extra_num_cols)."""
    if "date" not in tbl.columns:
        tbl = tbl.copy()
        tbl["date"] = tbl["time"].dt.normalize()
    bad = set(flag_regime_shift(tbl))
    if bad:
        tbl = tbl[~tbl["Station"].isin(bad)]
    feats = build(tbl, keep_na=keep_na,
                  horizon_steps=horizon_steps, horizon_label=horizon_label)
    extra: list[str] = []
    soil = load_soil()
    need = int(tbl["Station"].nunique()) if "Station" in tbl else 0
    if not soil.empty and soil["Station"].nunique() >= need:
        # Static soil is redundant with st_id ordinals (test 2.242 -> 2.277 m);
        # data stays available for the app, gated off by default.
        if ENABLE_SOIL_FEATURES:
            feats = feats.merge(soil[["Station"] + STATIC_SOIL_COLS],
                                on="Station", how="left")
            extra += list(STATIC_SOIL_COLS)
    else:
        have = soil["Station"].nunique() if not soil.empty else 0
        print(f"soil: SKIPPED (have {have}/{need} stations) — incomplete fetch", flush=True)
    assess = load_extraction()
    if not assess.empty:
        grid = extraction_year_grid(assess, feats["year"]).rename(
            columns={"district": "District", "year": "assess_year"})
        feats = feats.merge(grid[["District", "assess_year"] + STATIC_EXTRACTION_COLS],
                            left_on=["District", "year"],
                            right_on=["District", "assess_year"],
                            how="left").drop(columns=["assess_year"])
        extra += list(STATIC_EXTRACTION_COLS)
    if extra:
        for c in extra:
            feats[c] = feats[c].astype(np.float32)
    if ENABLE_RAIN_EXP:
        feats = rain_exp_30d(tbl, feats)
        feats[RAIN_EXP_COL] = feats[RAIN_EXP_COL].astype(np.float32)
        extra.append(RAIN_EXP_COL)
    return feats, extra


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Purged split by ANCHOR date AND label-observation time: train < Oct 2025,
    val = Oct-Dec 2025, test = 2026. Rows reaching the next boundary are purged."""
    d, tt = df["date"], df["target_time"]
    return {
        "train": df[(d < VAL_START) & (tt < VAL_START)],
        "val": df[(d >= VAL_START) & (d < TEST_START) & (tt < TEST_START)],
        "test": df[d >= TEST_START],
    }


def features_main() -> None:
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl["date"] = tbl["time"].dt.normalize()
    tbl = drop_gwl_spikes(tbl)
    reg = flag_regime_shift(tbl)
    if reg:
        print(f"regime-shift stations dropped: {len(reg)} -> {reg[:5]}...", flush=True)
        tbl = tbl[~tbl["Station"].isin(reg)]
    feats, extra = build_full(tbl)
    parts = split(feats)
    FEATS.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(FEATS / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows", flush=True)
    prep = {
        "horizons": list(HORIZONS), "steps_per_day": STEPS_PER_DAY,
        "horizon_steps": HORIZON_STEPS, "num_cols": NUM_COLS + extra,
        "hydro_num_cols": HYDRO_COLS,
        "split": split.__doc__.strip().split("\n")[0],
        "stations": sorted(feats["Station"].astype(str).unique(), key=str),
        "districts": sorted(feats["District"].astype(str).unique(), key=str),
        "extra_features": {
            "soil": STATIC_SOIL_COLS if "sand_0_30" in feats.columns else [],
            "extraction": STATIC_EXTRACTION_COLS if "ann_gw_draft_mcm" in feats.columns else [],
            "lulc": STATIC_LULC_COLS if "lulc_cropland_pct" in feats.columns else [],
            "hydrology": HYDRO_COLS,
        },
        "train_rows": int(len(parts["train"])), "val_rows": int(len(parts["val"])),
        "test_rows": int(len(parts["test"])),
        "built_at": str(pd.Timestamp.now())[:19],
    }
    (FEATS / "prep.json").write_text(json.dumps(prep, indent=2))
    base = math.sqrt(((parts["test"]["target"] - parts["test"]["gwl"]) ** 2).mean())
    print(f"persistence RMSE on test (30d, gwl as-is): {base:.3f} m", flush=True)


# =============================================================== stage: train
def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def train_main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    train = pd.read_parquet(FEATS / "train.parquet")
    val = pd.read_parquet(FEATS / "val.parquet")
    xcols = NUM + ["st_id", "dist_id"]
    # 6h grid but sample 1 slot/day per station: grid-derived values identical.
    train = train[train.groupby("Station")["time"].cumcount() % 4 == 0]
    Xtr, Xva = train[xcols], val[xcols]
    ytr_d, yva_d = train["target_d"].values, val["target_d"].values
    yva_lvl = val["target"].values

    def make_xgb(**kw):
        return xgb.XGBRegressor(**dict(XGB_PARAMS, **kw))

    try:
        dxgb = make_xgb()
        dxgb.fit(Xtr, ytr_d, eval_set=[(Xva, yva_d)], verbose=False,
                 early_stopping_rounds=60)
    except TypeError:
        dxgb = make_xgb(early_stopping_rounds=60)
        dxgb.fit(Xtr, ytr_d, eval_set=[(Xva, yva_d)], verbose=False)
    best = getattr(dxgb, "best_iteration", None)
    n_trees = best + 1 if best is not None else dxgb.n_estimators
    xgb_val_d = rmse(yva_d, dxgb.predict(Xva))
    xgb_val_lvl = rmse(yva_lvl, val["gwl"].values + dxgb.predict(Xva))
    print(f"XGBoost: best_iteration={n_trees} val_rmse(delta)={xgb_val_d:.4f} "
          f"val_rmse(level)={xgb_val_lvl:.4f} m")

    colt = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
         ["Station", "District"]),
    ], remainder="drop")
    pipe = Pipeline([("colt", colt), ("ridge", RidgeCV(alphas=(0.1, 1, 10, 100)))])
    pipe.fit(train[NUM + ["Station", "District"]], ytr_d)
    ridge_val_d = rmse(yva_d, pipe.predict(val[NUM + ["Station", "District"]]))
    ridge_val_lvl = rmse(yva_lvl, val["gwl"].values
                         + pipe.predict(val[NUM + ["Station", "District"]]))
    print(f"Ridge: alpha={pipe.named_steps['ridge'].alpha_} "
          f"val_rmse(delta)={ridge_val_d:.4f} val_rmse(level)={ridge_val_lvl:.4f} m")

    MODELS.mkdir(exist_ok=True)
    joblib.dump(dxgb, MODELS / "xgb_multihorizon.joblib")
    joblib.dump(pipe, MODELS / "linear_multihorizon.joblib")
    config = {
        "model_type": "pooled global model, single 30-day horizon (6h grid, 120 steps)",
        "target": "delta (GWL_{t+120} - GWL_t); level = anchor + delta; anchor never predicted",
        "horizons": prep["horizons"], "num_cols": NUM,
        "stations": prep["stations"], "districts": prep["districts"],
        "train_rows": prep["train_rows"], "val_rows": prep["val_rows"],
        "test_rows": prep["test_rows"],
        "xgb": {"n_trees": int(n_trees), "val_rmse_delta": round(xgb_val_d, 4),
                "val_rmse_level": round(xgb_val_lvl, 4)},
        "ridge": {"alpha": float(pipe.named_steps["ridge"].alpha_),
                  "val_rmse_delta": round(ridge_val_d, 4),
                  "val_rmse_level": round(ridge_val_lvl, 4)},
        "trained_at": str(pd.Timestamp.now())[:19],
    }
    (MODELS / "feature_config.json").write_text(json.dumps(config, indent=2))
    print(f"saved -> {MODELS}/xgb_multihorizon.joblib, linear_multihorizon.joblib, "
          "feature_config.json")
    _write_model_metadata(config, prep, xgb_val_lvl, ridge_val_lvl)


def _write_model_metadata(config: dict, prep: dict, xgb_val_lvl: float,
                          ridge_val_lvl: float) -> None:
    """Train-time facts + any existing validation artifacts, for deployed hosts."""
    meta: dict = {
        "generated_by": "06_train.py",  # kept for artifact compatibility
        "architecture": config["model_type"], "target": config["target"],
        "horizons": config["horizons"], "n_stations": len(prep.get("stations", [])),
        "n_districts": len(prep.get("districts", [])),
        "train_val_rows": config["train_rows"] + config["val_rows"],
        "test_rows": config["test_rows"],
        "val_rmse_level_m": {"xgb": float(xgb_val_lvl), "ridge": float(ridge_val_lvl)},
        "split": prep.get("split", {}), "trained_at": config["trained_at"],
        "generated_at": str(pd.Timestamp.now())[:19],
        "source_archive": "data/processed/common.parquet", "validated": {},
    }

    def _try(path: Path, section: str) -> None:
        if not path.exists():
            return
        try:
            if path.suffix == ".csv":
                meta["validated"][section] = pd.read_csv(path).head(20).to_dict("records")
            elif path.suffix == ".json":
                meta["validated"][section] = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001 - metadata is best-effort
            meta["validated"][section] = {"error": str(e)}

    for path, section in (
        (MODELS / "quantile_calibration.json", "quantile_calibration"),
        (ROOT / "outputs" / "honest_metrics.csv", "honest_metrics"),
        (ROOT / "outputs" / "spatial_cv_metrics.csv", "spatial_cv"),
        (ROOT / "outputs" / "spatial_cv_summary.json", "spatial_cv_summary"),
    ):
        _try(path, section)
    (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {MODELS}/model_metadata.json")


# ============================================================= stage: correlate
def add_windows(tbl: pd.DataFrame) -> pd.DataFrame:
    tbl = tbl.copy()
    for w in RAIN_WINDOWS:
        if "rain" in tbl.columns:
            tbl[f"rain_{w}d"] = (tbl.groupby("Station")["rain"].transform(
                lambda s: s.rolling(w).sum()) if w > 1 else tbl["rain"])
    tbl["doy"] = tbl["date"].dt.dayofyear
    return tbl


def deseason(tbl: pd.DataFrame, col: str) -> pd.Series:
    return tbl[col] - tbl.groupby(["Station", "doy"])[col].transform("mean")


def per_station_corr(tbl: pd.DataFrame, a: str, b: str, how: str) -> pd.DataFrame:
    def _f(g: pd.DataFrame) -> float:
        s = g[["_a", "_b"]].dropna()
        if len(s) < 20:
            return np.nan
        try:
            return s["_a"].corr(s["_b"], method=how)
        except Exception:  # noqa: BLE001
            return np.nan
    sub = tbl[["Station", a, b]].rename(columns={a: "_a", b: "_b"})
    out = sub.groupby("Station").apply(_f, include_groups=False).rename("corr").reset_index()
    out[f"n_{a}_{b}_n"] = sub.groupby("Station").apply(
        lambda g: int(g[["_a", "_b"]].dropna().shape[0]), include_groups=False).values
    return out


def per_station_corr_many(tbl: pd.DataFrame, target: str, cols: list[str],
                          how: str, min_rows: int = 20) -> pd.DataFrame:
    """Vectorised per-station correlation of one target with many drivers."""
    sub = tbl[["Station"] + cols].copy()
    sub[target] = sub[target].astype("float32")
    for c in cols:
        sub[c] = sub[c].astype("float32")
    feat = list(dict.fromkeys([target] + cols))
    mat = sub.groupby("Station")[feat].corr(method=how, min_periods=min_rows)
    return mat.xs(target, level=1)[cols].reset_index()


def lagged_med(tbl: pd.DataFrame, lead: int, min_rows: int = 180) -> float:
    def _f(g: pd.DataFrame) -> float:
        s = pd.DataFrame({"x": g["rain"].shift(lead), "y": g[TARGET]}).dropna()
        return s["y"].corr(s["x"], method="pearson") if len(s) >= min_rows else np.nan
    return float(tbl.groupby("Station").apply(_f, include_groups=False).median())


def correlate_main() -> None:
    tbl = add_windows(pd.read_parquet(config_aligned_daily()))
    tbl["date"] = pd.to_datetime(tbl["date"], errors="coerce")
    tbl = tbl.sort_values(["Station", "date"]).reset_index(drop=True)
    print(f"table: {len(tbl):,} rows / {tbl['Station'].nunique():,} stations", flush=True)
    report: list[dict] = []
    print(f"\ntarget: {TARGET}")
    skip = {"Station", "date", TARGET, "doy", "District", "Tehsil"}
    drivers = [c for c in tbl.columns if c not in skip
               and not c.endswith("_n") and not c.endswith("_dist_km")]
    print("drivers:", ", ".join(drivers), flush=True)
    for col in drivers:
        for how in ("spearman", "pearson"):
            r = per_station_corr(tbl, TARGET, col, how)
            med = r["corr"].median()
            report.append({"driver": col, "metric": how, "mode": "raw",
                           "corr": None if pd.isna(med) else round(float(med), 4),
                           "stations_ok": int(r["corr"].notna().sum())})
        if col in LEVEL_COLS or col.startswith("rain_"):
            sub = tbl.assign(_t=tbl.groupby("Station")[TARGET].diff(),
                             _d=tbl.groupby("Station")[col].diff())
            r = sub[["Station", "_t", "_d"]].groupby("Station").apply(
                lambda g: (g["_t"].corr(g["_d"], method="spearman")
                           if g[["_t", "_d"]].dropna().shape[0] >= 20 else np.nan),
                include_groups=False)
            med = r.median()
            report.append({"driver": col, "metric": "spearman", "mode": "diff",
                           "corr": None if pd.isna(med) else round(float(med), 4),
                           "stations_ok": int(r.notna().sum())})
        if col in LEVEL_COLS:
            sub = tbl.assign(_t=deseason(tbl, TARGET), _d=deseason(tbl, col))
            r = sub[["Station", "_t", "_d"]].groupby("Station").apply(
                lambda g: g["_t"].corr(g["_d"], method="spearman")
                if len(g.dropna()) >= 20 else np.nan, include_groups=False)
            med = r.median()
            report.append({"driver": col, "metric": "spearman", "mode": "deseason",
                           "corr": None if pd.isna(med) else round(float(med), 4),
                           "stations_ok": int(r.notna().sum())})

    # Engineered model features on the 6h grid, so the app's correlation matrix
    # and the model's importance chart line up feature-for-feature.
    print("\nbuilding 6h feature frame (same pipeline as the model)...", flush=True)
    t6 = pd.read_parquet(ALIGNED)
    t6["time"] = pd.to_datetime(t6["time"], errors="coerce")
    t6["date"] = t6["time"].dt.normalize()
    eng, _ = build_full(drop_gwl_spikes(t6))
    print(f"engineered frame: {len(eng):,} rows / {eng['Station'].nunique():,} stations",
          flush=True)
    for how in ("spearman", "pearson"):
        corr = per_station_corr_many(eng, TARGET, ENG_COLS, how)
        for col in ENG_COLS:
            med = corr[col].median()
            report.append({"driver": col, "metric": how, "mode": "raw",
                           "corr": None if pd.isna(med) else round(float(med), 4),
                           "stations_ok": int(corr[col].notna().sum())})
    out = pd.DataFrame(report).sort_values(["driver", "metric", "mode"]).reset_index(drop=True)
    out.to_csv(OUT / "correlation_report.csv", index=False)
    print(out.to_string(index=False), flush=True)

    dist_rows = []
    for a, b in (("rain_7d", "rain_30d"), ("temp", "river_level"),
                 ("humidity", "solar")):
        if a not in tbl or b not in tbl:
            continue
        for dist, g in tbl.groupby("District"):
            s = g[[a, b, TARGET]].dropna()
            if len(s) < 30:
                continue
            for col in (a, b):
                dist_rows.append({
                    "District": dist, "driver": col, "n": int(len(s)),
                    "spearman": round(float(s[TARGET].corr(s[col], method="spearman")), 4),
                    "pearson": round(float(s[TARGET].corr(s[col], method="pearson")), 4),
                })
    pd.DataFrame(dist_rows).to_csv(OUT / "correlation_by_district.csv", index=False)
    lags = list(range(0, 31))
    curves = pd.DataFrame({"lag": lags, "pearson_med": [lagged_med(tbl, k) for k in lags]})
    curves.to_csv(OUT / "lag_curves.csv", index=False)
    best = curves.iloc[curves["pearson_med"].abs().idxmax()]
    print(f"\nbest rain lag: {int(best['lag'])} days | "
          f"med corr {best['pearson_med']:.4f}", flush=True)
    _correlation_plots(curves, best, tbl)
    print("\nall reports saved under outputs/", flush=True)


def config_aligned_daily() -> Path:
    return ROOT / "data" / "aligned" / "table.parquet"


def _correlation_plots(curves: pd.DataFrame, best, tbl: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly unavailable — skipped plots", flush=True)
        return
    fig = go.Figure(go.Scatter(x=curves["lag"], y=curves["pearson_med"],
                               mode="lines+markers"))
    fig.add_vline(x=int(best["lag"]), line_dash="dot",
                  annotation_text=f"best lag {int(best['lag'])}d")
    fig.update_layout(title="GWL vs rainfall — median Pearson by lag (days)")
    fig.write_html(OUT / "lag_curve.html")
    top = list(tbl.dropna(subset=["rain_30d"]).groupby("Station").size()
               .sort_values(ascending=False).head(3).index)
    if not top:
        return
    figs = make_subplots(rows=len(top), cols=1, shared_xaxes=True, subplot_titles=top)
    for i, st in enumerate(top, 1):
        g = tbl[tbl["Station"] == st].dropna(subset=["rain_30d", TARGET])
        if g.empty:
            continue
        figs.add_trace(go.Scatter(x=g["date"], y=g[TARGET], name="GWL (m)"), row=i, col=1)
        figs.add_trace(go.Scatter(x=g["date"], y=g["rain_30d"],
                                  name="rain 30d (mm)", yaxis=f"y{i+1}"), row=i, col=1)
    figs.update_layout(height=350 * len(top))
    figs.write_html(OUT / "gwl_vs_rain_top.html")


# =============================================================== stage: evaluate
def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    true, pred = np.asarray(true, dtype="float64"), np.asarray(pred, dtype="float64")
    ok = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[ok], pred[ok]
    if true.size == 0:
        nan = np.nan
        return {"rmse": nan, "mae": nan, "r2": nan, "resid_std": nan,
                **{k: nan for k in TOLERANCES}, **{f"p{q}": nan for q in (5, 50, 95)}}
    err = pred - true
    ss_tot = float(np.sum((true - true.mean()) ** 2)) or np.nan
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot,
        "resid_std": float(err.std()),
    }
    out.update({k: float(np.mean(np.abs(err) <= tol)) for k, tol in TOLERANCES.items()})
    out.update({f"p{q}": float(np.percentile(err, q)) for q in (5, 50, 95)})
    return out


def add_metrics(rows: list[dict], label: str, h: int, true, pred, n_feat_days) -> None:
    rows.append({"model": label, "horizon": h, "rows": int(len(true)),
                 "feat_days": int(n_feat_days), **metrics(true, pred)})


def evaluate_main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    dxgb = joblib.load(MODELS / "xgb_multihorizon.joblib")
    pipe = joblib.load(MODELS / "linear_multihorizon.joblib")
    train = pd.read_parquet(FEATS / "train.parquet")
    te = pd.read_parquet(FEATS / "test.parquet").copy()
    tr = train.copy()
    tr["doy"] = (tr["date"] + pd.to_timedelta(tr["horizon"], unit="D")).dt.dayofyear
    clim = tr.groupby(["Station", "doy"])["target"].mean().rename("clim")
    stmean = tr.groupby("Station")["target"].mean().rename("stmean")
    te["doy"] = (te["date"] + pd.to_timedelta(te["horizon"], unit="D")).dt.dayofyear
    te = te.merge(clim.reset_index(), on=["Station", "doy"], how="left")
    te = te.merge(stmean.reset_index(), on="Station", how="left")
    pred_seas = te["clim"].fillna(te["stmean"]).values
    true, true_d = te["target"].values, te["target_d"].values
    xcols = NUM + ["st_id", "dist_id"]
    # Models were fit on delta; report on level = anchor + delta.
    pred_xgb = te["gwl"].values + dxgb.predict(te[xcols])
    pred_ridge = te["gwl"].values + pipe.predict(te[NUM + ["Station", "District"]])
    pred_persist = te["gwl"].values
    te["feat_days"] = te[DRIVER_COLS].notna().any(axis=1).astype(np.int8)

    rows: list[dict] = []
    for h in HORIZONS:
        m = te["horizon"] == h
        nfd = int(te.loc[m, "feat_days"].sum())
        preds = {"xgboost": pred_xgb, "ridge": pred_ridge,
                 "persistence": pred_persist, "climatology": pred_seas}
        for label, p in preds.items():
            add_metrics(rows, label, h, true[m], p[m], nfd)
        for label, p in preds.items():
            dm = metrics(true_d[m], (p - te["gwl"].values)[m])
            rows.append({"model": label + " (delta)", "horizon": h,
                         "rows": int(m.sum()), "feat_days": nfd, **dm})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "model_metrics.csv", index=False)
    print(metrics_df.round(3).to_string(index=False), flush=True)

    preds = {"xgboost": pred_xgb, "ridge": pred_ridge,
             "persistence": pred_persist, "climatology": pred_seas}
    dist_rows = []
    for (d, h), sub in te.groupby(["District", "horizon"]):
        for label, p in preds.items():
            r = metrics(sub["target"].values, p[sub.index])
            dist_rows.append({"district": d, "horizon": h, "model": label,
                              "rows": len(sub), "rmse": r["rmse"],
                              "mae": r["mae"], "within_05": r["within_05"]})
    pd.DataFrame(dist_rows).to_csv(OUT / "eval_by_district.csv", index=False)

    st_rows = []
    for (s_, h), sub in te.groupby(["Station", "horizon"]):
        row = {"station": s_, "horizon": h, "district": sub["District"].iloc[0],
               "rows": len(sub), "feat_days": int(sub["feat_days"].sum())}
        for label, p in preds.items():
            row[label + "_rmse"] = np.sqrt(np.mean(
                (sub["target"].values - p[sub.index]) ** 2))
        st_rows.append(row)
    st_df = pd.DataFrame(st_rows)
    st_df.to_csv(OUT / "eval_by_station.csv", index=False)
    print(f"driver-rich test stations (>=90 feature-days in 2026): "
          f"{int((st_df.groupby('station')['feat_days'].sum() >= 90).sum())}"
          f" / {st_df['station'].nunique()}")

    # Predictions for the app + honest validation; non-overlapping 30d stride.
    out = te[["Station", "District", "date", "time", "horizon", "target", "gwl",
              "feat_days"]].copy()
    out["xgb"], out["ridge"], out["persist"], out["clim"] = (
        pred_xgb, pred_ridge, pred_persist, pred_seas)
    out["err_xgb"], out["err_ridge"] = pred_xgb - true, pred_ridge - true
    try:
        qmods = {q: joblib.load(MODELS / f"xgb_{q}.joblib") for q in ("q05", "q50", "q95")}
        qcal = json.loads((MODELS / "quantile_calibration.json").read_text())
        qlev = {q: te["gwl"].values + m.predict(te[xcols]) for q, m in qmods.items()}
        kq = float(qcal.get("widen_factor_k", 1.0))
        med = qlev["q50"]
        lo, hi = np.minimum(qlev["q05"], med), np.maximum(qlev["q95"], med)
        out["q05_lvl"], out["q95_lvl"] = med - kq * (med - lo), med + kq * (hi - med)
        print("calibrated quantile test band added (q05_lvl/q95_lvl)")
    except Exception as e:  # noqa: BLE001 - quantile models are optional
        print(f"quantile band skipped: {e}")
    out = out.sort_values(["Station", "time"])
    idx = out.groupby("Station").cumcount()
    out["step_idx"] = idx
    out["window_id"] = idx // 120
    out["stride"] = idx % 120 == 0
    out.to_parquet(OUT / "predictions_2026.parquet", index=False)
    print(f"predictions -> outputs/predictions_2026.parquet  ({len(out):,} rows)")

    gain = dxgb.get_booster().get_score(importance_type="gain")
    freq = dxgb.get_booster().get_score(importance_type="weight")
    imp = pd.DataFrame({"feature": list(dxgb.feature_names_in_)})
    imp["gain"] = imp["feature"].map(lambda f: gain.get(f, 0.0))
    imp["freq"] = imp["feature"].map(lambda f: freq.get(f, 0))
    imp.sort_values("gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    names = pipe.named_steps["colt"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": np.asarray(pipe.named_steps["ridge"].coef_)})
    coefs["abs_coef"] = coefs["coef"].abs()
    coefs.to_csv(OUT / "feature_coefficients.csv", index=False)
    print(f"xgb feature names had no gain: {(imp['gain'] == 0).sum()}")

    mm = metrics_df.pivot(index="horizon", columns="model", values="rmse")
    print("\nRMSE improvement vs persistence (30 d):")
    for h in HORIZONS:
        for lbl in ("xgboost", "ridge"):
            pct = 100 * (mm.loc[h, "persistence"] - mm.loc[h, lbl]) / mm.loc[h, "persistence"]
            print(f"  h={h}: {lbl:8s} {mm.loc[h, lbl]:.3f} m  vs persist "
                  f"{mm.loc[h, 'persistence']:.3f} m  ({pct:+.1f}%)")

    # Overlap-window residual diagnostics (regenerates outputs/honest_metrics.csv,
    # /overlap.json, /residual_acf.json, /residual_spatial.json).
    for tool in ("overlap", "residual_acf", "spatial_residual"):
        script = ROOT / "validation" / f"{tool}.py"
        if not script.exists():
            continue
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        print(f"validation {tool}: " + ("ok" if r.returncode == 0 else
              f"FAILED\n{r.stdout[-800:]}\n{r.stderr[-800:]}"))


# ================================================ causal one-step (+6h) features
H6_STEPS, H6_LABEL = 1, 1          # +6h on the 6-hourly grid


def features6h_main() -> None:
    """Same causal features as the 30d model, one-step delta target."""
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl = drop_gwl_spikes(tbl)
    tbl["date"] = tbl["time"].dt.normalize()
    reg = flag_regime_shift(tbl)
    if reg:
        print(f"regime-shift stations dropped: {len(reg)} -> {reg[:5]}...", flush=True)
        tbl = tbl[~tbl["Station"].isin(reg)]
    feats, extra = build_full(tbl, keep_na=True,
                              horizon_steps=H6_STEPS, horizon_label=H6_LABEL)
    feats = feats.dropna(subset=["target", "gwl"])
    parts = split(feats)
    FEATS_6H.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part = part.sort_values(["Station", "time"]).reset_index(drop=True)
        part.to_parquet(FEATS_6H / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows", flush=True)
    (FEATS_6H / "prep_6h.json").write_text(json.dumps({
        "horizons": [H6_LABEL], "steps_per_day": STEPS_PER_DAY,
        "horizon_steps": H6_STEPS, "num_cols": NUM_COLS + extra,
        "stations": sorted(feats["Station"].astype(str).unique(), key=str),
        "districts": sorted(feats["District"].astype(str).unique(), key=str),
        "split": split.__doc__.strip().split("\n")[0],
        "train_rows": int(len(parts["train"])), "val_rows": int(len(parts["val"])),
        "test_rows": int(len(parts["test"])), "built_at": str(pd.Timestamp.now())[:19],
    }, indent=2))
    tst = parts["test"]
    base = math.sqrt(((tst["target"] - tst["gwl"]) ** 2).mean())
    print(f"one-step persistence RMSE on test (+6h, gwl as-is): {base:.3f} m", flush=True)


# ========================================================= drop-group ablation
ABLATE_GROUPS = {
    "rain": ["rain_1d", "rain_7d", "rain_30d"],
    "weather": ["temp", "temp_7d", "humidity", "humidity_7d", "solar",
                "wind_speed", "pressure"],
    "river_canal": ["river_level", "canal_level"],
    "calendar": ["year", "month_sin", "month_cos", "doy_sin", "doy_cos",
                 "hour_sin", "hour_cos", "monsoon"],
    "gwl_lag": ["lag1", "lag4", "lag8", "lag28", "lag120",
                "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std"],
}


def stride_rmse(df: pd.DataFrame, pred) -> float:
    d = df.sort_values(["Station", "time"]).reset_index(drop=True)
    stride = (d.groupby("Station").cumcount() % 120 == 0).values
    return rmse(d.loc[stride, "target"].values, np.asarray(pred)[stride])


def ablate_main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    train = pd.read_parquet(FEATS / "train.parquet")
    val = pd.read_parquet(FEATS / "val.parquet")
    test = pd.read_parquet(FEATS / "test.parquet")
    train_mask = train.groupby("Station")["time"].cumcount() % 4 == 0
    groups = {k: [c for c in cols if c in NUM] for k, cols in ABLATE_GROUPS.items()}
    configs: list[tuple[str, list[str]]] = [("baseline", NUM)]
    configs += [(f"drop_{k}", [c for c in NUM if c not in cols])
                for k, cols in groups.items() if cols]
    drivers = {c for grp in ("rain", "weather", "river_canal") for c in groups[grp]}
    configs.append(("no_drivers", [c for c in NUM if c not in drivers]))
    configs.append(("no_gwl_history", [c for c in NUM if c not in groups["gwl_lag"]]))

    rows: list[dict] = []
    for name, feats in configs:
        xcols = feats + ["st_id", "dist_id"]
        trf = train[train_mask]
        vf = val if len(val) else trf
        Xtr, ytr = trf[xcols], trf["target_d"].values
        Xva, yva = vf[xcols], vf["target_d"].values
        dxgb = xgb.XGBRegressor(**XGB_PARAMS)
        try:
            dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                     early_stopping_rounds=60)
        except TypeError:
            dxgb.set_params(early_stopping_rounds=60)
            dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = test["gwl"].values + dxgb.predict(test[xcols])
        err = pred - test["target"].values
        row = {
            "config": name, "num_feats": len(feats),
            "n_trees": int(getattr(dxgb, "best_iteration", 0) + 1),
            "val_rmse_delta": round(rmse(yva, dxgb.predict(Xva)), 4),
            "raw_rmse": round(rmse(test["target"].values, pred), 4),
            "raw_mae": round(float(np.mean(np.abs(err))), 4),
            "raw_r2": round(1 - np.sum(err ** 2) / float(
                np.sum((test["target"].values - test["target"].values.mean()) ** 2)), 4),
            "stride_rmse": round(stride_rmse(test, pred), 4),
        }
        rows.append(row)
        print(f"{name:16s} feats={row['num_feats']:2d}  trees={row['n_trees']:3d}  "
              f"raw_rmse={row['raw_rmse']:.3f}  stride_rmse={row['stride_rmse']:.3f}  "
              f"val_rmse={row['val_rmse_delta']:.3f}  r2={row['raw_r2']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation.csv", index=False)
    base = df.set_index("config").loc["baseline"]
    print("\n=== delta vs baseline (raw RMSE) ===")
    for _, r in df.iterrows():
        if r["config"] != "baseline":
            print(f"  {r['config']:16s}  {r['raw_rmse'] - base['raw_rmse']:+.4f}  m   "
                  f"({base['raw_rmse']:.3f} -> {r['raw_rmse']:.3f})")


# ============================================ pooled quantile XGBoost + calibration
QUANTILE_ALPHAS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
COVERAGE_TARGET = 0.80          # nominal 90% interval calibrated on non-overlap stride
COVERAGE_TARGET_RAW = 0.85


def q_coverage(true, lo, hi) -> float:
    inside = ((np.asarray(true) >= np.asarray(lo))
              & (np.asarray(true) <= np.asarray(hi)))
    return float(inside.mean())


def quantile_main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    xcols = list(prep["num_cols"]) + ["st_id", "dist_id"]
    train = pd.read_parquet(FEATS / "train.parquet")
    val = pd.read_parquet(FEATS / "val.parquet")
    test = pd.read_parquet(FEATS / "test.parquet")
    tr = train[train.groupby("Station")["time"].cumcount() % 4 == 0]
    Xtr, ytr = tr[xcols], tr["target_d"].values
    Xva, yva = val[xcols], val["target_d"].values

    test = test.sort_values(["Station", "time"]).reset_index(drop=True)
    stride = (test.groupby("Station").cumcount() % 120 == 0).values
    Xte, true_lvl, anchor = test[xcols], test["target"].values, test["gwl"].values

    pred_d: dict[str, np.ndarray] = {}
    for name, alpha in QUANTILE_ALPHAS.items():
        m = xgb.XGBRegressor(**Q_XGB_PARAMS, quantile_alpha=alpha)
        try:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=60)
        except TypeError:
            m.set_params(early_stopping_rounds=60)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred_d[name] = m.predict(Xte)
        joblib.dump(m, MODELS / f"xgb_{name}.joblib")
        print(f"{name}: trained  best_iter={int(getattr(m, 'best_iteration', 0) + 1)}")

    lo_raw, hi_raw = anchor + pred_d["q05"], anchor + pred_d["q95"]
    med = anchor + pred_d["q50"]
    k = 1.0
    for _ in range(60):
        lo = med - k * (med - lo_raw)
        hi = med + k * (hi_raw - med)
        if q_coverage(true_lvl[stride], lo[stride], hi[stride]) >= COVERAGE_TARGET:
            break
        k += 0.1
    lo, hi = med - k * (med - lo_raw), med + k * (hi_raw - med)
    half_width = (hi - lo) / 2
    report = {
        "quantiles": list(QUANTILE_ALPHAS), "target_coverage_stride": COVERAGE_TARGET,
        "widen_factor_k": round(k, 2),
        "coverage_stride": round(q_coverage(true_lvl[stride], lo[stride], hi[stride]), 4),
        "coverage_raw": round(q_coverage(true_lvl, lo, hi), 4),
        "coverage_stride_raw_q": round(
            q_coverage(true_lvl[stride], lo_raw[stride], hi_raw[stride]), 4),
        "half_width_median_m": round(float(np.median(half_width)), 3),
        "half_width_p90_m": round(float(np.percentile(half_width, 90)), 3),
        "half_width_mean_m": round(float(np.mean(half_width)), 3),
        "anchor_basis": "level = today's GWL + quantile(delta 30d)",
    }
    (MODELS / "quantile_calibration.json").write_text(json.dumps(report, indent=2))
    print("\n=== calibration ===")
    print(json.dumps(report, indent=2))


# ============================================== VIF / permutation / OAT diagnostics
DIAG_REPEATS, DIAG_SEED = 5, 42


def vif_report(X: np.ndarray, feature_cols: list[str]) -> dict:
    """VIF per feature; constant columns cannot have a VIF and are excluded."""
    X = np.asarray(X, float)
    var = X.var(axis=0)
    const_ix = [i for i in range(X.shape[1]) if var[i] <= 1e-12]
    keep = [i for i in range(X.shape[1]) if i not in const_ix]
    Xc = X[:, keep] - X[:, keep].mean(axis=0)
    p = len(keep)
    corr = np.corrcoef(Xc.T) if p > 1 else np.array([[1.0]])
    corr = corr + np.eye(p) * 1e-9
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    vifs = np.diag(inv)
    out = {feature_cols[i]: (float("nan") if i in const_ix
                            else float(vifs[keep.index(i)])) for i in range(X.shape[1])}
    return {
        "feature": list(out), "vif": [out[f] for f in out],
        "constant_excluded": [feature_cols[i] for i in const_ix],
        "highly_correlated_gt10": [f for f, v in out.items() if v == v and v > 10],
    }


def permutation_report(model, X: np.ndarray, y: np.ndarray,
                       feature_cols: list[str]) -> list[dict]:
    from sklearn.inspection import permutation_importance

    res = permutation_importance(model, X, y, scoring="neg_root_mean_squared_error",
                                 n_repeats=DIAG_REPEATS, random_state=DIAG_SEED, n_jobs=1)
    return [{"rank": r + 1, "feature": feature_cols[i],
             "importance_mean": float(res.importances_mean[i]),
             "importance_std": float(res.importances_std[i])}
            for r, i in enumerate(np.argsort(res.importances_mean)[::-1])]


def sensitivity_report(model, feature_cols: list[str], X: np.ndarray,
                       n_top: int = 8) -> dict:
    """One-at-a-time 2.5->97.5 percentile swing of the forecast change."""
    top = np.argsort(np.asarray(model.feature_importances_))[::-1][:n_top]
    lo = np.nanpercentile(X[:, top], 2.5, axis=0)
    hi = np.nanpercentile(X[:, top], 97.5, axis=0)
    base = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    swing = []
    for i, j in enumerate(top):
        xl, xh = base.copy(), base.copy()
        xl[:, j], xh[:, j] = lo[i], hi[i]
        swing.append(float(np.mean(np.abs(model.predict(xh) - model.predict(xl)))))
    return {"features": [feature_cols[i] for i in top],
            "swing_mean_m": swing, "n_rows": int(len(base))}


def diagnostics_main() -> None:
    from serving_core import load_feature_config

    t0 = time.time()
    xcols = load_feature_config()["num_cols"] + ["st_id", "dist_id"]
    test = pd.read_parquet(FEATS / "test.parquet")
    stride = (test.groupby("Station").cumcount() % 120 == 0).values
    X = test.loc[stride, xcols].to_numpy(float)
    y = test.loc[stride, "target"].to_numpy(float)
    print(f"stride-subset for diagnostics: {len(y):,} rows ({len(test):,} raw)")

    model = joblib.load(MODELS / "xgb_multihorizon.joblib")
    fnames = list(getattr(model, "feature_names_in_", np.asarray(xcols)))
    col_ix = [fnames.index(c) for c in fnames]
    X = X[:, col_ix]
    perm = permutation_report(model, X, y, fnames)
    pd.DataFrame(perm).to_csv(OUT / "diagnostics_permutation.csv", index=False)
    print("permutation importance (stride RMSE, mean patience per feature):")
    for p in perm[:10]:
        print(f"  {p['rank']:>2}. {p['feature']:<20} {p['importance_mean']:+.4f} "
              f"(+/-{p['importance_std']:.4f})")

    full = test.loc[:, xcols].to_numpy(float)
    rng = np.random.default_rng(DIAG_SEED)
    samp = full[rng.choice(len(full), size=min(50_000, len(full)), replace=False)].astype(float)
    nan = np.isnan(samp)
    samp[nan] = np.take(np.nanmean(samp, axis=0), np.nonzero(nan)[1])
    vif = vif_report(samp[:, col_ix], fnames)
    print(f"\nVIF (mean-imputed sample, {len(samp):,} rows) > 10 (collinear): "
          f"{vif['highly_correlated_gt10'] or 'none'}")
    if vif["constant_excluded"]:
        print("VIF constant-excluded:", vif["constant_excluded"])
    sens = sensitivity_report(model, fnames, X)
    print("\nOAT sensitivity (mean |Δ 30-day change|, 2.5→97.5 pct):")
    for f, s in zip(sens["features"], sens["swing_mean_m"]):
        print(f"  {f:<20} {s:.3f} m")

    (OUT / "diagnostics.json").write_text(json.dumps({
        "model": "xgb_multihorizon (pooled, delta 30d)",
        "basis": "stride (non-overlap) test subset", "n_stride_rows": int(len(y)),
        "permutation": perm, "vif": vif, "sensitivity": sens,
        "generated_at": str(pd.Timestamp.now())[:19], "seconds": round(time.time() - t0, 1),
    }, indent=1))
    print(f"\ndone in {time.time()-t0:.1f}s -> outputs/diagnostics.json "
          "+ diagnostics_permutation.csv")


# ============================================================ fleet scan artifacts
FLEET_SNAP = OUT / "fleet_forecast_snapshot.json"
FLEET_CSV = OUT / "fleet_forecast.csv"
FLEET_DIST_CSV = OUT / "fleet_district.csv"
FLEET_MIN_OBS, FLEET_MIN_SPAN_DAYS, FLEET_HORIZON = 2000, 730, 30


def fleet_backends() -> dict:
    try:
        from serving_core import traj_backend
        traj = traj_backend()
    except Exception:  # noqa: BLE001 - metadata only
        traj = None
    try:
        from serving_core import forward_backend
        fwd = forward_backend()
    except Exception:  # noqa: BLE001 - metadata only
        fwd = None
    return {"traj_backend": traj, "forward_backend": fwd}


def fleet_write_snapshot(done: dict[str, dict], stations: list[str]) -> None:
    FLEET_SNAP.write_text(json.dumps({
        "horizon_days": FLEET_HORIZON, "decline_threshold_m": DECLINE_M,
        "eligible_stations": len(stations),
        "scanned_stations": sum(1 for s in done.values() if "error" not in s),
        "generated_at": str(pd.Timestamp.now())[:19],
        **fleet_backends(),
        "stations": [done[s] for s in stations if s in done],
    }, indent=1))


def fleet_write_csvs(done: dict[str, dict]) -> None:
    rows = [d for d in done.values() if "error" not in d]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(FLEET_CSV, index=False)
    good = df[df["category"] != "unreliable"]
    dist = (good.groupby("district")
            .agg(n=("station", "count"),
                 n_decline=("change_30d_m", lambda x: int((x <= -DECLINE_M).sum())),
                 n_recover=("change_30d_m", lambda x: int((x >= DECLINE_M).sum())),
                 median_change_m=("change_30d_m", "median"))
            .sort_values("n_decline", ascending=False))
    dist["decline_share"] = (dist["n_decline"] / dist["n"]).round(3)
    dist.to_csv(FLEET_DIST_CSV)
    n_decline = int((good["change_30d_m"] <= -DECLINE_M).sum())
    n_recover = int((good["change_30d_m"] >= DECLINE_M).sum())
    n_bad = int((df["category"] == "unreliable").sum())
    print(f"\nfleet: {len(good)} scored (excl {n_bad} unreliable) | "
          f"decline {n_decline} | recovering {n_recover} | stable "
          f"{len(good)-n_decline-n_recover}")
    print("worst 8 (30d change):")
    print(good.nsmallest(8, "change_30d_m")[["station", "district", "change_30d_m", "category"]]
          .to_string(index=False))


def fleet_main() -> None:
    ap = argparse.ArgumentParser(prog="pipeline_data fleet", description="Fleet 30d scan")
    ap.add_argument("--limit", type=int, default=0, help="max stations (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip first N stations")
    ap.add_argument("--force", action="store_true", help="ignore existing snapshot")
    args = ap.parse_args()

    import serving_core as serving
    from serving_core import active_trajectory_forecast, load_table_6h

    table6 = load_table_6h()
    obs_range = table6.groupby("Station")["gwl"].agg(omin="min", omax="max")
    g = table6.groupby("Station").agg(
        n=("gwl", "count"), first=("time", "min"), last=("time", "max")).reset_index()
    g["span_days"] = (g["last"] - g["first"]).dt.days
    eligible = g[(g["n"] >= FLEET_MIN_OBS) & (g["span_days"] >= FLEET_MIN_SPAN_DAYS)]
    stations = eligible.sort_values("n", ascending=False)["Station"].tolist()

    done: dict[str, dict] = {}
    if not args.force and FLEET_SNAP.exists():
        try:
            done = {s["station"]: s for s in json.loads(FLEET_SNAP.read_text())["stations"]}
            print(f"resume: {len(done)} stations already in snapshot")
        except Exception as e:  # noqa: BLE001
            print(f"snapshot unreadable, starting fresh: {e}")

    todo = [s for s in stations[args.start:args.start + (args.limit or len(stations))]
            if s not in done]
    print(f"eligible {len(stations)} stations — scanning {len(todo)}")
    t0, now, pooled_failures = time.time(), pd.Timestamp.now(), 0
    for i, station in enumerate(todo, 1):
        traj_out = active_trajectory_forecast(station) or {}
        points = traj_out.get("trajectory") or []
        if not points:
            done[station] = {"station": station, "district": None,
                             "error": traj_out.get("error", "no trajectory")}
        else:
            end = points[-1]
            anchor = float(traj_out["anchor_gwl"])
            point_level = float(end["q50"])   # legacy schema name xgb_level
            q05, q95 = float(end["q05"]), float(end["q95"])
            band_half = (q95 - q05) / 2.0
            chg = point_level - anchor
            anchor_valid = bool(np.isfinite(anchor) and np.isfinite(point_level))
            anchor_time = pd.Timestamp(traj_out["anchor_time"])
            age_days = int((now - anchor_time).days) if pd.notna(anchor_time) else None
            ore = obs_range.loc[station] if station in obs_range.index else (np.nan, np.nan)
            olo, ohi = float(ore["omin"]), float(ore["omax"])
            plausible = (anchor_valid and abs(chg) <= 12.0
                         and (not np.isfinite(olo) or (olo - 25.0 <= point_level <= ohi + 25.0))
                         and (age_days is None or age_days <= 45))
            chg_pooled = None
            try:
                v = (serving.forward_forecast(station) or {}).get("pred_xgb")
                if v is not None and np.isfinite(v):
                    chg_pooled = float(v)
            except Exception as e:  # noqa: BLE001 - pooled view is informational
                pooled_failures += 1
                print(f"  pooled view unavailable for {station}: "
                      f"{type(e).__name__}: {e}", flush=True)
            sub = table6.loc[table6["Station"] == station, "District"]
            done[station] = {
                "station": station, "district": str(sub.iloc[0]) if len(sub) else None,
                "anchor": round(anchor, 3) if np.isfinite(anchor) else None,
                "date_from": str(traj_out["anchor_time"]), "age_days": age_days,
                "anchor_valid": anchor_valid,
                "xgb_level": round(point_level, 3) if np.isfinite(point_level) else None,
                "q05_level": round(q05, 3) if np.isfinite(q05) else None,
                "q50_level": round(point_level, 3) if np.isfinite(point_level) else None,
                "q95_level": round(q95, 3) if np.isfinite(q95) else None,
                "change_30d_m": round(chg, 3) if np.isfinite(chg) else None,
                "change_pooled_m": round(chg_pooled, 3) if chg_pooled is not None else None,
                "band_half_m": round(band_half, 3) if np.isfinite(band_half) else None,
                "plausible": bool(plausible),
                "category": category(chg, anchor_valid=anchor_valid, plausible=plausible),
            }
        if i % 25 == 0 or i == len(todo):
            fleet_write_snapshot(done, stations)
            print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    fleet_write_snapshot(done, stations)
    fleet_write_csvs(done)
    print(f"scan complete: {len(done)} stations in {time.time()-t0:.0f}s "
          f"| pooled view missing for {pooled_failures}")
