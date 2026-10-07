"""06_features — feature/target engineering on the 6-hourly track.

One feature row per (station, 6h-slot); the ONLY forecast horizon is 30 days =
120 six-hour steps ahead (production-consistent cadence, see config).
'gwl' (the current reading) is the anchor feature — it is NEVER predicted; the
predicted quantity is the 30-day change GWL(t+120) - GWL(t).

The frame carries the causal hydrology-informed block (HYDRO_COLS, built by
_hydrology and shared with the refresh/serving path) plus ``target_time`` —
the timestamp of the row that supplies the label — so split() can purge any
row whose target is observed inside a later split (see split()).

Outputs -> ml/data/features/{train,val,test}.parquet + prep.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _hydrology import (API_TAUS, DEV_WINS, gwl_imputed_flag, gwl_observed,
                        gwl_trend_block, rain_memory_block)
from _soil import SOIL_COLS, EXTRACTION_COLS, load_soil, load_extraction, extraction_year_grid
from _lulc import LULC_COLS, load_lulc

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
FEATS = ROOT / "data" / "features"

HORIZONS = (30,)            # days
STEPS_PER_DAY = 4           # 6-hourly grid
HORIZON_STEPS = 30 * STEPS_PER_DAY

# time-split boundaries (anchor time); split() additionally purges by target time
VAL_START = pd.Timestamp("2025-10-01")    # validation begins
TEST_START = pd.Timestamp("2026-01-01")   # test begins

NUM_COLS = [
    # day-0 anchor (feature only, never a target)
    "gwl",
    # GWL memory in 6h steps: 1=6h, 4=1d, 8=2d, 28=7d, 120=30d
    "lag1", "lag4", "lag8", "lag28", "lag120",
    "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
    # drivers on the 6h grid
    "rain_1d", "rain_7d", "rain_30d",
    "temp", "temp_7d", "humidity", "humidity_7d", "solar", "wind_speed",
    "pressure", "river_level", "canal_level",
    # calendar / diurnal on a 6h grid
    "year", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "hour_sin", "hour_cos", "monsoon",
]

# hydrology-informed block (causal; built by _hydrology, carried in every
# feature frame — batch training, backtesting and serving alike).
# NOT part of the trained model schema yet: wiring them into NUM_COLS requires
# a full retrain of the frozen artifacts (out of scope by design), and the
# refresh gate refuses any candidate whose feature_cols differ from production.
# When that retrain happens, append HYDRO_COLS to the trained num_cols.
HYDRO_COLS = [
    # groundwater state / trend
    "gwl_dev7", "gwl_dev30",                      # deviation from trailing mean (m)
    "gwl_slope7", "gwl_slope30", "gwl_slope90",   # Theil-Sen slope (m/day)
    "gwl_obs7", "gwl_obs30", "gwl_obs90",         # observed-day coverage (0..1)
    "gwl_imputed",                                # 1 = forward-filled reading
    # rainfall memory (exponential API, 6h decay per tau)
    "rain_api7", "rain_api22", "rain_api60",      # mm — numeric forcing (incl.
                                                  # future forecast/climatology
                                                  # in the serving path)
    "rain_obs7", "rain_obs22", "rain_obs60",      # fraction of the tau window
                                                  # backed by an ACTUAL rain
                                                  # observation (0..1); forecast/
                                                  # climatology never counts
]

# static per-station soil texture (added in build_full when available)
STATIC_SOIL_COLS = SOIL_COLS
# experiment gate — static soil is redundant with st_id ordinals (see build_full)
ENABLE_SOIL_FEATURES = False
# district-year CGWB extraction (added in build_full when available)
STATIC_EXTRACTION_COLS = EXTRACTION_COLS
# static per-district ISRO Bhuvan LULC land-cover shares (added in build_full)
STATIC_LULC_COLS = LULC_COLS

# experiment gate — future-30d rain expectation from a TRAILING district-day
# climatology of observed NWIC rainfall (no future leakage). This measures the
# ceiling of any "expected rain next month" feature (CFS being its noisy proxy).
ENABLE_RAIN_EXP = False
RAIN_EXP_COL = "rain_exp_30d"

KEEP = ["Station", "District", "time", "date", "horizon", "target", "target_d",
        "target_time", "st_id", "dist_id"] + NUM_COLS + HYDRO_COLS


def drop_gwl_spikes(tbl: pd.DataFrame, z: float = 30.0) -> pd.DataFrame:
    """Per-station robust spike mask (median + 30*MAD)."""
    def mask(g: pd.Series) -> np.ndarray:
        med = g.median()
        mad = (g - med).abs().median() * 1.4826
        return np.abs(g - med) > z * max(mad, 1e-6)
    out = tbl.copy()
    out["_spike"] = out.groupby("Station")["gwl"].transform(mask)
    bad = int(out["_spike"].sum())
    out.loc[out["_spike"], "gwl"] = np.nan
    out = out.drop(columns=["_spike"])
    if bad:
        print(f"gwl spikes -> NaN: {bad:,} rows", flush=True)
    return out


def flag_regime_shift(tbl: pd.DataFrame, max_delta: float = 15.0) -> list[str]:
    """Stations whose per-year median GWL jumps >max_delta m from the 2021-2024
    baseline. Such persistent offsets are telemetry datum errors (same well
    switching to ~-95 m overnight), so the station is dropped wholesale."""
    df = tbl[["Station", "date", "gwl"]].copy()
    df["year"] = df["date"].dt.year
    med = df.dropna(subset=["gwl"]).groupby(["Station", "year"])["gwl"].median().unstack()
    base_yrs = [y for y in range(2021, 2025) if y in med.columns]
    if not base_yrs:
        return []
    base = med[base_yrs].median(axis=1)
    dev = med.sub(base, axis=0).abs().dropna(axis=1, how="all")
    bad = dev[dev.max(axis=1) > max_delta].index.tolist()
    return sorted(str(s) for s in bad)


def build(tbl: pd.DataFrame, keep_na: bool = False,
          horizon_steps: int = HORIZON_STEPS, horizon_label: int = 30) -> pd.DataFrame:
    df = tbl.sort_values(["Station", "time"]).copy()
    df["date"] = df["time"].dt.normalize()

    # --- GWL memory (6h steps) ---
    g = df.groupby("Station", sort=False)["gwl"]
    for k, steps in (("lag1", 1), ("lag4", 4), ("lag8", 8), ("lag28", 28), ("lag120", 120)):
        df[k] = g.shift(steps)
    for name, win in (("gwl_roll7", 28), ("gwl_roll30", 120)):
        df[f"{name}_mean"] = g.transform(lambda s: s.rolling(win, min_periods=1).mean())
        df[f"{name}_std"] = g.transform(lambda s: s.rolling(win, min_periods=1).std())

    # --- driver windows (6h sums/means over 1/7/30 days) ---
    if "rain" in df.columns:
        rg = df.groupby("Station", sort=False)["rain"]
        df["rain_1d"] = rg.transform(lambda s: s.rolling(4, min_periods=1).sum())
        df["rain_7d"] = rg.transform(lambda s: s.rolling(28, min_periods=1).sum())
        df["rain_30d"] = rg.transform(lambda s: s.rolling(120, min_periods=1).sum())
    for col, win in (("temp", 28), ("humidity", 28)):
        if col in df.columns:
            df[f"{col}_7d"] = df.groupby("Station", sort=False)[col].transform(
                lambda s: s.rolling(win, min_periods=1).mean())

    # --- hydrology-informed block (causal; shared with refresh/future_drivers
    #     via _hydrology so batch training and serving stay identical) ---
    obs_flags = df["gwl_obs"] if "gwl_obs" in df.columns else None
    observed = gwl_observed(df["gwl"], obs_flags)
    df["gwl_imputed"] = gwl_imputed_flag(df["gwl"], obs_flags)
    for w in DEV_WINS:
        df[f"gwl_dev{w}"] = df["gwl"] - df[f"gwl_roll{w}_mean"]
    trend = gwl_trend_block(df["time"], df["gwl"], observed, df["Station"])
    for c in trend.columns:
        df[c] = trend[c]
    rain_cols = [f"rain_api{int(t)}" for t in API_TAUS] + \
                [f"rain_obs{int(t)}" for t in API_TAUS]
    if "rain" in df.columns:
        for _, idx in df.groupby("Station", sort=False).groups.items():
            # Rainfall provenance: unlike GWL there is NO forward-fill or
            # interpolation anywhere in the rainfall path (03_merge_normalize
            # drops NaN rows; 04_align/04b_align_6h sum-or-mean over the gauges
            # and left-join, so a slot with no gauge report stays NaN). A finite
            # value is therefore a real observation, and a missing slot must not
            # count as observed. Build the mask explicitly instead of inferring
            # it, so the meaning stays correct if an imputation step is ever
            # added upstream (then thread that step's flag in here).
            rain_vals = df.loc[idx, "rain"].to_numpy(dtype=np.float64)
            rain_seen = np.isfinite(rain_vals)
            blk = rain_memory_block(rain_vals, rain_seen)
            for c, a in blk.items():
                df.loc[idx, c] = a
    else:
        for c in rain_cols:
            df[c] = np.nan

    # --- calendar / diurnal ---
    t = df["time"]
    mo = t.dt.month
    doy = t.dt.dayofyear
    hr = t.dt.hour + t.dt.minute / 60.0
    df["year"] = t.dt.year
    df["month_sin"] = np.sin(2 * np.pi * mo / 12)
    df["month_cos"] = np.cos(2 * np.pi * mo / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["monsoon"] = doy - 152

    # --- the target horizon (space-separated so the same builder serves the
    #     30-day endpoint model and the recursive +6h one-step model) ---
    df["target"] = g.shift(-horizon_steps)
    # actual timestamp of the row that supplies the label (same shift positions
    # as `target`), so split() can purge by the OBSERVED target time rather
    # than assuming anchor + horizon on a possibly gappy axis.
    df["target_time"] = df.groupby("Station", sort=False)["time"].shift(-horizon_steps)
    df["horizon"] = horizon_label
    df["target_d"] = df["target"] - df["gwl"]

    if not keep_na:
        df = df.dropna(subset=["target", "gwl"])

    # stable ordinal encodings shared across train/val/test + app
    stations = sorted(df["Station"].astype(str).unique())
    districts = sorted(df["District"].astype(str).unique())
    df["st_id"] = df["Station"].astype(str).map({s: i for i, s in enumerate(stations)}).astype(np.int32)
    df["dist_id"] = df["District"].astype(str).map({d: i for i, d in enumerate(districts)}).astype(np.int32)

    for c in NUM_COLS + HYDRO_COLS:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    # Column contract: return EVERY column the builder produced, in `KEEP` order
    # first and any extras after it. Selecting `df[KEEP]` here would silently drop
    # the columns that are computed rather than stored in the input table — the
    # rain-memory block (`rain_api*`/`rain_obs*`), `gwl_imputed`, and the driver
    # columns when a caller passes a partial frame — and callers that legitimately
    # need them (`tests/test_hydro_rain_obs.py`, bare-fixture use) then fail.
    ordered = [c for c in KEEP if c in df.columns]
    extras = [c for c in df.columns if c not in set(ordered)]
    return df[ordered + extras]


def rain_exp_30d(tbl: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Trailing-climate "expected rainfall over the next 30 days" per (District, date).

    For row at date d of year y: climo daily-rain profile for each day-of-year is
    the observed district mean over years < y (never the same year — no leakage),
    then the feature sums the 30 profile values that follow d. Rows without any
    prior year are NaN (train handles via XGBoost trees / median imputation).
    """
    t = tbl[["District", "time", "rain"]].copy()
    t["date"] = t["time"].dt.normalize()
    daily = t.groupby(["District", "date"])["rain"].sum().reset_index()
    daily["year"] = daily["date"].dt.year
    daily["doy"] = daily["date"].dt.dayofyear

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
            s = 0.0
            ok = True
            for k in range(1, 31):
                v = prof.get((int(row["doy"]) + k - 1) % 366 + 1, np.nan)
                if pd.isna(v):
                    ok = False
                    break
                s += v
            out.append((district, row["date"], s if ok else np.nan))
    exp = pd.DataFrame(out, columns=["District", "date", RAIN_EXP_COL])
    return feats.merge(exp, on=["District", "date"], how="left")


def build_full(tbl: pd.DataFrame, keep_na: bool = False,
               horizon_steps: int = HORIZON_STEPS,
               horizon_label: int = 30) -> tuple[pd.DataFrame, list[str]]:
    """build() + static soil texture + district-year extraction.

    Returns (features, extra_num_cols) so train/test/val AND the app's
    forward-forecast share byte-identical columns.
    """
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
        # NOTE(experiment): static soil texture is redundant with the st_id ordinals
        # in the pooled model (test XGB 30d 2.242 -> 2.277 m, best_iteration 15 -> 2).
        # Kept out of the shipped feature set; data stays available for the app.
        # Flip ENABLE_SOIL_FEATURES to include it if desired.
        pass
        if ENABLE_SOIL_FEATURES:
            feats = feats.merge(soil[["Station"] + STATIC_SOIL_COLS], on="Station", how="left")
            extra += list(STATIC_SOIL_COLS)
    else:
        have = soil["Station"].nunique() if not soil.empty else 0
        print(f"soil: SKIPPED (have {have}/{need} stations) — incomplete fetch", flush=True)

    assess = load_extraction()
    if not assess.empty:
        grid = extraction_year_grid(assess, feats["year"]).rename(
            columns={"district": "District", "year": "assess_year"})
        feats = feats.merge(
            grid[["District", "assess_year"] + STATIC_EXTRACTION_COLS],
            left_on=["District", "year"], right_on=["District", "assess_year"],
            how="left").drop(columns=["assess_year"])
        extra += list(STATIC_EXTRACTION_COLS)

    lulc = load_lulc()
    if not lulc.empty:
        # NOTE(experiment): static per-district LULC shares are redundant with the
        # st_id/dist_id ordinals already in the pooled model (val unchanged, XGB
        # best_iteration collapsed to 1). Kept out of the shipped feature set; the
        # data stays available for the app's Sources page and correlation_lulc.csv.
        pass

    if extra:
        for c in extra:
            feats[c] = feats[c].astype(np.float32)

    if ENABLE_RAIN_EXP:
        feats = rain_exp_30d(tbl, feats)
        feats[RAIN_EXP_COL] = feats[RAIN_EXP_COL].astype(np.float32)
        extra.append(RAIN_EXP_COL)

    return feats, extra


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Purged time split: train < 2025-10-01, val = fall/winter 2025, test = 2026.

    Partitioning is by ANCHOR date AND target time: a train row is kept only
    while its label is observed before 2025-10-01 (the validation boundary)
    and a val row only while its label is observed before 2026-01-01 (the
    test boundary).  Rows whose target timestamp reaches the next boundary
    are purged, so no validation label ever comes from inside the test period
    (and no training label from inside validation) — early stopping cannot
    see observations that belong to a later split.  Test rows are unbounded.
    """
    d = df["date"]
    tt = df["target_time"]
    train = df[(d < VAL_START) & (tt < VAL_START)]
    val = df[(d >= VAL_START) & (d < TEST_START) & (tt < TEST_START)]
    test = df[d >= TEST_START]
    return {"train": train, "val": val, "test": test}


def main() -> None:
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl["date"] = tbl["time"].dt.normalize()
    tbl = drop_gwl_spikes(tbl)
    reg = flag_regime_shift(tbl)
    if reg:
        print(f"regime-shift stations dropped: {len(reg)} -> {reg[:5]}...", flush=True)
        tbl = tbl[~tbl["Station"].isin(reg)]

    feats, extra = build_full(tbl)
    num_cols = NUM_COLS + extra
    parts = split(feats)

    FEATS.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(FEATS / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows", flush=True)

    prep = {
        "horizons": list(HORIZONS),
        "steps_per_day": STEPS_PER_DAY,
        "horizon_steps": HORIZON_STEPS,
        "num_cols": num_cols,
        "hydro_num_cols": HYDRO_COLS,   # computed/carried in every frame; not in
                                        # num_cols (trained schema is frozen —
                                        # enabling requires a deliberate retrain)
        "split": split.__doc__.strip().split("\n")[0],
        "stations": sorted(feats["Station"].astype(str).unique(), key=str),
        "districts": sorted(feats["District"].astype(str).unique(), key=str),
        "extra_features": {
            "soil": STATIC_SOIL_COLS if "sand_0_30" in feats.columns else [],
            "extraction": STATIC_EXTRACTION_COLS if "ann_gw_draft_mcm" in feats.columns else [],
            "lulc": STATIC_LULC_COLS if "lulc_cropland_pct" in feats.columns else [],
            "hydrology": HYDRO_COLS,
        },
        "train_rows": int(len(parts["train"])),
        "val_rows": int(len(parts["val"])),
        "test_rows": int(len(parts["test"])),
        "built_at": str(pd.Timestamp.now())[:19],
    }
    (FEATS / "prep.json").write_text(json.dumps(prep, indent=2))
    # sanity: persistence floor on test = carry today's reading 30 days forward
    base = math.sqrt(((parts["test"]["target"] - parts["test"]["gwl"]) ** 2).mean())
    print(f"persistence RMSE on test (30d, gwl as-is): {base:.3f} m", flush=True)


if __name__ == "__main__":
    main()