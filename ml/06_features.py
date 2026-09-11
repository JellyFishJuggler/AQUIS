"""06_features — feature/target engineering on the 6-hourly track.

One feature row per (station, 6h-slot); the ONLY forecast horizon is 30 days =
120 six-hour steps ahead (production-consistent cadence, see config).
'gwl' (the current reading) is the anchor feature — it is NEVER predicted; the
predicted quantity is the 30-day change GWL(t+120) - GWL(t).

Outputs -> ml/data/features/{train,val,test}.parquet + prep.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _soil import SOIL_COLS, EXTRACTION_COLS, load_soil, load_extraction, extraction_year_grid
from _lulc import LULC_COLS, load_lulc

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
FEATS = ROOT / "data" / "features"

HORIZONS = (30,)            # days
STEPS_PER_DAY = 4           # 6-hourly grid
HORIZON_STEPS = 30 * STEPS_PER_DAY

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
        "st_id", "dist_id"] + NUM_COLS


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
    df["horizon"] = horizon_label
    df["target_d"] = df["target"] - df["gwl"]

    if not keep_na:
        df = df.dropna(subset=["target", "gwl"])

    # stable ordinal encodings shared across train/val/test + app
    stations = sorted(df["Station"].astype(str).unique())
    districts = sorted(df["District"].astype(str).unique())
    df["st_id"] = df["Station"].astype(str).map({s: i for i, s in enumerate(stations)}).astype(np.int32)
    df["dist_id"] = df["District"].astype(str).map({d: i for i, d in enumerate(districts)}).astype(np.int32)

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    return df[KEEP]


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
    """Time-based split: train < 2025-10-01, val = fall/winter 2025, test = 2026."""
    d = df["date"]
    train = df[d < pd.Timestamp("2025-10-01")]
    val = df[(d >= pd.Timestamp("2025-10-01")) & (d < pd.Timestamp("2026-01-01"))]
    test = df[d >= pd.Timestamp("2026-01-01")]
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
        "stations": sorted(feats["Station"].astype(str).unique(), key=str),
        "districts": sorted(feats["District"].astype(str).unique(), key=str),
        "extra_features": {
            "soil": STATIC_SOIL_COLS if "sand_0_30" in feats.columns else [],
            "extraction": STATIC_EXTRACTION_COLS if "ann_gw_draft_mcm" in feats.columns else [],
            "lulc": STATIC_LULC_COLS if "lulc_cropland_pct" in feats.columns else [],
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