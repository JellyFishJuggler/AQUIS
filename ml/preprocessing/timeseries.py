"""Timeseries preprocessing pipeline for AQUIS groundwater forecasting."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GWL_COL = "Groundwater Level Telemetry 6 Hourly (meter)"
TIME_COL = "Data Acquisition Time"
STATION_COL = "Station"
AGENCY_COL = "Agency"
SLNO_COL = "SlNo"
DISTRICT_COL = "District"

DISTRICT_MAPPING = {
    "Allahabad": "PRAYAGRAJ",
    "Amethi (C.S.M. Nagar)": "AMETHI",
    "Amroha (J.P.Nagar)": "AMROHA",
    "BAGHPAT": "BAGPAT",
    "G.B. Nagar": "G.B.NAGAR",
    "Kansiram Nagar": "KASGANJ",
    "SIDDHARTHNAGAR": "SIDDHARTH NAGAR",
    "Sambhal": "SAMBHAL",
}


def normalize_district(name: str) -> str:
    """Normalize district name to match back-end data."""
    return DISTRICT_MAPPING.get(name, name.upper())


def load_and_clean(parquet_path: str | Path) -> pd.DataFrame:
    """Load parquet, parse timestamps, drop duplicates, sort per station."""
    df = pd.read_parquet(parquet_path)
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=[TIME_COL, GWL_COL, STATION_COL])
    df = df.drop_duplicates(subset=[STATION_COL, TIME_COL])
    df = df.sort_values([STATION_COL, TIME_COL]).reset_index(drop=True)
    return df


def station_slug(station_name: str, agency: str, sl_no: int) -> str:
    """Generate unique case-sensitive slug from Station + Agency (SlNo is row index, not station ID)."""
    safe_station = station_name.replace("/", "_").replace("\\", "_")
    safe_agency = agency.replace("/", "_").replace("\\", "_")
    base = f"{safe_station}_{safe_agency}"
    slug = base[:180]
    import hashlib
    hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
    return f"{slug}_{hash_suffix}"


def build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add time_hours = hours since station's first reading."""
    df = df.copy()
    first_ts = df.groupby(STATION_COL)[TIME_COL].transform("min")
    df["time_hours"] = (df[TIME_COL] - first_ts).dt.total_seconds() / 3600.0
    return df


def detect_gaps(df: pd.DataFrame, threshold_hours: float = 72.0) -> dict[str, list[dict]]:
    """Flag gaps > threshold_hours per station. Return gap metadata."""
    gaps = {}
    for station, grp in df.groupby(STATION_COL):
        grp = grp.sort_values(TIME_COL)
        time_diff = grp[TIME_COL].diff().dt.total_seconds() / 3600.0
        gap_mask = time_diff > threshold_hours
        if gap_mask.any():
            gap_starts = grp[TIME_COL][gap_mask]
            gap_ends = grp[TIME_COL].shift(-1)[gap_mask]
            durations = time_diff[gap_mask]
            gaps[station] = [
                {
                    "start": str(start),
                    "end": str(end) if pd.notna(end) else None,
                    "duration_hours": float(dur),
                }
                for start, end, dur in zip(gap_starts, gap_ends, durations, strict=False)
            ]
        else:
            gaps[station] = []
    return gaps


def detect_sentinel_values(df: pd.DataFrame, repeat_threshold: int = 10) -> pd.Series:
    """
    Flag sentinel/corrupted values per station using IQR on diffs + repeated values.
    Returns boolean mask (True = valid, False = sentinel).
    """
    valid_mask = pd.Series(True, index=df.index)

    for station, grp in df.groupby(STATION_COL):
        idx = grp.index
        gwl = grp[GWL_COL].values
        diffs = np.diff(gwl, prepend=gwl[0])

        q1 = np.percentile(diffs, 25)
        q3 = np.percentile(diffs, 75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr

        sentinel_diff = (diffs < lower) | (diffs > upper)

        repeated = pd.Series(gwl).rolling(window=repeat_threshold, min_periods=repeat_threshold).apply(
            lambda x: 1 if x.nunique() == 1 else 0
        ).fillna(0).astype(bool).values

        station_invalid = sentinel_diff | repeated
        valid_mask.loc[idx] = ~station_invalid

    return valid_mask


def load_exogenous_features(backend_csv: str | Path) -> pd.DataFrame:
    """Load district-level static features from back-end data.csv."""
    df = pd.read_csv(backend_csv)
    up_df = df[df["STATE"] == "UTTAR PRADESH"].copy()

    feature_cols = {
        "Rainfall (mm)_Total": "district_rainfall_mm",
        "Ground Water Recharge (ham)_Total_Total": "district_recharge_total_ham",
        "Ground Water Recharge (ham)_Rainfall Recharge_Total": "district_recharge_rainfall_ham",
        "Ground Water Recharge (ham)_Canals_Total": "district_recharge_canals_ham",
        "Ground Water Recharge (ham)_Surface Water Irrigation_Total": "district_recharge_irrigation_ham",
        "Ground Water Recharge (ham)_Ground Water Irrigation_Total": "district_recharge_gw_irr_ham",
        "Ground Water Recharge (ham)_Tanks and Ponds_Total": "district_recharge_tanks_ham",
        "Ground Water Extraction for all uses (ha.m)_Total_Total": "district_extraction_total_ham",
        "Ground Water Extraction for all uses (ha.m)_Domestic_Total": "district_extraction_domestic_ham",
        "Ground Water Extraction for all uses (ha.m)_Industrial_Total": "district_extraction_industrial_ham",
        "Ground Water Extraction for all uses (ha.m)_Irrigation_Total": "district_extraction_irrigation_ham",
        "Stage of Ground Water Extraction (%)_Total_Total": "district_extraction_stage_pct",
        "Annual Ground water Recharge (ham)_Total_Total": "district_annual_recharge_ham",
        "Annual Extractable Ground water Resource (ham)_Total_Total": "district_extractable_resource_ham",
    }

    avail_cols = {k: v for k, v in feature_cols.items() if k in up_df.columns}
    exog = up_df[["DISTRICT"] + list(avail_cols.keys())].copy()
    exog = exog.rename(columns=avail_cols)
    exog["DISTRICT"] = exog["DISTRICT"].str.upper()

    for col in exog.columns:
        if col != "DISTRICT":
            exog[col] = pd.to_numeric(exog[col], errors="coerce")

    exog = exog.fillna(exog.median(numeric_only=True))
    return exog


def attach_exogenous_features(df: pd.DataFrame, exog_df: pd.DataFrame) -> pd.DataFrame:
    """Join district exogenous features to station dataframe."""
    df = df.copy()
    df["district_norm"] = df[DISTRICT_COL].apply(normalize_district)
    exog_df = exog_df.copy()
    exog_df["district_norm"] = exog_df["DISTRICT"].str.upper()

    feature_cols = [c for c in exog_df.columns if c not in ["DISTRICT", "district_norm"]]
    df = df.merge(exog_df[["district_norm"] + feature_cols], on="district_norm", how="left")

    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(exog_df[col].median())

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all causal features: lags, rolling, seasonal, trend."""
    df = df.copy()
    df = df.sort_values([STATION_COL, TIME_COL])

    max_lag = 120

    for station, grp in df.groupby(STATION_COL):
        idx = grp.index
        gwl = grp[GWL_COL].values
        n = len(gwl)

        lags = [1, 2, 3, 4, 7, 14, 28, 60, 120]
        for lag in lags:
            col = f"lag_{lag}"
            lagged = np.full(n, np.nan)
            lagged[lag:] = gwl[:-lag]
            df.loc[idx, col] = lagged

        windows = [7, 28, 60, 120]
        for w in windows:
            roll_mean = pd.Series(gwl).rolling(w, min_periods=1).mean().values
            roll_std = pd.Series(gwl).rolling(w, min_periods=1).std().fillna(0).values
            df.loc[idx, f"roll_{w}_mean"] = roll_mean
            df.loc[idx, f"roll_{w}_std"] = roll_std

        trend_28 = pd.Series(gwl).rolling(28, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 28 else np.nan
        ).values
        trend_60 = pd.Series(gwl).rolling(60, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 60 else np.nan
        ).values
        df.loc[idx, "trend_28"] = trend_28
        df.loc[idx, "trend_60"] = trend_60

    doy = df[TIME_COL].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_doy_2"] = np.sin(4 * np.pi * doy / 365.25)
    df["cos_doy_2"] = np.cos(4 * np.pi * doy / 365.25)
    df["year"] = df[TIME_COL].dt.year

    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 80/20 split per station. No shuffle."""
    train_parts = []
    test_parts = []
    for station, grp in df.groupby(STATION_COL):
        grp = grp.sort_values(TIME_COL)
        n = len(grp)
        split_idx = int(n * train_frac)
        if split_idx < 10:
            split_idx = min(10, n - 1)
        train_parts.append(grp.iloc[:split_idx])
        test_parts.append(grp.iloc[split_idx:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def prepare_feature_matrix(df: pd.DataFrame, target_col: str = GWL_COL) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X and target y, dropping rows with NaN features."""
    exclude = {STATION_COL, AGENCY_COL, SLNO_COL, DISTRICT_COL, TIME_COL, "district_norm", target_col, "RL_MSL", "slug",
               "State", "Tehsil", "Block", "Village", "River", "Basin", "Tributary", "Subtributary", "SubSubtributary", "Local River",
               "Latitude", "Longitude", "_id", "SlNo", "State LGD Code", "District LGD Code"}
    feature_cols = [c for c in df.columns if c not in exclude]

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols

    valid = df[feature_cols].notna().all(axis=1)
    X = df.loc[valid, feature_cols].values.astype(float)
    y = df.loc[valid, target_col].values.astype(float)
    return X, y, feature_cols


def full_pipeline(
    parquet_path: str | Path,
    backend_csv: str | Path,
    station_slug_filter: str | None = None,
) -> dict[str, Any]:
    """Full preprocessing pipeline for a station (or all)."""
    df = load_and_clean(parquet_path)

    df["slug"] = df.apply(lambda r: station_slug(r[STATION_COL], r[AGENCY_COL], r[SLNO_COL]), axis=1)

    if station_slug_filter:
        df = df[df["slug"] == station_slug_filter].copy()
        if df.empty:
            raise ValueError(f"No data for station slug: {station_slug_filter}")

    df = build_time_index(df)
    gaps = detect_gaps(df)
    sentinel_mask = detect_sentinel_values(df)
    df = df[sentinel_mask].copy()

    exog = load_exogenous_features(backend_csv)
    df = attach_exogenous_features(df, exog)

    df = build_features(df)

    train_df, test_df = time_split(df)

    # Get clean feature columns using prepare_feature_matrix logic
    exclude = {STATION_COL, AGENCY_COL, SLNO_COL, DISTRICT_COL, TIME_COL, "district_norm", GWL_COL, "slug",
               "State", "Tehsil", "Block", "Village", "River", "Basin", "Tributary", "Subtributary", "SubSubtributary", "Local River",
               "Latitude", "Longitude", "_id", "SlNo", "State LGD Code", "District LGD Code", "RL_MSL"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    feature_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    return {
        "train": train_df,
        "test": test_df,
        "gaps": gaps,
        "sentinel_excluded": (~sentinel_mask).sum(),
        "stations": df["slug"].unique().tolist(),
        "feature_cols": feature_cols,
    }