"""17_recursive — genuine 6-hourly 120-step recursive GWL trajectory engine.

Strictly causal one-step-at-a-time forecast:
  1. anchor  = latest observed GWL (its REJECTION-SPIKE-CLEANED value from the
     training features path); NEVER predicted.
  2. features at time t are the SAME causal features the +6h model was trained
     on (06_features.build, horizon_steps=1): lags, rolling memory, driver
     windows, calendar. Only observables through t are used.
  3. predict delta = GWL(t+6h) - GWL(t) with the +6h point model (q50 for the
     band paths), level = gwl(t) + delta.
  4. append the predicted GWL (and the scenario's future drivers for t+6h) to
     the synthetic history, regenerate causal lag/rolling features, repeat.
  5. 120 steps -> one ML-generated trajectory from the anchor to +30 d.

No interpolation anywhere. Future drivers come ONLY from 16_future_drivers
(climatology / persistence / dry scenario) — never future observed values.

Output: columns timestamp, gwl_q05, gwl_q50, gwl_q95, horizon_hours plus
scenario + quality metadata. Used by Forecat page + 18_backtest_6h.
"""

from __future__ import annotations

import json
from pathlib import Path

import importlib

import numpy as np
import pandas as pd

F = importlib.import_module("06_features")
from importlib import import_module
scenario_frame = import_module("16_future_drivers").scenario_frame

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
CFG6 = ROOT / "models" / "6h_feature_config.json"

BOUND = 160          # rows kept in the synthetic-feature window (covers lag120/roll30)
OBS_WINDOW = 130     # observed rows carried into the synthetic history
STEP_H = pd.Timedelta(hours=6)
STEPS = 120
OUT_OF_RANGE_M = 25.0

_MODEL_CACHE: dict | None = None


def load_models() -> dict:
    """One-time model cache (shared by the app, the fleet, the backtest)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import joblib
        _MODEL_CACHE = {
            "point": joblib.load(ROOT / "models" / "6h_xgb_point.joblib"),
            "q05": joblib.load(ROOT / "models" / "6h_xgb_q05.joblib"),
            "q50": joblib.load(ROOT / "models" / "6h_xgb_q50.joblib"),
            "q95": joblib.load(ROOT / "models" / "6h_xgb_q95.joblib"),
        }
    return _MODEL_CACHE


def load_trainset_ordinals() -> tuple[list[str], list[str]]:
    cfg = json.loads(CFG6.read_text())
    return list(cfg["stations"]), list(cfg["districts"])


def resolve_station(name: str, st_names: list[str]) -> str | None:
    """Case-insensitive lookup: config names keep original mixed casing."""
    key = str(name).strip().upper()
    for s in st_names:
        if s.upper() == key:
            return s
    return None


def _row_features(hist: pd.DataFrame, station: str, district: str,
                  st_id: int, dist_id: int, num_cols: list[str]) -> pd.Series:
    """One feature row at the last timestamp, mirroring F.build() formulas.

    ``hist`` delivers Station/time/gwl + driver columns for the rolling window
    ending at the row we are predicting. st_id/dist_id are mapped through the
    TRAINING ordinals so the pooled model sees consistent station identity.
    """
    df = hist.tail(BOUND).copy()
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
        df["rain_1d"] = rg.transform(lambda s: s.rolling(4, min_periods=1).sum())
        df["rain_7d"] = rg.transform(lambda s: s.rolling(28, min_periods=1).sum())
        df["rain_30d"] = rg.transform(lambda s: s.rolling(120, min_periods=1).sum())
    for col, win in (("temp", 28), ("humidity", 28)):
        if col in df.columns:
            df[f"{col}_7d"] = df.groupby("Station", sort=False)[col].transform(
                lambda s: s.rolling(win, min_periods=1).mean())
    t = df["time"]
    mo, doy, hr = t.dt.month, t.dt.dayofyear, t.dt.hour + t.dt.minute / 60.0
    df["year"] = t.dt.year
    df["month_sin"] = np.sin(2 * np.pi * mo / 12)
    df["month_cos"] = np.cos(2 * np.pi * mo / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["monsoon"] = doy - 152

    want = num_cols + ["st_id", "dist_id"]
    row = df.iloc[-1]
    vals = {}
    for c in want:
        if c == "st_id":
            vals[c] = st_id
        elif c == "dist_id":
            vals[c] = dist_id
        else:
            vals[c] = row[c] if c in row else np.nan
    return pd.Series(vals).astype(np.float32)


def recursive_trajectory(station: str, scenario: str = "climatology",
                         steps: int = STEPS,
                         obs_tbl: pd.DataFrame | None = None,
                         models: dict | None = None,
                         ordinals: tuple[list[str], list[str]] | None = None,
                         driver_cache: dict | None = None) -> dict:
    """Full +6h..+30d trajectory for one station. Returns dict with:

    - anchor, anchor_time, district, q05/q50/q95 level arrays (length ``steps``)
    - times (length ``steps``), horizon_hours (6..720)
    - scenario_meta, reliable (False if level escapes the observed range ±25 m)
    - error key when the station can't be scored.
    """
    station = str(station).strip().upper()
    if models is None:
        models = load_models()
    st_names, dist_names = ordinals if ordinals is not None else load_trainset_ordinals()
    resolved = resolve_station(station, st_names)
    if resolved is None:
        return {"station": station, "error": "station not in 6h training set"}
    station = resolved

    if obs_tbl is None:
        obs_tbl = pd.read_parquet(ALIGNED)
        obs_tbl["time"] = pd.to_datetime(obs_tbl["time"], errors="coerce")

    raw = obs_tbl[obs_tbl["Station"].str.upper() == station.upper()].sort_values("time").reset_index(drop=True)
    if raw.empty:
        return {"station": station, "error": "no telemetry"}

    hist = F.drop_gwl_spikes(raw).copy()          # same spike mask as training
    obs = hist[hist["gwl"].notna()]
    if obs.empty:
        return {"station": station, "error": "no clean GWL"}
    anchor = float(obs["gwl"].iloc[-1])
    anchor_time = pd.to_datetime(obs["time"].iloc[-1])
    district = str(raw["District"].iloc[0])
    dist_id = dist_names.index(district) if district in dist_names else -1
    st_id = st_names.index(station)

    olo, ohi = float(hist["gwl"].min()), float(hist["gwl"].max())
    in_range = lambda L: (olo - OUT_OF_RANGE_M) <= L <= (ohi + OUT_OF_RANGE_M)

    # scenario drivers for every future 6h step
    if driver_cache is not None and station in driver_cache:
        fdrv, smeta = driver_cache[station]
    else:
        fdrv, smeta = scenario_frame(station, district, raw, steps, scenario)
    fdrv = fdrv.set_index("time")
    fdrv.index = pd.to_datetime(fdrv.index)

    # synthetic history = observed tail (driver cols filled w/ NaN where unknown)
    synth = obs.tail(OBS_WINDOW).copy()
    for col in F.NUM_COLS:
        if col not in synth.columns and col not in ("gwl", "rain", "temp",
                                                    "humidity", "solar",
                                                    "wind_speed", "pressure",
                                                    "river_level", "canal_level"):
            continue
    # align driver columns present in train
    for c in ("rain", "temp", "humidity", "solar", "wind_speed", "pressure",
              "river_level", "canal_level"):
        if c not in synth.columns:
            synth[c] = np.nan

    lo, med, hi = [], [], []
    times = []
    reliable = True
    fnames = list(models["point"].feature_names_in_)
    num_cols = list(json.loads(CFG6.read_text())["num_cols"])

    for k in range(1, steps + 1):
        t_next = anchor_time + STEP_H * k
        times.append(t_next)
        row = _row_features(synth, station, district, st_id, dist_id, num_cols)
        X = row.reindex(fnames).to_frame().T
        q05 = float(models["q05"].predict(X)[0])
        q50 = float(models["q50"].predict(X)[0])
        q95 = float(models["q95"].predict(X)[0])

        # quantile paths are recursive too: each carries its own level.
        # clamp per step so the band stays ordered lo <= med <= hi.
        l05 = (lo[-1] + q05) if lo else anchor + q05
        l50 = (med[-1] + q50) if med else anchor + q50
        l95 = (hi[-1] + q95) if hi else anchor + q95
        l05, l50, l95 = min(l05, l50), l50, max(l95, l50)
        lo.append(l05); med.append(l50); hi.append(l95)

        if not in_range(l50):
            reliable = False

        # append the QUANTILE-MEDIAN level to the synthetic history as the
        # driving GWL value (same convention as the app's q50 headline path)
        level = l50
        drv = fdrv.loc[t_next] if t_next in fdrv.index else pd.Series(dtype=float)
        newrow = {"Station": station, "District": district, "time": t_next,
                  "gwl": level}
        for c in ("rain", "temp", "humidity", "solar", "wind_speed", "pressure",
                  "river_level", "canal_level"):
            newrow[c] = drv.get(c) if not drv.empty and c in drv else np.nan
        synth = pd.concat([synth, pd.DataFrame([newrow])], ignore_index=True)
        synth = synth.tail(BOUND).reset_index(drop=True)

    return {
        "station": station,
        "district": district,
        "anchor": anchor,
        "anchor_time": str(anchor_time),
        "times": [str(x) for x in times],
        "horizon_hours": [int(6 * k) for k in range(1, steps + 1)],
        "gwl_q05": lo,
        "gwl_q50": med,
        "gwl_q95": hi,
        "scenario": scenario,
        "scenario_meta": smeta,
        "reliable": reliable,
        "obs_range": [olo, ohi],
    }


def main() -> None:
    for st in ("ASHADHA PRATHMIK VIDYALAYA",):
        out = recursive_trajectory(st, scenario="climatology")
        if "error" in out:
            print(out); continue
        q50 = out["gwl_q50"]
        print(f"\n{st}  anchor {out['anchor']:.2f} m  reliable={out['reliable']}")
        for k in (1, 4, 12, 28, 60, 120):     # 6h | 1d | 3d | 7d | 15d | 30d
            i = k - 1
            print(f"  +{out['horizon_hours'][i]}h  q05={out['gwl_q05'][i]:7.2f} "
                  f"q50={q50[i]:7.2f} q95={out['gwl_q95'][i]:7.2f}")
        print("  scenario:", json.dumps(out["scenario_meta"], indent=1))


if __name__ == "__main__":
    main()