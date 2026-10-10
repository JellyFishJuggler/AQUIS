"""dl_rebuild/build_data — transferable-only dataset on the frozen honest sprint.

Same input (tracks/base_features.parquet, 6h grid, single 30-day horizon),
same folds and stride as exp_harness so every DL gate is apples-to-apples
with the E0/XGB/persistence numbers already in arm_reports.json.  The ONE
deliberate difference vs E0: st_id/dist_id (identity) are REMOVED and the
schema is purely transferable — trailing GWL trajectory + exogenous drivers
+ calendar.  NaN columns get a per-column binary mask (driver-optional:
river/canal can be missing for a whole geography, as with a Delhi port).

Cache: dl_rebuild/cache/*.npz  (loads in a few seconds on later runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exp_harness as H

ROOT = Path(__file__).resolve().parents[2]      # ml/
HERE = Path(__file__).resolve().parent          # dl_rebuild
CACHE = HERE / "cache"

DENSE = [
    "gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
    "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
    "gwl_slope7", "gwl_slope30", "gwl_slope90", "gwl_dev7", "gwl_dev30",
    "rain_1d", "rain_7d", "rain_30d", "rain_api7", "rain_api22", "rain_api60",
    "temp", "temp_7d", "humidity", "humidity_7d", "solar", "wind_speed",
    "pressure", "river_level", "canal_level", "gwl_obs7", "gwl_obs30",
    "gwl_obs90", "gwl_imputed",
]
SEASONAL = ["month_sin", "month_cos", "doy_sin", "doy_cos",
            "hour_sin", "hour_cos", "monsoon"]
SEQ_CHANNELS = ["gwl", "dw"]         # dw = first difference of gwl


def load() -> pd.DataFrame:
    df = H.load_frame()
    return df


def build() -> dict:
    t0 = pd.Timestamp.now()
    df = load()
    print(f"frame {len(df):,} rows / {df['Station'].nunique()} st "
          f"({(pd.Timestamp.now()-t0).total_seconds():.0f}s)", flush=True)

    # ---- identity removed; station key kept in a side column only --------
    st = df["st_id"].to_numpy(np.int32)

    cols = DENSE + SEASONAL
    X = df[cols].to_numpy("float32")
    M = np.isfinite(X).astype(np.float32)
    X = np.where(M, X, 0.0).astype(np.float32)          # fill; mask carries NaN
    FTR = np.concatenate([X, M], axis=1).astype(np.float32)   # (n, 2*len(cols))
    colnames = cols + [f"mask.{c}" for c in cols]

    # ---- trailing GWL trajectory (per station, time-ascending) -----------
    # assume station-major, time-sorted order (as produced by 06_features)
    gwl = np.where(np.isfinite(df["gwl"].to_numpy(float)),
                   df["gwl"].to_numpy(float), np.nan).astype(np.float32)
    dw = np.empty_like(gwl)
    dw[0] = 0.0
    dw[1:] = gwl[1:] - gwl[:-1]
    G = np.stack([gwl, dw], axis=1)                     # (n, 2)

    y = df["target_d"].to_numpy(float)
    tgt_time = df["target_time"].to_numpy("datetime64[ns]")
    date = df["date"].to_numpy("datetime64[ns]")
    district = df["District"].astype(str).to_numpy()
    fin = np.isfinite(y)
    dist = df["dist_id"].to_numpy(np.int32)

    # ---- per-station contiguous runs + window bookkeeping (W later) ------
    rid = np.cumsum(np.r_[True, np.diff(st) != 0]) - 1
    rlen = np.bincount(rid)
    run_start = np.repeat(np.cumsum(np.r_[0, rlen[:-1]]), rlen)   # (n,)
    run_end = np.repeat(np.cumsum(rlen), rlen)                    # exclusive

    out = {
        "FTR": FTR, "G": G, "y": y.astype(np.float32), "st": st,
        "n": len(df),
        "tgt_time": tgt_time.astype("int64"), "date": date.astype("int64"),
        "fin": fin.astype(np.int8),
        "district": district,
        "run_start": run_start, "run_end": run_end,
        "colnames": colnames,
        "meta": {"rows": int(len(df)), "stations": int(np.unique(st).size),
                 "cols": len(FTR[0]), "seq_channels": SEQ_CHANNELS},
    }
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE / "data.npz", FTR=FTR, G=G, y=out["y"], st=st,
        tgt_time=out["tgt_time"], date=out["date"], fin=out["fin"],
        run_start=run_start, run_end=run_end)
    (CACHE / "colnames.json").write_text(json.dumps({"cols": colnames,
                                                     "meta": out["meta"]},
                                                    indent=1))
    with open(CACHE / "district.txt", "w") as f:
        f.write("\n".join(district))
    print(f"cached build ({(pd.Timestamp.now()-t0).total_seconds():.0f}s) "
          f"FTR={FTR.shape} G={G.shape}", flush=True)
    return out


def ly_load() -> dict:
    if not (CACHE / "data.npz").exists():
        return build()
    d = np.load(CACHE / "data.npz")
    cm = json.loads((CACHE / "colnames.json").read_text())
    out = {
        "FTR": d["FTR"], "G": d["G"], "y": d["y"], "st": d["st"],
        "n": int(len(d["G"])),
        "tgt_time": d["tgt_time"], "date": d["date"], "fin": d["fin"],
        "district": np.array((CACHE / "district.txt").read_text().split("\n")
                             if (CACHE / "district.txt").exists()
                             else []),
        "run_start": d["run_start"], "run_end": d["run_end"],
        "colnames": cm["cols"], "meta": cm["meta"],
    }
    return out


if __name__ == "__main__":
    out = build()
    print("cols:", len(out["colnames"]), "| fill-nan vs persists ok")