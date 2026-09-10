"""30_traj_datasets — multi-horizon training long-frame for the direct trajectory v2.

Direct multi-horizon paradigm (no recursion): one shared model predicts the
6-hourly target GWL(t+h) - GWL(t) for ANY horizon h, given the same feature
vector as the production builder (past windows / lags only, i.e. information
available at time t) plus explicit horizon features (h, h_sin, h_cos). At
inference the anchor row is scored with h = 1..120 -> 120 genuine independent
6-hourly forecast deltas; level = observed anchor + delta (anchor never modeled).

Targets are taken at the EXACT time t + h*6h on each station's observed axis
(the aligned 6h grid has gaps; shift-based targets would silently attach stale
readings). Rows/horizons with no observed target are dropped.

Splits: reuse the pipeline time split (train < 2025-10-01, val = fall/winter
2025). The train frame is row-subsampled (stride) x horizon-subsampled (dense
short horizons -> sparse long) to keep the long-form frame ~7M rows within the
8 GB box; the validation frame keeps ALL rows x ALL 120 horizons for a clean
per-horizon curve.

Outputs -> data/features_traj/{train,val}.parquet + prep_traj.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import importlib

F = importlib.import_module("06_features")

ROOT = Path(__file__).resolve().parent
FE6 = ROOT / "data" / "features_6h"
OUT = ROOT / "data" / "features_traj"
OUT.mkdir(parents=True, exist_ok=True)

MAX_H = 120                 # 30 days on the 6h grid
TRAIN_STRIDE = 8            # row subsampling for the train long-form
HORIZONS_TRAIN = list(range(1, 21)) + [24, 28, 32, 36, 40, 48, 56, 64, 80, 96, 120]
HORIZONS_VAL = list(range(1, MAX_H + 1))

NUM_COLS = list(F.NUM_COLS)
META = ["Station", "District", "time", "st_id", "dist_id"]
HCOL = ["h", "h_sin", "h_cos"]


def _horizon_feats(h: int) -> dict:
    return {"h": h, "h_sin": float(np.sin(2 * np.pi * h / MAX_H)),
            "h_cos": float(np.cos(2 * np.pi * h / MAX_H))}


def _axis_and_targets(parts: list[pd.DataFrame]):
    """Per-station observed time axis (unique, sorted) + cleaned gwl values."""
    t = pd.concat([p[["time", "gwl"]] for p in parts], ignore_index=True)
    t = t.sort_values("time").drop_duplicates("time").dropna(subset=["gwl"])
    return t["time"].to_numpy(), t["gwl"].to_numpy(dtype=np.float64)


def _expand(rows: pd.DataFrame, times_st: np.ndarray, gwl_st: np.ndarray,
            horizons: list[int]) -> pd.DataFrame:
    """For each candidate row, emit (h, target_delta, horizon feats, row feats).

    target_delta = gwl(t+h*6h) - gwl(t) when gwl(t+h*6h) is observed at the
    exact time; rows/horizons without an exact observation are skipped.
    """
    t0s = rows["time"].to_numpy(dtype="datetime64[ns]")
    g0 = rows["gwl"].to_numpy(dtype=np.float64)
    out: list[pd.DataFrame] = []
    for h in horizons:
        tt = t0s + np.timedelta64(h * 6, "h")
        q = np.searchsorted(times_st, tt)
        qq = np.clip(q, 0, len(times_st) - 1)
        ok = q < len(times_st)
        ok &= times_st[qq] == tt
        delta = np.full(len(rows), np.nan)
        delta[ok] = gwl_st[qq[ok]] - g0[ok]
        keep = ok & np.isfinite(delta)
        if not keep.any():
            continue
        idx = np.where(keep)[0]
        hf = _horizon_feats(h)
        chunk = rows.iloc[idx].assign(
            h=h, h_sin=hf["h_sin"], h_cos=hf["h_cos"],
            target_delta=np.asarray(delta[keep], dtype=np.float32))
        out.append(chunk)
    if not out:
        return pd.DataFrame(columns=list(rows.columns) + HCOL + ["target_delta"])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    t0 = time.time()
    tr = pd.read_parquet(FE6 / "train.parquet", columns=META + NUM_COLS)
    va = pd.read_parquet(FE6 / "val.parquet", columns=META + NUM_COLS)
    te = pd.read_parquet(FE6 / "test.parquet", columns=META + NUM_COLS)

    for df in (tr, va, te):
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # all feature rows define the station's observed axis (cleaned gwl)
    axis = pd.concat([tr, va, te], ignore_index=True)
    stations = sorted(axis["Station"].astype(str).unique())
    print(f"stations: {len(stations):,}", flush=True)

    train_rows, val_rows = [], []
    for i, st in enumerate(stations, 1):
        sub = axis[axis["Station"].astype(str) == st]
        times_st, gwl_st = _axis_and_targets([sub])

        rows_tr = tr[tr["Station"].astype(str) == st].iloc[::TRAIN_STRIDE]
        rows_va = va[va["Station"].astype(str) == st]
        if len(rows_tr):
            train_rows.append(_expand(rows_tr, times_st, gwl_st, HORIZONS_TRAIN))
        if len(rows_va):
            val_rows.append(_expand(rows_va, times_st, gwl_st, HORIZONS_VAL))
        if i % 100 == 0:
            print(f"  {i}/{len(stations)} stations "
                  f"(train so far {sum(len(x) for x in train_rows):,})", flush=True)

    trb = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    vab = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()

    for df in (trb, vab):
        if not df.empty:
            df["h"] = df["h"].astype(np.int16)
            df["st_id"] = df["st_id"].astype(np.int32)
            df["dist_id"] = df["dist_id"].astype(np.int32)
            for c in NUM_COLS:
                df[c] = df[c].astype(np.float32)
            df["target_delta"] = df["target_delta"].astype(np.float32)

    trb.to_parquet(OUT / "train.parquet", index=False)
    vab.to_parquet(OUT / "val.parquet", index=False)

    prep = {
        "paradigm": "direct multi-horizon shared model (no recursion)",
        "target": "GWL(t+h*6h) - GWL(t), exact observed time match (no stale shift)",
        "train_stride": TRAIN_STRIDE,
        "h_train_list": HORIZONS_TRAIN,
        "h_val_all": MAX_H,
        "num_cols": NUM_COLS,
        "horizon_cols": HCOL,
        "train_rows": int(len(trb)),
        "val_rows": int(len(vab)),
        "stations": int(len(stations)),
        "split": "train < 2025-10-01 (row-subsampled x horizon-subsampled), val = all",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "prep_traj.json").write_text(json.dumps(prep, indent=2))
    print(f"\ntrain rows: {len(trb):,} | val rows: {len(vab):,}  ({(time.time() - t0):.0f}s)")
    if not trb.empty:
        per_h = trb.groupby("h").size()
        print(per_h.head(12).to_string())


if __name__ == "__main__":
    main()