"""14_6h_features — causal one-step (+6h) feature/target set for the recursive
trajectory model.

Identical feature schema as the frozen 30-day endpoint model (06_features):
the SAME causal features (lags, rolling memory, driver windows, calendar,
static soil/extraction if enabled) build the row at time t; only the TARGET
changes to the one-step delta GWL(t+6h) - GWL(t) (horizon_steps=1).

No future information is used: every feature is observable at time t. The
trained model is then rolled forward recursively by the 17_recursive engine.

Outputs -> ml/data/features_6h/{train,val,test}.parquet + prep_6h.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import importlib

feats_mod = importlib.import_module("06_features")

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
FEATS = ROOT / "data" / "features_6h"

HORIZON_STEPS = 1          # +6h on the 6-hourly grid
HORIZON_LABEL = 1


def main() -> None:
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl = feats_mod.drop_gwl_spikes(tbl)
    tbl["date"] = tbl["time"].dt.normalize()
    reg = feats_mod.flag_regime_shift(tbl)
    if reg:
        print(f"regime-shift stations dropped: {len(reg)} -> {reg[:5]}...", flush=True)
        tbl = tbl[~tbl["Station"].isin(reg)]

    feats, extra = feats_mod.build_full(
        tbl, keep_na=True, horizon_steps=HORIZON_STEPS, horizon_label=HORIZON_LABEL)
    # one-step rows must have a valid +6h target AND a valid current reading
    feats = feats.dropna(subset=["target", "gwl"])
    num_cols = feats_mod.NUM_COLS + extra
    parts = feats_mod.split(feats)

    FEATS.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part = part.sort_values(["Station", "time"]).reset_index(drop=True)
        part.to_parquet(FEATS / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows", flush=True)

    prep = {
        "horizons": [HORIZON_LABEL],
        "steps_per_day": feats_mod.STEPS_PER_DAY,
        "horizon_steps": HORIZON_STEPS,
        "num_cols": num_cols,
        "stations": sorted(feats["Station"].astype(str).unique(), key=str),
        "districts": sorted(feats["District"].astype(str).unique(), key=str),
        "split": feats_mod.split.__doc__.strip().split("\n")[0],
        "train_rows": int(len(parts["train"])),
        "val_rows": int(len(parts["val"])),
        "test_rows": int(len(parts["test"])),
        "built_at": str(pd.Timestamp.now())[:19],
    }
    (FEATS / "prep_6h.json").write_text(json.dumps(prep, indent=2))

    # persistence floor on test: carry the current reading forward one step
    tst = parts["test"]
    base = math.sqrt(((tst["target"] - tst["gwl"]) ** 2).mean())
    print(f"one-step persistence RMSE on test (+6h, gwl as-is): {base:.3f} m", flush=True)


if __name__ == "__main__":
    main()