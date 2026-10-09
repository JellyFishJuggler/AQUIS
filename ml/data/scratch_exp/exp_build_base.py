"""exp_build_base — build the shared base feature frame ONCE from the frozen
backup table (backups/20261007_pre_ablation/aligned/table_6h.parquet) and cache
it under data/scratch_exp/tracks/. All Track-1 arms derive their columns from
this single construction so every arm shares byte-identical base features.

Matches the production path exactly: drop_gwl_spikes -> build_full(keep_na=True)
(hydrology block + target/target_d/target_time + st_id/dist_id included).
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]        # ml/
sys.path.insert(0, str(ROOT))
F = importlib.import_module("06_features")

HERE = ROOT / "data" / "scratch_exp"
FROZEN = HERE / "frozen" / "table_6h.parquet"
OUT = HERE / "tracks" / "base_features.parquet"

KEEP_EXTRA = ("rain",)  # raw rain kept for arm-specific cumulative windows


def main() -> None:
    t0 = time.time()
    tbl = pd.read_parquet(FROZEN)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tbl["date"] = tbl["time"].dt.normalize()
    print(f"frozen table: {len(tbl):,} rows / {tbl.Station.nunique():,} st "
          f"{tbl['time'].min()}..{tbl['time'].max()}", flush=True)

    tbl = F.drop_gwl_spikes(tbl)
    feats, extra = F.build_full(tbl, keep_na=True)
    print(f"base feats: {len(feats):,} rows / {feats.Station.nunique():,} st "
          f"(extra_cols={extra})", flush=True)

    cols = [c for c in feats.columns if c in
            ("Station", "District", "date", "time", "target", "target_d",
             "target_time", "st_id", "dist_id") or c in F.NUM_COLS
            or c in F.HYDRO_COLS or c in KEEP_EXTRA] + [c for c in extra if c in feats]
    feats[cols].to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(cols)} cols, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()