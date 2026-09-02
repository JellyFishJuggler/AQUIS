"""Memory-light check: confirm no station's data is truncated at the tail.

The forecast now anchors on full_pipeline['full'] (the full cleaned series),
NOT train_df (the 80/20 train tail). The only remaining way a station's plotted
observed range could stop short of its true late reading is if
detect_sentinel_values wrongly flags the trailing readings.

Here we verify, for every trained station, that the last valid (non-sentinel)
reading extends to a fresh date (within 90 days of the station's true max) by
reading only the needed columns from common.parquet and re-running the sentinel
scan in isolation (no heavy full_pipeline / station_slugs merge).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pyarrow.parquet as pq  # noqa: E402
import pandas as pd  # noqa: E402

from ml.preprocessing.timeseries import (  # noqa: E402
    TIME_COL, STATION_COL, AGENCY_COL, GWL_COL, load_and_clean,
    station_slug, detect_sentinel_values,
)
from ml.models.xgboost_quantile import ARTIFACTS_DIR  # noqa: E402

MASTER = _ROOT / "ml" / "data" / "processed" / "common.parquet"

# Only the columns needed; avoids loading lat/long/exogenous bloat.
COLS = [STATION_COL, AGENCY_COL, TIME_COL, GWL_COL]


def main() -> None:
    trained = sorted(d.name for d in ARTIFACTS_DIR.iterdir() if d.is_dir())
    print(f"trained stations: {len(trained)}", flush=True)

    pf = pq.ParquetFile(MASTER)
    batches = pf.iter_batches(columns=COLS, batch_size=200_000)
    df = pd.concat(b.to_pandas() for b in batches).reset_index(drop=True)
    # station_slug needs Agency; build the slug map for our columns only.
    pairs = df[[STATION_COL, AGENCY_COL]].drop_duplicates()
    slug_map = {}
    for _, r in pairs.iterrows():
        slug_map[(str(r[STATION_COL]).strip(), str(r[AGENCY_COL]).strip())] = station_slug(
            str(r[STATION_COL]).strip(), str(r[AGENCY_COL]).strip(), 0
        )
    df["slug"] = df.apply(lambda r: slug_map.get((str(r[STATION_COL]).strip(), str(r[AGENCY_COL]).strip())), axis=1)
    df = df.dropna(subset=[STATION_COL, TIME_COL, GWL_COL]).copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    print(f"loaded {len(df)} rows (projected columns only)", flush=True)

    rows = []
    for i, slug in enumerate(trained, 1):
        g = df[df["slug"] == slug]
        if g.empty:
            rows.append({"slug": slug, "ok": False, "reason": "not in common.parquet"})
            continue
        g = g.sort_values(TIME_COL)
        true_max = g[TIME_COL].max()
        # Run sentinel scan on this station's frame directly (mirrors full_pipeline).
        sent = detect_sentinel_values(g[[STATION_COL, TIME_COL, GWL_COL]].copy())
        kept = g[sent]
        kept_max = kept[TIME_COL].max()
        ok = kept_max == true_max or (true_max - kept_max).days <= 90
        rows.append({
            "slug": slug,
            "true_max": str(true_max),
            "kept_max": str(kept_max),
            "n_kept": int(len(kept)),
            "n_dropped": int((~sent).sum()),
            "ok": ok,
            "reason": "" if ok else f"sentinel scan dropped tail (true_max {true_max}, kept_max {kept_max})",
        })
        if i % 20 == 0:
            print(f"  ...{i}/{len(trained)}", flush=True)

    rep = pd.DataFrame(rows)
    print(f"OK: {rep['ok'].sum()} | FAIL: {(~rep['ok']).sum()}")
    bad = rep[~rep["ok"]]
    if len(bad):
        print("FAILURES:\n", bad.to_string())
    out = _ROOT / "ml" / "reports" / "tail_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()