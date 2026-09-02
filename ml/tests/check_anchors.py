"""Per-station anchor check: verify each trained station's full-pipeline data
extends to its true max timestamp in common.parquet (no truncation).

Fixes the INVESTIGATE requirement: confirm _future_forecast anchors on the
station's actual last observation, not the 80/20 train-split tail.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from ml.preprocessing.timeseries import load_and_clean, station_slugs, full_pipeline  # noqa: E402
from ml.models.xgboost_quantile import ARTIFACTS_DIR  # noqa: E402

MASTER = _ROOT / "ml" / "data" / "processed" / "common.parquet"
BACKEND = _ROOT / "back-end" / "db" / "data.csv"


def main() -> None:
    df = load_and_clean(MASTER)
    df["slug"] = station_slugs(df)
    # True per-station max timestamp from common.parquet (single load).
    true_max = df.groupby("slug")["Data Acquisition Time"].max()
    del df

    rows = []
    slugs = sorted(d.name for d in ARTIFACTS_DIR.iterdir() if d.is_dir())
    for i, slug in enumerate(slugs, 1):
        if slug not in true_max.index:
            rows.append({"slug": slug, "common_max": None, "pipe_max": None, "ok": False, "reason": "not in common.parquet"})
            continue
        pipe = full_pipeline(MASTER, BACKEND, station_slug_filter=slug)
        full = pipe["full"]
        pipe_max = pd.to_datetime(full["Data Acquisition Time"]).max()
        common_max = true_max.get(slug)
        ok = pipe_max == common_max
        rows.append({
            "slug": slug,
            "common_max": str(common_max),
            "pipe_max": str(pipe_max),
            "ok": ok,
            "reason": "" if ok else "full_pipeline max != common.parquet max",
        })
        del pipe, full
        if i % 20 == 0:
            print(f"  ...{i}/{len(slugs)} done", flush=True)

    rep = pd.DataFrame(rows)
    print(f"checked {len(rep)} stations")
    print("OK:", rep["ok"].sum(), "| FAIL:", (~rep["ok"]).sum())
    bad = rep[~rep["ok"]]
    if len(bad):
        print("\nFAILURES:")
        print(bad.to_string())
    else:
        print("\nAll stations anchor on their true last observation.")

    out = _ROOT / "ml" / "reports" / "anchor_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()