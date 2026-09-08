"""Precompute the fleet-wide forecast snapshot for the Data Assistant.

For every station that already has trained recursive forecast artifacts, run the
same forecast summary the assistant uses per-station and write the results to
``ml/agent/fleet_forecast_snapshot.json``. The assistant's ``_fleet_forecast_scan``
reads this file so users can ask "which stations are forecast to decline >0.3 m
over the next 60 days?" in one shot instead of station-by-station.

Run:  python -m ml.agent.build_fleet_forecast
Idempotent: re-running just updates the snapshot. No training is triggered —
stations without a model are skipped.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.agent.data_assistant import _forecast_summary  # noqa: E402
from ml.models.xgboost_quantile import ARTIFACTS_DIR  # noqa: E402

OUT = Path(__file__).resolve().parent / "fleet_forecast_snapshot.json"


def _one(slug: str):
    t0 = time.time()
    f = _forecast_summary(slug)
    if not f:
        return slug, None
    return slug, {
        "slug": slug,
        "day7_pred": f["day7_pred"],
        "day30_pred": f["day30_pred"],
        "day60_pred": f["day60_pred"],
        "change_30d_pred": f["change_30d_pred"],
        "change_60d_pred": f["change_60d_pred"],
        "direction": f["direction"],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    trained = [d.name for d in (ARTIFACTS_DIR).iterdir() if d.is_dir()]
    if not trained:
        print("no trained models found", file=sys.stderr)
        return 1
    print(f"{len(trained)} trained stations to score", flush=True)
    results: dict[str, dict] = {}
    workers = min(4, len(trained))
    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, s): s for s in trained}
        for fut in as_completed(futs):
            slug = futs[fut]
            slug_res, res = fut.result()
            if res:
                results[slug_res] = res
            done += 1
            print(f"[{done}/{len(trained)}] {slug} {'ok' if res else 'FAIL'}", flush=True)
    OUT.write_text(
        json.dumps(
            {"computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "stations": results},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {OUT} ({len(results)}/{len(trained)} stations), "
        f"took {time.time() - t_start:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())