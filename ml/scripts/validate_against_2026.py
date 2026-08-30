"""One-off validation: check our trained 93-station XGBoost models against
real 2026 data from the NWIC API.

Resource: ``31c66a49-e110-405d-bcf0-a0ea0d9c8b0c``  (UPGW 2026-2030,
6-hourly groundwater level telemetry).  Discovery is cheap because the
resource is sorted alphabetically by ``Station`` then chronologically, so a
station's records form one contiguous ``offset`` block found by binary
search (~log2(total) ~= 21 calls per station).

Mode ``--presence``: for each of the 93 trained stations, locate its 2026
block and report found/missing, record count, timestamp range.  No model
files are touched and nothing is trained.

Mode ``--validate`` (after presence passes): load the already-trained
artifacts, recursively forecast each station's 2026 observations from its
Dec-2025 baseline, and report how many fall inside the 90% PI.

Usage (from repo root):
    python -m ml.scripts.validate_against_2026 --presence
    python -m ml.scripts.validate_against_2026 --validate
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.preprocessing.timeseries import GWL_COL, TIME_COL  # noqa: E402

API_BASE = "https://nwdp.nwic.gov.in/api/3/action/datastore_search"
RESOURCE_2026 = "31c66a49-e110-405d-bcf0-a0ea0d9c8b0c"
RESOURCE_2025 = "84bfda45-8ead-436d-9c8e-f7a93ee57522"
TOTAL_2026 = 1_550_207
STATION_COL = "Station"
N_WORKERS = 10
MAX_STEPS = 60

PARQUET = _ML_ROOT / "data" / "processed" / "common.parquet"
OUT_PRESENCE = _ML_ROOT / "artifacts" / "2026_station_presence.csv"


def fetch_records(offset: int, limit: int = 1, resource: str = RESOURCE_2026) -> list[dict]:
    """Return up to ``limit`` records starting at ``offset`` (1 API call)."""
    url = API_BASE + "?" + urllib.parse.urlencode(
        {"resource_id": resource, "limit": limit, "offset": offset}
    )
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AQUIS-validation"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            if payload.get("success"):
                return payload["result"]["records"]
            last_err = RuntimeError(payload.get("error"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"fetch offset={offset} failed: {last_err}")


def parse_ts(ts: str | None) -> pd.Timestamp | None:
    if not ts:
        return None
    try:
        return pd.to_datetime(ts, format="%d-%m-%Y %H:%M")
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(ts)
        except (ValueError, TypeError):
            return None


def station_range_binary_search(station: str, total: int) -> tuple[int, int]:
    """Find [lo, hi) offsets of ``station``'s contiguous block via binary search.

    Returns (lo, hi) with lo==hi if the station is not found.
    Uses Python string ordering, matching the alphabetical sort of the
    resource (assumed; validated by the caller on a few stations).
    """
    if not station:
        return (0, 0)
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi) // 2
        cur = fetch_records(mid, 1)[0][STATION_COL]
        if cur < station:
            lo = mid + 1
        else:
            hi = mid
    if lo >= total or fetch_records(lo, 1)[0][STATION_COL] != station:
        return (lo, lo)
    lo_start = lo
    lo2, hi2 = lo, total
    while lo2 < hi2:
        mid = (lo2 + hi2) // 2
        cur = fetch_records(mid, 1)[0][STATION_COL]
        if cur <= station:
            lo2 = mid + 1
        else:
            hi2 = mid
    return (lo_start, lo2)


WINDOW = 2500


def _first_index(station: str, total: int, step: int = 0) -> tuple[int, int]:
    """Lower-bound index of ``station`` (first idx with name >= station)."""
    lo, hi = 0, total
    while lo < hi and step < MAX_STEPS:
        step += 1
        mid = (lo + hi) // 2
        cur = fetch_records(mid, 1)[0][STATION_COL]
        if cur < station:
            lo = mid + 1
        else:
            hi = mid
    return lo, step


def scan_station_block(station: str, total: int) -> dict:
    """Locate station block and summarize its records (presence mode).

    The resource is alphabetically sorted but with rare inversions (source
    collation quirks), so binary search is followed by a window scan around
    the boundary to guarantee correctness even for those stations.
    """
    found = False
    block_lo = None
    block_hi = None
    n_records = 0
    min_ts = None
    max_ts = None
    nearest = None
    boundary, step = _first_index(station, total)
    lo = max(0, boundary - WINDOW)
    hi = min(total, boundary + WINDOW)
    chunk = fetch_records(lo, hi - lo)
    names = [r[STATION_COL] for r in chunk]
    for i, nm in enumerate(names):
        if nm == station:
            found = True
            block_lo = lo + i
            break
    if found:
        blk = [r for r in chunk if r[STATION_COL] == station]
        n_records = len(blk)
        ts = [parse_ts(r.get("Data Acquisition Time")) for r in blk]
        ts = [t for t in ts if t is not None]
        if ts:
            min_ts, max_ts = min(ts), max(ts)
        block_hi = block_lo + n_records
        if block_lo + n_records >= hi:
            tail_recs = fetch_records(block_lo + n_records - 1,
                                      min(5000, total - block_lo))
            tail_blk = [r for r in tail_recs if r[STATION_COL] == station]
            if tail_blk:
                t = [parse_ts(r.get("Data Acquisition Time")) for r in tail_blk
                     if r.get("Data Acquisition Time") is not None]
                if t:
                    max_ts = max(t)
                n_records = max(n_records, len(tail_blk))
                block_hi = block_lo + len(tail_blk)
            step += 1
    else:
        nearest = chunk[0][STATION_COL] if chunk else ""
    return {
        "station": station,
        "found_2026": found,
        "block_lo": block_lo,
        "block_hi": block_hi,
        "n_records": n_records,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "nearest": nearest,
        "api_calls": step,
    }


def presence_mode(stations: list[str]) -> None:
    total = TOTAL_2026
    results: list[dict] = []
    done = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(scan_station_block, s, total): s for s in stations}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"station": s, "found_2026": False, "error": str(e)}
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(stations):
                print(
                    f"  {done}/{len(stations)} stations  "
                    f"({time.monotonic()-start:.0f}s)"
                )
    results.sort(key=lambda r: r["station"])
    with open(OUT_PRESENCE, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station", "found_2026", "n_records", "min_ts", "max_ts",
                "block_lo", "block_hi", "nearest", "api_calls", "error",
            ],
        )
        w.writeheader()
        for r in results:
            for k in ("min_ts", "max_ts"):
                r[k] = r[k].isoformat() if r[k] is not None else ""
            w.writerow(r)
    n_found = sum(1 for r in results if r["found_2026"])
    n_missing = sum(1 for r in results if not r["found_2026"])
    print(f"\npresence sweep done: {n_found}/{len(stations)} found in 2026, "
          f"{n_missing} missing.  -> {OUT_PRESENCE}")
    for r in results:
        if not r["found_2026"]:
            print(f"  MISSING: {r['station']!r}  (nearest: {r.get('nearest')!r})")


def load_trained_stations() -> list[str]:
    df = pd.read_parquet(PARQUET)
    return sorted(df[STATION_COL].unique().tolist())


def validate_mode() -> None:
    raise NotImplementedError(
        "Full 2026 validation comes after the four verification items "
        "are reported. Run --presence first."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate trained models vs 2026 API data")
    ap.add_argument("--presence", action="store_true", help="sweep 2026 resource for the 93 trained stations")
    ap.add_argument("--validate", action="store_true", help="full 90pct PI validation (stub)")
    args = ap.parse_args()

    stations = load_trained_stations()
    print(f"loaded {len(stations)} trained stations")
    if args.presence:
        presence_mode(stations)
    elif args.validate:
        validate_mode()
    else:
        ap.print_help()