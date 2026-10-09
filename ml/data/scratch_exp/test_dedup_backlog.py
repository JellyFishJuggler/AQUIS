"""Backlog equivalence + timing check for the verification de-dup fix.

Run:  venv/bin/python data/scratch_exp/test_dedup_backlog.py

Works on a COPY of the production ledger (v4-backlog) and replays the de-dup
decision the real verify_forecasts cycle makes over a candidate set derived from
the full 8056-pod forecast_archive, plus every existing ledger key (guaranteed
dups) and some brand-new keys. Asserts the old boolean-scan and the new key-set
keep/skip exactly the same rows, and prints timing before/after.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from refresh import verification  # noqa: E402
from refresh.publish import FORECAST_ARCHIVE  # noqa: E402

SRC_LEDGER = ROOT / "data" / "refresh" / "verification.parquet"
COPY_LEDGER = ROOT / "data" / "scratch_exp" / "ledger_copy.parquet"
HORIZONS = (6, 12, 24, 48, 120, 720)


def _old_dup_exists(ledger: pd.DataFrame, row: dict) -> bool:
    if ledger.empty:
        return False
    d = ledger[(ledger["station"] == row["station"])
               & (ledger["anchor_time"] == row["anchor_time"])
               & (ledger["horizon_h"] == row["horizon_h"])
               & (ledger["target_time"] == row["target_time"])]
    return not d.empty


def build_candidates(ledger: pd.DataFrame, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    files = sorted(FORECAST_ARCHIVE.glob("*.json"))
    for p in files:
        try:
            pod = json.loads(p.read_text())
        except Exception:
            continue
        station = pod.get("station")
        anchor = pod.get("anchor_time")
        if not station or not anchor:
            continue
        a = pd.Timestamp(anchor)
        for h in HORIZONS:
            rows.append({"station": station, "anchor_time": a,
                         "horizon_h": h,
                         "target_time": a + pd.Timedelta(hours=h)})
        if limit and len(rows) >= limit:
            break
    # every existing ledger key (must be skipped) + a few genuinely-new
    for r in ledger.sample(min(2000, len(ledger)), random_state=7).itertuples(index=False):
        rows.append({"station": r.station, "anchor_time": r.anchor_time,
                     "horizon_h": r.horizon_h, "target_time": r.target_time})
    for i in range(200):
        rows.append({"station": f"NEW-{i}", "anchor_time": pd.Timestamp("2026-12-01"),
                     "horizon_h": 6, "target_time": pd.Timestamp("2026-12-01 06:00")})
    return rows


def main() -> int:
    COPY_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger_copy = pd.read_parquet(SRC_LEDGER)
    ledger_copy.to_parquet(COPY_LEDGER, index=False)
    assert ledger_copy.equals(pd.read_parquet(COPY_LEDGER)), "copy mismatch"

    # point the module at the copy so nothing touches production
    verification.VERIF_LEDGER = COPY_LEDGER
    ledger = verification._load_ledger()

    print(f"ledger rows: {len(ledger):,}  (copy @ {COPY_LEDGER})")
    print(f"archive pods: {len(list(FORECAST_ARCHIVE.glob('*.json'))):,}")

    candidates = build_candidates(ledger)
    candidates = candidates  # full backlog scale
    print(f"candidate rows: {len(candidates):,}")

    # --- old: per-row boolean scan of the whole ledger ---
    t0 = time.perf_counter()
    old_kept = []
    for r in candidates:
        if not _old_dup_exists(ledger, r):
            old_kept.append((r["station"], r["anchor_time"], r["horizon_h"],
                             r["target_time"]))
    t_old = time.perf_counter() - t0

    # --- new: build key set once, O(1) membership ---
    t0 = time.perf_counter()
    keyset = verification._ledger_key_set(ledger)
    t_keyset = time.perf_counter() - t0
    t0 = time.perf_counter()
    new_kept = []
    for r in candidates:
        if verification._row_key(r) not in keyset:
            new_kept.append((r["station"], r["anchor_time"], r["horizon_h"],
                             r["target_time"]))
    t_new = time.perf_counter() - t0

    same = old_kept == new_kept
    print(f"\nkept: old={len(old_kept)} new={len(new_kept)}  identical={same}")
    print(f"skipped: {len(candidates) - len(new_kept)}")
    print(f"\ntiming over {len(candidates):,} candidates:")
    print(f"  OLD boolean scan : {t_old:8.3f}s  ({t_old/len(candidates)*1000:.2f} ms/row)")
    print(f"  NEW key-set build: {t_keyset:8.3f}s")
    print(f"  NEW membership   : {t_new:8.3f}s  ({t_new/len(candidates)*1000:.4f} ms/row)")
    print(f"  speedup (scan vs set): {t_old / max(t_new, 1e-9):.0f}x")
    if not same:
        print("EQUIVALENCE FAILED")
        return 1
    print("EQUIVALENCE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
