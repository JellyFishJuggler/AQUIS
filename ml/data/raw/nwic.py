"""Parameterized NWIC/NWDP groundwater ingestion for ALL AQUIS districts.

This replaces the original single-resource ``ingestion.py`` with a reusable,
district-parameterized implementation. It supports:
  * any resource_id (archive 2021-2025 or live 2026+, air-temp, etc.)
  * offset/limit pagination (default limit=5000 to bound request count)
  * an optional early window so we can bound pulls (date-bounded, not full history)
  * per-district checkpoint/resume so a long run can be interrupted and resumed
  * type normalization, dedup, chronological sort, missing-record detection
  * robust exception handling with a per-district failure log

The district filter is generated dynamically for every applicable district; it is
never hard-coded to a single example district.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://nwdp.nwic.gov.in/api/3/action/datastore_search"

# Verified working resource IDs.
RES_GROUNDWATER_ARCHIVE = "84bfda45-8ead-436d-9c8e-f7a93ee57522"  # 2021-2025
RES_GROUNDWATER_LIVE = "31c66a49-e110-405d-bcf0-a0ea0d9c8b0c"  # 2026+ (corrected ID)
RES_AIRTEMP_LIVE = "c2c7b5fa-615d-4b95-8a2d-18eb5297f7b0"
RES_AIRTEMP_ARCHIVE = "feb54802-f6cc-4061-b873-8311ae04df0e"
RES_DAILY_RAINFALL = "8752174f-1d17-4aaf-8058-2eb396f50157"

STATE = "Uttar Pradesh"
AGENCY_UPGW = "UPGW"

TIME_FMT = "%d-%m-%Y %H:%M"
TIME_FIELD = "Data Acquisition Time"
GWL_FIELD = "Groundwater Level Telemetry 6 Hourly (meter)"
STATION_FIELD = "Station"
DISTRICT_FIELD = "District"
STATE_FIELD = "State"
AGENCY_FIELD = "Agency"

# API timestamp format sample: "13-09-2023 00:00" (DD-MM-YYYY HH:MM)
OUT_DIR = Path(__file__).resolve().parent.parent / "processed" / "nwic"


# Canonical district spellings actually present in the NWIC archive. Some UP districts
# were renamed; this maps the AQUIS config name to the archive's District value. Values
# verified by probing ``total`` against the archive resource (see data probe).
DISTRICT_ALIASES: dict[str, list[str]] = {
    "KANSIRAM NAGAR": ["KASGANJ"],  # AQUIS spells it 'Kansiram'; archive uses 'KASGANJ'
    "KANSHIRAM NAGAR": ["KASGANJ"],  # tolerate the misspelled variant too
    "SIDDHARTHNAGAR": ["SIDDHARTHNAGAR"],
    # Districts with no UPGW record in the NWIC 2021+ archive (probed total=0 for all
    # plausible spellings, including renamed forms). Kept empty to signal "no coverage".
    "ALLAHABAD": [],
    "PRAYAGRAJ": [],
    "AMETHI (C.S.M. NAGAR)": [],
    "AMROHA (J.P.NAGAR)": [],
    "SAMBHAL": [],
    "G.B. NAGAR": [],
}


def _api_caps_district(name: str) -> str:
    """Convert a raw AQUIS district name to NWIC API-capitalization (uppercased)."""
    s = str(name).strip().upper()
    s = DISTRICT_ALIASES.get(s, [s])
    return s[0] if s else ""


def fetch_district(
    resource_id: str,
    district: str,
    *,
    state: str = STATE,
    agency: str | None = AGENCY_UPGW,
    start: str | None = "2021-01-01",
    end: str | None = None,
    limit: int = 5000,
    max_retries: int = 5,
    sleep: float = 0.2,
) -> list[dict[str, Any]]:
    """Fetch a district's records from a NWIC resource with offset pagination.

    Rows outside the ``[start, end)`` window are dropped client-side (the API does not
    support date filters). Tries each known district alias in order and returns the
    first non-empty result. Returns raw record dicts.
    """
    candidates = [district] + DISTRICT_ALIASES.get(str(district).strip().upper(), [])
    for candidate in candidates:
        rows = _fetch_district_alias(resource_id, candidate, state=state, agency=agency,
                                     start=start, end=end, limit=limit,
                                     max_retries=max_retries, sleep=sleep)
        if rows:
            return rows
    return []


def _fetch_district_alias(
    resource_id: str,
    district: str,
    *,
    state: str,
    agency: str | None,
    start: str | None,
    end: str | None,
    limit: int,
    max_retries: int,
    sleep: float,
) -> list[dict[str, Any]]:
    filt: dict[str, Any] = {"State": state}
    if district:
        filt[DISTRICT_FIELD] = str(district).strip().upper()
    if agency:
        filt[AGENCY_FIELD] = agency

    params = {
        "resource_id": resource_id,
        "filters": json.dumps(filt),
        "limit": limit,
    }

    start_dt = pd.to_datetime(start) if start else None
    end_dt = pd.to_datetime(end) if end else None

    session = requests.Session()
    records: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None

    while True:
        params["offset"] = offset
        result = _get(session, params, max_retries=max_retries)
        if result is None:
            raise RuntimeError(f"NWIC fetch failed for district={district}, resource={resource_id}")

        if total is None:
            total = result.get("total", 0)

        batch = result.get("records", [])
        if not batch:
            break

        for rec in batch:
            ts_str = rec.get("Data Acquisition Time") or rec.get("Date")
            ts = _parse_ts(ts_str)
            if ts is None:
                continue
            if start_dt is not None and ts < start_dt:
                continue
            if end_dt is not None and ts >= end_dt:
                continue
            records.append(rec)

        offset += len(batch)
        if total and offset >= total:
            break
        time.sleep(sleep)

    return records


def _get(session: requests.Session, params: dict, max_retries: int) -> dict | None:
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=(10, 120))
            if resp.status_code == 429:
                raise RuntimeError("rate limited")
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                raise RuntimeError(f"API success=false: {data.get('error')}")
            return data.get("result")
        except Exception as e:  # noqa: BLE001 - robust network handling
            if attempt == max_retries:
                print(f"  [warn] failed after {attempt} attempts for {params.get('offset')}: {e}")
                return None
            time.sleep(attempt * 5)
    return None


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, format=TIME_FMT, errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Cast timestamps/numerics, normalize district, then deduplicate + sort."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    ts = df.get("Data Acquisition Time")
    if ts is not None:
        df["Data Acquisition Time"] = ts.map(_parse_ts)
    date_col = df.get("Date")
    if date_col is not None:
        df["Date"] = date_col.map(_parse_ts)

    for col in (GWL_FIELD, "RL_MSL", "Daily Actual", "Daily Normal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if DISTRICT_FIELD in df.columns:
        df[DISTRICT_FIELD] = df[DISTRICT_FIELD].astype(str).str.strip()

    # Dedup on station + timestamp (and district to avoid cross-district collision).
    keys = [STATION_FIELD, "Data Acquisition Time"]
    if DISTRICT_FIELD in df.columns:
        keys = [DISTRICT_FIELD] + keys
    if any(k in df.columns for k in keys):
        df = df.drop_duplicates(subset=[k for k in keys if k in df.columns])

    time_key = "Data Acquisition Time" if "Data Acquisition Time" in df.columns else "Date"
    if time_key in df.columns:
        df = df.sort_values(time_key).reset_index(drop=True)

    return df


def checkpoint_file(resource_id: str, district: str) -> Path:
    return OUT_DIR / "checkpoints" / f"{resource_id[:8]}_{_api_caps_district(district)}.done"


def fetch_district_normalized(
    resource_id: str,
    district: str,
    *,
    start: str | None,
    end: str | None,
    force: bool = False,
    **kw: Any,
) -> pd.DataFrame:
    """Fetch + normalize a district, honouring a per-district checkpoint (resume-safe)."""
    if not force and checkpoint_file(resource_id, district).exists():
        print(f"  [skip] {district} already fetched for {resource_id[:8]} (use --force to redo)")
        return pd.DataFrame()

    print(f"  fetching {district} from {resource_id[:8]} window=[{start}..{end}] ...")
    records = fetch_district(resource_id, district, start=start, end=end, **kw)
    df = normalize_records(records)
    checkpoint_file(resource_id, district).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file(resource_id, district).touch()
    return df


def get_aquis_districts(parquet_path: Path | str) -> list[str]:
    """List of districts represented in the current AQUIS station configuration."""
    df = pd.read_parquet(parquet_path)
    return sorted(df[DISTRICT_FIELD].dropna().unique().tolist())


def validate_transition(
    archive_df: pd.DataFrame,
    live_df: pd.DataFrame,
    transition_ts: str = "2026-01-01 00:00:00",
) -> dict[str, Any]:
    """Validate the archive->live groundwater transition around the seam.

    Checks that the archive ends at/before and the live begins at/after the transition,
    that there is no unhandled overlap, and reports any gap at the seam. This is a
    validation report; it never fabricates continuity.
    """
    trans = pd.to_datetime(transition_ts)
    report: dict[str, Any] = {
        "transition": transition_ts,
        "archive_last": None,
        "live_first": None,
        "archive_rows": len(archive_df),
        "live_rows": len(live_df),
        "overlap_between": [],
        "seam_gap_hours": None,
        "ok": True,
        "notes": [],
    }

    acol = "Data Acquisition Time"
    if not archive_df.empty and acol in archive_df:
        archive_last = pd.to_datetime(archive_df[acol]).max()
        report["archive_last"] = str(archive_last)
        if archive_last < trans:
            report["seam_gap_hours"] = float((trans - archive_last).total_seconds() / 3600)
            report["notes"].append(
                f"archive ends {archive_last} before {transition_ts} (gap {report['seam_gap_hours']:.0f}h)"
            )
        else:
            report["notes"].append(f"archive extends into transition window (ends {archive_last})")

    if not live_df.empty and acol in live_df:
        live_first = pd.to_datetime(live_df[acol]).min()
        report["live_first"] = str(live_first)
        # Overlap: live records before transition are unexpected.
        pre_trans = live_df[pd.to_datetime(live_df[acol]) < trans]
        if not pre_trans.empty:
            report["overlap_between"].append(str(live_first))
            report["ok"] = False
            report["notes"].append(f"live has {len(pre_trans)} rows before {transition_ts} (overlap)")

    # Seam gap: if archive_last < live_first and both exist.
    if report["archive_last"] and report["live_first"]:
        g = (pd.to_datetime(report["live_first"]) - pd.to_datetime(report["archive_last"])).total_seconds() / 3600
        if g > 0:
            report["seam_gap_hours"] = float(g)
            report["notes"].append(f"seam gap of {g:.0f}h between archive end and live start")

    if report["seam_gap_hours"] and report["seam_gap_hours"] > 24:
        report["ok"] = False
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NWIC district ingestion (archive/live/rainfall)")
    parser.add_argument("--resource", required=True, help="resource_id")
    parser.add_argument("--districts", nargs="*", help="districts (default: all AQUIS districts)")
    parser.add_argument("--parquet", default=str(Path(__file__).resolve().parent.parent / "processed" / "common.parquet"))
    parser.add_argument("--start", default="2021-01-01", help="inclusive start window")
    parser.add_argument("--end", default=None, help="exclusive end window")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--force", action="store_true", help="ignore checkpoints")
    args = parser.parse_args()

    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    districts = args.districts or get_aquis_districts(args.parquet)
    if not districts:
        raise SystemExit("No districts found; cannot run district-parameterized ingestion.")

    all_frames = []
    failures = []
    for d in districts:
        try:
            df = fetch_district_normalized(args.resource, d, start=args.start, end=args.end, force=args.force)
            if not df.empty:
                fname = OUT_DIR / f"{args.resource[:8]}_{_api_caps_district(d)}.parquet"
                df.to_parquet(fname, index=False)
                all_frames.append(df)
                print(f"  -> {len(df)} rows saved to {fname.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {d}: {e}")
            failures.append({"district": d, "error": str(e)})

    if all_frames:
        master = pd.concat(all_frames, ignore_index=True)
        master_path = OUT_DIR / f"{args.resource[:8]}_all_districts.parquet"
        master.to_parquet(master_path, index=False)
        print(f"\nMaster: {len(master)} rows -> {master_path.name}")

    if failures:
        with open(OUT_DIR / f"{args.resource[:8]}_failures.json", "w") as fh:
            json.dump(failures, fh, indent=2, default=str)
        print(f"\nFailures ({len(failures)}): {failures}")