"""18_cwc_river_forecast — CWC / FMISC 7-day river water-level forecast fetch.

Pulls the "07 Days Advance Water Level Forecast of Uttar Pradesh Rivers"
dashboard published by the Central Water Commission (FMISC, Irrigation &
Water Resources Dept., Govt. of UP). The dashboard lists every CWC flood
station on UP rivers (Ganga, Yamuna, Ghaghra, Gomti, Ramganga, Rapti, SAI,
Betwa, Ken, Gandak) with:

  * site metadata            — CWC station, river, district, state
  * reference levels         — warning / danger / highest-flood level (m)
  * latest observed level    — Short-Range Forecast column (datetime, m)
  * day-1 .. day-7 forecast  — each a datetime + flood condition + level (m)

All amounts are water LEVELS in metres at the gauge — exactly the units of
the model's ``river_level`` feature. The forecast is refreshed on the site
roughly every 3 h, so a fetch-per-refresh-cycle gives a self-correcting
7-day river forecast (no future observation is ever used).

This module is dependency-light on purpose: it parses the dashboard with the
standard-library ``html.parser`` and only needs pandas for the parquet cache,
mirroring the decode-once/merge pattern of 20_openmeteo_fetch.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_URL = "http://117.250.2.226:84/FMISCFloodDashboard/up_flood_dashboard.html"
OUT_PARQUET = ROOT / "data" / "cfs" / "river_forecast_cwc.parquet"
OUT_META = ROOT / "data" / "meta" / "river_forecast_cwc_meta.json"

FORECAST_DAYS = 7
DAY_BASE = 11                     # column index of Day-1 (dt, cond, wl) per row
SNAPSHOT_COLS_BASE = {
    "observed_dt": 8, "observed_cond": 9, "observed_wl": 10,
}

RIVER_NAMES = {"BETWA", "GANDAK", "GANGA", "GHAGRA", "GOMTI", "KEN",
               "RAMGANGA", "RAPTI", "SAI", "YAMUNA"}


def _norm(s: object) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def _num(s: object) -> float:
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def _parse_dt(s: str) -> pd.Timestamp:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%y %H:%M"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT


class _CwcTableParser(HTMLParser):
    """Collects the first <table> of the dashboard as raw rows of cells."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_tr = False
        self.cells: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:  # noqa: D102
        if tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._in_tr = True
            self.cells = []
        elif tag in ("th", "td") and self._in_tr:
            self.cells.append("")

    def handle_data(self, data: str) -> None:  # noqa: D102
        if self._in_tr and self.cells:
            self.cells[-1] = (self.cells[-1] + " " + data).strip()

    def handle_endtag(self, tag: str) -> None:  # noqa: D102
        if tag in ("th", "td") and self._in_tr and self.cells:
            self.cells[-1] = self.cells[-1].strip()
        elif tag == "tr" and self._in_tr:
            self.rows.append(list(self.cells))
            self.cells = []
            self._in_tr = False
        elif tag == "table":
            self._in_table = False


def _align_row(row: list[str], header: list[str]) -> list[str] | None:
    """Align a data row to the header's column semantics.

    The dashboard occasionally renders a stray leading empty cell, shifting
    every value by one column. Locate the header's "Sr.No" column and the
    first numeric cell of the data row, then slide the row so the numeric
    cell lands on "Sr.No". Returns the aligned row (padded to >= 33 cells)
    or None when the row is not a station row.
    """
    hr_sr = next((i for i, x in enumerate(header) if "Sr.No" in str(x)), None)
    nums = [k for k, x in enumerate(row) if re.fullmatch(r"\d+", str(x).strip() or "")]
    if not nums:
        return None
    sr_pos = hr_sr if hr_sr is not None else 0
    offset = nums[0] - sr_pos
    if offset > 0:
        row = row[offset:] if offset < len(row) else []
        row = list(row) + [""] * (sr_pos if offset - sr_pos > 0 else 0)
    elif offset < 0:
        row = [""] * (-offset) + list(row)
    while len(row) < 33:
        row.append("")
    return row[:33]


def _row_to_record(row: list[str], header: list[str]) -> dict | None:
    aligned = _align_row(row, header)
    if aligned is None or not re.fullmatch(r"\d+", str(aligned[0]).strip() or ""):
        return None
    rec: dict = {
        "sr_no": int(aligned[0]),
        "site": aligned[1].strip(),
        "river": _norm(aligned[2]),
        "district": _norm(aligned[3]),
        "state": _norm(aligned[4]),
        "wl_c": _num(aligned[5]), "dl_c": _num(aligned[6]), "hfl_c": _num(aligned[7]),
    }
    for name, idx in SNAPSHOT_COLS_BASE.items():
        if name.endswith("_dt"):
            rec[name] = _parse_dt(aligned[idx])
        elif name.endswith("_wl"):
            rec[name] = _num(aligned[idx])
        else:
            rec[name] = str(aligned[idx]).strip()
    for day in range(1, FORECAST_DAYS + 1):
        base = DAY_BASE + 3 * (day - 1)
        rec[f"d{day}_dt"] = _parse_dt(aligned[base])
        rec[f"d{day}_cond"] = str(aligned[base + 1]).strip()
        rec[f"d{day}_wl"] = _num(aligned[base + 2])
    return rec


def parse_html(html: str) -> pd.DataFrame:
    """Parse the dashboard HTML into a long snapshot (one row per CWC site).

    Returns a frame keyed by ``site`` with metadata, observed level and the
    seven day-forecast levels/datetimes (columns ``d1_dt``, ``d1_wl``, ...).
    """
    parser = _CwcTableParser()
    parser.feed(html)
    header = next((r for r in parser.rows if len(r) >= 9 and "Sr.No" in str(r[0])),
                  parser.rows[1] if len(parser.rows) > 1 else [])
    recs = [_row_to_record(r, header) for r in parser.rows]
    recs = [r for r in recs if r]
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["river"] = df["river"].where(df["river"].isin(RIVER_NAMES), df["river"])
    return df


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def fetch_forecast(url: str | None = None, timeout: int = 30,
                   out_parquet: Path | None = None, out_meta: Path | None = None,
                   keep_last: int = 40, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fetch the dashboard, parse it, and persist a rolling snapshot.

    The snapshot keeps ``fetched_at`` history (drop-in for the pattern where a
    failed fetch keeps the previous file and is reported not-ok). Returns
    ``(df, meta)``.
    """
    url = url or DEFAULT_URL
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="replace")
    df = parse_html(html)
    if df.empty:
        raise RuntimeError(f"cwc: no river rows parsed from {url}")

    fetched_at = _utcnow_iso()
    df.insert(0, "fetched_at", fetched_at)

    if out_parquet is not None and not dry_run:
        merged = df
        if out_parquet.exists():
            prev = pd.read_parquet(out_parquet)
            merged = pd.concat([prev, df], ignore_index=True)
            snaps = sorted(merged["fetched_at"].unique())
            if len(snaps) > keep_last:
                merged = merged[merged["fetched_at"].isin(snaps[-keep_last:])]
        merged = merged.reset_index(drop=True)
        _atomic_write_parquet(out_parquet, merged)
        meta = {
            "source": "cwc-fmisc-up-flood-forecast",
            "forecast_days": FORECAST_DAYS,
            "sites": int(df["site"].nunique()),
            "rivers": sorted(df["river"].unique().tolist()),
            "districts": sorted(df["district"].unique().tolist()),
            "fetched_at": fetched_at,
            "cached_rows": int(len(merged)),
        }
        if out_meta is not None:
            _atomic_write_json(out_meta, meta)
        return df, meta

    meta = {
        "source": "cwc-fmisc-up-flood-forecast",
        "forecast_days": FORECAST_DAYS,
        "sites": int(df["site"].nunique()),
        "fetched_at": fetched_at,
    }
    return df, meta


def latest_forecast(out_parquet: Path | None = None) -> pd.DataFrame:
    """Most recent snapshot from the cached parquet (empty frame if none)."""
    path = out_parquet or OUT_PARQUET
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty or "fetched_at" not in df.columns:
        return pd.DataFrame()
    last = df["fetched_at"].max()
    return df[df["fetched_at"] == last].reset_index(drop=True)


def main() -> None:
    df, meta = fetch_forecast(out_parquet=OUT_PARQUET, out_meta=OUT_META)
    print(f"cwc river forecast: {meta['sites']} sites "
          f"({len(meta['rivers'])} rivers, {len(meta['districts'])} districts) "
          f"@ {meta['fetched_at']}")
    return None


if __name__ == "__main__":
    main()