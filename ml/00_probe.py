"""00_probe — one-shot metadata probe of every configured NWIC resource.

For each source (archive + live) it records the *total* record count and the
*value column* name (auto-detected when config has no hint). No data is stored;
results go to ml/data/meta/probe.json for later steps.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data.raw.nwic import BASE_URL, _get, _parse_ts  # noqa: E402

import config  # noqa: E402


def probe_resource(session: requests.Session, resource_id: str) -> dict:
    params = {"resource_id": resource_id, "limit": 1}
    result = _get(session, params, max_retries=5)
    if result is None:
        return {"ok": False, "total": None, "fields": [], "sample": None}
    records = result.get("records", [])
    fields = [f.get("id") for f in result.get("fields", [])] or (list(records[0]) if records else [])
    return {
        "ok": True,
        "total": result.get("total"),
        "fields": fields,
        "sample": records[0] if records else None,
    }


_META = config.META_COLS
_NUMERICS = {"integer", "number", "numeric", "float", "int"}


def detect_value_col(fields: list[str], sample: dict | None, hint: str | None) -> str | None:
    """Pick the numeric measurement column, honouring the config hint when present."""
    if hint is not None and hint in fields:
        return hint
    cols = [f for f in fields if f not in _META and "id" not in f.lower()
            and not f.lower().startswith("_")]
    if sample:
        for c in cols:
            v = sample.get(c)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return c
    return cols[0] if cols else None


def main() -> None:
    session = requests.Session()
    probe = {}
    for name, src in config.SOURCES.items():
        if not src.get("enabled"):
            probe[name] = {"enabled": False}
            continue
        print(f"[probe] {name}  archive={src['archive'][:8]} live={src['live'][:8]}")
        arch = probe_resource(session, src["archive"])
        time.sleep(0.3)
        live = probe_resource(session, src["live"])
        time.sleep(0.3)
        arch_col = detect_value_col(arch["fields"], arch["sample"], src.get("value_hint"))
        live_col = detect_value_col(live["fields"], live["sample"], src.get("value_hint"))
        probe[name] = {
            "enabled": True,
            "archive": {
                "id": src["archive"],
                "ok": arch["ok"],
                "total": arch["total"],
                "value_col": arch_col,
            },
            "live": {
                "id": src["live"],
                "ok": live["ok"],
                "total": live["total"],
                "value_col": live_col,
            },
            "agg": src["agg"],
            "kind": src["kind"],
        }
        print(
            f"    archive total={arch['total']} col={arch_col!r} | "
            f"live total={live['total']} col={live_col!r}"
        )

    out = config.META / "probe.json"
    out.write_text(json.dumps(probe, indent=2))
    print(f"\nprobe.json saved -> {out}")


if __name__ == "__main__":
    main()