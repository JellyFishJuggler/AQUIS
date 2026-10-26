"""_traj_dispatch.py - runtime backend selection for the 6-hourly trajectory.

Reads ``data/refresh/runtime.json`` -> ``{"traj_backend": "xgb" | "dl"}`` and
routes to the matching engine. Kept free of Flask/streamlit imports so it can be
unit-tested in the torch-only environment (tests/test_trajectory_dispatch.py).

The DL (quantile GRU) engine is the default so a fresh clone serves it without
any runtime.json. XGBoost remains the fallback: if ``dl`` cannot load (e.g. torch
absent), we log and serve xgb rather than fail the request. Set
``{"traj_backend": "xgb"}`` in runtime.json to pin the incumbent engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "data" / "refresh" / "runtime.json"


def traj_backend() -> str:
    """Active trajectory backend from runtime.json (default ``dl``)."""
    try:
        if RUNTIME.exists():
            return str(json.loads(RUNTIME.read_text()).get("traj_backend", "dl")).lower()
    except Exception:  # noqa: BLE001 - corrupt runtime file must not break the app
        pass
    return "dl"


def active_trajectory_forecast(station: str):
    """Dispatch a trajectory forecast to the selected backend."""
    if traj_backend() == "dl":
        try:
            from _dl import trajectory_forecast as dl_fc
            return dl_fc(station)
        except Exception as e:  # noqa: BLE001
            print(f"[traj_backend=dl] falling back to xgb: {type(e).__name__}: {e}",
                  file=sys.stderr)
    from _trajectory import trajectory_forecast as xgb_fc
    return xgb_fc(station)
