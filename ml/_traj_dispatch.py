"""_traj_dispatch.py - runtime backend selection for the 6-hourly trajectory.

Reads ``data/refresh/runtime.json`` -> ``{"traj_backend": "xgb" | "dl"}`` and
routes to the matching engine. Kept free of Flask/streamlit imports so it can be
unit-tested in the torch-only environment (tests/test_trajectory_dispatch.py).

Backend policy: the DL (quantile GRU) engine is authoritative. When
``traj_backend`` is ``dl`` and the GRU cannot be loaded (torch missing, weights
or config unreadable, incompatible checkpoint), the dispatcher does **not**
silently serve XGBoost: it returns an ``error`` payload with
``reason="backend_unavailable"`` and ``degraded=True`` so /health, /forecast and
the Streamlit pages surface the failure instead of quietly publishing a
different model's numbers. XGBoost is only used when it is explicitly requested
(``traj_backend: xgb``) or when a caller opts in with ``allow_fallback=True``.
A fresh clone with no runtime.json serves DL.

``backend_status()`` is the cheap, cached loadability probe behind /health and
/fleet/scan.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "data" / "refresh" / "runtime.json"

VALID_BACKENDS = ("dl", "xgb")

TRAJ_DL_CFG = ROOT / "models" / "traj_dl_config.json"
TRAJ_DL_MODEL = ROOT / "models" / "traj_dl.pt"
TRAJ_DL_CALIB = ROOT / "models" / "traj_dl_calibration.json"
TRAJ_DL_RELI = ROOT / "models" / "traj_dl_reliability.json"
ALIGNED_6H = ROOT / "data" / "aligned" / "table_6h.parquet"

_PROBE_CACHE: dict = {}
_PROBE_SIG = None


def traj_backend() -> str:
    """Requested trajectory backend from runtime.json (default ``dl``)."""
    try:
        if RUNTIME.exists():
            return str(json.loads(RUNTIME.read_text()).get("traj_backend", "dl")).lower()
    except Exception:  # noqa: BLE001 - corrupt runtime file must not break the app
        pass
    return "dl"


def _probe_signature():
    """Mtime signature of everything the probe depends on (cache invalidation)."""
    parts = []
    for p in (TRAJ_DL_CFG, TRAJ_DL_MODEL, TRAJ_DL_CALIB, TRAJ_DL_RELI, ALIGNED_6H):
        try:
            parts.append((p.name, p.stat().st_mtime_ns))
        except OSError:
            parts.append((p.name, None))
    return tuple(parts)


def dl_probe(*, use_cache: bool = True) -> dict:
    """Can the GRU trajectory engine actually be loaded in this process?

    Cheap: imports torch, builds the (tiny) GRU and loads the committed state
    dict - no inference, no full-table read. Cached per artifact mtime, so a
    redeploy/retrain invalidates it automatically. Returns
    ``{"ok": bool, "error": str | None, "checked_at": iso}``.
    """
    global _PROBE_SIG
    sig = _probe_signature()
    if use_cache and _PROBE_CACHE and sig == _PROBE_SIG:
        return _PROBE_CACHE

    err = None
    try:
        import importlib

        import torch  # noqa: F401 - the failure mode we are guarding against

        missing = [str(p) for p in (TRAJ_DL_CFG, TRAJ_DL_MODEL,
                                    TRAJ_DL_CALIB, ALIGNED_6H) if not p.exists()]
        if missing:
            raise FileNotFoundError("missing: " + ", ".join(missing))
        cfg = json.loads(TRAJ_DL_CFG.read_text())
        # calibration/reliability are optional at runtime but must parse if present
        if TRAJ_DL_CALIB.exists():
            json.loads(TRAJ_DL_CALIB.read_text())
        if TRAJ_DL_RELI.exists():
            json.loads(TRAJ_DL_RELI.read_text())
        dlmod = importlib.import_module("34_train_traj_dl")
        net = dlmod.TrajGRU(len(cfg["feature_cols"]), hidden=int(cfg["hidden"]))
        net.load_state_dict(torch.load(TRAJ_DL_MODEL, map_location="cpu",
                                       weights_only=True))
        net.eval()
    except Exception as exc:  # noqa: BLE001 - probe turns any failure into status
        err = f"{type(exc).__name__}: {exc}"

    result = {"ok": err is None, "error": err,
              "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _PROBE_CACHE.clear()
    _PROBE_CACHE.update(result)
    _PROBE_SIG = sig
    return result


def backend_status(*, probe: bool = True) -> dict:
    """Resolved backend for status endpoints.

    Returns ``requested`` (runtime.json), ``backend`` (engine the dispatcher
    serves), ``degraded`` (requested dl but the GRU is not loadable) and the
    load ``error`` when there is one.
    """
    requested = traj_backend()
    if requested == "dl":
        probe_out = dl_probe() if probe else {"ok": True, "error": None}
        return {
            "requested": requested,
            "backend": "dl",
            "degraded": not probe_out["ok"],
            "error": probe_out.get("error"),
        }
    if requested == "xgb":
        return {"requested": requested, "backend": "xgb",
                "degraded": False, "error": None}
    return {
        "requested": requested,
        "backend": "xgb",
        "degraded": False,
        "error": None,
        "fallback_reason": f"unknown traj_backend {requested!r} -> xgb",
    }


def active_trajectory_forecast(station: str, *, allow_fallback: bool = False):
    """Dispatch a trajectory forecast to the selected backend.

    With ``traj_backend=dl`` a load/scoring failure is returned as an error
    payload (``reason="backend_unavailable"``, ``degraded=True``); it never
    silently switches to XGBoost unless the caller passes ``allow_fallback``.
    """
    requested = traj_backend()
    if requested == "dl":
        try:
            from _dl import trajectory_forecast as dl_fc
            return dl_fc(station)
        except Exception as e:  # noqa: BLE001 - surface, never hide
            detail = f"{type(e).__name__}: {e}"
            print(f"[traj_backend=dl] GRU trajectory engine unavailable: {detail}",
                  file=sys.stderr)
            if not allow_fallback:
                return {
                    "error": (f"traj_backend=dl but the GRU trajectory engine "
                              f"failed to load: {detail}"),
                    "reason": "backend_unavailable",
                    "backend": "dl",
                    "degraded": True,
                }
            print("[traj_backend=dl] explicit allow_fallback -> xgb", file=sys.stderr)
    from _trajectory import trajectory_forecast as xgb_fc
    return xgb_fc(station)
