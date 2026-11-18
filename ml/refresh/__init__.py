"""refresh — AQUIS 6-hourly production refresh pipeline.

Two decoupled cadences (data/refresh/refresh_config.json):

  * FORECAST REFRESH (default 01:00/07:00/13:00/19:00 IST): fetch GWL + drivers,
    rebuild aligned/climatology data, forecast every station from its latest
    observed anchor (never a predicted anchor), then publish atomically.
  * MODEL UPDATE (default daily): full refit on a moving window, honest
    non-overlap candidate vs incumbent backtest on the SAME anchors, promote
    only if every quality gate passes.

The implementation now lives in four modules: refresh_core (config/state/
schedule/publish), refresh_data (sources/features/future_drivers),
refresh_engines (inference/verification/backtest/model_update) and
refresh_service (pipeline/scheduler/CLI). The former module names are kept as
aliases so existing imports and artifacts keep working.
"""
from __future__ import annotations

import sys

from . import refresh_core, refresh_data, refresh_engines, refresh_service

_ALIASES = {
    "config": refresh_core,
    "state": refresh_core,
    "schedule": refresh_core,
    "publish": refresh_core,
    "sources": refresh_data,
    "features": refresh_data,
    "future_drivers": refresh_data,
    "inference": refresh_engines,
    "verification": refresh_engines,
    "backtest": refresh_engines,
    "model_update": refresh_engines,
    "pipeline": refresh_service,
    "scheduler": refresh_service,
}
for _old, _module in _ALIASES.items():
    sys.modules[f"{__name__}.{_old}"] = _module
    globals()[_old] = _module
del _old, _module
