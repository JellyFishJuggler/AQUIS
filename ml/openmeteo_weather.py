"""Compatibility shim: implementation lives in ``pipelines/openmeteo_weather.py``.

Kept so ``python openmeteo_weather.py`` and ``import openmeteo_weather`` keep
working. New code should import :mod:`pipelines.openmeteo_weather`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.openmeteo_weather import *  # noqa: F401,F403,E402
from pipelines.openmeteo_weather import main  # noqa: E402

if __name__ == "__main__":
    main()
