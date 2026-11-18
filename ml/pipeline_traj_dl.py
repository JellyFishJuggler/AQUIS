"""Compatibility shim: implementation lives in ``pipelines/traj_dl.py``.

Kept for callers that still ``import pipeline_traj_dl``. New code should import
:mod:`pipelines.traj_dl` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.traj_dl import *  # noqa: F401,F403,E402
