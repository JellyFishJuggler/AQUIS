"""Compatibility shim: implementation lives in ``pipelines/model.py``.

Kept for callers that still ``import pipeline_model``. New code should import
:mod:`pipelines.model` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.model import *  # noqa: F401,F403,E402
