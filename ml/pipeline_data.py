"""Compatibility shim: implementation lives in ``pipelines/data.py``.

Kept so ``python -m pipeline_data <stage>`` and ``import pipeline_data`` keep
working. New code should import :mod:`pipelines.data` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.data import *  # noqa: F401,F403,E402
from pipelines.data import ALIASES, STAGES, main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
