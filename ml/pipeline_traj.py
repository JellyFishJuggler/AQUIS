"""Compatibility shim: implementation lives in ``pipelines/traj.py``.

Kept so ``python -m pipeline_traj <stage>`` and ``import pipeline_traj`` keep
working. New code should import :mod:`pipelines.traj` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.traj import *  # noqa: F401,F403,E402
from pipelines.traj import ALIASES, STAGES, main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
