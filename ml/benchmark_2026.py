"""Compatibility shim: implementation lives in ``experiments/benchmark_2026.py``.

Kept so ``python benchmark_2026.py`` and ``import benchmark_2026`` keep working.
New code should import :mod:`experiments.benchmark_2026`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.benchmark_2026 import *  # noqa: F401,F403,E402
from experiments.benchmark_2026 import main  # noqa: E402

if __name__ == "__main__":
    main()
