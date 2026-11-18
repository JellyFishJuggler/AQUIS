"""Compatibility shim: implementation lives in ``checks/gate_check.py``.

Kept so ``python gate_check.py`` and ``import gate_check`` keep working. New
code should import :mod:`checks.gate_check`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from checks.gate_check import *  # noqa: F401,F403,E402
from checks.gate_check import main  # noqa: E402

if __name__ == "__main__":
    main()
