"""Shared helpers for the ml test suite.

Tests are data-gated: they skip cleanly when a required artifact is missing so the
suite works on a fresh clone. Run with `python -m unittest discover -s tests -v`
from the ml directory (all tests must stay stdlib-only — no pytest dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def has(path: str) -> bool:
    return (ROOT / path).exists()