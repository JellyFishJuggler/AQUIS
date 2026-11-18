"""CLI shim — `python -m refresh.cli` and run_forecast.pyw stay unchanged."""
from __future__ import annotations

import sys

from .refresh_service import main


if __name__ == "__main__":
    sys.exit(main())
