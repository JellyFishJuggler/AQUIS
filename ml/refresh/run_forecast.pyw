"""Console-free entry point for the scheduled refresh.

Why this exists
---------------
The scheduled task used to run ``run_forecast.cmd``. That cmd allocates a
console, and a console process is killed with ``STATUS_CONTROL_C_EXIT``
(exit code 3221225786 / 0xC000013A) when the user's console session closes
or the machine sleeps. That is exactly what happened on 2026-11-05 01:00 —
the log showed ``refresh start`` followed by ``^C`` five seconds later.

``pythonw.exe`` never allocates a console, so there is nothing for a
CTRL_C/CTRL_CLOSE event to reach. This wrapper points stdout/stderr at the
same log the old cmd wrote, so nothing is lost.

Run manually exactly like the task does:

    .venv\\Scripts\\pythonw.exe refresh\\run_forecast.pyw
    .venv\\Scripts\\pythonw.exe refresh\\run_forecast.pyw forecast --stations 2

Any arguments are forwarded to ``refresh.cli``; no arguments means
``forecast`` (the scheduled behaviour).
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "refresh" / "run_forecast.log"


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    # python puts the *script's* directory (refresh/) on sys.path, not ml/,
    # so `import refresh.cli` would fail without this.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Open the log ourselves: under pythonw sys.stdout is None, so every
    # print() in refresh.cli would otherwise fail.
    with open(LOG, "a", encoding="utf-8", buffering=1) as fh:
        sys.stdout = fh
        sys.stderr = fh
        print(f"[{_stamp()}] refresh start (pythonw, pid {os.getpid()})")
        code = 1
        try:
            from refresh.cli import main as cli_main

            argv = sys.argv[1:] or ["forecast"]
            code = cli_main(argv) or 0
        except SystemExit as exc:  # argparse / explicit exits
            code = int(exc.code or 0)
        except BaseException:  # noqa: BLE001 - must be logged, not raised
            traceback.print_exc()
            code = 1
        finally:
            print(f"[{_stamp()}] refresh exit {code}")
            fh.flush()
        return code


if __name__ == "__main__":
    raise SystemExit(main())
