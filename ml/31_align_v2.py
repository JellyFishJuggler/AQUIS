"""DEPRECATED — superseded by `23_align_v2.py`. Do not run this file.

This path existed briefly and was renamed out of the 30-33 block, because
`31_align_v2.py` collided with `31_train_traj.py` in the trajectory chain. The copy
here also predates the coordinate hardening fix (gauge deduplication and rejection
of `(0, 0)` "null island" coordinates), so it can build links the current script
would reject.

Run instead:
    venv/bin/python 23_align_v2.py

This stub refuses to execute, so a stale invocation fails loudly instead of
silently producing a table from the older association logic.
"""
from __future__ import annotations

import sys

_MESSAGE = (
    "31_align_v2.py is DEPRECATED and must not be run.\n"
    "It was renamed to 23_align_v2.py to free the 30-33 block for the trajectory\n"
    "chain (30_traj_datasets, 31_train_traj, 32_backtest_traj,\n"
    "33_traj_reliability), and it lacks the coordinate-hardening fixes.\n\n"
    "Run instead:\n"
    "    venv/bin/python 23_align_v2.py\n"
)


def main() -> None:
    sys.stderr.write(_MESSAGE)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
