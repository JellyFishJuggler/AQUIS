"""DEPRECATED — superseded by `22_association_audit.py`. Do not run this file.

This path existed briefly and was renamed out of the 30-33 block, because
`30_association_audit.py` sorted immediately before `30_traj_datasets.py` in the
trajectory chain and made the ten-step sequence ambiguous. The copy here also
predates two coordinate hardening fixes (gauge deduplication and rejection of
`(0, 0)` "null island" coordinates), so it can report link counts that differ from
the current script.

Run instead:
    venv/bin/python 22_association_audit.py

This stub refuses to execute, so a stale invocation fails loudly instead of
silently producing the older numbers.
"""
from __future__ import annotations

import sys

_MESSAGE = (
    "30_association_audit.py is DEPRECATED and must not be run.\n"
    "It was renamed to 22_association_audit.py to free the 30-33 block for the\n"
    "trajectory chain (30_traj_datasets, 31_train_traj, 32_backtest_traj,\n"
    "33_traj_reliability), and it lacks the coordinate-hardening fixes.\n\n"
    "Run instead:\n"
    "    venv/bin/python 22_association_audit.py\n"
)


def main() -> None:
    sys.stderr.write(_MESSAGE)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
