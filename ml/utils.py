"""Shared utilities for the AQUIS ML pipeline."""


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string.

    Examples::

        >>> format_duration(8)
        '8s'
        >>> format_duration(74)
        '1m 14s'
        >>> format_duration(189)
        '3m 09s'
    """
    m, s = divmod(int(seconds), 60)
    if m == 0:
        return f"{s}s"
    return f"{m}m {s:02d}s"
