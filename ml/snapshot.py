"""snapshot.py — PNG export ("Snapshot") of the forecast card.

Renders a dark-theme, shareable dashboard image of the **currently displayed**
station's forecast card with matplotlib (already a project dependency — no new
packages). It captures only the forecast card content: station name, forecast
title, anchor + forecast period, observed history, the q50 trajectory, the
q05–q95 envelope, the "Forecast starts" boundary, confidence, key metrics and
direction. The trajectory v2 forecast is the only forecast rendered.

Pure presentation: the genuine q05/q50/q95 values are passed in and only drawn;
nothing is smoothed, interpolated or fabricated.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import numpy as np
import pandas as pd

# dark theme (matches .streamlit config.toml)
BG = "#0a0a0a"
PANEL = "#14161d"
TEXT = "#e0e0e0"
MUTED = "#9aa0a6"
GRID = "#232834"
COL_OBSERVED = "#6c7683"
COL_TRAJECTORY = "#4ecca3"
COL_ANCHOR = "#e0e0e0"
CONF_COLORS = {"HIGH": "#34d399", "DIRECTIONAL": "#fbbf24", "LOW": "#f472b6"}

FOOTER = "AQUIS · 30-day groundwater outlook"


def _driver_summary(pts: pd.DataFrame) -> str:
    labels = sorted({str(s).strip().lower() for s in pts["driver_source"].astype(str)})
    parts = []
    if any(s.startswith("open-meteo") for s in labels):
        parts.append("Open-Meteo (days 1–16)")
    if any(s.startswith("climatology") for s in labels):
        parts.append("climatology (beyond)")
    if any(s.startswith("driver source unavailable") for s in labels):
        parts.append("limited source (confidence downgraded)")
    return ", ".join(parts) if parts else "n/a"


def trajectory_snapshot_png(*, traj: dict, tail: pd.DataFrame | None = None,
                            station: str = "AQUIS", now=None, dpi: int = 150) -> bytes:
    """Render the trajectory forecast card to PNG bytes.

    ``traj`` — the trajectory_forecast() output dict (values as-is);
    ``tail`` — optional observed history [date, gwl] (daily mean, <= anchor).
    """
    pts = pd.DataFrame(traj["trajectory"])
    if pts.empty:
        raise ValueError("no trajectory points to snapshot")
    for col in ("time", "q05", "q50", "q95", "confidence_level"):
        if col not in pts.columns:
            raise ValueError(f"trajectory missing column {col!r}")
    pts = pts.copy()
    pts["time"] = pd.to_datetime(pts["time"], errors="coerce")
    anchor_t = pd.to_datetime(traj["anchor_time"])
    anchor_gwl = float(traj["anchor_gwl"])
    t30 = traj["trajectory_30d"]
    dir_ = traj["direction"]
    oc = traj["overall_confidence"]
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)

    fig = plt.figure(figsize=(11.4, 6.8), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)

    # observed history (last context window up to the anchor) — muted dashed
    if tail is not None and len(tail) >= 2:
        t = tail.copy()
        t["date"] = pd.to_datetime(t["date"])
        ax.plot(t["date"], t["gwl"], color=COL_OBSERVED, linestyle="--",
                linewidth=1.3, zorder=2, alpha=0.85)

    # q05–q95 uncertainty envelope (subtle translucent band)
    ax.fill_between(pts["time"], pts["q05"], pts["q95"],
                    color=COL_TRAJECTORY, alpha=0.14, linewidth=0, zorder=3)

    # 6-hour q50 trajectory (primary element, genuine values)
    ax.plot(pts["time"], pts["q50"], color=COL_TRAJECTORY, linewidth=2.4, zorder=4)

    # confidence tiers — subtle dots
    for lvl, col in CONF_COLORS.items():
        sub = pts[pts["confidence_level"] == lvl]
        if len(sub):
            ax.scatter(sub["time"], sub["q50"], color=col, s=13, alpha=0.55,
                       marker="o", linewidths=0, zorder=5)

    # "Forecast starts" boundary at the anchor
    ax.axvline(anchor_t, color=COL_ANCHOR, linestyle="--", linewidth=1.0,
               alpha=0.8, zorder=6)
    vals = np.concatenate([pts[["q05", "q50", "q95"]].to_numpy().ravel()] +
                          ([tail["gwl"].to_numpy()] if tail is not None and len(tail) else []))
    top_y = float(np.nanmax(vals))
    ax.text(anchor_t + pd.Timedelta(days=0.15), top_y, "Forecast starts",
            va="top", ha="left", color=MUTED, fontsize=9, zorder=7)

    # axes — clean, sub-horizontal gridlines only, sparse date labels
    ax.set_ylabel("Groundwater level (m)", color=MUTED, fontsize=10)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#3a4152")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.5)
    ax.grid(axis="x", visible=False)

    # compact legend inside the panel
    handles = [
        Line2D([], [], color=COL_OBSERVED, linestyle="--", linewidth=1.3,
               label="Observed"),
        Line2D([], [], color=COL_TRAJECTORY, linewidth=2.4, label="Forecast"),
        Patch(facecolor=COL_TRAJECTORY, alpha=0.22, label="90% uncertainty"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                    fontsize=8, labelcolor=TEXT, ncol=3)
    leg.get_frame().set_facecolor(PANEL)

    fig.subplots_adjust(left=0.075, right=0.97, top=0.78, bottom=0.30)
    fig.text(0.07, 0.86, station, color=TEXT, fontsize=14, weight="bold")
    fig.text(0.07, 0.81, "30-day groundwater outlook · 120 genuine 6-hour forecast points",
             color=MUTED, fontsize=10)
    fig.text(0.07, 0.765,
             f"Anchor {anchor_t:%d %b %Y %H:%M} · Forecast {pts['time'].iloc[0]:%d %b} → "
             f"{pts['time'].iloc[-1]:%d %b %Y}",
             color=MUTED, fontsize=9)

    # metric cards as a clean 3x2 grid
    cells = [
        ("Anchor GWL (m)", f"{anchor_gwl:.2f}"),
        ("+24 h (m)", f"{pts['q50'].iloc[4]:.2f}"),
        ("+7 d (m)", f"{pts['q50'].iloc[27]:.2f}"),
        ("+30 d (m)", f"{t30['level']:.2f}"),
        ("30 d change (m)", f"{t30['change']:+.2f}"),
        ("Confidence", oc["level"]),
    ]
    cols = (0.07, 0.40, 0.72)
    rows = (0.185, 0.095)
    for i, (label, value) in enumerate(cells):
        cx, cy = cols[i % 3], rows[i // 3]
        fig.text(cx, cy + 0.05, label.upper(), color=MUTED, fontsize=7.5)
        fig.text(cx, cy + 0.015, value, color=TEXT, fontsize=13, weight="bold")

    # direction line
    glyph, dlabel, dcolor = {"rising": ("↑", "Rising", "#34d399"),
                             "declining": ("↓", "Falling", "#f472b6"),
                             "stable": ("→", "Stable", "#fbbf24")}.get(
        dir_["label"], ("→", dir_["label"], "#fbbf24"))
    fig.text(0.07, 0.055, f"{glyph} {dlabel} · {dir_['change_q50_30d']:+.2f} m over 30 days",
             color=dcolor, fontsize=10, weight="bold")
    fig.text(0.40, 0.055, f"Confidence · {oc['level']}: {oc.get('reason') or '—'}",
             color=MUTED, fontsize=9)
    fig.text(0.07, 0.028, f"Drivers: {_driver_summary(pts)}", color=MUTED, fontsize=8)

    fig.text(0.955, 0.028, f"{FOOTER} · {now:%Y-%m-%d %H:%M}", ha="right",
             color="#6c7683", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


__all__ = ["trajectory_snapshot_png", "FOOTER", "CONF_COLORS"]