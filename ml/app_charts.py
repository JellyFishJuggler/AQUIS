"""app_charts.py — presentation helpers for the Forecast page chart.

Streamlit-free on purpose: these functions take forecast *values* as input and
only decide how they are drawn (colours, line weights, axes, tooltips, layering,
interaction). Nothing here changes, smooths, interpolates, resamples or
re-aggregates a forecast point — the caller passes the genuine 120-step
trajectory exactly as produced by the backend.

Visual contract (matches the app's dark dashboard theme):

* one forecast card: observed history (muted) -> forecast q50 (brand teal) ->
  q05/q95 uncertainty envelope -> a single "Forecast starts" boundary marker;
* clean axes: a handful of readable date labels, sub-horizontal gridlines only;
* a compact 3-item legend (Observed / Forecast / 90% uncertainty) is rendered
  by the page, not by Altair (so the chart carries no detached legend);
* confidence tiers are shown as subtle per-horizon dots; the exact tier is
  preserved in the tooltip and the detail table — never a noisy rainbow.

The trajectory v2 forecast is the only forecast presented — the page
intentionally carries no other forecast model, marker or reference.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# --- dark-theme palette (single source of truth) ----------------------------
BG = "#0a0a0a"
PANEL = "#14161d"
COL_OBSERVED = "#6c7683"
COL_TRAJECTORY = "#4ecca3"   # app primaryColor
COL_ANCHOR = "#e0e0e0"
AXIS_LABEL = "#9aa0a6"
AXIS_TITLE = "#c8cdd3"
GRID = "#232834"
DOMAIN = "#3a4152"

CONF_COLORS = {"HIGH": "#34d399", "DIRECTIONAL": "#fbbf24", "LOW": "#f472b6"}
CONF_ORDER = ["HIGH", "DIRECTIONAL", "LOW"]

TIME_TIP = "%d %b %Y · %H:%M"   # "10 Sep 2026 · 18:00" — tooltip timestamps
AXIS_TIP = "%b %d"              # "Sep 05" — clean axis labels

_Y_TITLE = "Groundwater level (m)"
_OBS_DAYS = 15                  # observed context shown before the anchor


def _xaxis(*, domain_start, domain_end) -> alt.X:
    return alt.X(
        "time:T",
        scale=alt.Scale(domain=[domain_start, domain_end]),
        axis=alt.Axis(format=AXIS_TIP, tickCount=6, labelAngle=0, grid=False,
                      labelColor=AXIS_LABEL, titleColor=AXIS_TITLE,
                      labelFontSize=11, domainColor=DOMAIN, tickColor=DOMAIN,
                      title=None),
    )


def _yaxis_config() -> alt.Axis:
    return alt.Axis(labelColor=AXIS_LABEL, titleColor=AXIS_TITLE,
                    labelFontSize=11, titleFontSize=12, grid=True,
                    gridColor=GRID, gridOpacity=0.4, domain=False,
                    tickColor=DOMAIN)


def _yenco(field: str) -> alt.Y:
    return alt.Y(f"{field}:Q", title=_Y_TITLE, axis=_yaxis_config())


def _walk(d):
    """Yield every (nested) dict in a Vega-Lite spec — used by tests to find
    layers/marks without depending on Altair's internal layer ordering."""
    if isinstance(d, dict):
        yield d
        for v in d.values():
            yield from _walk(v)
    elif isinstance(d, list):
        for v in d:
            yield from _walk(v)


def trajectory_chart(*, pts: pd.DataFrame, tail: pd.DataFrame | None = None,
                     anchor_t, height: int = 480,
                     interactive: bool = True) -> alt.LayerChart:
    """Layered, dark-themed trajectory chart.

    ``pts``      — the genuine 120-step frame [time, q05, q50, q95,
                   confidence_level, driver_source] (never altered here);
    ``tail``     — optional observed history [date, gwl] (daily mean, <= anchor);
    ``anchor_t`` — forecast boundary timestamp (observed -> forecast).
    """
    pts = pts.copy()
    pts["time"] = pd.to_datetime(pts["time"])
    pts["kind"] = "Forecast"
    anchor_t = pd.Timestamp(anchor_t)

    domain_start = anchor_t - pd.Timedelta(days=_OBS_DAYS)
    domain_end = anchor_t + pd.Timedelta(days=30)

    if len(pts):
        top_y = float(pts[["q05", "q50", "q95"]].max().max())
        if tail is not None and len(tail):
            top_y = max(top_y,
                        float(pd.to_numeric(tail["gwl"], errors="coerce").max(skipna=True)))
    else:
        top_y = float("nan")

    layers: list[alt.Chart] = []

    # A) observed history — muted, secondary, last days up to the anchor
    if tail is not None and len(tail) >= 2:
        t = tail.copy()
        t["time"] = pd.to_datetime(t["date"])
        t["kind"] = "Observed"
        layers.append(
            alt.Chart(t)
            .mark_line(color=COL_OBSERVED, strokeDash=[5, 3], strokeWidth=1.4)
            .encode(
                _xaxis(domain_start=domain_start, domain_end=domain_end),
                _yenco("gwl"),
                tooltip=[
                    alt.Tooltip("kind:N"),
                    alt.Tooltip("time:T", title="Timestamp", format=TIME_TIP),
                    alt.Tooltip("gwl:Q", title="GWL", format=".2f"),
                ],
            )
        )

    # B) q05-q95 uncertainty envelope — subtle translucent band
    band = alt.Chart(pts).mark_errorband(
        extent="ci", color=COL_TRAJECTORY, opacity=0.14).encode(
        _xaxis(domain_start=domain_start, domain_end=domain_end),
        alt.Y("q05:Q", title=_Y_TITLE, axis=_yaxis_config()), alt.Y2("q95:Q"),
        tooltip=[
            alt.Tooltip("kind:N"),
            alt.Tooltip("time:T", title="Timestamp", format=TIME_TIP),
            alt.Tooltip("q05:Q", title="q05", format=".2f"),
            alt.Tooltip("q95:Q", title="q95", format=".2f"),
        ],
    )
    layers.append(band)

    # C) forecast q50 — the primary visual element (brand teal, continuous)
    layers.append(
        alt.Chart(pts)
        .mark_line(color=COL_TRAJECTORY, strokeWidth=2.5)
        .encode(
            _xaxis(domain_start=domain_start, domain_end=domain_end),
            _yenco("q50"),
            tooltip=[
                alt.Tooltip("kind:N"),
                alt.Tooltip("time:T", title="Timestamp", format=TIME_TIP),
                alt.Tooltip("q50:Q", title="GWL", format=".2f"),
                alt.Tooltip("q05:Q", title="90% interval · q05", format=".2f"),
                alt.Tooltip("q95:Q", title="90% interval · q95", format=".2f"),
                alt.Tooltip("confidence_level:N", title="Confidence"),
                alt.Tooltip("driver_source:N", title="Driver source"),
            ],
        )
    )

    # confidence per horizon — subtle dots, tier kept in tooltip + table
    layers.append(
        alt.Chart(pts)
        .mark_point(filled=True, size=22, opacity=0.45)
        .encode(
            _xaxis(domain_start=domain_start, domain_end=domain_end),
            _yenco("q50"),
            color=alt.Color(
                "confidence_level:N",
                scale=alt.Scale(domain=CONF_ORDER,
                                range=[CONF_COLORS[c] for c in CONF_ORDER]),
                legend=None,
            ),
        )
    )

    # forecast boundary — one vertical marker, labelled, at the anchor
    layers.append(
        alt.Chart(pd.DataFrame({"time": [anchor_t]}))
        .mark_rule(color=COL_ANCHOR, strokeDash=[6, 4], strokeWidth=1.1)
        .encode(_xaxis(domain_start=domain_start, domain_end=domain_end))
    )
    if top_y == top_y:   # not NaN
        layers.append(
            alt.Chart(pd.DataFrame({"time": [anchor_t], "q50": [top_y]}))
            .mark_text(align="left", dx=8, dy=-10, color=AXIS_LABEL, fontSize=11)
            .encode(_xaxis(domain_start=domain_start, domain_end=domain_end),
                    alt.Y("q50:Q", title=None), text=alt.value("Forecast starts"))
        )

    chart = (
        alt.layer(*layers)
        .properties(height=height)
        .configure(background=BG)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, domainColor=DOMAIN, tickColor=DOMAIN,
                        labelColor=AXIS_LABEL, titleColor=AXIS_TITLE,
                        labelFontSize=11, titleFontSize=12)
        .configure_legend(orient="top", labelColor=AXIS_LABEL, titleColor=AXIS_TITLE)
    )
    if interactive:
        chart = chart.interactive()
    return chart


__all__ = ["trajectory_chart", "CONF_COLORS", "CONF_ORDER", "COL_OBSERVED",
           "COL_TRAJECTORY", "COL_ANCHOR", "BG", "PANEL", "AXIS_LABEL",
           "AXIS_TITLE", "GRID", "DOMAIN", "TIME_TIP", "AXIS_TIP",
           "_OBS_DAYS", "_walk"]