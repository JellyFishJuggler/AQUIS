"""_charts — Plotly figures for the hydrology views.

Written because `st.line_chart` anchors its y-axis at zero, which flattens a
groundwater hydrograph into an unusable band: levels sit around −20 m while the
whole signal spans a metre or two, so ~90% of the plot height is empty and the
recession detail is invisible. These builders set the axis range from the data
(with sensible fallbacks), use `connectgaps=False` so telemetry gaps read as
gaps, and layer the forecast band as a filled ribbon.

One shared palette, so 2D/3D map fills and chart series agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- palette -----------------------------------------------------------------
LEVEL = "#4da3ff"          # observed level
CARRIED = "#ffb84d"        # carried-forward slots
BAND = "rgba(120, 200, 255, 0.22)"
MEDIAN = "#8ad4ff"
RAIN = "#5bc8af"
DIM = "rgba(255,255,255,0.35)"


def _figure(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    """Shared layout. Title and legend are stacked, never overlapping.

    The legend sits in the top margin and the title above it, so a titled chart
    needs a taller top margin than an untitled one. Previously both defaulted
    into the same band of pixels, which drew the legend across the title text
    ("Antecedent precipitation index, 2 years" with "API 7 d / API 22 d ..." on
    top of it).
    """
    if title:
        top_margin, legend_y, title_y = 76, 1.02, 0.99
    else:
        top_margin, legend_y, title_y = 34, 1.02, 1.0
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=top_margin, b=8),
        title=dict(text=title, y=title_y, yanchor="top", x=0, xanchor="left",
                   font=dict(size=14)) if title else None,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=legend_y,
                    xanchor="left", x=0, font=dict(size=11)),
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=False, rangeslider=dict(visible=False))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                     zeroline=False)
    return _guard_legend(fig)


def _guard_legend(fig: go.Figure) -> go.Figure:
    """Chart-config guard: no legend entry may render as "undefined".

    A trace that would show in the legend but has no `name` renders as the
    literal "undefined" (or "n" for the empty string) in Plotly. This patches
    the trace config post-hoc, so a builder can never leak a nameless legend
    entry. Traces deliberately hidden from the legend keep `showlegend=False`.
    """
    for tr in fig.data:
        if getattr(tr, "showlegend", None) is not False and not (
                getattr(tr, "name", "") or "").strip():
            tr.name = "series"
    return fig


def _padded_range(vals: pd.Series | np.ndarray, frac: float = 0.10) -> list[float] | None:
    """Data-driven y range. Falls back to None (Plotly auto) when unusable.

    A flat series would otherwise collapse to a zero-height range, so a floor of
    ±0.5 m is enforced once the spread is tiny.
    """
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if v.empty:
        return None
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if not np.isfinite(span):
        return None
    pad = max(span * frac, 0.25)
    return [lo - pad, hi + pad]


# ---------------------------------------------------------------------------
# hydrograph
# ---------------------------------------------------------------------------
def hydrograph(df: pd.DataFrame, *, days: int = 730, height: int = 420,
               show_carried: bool = True, title: str | None = None) -> go.Figure:
    """Observed level with carried-forward slots marked.

    `df` needs a DatetimeIndex and a `level_m` column; an optional
    `carried_forward_m` column marks the telemetry-gap slots.
    """
    x = df[df.index >= (df.index.max() - pd.Timedelta(days=days))] if len(df) else df
    fig = go.Figure()
    if x.empty:
        return _figure(fig, height, title)

    level = pd.to_numeric(x["level_m"], errors="coerce")
    if show_carried and "carried_forward_m" in x.columns:
        car = pd.to_numeric(x["carried_forward_m"], errors="coerce")
        fig.add_trace(go.Scatter(
            x=x.index, y=car, mode="markers", name="carried forward",
            marker=dict(color=CARRIED, size=4, symbol="circle-open"),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>carried %{y:.3f} m<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=x.index, y=level, mode="lines", name="level (m)",
        line=dict(color=LEVEL, width=1.4),
        connectgaps=False,                       # a gap in telemetry is a gap
        hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.3f} m<extra></extra>",
    ))

    fig.update_yaxes(range=_padded_range(level), title_text="level (m)")
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="all"),
            ],
            bgcolor="rgba(255,255,255,0.08)", activecolor="rgba(255,255,255,0.25)",
            font=dict(size=11),
        ),
    )
    return _figure(fig, height, title)


def hydrograph_with_forecast(df: pd.DataFrame, *, days_tail: int = 90,
                             height: int = 440,
                             title: str | None = None) -> go.Figure:
    """Observed tail plus the model's q05/q50/q95 paths as a filled ribbon.

    Expected columns: `level (m)`, `carried forward`, `forecast q05`,
    `forecast median`, `forecast q95` on a shared DatetimeIndex.
    """
    fig = go.Figure()
    if df is None or df.empty:
        return _figure(fig, height, title)

    obs = df[["level (m)"]].dropna() if "level (m)" in df.columns else pd.DataFrame()
    if not obs.empty:
        tail = obs[obs.index >= obs.index.max() - pd.Timedelta(days=days_tail)]
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail["level (m)"], mode="lines", name="observed",
            line=dict(color=LEVEL, width=1.6), connectgaps=False,
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.3f} m<extra></extra>",
        ))
    if "carried forward" in df.columns:
        car = pd.to_numeric(df["carried forward"], errors="coerce").dropna()
        car = car[car.index >= (df.index.max() - pd.Timedelta(days=days_tail + 7))]
        if not car.empty:
            fig.add_trace(go.Scatter(
                x=car.index, y=car, mode="markers", name="carried forward",
                marker=dict(color=CARRIED, size=4, symbol="circle-open"),
                hovertemplate="%{x|%d %b %Y %H:%M}<br>carried %{y:.3f} m<extra></extra>",
            ))

    band = [c for c in ("forecast q05", "forecast q50", "forecast q95")
            if c in df.columns]
    if {"forecast q05", "forecast q95"} <= set(band):
        q05 = pd.to_numeric(df["forecast q05"], errors="coerce")
        q95 = pd.to_numeric(df["forecast q95"], errors="coerce")
        q50 = (pd.to_numeric(df["forecast q50"], errors="coerce")
               if "forecast q50" in df.columns else (q05 + q95) / 2)
        # The band is two traces (upper then lower, filled between). Only ONE of
        # them may carry the legend name — a trace with `name=""` renders as
        # "undefined" in the legend, and naming both duplicates the entry.
        fig.add_trace(go.Scatter(
            x=df.index, y=q95, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=q05, mode="lines", name="90% interval",
            line=dict(width=0), fill="tonexty", fillcolor=BAND,
            hovertemplate="q05 %{y:.3f} m<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=q50, mode="lines", name="forecast median",
            line=dict(color=MEDIAN, width=2.2),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>median %{y:.3f} m<extra></extra>",
        ))

    rng = _padded_range(pd.concat(
        [pd.to_numeric(df[c], errors="coerce") for c in df.columns
         if c.startswith(("level", "forecast"))], ignore_index=True))
    fig.update_yaxes(range=rng, title_text="level (m)")
    return _figure(fig, height, title)


# ---------------------------------------------------------------------------
# rainfall / levels together
# ---------------------------------------------------------------------------
def rain_and_level(rain30: pd.Series, level: pd.Series, *, height: int = 360,
                   days: int = 730) -> go.Figure:
    """Twin-axis view: 30-day rainfall (bars) against observed level (line)."""
    fig = go.Figure()
    r = pd.to_numeric(rain30, errors="coerce").dropna()
    l = pd.to_numeric(level, errors="coerce").dropna()
    if not r.empty:
        r = r[r.index >= r.index.max() - pd.Timedelta(days=days)]
        fig.add_trace(go.Bar(
            x=r.index, y=r, name="rain, 30 d (mm)",
            marker=dict(color=RAIN, opacity=0.55),
            hovertemplate="%{x|%d %b %Y}<br>30 d rain %{y:.1f} mm<extra></extra>",
        ))
    if not l.empty:
        l = l[l.index >= l.index.max() - pd.Timedelta(days=days)]
        fig.add_trace(go.Scatter(
            x=l.index, y=l, name="level (m)", yaxis="y2", mode="lines",
            line=dict(color=LEVEL, width=1.6), connectgaps=False,
            hovertemplate="%{x|%d %b %Y}<br>%{y:.3f} m<extra></extra>",
        ))
    fig.update_layout(
        yaxis=dict(title_text="30 d rain (mm)", showgrid=True,
                   gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title_text="level (m)", overlaying="y", side="right",
                    showgrid=False),
    )
    return _figure(fig, height)


def forecast_bands(df: pd.DataFrame, *, height: int = 320,
                   title: str | None = None) -> go.Figure:
    """Generic interval plot. `df` needs `time`, `q05`, `q50`, `q95`."""
    fig = go.Figure()
    if df is None or df.empty:
        return _figure(fig, height, title)
    d = df.copy()
    if "time" in d.columns:
        d = d.set_index("time")
    fig.add_trace(go.Scatter(x=d.index, y=d["q95"], mode="lines",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.index, y=d["q05"], mode="lines", name="90% interval",
                             line=dict(width=0), fill="tonexty", fillcolor=BAND))
    fig.add_trace(go.Scatter(x=d.index, y=d["q50"], mode="lines", name="median",
                             line=dict(color=MEDIAN, width=2.2)))
    fig.update_yaxes(range=_padded_range(d[["q05", "q95"]].to_numpy().ravel()),
                     title_text="level (m)")
    return _figure(fig, height, title)


# ---------------------------------------------------------------------------
# generic series — data-scaled, for everything that was st.line_chart
# ---------------------------------------------------------------------------
def paired_actual_forecast(frame: pd.DataFrame, *, height: int = 420,
                           title: str | None = None,
                           actual_col: str = "actual",
                           forecast_col: str = "forecast",
                           forecast_name: str = "30-day forecast",
                           daily_median: bool = False) -> go.Figure:
    """Two side-by-side panels: levels on the left, forecast error on the right.

    A single panel cannot carry both. Plotted together, a 6-hourly observed series
    draws hundreds of vertical excursions while the issued forecasts are a handful
    of points per month, so the forecast disappears into the actual's envelope.
    Splitting them gives each its own scale:

      left  — actual and forecast level, so the forecast's *position* is readable;
      right — `forecast − actual`, so its *accuracy* is readable, including
              systematic bias (bars persistently above or below zero).

    Uses `make_subplots`; the left panel keeps the data-scaled y-axis and the
    right is centred on zero.
    """
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=False, column_widths=[0.62, 0.38],
        horizontal_spacing=0.09,
        subplot_titles=("level (m)", "forecast − actual (m)"),
    )
    if frame is None or frame.empty or actual_col not in frame.columns:
        fig.update_layout(height=height, template="plotly_dark")
        return fig

    idx = pd.to_datetime(frame.index)
    a = pd.to_numeric(frame[actual_col], errors="coerce")
    f = (pd.to_numeric(frame[forecast_col], errors="coerce")
         if forecast_col in frame.columns else pd.Series(np.nan, index=frame.index))
    a.index, f.index = idx, idx
    if daily_median:
        a = a.groupby(idx.normalize()).median()
        f = f.groupby(idx.normalize()).median()
    err = (f - a).dropna()

    # ---- left: levels ------------------------------------------------------
    if a.notna().any():
        fig.add_trace(go.Scatter(
            x=a.index, y=a, mode="lines", name="actual",
            line=dict(color=LEVEL, width=1.2), connectgaps=False,
            hovertemplate="%{x|%d %b %Y}<br>actual %{y:.3f} m<extra></extra>",
        ), row=1, col=1)
    if f.notna().any():
        fig.add_trace(go.Scatter(
            x=f.index, y=f, mode="lines+markers", name=forecast_name,
            line=dict(color=CARRIED, width=1.8), marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{x|%d %b %Y}<br>forecast %{y:.3f} m<extra></extra>",
        ), row=1, col=1)
    lvl = pd.concat([a.dropna(), f.dropna()])
    if not lvl.empty:
        fig.update_yaxes(range=_padded_range(lvl), row=1, col=1)

    # ---- right: error ------------------------------------------------------
    if not err.empty:
        fig.add_trace(go.Bar(
            x=err.index, y=err.values, name="forecast − actual",
            marker=dict(color=[CARRIED if v > 0 else LEVEL for v in err.values],
                        opacity=0.85),
            hovertemplate="%{x|%d %b %Y}<br>%{y:+.3f} m<extra></extra>",
        ), row=1, col=2)
        bound = float(np.nanmax(np.abs(err.values))) or 0.1
        fig.update_yaxes(range=[-bound * 1.15, bound * 1.15], row=1, col=2)
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.35)", width=1),
                      row=1, col=2, showlegend=False)
        mean_err = float(err.mean())
        fig.add_hline(y=mean_err, line=dict(color=DIM, width=1, dash="dot"),
                      row=1, col=2, showlegend=False)

    fig.update_layout(
        height=height, barmode="relative", bargap=0.35,
        margin=dict(l=8, r=8, t=76 if title else 48, b=8),
        title=dict(text=title, y=0.99, yanchor="top", x=0, xanchor="left",
                   font=dict(size=14)) if title else None,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                     zeroline=False)
    return _guard_legend(fig)


def series(frame: pd.DataFrame, *, columns: list[str] | None = None,
           height: int = 320, title: str | None = None, colors: list[str] | None = None,
           y_title: str | None = None, markers_for: list[str] | None = None,
           connect_gaps: bool = True, daily_median: bool = False,
           y_range: list[float] | None = None) -> go.Figure:
    """Multi-line chart with a y-axis scaled to the DATA, not to zero.

    `frame` must have a DatetimeIndex. This exists because `st.line_chart`
    anchors at zero, which makes a groundwater hydrograph unreadable when the
    signal is a metre or two sitting at −20 m.

    `daily_median` collapses each series to a calendar-daily median first. Raw
    6-hourly levels oscillate through the day, and that oscillation visually
    swamps the multi-day recession signal in a long-range view.
    """
    fig = go.Figure()
    if frame is None or frame.empty:
        return _figure(fig, height, title)
    cols = columns or [c for c in frame.columns
                       if pd.api.types.is_numeric_dtype(frame[c])]
    palette = colors or [LEVEL, MEDIAN, CARRIED, RAIN, "#d98cff", "#ff8f8f"]
    markers_for = markers_for or []
    for i, c in enumerate(cols):
        if c not in frame.columns:
            continue
        y = pd.to_numeric(frame[c], errors="coerce")
        if daily_median and not y.dropna().empty:
            y = y.groupby(pd.to_datetime(y.index).normalize()).median()
        fig.add_trace(go.Scatter(
            x=y.index, y=y,
            mode="lines+markers" if c in markers_for else "lines",
            name=str(c),
            line=dict(color=palette[i % len(palette)], width=1.8),
            marker=dict(size=6) if c in markers_for else None,
            connectgaps=connect_gaps,
            hovertemplate="%{x|%d %b %Y}<br>%{y:.4f}<extra>" + str(c) + "</extra>",
        ))
    if y_range is None:
        stack = []
        for c in cols:
            if c in frame.columns:
                y = pd.to_numeric(frame[c], errors="coerce")
                if daily_median and not y.dropna().empty:
                    y = y.groupby(pd.to_datetime(y.index).normalize()).median()
                stack.append(y)
        y_range = _padded_range(pd.concat(stack, ignore_index=True)) if stack else None
    if y_range:
        fig.update_yaxes(range=y_range)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return _figure(fig, height, title)


def bars(index: pd.Index, values, *, height: int = 280, title: str | None = None,
         name: str = "count", color: str = LEVEL, y_title: str | None = None) -> go.Figure:
    """Simple bar chart (histograms, distributions, per-fold metrics)."""
    fig = go.Figure()
    v = pd.Series(values)
    if v.empty:
        return _figure(fig, height, title)
    fig.add_trace(go.Bar(x=list(index), y=v.to_numpy(), name=name,
                         marker=dict(color=color),
                         hovertemplate="%{x}<br>%{y:.4g}<extra></extra>"))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return _figure(fig, height, title)


def validation(actual: pd.Series, forecast_median: pd.Series, *,
               q05: pd.Series | None = None, q95: pd.Series | None = None,
               height: int = 400, title: str | None = None,
               daily_median: bool = False) -> go.Figure:
    """Issued-forecast vs realised-actual: actual line + forecast markers + band.

    `daily_median` collapses both series to a calendar-daily median first. The
    raw 6-hourly series oscillates through the day, which visually swamps the
    multi-day signal that actually matters here.
    """
    fig = go.Figure()
    a, f = pd.to_numeric(actual, errors="coerce"), pd.to_numeric(forecast_median, errors="coerce")
    parts = [s for s in (a, f) if s is not None and not s.empty]
    if not parts:
        return _figure(fig, height, title)

    def _maybe_daily(s: pd.Series) -> pd.Series:
        if not daily_median or s.empty:
            return s
        return s.groupby(pd.to_datetime(s.index).normalize()).median()

    a, f = _maybe_daily(a), _maybe_daily(f)
    band_lo = _maybe_daily(pd.to_numeric(q05, errors="coerce")) if q05 is not None else None
    band_hi = _maybe_daily(pd.to_numeric(q95, errors="coerce")) if q95 is not None else None

    if band_hi is not None and band_lo is not None and len(band_hi) and len(band_lo):
        fig.add_trace(go.Scatter(x=band_hi.index, y=band_hi, mode="lines", name="q95",
                                 line=dict(width=0), showlegend=False,
                                 hovertemplate="q95 %{y:.3f} m<extra></extra>"))
        fig.add_trace(go.Scatter(x=band_lo.index, y=band_lo, mode="lines",
                                 name="issued q05–q95", line=dict(width=0),
                                 fill="tonexty", fillcolor=BAND,
                                 hovertemplate="q05 %{y:.3f} m<extra></extra>"))
    if not a.empty:
        fig.add_trace(go.Scatter(
            x=a.index, y=a, mode="lines", name="actual",
            line=dict(color=LEVEL, width=1.6), connectgaps=True,
            hovertemplate="%{x|%d %b %Y}<br>actual %{y:.3f} m<extra></extra>",
        ))
    if not f.empty:
        fig.add_trace(go.Scatter(
            x=f.index, y=f, mode="lines+markers", name="issued forecast (30 d)",
            line=dict(color=CARRIED, width=1.8), marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{x|%d %b %Y}<br>forecast %{y:.3f} m<extra></extra>",
        ))
    stack = [s for s in (a, f, band_lo, band_hi) if s is not None and len(s)]
    if stack:
        fig.update_yaxes(range=_padded_range(pd.concat(stack, ignore_index=True)),
                         title_text="level (m)")
    suffix = " (daily medians)" if daily_median else ""
    return _figure(fig, height, (title or "") + suffix or None)
