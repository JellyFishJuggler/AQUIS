"""AQUIS station-locked data assistant (LLM-phrased, local Ollama).

Ported from ``ml/agent/data_assistant.py``, adapted to the ml explorer app:
  * data      -> ``data/aligned/table_6h.parquet`` (Station / time / gwl / District)
  * forecast  -> the app's pooled XGBoost 30-day change model (same as the
                 Forecast page, via ``_model.forward_forecast``)
  * scope     -> one station, pinned (no district / fleet intent routing)
  * model     -> ``llama3.2:3b`` via Ollama (local); overridable with
                 ``AQUIS_OLLAMA_MODEL`` in the repo ``.env`` (same read pattern as
                 ``LULC_STATISTICS_API_KEY`` in 08_lulc).

The LLM only phrases answers from pre-computed, deterministic facts — it never
invents numbers. If Ollama is down, callers can still use ``StationAssistant.facts``
directly (the page shows an instant data panel regardless of LLM availability).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_BASE = Path(__file__).resolve().parent
REPO = _BASE.parent

STATION_COL = "Station"
DISTRICT_COL = "District"
GWL_COL = "gwl"
TIME_COL = "time"

# Drivers present in the aligned 6h table that can explain GWL movement.
DRIVER_COLS = ["rain", "temp", "humidity", "solar", "wind_speed", "pressure",
               "river_level", "canal_level"]

_df: pd.DataFrame | None = None
_station_index: set[str] | None = None
_district_index: set[str] | None = None
_district_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Config (matches 08_lulc.read_env pattern)
# ---------------------------------------------------------------------------
def read_env(repo: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    f = repo / ".env"
    if not f.exists():
        return env
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip("'\"")
    return env


def ollama_model() -> str:
    env = read_env(REPO)
    return (env.get("AQUIS_OLLAMA_MODEL")
            or os.environ.get("AQUIS_OLLAMA_MODEL")
            or "llama3.2:3b").strip()


def ollama_host() -> str:
    env = read_env(REPO)
    host = (env.get("AQUIS_OLLAMA_HOST")
            or os.environ.get("AQUIS_OLLAMA_HOST")
            or "http://127.0.0.1:11434").strip()
    return host.rstrip("/")


def ollama_status() -> dict:
    """Live check: is the Ollama server reachable and the configured model pulled?

    Talks directly to the Ollama HTTP API (no ``ollama`` pip client / CLI
    dependency, so it works from any interpreter that has ``requests``).
    """
    model = ollama_model()
    models: set[str] = set()
    try:
        r = requests.get(f"{ollama_host()}/api/tags", timeout=5)
        r.raise_for_status()
        models = {m.get("name") for m in r.json().get("models", [])}
        server, err = True, None
    except Exception as e:  # noqa: BLE001 - never break the page on a probe
        return {"server": False, "models": models, "model": model, "error": str(e)}
    return {"server": server, "models": models, "model": model,
            "error": err or (f"model '{model}' not pulled" if model not in models else None)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        df = pd.read_parquet(_BASE / "data" / "aligned" / "table_6h.parquet")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        _df = df.sort_values([STATION_COL, TIME_COL]).reset_index(drop=True)
    return _df


# ---------------------------------------------------------------------------
# Cleaning / series stats (ported from ml/agent/data_assistant.py)
# ---------------------------------------------------------------------------
def _clean_series(
    series: pd.Series, times: pd.Series, z_thr: float = 5.0, jump_thr: float = 15.0
) -> tuple[pd.Series, pd.Series, int]:
    """Drop implausible GWL readings (sensor spikes) from a station's series.

    Ported from ``ml/agent/data_assistant._clean_series``: a MAD robust-z filter
    (z < ``z_thr``) plus a jump guard that drops a 6-hourly reading deviating from
    BOTH neighbours by more than ``jump_thr`` m. ``times`` stays parallel to ``vals``.
    """
    vals = series.dropna()
    times = times[vals.index]
    n0 = len(vals)
    if n0 < 3:
        return vals, times, 0
    med = vals.median()
    mad = (vals - med).abs().median()
    if mad > 0:
        z = (0.6745 * (vals - med) / mad).abs()
        keep = z < z_thr
    else:
        q1, q3 = vals.quantile(0.05), vals.quantile(0.95)
        iqr = q3 - q1
        keep = (vals >= q1 - 5 * iqr) & (vals <= q3 + 5 * iqr) if iqr > 0 else pd.Series(True, index=vals.index)
    vals, times = vals[keep], times[keep]
    if len(vals) > 3:
        order = times.argsort()  # index positions in chronological order
        prev = vals.iloc[order].shift(1)
        nxt = vals.iloc[order].shift(-1)
        jump = ((vals.iloc[order] - prev).abs() > jump_thr) & ((vals.iloc[order] - nxt).abs() > jump_thr)
        drop_idx = vals.iloc[order][jump].index
        if len(drop_idx):
            vals = vals.drop(drop_idx)
            times = times.drop(drop_idx)
    return vals, times, n0 - len(vals)


def _series_stats(series: pd.Series, times: pd.Series) -> dict:
    """Facts for a single station's GWL series (``times`` == parallel timestamps)."""
    vals, t, outliers = _clean_series(series, times)
    if vals.empty:
        return {}
    last = float(vals.iloc[-1])
    last_t = t.iloc[-1]

    def _change(days: int) -> float | None:
        thr = last_t - pd.Timedelta(days=days)
        past = vals[t <= thr]
        if past.empty:
            return None
        return float(last - float(past.iloc[-1]))

    return {
        "last": last,
        "last_date": str(last_t.date()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "span": float(vals.max() - vals.min()),
        "n_obs": int(vals.count()),
        "outliers": int(outliers),
        "change_7d": _change(7),
        "change_30d": _change(30),
        "change_60d": _change(60),
        "change_180d": _change(180),
    }


# ---------------------------------------------------------------------------
# Driver impact (which factor explains GWL movement at this station)
# ---------------------------------------------------------------------------
def station_names() -> list[str]:
    global _station_index
    if _station_index is None:
        _station_index = set(get_df()[STATION_COL].astype(str).unique())
    return sorted(_station_index)


def district_names() -> list[str]:
    global _district_index
    if _district_index is None:
        _district_index = set(get_df()[DISTRICT_COL].astype(str).str.strip().unique())
    return sorted(_district_index)


def _driver_correlations(station: str, min_n: int = 30) -> list[dict]:
    """Spearman correlation of each driver vs GWL over the station's 6h history.

    Uses the aligned table aggregate (daily-mean of each col is already aligned by
    the 04b pipeline), so a negative corr means "driver high -> level low" and vice
    versa. Only drivers with >= ``min_n`` co-observed values are reported.
    """
    g = get_df()[get_df()[STATION_COL] == station]
    if g.empty:
        return []
    from scipy.stats import spearmanr

    out = []
    for col in DRIVER_COLS:
        if col not in g.columns:
            continue
        sub = g[[GWL_COL, col]].dropna()
        if len(sub) < min_n or sub[col].nunique() < 2 or sub[GWL_COL].nunique() < 2:
            continue
        r, p = spearmanr(sub[GWL_COL], sub[col])
        if not np.isfinite(r):
            continue
        out.append({"driver": col, "corr": float(r), "p": float(p), "n": int(len(sub))})
    out.sort(key=lambda d: abs(d["corr"]), reverse=True)
    return out


def _rain_recent(station: str) -> dict:
    """Cumulative rain over the last 7 / 30 / 90 days (mm) from the aligned table."""
    g = get_df()[get_df()[STATION_COL] == station]
    g = g[g["rain"].notna()]
    if g.empty:
        return {"rain_7d": None, "rain_30d": None, "rain_90d": None, "last_rain_date": None}
    last = g[TIME_COL].max()
    out: dict = {"last_rain_date": str(last.date())}
    for days in (7, 30, 90):
        recent = g[g[TIME_COL] >= last - pd.Timedelta(days=days)]["rain"]
        out[f"rain_{days}d"] = float(recent.sum()) if len(recent) else None
    return out


def _annual_facts(station: str, last_years: int = 3) -> list[dict]:
    """Per-year mean/min GWL + total rain for the last N calendar years.

    Lets the assistant compare e.g. this year vs last year (which factor moved most).
    """
    g = get_df()[get_df()[STATION_COL] == station]
    if g.empty:
        return []
    g = g.copy()
    g["year"] = g[TIME_COL].dt.year
    rows = []
    for year, sub in g.groupby("year"):
        v = sub[GWL_COL].dropna()
        if v.empty:
            continue
        rain = sub[sub["rain"].notna()]["rain"].sum()
        rows.append({
            "year": int(year),
            "mean": float(v.mean()),
            "min": float(v.min()),
            "max": float(v.max()),
            "rain_mm": float(rain),
            "n": int(v.count()),
        })
    rows.sort(key=lambda r: r["year"])
    for prev, cur in zip(rows, rows[1:]):
        cur["mean_delta_prev"] = round(cur["mean"] - prev["mean"], 3)
        cur["rain_delta_prev"] = round(cur["rain_mm"] - prev["rain_mm"], 1)
    return rows[-last_years:]


# ---------------------------------------------------------------------------
# Temporal records (answer "what was the level on <date / time / month / year>?")
# ---------------------------------------------------------------------------
_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_FULL = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
_MONTH_BLOB = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"


def _mkdate(y: int, m: int, d: int):
    try:
        return pd.Timestamp(y, m, d).date()
    except ValueError:
        return None


def _mon(word: str) -> int | None:
    return _MONTH_NUM.get(word[:3].casefold())


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    out, last = [], 0
    for s, e in sorted(spans):
        out.append(text[last:s])
        last = e
    out.append(text[last:])
    return "".join(out)


def _parse_time_refs(question: str, since) -> list[dict]:
    """Extract date / month / year / relative-time references from a question.

    ``since`` is the station's last-reading date; relative words ("yesterday",
    "last week") and year-less dates resolve against it so answers always match
    what the telemetry actually contains.
    """
    refs: list[dict] = []
    text = question
    spent: list[tuple[int, int]] = []
    flags = re.IGNORECASE

    def _flush() -> None:
        nonlocal text, spent
        text = _strip_spans(text, spent)
        spent = []

    # ISO YYYY-MM-DD
    for m in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text, flags):
        d = _mkdate(int(m[1]), int(m[2]), int(m[3]))
        if d is None or len(refs) >= 4:
            continue
        spent.append(m.span())
        refs.append({"kind": "date", "date": d})
    _flush()

    # "15 March 2023" / "15th March" (day first)
    for m in re.finditer(
            rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_BLOB})\.?[,\s]+((?:19|20)\d{{2}})?(?!\d)",
            text, flags):
        mon = _mon(m[2])
        day = int(m[1])
        if mon is None or not (1 <= day <= 31) or len(refs) >= 4:
            continue
        yr = int(m[3]) if m[3] else _resolve_year_no(mon, day, since)
        d = _mkdate(yr, mon, day)
        if d is None:
            continue
        spent.append(m.span())
        refs.append({"kind": "date", "date": d})
    _flush()

    # "March 15, 2023" / "March 15th" (month first)
    for m in re.finditer(
            rf"(?<!\d)({_MONTH_BLOB})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?[,\s]+((?:19|20)\d{{2}})?(?!\d)",
            text, flags):
        mon = _mon(m[1])
        day = int(m[2])
        if mon is None or not (1 <= day <= 31) or len(refs) >= 4:
            continue
        yr = int(m[3]) if m[3] else _resolve_year_no(mon, day, since)
        d = _mkdate(yr, mon, day)
        if d is None:
            continue
        spent.append(m.span())
        refs.append({"kind": "date", "date": d})
    _flush()

    # "March 2023" (month + year)
    for m in re.finditer(rf"(?<!\w)({_MONTH_BLOB})\.?[,\s]+((?:19|20)\d{{2}})(?!\d)", text, flags):
        mon = _mon(m[1])
        if mon is None or len(refs) >= 4:
            continue
        spent.append(m.span())
        refs.append({"kind": "month", "year": int(m[2]), "month": mon})
    _flush()

    # dd/mm/yyyy or dd-mm-yyyy (day first, India convention) — after verbose patterns
    for m in re.finditer(r"\b(\d{1,2})[-/](\d{1,2})[-/]((?:19|20)\d{2})\b", text, flags):
        day, mon, yr = int(m[1]), int(m[2]), int(m[3])
        d = _mkdate(yr, mon, day)
        if d is None or len(refs) >= 4:
            continue
        spent.append(m.span())
        refs.append({"kind": "date", "date": d})
    _flush()

    # standalone year
    for m in re.finditer(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text, flags):
        if len(refs) >= 4:
            break
        spent.append(m.span())
        refs.append({"kind": "year", "year": int(m[1])})
    _flush()

    # standalone month ("in June") -> most recent such month <= since
    for m in re.finditer(rf"(?<!\w)({_MONTH_BLOB})\.?(?!\w)", text, flags):
        mon = _mon(m[1])
        if mon is None or len(refs) >= 4:
            continue
        spent.append(m.span())
        if mon > since.month:
            yr = since.year - 1
        else:
            yr = since.year
        refs.append({"kind": "month", "year": yr, "month": mon})
    _flush()

    # relative terms (resolved against the station's latest reading)
    low = question.casefold()

    def _has(phrase: str) -> bool:
        return phrase in low

    if len(refs) < 4:
        if _has("day before yesterday") or _has("two days ago"):
            refs.append({"kind": "date", "date": since - pd.Timedelta(days=2)})
        elif _has("yesterday"):
            refs.append({"kind": "date", "date": since - pd.Timedelta(days=1)})
        elif _has("today"):
            refs.append({"kind": "date", "date": since})
    if len(refs) < 4:
        if _has("last week") or _has("past week") or _has("last 7 days"):
            refs.append({"kind": "range", "days": 7, "label": "last week"})
        elif _has("past month") or _has("last 30 days") or _has("past 30 days") or _has("last month."):
            refs.append({"kind": "range", "days": 30, "label": "last 30 days"})
    if len(refs) < 4:
        if _has("this month"):
            refs.append({"kind": "month", "year": since.year, "month": since.month})
        elif _has("last month"):
            refs.append({"kind": "month", "year": since.year if since.month > 1 else since.year - 1,
                         "month": since.month - 1 if since.month > 1 else 12})
        elif _has("this year"):
            refs.append({"kind": "year", "year": since.year})
        elif _has("last year"):
            refs.append({"kind": "year", "year": since.year - 1})
    if len(refs) < 4:
        for name, wd in _WEEKDAYS.items():
            if _has(name):
                off = (since.weekday() - wd) % 7
                refs.append({"kind": "date", "date": since - pd.Timedelta(days=off),
                             "label": name.capitalize()})
                break

    # de-duplicate identical references, keep the most specific first
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in refs:
        if r["kind"] == "date":
            key = ("d", r["date"])
        elif r["kind"] == "month":
            key = ("m", r["year"], r["month"])
        elif r["kind"] == "year":
            key = ("y", r["year"])
        else:
            key = ("r", r.get("days"), r.get("label"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out[:4]


def _resolve_year_no(mon: int, day: int, since) -> int:
    """Year for a year-less date: the most recent occurrence that is <= ``since``."""
    yr = since.year
    if (mon, day) > (since.month, since.day):
        yr -= 1
    return yr


def _rain_str(day: pd.DataFrame) -> str:
    if "rain" not in day.columns:
        return "n/a"
    s = day["rain"]
    return "n/a" if s.notna().sum() == 0 else f"{float(s.sum()):.1f}"


def _date_lines(lines: list[str], g: pd.DataFrame, d, label: str | None = None) -> None:
    label = label or str(d)
    day = g[g[TIME_COL].dt.date == d]
    if day.empty:
        gv = g[g[GWL_COL].notna()][[TIME_COL, GWL_COL]]
        if gv.empty:
            lines.append(f"- {label}: no GWL data for this station.")
            return
        diff = (gv[TIME_COL].dt.normalize() - pd.Timestamp(d)).abs().dt.days
        i = diff.idxmin()
        off = int(diff.loc[i])
        if off > 60:
            lines.append(
                f"- {label}: no reading within ~60 days (station data spans "
                f"{gv[TIME_COL].min().date()} to {gv[TIME_COL].max().date()}).")
            return
        r = gv.loc[i]
        lines.append(
            f"- {label}: no exact reading; nearest is {r[TIME_COL]:%Y-%m-%d %H:%M} "
            f"= {r[GWL_COL]:.2f} m ({off} day(s) away).")
        return
    vals = day[GWL_COL].dropna()
    if vals.empty:
        lines.append(f"- {label}: GWL missing that day; rain {_rain_str(day)} mm.")
        return
    mean = float(vals.mean())
    rdgs = day.dropna(subset=[GWL_COL])[[TIME_COL, GWL_COL]]
    rstr = "; ".join(f"{t:%H:%M}={v:.2f}" for t, v in zip(rdgs[TIME_COL], rdgs[GWL_COL]))
    prev = g[(g[TIME_COL].dt.date == (pd.Timestamp(d) - pd.Timedelta(days=1)).date())][GWL_COL].dropna()
    prev_s = f"; prev day {prev.mean():.2f} m" if not prev.empty else ""
    lines.append(
        f"- {label}: GWL mean {mean:.2f} m over {len(vals)} readings "
        f"(min {float(vals.min()):.2f}, max {float(vals.max()):.2f}); "
        f"rain {_rain_str(day)} mm; readings: {rstr}{prev_s}"
    )


def _month_lines(lines: list[str], g: pd.DataFrame, year: int, month: int,
                 label: str | None = None) -> None:
    label = label or f"{_MONTH_FULL[month]} {year}"
    sub = g[(g[TIME_COL].dt.year == year) & (g[TIME_COL].dt.month == month)]
    vals = sub[GWL_COL].dropna()
    if vals.empty:
        lines.append(f"- {label}: no GWL readings that month.")
        return
    rain = float(sub["rain"].sum()) if "rain" in sub.columns and sub["rain"].notna().sum() else np.nan
    r = f"; rain {rain:.1f} mm" if np.isfinite(rain) else ""
    lines.append(
        f"- {label}: GWL mean {float(vals.mean()):.2f} m over {len(vals)} readings "
        f"(min {float(vals.min()):.2f}, max {float(vals.max()):.2f}){r}."
    )


def _year_lines(lines: list[str], g: pd.DataFrame, year: int) -> None:
    sub = g[g[TIME_COL].dt.year == year]
    vals = sub[GWL_COL].dropna()
    if vals.empty:
        lines.append(f"- {year}: no GWL readings that year.")
        return
    rain = float(sub["rain"].sum()) if "rain" in sub.columns and sub["rain"].notna().sum() else np.nan
    r = f"; rain {rain:.1f} mm" if np.isfinite(rain) else ""
    lines.append(
        f"- {year}: GWL mean {float(vals.mean()):.2f} m over {len(vals)} readings "
        f"(min {float(vals.min()):.2f}, max {float(vals.max()):.2f}){r}."
    )


def _range_lines(lines: list[str], g: pd.DataFrame, days: int, label: str | None = None) -> None:
    label = label or f"last {days} days"
    last = g[TIME_COL].max()
    sub = g[g[TIME_COL] >= last - pd.Timedelta(days=days)]
    rows = []
    for d, day in sub.groupby(sub[TIME_COL].dt.date):
        v = day[GWL_COL].dropna()
        if v.empty:
            continue
        rows.append((d, float(v.mean()), float(v.min()), float(v.max()), len(v)))
    rows.sort()
    if not rows:
        lines.append(f"- {label}: no readings in this window.")
        return
    if days <= 14:
        for d, mean, lo, hi, n in rows:
            r = day_g = sub[sub[TIME_COL].dt.date == d]
            lines.append(
                f"- {d.strftime('%Y-%m-%d')}: GWL mean {mean:.2f} m "
                f"({lo:.2f}..{hi:.2f}, n={n}); rain {_rain_str(day_g)} mm")
    else:
        means = [r[1] for r in rows]
        alo = min(r[2] for r in rows)
        ahi = max(r[3] for r in rows)
        lines.append(
            f"- {label}: {len(rows)} days with data, mean GWL "
            f"{float(np.mean(means)):.2f} m (min {alo:.2f}, max {ahi:.2f}).")
        for d, mean, _lo, _hi, _n in rows[-7:]:
            day_g = sub[sub[TIME_COL].dt.date == d]
            lines.append(
                f"- {d.strftime('%Y-%m-%d')}: GWL mean {mean:.2f} m; "
                f"rain {_rain_str(day_g)} mm")


def _append_ref(lines: list[str], g: pd.DataFrame, ref: dict) -> None:
    kind = ref["kind"]
    if kind == "date":
        _date_lines(lines, g, ref["date"], ref.get("label"))
    elif kind == "month":
        _month_lines(lines, g, ref["year"], ref["month"])
    elif kind == "year":
        _year_lines(lines, g, ref["year"])
    elif kind == "range":
        _range_lines(lines, g, ref["days"], ref.get("label"))


def _temporal_context(station: str, question: str) -> list[str]:
    """Exact observed records for the date/time the user asked about.

    Parses date / month / year / relative-time mentions out of the question and
    returns deterministic facts (daily means, 6-hourly readings, rain) so the LLM
    can answer "what was the level on <when>" without inventing numbers.
    """
    g = get_df()[get_df()[STATION_COL] == station]
    if g.empty or not question or not question.strip():
        return []
    since = g[TIME_COL].max().date()
    refs = _parse_time_refs(question, since)
    lines: list[str] = []
    for ref in refs:
        _append_ref(lines, g, ref)
    return lines


def _fleet_recency(days: int = 14) -> dict:
    """Fleet-wide fact: how many stations have their MOST RECENT reading on each date.

    Lets the LLM answer questions like "how many stations have their latest reading
    on 11 September" with an exact count instead of guessing — the counts are handed
    to it verbatim. Reuses the cached recency Series from ``_utils.station_recency``
    (built once per TTL window on top of ``table_6h``).
    """
    try:
        from _utils import station_recency

        t = station_recency()
        counts = t.dt.normalize().value_counts().sort_index(ascending=False)
        recent = {str(d.date()): int(c) for d, c in counts.head(days).items()}
        return {
            "recent_dates": recent,
            "stations_with_data": int(t.count()),
            "latest_date": str(t.max().date()),
        }
    except Exception:  # noqa: BLE001 - recency is a convenience fact, never fatal
        return {}


def _district_context(district: str, top_k: int = 5) -> dict:
    """District-level context: median level, spread, and the most-stressed stations.

    ``top_k`` most-declining stations over ~180d (by last-observed vs 8 months ago)
    give the assistant concrete "who is worst affected" facts inside the district.
    Cached per district (the table is static within a session).
    """
    global _district_cache
    if district in _district_cache:
        return _district_cache[district]
    df = get_df()
    d = df[df[DISTRICT_COL].astype(str).str.strip() == district]
    if d.empty:
        return {}
    levels = d[GWL_COL].dropna()
    ctx: dict = {
        "median": float(np.median(levels)) if levels.count() else None,
        "mean": float(levels.mean()) if levels.count() else None,
        "min_level": float(levels.min()) if levels.count() else None,
        "max_level": float(levels.max()) if levels.count() else None,
        "n_stations": int(d[STATION_COL].nunique()),
    }
    stressed = []
    for st, sub in d.groupby(STATION_COL):
        v, t, _o = _clean_series(sub[GWL_COL], sub[TIME_COL])
        if v.empty:
            continue
        last = float(v.iloc[-1])
        last_t = t.iloc[-1]
        thr = last_t - pd.Timedelta(days=180)
        past = v[t <= thr]
        chg = float(last - past.iloc[-1]) if len(past) else None
        stressed.append({"station": str(st), "level": last, "change_180d": chg})
    stressed.sort(key=lambda s: (s["change_180d"] if s["change_180d"] is not None else 1e9))
    ctx["n_analysed"] = len(stressed)
    ctx["most_stressed"] = [
        {k: (s[k] if k != "level" else round(s[k], 2))
         for k in ("station", "level", "change_180d")}
        for s in stressed[:top_k] if s.get("change_180d") is not None and s["change_180d"] < 0
    ]
    _district_cache[district] = ctx
    return ctx


# ---------------------------------------------------------------------------
# Precautions / suggested strategies (deterministic rule layer)
# ---------------------------------------------------------------------------
def _precautions(facts: dict) -> list[dict]:
    """Rule-based advisories from the computed facts.

    Each entry carries ``level`` (info / watch / action) + a plain-language
    ``why`` so the LLM can phrase it as a precaution or strategy without inventing
    numbers. Rules stay conservative: one clear reason per precaution.
    """
    out: list[dict] = []
    if not facts:
        return out
    last = facts.get("last")
    change_30d = facts.get("change_30d")
    change_180d = facts.get("change_180d")
    dist_median = facts.get("district_median")
    f = facts.get("forecast") or {}
    rain = facts.get("rain_recent") or {}
    annual = facts.get("annual") or []

    if last is not None and dist_median is not None and last > dist_median + 3.0:
        out.append({
            "level": "watch",
            "title": "Level notably deeper than district median",
            "why": (f"this station reads {last - dist_median:.1f} m below the "
                    f"district median ({dist_median:.1f} m)"),
        })
    if change_180d is not None and change_180d <= -1.0:
        out.append({
            "level": "action",
            "title": "Sustained 6-month decline",
            "why": f"level fell {abs(change_180d):.2f} m over ~180 days",
        })
    elif change_30d is not None and change_30d <= -0.5:
        out.append({
            "level": "watch",
            "title": "Active drawdown in the last month",
            "why": f"level fell {abs(change_30d):.2f} m in the last 30 days",
        })
    if rain.get("rain_30d") is not None and rain["rain_30d"] < 10:
        out.append({
            "level": "watch",
            "title": "Low recent recharge rain",
            "why": f"only {rain['rain_30d']:.0f} mm of rain observed in the last 30 days",
        })
    pf = f.get("change_30d_pred")
    if pf is not None and pf <= -0.5:
        out.append({
            "level": "watch",
            "title": "Model projects further fall",
            "why": f"the 30-day model expects ~{abs(pf):.2f} m decline",
        })
    if f.get("high_uncertainty"):
        out.append({
            "level": "info",
            "title": "Forecast unusually uncertain",
            "why": (f"this station's honest error is {f.get('station_stride_rmse'):.2f} m, "
                    "over 2x the fleet median"),
        })
    if annual and len(annual) >= 2:
        y0, y1 = annual[-2], annual[-1]
        if y1.get("mean_delta_prev") is not None and y1["mean_delta_prev"] <= -1.0:
            out.append({
                "level": "action",
                "title": f"Year-over-year drop ({y0['year']} -> {y1['year']})",
                "why": (f"mean level fell {abs(y1['mean_delta_prev']):.2f} m "
                        f"({y1['mean']:.2f} m now vs {y0['mean']:.2f} m)"),
            })
    return out


# ---------------------------------------------------------------------------
# Forecast (ml pooled 30-day model)
# ---------------------------------------------------------------------------
def _forecast_summary(station: str) -> dict | None:
    from _model import forward_forecast, load_station_summary

    fc = forward_forecast(station)
    if not fc:
        return None
    g = get_df()[get_df()[STATION_COL] == station]
    cvals, _tobs, _o = _clean_series(g[GWL_COL], g[TIME_COL])
    obs_min = float(cvals.min()) if len(cvals) else None
    obs_max = float(cvals.max()) if len(cvals) else None
    change = fc["pred_xgb"]
    day30 = fc["xgb_level"]
    plausible = abs(change) <= 12.0 and (
        obs_min is None or obs_max is None or (obs_min - 25.0 <= day30 <= obs_max + 25.0)
    )
    band_half = fc.get("band_half")
    sm = fc.get("station_stride_rmse")
    med = None
    if sm is not None:
        sums = load_station_summary()
        med = float(sums["xgb_stride_rmse"].median()) if len(sums) else float(sm)
    high_unc = sm is not None and sm > 2.0 * med
    return {
        "anchor": fc["anchor"],
        "day30_pred": day30,
        "change_30d_pred": float(change),
        "direction": "expected rise" if change >= 0 else "expected decline",
        "plausible": bool(plausible),
        "obs_min": obs_min,
        "obs_max": obs_max,
        "band_half": float(band_half) if band_half is not None else None,
        "q05_level": fc.get("q05_level"),
        "q95_level": fc.get("q95_level"),
        "station_stride_rmse": float(sm) if sm is not None else None,
        "high_uncertainty": bool(high_unc),
    }


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------
class StationAssistant:
    """Answer questions about one pinned station from live AQUIS data."""

    def __init__(self, model: str | None = None):
        self.model = model or ollama_model()
        self._llm = None

    def _invoke_llm(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{ollama_host()}/api/chat",
                json={"model": self.model, "stream": False,
                      "messages": [{"role": "user", "content": prompt}],
                      "options": {"temperature": 0.2}},
                timeout=600,
            )
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama server unreachable at {ollama_host()} — "
                f"start it with `ollama serve` and ensure `{self.model}` is pulled "
                f"(`ollama pull {self.model}`). ({e})"
            ) from e
        if not content:
            raise RuntimeError(f"Ollama returned an empty answer for model '{self.model}'.")
        return content

    def facts(self, station: str) -> dict:
        df = get_df()
        g = df[df[STATION_COL] == station]
        if g.empty:
            return {}
        s = _series_stats(g[GWL_COL], g[TIME_COL])
        if not s:
            return {}
        dist = str(g[DISTRICT_COL].iloc[0])
        dg = df[df[DISTRICT_COL] == dist][GWL_COL].dropna()
        facts = {"level": "station", "station": station, "district": dist}
        facts.update(s)
        facts["station_names"] = station_names()
        facts["district_names"] = district_names()
        facts["district_median"] = float(np.median(dg)) if dg.count() else None
        facts["district_n_stations"] = int(df[df[DISTRICT_COL] == dist][STATION_COL].nunique())
        facts["drivers"] = _driver_correlations(station)
        facts["rain_recent"] = _rain_recent(station)
        facts["annual"] = _annual_facts(station)
        facts["district_context"] = _district_context(dist)
        facts["precautions"] = _precautions(facts)
        facts["forecast"] = _forecast_summary(station)
        facts["fleet_recency"] = _fleet_recency()
        return facts

    def facts_for_mentions(self, question: str, base_station: str | None = None) -> list[dict]:
        """Facts for any station/district named in the question (beyond the pinned one).

        Lets the assistant answer comparisons like "how is X doing vs my station" or
        "what does the rain data say for Y". Returns a list of fact dicts (one per
        mention), excluding the pinned station itself to avoid duplication.
        """
        if not question:
            return []
        q = question.casefold()
        found: list[dict] = []
        seen: set[str] = set()
        # exact-ish station mentions (whole name, case-insensitive)
        for name in station_names():
            if base_station and name.casefold() == str(base_station).casefold():
                continue
            if name.casefold() in q:
                if name in seen:
                    continue
                seen.add(name)
                found.append(self.facts(name))
        # district mentions (only when no station already resolved)
        if not found:
            for dist in district_names():
                if base_station and dist.casefold() == str(base_station).casefold():
                    continue
                if dist.casefold() in q:
                    if dist in seen:
                        continue
                    seen.add(dist)
                    ctx = _district_context(dist)
                    if ctx:
                        found.append({
                            "level": "district",
                            "station": f"(district {dist})",
                            "district": dist,
                            "district_context": ctx,
                            "district_median": ctx.get("median"),
                            "district_n_stations": ctx.get("n_stations"),
                        })
        return found

    def answer(self, question: str, station: str, history: list[dict] | None = None) -> dict:
        facts = self.facts(station)
        if not facts:
            return {
                "answer": f"No telemetry found for station '{station}'.",
                "facts": {},
                "station": station,
            }
        mentions = self.facts_for_mentions(question, base_station=station)
        temporal = _temporal_context(station, question)
        prompt = _build_prompt(question, facts, mentions=mentions, history=history,
                               temporal=temporal)
        answer = self._invoke_llm(prompt)
        return {"answer": answer, "facts": facts, "station": station,
                "mentions": mentions, "temporal": temporal}


# ---------------------------------------------------------------------------
# Prompt (station branch only — ported from ml/agent/data_assistant.py)
# ---------------------------------------------------------------------------
def _build_prompt(question: str, facts: dict, mentions: list[dict] | None = None,
                  history: list[dict] | None = None,
                  temporal: list[str] | None = None) -> str:
    lines = [
        "You are the AQUIS groundwater advisory assistant. Answer the user's question",
        "using ONLY the given observed facts. No speculation, no invented numbers.",
        "Be insightful but concise (max ~10 lines unless the user asks for detail).",
        "Use metres (m) for levels/changes. State dates where known.",
        "Never refuse or say data is unavailable — use the station/district names in the",
        "facts below.",
        "",
        "GWL is a water-table LEVEL in metres (not depth-to-water). A RISING value means",
        "the water table rose / recharge; a FALLING value means drawdown.",
        "",
        "When the user asks about a SPECIFIC date, time, month, or year (e.g. 'what was",
        "the level on 15 March 2023', 'what happened last week', 'level in June 2021',",
        "'reading at 06:00 on <date>'): answer ONLY from the HISTORICAL RECORDS block",
        "below — give the exact reading (mean, per-time values, rain) for that moment.",
        "If the asked date has no reading, report the nearest one in the block and say",
        "what it is — never guess a number.",
        "When the user asks about FACTORS or 'what is driving the level': use the",
        "DRIVER CORRELATIONS (Spearman). A strongly negative corr with rain is normal",
        "in aquifer terms only if stated carefully — prefer: 'rain is the dominant",
        "driver here (corr r=-0.xx)' and translate sign into plain meaning.",
        "When asked about RECENT YEARS: compare the ANNUAL rows (mean + total rain).",
        "When asked for PRECAUTIONS / STRATEGY: pick the matching PRECAUTIONS entries",
        "and phrase them as practical suggestions (monitor, conserve, recharge watch).",
        "Never attach totals or changes to a station that are not in its facts.",
        "When asked a FLEET / all-stations question (e.g. 'how many stations have their",
        "latest reading on <date>'): answer from the FLEET RECENT-UPDATE block below —",
        "give the exact station count for that date, or say so if the date is not present",
        "and give the closest listed dates. Never invent a count.",
    ]
    if history:
        lines += [
            "",
            "CONVERSATION (context only):",
        ]
        for msg in history:
            role = "user" if msg.get("role") == "user" else "assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
        lines.append(
            "If old conversation numbers differ from the latest FACTS below, "
            "ALWAYS trust the FACTS."
        )
    lines += ["", f"PINNED STATION: {facts['station']} ({facts.get('district')})"]
    lines.append(f"latest level: {facts['last']:.2f} m on {facts.get('last_date')}")
    lines.append(
        f"range: {facts['min']:.2f} to {facts['max']:.2f} m "
        f"(span {facts['span']:.2f} m, {facts['n_obs']} observations)"
    )
    if facts.get("outliers"):
        lines.append(
            f"note: {facts['outliers']} outlier reading(s) excluded by quality filter")
    c7, c30, c60, c180 = (
        facts.get("change_7d"), facts.get("change_30d"),
        facts.get("change_60d"), facts.get("change_180d"),
    )
    lines.append(
        "level change: 7d=%s | 30d=%s | 60d=%s | 180d=%s m"
        % (_fmt(c7), _fmt(c30), _fmt(c60), _fmt(c180))
    )
    f = facts.get("forecast")
    if f:
        band = f"90% band +/- {f['band_half']:.2f} m" if f.get("band_half") is not None \
            else "90% band unavailable"
        interval = f"; q05/q95 interval {f['q05_level']:.2f}..{f['q95_level']:.2f} m" \
            if f.get("q05_level") is not None else ""
        lines.append(
            f"model forecast (+30 d): level={f['day30_pred']:.2f} m, "
            f"change 30d={f['change_30d_pred']:+.3f} m ({f['direction']}); "
            f"{band} (anchor {f['anchor']:.2f} m){interval}"
        )
        if not f.get("plausible", True):
            lines.append(
                "note: model is UNSTABLE for this station (predictions runaway outside "
                "observed range); treat the forecast numbers above as unreliable, "
                "give only the observed facts"
            )
        if f.get("high_uncertainty") and f.get("station_stride_rmse") is not None:
            lines.append(
                f"note: high-uncertainty station — honest non-overlap RMSE "
                f"{f['station_stride_rmse']:.2f} m (over 2× the fleet median); "
                "stress that the interval is unusually wide here"
            )
    if facts.get("district_median") is not None:
        lines.append(
            f"district median ({facts['district']}): {facts['district_median']:.2f} m "
            f"across {facts['district_n_stations']} stations"
        )
    rain = facts.get("rain_recent") or {}
    if rain.get("rain_7d") is not None or rain.get("rain_30d") is not None:
        lines.append(
            f"recent rain: 7d={_fmt(rain.get('rain_7d'))} | 30d={_fmt(rain.get('rain_30d'))} "
            f"| 90d={_fmt(rain.get('rain_90d'))} mm (last rain "
            f"{rain.get('last_rain_date')})"
        )
    annual = facts.get("annual") or []
    if annual:
        rows = " | ".join(
            f"{a['year']}: mean {a['mean']:.2f} m, rain {a.get('rain_mm', 0):.0f} mm"
            for a in annual)
        lines.append(f"annual history (last {len(annual)} yrs): {rows}")
        for prev, cur in zip(annual, annual[1:]):
            if cur.get("mean_delta_prev") is not None:
                lines.append(
                    f"  {prev['year']}->{cur['year']}: mean level "
                    f"{cur['mean_delta_prev']:+.3f} m, rain delta "
                    f"{cur.get('rain_delta_prev', 0):+.0f} mm")
    drivers = facts.get("drivers") or []
    if drivers:
        top = drivers[:3]
        if top:
            top_s = ", ".join(
                f"{d['driver']} r={d['corr']:+.2f} (n={d['n']})" for d in top)
            lines.append(f"top drivers of level: {top_s}")
    fr = facts.get("fleet_recency") or {}
    if fr.get("recent_dates"):
        lines.append(
            f"FLEET RECENT-UPDATE (all stations): {fr['stations_with_data']} stations "
            f"with data; most recent update date overall: {fr.get('latest_date')}."
        )
        for d, n in fr["recent_dates"].items():
            lines.append(f"  stations whose latest reading is on {d}: {n}")
    dc = facts.get("district_context") or {}
    if dc and dc.get("most_stressed"):
        worst = ", ".join(
            f"{s['station']} ({s['change_180d']:.2f} m/180d)" if s['change_180d'] is not None
            else s['station'] for s in dc["most_stressed"][:3])
        lines.append(f"most-stressed nearby stations (180d): {worst}")
    prec = facts.get("precautions") or []
    if prec:
        lines.append("PRECAUTIONS (rule-based; phrase as suggestions when asked):")
        for p in prec:
            lines.append(f"  [{p['level']}] {p['title']} — {p['why']}")
    # extra stations / districts the user mentioned
    if mentions:
        lines.append("")
        lines.append("ALSO ASKED ABOUT (facts for comparison):")
        for m in mentions:
            if m.get("level") == "district":
                ctx = m.get("district_context") or {}
                lines.append(f"- district {m['district']}: median {m.get('district_median'):.2f} m"
                             f" across {m.get('district_n_stations')} stations"
                             + (f"; most stressed: {', '.join(
                                 s['station'] for s in ctx['most_stressed'][:3])}"
                                if ctx.get("most_stressed") else ""))
            else:
                lines.append(
                    f"- station {m['station']} ({m.get('district')}): "
                    f"latest {m['last']:.2f} m on {m.get('last_date')}, "
                    f"30d change {_fmt(m.get('change_30d'))} m, "
                    f"180d {_fmt(m.get('change_180d'))} m"
                )
        lines.append("Use these only if the question compares or refers to them.")
    if temporal:
        lines.append("")
        lines.append("HISTORICAL RECORDS (observed values for the time the user asked about):")
        lines.extend(temporal)
        lines.append(
            "Answer date/time questions with these exact values only. If a date is not "
            "covered, say the nearest available reading shown here."
        )
    lines += ["", f"QUESTION: {question}", "ANSWER:"]
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{'+' if x >= 0 else ''}{x:.3f}"