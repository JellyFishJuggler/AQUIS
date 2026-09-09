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
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

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


def ollama_status() -> dict:
    """Live check: is the Ollama server reachable and the configured model pulled?"""
    model = ollama_model()
    try:
        res = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10)
    except Exception as e:  # noqa: BLE001 - never break the page on a probe
        return {"server": False, "models": set(), "model": model, "error": str(e)}
    models: set[str] = set()
    if res.returncode == 0:
        for line in res.stdout.splitlines()[1:]:
            if line.strip():
                models.add(line.split()[0])
    return {
        "server": res.returncode == 0,
        "models": models,
        "model": model,
        "error": res.stderr.strip() or None,
    }


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

    def _get_llm(self):
        if self._llm is None:
            import ollama

            self._llm = ollama.Client(timeout=None)
        return self._llm

    def _invoke_llm(self, prompt: str) -> str:
        resp = self._get_llm().chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return resp["message"]["content"]

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
        prompt = _build_prompt(question, facts, mentions=mentions, history=history)
        answer = self._invoke_llm(prompt)
        return {"answer": answer, "facts": facts, "station": station,
                "mentions": mentions}


# ---------------------------------------------------------------------------
# Prompt (station branch only — ported from ml/agent/data_assistant.py)
# ---------------------------------------------------------------------------
def _build_prompt(question: str, facts: dict, mentions: list[dict] | None = None,
                  history: list[dict] | None = None) -> str:
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
        "When the user asks about FACTORS or 'what is driving the level': use the",
        "DRIVER CORRELATIONS (Spearman). A strongly negative corr with rain is normal",
        "in aquifer terms only if stated carefully — prefer: 'rain is the dominant",
        "driver here (corr r=-0.xx)' and translate sign into plain meaning.",
        "When asked about RECENT YEARS: compare the ANNUAL rows (mean + total rain).",
        "When asked for PRECAUTIONS / STRATEGY: pick the matching PRECAUTIONS entries",
        "and phrase them as practical suggestions (monitor, conserve, recharge watch).",
        "Never attach totals or changes to a station that are not in its facts.",
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
    lines += ["", f"QUESTION: {question}", "ANSWER:"]
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{'+' if x >= 0 else ''}{x:.3f}"