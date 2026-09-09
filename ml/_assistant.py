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

_df: pd.DataFrame | None = None


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
        facts["district_median"] = float(np.median(dg)) if dg.count() else None
        facts["district_n_stations"] = int(df[df[DISTRICT_COL] == dist][STATION_COL].nunique())
        facts["forecast"] = _forecast_summary(station)
        return facts

    def answer(self, question: str, station: str, history: list[dict] | None = None) -> dict:
        facts = self.facts(station)
        if not facts:
            return {
                "answer": f"No telemetry found for station '{station}'.",
                "facts": {},
                "station": station,
            }
        prompt = _build_prompt(question, facts, history=history)
        answer = self._invoke_llm(prompt)
        return {"answer": answer, "facts": facts, "station": station}


# ---------------------------------------------------------------------------
# Prompt (station branch only — ported from ml/agent/data_assistant.py)
# ---------------------------------------------------------------------------
def _build_prompt(question: str, facts: dict, history: list[dict] | None = None) -> str:
    lines = [
        "You are the AQUIS groundwater assistant. Answer the user's question using",
        "ONLY the given observed facts. No speculation. Be concise (max ~6 lines).",
        "Use metres (m) for levels/changes. State dates where known.",
        "Never refuse or say data is unavailable — use the station/district names in the",
        "facts below.",
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
    lines += ["", "FACTS:"]
    lines.append(f"station: {facts['station']}  district: {facts.get('district')}")
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
    lines += ["", f"QUESTION: {question}", "ANSWER:"]
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{'+' if x >= 0 else ''}{x:.3f}"