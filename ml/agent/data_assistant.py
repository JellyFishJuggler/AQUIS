"""AQUIS Data Assistant — answers questions about observed trends/forecasts.

Unlike the docs-RAG assistant, this answers from the LIVE AQUIS data itself:
trends, latest levels and (where a trained model exists) short-horizon forecasts
for a district or a single station.

Design is intentionally deterministic + reliable:
  1. Resolve the target (district or station) from the prompt/selection.
  2. Compute observed facts from ``common.parquet`` (latest level, change over
     7/30/180 days, per-station counts, district median, rising vs falling).
  3. If the selected *station* has trained forecast artifacts, add a 30-day
     forecast summary.
  4. Ask the open-source LLM (llama3.2:3b via Ollama + LangChain) to phrase a
     concise, factual answer from ONLY those facts — no free-form agent calls.

Rising/falling convention: GWL values are water-table elevations in metres, so
"rising level" = GWL increased, "declining" = GWL decreased.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing.timeseries import load_and_clean, station_slug

_ML_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = _ML_ROOT / "data" / "processed" / "common.parquet"
ARTIFACTS_DIR = _ML_ROOT / "artifacts"
BACKEND_CSV = _ML_ROOT.parent / "back-end" / "db" / "data.csv"

GWL_COL = "Groundwater Level Telemetry 6 Hourly (meter)"
TIME_COL = "Data Acquisition Time"
STATION_COL = "Station"
DISTRICT_COL = "District"
AGENCY_COL = "Agency"
SLNO_COL = "SlNo"


# ---------------------------------------------------------------------------
# Loading / helpers
# ---------------------------------------------------------------------------
_df: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = load_and_clean(PARQUET_PATH)
    return _df


def _match_names(query: str) -> dict:
    """Return {'district': str|None, 'station': str|None, 'station_slug': str|None}."""
    df = _get_df()
    q = query.lower().strip()
    out = {"district": None, "station": None, "station_slug": None}
    if not q:
        return out

    districts = sorted(df[DISTRICT_COL].dropna().astype(str).unique())
    stations = sorted(df[STATION_COL].dropna().astype(str).unique())

    # Prefer an exact station-name mention (more specific than a district).
    for s in stations:
        if s.lower() in q:
            out["station"] = s
            break
    if out["station"] is None:
        for d in districts:
            if d.lower() in q:
                out["district"] = d
                break
    else:
        row = df[df[STATION_COL] == out["station"]].iloc[0]
        out["district"] = str(row[DISTRICT_COL])
        out["station_slug"] = station_slug(row[STATION_COL], row[AGENCY_COL], row[SLNO_COL])
    return out


def _clean_series(
    series: pd.Series, times: pd.Series, z_thr: float = 5.0, jump_thr: float = 15.0
) -> tuple[pd.Series, pd.Series, int]:
    """Drop implausible GWL readings (sensor spikes) from a station's series.

    Two robust guards:
      1. MAD filter — drops values with a large robust z-score vs the station median
         (catches glitches like reading 1004 m at a ~-3 m well).
      2. Jump guard — drops a 6-hourly reading that deviates from BOTH neighbours by
         more than ``jump_thr`` m.

    Returns (vals, times, n_dropped) with ``times`` kept parallel to ``vals``.
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
    """Facts for a single station's GWL series (times == parallel timestamps)."""
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
        "last_ts": str(last_t),
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


def _ensure_forecast_model(station_slug_str: str) -> float | None:
    """Train the station's forecast model on demand if missing; returns train seconds.

    Uses the exact same per-station training path as `ml/training/train_forecast.py`
    (writes real models + metadata + interpretability into ml/artifacts/<slug>), so a
    forecast is available for ANY station — not only the fleet-retrained ones.
    """
    import subprocess
    import sys
    import time

    art = ARTIFACTS_DIR / station_slug_str
    if (art / "recursive" / "xgb_point.joblib").exists():
        return None
    root = _ML_ROOT.parent
    t0 = time.time()
    try:
        res = subprocess.run(
            [sys.executable, "-u", "-m", "ml.training.train_forecast", "--station", station_slug_str],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=900,
            env=None,
        )
    except Exception:  # noqa: BLE001
        return None
    if res.returncode != 0:
        return None
    return time.time() - t0


def _forecast_summary(station_slug_str: str) -> dict | None:
    """Forecast summary (~next 2 months) from a (trained) recursive model.

    Trains the station on demand if no model exists yet, so the assistant can always
    give a forecast estimate. Returns None only if training/forecasting truly fails.
    Adds ``plausible`` = False when recursive predictions blow up (error
    accumulation), so callers can flag unstable models instead of quoting nonsense.
    """
    try:
        from ml.models.xgboost_quantile import load_models, predict_recursive
        from ml.preprocessing.timeseries import prepare_feature_matrix
        from ml.preprocessing.timeseries import GWL_COL as GC

        train_secs = _ensure_forecast_model(station_slug_str)

        art = ARTIFACTS_DIR / station_slug_str
        if not (art / "recursive" / "xgb_point.joblib").exists():
            return None

        from ml.preprocessing.timeseries import full_pipeline
        pipe = full_pipeline(PARQUET_PATH, BACKEND_CSV, station_slug_filter=station_slug_str)
        if not pipe:
            return None
        fc = pipe["feature_cols"]
        tail = pipe.get("full", pipe["train"]).tail(1)
        X, _, _ = prepare_feature_matrix(tail[fc + [GC]])
        if len(X) == 0:
            return None
        models = load_models(art)
        rec = predict_recursive(models, X[0], 62, fc, models.get("error_correction"))
        pts = np.asarray(rec["point"])
        if len(pts) < 60:
            return None
        chg60 = float(pts[59] - pts[0])
        obs_row = pipe.get("full", pipe["train"])
        obs_min = float(obs_row[GC].min()) if GC in obs_row.columns and obs_row[GC].count() else None
        obs_max = float(obs_row[GC].max()) if GC in obs_row.columns and obs_row[GC].count() else None
        plausible = abs(chg60) <= 12.0 and (
            obs_min is None or obs_max is None or
            (obs_min - 25.0 <= float(pts[59]) <= obs_max + 25.0)
        )
        return {
            "day7_pred": float(pts[6]),
            "day30_pred": float(pts[29]),
            "day60_pred": float(pts[59]),
            "change_30d_pred": float(pts[29] - pts[0]),
            "change_60d_pred": chg60,
            "direction": "expected rise" if chg60 >= 0 else "expected decline",
            "plausible": bool(plausible),
            "obs_min": obs_min,
            "obs_max": obs_max,
            "trained_on_demand": train_secs is not None,
            "train_secs": train_secs,
        }
    except Exception:  # noqa: BLE001 - forecast is optional, never fail the answer
        return None


# ---------------------------------------------------------------------------
# Fleet-wide tools (cross-station / cross-district questions)
# ---------------------------------------------------------------------------
SNAPSHOT_PATH = _ML_ROOT / "agent" / "fleet_forecast_snapshot.json"
_rank_cache: dict[int, dict] = {}


def _district_recovery_rank(window_days: int = 60, top: int = 5,
                            min_stations: int = 3) -> dict:
    """Rank districts by median observed level change over ``window_days``.

    "Recovery" = rising level (GWL up); "decline" = falling. Computed on CLEANED
    per-station series, median across each district's stations. Memoized by window.
    """
    global _rank_cache
    if window_days in _rank_cache:
        return _rank_cache[window_days]
    df = _get_df()
    rows: list[tuple[str, int, float, str]] = []
    for d, g in df.groupby(DISTRICT_COL):
        lst = pd.Timestamp(g[TIME_COL].max())
        thr = lst - pd.Timedelta(days=window_days)
        changes = []
        for _st, sg in g.groupby(STATION_COL):
            vals, t, _o = _clean_series(sg[GWL_COL], sg[TIME_COL])
            if vals.empty:
                continue
            past = vals[t <= thr]
            if not past.empty:
                changes.append(float(vals.iloc[-1] - float(past.iloc[-1])))
        if len(changes) >= max(1, min_stations):
            rows.append((str(d), len(changes), float(np.median(changes)), str(lst.date())))
    rows.sort(key=lambda r: (r[2], r[1]), reverse=True)  # median change desc
    result = {
        "window_days": window_days,
        "as_of": str(pd.Timestamp(_get_df()[TIME_COL].max()).date()),
        "districts": [(d, n, float(c), dt) for d, n, c, dt in rows[:top]],
    }
    _rank_cache[window_days] = result
    return result


def _fleet_forecast_scan(district: str | None = None, threshold: float = 0.3,
                         horizon: int = 60) -> dict:
    """Stations whose model forecast predicts a level DECLINE >= ``threshold`` m.

    Reads the precomputed snapshot (built by `ml/agent/build_fleet_forecast.py`).
    Predictions are VERIFIED against the station's observed level range before being
    reported: the recursive model occasionally blows up (error accumulation yields
    absurd day-60 levels), and those stations are returned under ``unreliable`` so
    the LLM never quotes nonsense. Returns {"available": bool, ...}.
    """
    if not SNAPSHOT_PATH.exists():
        return {"available": False, "matches": [], "threshold": threshold}
    try:
        data = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - snapshot must never break the chat
        return {"available": False, "matches": [], "threshold": threshold}
    stations = data.get("stations", {})
    df = _get_df()

    # One groupby pass -> slug -> station name (optionally restricted to a district).
    slug2name: dict[str, str] = {}
    dsel = df[df[DISTRICT_COL] == district] if district else df
    for st, g in dsel.groupby(STATION_COL):
        r = g.iloc[0]
        slug2name[station_slug(st, r[AGENCY_COL], r[SLNO_COL])] = str(st)

    ch_key = f"change_{horizon}d_pred"
    day_key = f"day{horizon}_pred"
    cand: list[dict] = []
    for slug, f in stations.items():
        if district and slug not in slug2name:
            continue
        ch = f.get(ch_key)
        if ch is None:
            continue
        if ch <= -abs(threshold):
            cand.append({
                "name": slug2name.get(slug, slug),
                "slug": slug,
                "change": float(ch),
                "day_end": float(f.get(day_key, 0.0)),
            })
    if not cand:
        return {
            "available": True, "matches": [], "n_matches": 0, "unreliable": [],
            "threshold": float(threshold), "horizon": int(horizon),
            "as_of": data.get("computed_at", ""),
        }

    # Verify candidates against observed level ranges (only the candidate subset).
    sub = df[df[STATION_COL].isin({c["name"] for c in cand})]
    obs: dict[str, tuple[float | None, float | None]] = {}
    for st, g in sub.groupby(STATION_COL):
        v, t, _o = _clean_series(g[GWL_COL], g[TIME_COL])
        if v.empty:
            continue
        obs[str(st)] = (float(v.min()), float(v.max()))

    matches, unreliable = [], []
    for c in cand:
        mn, mx = obs.get(c["name"], (None, None))
        if mn is None or mx is None or (
            abs(c["change"]) <= 12.0 and (mn - 25.0 <= c["day_end"] <= mx + 25.0)
        ):
            matches.append(c)
        else:
            unreliable.append(c)
    matches.sort(key=lambda m: m["change"])
    unreliable.sort(key=lambda m: m["change"])
    return {
        "available": True,
        "matches": matches[:12],
        "n_matches": len(matches),
        "unreliable": unreliable[:5],
        "n_unreliable": len(unreliable),
        "threshold": float(threshold),
        "horizon": int(horizon),
        "as_of": data.get("computed_at", ""),
    }


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------
class DataAssistant:
    """Answer data queries about AQUIS districts/stations using the LLM."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_ollama import ChatOllama

            self._llm = ChatOllama(model=self.model, temperature=0.2)
        return self._llm

    def answer(self, question: str, station: str | None = None,
               station_slug: str | None = None) -> dict:
        """Answer a question about AQUIS data.

        Pass ``station`` (display name) to LOCK the answer to that one station: the
        assistant only talks about that station (no district/fleet matching, no
        switching targets). Without it, the pre-existing name-matching behaviour
        (incl. fleet-wide ranking/forecast-scan intents) applies.
        """
        q = (question or "").lower()
        df = _get_df()

        if station:
            sel = df[df[STATION_COL] == station]
            if sel.empty:
                return {
                    "answer": f"No telemetry found for station '{station}'.",
                    "facts": {},
                    "match": {"station": station, "district": None, "station_slug": None},
                }
            row = sel.sort_values(TIME_COL).iloc[-1]
            from ml.preprocessing.timeseries import station_slug as _make_slug
            m = {
                "station": str(row[STATION_COL]),
                "district": str(row[DISTRICT_COL]),
                "station_slug": station_slug or _make_slug(
                    row[STATION_COL], row[AGENCY_COL], row[SLNO_COL]),
            }
            facts = self._collect_facts(df, m)
        else:
            m = _match_names(question)

            # --- Special intents that need fleet-wide tools -----------------
            cross_district = ("which district" in q or "districts " in q or " district" in q) and any(
                w in q for w in ("recover", "rise", "risen", "rises", "declin", "fall", "drop",
                                 "fastest", "highest", "most")
            )
            scan_forecast = "forecast" in q and any(w in q for w in ("stations", "station")) and any(
                w in q for w in ("declin", "fall", "drop")
            )
            if cross_district:
                facts = _district_recovery_rank(window_days=60)
                facts["level"] = "rank_district"
                m = {"station": None, "district": None}
            elif scan_forecast:
                thr = 0.3
                for tok in re.findall(r"\d+(?:\.\d+)?", q):
                    v = float(tok)
                    if 0.05 <= v <= 10:
                        thr = v
                        break
                district = m.get("district") if m.get("district") else None
                facts = _fleet_forecast_scan(district=district, threshold=thr)
                facts["level"] = "forecast_scan"
                facts["n_scan"] = 0 if not facts.get("available") else None
                m = {"station": None, "district": district}
            else:
                if not m["station"] and not m["district"]:
                    return {
                        "answer": "I couldn't find that district or station in the AQUIS data. "
                                  "Try naming a district (e.g. 'Allahabad') or a station name.",
                        "facts": {},
                        "match": m,
                    }
                facts = self._collect_facts(df, m)

        if not facts:
            return {"answer": "No usable data found for that target.", "facts": {}, "match": m}
        prompt = self._build_prompt(question, facts)
        answer = self._get_llm().invoke(prompt).content
        return {"answer": answer, "facts": facts, "match": m}

    def fast_facts(self, station: str | None = None, district: str | None = None) -> dict:
        """Deterministic facts for a station OR district — fast, no LLM, no forecast.

        Returns a plain dict of numbers/labels suitable for direct rendering. Raises
        nothing; returns an empty dict if the target is unknown.
        """
        df = _get_df()
        facts: dict[str, object] = {}
        if station:
            g = df[df[STATION_COL] == station]
            s = _series_stats(g[GWL_COL], g[TIME_COL])
            if not s:
                return facts
            facts["level"] = "station"
            facts["station"] = station
            facts["district"] = str(g[DISTRICT_COL].iloc[0])
            facts.update(s)
            dg = df[df[DISTRICT_COL] == facts["district"]][GWL_COL].dropna()
            facts["district_median"] = float(np.median(dg)) if dg.count() else None
            facts["district_n_stations"] = int(df[df[DISTRICT_COL] == facts["district"]][STATION_COL].nunique())
            return facts
        if district:
            dsel = df[df[DISTRICT_COL] == district]
            facts["level"] = "district"
            facts["district"] = district
            per_st: dict[str, dict] = {}
            for st, g in dsel.groupby(STATION_COL):
                s = _series_stats(g[GWL_COL], g[TIME_COL])
                if s:
                    per_st[str(st)] = s
            if not per_st:
                return facts
            latest = [v["last"] for v in per_st.values()]
            rising = sum(1 for v in per_st.values() if (v.get("change_30d") or 0) > 0)
            falling = sum(1 for v in per_st.values() if (v.get("change_30d") or 0) < 0)
            facts["n_stations"] = len(per_st)
            facts["rising_30d"] = rising
            facts["falling_30d"] = falling
            facts["district_median_latest"] = float(np.median(latest))
            facts["district_min"] = float(min(latest))
            facts["district_max"] = float(max(latest))
            facts["stations"] = per_st
            return facts
        return facts

    def _collect_facts(self, df: pd.DataFrame, m: dict) -> dict:
        facts: dict[str, object] = {}
        if m["station"]:
            g = df[df[STATION_COL] == m["station"]]
            s = _series_stats(g[GWL_COL], g[TIME_COL])
            if not s:
                return facts
            facts["level"] = "station"
            facts["station"] = m["station"]
            facts["district"] = m["district"]
            facts.update(s)
            facts["forecast"] = _forecast_summary(m["station_slug"]) if m["station_slug"] else None
            if m["district"]:
                dg = df[df[DISTRICT_COL] == m["district"]][GWL_COL].dropna()
                facts["district_median"] = float(np.median(dg)) if dg.count() else None
                facts["district_n_stations"] = int(df[df[DISTRICT_COL] == m["district"]][STATION_COL].nunique())
            return facts

        # District level.
        dsel = df[df[DISTRICT_COL] == m["district"]]
        facts["level"] = "district"
        facts["district"] = m["district"]
        per_st: dict[str, dict] = {}
        for st, g in dsel.groupby(STATION_COL):
            s = _series_stats(g[GWL_COL], g[TIME_COL])
            if s:
                per_st[str(st)] = s
        if not per_st:
            return facts
        latest = [s["last"] for s in per_st.values()]
        rising = sum(1 for s in per_st.values() if (s.get("change_30d") or 0) > 0)
        falling = sum(1 for s in per_st.values() if (s.get("change_30d") or 0) < 0)
        facts["n_stations"] = len(per_st)
        facts["rising_30d"] = rising
        facts["falling_30d"] = falling
        facts["district_median_latest"] = float(np.median(latest))
        facts["district_min"] = float(min(latest))
        facts["district_max"] = float(max(latest))
        top = sorted(per_st.items(), key=lambda kv: kv[1].get("change_30d", 0) or 0, reverse=True)[:3]
        facts["top_rises"] = [(k, float(v.get("change_30d") or 0)) for k, v in top]
        decl = sorted(per_st.items(), key=lambda kv: kv[1].get("change_30d", 0) or 0)
        facts["top_declines"] = [(k, float(v.get("change_30d") or 0)) for k, v in decl[:3]]
        facts["declines_named"] = [(k, float(v.get("change_30d") or 0)) for k, v in decl[:10]]
        rec60 = sorted(
            per_st.items(), key=lambda kv: kv[1].get("change_60d", -1e18) or -1e18, reverse=True
        )
        facts["top_recovery_60d"] = [(k, float(v.get("change_60d") or 0)) for k, v in rec60[:5]]
        ch30 = [v.get("change_30d") for v in per_st.values() if v.get("change_30d") is not None]
        ch60 = [v.get("change_60d") for v in per_st.values() if v.get("change_60d") is not None]
        facts["district_median_change_30d"] = float(np.median(ch30)) if ch30 else None
        facts["district_median_change_60d"] = float(np.median(ch60)) if ch60 else None
        fresh_cut = _to_naive(pd.Timestamp(dsel[TIME_COL].max()) - pd.Timedelta(days=14))
        stale = [(k, v.get("last_date", "")) for k, v in per_st.items()
                 if _to_naive(pd.Timestamp(v.get("last_ts", ""))) < fresh_cut]
        facts["stale_stations"] = sorted(stale, key=lambda x: x[1])[:8]
        return facts

    def _build_prompt(self, question: str, facts: dict) -> str:
        lines = [
            "You are the AQUIS groundwater assistant. Answer the user's question using",
            "ONLY the given observed facts. No speculation. Be concise (max ~8 lines).",
            "Use metres (m) for levels/changes. State dates where known.",
            "Never refuse or say data is unavailable — use the station/district names in the",
            "facts below, and if a comparison is asked, answer it from the facts provided.",
            "", "FACTS:",
        ]
        if facts.get("level") == "station":
            lines.append(f"station: {facts['station']}  district: {facts.get('district')}")
            lines.append(f"latest level: {facts['last']:.2f} m on {facts.get('last_date')}")
            lines.append(f"range: {facts['min']:.2f} to {facts['max']:.2f} m (span {facts['span']:.2f} m, {facts['n_obs']} observations)")
            if facts.get("outliers"):
                lines.append(f"note: {facts['outliers']} outlier reading(s) excluded by quality filter")
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
                lines.append(
                    f"model forecast (next 2 months): day7={f['day7_pred']:.2f}, "
                    f"day30={f['day30_pred']:.2f}, day60={f['day60_pred']:.2f} m; "
                    f"change 30d={f['change_30d_pred']:+.3f}, 60d={f['change_60d_pred']:+.3f} m "
                    f"({f['direction']})"
                )
                if not f.get("plausible", True):
                    lines.append(
                        "note: model is UNSTABLE for this station (predictions runaway); "
                        "treat the forecast numbers above as unreliable, give only the observed facts"
                    )
            if facts.get("district_median") is not None:
                lines.append(
                    f"district median ({facts['district']}): {facts['district_median']:.2f} m "
                    f"across {facts['district_n_stations']} stations"
                )
        elif facts.get("level") == "district":
            lines.append(f"district: {facts['district']}")
            lines.append(
                f"stations: {facts['n_stations']} | rising (last 30d): {facts['rising_30d']} | "
                f"falling: {facts['falling_30d']}"
            )
            lines.append(
                f"latest median: {facts['district_median_latest']:.2f} m | min {facts['district_min']:.2f} | "
                f"max {facts['district_max']:.2f} m"
            )
            if facts.get("district_median_change_30d") is not None:
                lines.append(
                    f"district median change: 30d={facts['district_median_change_30d']:+.3f} m | "
                    f"60d={facts['district_median_change_60d']:+.3f} m"
                )
            if facts.get("top_rises"):
                lines.append("largest 30-day rises: " + ", ".join(
                    f"{k} {v:+.2f}m" for k, v in facts["top_rises"])
                )
            if facts.get("top_declines"):
                lines.append("largest 30-day declines: " + ", ".join(
                    f"{k} {v:+.2f}m" for k, v in facts["top_declines"])
                )
            if facts.get("declines_named"):
                lines.append(
                    "stations falling over last 30d (up to 10): " + ", ".join(
                        f"{k} {v:+.2f}m" for k, v in facts["declines_named"]
                    )
                )
            if facts.get("top_recovery_60d"):
                lines.append("fastest 60-day recovery (Jul->Sep): " + ", ".join(
                    f"{k} {v:+.2f}m" for k, v in facts["top_recovery_60d"])
                )
            if facts.get("stale_stations"):
                lines.append("stale/no recent data (last update): " + ", ".join(
                    f"{k} ({d})" for k, d in facts["stale_stations"]
                ))
        elif facts.get("level") == "rank_district":
            lines.append(f"district comparison window: last {facts['window_days']} days (as of {facts['as_of']})")
            lines.append("districts ranked by median level change (recovery = up):")
            for d, n, c, dt in facts["districts"]:
                lines.append(f"  {d}: {c:+.3f} m median across {n} stations (recent data to {dt})")
        elif facts.get("level") == "forecast_scan":
            lines.append(f"predicted 60-day declines of >= {facts['threshold']} m (model forecasts, as of {facts.get('as_of', '?')}):")
            if not facts.get("available"):
                lines.append("<no precomputed fleet forecast snapshot on this host>")
            elif not facts.get("matches"):
                lines.append(f"none of the scored stations exceed the threshold (or only unstable models did)")
            else:
                for mm in facts["matches"]:
                    lines.append(
                        f"  {mm['name']}: forecast {mm['change']:+.3f} m over 60d "
                        f"(level at day60 ≈ {mm['day_end']:.2f} m)"
                    )
            if facts.get("n_unreliable"):
                lines.append(
                    f"note: {facts['n_unreliable']} further stations had unstable "
                    f"models (predictions outside observed ranges) and were excluded"
                )
        lines += ["", f"QUESTION: {question}", "ANSWER:"]
        return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{'+' if x >= 0 else ''}{x:.3f}"


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts