#!/usr/bin/env python
"""Render the app's forecast charts with REAL data to standalone HTML,
screenshot them with headless Chrome, and dump every trace's legend name.

Chart-config-only probe: reproduces exactly the figures the dashboard builds.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app_pages"))
OUT = ROOT / "data" / "scratch_assoc" / "ui_render"
OUT.mkdir(parents=True, exist_ok=True)

import app_pages._charts as C
import app_pages._data as D

OPEN = "Varansi Harahua Gurwat"

# ---- 1. evaluation: paired_actual_forecast (stride backtest) ------------
p = D.predictions_2026()
p30 = p[p["horizon"] == 30].copy()
p30["target_time"] = pd.to_datetime(p30["time"]) + pd.to_timedelta(30, unit="D")
p30 = p30[p30["target_time"].dt.year.eq(2026)]
p30["actual"] = pd.to_numeric(p30["target"], errors="coerce")
p30["forecast"] = pd.to_numeric(p30["xgb"], errors="coerce")
shown = p30.dropna(subset=["actual", "forecast"]).copy()
shown = shown[shown["Station"].astype(str) == OPEN]
shown["error"] = shown["forecast"] - shown["actual"]
cf = shown.sort_values("target_time").set_index("target_time")
cf["error"] = cf["forecast"] - cf["actual"]
fig1 = C.paired_actual_forecast(
    cf, height=500,
    forecast_name="30-day forecast — honest backtest (stride grid)",
    daily_median=True)
fig1.update_layout(width=1400, height=560, margin=dict(t=40, b=30))
fig1.write_html(str(OUT / "01_paired_backtest.html"), include_plotlyjs="cdn")

# ---- 2. evaluation: published trajectory vs realised (live archive) -----
av = D.archive_validation(OPEN, horizon_h=720)
led = D.verification_ledger()
fig2 = C.paired_actual_forecast(
    pd.DataFrame(), height=500, forecast_name="placeholder", daily_median=True)
if led is not None and len(led):
    led = led[led["horizon_h"] == 720].copy()
    led["target_time"] = pd.to_datetime(led["target_time"])
    led = led[led["target_time"].dt.year.eq(2026)]
    led["actual"] = pd.to_numeric(led["actual"], errors="coerce")
    led["q50"] = pd.to_numeric(led["q50"], errors="coerce")
    led = led[led["station"].astype(str) == OPEN].dropna(subset=["actual", "q50"])
    if len(led):
        series = (led.sort_values("target_time").set_index("target_time")[["actual", "q50"]]
                  .rename(columns={"q50": "forecast median"}))
        series["error"] = series["forecast median"] - series["actual"]
        fig2 = C.paired_actual_forecast(
            series, height=500, forecast_col="forecast median",
            forecast_name="30-day forecast — live issued (archive)",
            title="published trajectory vs realised", daily_median=True)
fig2.update_layout(width=1400, height=560, margin=dict(t=70, b=30))
fig2.write_html(str(OUT / "02_paired_live.html"), include_plotlyjs="cdn")

# ---- 3. station card: hydrograph_with_forecast --------------------------
hist = D.station_history(OPEN, tail_days=400)
hdf = pd.DataFrame(index=pd.to_datetime(hist["time"])) if (hist is not None and len(hist)) else pd.DataFrame()
if hist is not None and len(hist):
    hdf["level (m)"] = pd.to_numeric(hist["gwl"], errors="coerce")
    hdf["carried forward"] = pd.to_numeric(hist["gwl"], errors="coerce")
if len(av):
    idx = pd.to_datetime(av["target_time"]).dropna()
    q05 = pd.to_numeric(av["q05"], errors="coerce").to_numpy()
    q50 = pd.to_numeric(av["q50"], errors="coerce").to_numpy()
    q95 = pd.to_numeric(av["q95"], errors="coerce").to_numpy()
    f = pd.DataFrame({"forecast q05": q05, "forecast median": q50,
                      "forecast q95": q95}, index=idx)
    hdf = pd.concat([hdf, f], axis=0).sort_index()
fig3 = C.hydrograph_with_forecast(hdf, height=520, title=OPEN)
fig3.update_layout(width=1400, height=580, margin=dict(t=70, b=30))
fig3.write_html(str(OUT / "03_station_card.html"), include_plotlyjs="cdn")

# ---- 4. forecast explorer: validation(actual q50 band) ------------------
if len(av):
    a = pd.Series(pd.to_numeric(av["actual"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(av["target_time"]).dropna())
    q50 = pd.Series(pd.to_numeric(av["q50"], errors="coerce").to_numpy(),
                    index=a.index)
    q05 = pd.Series(pd.to_numeric(av["q05"], errors="coerce").to_numpy(), index=a.index)
    q95 = pd.Series(pd.to_numeric(av["q95"], errors="coerce").to_numpy(), index=a.index)
    fig4 = C.validation(a, q50, q05=q05, q95=q95, height=500, daily_median=True)
else:
    fig4 = C.validation(pd.Series(dtype=float), pd.Series(dtype=float))
fig4.update_layout(width=1400, height=560, margin=dict(t=40, b=30))
fig4.write_html(str(OUT / "04_archive_validation.html"), include_plotlyjs="cdn")

report = {"station": OPEN, "figures": {}}
for name, fig in [("01_paired_backtest", fig1), ("02_paired_live", fig2),
                  ("03_station_card", fig3), ("04_archive_validation", fig4)]:
    report["figures"][name] = {
        "traces": [
            {"name": getattr(t, "name", None) or "",
             "showlegend": bool(getattr(t, "showlegend", False) is not False)}
            for t in fig.data
        ],
        "legend_title": None,
    }
    lt = fig.layout.legend.title
    try:
        report["figures"][name]["legend_title"] = lt.text if lt is not None and getattr(lt, "text", None) is not None else None
    except Exception:
        pass
json.dump(report, open(OUT / "trace_names.json", "w"), indent=2)

bad = [f"{name}:{getattr(t, 'name', '')}" for name, fig in
       [("01", fig1), ("02", fig2), ("03", fig3), ("04", fig4)]
       for t in fig.data
       if not (getattr(t, "name", "") or "").strip()
       and getattr(t, "showlegend", False) is not False]
print("legend-visible traces with blank/None name:", bad if bad else "NONE")
for name in report["figures"]:
    print(name, "->", [t["name"] for t in report["figures"][name]["traces"]])
print("wrote 4 html + trace_names.json")