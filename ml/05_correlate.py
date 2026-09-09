"""05_correlate — Spearman/Pearson, lagged CCF, rain windows, de-seasonalized.

Target: daily groundwater level (gwl). Drivers pulled from the aligned table.
Per-station correlations are aggregated with the median (robust to datum mixes);
pooled N is reported. Outputs:
  ml/outputs/correlation_report.csv        — ranked feature strength
  ml/outputs/correlation_by_district.csv
  ml/outputs/lag_curves.csv                — median lagged correlation vs rain
  ml/outputs/scatter_top.html, lag_heat.html — plotly charts
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config  # noqa: E402

TARGET = "gwl"
RAIN_WINDOWS = (1, 7, 30)

# (column, is_within_station_deseason) — mobility drivers we also deseasonalise.
LEVEL_COLS = ["temp", "humidity", "solar", "wind_speed", "pressure", "river_level",
              "canal_level", "canal_discharge",
              "res_okhla", "res_okhla_agra_canal", "res_matatila",
              "res_ganga_rmc", "res_ganga_1"]


def load_table() -> pd.DataFrame:
    tbl = pd.read_parquet(config.ALIGNED / "table.parquet")
    tbl["date"] = pd.to_datetime(tbl["date"], errors="coerce")
    return tbl.sort_values(["Station", "date"]).reset_index(drop=True)


def add_windows(tbl: pd.DataFrame) -> pd.DataFrame:
    tbl = tbl.copy()
    for w in RAIN_WINDOWS:
        if "rain" in tbl.columns:
            tbl[f"rain_{w}d"] = (
                tbl.groupby("Station")["rain"].transform(lambda s: s.rolling(w).sum()) if w > 1
                else tbl["rain"])
    # only where enough history exists
    tbl["doy"] = tbl["date"].dt.dayofyear
    return tbl


def deseason(tbl: pd.DataFrame, col: str) -> pd.Series:
    mu = tbl.groupby(["Station", "doy"])[col].transform("mean")
    return tbl[col] - mu


def per_station_corr(tbl: pd.DataFrame, a: str, b: str,
                     how: str) -> pd.DataFrame:
    """Per-station correlation (pearson/spearman); NaN when too few shared days."""
    def _f(g: pd.DataFrame) -> float:
        s = g[["_a", "_b"]].dropna()
        if len(s) < 20:
            return np.nan
        try:
            if how == "spearman":
                return s["_a"].corr(s["_b"], method="spearman")
            return s["_a"].corr(s["_b"], method="pearson")
        except Exception:  # noqa: BLE001
            return np.nan

    sub = tbl[["Station", a, b]].rename(columns={a: "_a", b: "_b"})
    out = sub.groupby("Station").apply(_f, include_groups=False)
    out = out.rename("corr").reset_index()
    out[f"{what_n(a,b)}_n"] = sub.groupby("Station").apply(
        lambda g: int(g[["_a", "_b"]].dropna().shape[0]), include_groups=False
    ).values
    return out


def what_n(a: str, b: str) -> str:
    return f"n_{a}_{b}"


def lagged_med(tbl: pd.DataFrame, lead: int, min_rows: int = 180) -> float:
    def _f(g: pd.DataFrame) -> float:
        s = pd.DataFrame({"x": g["rain"].shift(lead), "y": g[TARGET]}).dropna()
        if len(s) < min_rows:
            return np.nan
        return s["y"].corr(s["x"], method="pearson")
    c = tbl.groupby("Station").apply(_f, include_groups=False)
    return float(c.median())


def main() -> None:
    tbl = add_windows(load_table())
    print(f"table: {len(tbl):,} rows / {tbl['Station'].nunique():,} stations", flush=True)

    report: list[dict] = []
    print(f"\ntarget: {TARGET}")

    # raw Spearman + Pearson (per-station, median)
    skip = {"Station", "date", TARGET, "doy", "District", "Tehsil"}
    driver_cols = [c for c in tbl.columns
                   if c not in skip and not c.endswith("_n")
                   and not c.endswith("_dist_km")]
    print("drivers:", ", ".join(driver_cols), flush=True)
    for col in driver_cols:
        for how in ("spearman", "pearson"):
            r = per_station_corr(tbl, TARGET, col, how)
            med = r["corr"].median()
            report.append({
                "driver": col, "metric": how, "mode": "raw",
                "corr": None if pd.isna(med) else round(float(med), 4),
                "stations_ok": int(r["corr"].notna().sum()),
            })
        # first-difference Spearman — reacts to day-to-day change, datum-free
        if col in LEVEL_COLS or col.startswith("rain_"):
            sub = tbl.assign(_t=tbl.groupby("Station")[TARGET].diff(),
                             _d=tbl.groupby("Station")[col].diff())
            r = sub[["Station", "_t", "_d"]].groupby("Station").apply(
                lambda g: (pd.Series(g["_t"]).corr(pd.Series(g["_d"]), method="spearman")
                           if g[["_t", "_d"]].dropna().shape[0] >= 20 else np.nan),
                include_groups=False)
            med = r.median()
            report.append({
                "driver": col, "metric": "spearman", "mode": "diff",
                "corr": None if pd.isna(med) else round(float(med), 4),
                "stations_ok": int(r.notna().sum()),
            })
        # deseasonalised Spearman
        if col in LEVEL_COLS:
            sub = tbl.assign(_t=deseason(tbl, TARGET), _d=deseason(tbl, col))
            r = sub[["Station", "_t", "_d"]].groupby("Station").apply(
                lambda g: g["_t"].corr(g["_d"], method="spearman")
                if len(g.dropna()) >= 20 else np.nan, include_groups=False)
            med = r.median()
            report.append({
                "driver": col, "metric": "spearman", "mode": "deseason",
                "corr": None if pd.isna(med) else round(float(med), 4),
                "stations_ok": int(r.notna().sum()),
            })

    out = pd.DataFrame(report)
    out = out.sort_values(["driver", "metric", "mode"]).reset_index(drop=True)
    out.to_csv(config.OUT / "correlation_report.csv", index=False)
    print(out.to_string(index=False), flush=True)

    # per-district (pooled-pair spearman) for the top drivers
    dist_pairs = [("rain_7d", "rain_30d"), ("temp", "river_level"), ("humidity", "solar")]
    dist_rows = []
    for a, b in dist_pairs:
        if a not in tbl or b not in tbl:
            continue
        for dist, g in tbl.groupby("District"):
            s = g[[a, b, TARGET]].dropna()
            if len(s) < 30:
                continue
            for col in (a, b):
                dist_rows.append({
                    "District": dist, "driver": col,
                    "spearman": round(float(s[TARGET].corr(s[col], method="spearman")), 4),
                    "pearson": round(float(s[TARGET].corr(s[col], method="pearson")), 4),
                    "n": int(len(s)),
                })
    pd.DataFrame(dist_rows).to_csv(config.OUT / "correlation_by_district.csv", index=False)
    print(f"\nper-district -> {config.OUT / 'correlation_by_district.csv'}", flush=True)

    # lagged CCF vs rainfall
    lags = list(range(0, 31))
    lag_curves = pd.DataFrame({"lag": lags,
                               "pearson_med": [lagged_med(tbl, k) for k in lags]})
    lag_curves.to_csv(config.OUT / "lag_curves.csv", index=False)
    best = lag_curves.iloc[lag_curves["pearson_med"].abs().idxmax()]
    print(f"\nbest rain lag: {int(best['lag'])} days | med corr {best['pearson_med']:.4f}", flush=True)

    # -------- plots (plotly html) --------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = go.Figure(go.Scatter(x=lag_curves["lag"], y=lag_curves["pearson_med"],
                                   mode="lines+markers"))
        fig.add_vline(x=int(best["lag"]), line_dash="dot",
                      annotation_text=f"best lag {int(best['lag'])}d")
        fig.update_layout(title="GWL vs rainfall — median Pearson by lag (days)")
        fig.write_html(config.OUT / "lag_curve.html")

        # top stations by shared rain days
        top = (tbl.dropna(subset=["rain_30d"]).groupby("Station").size()
               .sort_values(ascending=False).head(3).index)
        if len(top):
            figs = make_subplots(rows=len(list(top)), cols=1, shared_xaxes=True,
                                 subplot_titles=list(top))
            for i, st in enumerate(top, 1):
                g = tbl[tbl["Station"] == st].dropna(subset=["rain_30d", TARGET])
                if g.empty:
                    continue
                figs.add_trace(go.Scatter(x=g["date"], y=g[TARGET], name="GWL (m)"),
                               row=i, col=1)
                figs.add_trace(go.Scatter(x=g["date"], y=g["rain_30d"],
                                          name="rain 30d (mm)", yaxis=f"y{i+1}"),
                               row=i, col=1)
            figs.update_layout(height=350 * max(len(list(top)), 1))
            figs.write_html(config.OUT / "gwl_vs_rain_top.html")
    except ImportError:
        print("plotly unavailable — skipped plots", flush=True)

    print("\nall reports saved under outputs/", flush=True)


if __name__ == "__main__":
    main()