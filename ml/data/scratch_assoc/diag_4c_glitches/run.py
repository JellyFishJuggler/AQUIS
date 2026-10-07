#!/usr/bin/env python
"""Diagnostic 4c - fleet-wide jump/glitch scan of the 6-hourly GWL series.

READ-ONLY on production: this script only READS
  data/aligned/table_6h.parquet and outputs/predictions_2026.parquet
and only WRITES into data/scratch_assoc/diag_4c_glitches/.

FIXED RULE (declared verbatim BEFORE looking at any results):
"Flag a 6-hourly GWL point at time t for station s if |gwl_t - gwl_{t-6h}| > max(2.5 * MAD_s, 2.0 m), where MAD_s is the station's median absolute deviation of the FULL series (median of |gwl - median(gwl)|). Only flag points where both gwl_t and gwl_{t-6h} are finite."

Event definition (declared in advance): consecutive flagged points within
2 steps (12 h) of each other form ONE event.

Event buckets (exactly one per event, evaluated in this order):
  (i)  SINGLE-POINT SPIKE that reverts: within 12 steps (72 h) of the flagged
       first time the series returns within 1.0 m of the pre-jump level.
  (ii) PERMANENT LEVEL SHIFT: the series does not return within 1.0 m of the
       pre-jump level within 30 days after the event.
  (iii) FAST RISE COINCIDING WITH RAIN: not (i), and within +/-1 day of the
       event start the station's rain (mm per 6-h, summed over +/-1 d) exceeds
       20 mm; if no rain data, note rain unavailable -> fast-rise-without-rain-
       evidence. (Events that are neither (i) nor (ii) fall here; the rain
       evidence flag records whether the >20 mm test passed.)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
ALIGNED = BASE / "data" / "aligned" / "table_6h.parquet"
PRED = BASE / "outputs" / "predictions_2026.parquet"
OUT = BASE / "data" / "scratch_assoc" / "diag_4c_glitches"

RULE = (
    "Flag a 6-hourly GWL point at time t for station s if "
    "|gwl_t - gwl_{t-6h}| > max(2.5 * MAD_s, 2.0 m), where MAD_s is the "
    "station's median absolute deviation of the FULL series (median of "
    "|gwl - median(gwl)|). Only flag points where both gwl_t and gwl_{t-6h} "
    "are finite."
)

STEP = pd.Timedelta(hours=6)
EVENT_GAP = pd.Timedelta(hours=12)      # group flagged pts within 2 steps (12 h)
REVERT_WIN = pd.Timedelta(hours=72)     # bucket (i): 12 steps
SHIFT_WIN = pd.Timedelta(days=30)       # bucket (ii): 30 days
RAIN_WIN = pd.Timedelta(days=1)         # bucket (iii): +/- 1 day
RAIN_MM = 20.0
REVERT_M = 1.0
PEAK5 = 5.0
FORECAST_HORIZON = pd.Timedelta(days=30)
EXCL_WIN = pd.Timedelta(days=1)

DASH_PREFIXES = ["Azamgarh", "Deoria", "Banda", "Meerut", "Varanasi", "ASHADHA"]

B_I = "i_single_point_spike_reverts"
B_II = "ii_permanent_level_shift"
B_III = "iii_fast_rise"


def load() -> pd.DataFrame:
    df = pd.read_parquet(ALIGNED, columns=["Station", "time", "gwl", "rain"])
    df = df.sort_values(["Station", "time"], kind="mergesort").reset_index(drop=True)
    return df


def flag_points(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the FIXED RULE (vectorised)."""
    g = df.groupby("Station", sort=False)
    df["med"] = g["gwl"].transform("median")
    df["_absdev"] = (df["gwl"] - df["med"]).abs()
    df["mad"] = df.groupby("Station", sort=False)["_absdev"].transform("median")
    df["prev_gwl"] = g["gwl"].shift(1)
    df["prev_time"] = g["time"].shift(1)
    df["jump"] = (df["gwl"] - df["prev_gwl"]).abs()

    threshold = np.maximum(2.5 * df["mad"], 2.0)
    df["threshold"] = threshold
    both_finite = df["gwl"].notna() & df["prev_gwl"].notna()
    is_prev_6h = df["prev_time"] == (df["time"] - STEP)
    df["flagged"] = both_finite & is_prev_6h & (df["jump"] > threshold)
    return df


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    fl = df.loc[df["flagged"]].copy()
    if fl.empty:
        return fl.iloc[0:0].assign(
            first_time=pd.Series(dtype="datetime64[us]"),
            last_time=pd.Series(dtype="datetime64[us]"),
        )
    gap = fl.groupby("Station", sort=False)["time"].diff()
    fl["new_event"] = gap.isna() | (gap > EVENT_GAP)
    fl["event_id"] = fl.groupby("Station", sort=False)["new_event"].cumsum()

    ev = (
        fl.groupby(["Station", "event_id"], sort=False)
        .agg(
            first_time=("time", "min"),
            last_time=("time", "max"),
            peak_jump_m=("jump", "max"),
            n_flagged_points=("jump", "size"),
            pre_jump_level_m=("prev_gwl", "first"),
            threshold_m=("threshold", "first"),
            station_median_m=("med", "first"),
            station_mad_m=("mad", "first"),
        )
        .reset_index()
    )
    # first flagged point of the event (for the pre-jump level / rain window)
    first_rows = fl[fl["new_event"]][["Station", "event_id", "time", "prev_gwl"]]
    first_rows = first_rows.rename(columns={"time": "first_time", "prev_gwl": "pre"})
    ev = ev.merge(first_rows, on=["Station", "event_id", "first_time"], how="left")
    ev["pre_jump_level_m"] = ev["pre"]
    ev = ev.drop(columns=["pre"])
    return ev


def classify(df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Buckets (i) -> (ii) -> (iii), one per event."""
    stations = {
        s: (
            idx["time"].to_numpy(dtype="datetime64[us]"),
            idx["gwl"].to_numpy(dtype=float),
            idx["rain"].to_numpy(dtype=float),
        )
        for s, idx in df.groupby("Station", sort=False)
    }

    bucket, detail, rain_sum, n_pts_30d, n_rets_72h = [], [], [], [], []
    for row in ev.itertuples(index=False):
        times, gwl, rain = stations[row.Station]
        pre = row.pre_jump_level_m
        t0 = np.datetime64(row.first_time.to_datetime64())

        lo = np.searchsorted(times, t0, side="left")
        hi72 = np.searchsorted(times, t0 + np.timedelta64(72, "h"), side="right")
        seg72 = gwl[lo:hi72]
        fin72 = seg72[np.isfinite(seg72)]
        n_ret = int(np.sum(np.abs(fin72 - pre) <= REVERT_M)) if fin72.size else 0

        hi30 = np.searchsorted(times, t0 + np.timedelta64(30 * 24, "h"), side="right")
        seg30 = gwl[lo:hi30]
        fin30 = seg30[np.isfinite(seg30)]
        n30 = int(fin30.size)
        returns_30d = bool(fin30.size and np.any(np.abs(fin30 - pre) <= REVERT_M))

        if n_ret > 0:
            b, d = B_I, f"reverts within 72 h ({n_ret} pt(s) within 1.0 m of pre-jump)"
        elif not returns_30d:
            b, d = B_II, "no return within 1.0 m of pre-jump in 30 days"
            if n30 == 0:
                d += " (no finite gwl points in window)"
        else:
            b = B_III
            rl = np.searchsorted(times, t0 - np.timedelta64(24, "h"), side="left")
            rh = np.searchsorted(times, t0 + np.timedelta64(24, "h"), side="right")
            rseg = rain[rl:rh]
            rfin = rseg[np.isfinite(rseg)]
            if rfin.size == 0:
                rsum = np.nan
                d = "fast-rise-without-rain-evidence (rain unavailable in +/-1 d window)"
            else:
                rsum = float(np.nansum(rfin))
                if rsum > RAIN_MM:
                    d = f"fast rise, rain {rsum:.1f} mm > 20 mm in +/-1 d"
                else:
                    d = f"fast-rise-without-rain-evidence (rain {rsum:.1f} mm <= 20 mm)"
            rain_sum.append(rsum)
            bucket.append(b)
            detail.append(d)
            n_pts_30d.append(n30)
            n_rets_72h.append(n_ret)
            continue

        rl = np.searchsorted(times, t0 - np.timedelta64(24, "h"), side="left")
        rh = np.searchsorted(times, t0 + np.timedelta64(24, "h"), side="right")
        rfin = rain[rl:rh]
        rfin = rfin[np.isfinite(rfin)]
        rain_sum.append(float(np.nansum(rfin)) if rfin.size else np.nan)
        bucket.append(b)
        detail.append(d)
        n_pts_30d.append(n30)
        n_rets_72h.append(n_ret)

    ev = ev.copy()
    ev["bucket"] = bucket
    ev["detail"] = detail
    ev["rain_sum_pm1d_mm"] = rain_sum
    ev["n_finite_pts_30d"] = n_pts_30d
    ev["n_revert_pts_72h"] = n_rets_72h
    return ev


def mad_mask_analysis(df: pd.DataFrame, ev: pd.DataFrame) -> dict:
    fl = df.loc[df["flagged"]].copy()
    fl["masked_30mad"] = (fl["gwl"] - fl["med"]).abs() > 30.0 * fl["mad"]
    # actual pipeline: 06_features.drop_gwl_spikes uses 30 * (1.4826 * MAD), floor 1e-6
    fl["masked_pipeline"] = (fl["gwl"] - fl["med"]).abs() > 30.0 * np.maximum(
        1.4826 * fl["mad"], 1e-6
    )
    key = ["Station", "event_id"] if "event_id" in fl.columns else None
    agg: dict = {
        "flagged_points_total": int(len(fl)),
        "flagged_points_masked_by_30mad": int(fl["masked_30mad"].sum()),
        "flagged_points_masked_by_pipeline_30x1p4826mad": int(fl["masked_pipeline"].sum()),
    }
    if key:
        fl2 = fl.merge(ev[["Station", "event_id"]], on=["Station", "event_id"], how="left")
        per_ev = fl2.groupby(["Station", "event_id"], sort=False).agg(
            any_30=("masked_30mad", "any"),
            all_30=("masked_30mad", "all"),
            any_pl=("masked_pipeline", "any"),
            all_pl=("masked_pipeline", "all"),
        )
        agg["events_any_flagged_point_masked_30mad"] = int(per_ev["any_30"].sum())
        agg["events_all_flagged_points_masked_30mad"] = int(per_ev["all_30"].sum())
        agg["events_any_flagged_point_masked_pipeline"] = int(per_ev["any_pl"].sum())
        agg["events_all_flagged_points_masked_pipeline"] = int(per_ev["all_pl"].sum())
    agg["note"] = (
        "task-spec mask = |gwl-median| > 30*MAD (MAD unscaled); actual "
        "06_features.drop_gwl_spikes uses 30*(1.4826*MAD) with floor 1e-6, "
        "so the pipeline-masked counts are a subset of the 30*MAD counts"
    )
    return agg


def near_event_mask(stations: pd.Series, times: pd.Series, windows: dict) -> np.ndarray:
    out = np.zeros(len(times), dtype=bool)
    tvals = times.to_numpy(dtype="datetime64[us]")
    codes, uniq = pd.factorize(stations, sort=False)
    for code in range(len(uniq)):
        w = windows.get(uniq[code])
        if w is None:
            continue
        starts, ends = w
        idx = np.flatnonzero(codes == code)
        tt = tvals[idx]
        j = np.searchsorted(starts, tt, side="right") - 1
        ok = j >= 0
        hit = np.zeros(len(idx), dtype=bool)
        hit[ok] = tt[ok] <= ends[j[ok]]
        out[idx[hit]] = True
    return out


def rmse_case(sub: pd.DataFrame, rng_seed: int, B: int = 2000) -> dict:
    ex_x = (sub["xgb"].to_numpy(float) - sub["target"].to_numpy(float)) ** 2
    ex_p = (sub["persist"].to_numpy(float) - sub["target"].to_numpy(float)) ** 2
    n = len(sub)
    rmse_x = float(np.sqrt(ex_x.mean()))
    rmse_p = float(np.sqrt(ex_p.mean()))
    ratio = rmse_x / rmse_p

    per = pd.DataFrame(
        {"Station": sub["Station"].to_numpy(), "sx": ex_x, "sp": ex_p}
    ).groupby("Station", sort=False)[["sx", "sp"]].sum()
    S = len(per)
    sx = per["sx"].to_numpy(float)
    sp = per["sp"].to_numpy(float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, S, size=(B, S))
    bx = sx[idx].sum(axis=1)
    bp = sp[idx].sum(axis=1)
    br = np.sqrt(bx / np.maximum(bp, 1e-12))
    ci = np.percentile(br, [2.5, 97.5])
    return {
        "n_rows": int(n),
        "n_stations": int(S),
        "rmse_xgb": rmse_x,
        "rmse_persist": rmse_p,
        "ratio_xgb_over_persist": ratio,
        "ratio_ci95": [float(ci[0]), float(ci[1])],
        "bootstrap": {"B": B, "seed": rng_seed, "resample": "stations_with_replacement"},
    }


def forecast_impact(ev: pd.DataFrame) -> dict:
    p = pd.read_parquet(PRED)
    s = p.loc[p["stride"]].copy()
    s = s[s["xgb"].notna() & s["target"].notna() & s["persist"].notna()].copy()
    s["anchor"] = s["time"]
    s["target_time"] = s["time"] + FORECAST_HORIZON

    windows: dict = {}
    for st, g in ev.groupby("Station", sort=False):
        lo = (g["first_time"].to_numpy(dtype="datetime64[us]")
              - np.timedelta64(24, "h"))
        hi = (g["last_time"].to_numpy(dtype="datetime64[us]")
              + np.timedelta64(24, "h"))
        order = np.argsort(lo)
        windows[st] = _merged_windows_events(lo[order], hi[order])

    near_a = near_event_mask(s["Station"], s["anchor"], windows)
    near_t = near_event_mask(s["Station"], s["target_time"], windows)
    excl = near_a | near_t

    out = {
        "anchor_definition": "stride row `time` (forecast issue time); target_time = anchor + 30 d",
        "exclusion": "row removed if anchor OR anchor+30d is within +/-1 d of any flagged event of the same station",
        "case_all_stride_rows": rmse_case(s, 42),
        "case_excluding_flagged_event_rows": rmse_case(s.loc[~excl].copy(), 42),
        "n_rows_excluded": int(excl.sum()),
        "n_stations_affected_by_exclusion": int(s.loc[excl, "Station"].nunique()),
    }
    return out


def _merged_windows_events(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return starts, ends
    s, e = [starts[0]], [ends[0]]
    for a, b in zip(starts[1:], ends[1:]):
        if a <= e[-1]:
            e[-1] = max(e[-1], b)
        else:
            s.append(a)
            e.append(b)
    return np.asarray(s), np.asarray(e)


def plot_dashboard(df: pd.DataFrame, ev: pd.DataFrame) -> tuple[Path, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(df["Station"].unique())
    matched: dict[str, list[str]] = {}
    for pfx in DASH_PREFIXES:
        matched[pfx] = [n for n in names if n.lower().startswith(pfx.lower())]

    sel = [n for pfx in DASH_PREFIXES for n in matched[pfx]]
    ncols = 5
    nrows = int(np.ceil(len(sel) / ncols)) if sel else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, st in enumerate(sel):
        ax = axes.flat[i]
        ax.set_visible(True)
        sub = df[df["Station"] == st]
        ax.plot(sub["time"], sub["gwl"], lw=0.5, color="0.35", zorder=1)
        fl = sub[sub["flagged"]]
        if len(fl):
            ax.plot(fl["time"], fl["gwl"], "o", ms=3.5, color="red", zorder=3)
        evs = ev[ev["Station"] == st]
        for r in evs.itertuples(index=False):
            ax.axvspan(r.first_time - STEP, r.last_time + STEP, color="red", alpha=0.18, zorder=2)
        n_ev = len(evs)
        buckets = evs["bucket"].value_counts().to_dict() if n_ev else {}
        tag = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in sorted(buckets.items()))
        ax.set_title(f"{st}\n{n_ev} event(s) {tag}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    for j in range(len(sel), axes.size):
        axes.flat[j].set_visible(False)
    fig.suptitle(
        "Diagnostic 4c - dashboard stations: 6-hourly GWL, flagged points (red), "
        "events (shaded)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png = OUT / "dashboard_stations.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    dash = {pfx: {"n_stations": len(v), "stations": v} for pfx, v in matched.items()}
    dash["varanasi_note"] = (
        "No station whose name starts with 'Varanasi' exists in "
        "data/aligned/table_6h.parquet - explicitly skipped."
    )
    dash["n_stations_plotted"] = len(sel)
    dash["plot"] = str(out_png.relative_to(BASE))
    return out_png, dash


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"RULE: {RULE}\n", flush=True)

    df = load()
    n_rows, n_stations = len(df), df["Station"].nunique()
    df = flag_points(df)
    n_flagged = int(df["flagged"].sum())
    print(f"loaded {n_rows:,} rows / {n_stations} stations; flagged {n_flagged:,} points", flush=True)

    ev = build_events(df)
    # link event_id back onto flagged rows for the mask analysis
    fl = df.loc[df["flagged"], ["Station", "time"]].copy()
    gap = fl.groupby("Station", sort=False)["time"].diff()
    fl["event_id"] = (gap.isna() | (gap > EVENT_GAP)).groupby(fl["Station"], sort=False).cumsum()
    df.loc[df["flagged"], "event_id"] = fl["event_id"]
    ev = classify(df, ev)

    # ---- 3a fleet counts ----
    counts = ev["bucket"].value_counts().to_dict()
    bucket_i = int(counts.get(B_I, 0))
    bucket_ii = int(counts.get(B_II, 0))
    bucket_iii = int(counts.get(B_III, 0))
    rain_ev = ev.loc[ev["bucket"] == B_III, "rain_sum_pm1d_mm"]
    iii_rain_gt20 = int((rain_ev > RAIN_MM).sum())
    iii_rain_le20 = int((rain_ev <= RAIN_MM).sum())
    iii_rain_na = int(rain_ev.isna().sum())

    fleet = {
        "n_stations_total": n_stations,
        "n_rows_total": n_rows,
        "n_flagged_points": n_flagged,
        "n_stations_with_ge1_event": int(ev["Station"].nunique()) if len(ev) else 0,
        "n_events": int(len(ev)),
        "events_bucket_i_spike_reverts": bucket_i,
        "events_bucket_ii_permanent_shift": bucket_ii,
        "events_bucket_iii_fast_rise": bucket_iii,
        "events_bucket_iii_with_rain_gt_20mm": iii_rain_gt20,
        "events_bucket_iii_without_rain_evidence": iii_rain_le20 + iii_rain_na,
        "events_bucket_iii_rain_unavailable": iii_rain_na,
        "events_peak_jump_ge_5m": int((ev["peak_jump_m"] >= PEAK5).sum()) if len(ev) else 0,
    }

    # ---- 3b MAD-mask modification analysis ----
    mask = mad_mask_analysis(df, ev)

    # ---- 3c forecast impact ----
    fimp = forecast_impact(ev)

    # ---- 3d dashboard plots ----
    png, dash = plot_dashboard(df, ev)

    # dashboard per-prefix event counts
    names = df["Station"].unique()
    for pfx in DASH_PREFIXES:
        sel = [n for n in names if str(n).lower().startswith(pfx.lower())]
        n_ev = int(ev.loc[ev["Station"].isin(sel)].shape[0]) if len(ev) else 0
        buckets = (
            ev.loc[ev["Station"].isin(sel), "bucket"].value_counts().to_dict()
            if len(ev)
            else {}
        )
        dash[pfx]["n_events"] = n_ev
        dash[pfx]["events_by_bucket"] = {k: int(v) for k, v in buckets.items()}
        dash[pfx]["n_stations_with_events"] = (
            int(ev.loc[ev["Station"].isin(sel), "Station"].nunique()) if len(ev) else 0
        )

    summary = {
        "diagnostic": "4c fleet-wide jump/glitch scan (6-hourly GWL)",
        "fixed_rule": RULE,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "aligned": str(ALIGNED.relative_to(BASE)),
            "predictions": str(PRED.relative_to(BASE)),
        },
        "event_definition": "consecutive flagged points within 2 steps (12 h) form one event",
        "fleet": fleet,
        "mad_mask_analysis": mask,
        "forecast_impact": fimp,
        "dashboard_stations": dash,
    }

    ev_out = ev.sort_values(["Station", "first_time"]).rename(
        columns={"Station": "station"}
    )
    (OUT / "events.csv").write_text(
        ev_out.to_csv(index=False)
    )
    (OUT / "fleet_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DIAGNOSTIC 4c SUMMARY ===")
    print(f"flagged points            : {fleet['n_flagged_points']:,}")
    print(f"stations with >=1 event   : {fleet['n_stations_with_ge1_event']} / {n_stations}")
    print(f"total events              : {fleet['n_events']}")
    print(f"  (i) spike reverts       : {bucket_i}")
    print(f"  (ii) permanent shift    : {bucket_ii}")
    print(f"  (iii) fast rise         : {bucket_iii}  (rain>20mm: {iii_rain_gt20}, "
          f"no rain evidence: {iii_rain_le20 + iii_rain_na}, rain unavailable: {iii_rain_na})")
    print(f"events peak |jump| >= 5 m : {fleet['events_peak_jump_ge_5m']}")
    print(f"MAD-mask: flagged pts masked (30*MAD)      : "
          f"{mask['flagged_points_masked_by_30mad']:,} / {mask['flagged_points_total']:,}")
    print(f"MAD-mask: flagged pts masked (pipeline)    : "
          f"{mask['flagged_points_masked_by_pipeline_30x1p4826mad']:,}")
    print(f"MAD-mask: events with any pt masked (30*MAD): "
          f"{mask.get('events_any_flagged_point_masked_30mad', 0)} / {fleet['n_events']}")
    ca, ce = fimp["case_all_stride_rows"], fimp["case_excluding_flagged_event_rows"]
    print(f"forecast ALL     : n={ca['n_rows']:,} rmse_xgb={ca['rmse_xgb']:.4f} "
          f"rmse_persist={ca['rmse_persist']:.4f} ratio={ca['ratio_xgb_over_persist']:.4f} "
          f"CI95=[{ca['ratio_ci95'][0]:.4f},{ca['ratio_ci95'][1]:.4f}]")
    print(f"forecast EXCL    : n={ce['n_rows']:,} (removed {fimp['n_rows_excluded']}) "
          f"rmse_xgb={ce['rmse_xgb']:.4f} rmse_persist={ce['rmse_persist']:.4f} "
          f"ratio={ce['ratio_xgb_over_persist']:.4f} "
          f"CI95=[{ce['ratio_ci95'][0]:.4f},{ce['ratio_ci95'][1]:.4f}]")
    print(f"Varanasi*        : {dash['varanasi_note']}")
    print("dashboard six-prefix event counts:")
    for pfx in DASH_PREFIXES:
        d = dash[pfx]
        print(f"  {pfx:<10} stations={d['n_stations']} with_events={d['n_stations_with_events']} "
              f"events={d['n_events']} {d['events_by_bucket']}")
    print(f"\nwrote {OUT/'run.py'}, events.csv, fleet_summary.json, summary.json, {png.name}")


if __name__ == "__main__":
    main()
