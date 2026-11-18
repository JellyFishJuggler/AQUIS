"""benchmark_2026 — rolling 30-day benchmark for 2026: persistence vs XGB vs DL.

Read-only with respect to ``ml/models`` and ``ml/data``. The XGBoost booster is
built from an *in-memory* copy of ``models/traj_xgb_q50.json`` whose
``base_score`` is rewritten to ``0E0`` (the tracked artifact was trained with
xgboost 3.4.1 and carries an out-of-range base_score that xgboost 2.x silently
clamps, adding a constant offset to every prediction); the copy is written to a
temporary directory outside the repo. The tracked JSON is never modified.

Protocol
--------
* Universe: stations the fleet scan scores (outputs/fleet_forecast.csv).
* Actuals: ``data/aligned/table_6h.parquet`` cleaned with
  ``pipelines.model.drop_gwl_spikes`` (the same cleaning the deployed engines use).
* Origins: 2026-01-10 then every 30 days while ``origin + 30d <= last obs``.
* Per origin each station is scored from a *causal* slice (rows with
  ``time <= origin`` only); the anchor is that slice's last observed GWL and the
  prediction target is ``anchor + 30d`` (the model's own horizon), not
  ``origin + 30d``.
* Excluded from the clean table (counted and reported, never silently dropped):
    - ``flag_regime_shift`` stations (persistent datum jumps),
    - anchor lag > 7 days,
    - "Luktra" anchors: fewer than 14 observations in the 7 days before the
      anchor (an isolated reading after a long gap is not a usable anchor),
    - no observed actual within +/-1 day of the target,
    - stations whose feature frame is empty (e.g. the flag drop inside
      ``build_full``).
* Metrics are level errors ``prediction - actual``: n, median bias, MAE, RMSE.
  Persistence predicts the anchor level.
* XGB / DL improvement over persistence is reported per origin and pooled, for
  MAE and RMSE.

Usage (from ml/):

    python -m experiments.benchmark_2026                # full run, writes outputs
    python -m experiments.benchmark_2026 --limit 12     # smoke test on 12 stations
    python -m experiments.benchmark_2026 --verify-xgb   # val-split base_score check only

    python benchmark_2026.py ...                        # backward-compatible root shim
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib  # noqa: E402

F = importlib.import_module("pipelines.model")

from core.thresholds import DECLINE_M, HIGH_DECLINE_M  # noqa: E402

ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
FLEET_CSV = ROOT / "outputs" / "fleet_forecast.csv"
TRAJ_CFG = ROOT / "models" / "traj_config.json"
TRAJ_DL_CFG = ROOT / "models" / "traj_dl_config.json"
TRAJ_XGB_Q50 = ROOT / "models" / "traj_xgb_q50.json"
VAL_PARQUET = ROOT / "data" / "features_traj" / "val.parquet"

OUTDIR = ROOT / "outputs" / "benchmark_2026"

ORIGIN_START = pd.Timestamp("2026-01-10")
ORIGIN_STEP = pd.Timedelta(days=30)
HORIZON = pd.Timedelta(days=30)
ANCHOR_MAX_LAG = pd.Timedelta(days=7)
ANCHOR_MIN_OBS_7D = 14          # "Luktra" guard: >= 50% of the 28 six-hourly slots
ACTUAL_TOL = pd.Timedelta(days=1)
MAX_ABS_ACTUAL_CHANGE = 12.0    # datum-jump guard on the realised 30d change
MODELS = ("persist", "xgb", "dl")
MODEL_LABEL = {"persist": "persistence", "xgb": "XGB (base_score=0)",
               "dl": "DL (GRU)"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_origins(last_obs: pd.Timestamp) -> list[pd.Timestamp]:
    out, cur = [], ORIGIN_START
    while cur + HORIZON <= last_obs:
        out.append(cur)
        cur = cur + ORIGIN_STEP
    return out


def load_xgb_booster() -> "object":
    """Load q50 from a base_score=0 in-memory copy (repo artifact untouched)."""
    import xgboost as xgb

    src = TRAJ_XGB_Q50.read_text(encoding="utf-8")
    fixed = re.sub(r'"base_score":"[^"]*"', '"base_score":"0E0"', src, count=1)
    if fixed == src:
        raise RuntimeError("base_score field not found in traj_xgb_q50.json")
    tmpdir = Path(tempfile.mkdtemp(prefix="aquis_bench_"))
    path = tmpdir / "traj_xgb_q50_base0.json"
    path.write_text(fixed, encoding="utf-8")
    booster = xgb.Booster(model_file=str(path))
    got = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    print(f"[xgb] in-memory copy base_score={got} (repo file untouched)", flush=True)
    return booster


def verify_xgb_base_score() -> dict:
    """Reproduce traj_config.json val metrics at h=120 with the base_score=0 copy."""
    import xgboost as xgb

    cfg = json.loads(TRAJ_CFG.read_text())
    fnames = list(cfg["feature_cols"])
    ref = next((r for r in cfg.get("val_per_h_metrics", []) if r["h"] == 120), {})
    d = pd.read_parquet(VAL_PARQUET,
                        columns=list(dict.fromkeys(fnames + ["target_delta", "h"])))
    g = d[d["h"] == 120]
    X = g[fnames].to_numpy(np.float32)
    y = g["target_delta"].to_numpy(np.float32)
    booster = load_xgb_booster()
    p = booster.predict(xgb.DMatrix(X, feature_names=fnames))
    got = {"n": int(len(g)), "bias": float(np.mean(p - y)),
           "mae": float(np.mean(np.abs(p - y))),
           "rmse": float(np.sqrt(np.mean((p - y) ** 2)))}
    print(f"[verify] val h=120  n={got['n']}  bias={got['bias']:+.4f}  "
          f"mae={got['mae']:.4f}  rmse={got['rmse']:.4f}", flush=True)
    print(f"[verify] traj_config  n={ref.get('n')}  bias={ref.get('bias', float('nan')):+.4f}  "
          f"mae={ref.get('mae', float('nan')):.4f}  rmse={ref.get('rmse', float('nan')):.4f}",
          flush=True)
    return {"measured": got, "config_h120": ref}


def nearest_obs(times: np.ndarray, vals: np.ndarray,
                target: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Observed (time, value) closest to ``target`` within ACTUAL_TOL."""
    t = np.datetime64(target)
    pos = int(np.searchsorted(times, t))
    best = None
    for i in (pos - 1, pos):
        if 0 <= i < len(times):
            dt = abs(pd.Timestamp(times[i]) - target)
            if dt <= ACTUAL_TOL and (best is None or dt < best[0]):
                best = (dt, pd.Timestamp(times[i]), float(vals[i]))
    return None if best is None else (best[1], best[2])


def metric_row(df: pd.DataFrame, origin: str, model: str, n_excluded: int = 0) -> dict:
    col = model
    err = df[col].to_numpy(dtype=float) - df["actual"].to_numpy(dtype=float)
    return {
        "origin": origin, "model": model, "n": int(len(df)),
        "median_bias": float(np.median(err)) if len(err) else float("nan"),
        "mae": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        "excluded": int(n_excluded),
    }


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
def run(limit_stations: int = 0, write: bool = True) -> dict:
    import serving.engines as engines

    t0 = time.time()
    print(f"[data] loading {ALIGNED.name}", flush=True)
    c = engines._load_traj()             # table already drop_gwl_spikes-cleaned
    tbl = c["tbl"]
    last_obs = pd.Timestamp(tbl["time"].max())
    origins = build_origins(last_obs)
    if not origins:
        raise SystemExit(f"no evaluable origin: last obs {last_obs}")

    fleet = pd.read_csv(FLEET_CSV)
    fleet_st = set(fleet["station"].astype(str))
    table_st = set(tbl["Station"].astype(str))

    regime_shift = set(F.flag_regime_shift(
        tbl[["Station", "time", "gwl"]].assign(date=tbl["time"].dt.normalize())))
    scored_st = sorted(fleet_st & table_st)
    universe = [s for s in scored_st if s not in regime_shift]
    if limit_stations:
        universe = universe[:limit_stations]
    obs_stations = scored_st

    print(f"[data] last obs {last_obs} | origins {len(origins)} "
          f"({origins[0].date()} -> {origins[-1].date()})", flush=True)
    print(f"[universe] fleet={len(fleet_st & table_st)} regime_shift={len(fleet_st & table_st & regime_shift)} "
          f"scored={len(universe)}", flush=True)

    # per-station positional index (no big copies) + observed arrays
    pos_by_station = {str(st): pos for st, pos in tbl.groupby("Station", sort=False).indices.items()}
    obs_by_station: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for st in obs_stations:
        sub = tbl.iloc[pos_by_station[st]]
        o = sub.dropna(subset=["gwl"])
        obs_by_station[st] = (o["time"].to_numpy(), o["gwl"].to_numpy(dtype=float))

    xgb_cfg = json.loads(TRAJ_CFG.read_text())
    fnames = list(xgb_cfg["feature_cols"])
    base_cols = [x for x in fnames if x not in ("h", "h_sin", "h_cos")]
    booster = load_xgb_booster()
    import xgboost as xgb

    hs = np.arange(1, 121, dtype=np.float32)
    xgb_X = np.hstack([
        np.zeros((120, len(base_cols)), dtype=np.float32),
        hs[:, None],
        np.sin(2 * np.pi * hs / 120)[:, None],
        np.cos(2 * np.pi * hs / 120)[:, None],
    ])

    rows: list[dict] = []
    counts: list[dict] = []
    for origin in origins:
        t_o = time.time()
        o_np = np.datetime64(origin)
        exclude = {"regime_shift": 0, "anchor_lag_gt_7d": 0, "anchor_sparse": 0,
                   "no_actual": 0, "feature_missing": 0, "error": 0}
        n_origin = 0
        # regime-shift stations that would otherwise have a usable anchor
        for st in (fleet_st & table_st & regime_shift):
            ti, _ = obs_by_station.get(st, (np.array([], dtype="datetime64[ns]"), None))
            if len(ti):
                k = np.searchsorted(ti, o_np, side="right") - 1
                if k >= 0 and (origin - pd.Timestamp(ti[k])) <= ANCHOR_MAX_LAG:
                    exclude["regime_shift"] += 1

        for st in universe:
            ti, gi = obs_by_station[st]
            k = int(np.searchsorted(ti, o_np, side="right") - 1)
            if k < 0:
                exclude["anchor_lag_gt_7d"] += 1
                continue
            anchor_t, anc = pd.Timestamp(ti[k]), float(gi[k])
            if (origin - anchor_t) > ANCHOR_MAX_LAG:
                exclude["anchor_lag_gt_7d"] += 1
                continue
            n_obs_7d = int(((ti > np.datetime64(anchor_t - pd.Timedelta(days=7)))
                            & (ti <= np.datetime64(anchor_t))).sum())
            if n_obs_7d < ANCHOR_MIN_OBS_7D:
                exclude["anchor_sparse"] += 1
                continue
            target = anchor_t + HORIZON
            hit = nearest_obs(ti, gi, target)
            if hit is None:
                exclude["no_actual"] += 1
                continue
            actual_t, actual = hit
            actual_chg = actual - anc
            if abs(actual_chg) > MAX_ABS_ACTUAL_CHANGE:
                exclude["no_actual"] += 1
                continue

            sub = tbl.iloc[pos_by_station[st]]
            hist_d = sub[sub["time"] <= anchor_t].copy()
            hist_d["date"] = hist_d["time"].dt.normalize()
            try:
                feats, _ = F.build_full(hist_d, keep_na=True,
                                        horizon_steps=120, horizon_label=30)
                feats = feats.sort_values("time")
            except Exception:  # noqa: BLE001 - counted, never fatal
                exclude["error"] += 1
                continue
            if feats.empty:
                exclude["feature_missing"] += 1
                continue
            feat_anchor = float(feats["gwl"].iloc[-1])
            if not np.isclose(feat_anchor, anc, atol=1e-4, rtol=0.0):
                # feature frame lost the anchor (spike/flag drop)
                exclude["feature_missing"] += 1
                continue
            # the float32 feature value is what both models actually see
            anc = feat_anchor
            actual_chg = actual - anc

            try:
                pred = engines._predict_deltas(c, feats)
                dl_q50 = float(pred[-1, 1]) + anc
            except Exception:  # noqa: BLE001
                exclude["error"] += 1
                continue

            xgb_X[:, :len(base_cols)] = feats.iloc[-1][base_cols].to_numpy(np.float32)
            xrow = xgb_X[-1:].copy()
            xgb_q50 = float(booster.predict(xgb.DMatrix(xrow, feature_names=fnames))[0]) + anc

            rows.append({
                "origin": str(origin.date()), "station": st,
                "district": str(feats["District"].iloc[-1]),
                "anchor_time": anchor_t, "anchor": anc,
                "anchor_lag_days": round((origin - anchor_t).total_seconds() / 86400, 2),
                "n_obs_7d": n_obs_7d,
                "target_time": target, "actual_time": actual_t, "actual": actual,
                "actual_change": actual_chg,
                "persist": anc, "xgb": xgb_q50, "dl": dl_q50,
                "persist_chg": 0.0, "xgb_chg": xgb_q50 - anc, "dl_chg": dl_q50 - anc,
            })
            n_origin += 1

        counts.append({"origin": str(origin.date()), "n": n_origin, **exclude})
        print(f"[origin] {origin.date()} n={n_origin} excluded={exclude} "
              f"({time.time() - t_o:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no evaluable rows")

    latest = str(origins[-1].date())
    out = summarise(df, counts, latest, last_obs)
    if write:
        write_outputs(df, out, latest)
    print(f"[done] {len(df)} rows in {time.time() - t0:.0f}s", flush=True)
    return out


def summarise(df: pd.DataFrame, counts: list[dict], latest: str,
              last_obs: pd.Timestamp) -> dict:
    per_origin = []
    for origin in sorted(df["origin"].unique()):
        g = df[df["origin"] == origin]
        for m in MODELS:
            per_origin.append(metric_row(g, origin, m))

    pooled = [metric_row(df, "pooled", m) for m in MODELS]
    pm = {r["model"]: r for r in pooled}

    improvements = []
    for r in per_origin:
        if r["model"] == "persist":
            continue
        base = next(x for x in per_origin
                    if x["origin"] == r["origin"] and x["model"] == "persist")
        improvements.append({
            "origin": r["origin"], "model": r["model"],
            "mae_impr_pct": 100 * (base["mae"] - r["mae"]) / base["mae"] if base["mae"] else float("nan"),
            "rmse_impr_pct": 100 * (base["rmse"] - r["rmse"]) / base["rmse"] if base["rmse"] else float("nan"),
        })
    pooled_impr = {
        "model": "xgb",
        "mae_impr_pct": 100 * (pm["persist"]["mae"] - pm["xgb"]["mae"]) / pm["persist"]["mae"],
        "rmse_impr_pct": 100 * (pm["persist"]["rmse"] - pm["xgb"]["rmse"]) / pm["persist"]["rmse"],
    }
    pooled_impr_dl = {
        "model": "dl",
        "mae_impr_pct": 100 * (pm["persist"]["mae"] - pm["dl"]["mae"]) / pm["persist"]["mae"],
        "rmse_impr_pct": 100 * (pm["persist"]["rmse"] - pm["dl"]["rmse"]) / pm["persist"]["rmse"],
    }

    decline = {}
    for th in (DECLINE_M, HIGH_DECLINE_M):
        flags = df["dl_chg"] <= -th
        hits = df["actual_change"] <= -th
        tp = int((flags & hits).sum())
        decline[f"{th:.1f}"] = {
            "flagged": int(flags.sum()), "actual": int(hits.sum()), "tp": tp,
            "precision": (tp / int(flags.sum())) if flags.sum() else None,
            "recall": (tp / int(hits.sum())) if hits.sum() else None,
        }

    latest_df = df[df["origin"] == latest]
    stats = {
        "dataset_last": str(last_obs), "rows": int(len(df)),
        "origins": sorted(df["origin"].unique().tolist()),
        "latest_origin": latest,
        "stations": int(df["station"].nunique()),
        "per_origin": per_origin, "pooled": pooled,
        "improvements": improvements,
        "pooled_improvement": [pooled_impr, pooled_impr_dl],
        "decline_dl": decline,
        "actual_median_change": {
            "pooled": float(df["actual_change"].median()),
            **{o: float(g["actual_change"].median())
               for o, g in df.groupby("origin")},
        },
        "latest": {
            "origin": latest, "n": int(len(latest_df)),
            "models": {m: metric_row(latest_df, latest, m) for m in MODELS},
            "actual_median_change": float(latest_df["actual_change"].median()),
            "decline_dl": {
                f"{th:.1f}": {
                    "flagged": int((latest_df["dl_chg"] <= -th).sum()),
                    "actual": int((latest_df["actual_change"] <= -th).sum()),
                    "tp": int(((latest_df["dl_chg"] <= -th)
                               & (latest_df["actual_change"] <= -th)).sum()),
                } for th in (DECLINE_M, HIGH_DECLINE_M)
            },
        },
        "exclusions": counts,
    }
    return stats


def write_outputs(df: pd.DataFrame, out: dict, latest: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    keep = ["origin", "station", "district", "anchor_time", "anchor",
            "anchor_lag_days", "n_obs_7d", "target_time", "actual_time", "actual",
            "actual_change", "persist", "xgb", "dl", "persist_chg", "xgb_chg", "dl_chg"]
    df[keep].to_csv(OUTDIR / "rows.csv", index=False)

    summ = pd.DataFrame(out["per_origin"] + out["pooled"])
    summ.to_csv(OUTDIR / "summary.csv", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(out, indent=2, default=str),
                                         encoding="utf-8")

    per = out["per_origin"]
    origins = out["origins"]
    md: list[str] = []
    md.append("# AQUIS 2026 rolling 30-day benchmark — persistence vs XGB vs DL\n")
    md.append(f"- Dataset: `data/aligned/table_6h.parquet`, last observation "
              f"**{out['dataset_last']}**, cleaned with `drop_gwl_spikes`.")
    md.append(f"- Origins (anchor-causal, +30d target = anchor + 30d): "
              f"{', '.join(origins)}")
    md.append(f"- Evaluable rows: **{out['rows']}** across "
              f"**{out['stations']}** stations.")
    md.append("- XGB is the tracked q50 artifact loaded from an in-memory copy with "
              "`base_score=0` (repo file untouched). DL is the deployed quantile GRU.")
    md.append("- Errors are level errors (`prediction - actual`); persistence predicts "
              "the anchor level.\n")

    md.append("## Per origin\n")
    rows = []
    for o in origins:
        for m in MODELS:
            r = next(x for x in per if x["origin"] == o and x["model"] == m)
            imp = next((x for x in out["improvements"]
                        if x["origin"] == o and x["model"] == m), None)
            rows.append([o, MODEL_LABEL[m], r["n"], f"{r['median_bias']:+.3f}",
                         f"{r['mae']:.3f}", f"{r['rmse']:.3f}",
                         "" if imp is None else f"{imp['mae_impr_pct']:+.1f}%",
                         "" if imp is None else f"{imp['rmse_impr_pct']:+.1f}%"])
    md.append(md_table(["origin", "model", "n", "median bias (m)", "MAE", "RMSE",
                        "ΔMAE vs persistence", "ΔRMSE vs persistence"], rows))

    md.append("\n## Pooled\n")
    rows = []
    for m in MODELS:
        r = next(x for x in out["pooled"] if x["model"] == m)
        imp = next((x for x in out["pooled_improvement"] if x["model"] == m), None)
        rows.append([MODEL_LABEL[m], r["n"], f"{r['median_bias']:+.3f}",
                     f"{r['mae']:.3f}", f"{r['rmse']:.3f}",
                     "" if imp is None else f"{imp['mae_impr_pct']:+.1f}%",
                     "" if imp is None else f"{imp['rmse_impr_pct']:+.1f}%"])
    md.append(md_table(["model", "n", "median bias (m)", "MAE", "RMSE",
                        "ΔMAE vs persistence", "ΔRMSE vs persistence"], rows))

    lt = out["latest"]
    md.append(f"\n## Latest evaluable origin — {latest} (closest to the current fleet season)\n")
    rows = []
    for m in MODELS:
        r = lt["models"][m]
        rows.append([MODEL_LABEL[m], r["n"], f"{r['median_bias']:+.3f}",
                     f"{r['mae']:.3f}", f"{r['rmse']:.3f}"])
    md.append(md_table(["model", "n", "median bias (m)", "MAE", "RMSE"], rows))
    md.append(f"\nActual median 30-day change at this origin: "
              f"**{lt['actual_median_change']:+.3f} m**.")

    md.append("\n## DL decline flags\n")
    rows = []
    for th, d in out["decline_dl"].items():
        rows.append([th, d["flagged"], d["actual"], d["tp"],
                     "n/a" if d["precision"] is None else f"{d['precision']:.3f}",
                     "n/a" if d["recall"] is None else f"{d['recall']:.3f}"])
    md.append(md_table(["threshold (m)", "flagged", "actual declines", "TP",
                        "precision", "recall"], rows))
    md.append("\nAt the latest origin:")
    rows = []
    for th, d in lt["decline_dl"].items():
        prec = "n/a" if not d["flagged"] else f"{d['tp'] / d['flagged']:.3f}"
        rec = "n/a" if not d["actual"] else f"{d['tp'] / d['actual']:.3f}"
        rows.append([th, d["flagged"], d["actual"], d["tp"], prec, rec])
    md.append(md_table(["threshold (m)", "flagged", "actual declines", "TP",
                        "precision", "recall"], rows))

    md.append("\n## Actual 2026 30-day change\n")
    rows = [[o, f"{v:+.3f}"] for o, v in out["actual_median_change"].items()]
    md.append(md_table(["origin", "median actual 30d change (m)"], rows))

    md.append("\n## Exclusions (clean table)\n")
    rows = [[c["origin"], c["n"], c["regime_shift"], c["anchor_lag_gt_7d"],
             c["anchor_sparse"], c["no_actual"], c["feature_missing"], c["error"]]
            for c in out["exclusions"]]
    md.append(md_table(["origin", "scored", "regime shift", "anchor lag >7d",
                        "sparse anchor", "no actual ±1d", "empty features", "errors"], rows))
    md.append("\n- `sparse anchor` is the Luktra guard: fewer than 14 observations in "
              "the 7 days before the anchor.\n"
              "- Regime-shift stations are excluded from both the forecast and the "
              "actuals (their levels carry a datum offset).\n"
              "- Predictions are produced from the station's causal slice only; the "
              "anchor is never a prediction.\n")

    (OUTDIR / "benchmark_2026.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[out] wrote {OUTDIR}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N stations (smoke test)")
    ap.add_argument("--verify-xgb", action="store_true",
                    help="only run the val-split base_score check")
    ap.add_argument("--no-write", action="store_true", help="do not write outputs")
    args = ap.parse_args()
    if args.verify_xgb:
        verify_xgb_base_score()
        return
    run(limit_stations=args.limit, write=not args.no_write)


if __name__ == "__main__":
    main()
