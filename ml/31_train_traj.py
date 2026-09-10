"""31_train_traj — shared direct multi-horizon XGBoost for the trajectory v2.

One pooled model per quantile (q05/q50/q95) predicts GWL(t+h)-GWL(t) from the
pipeline feature vector (past windows only) + explicit horizon features. The
anchor row is scored with h=1..120 -> 120 genuine independent deltas. No
recursion, no future observed values, anchor-level never predicted.

Early stopping on a time-ordered tail of the train frame (fit/eval split by
time). Validation = the full 120-horizon frame for a clean per-horizon curve.

Outputs -> models/traj_xgb_{q05,q50,q95}.joblib + models/traj_config.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import importlib

F = importlib.import_module("06_features")

ROOT = Path(__file__).resolve().parent
FTRAJ = ROOT / "data" / "features_traj"
OUTM = ROOT / "models"
OUTM.mkdir(parents=True, exist_ok=True)

ALPHAS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
MAX_H = 120
SEED = 42
EVAL_FRAC = 0.10      # time-ordered tail used for early stopping
PARAMS = dict(
    max_depth=7, learning_rate=0.05, min_child_weight=60,
    subsample=0.9, colsample_bytree=0.8,
    n_estimators=900, tree_method="hist", n_jobs=12, random_state=SEED,
)
EARLY = 25


def _load(sub: str) -> pd.DataFrame:
    return pd.read_parquet(FTRAJ / f"{sub}.parquet")


def _fnames() -> list[str]:
    return list(F.NUM_COLS) + ["h", "h_sin", "h_cos"]


def _per_h(va: pd.DataFrame, q5: np.ndarray, q50: np.ndarray,
           q9: np.ndarray, steps: list[int]) -> list[dict]:
    y = va["target_delta"].to_numpy()
    hh = va["h"].to_numpy()
    rows = []
    for h in steps:
        m = hh == h
        if not m.any():
            continue
        yh = y[m]; p05, p50, p95 = q5[m], q50[m], q9[m]
        e = yh - p50
        rows.append({
            "h": int(h), "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(p50 - yh)),
            "coverage": float(np.mean((p05 <= yh) & (yh <= p95))),
            "halfwidth_median": float(np.median((p95 - p05) / 2)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny subset, quick")
    args = ap.parse_args()

    t0 = time.time()
    tr = _load("train")
    va = _load("val")

    fn = _fnames()
    order = [c for c in fn if c in tr.columns]

    if args.smoke:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(tr), size=min(400_000, len(tr)), replace=False)
        tr = tr.iloc[np.sort(keep)].copy()
        va = va[va["h"] <= 30].copy()

    tr = tr.sort_values("time").reset_index(drop=True)
    n_eval = max(1, int(len(tr) * EVAL_FRAC))
    dtrain = xgb.DMatrix(tr.iloc[:-n_eval][order], label=tr.iloc[:-n_eval]["target_delta"])
    deval = xgb.DMatrix(tr.iloc[-n_eval:][order], label=tr.iloc[-n_eval:]["target_delta"])
    dval = xgb.DMatrix(va[order], label=va["target_delta"])
    print(f"fit {dtrain.num_row():,} | eval {deval.num_row():,} | val {dval.num_row():,} "
          f"| features {len(order)}", flush=True)

    preds: dict[str, np.ndarray] = {}
    for name, alpha in ALPHAS.items():
        p = dict(PARAMS, objective="reg:quantileerror", quantile_alpha=alpha)
        bst = xgb.train(p, dtrain, num_boost_round=PARAMS["n_estimators"],
                        evals=[(deval, "eval")], early_stopping_rounds=EARLY,
                        verbose_eval=False)
        bst.save_model(OUTM / f"traj_xgb_{name}.json")
        preds[name] = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        tail = np.sqrt(np.mean((va["target_delta"].to_numpy() - preds[name]) ** 2))
        print(f"  {name}: best_iter={bst.best_iteration} val_overall_rmse={tail:.4f}",
              flush=True)

    q5 = preds["q05"]; q50 = preds["q50"]; q9 = preds["q95"]
    q5 = np.minimum(q5, q50); q9 = np.maximum(q9, q50)   # enforce order at all h
    steps = [1, 2, 3, 4, 8, 12, 28, 60, 120] if not args.smoke else [1, 2, 8, 30]
    rows = _per_h(va, q5, q50, q9, steps)
    val_metrics = pd.DataFrame(rows)

    cfg = {
        "paradigm": "direct multi-horizon shared model (no recursion)",
        "target": "GWL(t+h*6h) - GWL(t); level = anchor + delta (anchor observed only)",
        "feature_cols": order,
        "quantiles": ALPHAS,
        "max_h": MAX_H,
        "early_stopping_tail_frac": EVAL_FRAC,
        "fit_rows": int(dtrain.num_row()),
        "eval_rows": int(deval.num_row()),
        "val_rows_scored": int(dval.num_row()),
        "smoke": args.smoke,
        "val_per_h_metrics": val_metrics.round(4).to_dict("records"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUTM / "traj_config.json").write_text(json.dumps(cfg, indent=2))

    print("\nper-horizon validation (delta RMSE m / coverage / half-width):")
    print(val_metrics.round(4).to_string(index=False))
    print(f"\nelapsed {(time.time() - t0):.0f}s -> models/traj_xgb_*.json + traj_config.json")


if __name__ == "__main__":
    main()