"""Model page — pipeline config, eval metrics, feature importances/coefficients."""

import altair as alt
import pandas as pd
import streamlit as st

from _model import (load_model_metrics, load_importance, load_coefficients,
                    load_feature_config, load_honest_metrics, load_overlap_summary,
                    load_residual_acf, load_residual_spatial, load_spatial_cv_metrics,
                    load_ablation, load_diagnostics, load_diagnostics_permutation)

HORIZON = 30
H_LABEL = "30 days"

st.set_page_config(page_title="AQUIS — model", page_icon=":material/model_training:", layout="wide")
st.title("Model — 30-day groundwater forecast")
st.caption("Pooled global model on a 6-hourly grid; the target is the **30-day change** "
           "GWL(t+120) − GWL(t). Today's GWL is an anchor feature — it is never predicted.")

cfg = load_feature_config()
basis = st.segmented_control(
    "Metric basis", options=["level", "delta"],
    format_func=lambda v: {"level": "GWL level (m)", "delta": "Change (m)"}[v],
    default="level", key="model_basis",
)

metrics = load_model_metrics()
m = metrics[metrics["model"].str.endswith(" (delta)") == (basis == "delta")]
m = m[m["model"].str.replace(" (delta)", "", regex=False).isin(["xgboost", "ridge", "persistence", "climatology"])]
if basis == "delta":
    m["model"] = m["model"].str.replace(" (delta)", "", regex=False)
m = m[m["horizon"] == HORIZON]
m = m.sort_values("rmse")

acol, bcol, ccol = st.columns(3)
acol.metric("Models benchmarked", f"{len(metrics['model'].unique())}")
acol.caption(f"{HORIZON}-day horizon (120 six-hour steps)")
honest = load_honest_metrics()
if not honest.empty:
    hx = honest[honest["model"] == "xgb"].iloc[0]
    hpers = honest[honest["model"] == "persist"].iloc[0]
    bcol.metric("Honest 30-d RMSE (non-overlap windows)", f"{hx['basis_stride_rmse']:.3f} m",
                delta=f"{(hx['basis_stride_rmse'] - hpers['basis_stride_rmse']):+.3f} m vs persistence")
    bcol.caption("1 independent 30-day window per station-row-block (85% fewer raw rows counted)")
else:
    bcol.metric("Best RMSE (30 d)", f"{m['rmse'].min():.3f} m")
cval = m.loc[m["model"] == "xgboost", "within_05"]
bup = m.loc[m["model"] == "persistence", "within_05"]
ccol.metric("XGBoost within ±0.5 m", f"{cval.iloc[0]*100:.1f}%",
            delta=f"{(cval.iloc[0]-bup.iloc[0])*100:+.1f}pp vs persistence"
            if len(bup) else None)

st.subheader(f"Test-set metrics, {H_LABEL} horizon")
show = m.rename(columns={
    "model": "Model", "rmse": "RMSE (m)", "mae": "MAE (m)", "r2": "R²",
    "within_025": "±0.25 m", "within_05": "±0.5 m", "within_10": "±1.0 m",
    "rows": "Rows", "feat_days": "Feature days",
})
st.dataframe(show[["Model", "RMSE (m)", "MAE (m)", "R²", "±0.25 m", "±0.5 m", "±1.0 m", "Rows"]],
             use_container_width=True, hide_index=True)

st.subheader("RMSE by model")
bars = alt.Chart(m).mark_bar(color="#4ecca3").encode(
    x=alt.X("rmse:Q", title="RMSE (m)"),
    y=alt.Y("model:N", title=None, sort="-x"),
    tooltip=["model", "rmse", "mae", "within_05"],
).properties(height=200)
st.altair_chart(bars, use_container_width=True)

with st.expander("Honest validation — spatial CV, overlap & residual autocorrelation", expanded=False):
    st.caption("The 6-hour grid means consecutive test rows share ~99% of the same 30-day "
               "target window. These checks re-score the forecast on **non-overlapping "
               "windows** (1 row per 30 days, `stride`), retrain on **geographically "
               "held-out station blocks**, and quantify residual autocorrelation.")
    honest = load_honest_metrics()
    ov = load_overlap_summary()
    if not honest.empty:
        hcols1, hcols2, hcols3 = st.columns(3)
        hx = honest[honest["model"] == "xgb"].iloc[0]
        hpers = honest[honest["model"] == "persist"].iloc[0]
        hcols1.metric("Raw RMSE (all rows)", f"{hx['basis_raw_rmse']:.3f} m")
        hcols2.metric("Non-overlap RMSE (stride)", f"{hx['basis_stride_rmse']:.3f} m",
                      delta=f"{(hx['basis_stride_rmse'] - hpers['basis_stride_rmse']):+.3f} m vs persistence")
        hcols3.metric("Independent 30-d windows", f"{ov.get('independent_windows', 0):,}",
                      delta=f"{(ov.get('effective_N_ratio', 0) * 100):.2f}% of raw rows")
        showh = honest.rename(columns={
            "model": "Model", "basis_raw_rmse": "Raw RMSE", "basis_stride_rmse": "Stride RMSE",
            "window_rmse_mean": "Window mean RMSE", "window_rmse_median": "Window med RMSE",
            "basis_raw_within_05": "Raw ±0.5 m", "basis_stride_within_05": "Stride ±0.5 m",
            "stride_rows": "Independent rows", "n_windows": "Windows",
        })
        st.dataframe(showh[["Model", "Raw RMSE", "Stride RMSE", "Window mean RMSE",
                            "Window med RMSE", "Raw ±0.5 m", "Stride ±0.5 m",
                            "Independent rows"]], use_container_width=True, hide_index=True)
    else:
        st.write("honest_metrics.csv missing — run `07_evaluate.py`.")

    scv = load_spatial_cv_metrics()
    if not scv.empty:
        st.markdown("**Spatial CV — leave-one-block-out station retraining** (RMSE on "
                    "stations whose whole region was unseen at train time):")
        st.dataframe(scv[["block", "n_train_stations", "n_test_stations", "test_rows",
                          "rmse", "mae", "r2"]].round(3), use_container_width=True, hide_index=True)
        st.caption(f"fold mean {scv['rmse'].mean():.2f} m / median {scv['rmse'].median():.2f} m "
                   f"(temporal-split headline is {hx['basis_raw_rmse']:.2f} m) — spatial "
                   f"generalisation is not the binding constraint; the 2026 temporal holdout is.")
    else:
        st.write("spatial_cv_metrics.csv missing — run `ml/validation/spatial_cv.py`.")

    d1, d2 = st.columns(2)
    with d1:
        acf = load_residual_acf()
        if acf:
            st.markdown("**Temporal residual autocorrelation** (6h lags):")
            st.dataframe(pd.DataFrame({
                "lag (6h steps)": list(acf.get("pooled_acf", {}).keys()),
                "ACF": [f"{v:.3f}" for v in acf.get("pooled_acf", {}).values() if v is not None],
            }), use_container_width=True, hide_index=True)
            st.caption(f"effective test sample ≈ **{acf.get('effective_sample_size_est'):,.0f} rows** "
                       f"(raw {acf.get('raw_test_rows'):,}) — lag-1 ACF {acf.get('pooled_acf', {}).get('lag1')}.")
    with d2:
        sp = load_residual_spatial()
        if sp:
            st.markdown("**Residual spatial autocorrelation** (per-station mean error):")
            st.metric("Moran's I", f"{sp['moran_I']:.3f}", f"p={sp['moran_p']:.3f}")
            st.metric("Geary's C", f"{sp['geary_C']:.3f}", f"p={sp['geary_p']:.3f}")
            st.caption(f"{sp['n_stations']} stations, k={sp['k']} nearest neighbours. "
                       "p<0.05 ⇒ error clusters in space (spatial structure under-modelled).")

with st.expander("Feature ablation — is every driver paying its way?", expanded=False):
    abl = load_ablation()
    if not abl.empty:
        base = abl.set_index("config").loc["baseline"]
        abl = abl.copy()
        abl["delta_raw_rmse"] = abl["raw_rmse"] - base["raw_rmse"]
        keep = abl[["config", "num_feats", "n_trees", "raw_rmse", "stride_rmse",
                    "delta_raw_rmse", "val_rmse_delta"]].round(4)
        st.dataframe(keep.rename(columns={
            "config": "Config", "num_feats": "Features", "n_trees": "Trees",
            "raw_rmse": "Raw RMSE", "stride_rmse": "Stride RMSE",
            "delta_raw_rmse": "Δ RMSE vs baseline", "val_rmse_delta": "Val RMSE",
        }), use_container_width=True, hide_index=True)
        st.caption("Positive Δ RMSE = dropping the group HURTS (group adds value). "
                   "Rain, weather and calendar are positive; river/canal and the GWL "
                   "lags are neutral-to-noise at 30 d given the anchor + calendar.")
    else:
        st.write("ablation.csv missing — run `10_ablate.py`.")

with st.expander("Diagnostics — collinearity (VIF), feature influence & sensitivity"):
    diag = load_diagnostics()
    dperm = load_diagnostics_permutation()
    if diag:
        st.caption("Runs on the **non-overlap (stride) test subset** "
                   f"({diag.get('n_stride_rows', 0):,} rows), so feature influence "
                   "isn't inflated by 120-step window overlap. River/weather drivers "
                   "are mostly missing in 2026, so VIF uses a mean-imputed sample.")
        d1, d2, d3 = st.columns(3)
        vifraw = diag.get("vif", {})
        vifd = pd.DataFrame({"feature": vifraw.get("feature", []),
                             "vif": vifraw.get("vif", [])}) if vifraw.get("feature") else pd.DataFrame()
        if not vifd.empty:
            vifd = vifd.dropna()
            topv = vifd.sort_values("vif", ascending=False).head(6)
            d1.metric("VIF > 10 (collinear)", f"{len(vifd[vifd['vif'] > 10]):d}")
            d1.caption("ranked: " + ", ".join(
                f"{r['feature']}={r['vif']:.0f}" for _, r in topv.iterrows()))
        sens = diag.get("sensitivity", {})
        feats = sens.get("features", [])
        swings = sens.get("swing_mean_m", [])
        if feats:
            d2.metric("Max single-driver swing",
                      f"{max(swings) if swings else 0:.2f} m",
                      delta=f"{feats[int(swings.index(max(swings)))] if swings else ''}")
            d2.caption("mean |Δ 30-day change| moving the driver 2.5→97.5 pct "
                       "(covariates held at each row's own values)")
        perm = diag.get("permutation", [])
        if perm:
            top = max(perm, key=lambda r: r["importance_mean"])
            d3.metric("Most important at stride",
                      f"{top['feature']} +{top['importance_mean']:.4f} m",
                      delta=f"of {diag.get('n_stride_rows', 0):,} rows")
            d3.caption("permutation RMSE increase — all features are ≈0: the pooled "
                       "30-d delta is near-agnostic to every single driver on the "
                       "independent windows (consistent with the honest re-score).")

        st.markdown("**Permutation importance** (stride RMSE increase when the feature "
                    "is shuffled; mean ± std over repeats):")
        if not dperm.empty:
            bar = alt.Chart(dperm.head(12)).mark_bar(color="#4ecca3").encode(
                x=alt.X("importance_mean:Q", title="Δ RMSE (m)"),
                y=alt.Y("feature:N", title=None, sort="-x"),
                tooltip=["feature", "importance_mean", "importance_std"],
            ).properties(height=260)
            st.altair_chart(bar, use_container_width=True)

        st.markdown("**VIF** (variance inflation factor, mean-imputed sample):")
        if not vifd.empty:
            st.dataframe(vifd.assign(vif=vifd["vif"].round(2)).rename(columns={
                "feature": "Feature", "vif": "VIF"}).sort_values("VIF", ascending=False),
                use_container_width=True, hide_index=True)
            st.caption(f"constant-excluded: {', '.join(diag.get('vif', {}).get('constant_excluded', [])) or '—'} "
                       "(zero variance in the sample). Nothing exceeds the classic VIF>10 "
                       "redline — the feature set is not collinear.")

        st.markdown("**Sensitivity** — forecast movement from each top driver "
                    "(2.5th → 97.5th percentile, covariates unchanged):")
        if feats and swings:
            sens_df = pd.DataFrame({"feature": feats, "swing_m": swings}).sort_values(
                "swing_m", ascending=False)
            sbar = alt.Chart(sens_df).mark_bar(color="#4ecca3").encode(
                x=alt.X("swing_m:Q", title="mean |Δ 30-d change| (m)"),
                y=alt.Y("feature:N", title=None, sort="-x"),
                tooltip=["feature", "swing_m"],
            ).properties(height=240)
            st.altair_chart(sbar, use_container_width=True)
            st.caption("Season (doy_cos), current level (gwl) and monsoon state move the "
                       "forecast most; the ordinal station/district IDs dominate gain but "
                       "carry little interpretable signal on their own.")
    else:
        st.write("diagnostics.json missing — run `12_diagnostics.py`.")

with st.expander("Feature importances / coefficients"):
    kind = st.radio("Source", ["xgboost gain", "ridge |coef|"], horizontal=True, key="imp_kind")
    if kind == "xgboost gain":
        imp = load_importance()
        imp = imp.sort_values("gain", ascending=False).head(20)
        imp["label"] = imp["feature"].fillna("(overall)")
        chart = alt.Chart(imp).mark_bar(color="#4ecca3").encode(
            x=alt.X("gain:Q", title="gain"),
            y=alt.Y("label:N", title=None, sort="-x"),
            tooltip=["feature", "gain"],
        )
    else:
        coef = load_coefficients()
        coef["abs_coef"] = coef["abs_coef"].abs()
        coef = coef.sort_values("abs_coef", ascending=False).head(20)
        chart = alt.Chart(coef).mark_bar(color="#4ecca3").encode(
            x=alt.X("abs_coef:Q", title="|coef|"),
            y=alt.Y("feature:N", title=None, sort="-x"),
            tooltip=["feature", "coef"],
        )
    st.altair_chart(chart, use_container_width=True)

with st.expander("Pipeline configuration"):
    st.json({
        "horizon_days": cfg["horizons"],
        "steps_per_day": cfg.get("steps_per_day"),
        "num_features": len(cfg["num_cols"]),
        "stations": len(cfg["stations"]),
        "districts": len(cfg["districts"]),
        "rows": {"train": cfg["train_rows"], "val": cfg["val_rows"], "test": cfg["test_rows"]},
        "extra_features": cfg.get("extra_features", {}),
        "built_at": cfg.get("built_at"),
    })
    st.caption("Static soil (8) and LULC (9) columns are fetched and shown in the app but "
               "gated out of the model — two experiments showed them redundant with the "
               "station/district ordinals (see README, items 6-7). Full lifecycle: "
               "`ml/MODEL_CARD.md`.")