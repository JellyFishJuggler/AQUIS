"""Page 3 — Feature explorer: what the model uses vs what is carried untrained."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _charts as C
import _data as D


@st.cache_data(ttl="10m", show_spinner=False)
def _groups() -> pd.DataFrame:
    return D.feature_group_table()


@st.cache_data(ttl="10m", show_spinner=False)
def _importance() -> pd.DataFrame | None:
    return D.feature_importance()


@st.cache_data(ttl="10m", show_spinner=False)
def _diagnostics() -> dict | None:
    return D.diagnostics()


@st.cache_data(ttl="10m", show_spinner=False)
def _lag_curves() -> pd.DataFrame | None:
    return D.lag_curves()


@st.cache_data(ttl="10m", show_spinner=False)
def _manifest() -> list | None:
    return D.manifest()


st.title("Feature explorer")
st.caption(
    "Every column the pipeline builds, grouped by modelling intent, with the "
    "trained / carried-untrained distinction made explicit."
)

groups = _groups()
if groups.empty:
    st.error(
        "**Data/artifact unavailable.** Could not import `06_features` to read "
        "`NUM_COLS` / `HYDRO_COLS` (missing module or dependency)."
    )
    st.stop()

num, hydro = D.hydro_columns()

# ---- the distinction, stated up front ---------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Trained numeric features", f"{len(num)}")
c2.metric("Ordinals (added by trainer)", "2")
c3.metric("Carried hydrology features", f"{len(hydro)}")

st.info(
    "**Two feature sets live in the same frame.** The frozen models consume "
    f"`num_cols` ({len(num)} numeric + `st_id`/`dist_id`). The "
    f"{len(hydro)} `HYDRO_COLS` are calculated in the base feature frames and "
    "the future-driver builder, but omitted from the trajectory long-frame and "
    "all frozen model schemas. The current inference models do not request them, "
    "so they are not present in the model input matrix. Promoting them requires "
    "a matched training/serving definition and a deliberate retrain. Nothing on "
    "this page changes model behaviour."
)

# ---- grouped inventory ------------------------------------------------
st.subheader("Feature inventory by group")
order = [
    "groundwater state / trend", "rainfall / weather",
    "seasonal / calendar", "spatial / static",
    "candidate hydrology (carried, untrained)",
]
chosen = st.multiselect(
    "Groups", order, default=order,
)
view = groups[groups["group"].isin(chosen)].copy()
view = view[["group", "feature", "status"]].reset_index(drop=True)

def _style(_row):
    return ["", "", "background-color: #e7f5ea"] if _row["status"] == "trained" \
        else (["", "", "background-color: #fff4e0"]
              if _row["status"] == "carried, untrained" else ["", "", ""])

st.dataframe(
    view.style.apply(_style, axis=1),
    width="stretch", hide_index=True, height=460,
)
st.caption("Green = used by the frozen models · Amber = computed but untrained.")

# ---- hydrology block detail -------------------------------------------
st.subheader("The candidate hydrology block")
st.markdown(
    """
These columns encode hydrologist reasoning that the current feature set lacks:
memory of past rainfall, the well's own trend, and how much of each window is
actually backed by observations.
"""
)
hid = pd.DataFrame([
    {"column": "gwl_dev7 / gwl_dev30", "meaning": "anchor GWL minus its own trailing 7 / 30 d mean (m)"},
    {"column": "gwl_slope7 / 30 / 90", "meaning": "Theil–Sen slope of the per-day median observed GWL (m/day)"},
    {"column": "gwl_obs7 / 30 / 90", "meaning": "fraction of the slope window made of observed (not carried) days"},
    {"column": "gwl_imputed", "meaning": "1 = this row's GWL was forward-filled, 0 = observed, NaN = no reading"},
    {"column": "rain_api7 / 22 / 60", "meaning": "exponential antecedent precipitation index, τ = 7 / 22 / 60 d (mm)"},
    {"column": "rain_obs7 / 22 / 60", "meaning": "fraction of the τ window backed by a real rain-gauge observation"},
])
st.dataframe(hid, width="stretch", hide_index=True)

st.warning(
    "**`rain_api*` and `rain_obs*` mean different things — do not conflate them.** "
    "`rain_api*` is the numeric memory term: it uses the rainfall *forcing* "
    "available at that step, including forecast and climatology values for future "
    "steps. `rain_obs*` is a *coverage* column: it counts only slots backed by an "
    "actual rain-gauge observation, over a fixed τ·4-slot denominator, so forecast "
    "and climatology rain never counts as observed and coverage decays across the "
    "horizon."
)
st.error(
    "**Train/serve mismatch — these features are NOT fair to train yet.** "
    "In batch, `rain_api*` is built from rainfall known *at the anchor*. In "
    "serving, the recursion is advanced across the forecast window using future "
    "forecast/climatology rainfall. Training on realized future rainfall is "
    "leakage and is explicitly disallowed, so **the hydrology block stays "
    "experimental/untrained** until either a point-in-time archive of numeric "
    "driver values exists, or the serving definition is pinned to anchor-only "
    "information. See `docs/ml-hydro-train-serve.md`."
)

# ---- what the fitted models actually used ------------------------------
st.subheader("What the fitted models actually relied on")
imp = _importance()
diag = _diagnostics()
t1, t2 = st.tabs(["XGBoost gain", "Permutation importance (stride)"])

with t1:
    if imp is None:
        st.info("**Artifact unavailable.** `outputs/feature_importance.csv` not found.")
    else:
        top = imp.head(20)
        st.dataframe(top, width="stretch", hide_index=True)
        st.bar_chart(top.set_index("feature")["gain"],
                     width="stretch", height=380)
        st.caption(
            "Gain from the pooled 30-day model. Because that model stopped at a "
            "very small number of trees, these scores are dominated by the "
            "seasonal and anchor-level columns."
        )

with t2:
    if not diag or not diag.get("permutation"):
        st.info("**Artifact unavailable.** `outputs/diagnostics.json` not found.")
    else:
        perm = pd.DataFrame(diag["permutation"])
        st.dataframe(perm, width="stretch", hide_index=True, height=380)
        st.caption(
            f"Basis: **{diag.get('basis', '—')}** "
            f"({diag.get('n_stride_rows', '—')} rows). Positive = shuffling the "
            "feature hurt RMSE. Several features score exactly 0.0, which is the "
            "signature of near-total missingness, not of irrelevance."
        )
        vif = diag.get("vif", {})
        if vif.get("constant_excluded"):
            st.warning(
                "Collinearity was **not** actually measured on this sample: "
                f"`{vif['constant_excluded']}` were constant and every VIF came "
                "back NaN. The 'no VIF > 10' claim in the model card is therefore "
                "vacuous for this data."
            )

# ---- recharge lag ------------------------------------------------------
st.subheader("Rainfall → groundwater recharge lag")
lc = _lag_curves()
if lc is None:
    st.info("**Artifact unavailable.** `outputs/lag_curves.csv` not found.")
else:
    st.dataframe(lc.head(40), width="stretch", hide_index=True)
    numcols = [c for c in lc.columns if pd.api.types.is_numeric_dtype(lc[c])]
    if numcols:
        # Bars, not a line: lag is a discrete day offset, and a connecting line
        # implies intermediate values that were never computed.
        st.plotly_chart(
            C.bars(lc[lc.columns[0]], lc[numcols[0]], height=340,
                   name="median Pearson r",
                   title="GWL vs rainfall, by lag (median across stations)",
                   y_title="median Pearson r"),
            width="stretch", config={"displayModeBar": False},
        )
    st.caption(
            "The corpus measured a best recharge lag of roughly three weeks. Note "
            "that the trained model's rainfall windows are only 1 / 7 / 30 days, so this "
        "response time is **not currently represented** in the trained features."
    )

# ---- source coverage ---------------------------------------------------
st.subheader("Driver coverage behind the features")
man = _manifest()
if not man:
    st.info("**Artifact unavailable.** `data/meta/manifest.json` not found.")
else:
    mdf = pd.DataFrame(man)
    keep = [c for c in ("source", "rows", "stations", "districts", "span", "min", "max")
            if c in mdf.columns]
    st.dataframe(mdf[keep], width="stretch", hide_index=True)
    st.caption(
        "The driver side is thin: point weather exists for a handful of gauges in "
        "three districts, rainfall for a few dozen gauges, and river observations "
        "stop well before the 2026 test period. Much of this sparsity is a data "
        "constraint, not a code defect."
    )
