"""Monitoring — fleet triage. Which wells need attention right now, and why.

Reads a recent window for all stations (row-group pushdown on `time`) and derives
descriptive indicators only: recency, data completeness, 30-day level change,
30-day rainfall, and where the current level sits in its own history. No model is
involved; this is the screen a hydrogeologist would open first.

Page contract: module-level script (run directly by `st.navigation`), not a
`render()` function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _data as D

WINDOW_DAYS = 120


@st.cache_data(ttl="10m", show_spinner="Reading the recent record for all stations…")
def _fleet(days: int = WINDOW_DAYS) -> pd.DataFrame | None:
    df = D.recent_window(days=days)
    if df is None or df.empty:
        return None
    gwl = pd.to_numeric(df["gwl"], errors="coerce")
    obs = gwl.notna()
    if "gwl_obs" in df.columns:
        obs = obs & (pd.to_numeric(df["gwl_obs"], errors="coerce") == 1.0)
    df = df.assign(_obs=obs, _gwl=gwl)
    last = df["time"].max()
    g = df.groupby("Station")

    rows = pd.DataFrame({
        "last_observed": g.apply(lambda x: x.loc[x["_obs"], "time"].max(),
                                 include_groups=False),
        "last_any": g["time"].max(),
        "n_slots": g.size(),
        "n_observed": g["_obs"].sum(),
        "latest_gwl": g["_gwl"].last(),
    })
    rows["coverage"] = (rows["n_observed"] / rows["n_slots"]).round(3)
    rows["recency_days"] = ((last - rows["last_observed"]).dt.total_seconds()
                            / 86400.0).round(1)

    # 30-day change in the observed daily median
    dd = (df[df["_obs"]].assign(date=df["time"].dt.normalize())
          .groupby(["Station", "date"])["_gwl"].median().reset_index())
    cut = last - pd.Timedelta(days=30)

    def _recent(gp: pd.DataFrame) -> pd.Series:
        cur = gp[gp["date"] >= cut]["_gwl"]
        old = gp[gp["date"] < cut]["_gwl"]
        return pd.Series({
            "gwl_30d_ago": float(old.iloc[-1]) if len(old) else np.nan,
            "gwl_now": float(cur.iloc[-1]) if len(cur) else np.nan,
        })

    rows = rows.join(dd.groupby("Station").apply(_recent, include_groups=False))
    rows["change_30d_m"] = (rows["gwl_now"] - rows["gwl_30d_ago"]).round(3)

    if "rain" in df.columns:
        dsum = (df.assign(date=df["time"].dt.normalize())
                  .groupby(["Station", "date"])["rain"].sum().reset_index())
        rows = rows.join(dsum[dsum["date"] >= cut].groupby("Station")["rain"]
                         .sum().rename("rain_30d_mm"))
        rows["rain_30d_mm"] = rows["rain_30d_mm"].round(1)
    if "District" in df.columns:
        rows = rows.join(df.drop_duplicates("Station").set_index("Station")["District"])
    return rows.reset_index()


def _condition(r: pd.Series) -> str:
    """Single-word triage label. Data problems outrank trends."""
    if pd.isna(r.get("recency_days")) or r["recency_days"] > 14:
        return "no recent data"
    if pd.notna(r.get("coverage")) and r["coverage"] < 0.5:
        return "sparse"
    ch = r.get("change_30d_m")
    if pd.isna(ch):
        return "cannot assess"
    if ch <= -0.5:
        return "falling fast"
    if ch <= -0.2:
        return "falling"
    if ch >= 0.5:
        return "rising fast"
    if ch >= 0.2:
        return "rising"
    return "stable"


st.title("Monitoring")
st.caption(
    "Every station in the record, screened on the recent window. Indicators are "
    f"descriptive statistics over the last {WINDOW_DAYS} days — no model, no forecast."
)

fleet = _fleet()
if fleet is None or fleet.empty:
    st.error(
        "**Data unavailable.** `ml/data/aligned/table_6h.parquet` could not be read, "
        "or the recent window holds no rows."
    )
    st.stop()

fleet = fleet.copy()
fleet["condition"] = fleet.apply(_condition, axis=1)

# ---- headline counts -------------------------------------------------------
c = fleet["condition"].value_counts()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Stations", f"{len(fleet):,}")
k2.metric("Falling", int(c.get("falling", 0) + c.get("falling fast", 0)))
k3.metric("Rising", int(c.get("rising", 0) + c.get("rising fast", 0)))
k4.metric("Stable", int(c.get("stable", 0)))
k5.metric("Data problems", int(c.get("no recent data", 0) + c.get("sparse", 0)))

st.caption(
    "Thresholds: falling ≤ −0.2 m / 30 d · rising ≥ +0.2 m / 30 d · "
    "'no recent data' = last observation older than 14 days · sparse = under 50% "
    "coverage. Every indicator is shown as a column so you can re-judge it."
)

# ---- filters ---------------------------------------------------------------
f1, f2, f3 = st.columns([2, 2, 3])
with f1:
    conds = sorted(fleet["condition"].unique())
    pick_cond = st.multiselect("Condition", conds, default=conds)
with f2:
    if "District" in fleet.columns:
        dists = sorted(fleet["District"].dropna().astype(str).unique())
        pick_dist = st.multiselect("District", dists, default=[])
    else:
        pick_dist = []
with f3:
    sort_by = st.selectbox(
        "Sort by",
        ["change_30d_m (asc)", "change_30d_m (desc)", "recency_days (desc)",
         "coverage (asc)", "rain_30d_mm (asc)", "station"], index=0)

view = fleet[fleet["condition"].isin(pick_cond)]
if pick_dist and "District" in view.columns:
    view = view[view["District"].astype(str).isin(pick_dist)]

sort_map = {
    "change_30d_m (asc)": ("change_30d_m", True),
    "change_30d_m (desc)": ("change_30d_m", False),
    "recency_days (desc)": ("recency_days", False),
    "coverage (asc)": ("coverage", True),
    "rain_30d_mm (asc)": ("rain_30d_mm", True),
    "station": ("Station", True),
}
col, asc = sort_map[sort_by]
if col in view.columns:
    view = view.sort_values(col, ascending=asc, na_position="last")

show = [c_ for c_ in ("Station", "District", "condition", "latest_gwl",
                      "change_30d_m", "rain_30d_mm", "recency_days", "coverage",
                      "n_observed") if c_ in view.columns]
st.dataframe(view[show], width="stretch", hide_index=True, height=420)
st.caption(f"{len(view):,} of {len(fleet):,} stations shown.")

with st.expander("Distribution of 30-day change", expanded=False):
    ch = fleet["change_30d_m"].dropna()
    if len(ch):
        hist = np.histogram(ch.clip(-3, 3), bins=48, range=(-3, 3))
        st.bar_chart(pd.DataFrame({"stations": hist[0]},
                                  index=np.round(hist[1][:-1], 2)))
        st.caption(
            f"median {ch.median():+.3f} m · p05 {ch.quantile(0.05):+.3f} · "
            f"p95 {ch.quantile(0.95):+.3f} m. Clipped to ±3 m for display."
        )

st.download_button("Download this screen (CSV)",
                   data=view[show].to_csv(index=False).encode(),
                   file_name="aquis_monitoring.csv", mime="text/csv")
st.info("Next: open **Station** in the sidebar to read a hydrograph, or "
        "**Water balance** to judge recharge conditions.")
