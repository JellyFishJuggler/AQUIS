"""Quality — can this station's record be trusted for a forecast?

Covers completeness, provenance (measured vs carried-forward), telemetry gaps, the
spike mask the pipeline applies, and datum-jump detection. Every check is
reproducible from the aligned table; none depends on the models.

The point of this page is that a hydrologist should be able to decide whether to
trust a station *before* reading its forecast.

(Source-coverage and artifact-availability reporting lives in `data_quality.py`;
this page is the per-station diagnostic.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _charts as C
import _data as D
import _hydro_metrics as M

SPIKE_Z = 30.0          # mirrors 06_features.drop_gwl_spikes
REGIME_JUMP_M = 15.0    # mirrors 06_features.flag_regime_shift


@st.cache_data(ttl="10m", show_spinner="Reading this station's record…")
def _frame(station: str) -> pd.DataFrame | None:
    return D.station_frame(station)


@st.cache_data(ttl="10m", show_spinner=False)
def _stations() -> pd.DataFrame:
    return D.station_universe()


@st.cache_data(ttl="10m", show_spinner="Scanning all stations for datum jumps…")
def _datum_scan() -> pd.DataFrame:
    """Annual medians per station, to flag >15 m datum jumps.

    One pass over three thin columns; the same rule the feature pipeline uses to
    drop a station wholesale (`06_features.flag_regime_shift`).
    """
    df = D.read_parquet_safe(D.ALIGNED / "table_6h.parquet",
                             columns=["Station", "time", "gwl"])
    if df is None or df.empty:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["gwl"] = pd.to_numeric(df["gwl"], errors="coerce")
    df = df.dropna(subset=["time", "gwl"])
    df["year"] = df["time"].dt.year
    med = df.groupby(["Station", "year"])["gwl"].median().unstack()
    base_yrs = [y for y in range(2021, 2025) if y in med.columns]
    if not base_yrs:
        return pd.DataFrame()
    base = med[base_yrs].median(axis=1)
    dev = med.sub(base, axis=0).abs()
    out = pd.DataFrame({"baseline_2021_24_m": base.round(2),
                        "max_annual_dev_m": dev.max(axis=1).round(2),
                        "worst_year": dev.idxmax(axis=1)})
    out = out[out["max_annual_dev_m"].notna()]
    return out.sort_values("max_annual_dev_m", ascending=False).reset_index()


def _pick() -> str | None:
    uni = _stations()
    if uni.empty:
        return None
    names = sorted(uni["Station"].astype(str))
    default = 0
    if st.session_state.get("focus_station") in names:
        default = names.index(st.session_state["focus_station"])
    station = st.selectbox("Station", names, index=default)
    st.session_state["focus_station"] = station
    return station


st.title("Quality")
st.caption(
    "Whether a station's record supports a forecast — checked independently of the "
    "models, which inherit whatever is wrong here."
)

if not _stations().shape[0]:
    st.error("**Data unavailable.** `data/aligned/table_6h.parquet` not found.")
    st.stop()
station = _pick()
if not station:
    st.error("**No stations found** in the aligned table.")
    st.stop()

df = _frame(station)
if df is None or df.empty:
    st.warning(f"**No record for {station}** in `table_6h.parquet`.")
    st.stop()

q = M.quality_block(df)
st.metric("Verdict", str(q.get("verdict", "—")).upper(),
          help="ok = fresh and ≥80% of slots observed · sparse = under 80% · "
               "stale = last observation more than 7 days old")

k = st.columns(5)
k[0].metric("6 h slots", f"{q.get('slots', 0):,}")
k[1].metric("Measured", f"{100 * float(q.get('observed_frac', np.nan)):.1f}%"
            if q.get("observed_frac") is not None else "—")
k[2].metric("Carried forward",
            f"{100 * float(q.get('carried_frac', 0)):.1f}%"
            if q.get("has_provenance") else "n/a")
k[3].metric("Last observation",
            f"{float(q.get('stale_days', np.nan)):.0f} d ago"
            if pd.notna(q.get("stale_days")) else "—")
k[4].metric("Provenance column",
            "present" if q.get("has_provenance") else "absent",
            help="Without `gwl_obs` the pipeline cannot distinguish a measurement "
                 "from a carried-forward value.")

if not q.get("has_provenance"):
    st.warning(
        "**This aligned table predates the provenance column.** `gwl_obs` is written "
        "by `04b_align_6h.gwl_track`; until that runs, every present value is treated "
        "as measured — so carried-forward readings are invisible here *and* in the "
        "hydrology features."
    )

monthly = q.get("_monthly")
if monthly is not None and len(monthly):
    st.subheader("Monthly completeness")
    strip = monthly.to_frame("observed fraction")
    strip.index = strip.index.astype(str)
    st.plotly_chart(
        C.bars(strip.index, strip["observed fraction"], height=240,
               name="observed fraction", color=C.RAIN,
               title="share of 6 h slots measured, by month",
               y_title="observed fraction"),
        width="stretch", config={"displayModeBar": False},
    )
    worst = monthly.nsmallest(5)
    st.caption("Weakest months: "
               + " · ".join(f"{k_} {100 * v:.0f}%" for k_, v in worst.items()))

# ---- spike and gap checks -------------------------------------------------
st.divider()
st.subheader("Spike and gap checks")
gwl = pd.to_numeric(df["gwl"], errors="coerce")
med = float(gwl.median())
mad = float((gwl - med).abs().median() * 1.4826)
spikes = (gwl - med).abs() > SPIKE_Z * max(mad, 1e-6)
t = df["time"].sort_values()
longest = float((t.diff().dt.total_seconds() / 3600.0).max()) if len(t) > 1 else np.nan
s = st.columns(3)
s[0].metric("Spikes flagged", f"{int(spikes.sum()):,}",
            help=f"|level − median| > {SPIKE_Z:.0f} × MAD — the mask "
                 "`06_features.drop_gwl_spikes` applies at build time.")
s[1].metric("Robust spread (MAD)", f"{mad:.3f} m")
s[2].metric("Longest gap", f"{longest:.0f} h" if np.isfinite(longest) else "—")
if int(spikes.sum()):
    st.caption("First 10 flagged slots — these become NaN at feature-build time.")
    st.dataframe(df.loc[spikes, ["time", "gwl"]].head(10),
                 width="stretch", hide_index=True)

# ---- datum jumps -----------------------------------------------------------
st.divider()
st.subheader("Datum-jump scan (all stations)")
scan = _datum_scan()
if scan.empty:
    st.info("Not enough multi-year history to scan for datum jumps.")
else:
    flagged = scan[scan["max_annual_dev_m"] > REGIME_JUMP_M]
    d = st.columns(2)
    d[0].metric("Stations flagged", f"{len(flagged):,}",
                help=f"Annual median deviating more than {REGIME_JUMP_M:.0f} m from the "
                     "2021–24 baseline — the telemetry datum-error signature.")
    d[1].metric("Scanned", f"{len(scan):,}")
    st.dataframe(scan.head(25), width="stretch", hide_index=True, height=320)
    this = scan[scan["Station"].astype(str) == str(station)]
    if not this.empty and float(this["max_annual_dev_m"].iloc[0]) > REGIME_JUMP_M:
        st.error(
            f"**{station} is itself flagged** (max annual deviation "
            f"{float(this['max_annual_dev_m'].iloc[0]):.1f} m). `06_features.build_full` "
            "drops such stations wholesale, so the pooled models will not forecast it."
        )
    st.caption(
        "A jump this large is almost always a sensor or datum change, not hydrology. "
        "Re-baselining the series is a data decision, not a modelling one."
    )
