"""_hydrology.py — causal hydrology-informed feature helpers (block 1).

Shared by the batch builder (``06_features.build``) and the serving paths
(``_trajectory``/``refresh.future_drivers.build_step_features``) so training,
backtesting and inference compute identical values from the same history.

Features produced here
----------------------
Groundwater state / trend
    gwl_dev7, gwl_dev30    deviation from the trailing 7/30-day mean (m);
                           computed by the callers as gwl - gwl_roll{w}_mean.
    gwl_slope7/30/90       Theil-Sen slope (robust, m/day) of the PER-DAY
                           MEDIAN observed GWL over the trailing 7/30/90
                           calendar days.  Daily medians first, then the
                           median of all pairwise day-to-day slopes, so a
                           single telemetry spike cannot tilt the trend.
    gwl_obs7/30/90         coverage: fraction of the w-day window whose days
                           held >= 1 observed (not forward-filled) reading.
    gwl_imputed            1.0 where the row's GWL is a carried-forward value,
                           0.0 where it is an observed reading, NaN if no GWL.

Rainfall memory (antecedent precipitation index, API)
    rain_api7/22/60        API_t = lam * API_{t-1} + rain_t on the 6-hour
                           grid with lam = exp(-(6h/24)/tau), tau in
                           {7, 22, 60} days (mm).
    rain_obs7/22/60        fraction of the tau-window's 6h slots backed by an
                           ACTUAL rainfall observation (fixed denominator = the
                           full window, so an incomplete window reads < 1 instead
                           of pretending to be complete).  Forecast and
                           climatology rain used by the serving path NEVER counts
                           as observed, and neither does a missing historical
                           slot; the numeric API value may still use that
                           forcing.

Causality contract (every function below)
-----------------------------------------
* trailing windows only — never centered, never full-series statistics.
* NaN whenever the history/window requirement is not met, always paired with
  an explicit coverage column — never a silently shortened window.
* the API recursion starts at the station's first observed rain and stays NaN
  before it (no fabricated warm-up); a missing rain slot contributes zero
  accumulation while the decay continues (memory of past rain decays whether
  or not rain is observed that slot).
* GWL provenance comes from the aligned table's ``gwl_obs`` column when the
  table carries it; on legacy tables (built before provenance existed) every
  present reading is treated as observed and ``gwl_imputed`` is 0.
* Rainfall provenance is an explicit observation MASK, not "finite value"
  (``rain_memory_block(..., rain_observed=...)``): the serving path passes a
  mask that is True for observed history only, so forecast/climatology rain can
  drive ``rain_api*`` without ever counting towards ``rain_obs*``.
* nothing here fits any parameter on validation/test data — the only
  station-specific reference is the observed history at or before the row.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import lfilter

STEP_H = 6                       # hours per grid slot
API_TAUS = (7.0, 22.0, 60.0)     # API time constants (days)
SLOPE_WINS = (7, 30, 90)         # trend windows (days)
DEV_WINS = (7, 30)               # deviation windows (days)
MIN_SLOPE_OBS_DAYS = 3           # a slope needs at least this many observed days
MIN_SLOPE_FRAC = 0.5             # ... and at least this fraction of the window


def api_alpha(tau_days: float, step_h: int = STEP_H) -> float:
    """Per-step retention of an exponential memory with time constant tau_days
    sampled every step_h hours: lam = exp(-(step_h/24)/tau_days)."""
    return float(np.exp(-(step_h / 24.0) / float(tau_days)))


def api_series(rain: np.ndarray, tau_days: float) -> np.ndarray:
    """Exponential antecedent precipitation index over a gapless 6h rain series.

    API_t = lam * API_{t-1} + rain_t, zero state before the first observed
    rain, NaN before it.  NaN rain slots are treated as "no accumulation"
    (decay continues).  Causal: output[t] depends only on rain[0..t].
    """
    x = np.asarray(rain, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    if not valid.any():
        return out
    lam = api_alpha(tau_days)
    filled = np.where(valid, x, 0.0)
    # y_t = lam*y_{t-1} + x_t with y_{-1} = 0  (b=[1], a=[1, -lam])
    y = lfilter([1.0], [1.0, -lam], filled)
    seen = np.maximum.accumulate(valid)          # False until first observed rain
    out[seen] = y[seen]
    return out


def rain_memory_block(rain: np.ndarray, rain_observed: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """API + observed-rain coverage for every configured tau.

    ``rain`` is a gapless 6-hourly series (station rows in time order) carrying
    the NUMERIC rainfall forcing — observations where available, and (in the
    serving path) forecast/climatology values for future steps.

    ``rain_observed`` is an optional boolean mask with the same length marking
    which slots are backed by an ACTUAL rainfall observation. When it is given,
    ``rain_obs{tau}`` counts mask-True slots only, so a future forecast or
    climatology value — and a missing historical slot — never contributes
    coverage. When it is None (legacy callers, batch frames whose rainfall has
    no imputation step) coverage falls back to "finite value" and behaviour is
    unchanged.

    Crucially the mask affects COVERAGE ONLY. ``rain_api{tau}`` always uses the
    numeric ``rain`` series, including forecast/climatology forcing, so the
    memory term reflects the best available rainfall knowledge at the step while
    the coverage column honestly reports how much of the window is observed.

    Returns ``rain_api{7,22,60}`` and ``rain_obs{7,22,60}`` arrays of equal
    length; coverage uses a fixed denominator of ``tau * 4`` slots so a short
    or sparse history reads as an incomplete window instead of 1.0.
    """
    x = np.asarray(rain, dtype=np.float64)
    n = x.size
    out: dict[str, np.ndarray] = {}
    if rain_observed is None:
        avail = np.isfinite(x).astype(np.float64)
    else:
        avail = np.asarray(rain_observed, dtype=bool).astype(np.float64)
        if avail.size != n:
            raise ValueError(
                f"rain_observed length {avail.size} != rain length {n}")
    csum = np.cumsum(avail)
    t = np.arange(n)
    for tau in API_TAUS:
        key = int(tau)
        out[f"rain_api{key}"] = api_series(x, tau)
        win = int(round(tau * 24.0 / STEP_H))
        lo = np.maximum(t - win + 1, 0)
        trailing = csum[t] - np.where(lo > 0, csum[np.maximum(lo - 1, 0)], 0.0)
        out[f"rain_obs{key}"] = trailing / float(win)
    return out


def theil_sen_trailing(y: np.ndarray, w: int,
                       min_obs: int = MIN_SLOPE_OBS_DAYS,
                       min_frac: float = MIN_SLOPE_FRAC
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Trailing Theil-Sen slope + observed coverage on a gapless DAILY series.

    ``y`` holds the daily values (NaN = day unobserved); ``w`` is the trailing
    window in days.  Returns ``(slope, coverage)`` arrays of len(y):

      slope[p]      median of (y[j] - y[i]) / (j - i) over all observed day
                    pairs i < j inside the trailing window ending at p
                    (m/day when y is GWL), or NaN when the window has fewer
                    than ``min_obs`` observed days or coverage < ``min_frac``;
      coverage[p]   observed days / w  (fixed denominator — an early or
                    sparse window reads < 1, never a silent 1.0).
    """
    y = np.asarray(y, dtype=np.float64)
    w = int(w)
    n = y.size
    slope = np.full(n, np.nan, dtype=np.float64)
    cov = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or w < 2:
        return slope, cov
    obs = np.isfinite(y)
    ii, jj = np.triu_indices(w, 1)
    den = (jj - ii).astype(np.float64)
    n_full = n - w + 1
    if n_full > 0:
        W = np.lib.stride_tricks.sliding_window_view(y, w)
        M = np.lib.stride_tricks.sliding_window_view(obs, w)
        cnt = M.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            delta = (W[:, jj] - W[:, ii]) / den
            delta[~(M[:, jj] & M[:, ii])] = np.nan
            s = np.nanmedian(delta, axis=1)
        c = cnt / float(w)
        bad = (cnt < min_obs) | (c < min_frac)
        s[bad] = np.nan
        slope[w - 1:] = s
        cov[w - 1:] = c
    for p in range(min(n, w - 1)):               # short-history prefix
        seg = obs[: p + 1]
        cnt = int(seg.sum())
        c = cnt / float(w)
        cov[p] = c
        if cnt < min_obs or c < min_frac:
            continue
        idx = np.nonzero(seg)[0]
        v = y[idx]
        i2, j2 = np.triu_indices(cnt, 1)
        slope[p] = float(np.median((v[j2] - v[i2]) / (j2 - i2)))
    return slope, cov


def gwl_observed(gwl: pd.Series, obs_flags) -> pd.Series:
    """True where the row's GWL is an actual reading (not forward-filled).

    ``obs_flags`` is the aligned table's ``gwl_obs`` column (1 observed,
    0 carried, NaN no reading) or None for legacy tables without provenance,
    in which case every present reading counts as observed.
    """
    present = gwl.notna()
    if obs_flags is None:
        return present
    obs = pd.to_numeric(obs_flags, errors="coerce")
    return present & (obs == 1.0)


def gwl_imputed_flag(gwl: pd.Series, obs_flags) -> pd.Series:
    """1.0 carried-forward, 0.0 observed, NaN where GWL is missing."""
    out = pd.Series(np.nan, index=gwl.index, dtype="float64")
    present = gwl.notna()
    if obs_flags is None:
        out[present] = 0.0
        return out
    obs = pd.to_numeric(obs_flags, errors="coerce")
    out[present] = (obs[present] == 0.0).astype(float)
    return out


def gwl_trend_block(times: pd.Series, gwl: pd.Series, observed: pd.Series,
                    station: pd.Series) -> pd.DataFrame:
    """Per-row trailing Theil-Sen slopes + observed-day coverage.

    Works on observed readings only (per-day median, NaN days excluded from
    the slope but counted against coverage), per station, trailing windows of
    SLOPE_WINS days.  Returns columns ``gwl_slope{w}`` / ``gwl_obs{w}``
    aligned to the input index; rows without enough history/coverage stay NaN.
    """
    cols = ([f"gwl_slope{w}" for w in SLOPE_WINS]
            + [f"gwl_obs{w}" for w in SLOPE_WINS])
    out = pd.DataFrame(np.nan, index=times.index, columns=cols, dtype="float64")
    if len(times) == 0:
        return out
    day = pd.to_datetime(times).dt.normalize()
    vals = gwl.where(observed)
    st = station.astype(str)
    for _, idx in st.groupby(st).groups.items():
        sub_day = day.loc[idx]
        daily = vals.loc[idx].groupby(sub_day).median()
        if daily.empty:
            continue
        cal = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        arr = daily.reindex(cal).to_numpy(dtype=np.float64)
        pos = (sub_day - cal[0]).dt.days.to_numpy()
        pos = np.clip(pos, 0, arr.size - 1)
        for w in SLOPE_WINS:
            s, c = theil_sen_trailing(arr, w)
            out.loc[idx, f"gwl_slope{w}"] = s[pos]
            out.loc[idx, f"gwl_obs{w}"] = c[pos]
    return out
