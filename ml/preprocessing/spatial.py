"""Spatial augmentation for AQUIS groundwater forecasting (paper's XGB-SF idea).

The paper (Dule & Sharma) augments the standard model with *spatial features*
(its top-performing variant, XGB-SF, attained the best internal R2). In the
paper these are raster textures from neighbouring pixels. AQUIS works with point
telemetry stations that carry real coordinates and elevation (RL_MSL), so we
adapt the same idea to a *station neighbourhood*: for each station we derive
static spatial textures from its k nearest neighbouring stations (mean/std/range
of RL_MSL elevation, plus the spacing/roughness of the neighbourhood).

All features here are STATIC per station (derived only from the fixed station
coordinates/elevation catalogue), so they are leak-free by construction and can be
safely attached to every chronological row of a station's series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_STATION_COL = "Station"
_LAT = "Latitude"
_LON = "Longitude"
_ELEV = "RL_MSL"

# Names of the spatial-augmentation columns added to the feature set.
SPATIAL_FEATURES = [
    "spatial_nb_elev_mean",   # focal mean elevation of k nearest stations
    "spatial_nb_elev_std",    # focal std (roughness of the local terrain)
    "spatial_nb_elev_range",  # focal elevation range (topographic relief)
    "spatial_nb_spread",      # mean distance to k nearest stations (station density)
]


def station_spatial_table(meta: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Compute static spatial-augmentation features for every station.

    ``meta`` must contain at least ``Station``, ``Latitude``, ``Longitude`` and
    ``RL_MSL`` (elevation may be NaN for some stations; those are excluded from
    the neighbourhood elevation statistics per-station where missing).

    Returns a DataFrame indexed by ``Station`` with the ``SPATIAL_FEATURES``.
    """
    meta = meta.copy()
    meta = meta.dropna(subset=[_LAT, _LON])
    if meta.empty:
        return pd.DataFrame(columns=SPATIAL_FEATURES)

    coords = meta[[_LON, _LAT]].to_numpy(dtype=float)
    elev = meta[_ELEV].to_numpy(dtype=float)
    elev_finite = np.isfinite(elev)
    elev_imputed = elev.copy()
    if elev_finite.any():
        elev_imputed[~elev_finite] = np.nanmedian(elev[elev_finite])
    else:
        elev_imputed[~elev_finite] = 0.0

    n = len(meta)
    k = max(3, min(k, n - 1))
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    if k + 1 == 1:
        idx = np.atleast_2d(idx)
    else:
        idx = np.atleast_2d(idx)
    # remove self-neighbour (first result is the station itself)
    nbr_idx = idx[:, 1:]

    rows = []
    for i in range(n):
        nb = nbr_idx[i]
        nb = nb[np.isfinite(nb)].astype(int)
        if len(nb) == 0:
            nb = np.array([i])
        en = elev_imputed[nb]
        dn = np.linalg.norm(coords[i] - coords[nb], axis=1)
        rows.append({
            "Station": meta[_STATION_COL].iloc[i],
            "spatial_nb_elev_mean": float(en.mean()),
            "spatial_nb_elev_std": float(en.std()) if len(en) > 1 else 0.0,
            "spatial_nb_elev_range": float(en.max() - en.min()) if len(en) > 1 else 0.0,
            "spatial_nb_spread": float(dn.mean()),
        })

    return pd.DataFrame(rows).set_index(_STATION_COL)


def attach_spatial_features(df: pd.DataFrame, spatial_table: pd.DataFrame) -> pd.DataFrame:
    """Left-join static spatial-augmentation features onto a station dataframe."""
    df = df.copy()
    if spatial_table is None or spatial_table.empty or _STATION_COL not in df.columns:
        for c in SPATIAL_FEATURES:
            df[c] = 0.0
        return df
    df = df.merge(spatial_table.reset_index(), on=_STATION_COL, how="left")
    for c in SPATIAL_FEATURES:
        if c not in df.columns or df[c].isna().any():
            df[c] = df.get(c, 0.0).fillna(0.0)
    return df
