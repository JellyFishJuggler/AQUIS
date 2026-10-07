"""_dbg_rain — throwaway diagnostic for the rain_obs serving offset.

Prints the mask, the window boundaries and the coverage at a few steps so the
off-by-one can be settled from observed numbers rather than reasoning.

Run:  venv/bin/python _dbg_rain.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _hydrology as H
from refresh.future_drivers import (build_step_features, future_weather_matrix,
                                    _rain_memory_steps)

N_HIST = 200
ANCHOR = pd.Timestamp("2026-09-05 00:00:00")


def fixture(n=N_HIST):
    rng = np.random.default_rng(7)
    times = pd.date_range(ANCHOR - pd.Timedelta(hours=(n - 1) * H.STEP_H), ANCHOR,
                          freq="6h")
    i = np.arange(n, dtype=float)
    return pd.DataFrame({
        "time": times,
        "gwl": -10.0 + 0.001 * i + rng.normal(0, 0.05, n),
        "rain": 1.0 + 0.1 * i,
        "temp": 30 + 0.01 * i, "humidity": 60 + 0.01 * i,
        "solar": 200.0, "wind_speed": 10.0, "pressure": 990.0,
        "river_level": 200.0, "canal_level": 300.0,
    })


FNAMES = ["gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
          "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
          "rain_1d", "rain_7d", "rain_30d", "temp", "temp_7d",
          "humidity", "humidity_7d", "solar", "wind_speed", "pressure",
          "river_level", "canal_level", "year", "month_sin", "month_cos",
          "doy_sin", "doy_cos", "hour_sin", "hour_cos", "monsoon",
          "h", "h_sin", "h_cos"]
COLS = ["rain_api7", "rain_obs7", "rain_obs22", "rain_obs60"]

def main() -> None:
    print("== constants ==")
    print("STEP_H      =", H.STEP_H)
    print("API_TAUS    =", H.API_TAUS)
    print("win(tau=7)  =", int(round(7 * 24.0 / H.STEP_H)))

    hist = fixture()
    fw, _ = future_weather_matrix(station="S1", district="ALIGARH", anchor_t=ANCHOR,
                                  steps=120, om_frame=None, climatology=None,
                                  rain_clim=None)
    cols_ref, n_hist, n_hist_obs = _rain_memory_steps(hist.sort_values("time"), fw,
                                                      ANCHOR, 120)
    print("\n== _rain_memory_steps ==")
    print("n_hist =", n_hist, " n_hist_obs =", n_hist_obs)
    for c in COLS:
        print(f"  ref {c}[0:3] = {np.round(cols_ref[c][:3], 6)}")
    print("  ref rain_api7[0]  =", np.round(cols_ref['rain_api7'][0], 6))
    print("  ref rain_api7[-1] =", np.round(cols_ref['rain_api7'][-1], 6))

    hist_rain = hist.sort_values("time")["rain"].to_numpy(float)
    series = np.concatenate([hist_rain, fw["rain"].to_numpy(float)])
    mask = np.concatenate([np.isfinite(hist_rain), np.zeros(fw["rain"].size, bool)])
    blk = H.rain_memory_block(series, mask)
    print("\n== raw block over full series ==")
    print("  blk rain_obs7[196:203] =", np.round(blk['rain_obs7'][196:203], 6))
    print("  blk rain_obs7[225:230] =", np.round(blk['rain_obs7'][225:230], 6))
    print("  blk rain_api7[199]     =", np.round(blk['rain_api7'][199], 6))

    X, _ = build_step_features(station="S1", district="ALIGARH", anchor_t=ANCHOR,
                               hist=hist, fnames=FNAMES + COLS, om_frame=None,
                               climatology=None, rain_clim=None, steps=120,
                               mode="future")
    df = pd.DataFrame(X, columns=FNAMES + COLS)
    print("\n== build_step_features output ==")
    for c in COLS:
        print(f"  df {c}[0:3] = {np.round(df[c].to_numpy()[:3], 6)}")
    print("  df rain_1d[0] =", round(float(df['rain_1d'].iloc[0]), 6),
          " expected =", round(float(hist['rain'].iloc[-3] + hist['rain'].iloc[-2]
                                     + hist['rain'].iloc[-1] + 8.0 / 4.0), 6))
    print("\n== step -> tau=7 window bounds (series coords) ==")
    for k in (0, 1, 2, 27, 28):
        end = n_hist - 1 + k
        lo = max(0, end - 28 + 1)
        print(f"  k={k:3d}  end={end}  window=[{lo},{end}]  "
              f"obs_in_window={int(mask[lo:end + 1].sum())}/28")


if __name__ == "__main__":
    main()
