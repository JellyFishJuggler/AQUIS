"""
GPR model definition for groundwater time series.

Composite kernel:
  - Matern(v=2.5): smooth long-term trend
  - ExpSineSquared: annual/monsoon periodic cycle
  - WhiteKernel: sensor/measurement noise
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    ExpSineSquared,
    Matern,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler


def build_kernel(x_scaler: StandardScaler):
    """
    Composite kernel for groundwater level time series.

    All length-scale and periodicity parameters operate in SCALED
    time_hours space.  The periodicity and its bounds are computed
    dynamically from the fitted scaler so they adapt to any station.
    """
    hours_per_year = 365.25 * 24
    scaled_period = hours_per_year / x_scaler.scale_[0]
    scaled_period_lo = 0.5 * scaled_period
    scaled_period_hi = 1.5 * scaled_period

    kernel = (
        Matern(length_scale=1.0, length_scale_bounds=(0.5, 100.0), nu=2.5)
        * ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        + ExpSineSquared(
            length_scale=0.5,
            length_scale_bounds=(0.1, 100.0),
            periodicity=scaled_period,
            periodicity_bounds=(0.3, 5.0),
        )
        * ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1))
    )
    return kernel


def build_gpr(
    x_scaler: StandardScaler,
    n_restarts: int = 5,
) -> GaussianProcessRegressor:
    """
    Build the GPR regressor.

    n=6,307 -> O(n^3) ~ 2.5e11 FLOPs.  Expect several minutes per
    fit.  If training on multiple pooled stations, consider sparse GP
    approximations (inducing points / SparseGP).
    """
    # TODO:
    # Replace sklearn internal optimizer restarts with manual restart loop
    # to expose per-restart progress (Restart 1/5, Restart 2/5, ...).
    # This would require iterating n_restarts_optimizer manually, calling
    # gpr.fit() with warm_start=True, and comparing log-marginal-likelihoods.
    kernel = build_kernel(x_scaler)
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=1e-6,
        normalize_y=False,
        random_state=42,
    )
