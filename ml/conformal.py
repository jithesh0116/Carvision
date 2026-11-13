"""Split conformal prediction intervals.

Method (documented for the report): after training, a separate *calibration* set that the model
never saw is used to compute absolute residuals on the log1p(price) scale,
    r_i = | log1p(y_i) - f(x_i) |.
With n calibration points and miscoverage alpha, q is the ceil((n+1)(1-alpha))-th smallest
residual. For a new car the interval is  expm1(f(x) -/+ q).  Under exchangeability this has
marginal coverage >= 1 - alpha (Vovk et al.; Lei et al., 2018). Limitation: the interval has a
constant *relative* width for all cars, so it is not adaptive to individual difficulty.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(abs_residuals, alpha: float) -> float:
    r = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(r)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        raise ValueError(f"Calibration set too small ({n} rows) for {100 * (1 - alpha):.0f}% coverage.")
    return float(r[k - 1])


def interval_from_log(pred_log, q: float):
    pred_log = np.asarray(pred_log, dtype=float)
    lower = np.clip(np.expm1(pred_log - q), 0, None)
    upper = np.expm1(pred_log + q)
    return lower, upper


def empirical_coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))
