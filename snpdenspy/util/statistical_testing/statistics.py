import numpy as np
from typing import Callable

# K-S statistics (two samples)
def ks_statistic_1d_two_samples(x, y) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    x.sort()
    y.sort()

    # Evaluate ECDF difference on the pooled sorted grid
    grid = np.unique(np.concatenate([x, y]))
    Fx = np.searchsorted(x, grid, side="right") / x.size
    Fy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(Fx - Fy)))

# K-S statistics (one sample vs model CDF)
def ks_statistic_1d_sample_model(x, model_cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    x_sorted = np.sort(x)
    ecdf = np.arange(1, n + 1, dtype=float) / n
    
    mcdf = np.asarray(model_cdf(x_sorted), dtype=float).reshape(-1)
    return float(np.max(np.abs(ecdf - mcdf)))

# Cramer-von Mises statistics (two samples)
def cvm_statistic_1d_two_samples(x, y) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    x.sort()
    y.sort()

    grid = np.unique(np.concatenate([x, y]))
    Fx = np.searchsorted(x, grid, side="right") / x.size
    Fy = np.searchsorted(y, grid, side="right") / y.size

    # Approximate integral by averaging over grid points (works well for unit tests)
    return float(np.mean((Fx - Fy) ** 2))

# Cramer-von Mises statistics (one sample vs model CDF)
def cvm_statistic_1d_sample_model(x, model_cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    x_sorted = np.sort(x)
    ecdf = np.arange(1, n + 1, dtype=float) / n
    
    mcdf = np.asarray(model_cdf(x_sorted), dtype=float).reshape(-1)
    return float(np.mean((ecdf - mcdf) ** 2))
