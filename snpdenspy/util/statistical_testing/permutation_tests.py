import numpy as np
from .statistics import (
    ks_statistic_1d_two_samples,
    cvm_statistic_1d_two_samples
)

def permutation_pvalue(stat_obs: float, stat_perm: np.ndarray, *, alternative: str = "greater") -> float:
    """
    Computes a permutation p-value with a +1 correction:
      p = (1 + #{T_perm >= T_obs}) / (B + 1)  for alternative="greater"
    """
    stat_perm = np.asarray(stat_perm)
    if alternative == "greater":
        return (1.0 + np.sum(stat_perm >= stat_obs)) / (stat_perm.size + 1.0)
    if alternative == "less":
        return (1.0 + np.sum(stat_perm <= stat_obs)) / (stat_perm.size + 1.0)
    if alternative == "two-sided":
        # assumes statistic is non-negative; for signed stats you'd use abs
        return (1.0 + np.sum(stat_perm >= stat_obs)) / (stat_perm.size + 1.0)
    raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")


def permute_two_sample_stats(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    *,
    n_perm: int = 1000,
    random_state=None,
) -> tuple[float, float]:
    """
    Generic two-sample permutation test. Returns (stat_obs, p_value).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = x.shape[0], y.shape[0]

    stat_obs = float(stat_fn(x, y))

    pooled = np.concatenate([x, y], axis=0)
    n = pooled.shape[0]

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    stat_perm = np.empty(int(n_perm), dtype=float)
    for b in range(int(n_perm)):
        idx = rng.permutation(n)
        xb = pooled[idx[:nx]]
        yb = pooled[idx[nx:]]
        stat_perm[b] = stat_fn(xb, yb)

    p = permutation_pvalue(stat_obs, stat_perm, alternative="greater")
    return stat_obs, p

# two wrappers of permute_two_sample_stats
def ks_test_1d_permutation(x, y, *, n_perm=2000, random_state=0) -> tuple[float, float]:
    return permute_two_sample_stats(
        np.asarray(x).ravel(),
        np.asarray(y).ravel(),
        ks_statistic_1d_two_samples,
        n_perm=n_perm,
        random_state=random_state,
    )

def cvm_test_1d_permutation(x, y, *, n_perm=2000, random_state=0) -> tuple[float, float]:
    return permute_two_sample_stats(
        np.asarray(x).ravel(),
        np.asarray(y).ravel(),
        cvm_statistic_1d_two_samples,
        n_perm=n_perm,
        random_state=random_state,
    )