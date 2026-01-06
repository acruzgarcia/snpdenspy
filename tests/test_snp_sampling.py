import numpy as np
import time

from snpdenspy.snp.snp_rv_continuous import SNPSimpleDistribution
from snpdenspy.snp.snp_sampler import snp_sampler

from snpdenspy.util.statistical_testing.permutation_tests import (
    ks_test_1d_permutation, 
    cvm_test_1d_permutation
)

# globals for tests
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

def assert_same_distribution(
    x,
    y,
    *,
    method="ks",          # "ks", "cvm", "energy"
    alpha=1e-3,
    n_perm=2000,
    random_state=0,
    verbose=False,
):
    """
    Raises AssertionError if the test rejects H0 at level alpha.
    Designed for deterministic unit tests (fixed random_state).
    """
    if method == "ks":
        stat, p = ks_test_1d_permutation(x, y, n_perm=n_perm, random_state=random_state)
    elif method == "cvm":
        stat, p = cvm_test_1d_permutation(x, y, n_perm=n_perm, random_state=random_state)
    else:
        raise ValueError("Unknown method.")

    if verbose:
        color = RED if p < alpha else GREEN
        print(f"Test stat: {stat: .3f}, p-value: {color}{p: .3f}{RESET}")

    if p < alpha:
        raise AssertionError(f"{method} rejects equality: p={p:.6g} < alpha={alpha} (stat={stat:.6g})")

    return stat, p

def time_execution(func, *args, **kwargs):
    """
    Execute function and return (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def test_statistical_testing_framework():
    
    import numpy as np
    rng = np.random.default_rng(78)
    
    # a and b are equals distributions, not c 
    a = rng.normal(0, 1, size=2000)
    b = rng.normal(0, 1, size=2000)
    c = rng.normal(2, 1.5, size=2000)

    # deterministic + low flakiness
    for m in ("ks","cvm"):
        assert_same_distribution(
            a, b, method=m, alpha=1e-3, n_perm=2000, random_state=123)
    
    # expected failing tets, capture exception
    for me in ("ks", "cvm"):
        try:
            assert_same_distribution(
                a, c, method=m, alpha=1e-3, n_perm=2000, random_state=249)      
        except AssertionError:
            pass    

def test_snp_rvs_sampling(include_timing=False):

    # --- Create Toy distribution ---
    toy = SNPSimpleDistribution()
    
    # Example coefficients to create a bimodal shape
    coeffs = [0.5, -0.8, 0.2, 0.0, 0.0]
    mu, sigma = 12, 5
    frozen_dist = toy(c1=coeffs[0], c2=coeffs[1], c3=coeffs[2], c4=coeffs[3], c5=coeffs[4], loc=mu, scale=sigma)

    nsamples = 1200
    samples_ppf, elapsed1 = time_execution(frozen_dist.rvs, size=nsamples) # samples with root-finding algo
    samples_accept_reject, elapsed2 = time_execution(
        snp_sampler, c_coeffs=coeffs, size=nsamples, loc=mu, scale=sigma)
    if include_timing:
        print(f"Execution time for {nsamples}: inv-cdf: {elapsed1:.4f}, AR: {elapsed2}")

    # deterministic + low flakiness
    assert_same_distribution(
        samples_ppf, samples_accept_reject, method="ks", alpha=1e-3, n_perm=2000, random_state=123)
    assert_same_distribution(
        samples_accept_reject, samples_ppf, method="cvm", alpha=1e-3, n_perm=2000, random_state=56)

def test_snp_vs_normal_sampling(include_timing=False):

    import numpy as np
    from snpdenspy.snp.snp_sampler import snp_sampler
    from scipy.stats import norm

    # Example coefficients to create a bimodal shape
    coeffs = [0, 0, 0, 0, 0]
    mu, sigma = 12, 5
    
    nsamples = 500
    samples_normal, elapsed1 = time_execution(norm.rvs, size=nsamples, loc=mu, scale=sigma)
    samples_accept_reject, elapsed2 = time_execution(
        snp_sampler,c_coeffs=coeffs, size=nsamples, loc=mu, scale=sigma)
    if include_timing:
        print(f"Execution time for {nsamples}: norm.rvs: {elapsed1:.4f}, AR: {elapsed2}")

    # deterministic + low flakiness
    assert_same_distribution(
        samples_normal, samples_accept_reject, method="ks", alpha=1e-3, n_perm=2000, random_state=123)
    assert_same_distribution(
        samples_normal, samples_accept_reject, method="cvm", alpha=1e-3, n_perm=2000, random_state=56)


if __name__ == "__main__":

    test_statistical_testing_framework()
    print(f"Test {test_statistical_testing_framework.__qualname__} {GREEN}passed{RESET}!")
    test_snp_rvs_sampling()
    print(f"Test {test_snp_rvs_sampling.__qualname__} {GREEN}passed{RESET}!")
    test_snp_vs_normal_sampling(include_timing=False)
    print(f"Test {test_snp_vs_normal_sampling.__qualname__} {GREEN}passed{RESET}!")
    
    