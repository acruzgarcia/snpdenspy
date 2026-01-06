
import numpy as np
from scipy.optimize import least_squares

from .snp_moments import (
    EZr_Hn_matrix,
    snp_standardized_raw_moments_from_c, 
    compute_raw_moments_from_loc_scale
)

def snp_fit_moment_matching(
    x: np.ndarray,
    K: int,
    fit_loc_scale=True,
):
    """
    Moment matching for SNP with polynomial P(z)=1+sum c_k H_k(z), density ∝ φ(z) P(z)^2.

    Parameters
    ----------
    x : array
        Sample data.
    K : int
        Number of c coefficients to fit (c1..cK).
    fit_loc_scale : bool
        If True, include loc and scale in optimization; else mat moments on standardized x.
    
    Returns
    -------
    result : OptimizeResult
        From scipy.optimize.least_squares.
    """

    x = np.asarray(x, dtype=float)
    # We fit as many moments as order of the poly + 2 (loca and scale, or 0 and 1) 
    n_moments_to_fit = K + 2
    moments_to_fit =  range(1, n_moments_to_fit + 1)
    empirical_moments_func = lambda x: np.array([np.mean(x ** m) for m in moments_to_fit], dtype=float) 

    # Initial guesses
    initial_c = np.zeros(K, dtype=float)

    if fit_loc_scale:
        initial_loc = float(np.mean(x))
        initial_scale = float(np.std(x, ddof=1))
        
        # weights
        # crude rule: higher moments noisier => downweight them
        w = np.array([1.0 / max(1.0, np.std(x) ** m) for m in moments_to_fit], dtype=float)
        p0 = np.concatenate([initial_c, [initial_loc, initial_scale]])    
    else:
        # If not fitting loc/scale, standardize data and match moments of Z directly
        mu0 = float(np.mean(x))
        s0 = float(np.std(x, ddof=1))
        x = (x - mu0) / s0
        w = np.ones(shape=len(moments_to_fit))
        p0 = initial_c.copy()

    # Sample raw moments of X
    sample_raw_moments = empirical_moments_func(x)

    # Precompute E[z^r H_m] for efficiency
    EZr_Hn_mat = EZr_Hn_matrix(n_moments_to_fit, 2*K)

    def model_moments_from_params(params):
        
        c = params[:K]

        # raw moments of standardized Z: mz[0]=E[Z], ..., mz[max_m-1]=E[Z^max_m]
        mz = snp_standardized_raw_moments_from_c(
            c_coeffs=c, 
            r_max=n_moments_to_fit,
            E_zn_Hn_matrix=EZr_Hn_mat)

        if fit_loc_scale:
            loc = float(params[K])
            scale = float(params[K + 1])
            
            mx = compute_raw_moments_from_loc_scale(moments_std=mz, loc=loc, scale=scale)
            
            # select only requested orders
            return np.array([mx[m] for m in moments_to_fit], dtype=float)
        else:
            # match Z moments directly
            return np.array([mz[m] for m in moments_to_fit], dtype=float)

    def residuals(params):
        
        mm = model_moments_from_params(params)
        res = (mm - sample_raw_moments) * np.sqrt(w)
        return res

    # Least-square optimization
    res = least_squares(residuals, p0, method="lm", max_nfev=2000, verbose=0)

    return {
        "c": res.x[:K],
        "loc": res.x[K] if fit_loc_scale else None, 
        "scale": res.x[K+1] if fit_loc_scale else None,
        "success": bool(res.success),
        "message": res.message,
        "opt_result": res        
    }
