import numpy as np
from scipy.optimize import brentq

from snpdenspy.snp.snp_eval_functions import (
    pdf_std_eval,
    cdf_std_eval,
    log_pdf_std_eval
)
from snpdenspy.snp.snp_moments import (
    compute_raw_moments_from_loc_scale,
    snp_standardized_raw_moments_from_c,
)

from snpdenspy.snp.snp_fit_mm import snp_fit_moment_matching
from snpdenspy.snp.snp_fit_mle import snp_fit_mle
from snpdenspy.snp.snp_sampler import snp_sampler

class SNPDistribution:
    """
    Semi-nonparametric distribution with Hermite polynomial squared expansion.

    This implementation follows the Gallant-Nychka approach, allowing an
    arbitrary number of coefficients (c_1..c_K). It is intentionally not a
    subclass of ``rv_continuous``; instead it exposes SciPy-like methods and a
    ``freeze``/``__call__`` helper that returns a frozen distribution instance.
    """

    def __init__(self, K: int):
        if K is None:
            raise ValueError("K (number of Hermite coefficients) must be provided.")
        if K <= 0:
            raise ValueError("K must be a positive integer.")
        self.K = int(K)

    # --- public API ------------------------------------------------------- #
    def freeze(self, c, loc: float = 0.0, scale: float = 1.0):
        """
        Return a frozen distribution with parameters fixed.
        """
        c_arr, loc, scale = self._validate_params(c, loc, scale)
        return FrozenSNPDistribution(self, c_arr, loc, scale)

    __call__ = freeze  # mimic SciPy's rv_continuous freeze semantics

    def pdf(self, x, c, loc: float = 0.0, scale: float = 1.0):
        c_arr, loc, scale = self._validate_params(c, loc, scale)
        z = (np.asarray(x) - loc) / scale
        return pdf_std_eval(z, c_coeffs=c_arr) / scale
    
    def logpdf(self, x, c, loc: float = 0.0, scale: float = 1.0):
        c_arr, loc, scale = self._validate_params(c, loc, scale)
        z = (np.asarray(x) - loc) / scale
        return log_pdf_std_eval(z, c_coeffs=c_arr) - np.log(scale)
    
    def loglik(self, x, c, loc: float = 0.0, scale: float = 1.0):
        return float(np.sum(self.logpdf(x=x, c=c, loc=loc, scale=scale)))

    def cdf(self, x, c, loc: float = 0.0, scale: float = 1.0):
        c_arr, loc, scale = self._validate_params(c, loc, scale)
        z = (np.asarray(x) - loc) / scale
        return cdf_std_eval(z, c_coeffs=c_arr)

    def ppf(self, q, c, loc: float = 0.0, scale: float = 1.0):
        c_arr, loc, scale = self._validate_params(c, loc, scale)
        q = np.asarray(q, dtype=float)

        def root_fn(z, target):
            return cdf_std_eval(z, c_coeffs=c_arr) - target

        z_out = np.empty_like(q, dtype=float)
        for idx, q_i in np.ndenumerate(q):
            if not 0.0 <= q_i <= 1.0:
                raise ValueError("Quantiles must be in [0, 1].")
            z_out[idx] = brentq(
                lambda z: root_fn(z, q_i),
                a=-12.0,
                b=+12.0,
                maxiter=100,
                xtol=1e-10,
            )

        return loc + scale * z_out

    def rvs(
        self,
        c,
        loc: float = 0.0,
        scale: float = 1.0,
        size=1,
        random_state=None,
        method: str = "ar",
    ):
        """
        Random variate generation.

        Parameters
        ----------
        method : {"icdf", "ar"}
            Sampling method to use.
        """
        method = str(method)
        if method == "icdf":
            return self.rvs_icdf(
                c=c, loc=loc, scale=scale, size=size, random_state=random_state
            )
        if method == "ar":
            return self.rvs_ar(
                c=c, loc=loc, scale=scale, size=size, random_state=random_state
            )
        raise ValueError('method must be either "inverse_cdf" or "accept_reject".')

    def rvs_ar(self, c, loc=0.0, scale=1.0, size=1, random_state=None):
        """
        Random variate generation via accept-reject method.
        """
        c_arr, loc, scale = self._validate_params(c, loc, scale)
        return snp_sampler(c_coeffs=c_arr, size=size,
                               random_state=random_state, loc=loc, scale=scale)

    def rvs_icdf(self, c, loc: float = 0.0, scale: float = 1.0, size=1, random_state=None):
        """
        Random variate generation via inverse transform on U(0,1).
        """
        rng = np.random.default_rng(random_state)
        u = rng.random(size=size)
        return self.ppf(u, c=c, loc=loc, scale=scale)

    def moment(self, order, c, loc: float = 0.0, scale: float = 1.0):
        """
        Raw moment(s) of the distribution (same contract as scipy.stats.moment).
        """
        c_arr, loc, scale = self._validate_params(c, loc, scale)
        n = np.asarray(order, dtype=int)
        if np.any(n < 0):
            raise ValueError("Moment order must be non-negative.")

        # Compute standardized raw moments up to max requested order.
        m_std = snp_standardized_raw_moments_from_c(
            c_coeffs=c_arr, r_max=int(np.max(n))
        )
        m_raw = compute_raw_moments_from_loc_scale(
            moments_std=m_std, loc=loc, scale=scale
        )
        return m_raw[n]
    
    def fit_moment_matching(self, x, fit_loc_scale: bool = True):
        
        x = np.asarray(x, dtype=float).reshape(-1)
        res = snp_fit_moment_matching(
            x, 
            K=self.K, 
            fit_loc_scale=fit_loc_scale)
        
        loc_fit = res.get("loc", None)
        scale_fit = res.get("scale", None)
        if loc_fit is None or scale_fit is None:
            # fallback when loc/scale were not part of the fit (moment-matching standardizes data)
            loc_fit = float(np.mean(x))
            scale_fit = float(np.std(x, ddof=1))

        # returns a tuple as scipy's fit method does
        return (res["c"], loc_fit, scale_fit)
    
    def fit_mle(
        self,
        x,
        fit_loc: bool = True,
        fit_scale: bool = True,
        constrain_std_moments: bool = False,
        loc_scale_fn=None,
        loc_scale_params0=None,
        loc_scale_bounds=None,
        loc_scale_kwargs=None,
        return_opt_result: bool = False,
        **kwargs,
    ):        
        x = np.asarray(x, dtype=float).reshape(-1)
        res = snp_fit_mle(
            x=x,
            K=self.K,
            fit_loc=fit_loc,
            fit_scale=fit_scale,
            constrain_std_moments=constrain_std_moments,
            loc_scale_fn=loc_scale_fn,
            loc_scale_params0=loc_scale_params0,
            loc_scale_bounds=loc_scale_bounds,
            loc_scale_kwargs=loc_scale_kwargs,
            **kwargs,
        )
        
        loc_fit = res.get("loc", None)
        if loc_fit is None:
            # fallback
            loc_fit = float(np.mean(x))
        
        scale_fit = res.get("scale", None)
        if scale_fit is None:
            # fallback
            scale_fit = float(np.std(x, ddof=1))

        if return_opt_result:
            # allow callers to access external_params / optimizer details
            return res
        # returns a tuple as scipy's fit method does
        return (res["c"], loc_fit, scale_fit)
    
    # --- helpers ---------------------------------------------------------- #
    def _validate_params(self, c, loc: float, scale: float):
        c_arr = np.asarray(c, dtype=float).reshape(-1)
        if c_arr.size != self.K:
            raise ValueError(f"Expected {self.K} coefficients, got {c_arr.size}.")

        scale = float(scale)
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        loc = float(loc)

        return c_arr, loc, scale


class FrozenSNPDistribution:
    """
    Frozen SNP distribution with fixed coefficients/loc/scale.
    """

    def __init__(self, dist: SNPDistribution, c: np.ndarray, loc: float, scale: float):
        self._dist = dist
        self.c = c
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return self._dist.pdf(x, c=self.c, loc=self.loc, scale=self.scale)

    def logpdf(self, x):
        return self._dist.logpdf(x, c=self.c, loc=self.loc, scale=self.scale)

    def loglik(self, x):
        return self._dist.loglik(x, c=self.c, loc=self.loc, scale=self.scale)

    def cdf(self, x):
        return self._dist.cdf(x, c=self.c, loc=self.loc, scale=self.scale)

    def ppf(self, q):
        return self._dist.ppf(q, c=self.c, loc=self.loc, scale=self.scale)

    def moment(self, order):
        return self._dist.moment(order, c=self.c, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None, method: str="ar"):
        return self._dist.rvs(
            c=self.c, loc=self.loc, scale=self.scale, size=size, 
            random_state=random_state, method=method
        )
    
