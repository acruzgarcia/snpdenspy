from scipy.stats import rv_continuous
import numpy as np
from scipy.optimize import brentq

from snpdenspy.snp.snp_eval_functions import (
    pdf_std_eval,
    cdf_std_eval,
    log_pdf_std_eval
)
from snpdenspy.snp.snp_moments import (
    snp_standardized_raw_moments_from_c,
)
from snpdenspy.snp.snp_sampler import snp_sampler


class SNPSimpleDistribution(rv_continuous):
    """
    Implementation of SNP (Semi-nonparametric distributions) 
    subclassing from scipy's rv_continuous.
    The class supports up to 5 coefficients describing the density, apart from loc and scale.
    It implements an efficient way of sampling from the distribution via accept-reject method
    and a suitable choice of the envelope density. 
    It does not implement any fitting algorithm.

    Please use SNPDistribution instead to have:
    - Unlimited number of coefficients.
    - Moment-matching and MLE fitting. 
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        if ("shapes" in kwargs):
            super().__init__(**kwargs)
        else:
            shapes = "c1,c2,c3,c4,c5"
            super().__init__(shapes=shapes, **kwargs)

    @staticmethod
    def _reshape_coefficient(coeffs: tuple) -> np.ndarray:
        """
        This method is provided because rv_continuous automatically broadcast 
        before calling _pdf()
        """
        return np.array([float(np.asarray(x).flat[0]) for x in coeffs],
                        dtype=float)
    
    def _pdf(self, z, c1, c2, c3, c4, c5):
        """
        Pdf evaluation
        """
        # We have to make sure that c1 .. cN are scalars
        c_coeffs = self._reshape_coefficient((c1, c2, c3, c4, c5))

        # pdf() in rv_continuous computes _pdf() / scale!        
        return pdf_std_eval(z, c_coeffs=c_coeffs) 

    def _logpdf(self, z, c1, c2, c3, c4, c5):
        """
        Log-pdf evaluation for standardized input.
        """
        # We have to make sure that c1 .. cN are scalars
        c_coeffs = self._reshape_coefficient((c1, c2, c3, c4, c5))

        # scale is handle in logpdf        
        return log_pdf_std_eval(z, c_coeffs=c_coeffs) 

    def _cdf(self, z, c1, c2, c3, c4, c5):
        """
        Cdf evaluation
        """
        # We have to make sure that c1 .. cN are scalars
        c_coeffs = self._reshape_coefficient((c1, c2, c3, c4, c5))
        return cdf_std_eval(z, c_coeffs=c_coeffs)
    
    def _ppf(self, q, c1, c2, c3, c4, c5):
        """
        Inverse quantile evaluation via root-finding of the cdf function.
        """
        q = np.asarray(q)

        # coefficients as scalars
        c = self._reshape_coefficient((c1, c2, c3, c4, c5))

        def func_root(z, q):
            return self._cdf(z, *c) - q

        out = np.empty_like(q, dtype=float)

        for idx, q_i in np.ndenumerate(q):
            # Robust bracket for Gaussian-like tails
            out[idx] = brentq(
                lambda z: func_root(z, q_i),
                a=-12.0,
                b=+12.0,
                maxiter=100,
                xtol=1e-10
            )

        return out

    def _rvs(self, c1, c2, c3, c4, c5, size=1, random_state=None):
        """
        Random variate generation via accept-reject method.
        """
        # coefficients as scalars
        c = self._reshape_coefficient((c1, c2, c3, c4, c5))

        # loc and scale are handled in rvs() methods
        return snp_sampler(c_coeffs=c, size=size, random_state=random_state, 
                           loc=0, scale=1)

    def _munp(self, n, c1, c2, c3, c4, c5):
        
        # n: scalar or array-like order(s) of raw moment(s) for Z (loc=0, scale=1)
        n = np.asarray(n, dtype=int)
        # coefficients as scalars
        c = self._reshape_coefficient((c1, c2, c3, c4, c5))

        # compute standardized raw moments up to max(n)
        m_std = snp_standardized_raw_moments_from_c(c_coeffs=c, r_max=int(np.max(n)))

        return m_std[n]  # raw moment(s) for Z; rv_continuous will handle loc/scale

    def _argcheck(self, c1, c2, c3, c4, c5):
        """
        Any real value for the parameters are valid.
        """
        # Accept any real numbers
        # return np.ones_like(c1, dtype=bool)
        return True

    def _fitstart(self, data, args=None):
        """
        Starting values for scipy's rv_continuous.fit: (c1..c5, loc, scale).
        """
        data = np.asarray(data, dtype=float)
        loc0 = float(np.mean(data))
        scale0 = float(np.std(data, ddof=1))
        return (0.0, 0.0, 0.0, 0.0, 0.0, loc0, scale0)
        
if __name__ == "__main__":
    pass
