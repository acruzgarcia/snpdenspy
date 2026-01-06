from scipy.stats import norm
import numpy as np
import math

from snpdenspy.hermite.hermite_funcs import (
    hermite_polys_eval,
    hermite_product
    )

def snp_norm_constant(c_coeffs):
    """
    Compute the normalizing constant of a squared Hermite expansion density.

    Consider the density defined on the standardized variable z as

        f(z) = φ(z) · ( 1 + ∑_{k=1}^K c_k H_k(z) )² ,

    where:
        - φ(z) is the standard normal density,
        - H_k(z) are probabilistic Hermite polynomials,
        - c_k are real coefficients.

    Due to the orthogonality of probabilistic Hermite polynomials with respect
    to the standard normal measure,

        ∫ H_j(z) H_k(z) φ(z) dz = k! · δ_{jk},

    all cross terms vanish when integrating f(z), and the integral simplifies to

        ∫ f(z) dz = 1 + ∑_{k=1}^K c_k² · k! .

    This function returns that integral value, which is the normalizing constant
    required to rescale the density so that it integrates to one.

    Parameters
    ----------
    c_coeffs : array-like
        Coefficients (c₁, …, c_K) of the Hermite expansion (excluding c0=1 constant term).

    Returns
    -------
    float
        The normalizing constant of the density.
    """
    return 1.0 + sum(c_coeffs[k]**2 * math.factorial(k+1) for k in range(len(c_coeffs)))        

def pdf_std_eval(z, c_coeffs):
    """
    Pdf evaluation
    """

    # poly_mono = self._ build_poly_mono(np.array([c1, c2, c3, c4, c5]))
    # poly = np.polyval(poly_mono, z)
    phi = np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
    H_z = hermite_polys_eval(len(c_coeffs), z)
    poly = 1 + sum(c_coeffs[k] * H_z[k+1] for k in range(len(c_coeffs)))
        
    # pdf() in rv_continuous computes _pdf() / scale!        
    return (phi * poly * poly) / snp_norm_constant(c_coeffs)

def cdf_std_eval(z, c_coeffs):
    """
    Cdf evaluation
    """
    
    # Build Hermite coefficients of P(z)^2
    c0_cK = np.concatenate(([1.0], c_coeffs))
    d_coeffs = hermite_product(c0_cK, c0_cK)
        
    # Normalizing constant
    #Z = self.normalizing_constant(c_coeffs)
    Z = d_coeffs[0] # faster

    phi = np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
        
    # Evaluated Hermite polynomials up tyo order 2K-1
    H_z = hermite_polys_eval(len(d_coeffs) - 1, z)
        
    # CDF
    cdf = d_coeffs[0] * norm.cdf(z)
    correction = sum(
        d_coeffs[n] * H_z[n - 1] for n in range(1, len(d_coeffs))
        )

    return (cdf - phi * correction) / Z

def log_pdf_std_eval(z, c_coeffs):
    """
    LogPdf evaluation
    """

    log_phi = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
    H_z = hermite_polys_eval(len(c_coeffs), z)
    poly = 1 + sum(c_coeffs[k] * H_z[k+1] for k in range(len(c_coeffs)))    
        
    log_Z = math.log(snp_norm_constant(c_coeffs))
    return log_phi + 2.0 * np.log(np.abs(poly)) - log_Z