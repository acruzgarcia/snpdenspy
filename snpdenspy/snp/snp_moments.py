import math
import numpy as np
from typing import Callable

from snpdenspy.hermite.hermite_funcs import hermite_product

def _double_factorial_odd(n: int) -> int:
    # n is odd >= 1: (n)!! = n*(n-2)*...*1
    out = 1
    k = n
    while k >= 1:
        out *= k
        k -= 2
    return out

def _normal_moment(m: int) -> float:
    """E[Z^m] for Z~N(0,1)."""
    if m < 0:
        return 0.0
    if m % 2 == 1:
        return 0.0
    if m == 0:
        return 1.0
    # (m-1)!!
    return float(_double_factorial_odd(m - 1))

def Ezr_Hn(r: int, n: int) -> float:
    """
    Computes E[Z^r H_n(Z)] for Z~N(0,1), probabilists' Hermite.
    Closed form via monomial expansion of H_n.
    """
    s = 0.0
    nf = math.factorial(n)
    for j in range(0, n // 2 + 1):
        m = r + n - 2*j
        mu = _normal_moment(m)
        if mu == 0.0:
            continue
        denom = math.factorial(j) * math.factorial(n - 2*j) * (2**j)
        s += ((-1)**j) * mu / denom
    return nf * s

def snp_standardized_raw_moments_from_c(
        c_coeffs: np.ndarray, 
        r_max: int, 
        E_zn_Hn_matrix: np.ndarray = None) -> np.ndarray:
    """
    """
    # Build d coefficients from c via hermite_product and normalize so d0=1
    c0_c = np.concatenate(([1.0], c_coeffs))
    d = hermite_product(c0_c, c0_c)
    d = d / float(d[0])  # enforce d0=1

    if E_zn_Hn_matrix is None:
        E_zr_Hn_eval = lambda r, n: Ezr_Hn(r, n) 
    else:
        # Check that the matrix has the right dimension
        if not E_zn_Hn_matrix.shape == (r_max + 1, len(d)):
            raise ValueError("Provided matrix E[z^n H_n] does not have the right dimension.")

        E_zr_Hn_eval = lambda r, n: E_zn_Hn_matrix[r, n]
    
    return _snp_standardized_raw_moments_from_d(
        d=d, r_max=r_max, E_zr_Hn_eval=E_zr_Hn_eval)

def _snp_standardized_raw_moments_from_d(
        d: np.ndarray, r_max: int, E_zr_Hn_eval: Callable[[int, int], float]) -> np.ndarray:
    """
    Compute standardized raw moments m_r = E[Z^r] for r=0..r_max
    where f_Z(z)=phi(z)*sum_{n=0}^M d_n H_n(z), with d0 assumed 1.

    Parameters
    ----------
    d : array-like, shape (M+1,)
        Hermite coefficients d_0..d_M of the *squared* expansion, normalized so d0=1.
    r_max : int
        Maximum raw moment order to compute.

    Returns
    -------
    m : np.ndarray, shape (r_max+1,)
        m[r] = E[Z^r]
    """
    d = np.asarray(d, dtype=float)
    M = len(d) - 1

    m = np.zeros(r_max + 1, dtype=float)
    for r in range(r_max + 1):
        acc = 0.0
        for n in range(M + 1):
            if d[n] == 0.0:
                continue
            acc += d[n] * E_zr_Hn_eval(r, n)
        m[r] = acc
    return m

def compute_raw_moments_from_loc_scale(moments_std: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """
    Given standardized raw moments m_std[k]=E[Z^k], return raw moments of X=loc+scale*Z.

    Returns M_X[r]=E[X^r] for r=0..len(moments_std)-1.
    """
    moments_std = np.asarray(moments_std, dtype=float)
    r_max = len(moments_std) - 1
    out = np.zeros(r_max + 1, dtype=float)
    for r in range(r_max + 1):
        s = 0.0
        for k in range(r + 1):
            s += math.comb(r, k) * (loc ** (r - k)) * (scale ** k) * moments_std[k]
        out[r] = s
    return out

def EZr_Hn_matrix(R: int, N: int) -> np.ndarray:
    """
    Compute I[r,n] = E[Z^r H_n(Z)] for 0<=r<=R, 0<=n<=N,
    where Z ~ N(0,1) and H_n are probabilists' Hermite polynomials.

    Recurrence:
        I(r,n) = I(r-1,n+1) + r*I(r-1,n-1),  r>=1
    Base:
        I(0,0)=1, I(0,n>0)=0

    Returns
    -------
    I : ndarray shape (R+1, N+1)
        Table of I[r,n].
    """
    if R < 0 or N < 0:
        raise ValueError("M and N must be non-negative integers.")

    # We need n+1 access, so allocate an extra column for each step in m.
    # At step m, n can effectively reach N+m, but we only return up to N.
    max_n = N + R + 1
    prev = np.zeros(max_n + 1, dtype=float)
    prev[0] = 1.0  # I(0,0)=1

    out = np.zeros((R + 1, N + 1), dtype=float)
    out[0, :] = prev[:N + 1]

    for m in range(1, R + 1):
        curr = np.zeros_like(prev)
        # compute curr[n] for n up to N (we only care up to N in output),
        # but curr depends on prev[n+1] and prev[n-1], so ensure indices exist.
        for n in range(0, N + 1):
            term1 = prev[n + 1]  # I(m-1,n+1)
            term2 = n * prev[n - 1] if n >= 1 else 0.0  # n*I(m-1,n-1)
            curr[n] = term1 + term2
        out[m, :] = curr[:N + 1]
        prev = curr

    return out
