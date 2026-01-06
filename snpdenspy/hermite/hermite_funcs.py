import numpy as np
from math import comb, factorial

def hermite_poly(n:int, method: str = "recurrency"):
    """
    Return the monomial coefficients of the 
    probabilists' Hermite polynomial He_n(x).

    Output format:
        [a0, a1, ..., an]
    such that:
        He_n(x) = a0 + a1*x + ... + an*x^n

    Parameters
    ----------
    n : int
        Order of the Hermite polynomial (probabilist convention).

    Returns
    -------
    coeffs : np.ndarray
        1D array of length n+1 with monomial coefficients.
    """

    if method == "explicit":
        return _hermite_poly_explicit(n)
    elif method == "recurrency":
        return _hermite_poly_recurrency(n)
    else:
        raise ValueError(f"No method {method} is implemented.")
 

def _hermite_poly_explicit(n: int):
    """
    Coeficientes monomiales (low->high) del Hermite probabilista He_n(x):
    He_0=1, He_1=x, He_{n+1} = x He_n - n He_{n-1}.

    This function uses explicit formula
    """
    coeffs = np.zeros(n + 1)

    # He_n(x) = n! * sum_{k=0..floor(n/2)} (-1)^k / (k!(n-2k)! 2^k) * x^(n-2k)
    nfact = factorial(n)
    max_k = n // 2

    for k in range(max_k + 1):
        power = n - 2*k
        coeff = nfact * ((-1)**k) / (factorial(k) * factorial(n - 2*k) * (2**k))
        coeffs[power] = coeff

    return coeffs

def _hermite_poly_recurrency(n: int) -> np.ndarray:
    """
    Coeficientes monomiales (low->high) del Hermite probabilista He_n(x):
    He_0=1, He_1=x, He_{n+1} = x He_n - n He_{n-1}.

    This function uses the recurrent formula
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])
    
    Hm1 = np.array([1.0])        # He_0
    Hm  = np.array([0.0, 1.0])   # He_1
    
    for k in range(1, n):
        xHm = np.concatenate(([0.0], Hm))        # x*He_k
        Hnext = xHm - k * np.pad(Hm1, (0, len(xHm)-len(Hm1)))
        Hm1, Hm = Hm, Hnext
        
    return Hm

def hermite_polys_eval(N, x):
    """
    Returns hermite arrays evaluated in x, 
        H[k] = H_k(x) for k=0..N
    """
    x = np.asarray(x)
    H = np.zeros((N+1,) + x.shape)

    H[0] = 1
    if N >= 1:
        H[1] = x

    for k in range(1, N):
        H[k+1] = x * H[k] - k * H[k-1]

    return H   # shape: (N+1, len(x))

def hermite_poly_to_monomial(hermite_coeffs: np.ndarray) -> np.ndarray:
    """
    Build P(z)= sum_{k=0}^K c_k He_k(z) in monomial basis (low->high).
    hermite_coeffs includes c0.
    """
    
    N = len(hermite_coeffs)

    out = np.zeros(N, dtype=float)  # low->high
    for k in range(N):
        hk = hermite_poly(k)     # low->high
        out[:len(hk)] += hermite_coeffs[k] * hk
    return out

def hermite_to_monomial_matrix(n: int) -> np.ndarray: 
    """
    It computes the (n+1)x(n+1) change-of-basis matrix using hermitenorm() function.
    Rows correspond to Hermite polynomials expressed in the monomial basis
    """
    
    M = np.zeros((n+1, n+1))
    for k in range(n+1):
        # use hermitenorm() from scioy.special 
        coeffs = _hermite_poly_explicit(k)
        M[:len(coeffs), k] = coeffs

    return M

def hermite_product(hermite_poly_a, hermite_poly_b):
    """
    Multiply two Hermite expansions.
    
    Parameters:
    hermite_poly_a : array_like
        Coefficients of the first Hermite expansion [a0, a1, ..., aM]
    hermite_poly_b : array_like
        Coefficients of the second Hermite expansion [b0, b1, ..., bN]
        
    Returns:
    hermite_poly_c : numpy array
        Coefficients of the product Hermite expansion
    """
    hermite_poly_a = np.atleast_1d(hermite_poly_a)
    hermite_poly_b = np.atleast_1d(hermite_poly_b)
    
    deg_a = len(hermite_poly_a) - 1
    deg_b = len(hermite_poly_b) - 1
    deg_c = deg_a + deg_b
    
    # Resulting coefficient array
    hermite_poly_c = np.zeros(deg_c + 1)
    
    # Product-to-Hermite formula
    for m in range(len(hermite_poly_a)):
        for n in range(len(hermite_poly_b)):
            for k in range(min(m, n) + 1):
                hermite_poly_c[m + n - 2*k] += hermite_poly_a[m] * hermite_poly_b[n] * factorial(k) * comb(m, k) * comb(n, k)
    
    return hermite_poly_c
