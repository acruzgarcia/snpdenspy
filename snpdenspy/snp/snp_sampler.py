import numpy as np
from scipy.stats import chi2
from math import gamma

from snpdenspy.hermite.hermite_funcs import (
    hermite_poly_to_monomial
)

def snp_sampler(c_coeffs, size, random_state=None, loc=0, scale=1) -> np.ndarray:
    """
    Devuelve muestras x ~ f(x) usando accept-reject method basado 
    en una densidad enveloppe que es una mixtura de chi distributions.
    """
    rng = np.random.default_rng(random_state)

    # coef Hermite: c0=1
    c = np.concatenate(([1.0], c_coeffs), dtype=float)
    
    # P en monomial, Q=P^2 en monomial low->high: Q(z)=sum b_m z^m
    # coefficients are low->high
    poly_mono = hermite_poly_to_monomial(c)
    b_coeffs = np.convolve(poly_mono, poly_mono)    
    abs_b_coeffs = np.abs(b_coeffs)

    # pesos de mixtura chi para r=|z|:
    # r^m e^{-r^2/2} = C_m * chi_{m+1}(r),  C_m=2^{(m+1)/2-1}Gamma((m+1)/2)
    
    m = np.arange(len(abs_b_coeffs))
    Cm = (2.0 ** ((m + 1) / 2.0 - 1.0)) * np.array([gamma((mi + 1) / 2.0) for mi in m])
    w = abs_b_coeffs * Cm
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("Envelope weights invalid (all zero or non-finite). Check coefficients.")
    w = w / w_sum

    # np.polyval espera high->low
    b_coeffs_ = b_coeffs[::-1]
    abs_b_coeffs_ = abs_b_coeffs[::-1]

    out = np.empty(int(np.prod(size)) if isinstance(size, tuple) else int(size), dtype=float)
    filled = 0

    while filled < out.size:
        
        # 1) sample component m with prob w_m
        mi = rng.choice(len(w), p=w)
        df = mi + 1

        # 2) sample r ~ chi_df  (r = sqrt(chi2(df)))
        r = np.sqrt(chi2.rvs(df, random_state=rng))
        # random sign
        z = r if rng.random() < 0.5 else -r

        # 3) accept prob = Q(z)/S(|z|)
        Q = np.polyval(b_coeffs_, z)
        if Q < 0:   # numérico (debería ser >=0)
            Q = 0.0
        
        S = np.polyval(abs_b_coeffs_, abs(z))
        if S <= 0:
            continue
        
        if rng.random() <= (Q / S):
            out[filled] = scale*z + loc
            filled += 1

    # reshape to requested size
    if isinstance(size, tuple):
        return out.reshape(size)
    
    return out