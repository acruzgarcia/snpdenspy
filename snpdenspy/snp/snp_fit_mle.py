import math
import numpy as np
from scipy.optimize import minimize

from snpdenspy.hermite.hermite_funcs import hermite_polys_eval
from snpdenspy.snp.snp_eval_functions import snp_norm_constant


def _build_minimize_options(verbose: bool, minimize_kwargs: dict) -> dict:
    """
    Merge user-supplied options with a verbosity flag (disp) for scipy.optimize.minimize.
    """
    options = minimize_kwargs.pop("options", {})
    if verbose and "disp" not in options:
        options = {**options, "disp": True}
    return options


def _factorial_terms(K: int) -> np.ndarray:
    return np.array([math.factorial(k + 1) for k in range(K)], dtype=float)


def _unpack_params(params, K, fit_loc, fit_scale, loc_fixed, scale_fixed):
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.size != K + int(fit_loc) + int(fit_scale):
        raise ValueError("Unexpected parameter vector size for the requested fit options.")

    c = params[:K]
    loc = float(params[K]) if fit_loc else float(loc_fixed)
    scale_idx = K + int(fit_loc)
    scale = float(params[scale_idx]) if fit_scale else float(scale_fixed)
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    return c, loc, scale


def _loglik_and_grad_full(x, c, loc, scale, poly_floor: float):
    """
    Compute log-likelihood and gradients w.r.t c, loc, scale for given params.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)
    K = c.size
    if K == 0:
        raise ValueError("At least one Hermite coefficient is required.")
    if scale <= 0.0:
        raise ValueError("scale must be positive.")

    z = (x - loc) / scale
    H = hermite_polys_eval(K, z)  # shape: (K+1, n)

    poly = 1.0 + np.dot(c, H[1:])  # 1 + sum c_k H_{k+1}(z)
    poly_safe = np.where(np.abs(poly) < poly_floor, poly_floor * np.sign(poly + 1e-12), poly)
    inv_poly = 1.0 / poly_safe

    idx = np.arange(1, K + 1, dtype=float)
    poly_prime = np.dot(c * idx, H[:-1])  # sum c_k (k+1) H_k(z)
    ratio_poly_prime = poly_prime * inv_poly

    log_phi = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)

    Z = snp_norm_constant(c)
    log_Z = math.log(Z)
    factorial_terms = _factorial_terms(K)

    log_pdf = -math.log(scale) + log_phi + 2.0 * np.log(np.abs(poly_safe)) - log_Z
    loglik = float(np.sum(log_pdf))

    grad_c = 2.0 * (H[1:] * inv_poly).sum(axis=1) - (2.0 * factorial_terms * c / Z) * x.size

    grad_loc = np.sum((z - 2.0 * ratio_poly_prime) / scale)
    grad_scale = np.sum((z * z - 1.0 - 2.0 * ratio_poly_prime * z) / scale)

    return loglik, grad_c, float(grad_loc), float(grad_scale)


def snp_neg_log_likelihood_and_grad(
    params,
    x,
    K: int,
    fit_loc: bool = True,
    fit_scale: bool = True,
    loc_fixed: float = 0.0,
    scale_fixed: float = 1.0,
    poly_floor: float = 1e-12,
):
    """
    Negative log-likelihood and gradient (matching params ordering) for optimizer use.
    """
    c, loc, scale = _unpack_params(params, K=K, fit_loc=fit_loc, fit_scale=fit_scale, loc_fixed=loc_fixed, scale_fixed=scale_fixed)
    loglik, grad_c, grad_loc, grad_scale = _loglik_and_grad_full(
        x=x, c=c, loc=loc, scale=scale, poly_floor=poly_floor
    )

    grads = [grad_c]
    if fit_loc:
        grads.append(np.array([grad_loc], dtype=float))
    if fit_scale:
        grads.append(np.array([grad_scale], dtype=float))
    grad_loglik = np.concatenate(grads)
    return -loglik, -grad_loglik


def snp_fit_mle(
    x: np.ndarray,
    K: int,
    fit_loc: bool = True,
    fit_scale: bool = True,
    method: str = "L-BFGS-B",
    poly_floor: float = 1e-12,
    initial_c=None,
    initial_loc=None,
    initial_scale=None,
    verbose: bool = False,
    **minimize_kwargs,
):
    """
    Maximum likelihood fit for SNP coefficients (and optionally loc/scale).

    Returns a dict mirroring fit_moment_matching.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if K <= 0:
        raise ValueError("K must be a positive integer.")

    c0 = np.zeros(K, dtype=float) if initial_c is None else np.asarray(initial_c, dtype=float).reshape(-1)
    if c0.size != K:
        raise ValueError(f"Expected {K} initial c coefficients, got {c0.size}.")

    loc0 = float(np.mean(x)) if initial_loc is None else float(initial_loc)
    scale0 = float(np.std(x, ddof=1)) if initial_scale is None else float(initial_scale)
    if scale0 <= 0.0:
        raise ValueError("Initial scale must be positive.")

    params0 = [c0]
    loc_fixed = loc0
    scale_fixed = scale0
    if fit_loc:
        params0.append(np.array([loc0], dtype=float))
    if fit_scale:
        params0.append(np.array([scale0], dtype=float))
    p0 = np.concatenate(params0)

    bounds = []
    bounds.extend([(None, None)] * K)
    if fit_loc:
        bounds.append((None, None))
    if fit_scale:
        bounds.append((1e-10, None))

    def objective(p):
        return snp_neg_log_likelihood_and_grad(
            p,
            x=x,
            K=K,
            fit_loc=fit_loc,
            fit_scale=fit_scale,
            loc_fixed=loc_fixed,
            scale_fixed=scale_fixed,
            poly_floor=poly_floor,
        )

    res = minimize(
        objective,
        p0,
        method=method,
        jac=True,
        bounds=bounds if bounds else None,
        options=_build_minimize_options(verbose=verbose, minimize_kwargs=minimize_kwargs),
        **minimize_kwargs
    )

    c_hat, loc_hat, scale_hat = _unpack_params(
        res.x,
        K=K,
        fit_loc=fit_loc,
        fit_scale=fit_scale,
        loc_fixed=loc_fixed,
        scale_fixed=scale_fixed,
    )

    return {
        "c": c_hat,
        "loc": loc_hat if fit_loc else loc_fixed,
        "scale": scale_hat if fit_scale else scale_fixed,
        "success": bool(res.success),
        "message": res.message,
        "opt_result": res,
    }
